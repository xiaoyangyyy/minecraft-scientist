"""Replay frozen sensitivity predictions before evaluator-only motion scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_dense_correspondence import _numeric_equal
from diagnose_motion_stability import file_hash, runtime_sources

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_stability import (
    MotionStabilityConfig,
    StableMotionEvidence,
    assess_motion_stability,
)
from minecraft_scientist.exploration.two_view_motion import TwoViewMotionConfig
from minecraft_scientist.exploration.visual_geometry import _opencv


def audit(route: Path, development: Path, sensitivity: Path) -> dict:
    config_path, report_path = sensitivity / "config.json", sensitivity / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    sources = runtime_sources()
    input_paths = {
        name: development / name
        for name in ("config.json", "report.json", "audit.json")
    }
    checks = {
        "preregistered_config_hash": file_hash(config_path) == report["config_sha256"],
        "input_file_set": set(input_paths) == set(config["input_sha256"]),
        "input_hashes": all(
            file_hash(path) == config["input_sha256"].get(name)
            for name, path in input_paths.items()
        ),
        "source_file_set": set(sources) == set(config["source_sha256"]),
        "runtime_sources": all(
            file_hash(path) == config["source_sha256"].get(name)
            for name, path in sources.items()
        ),
        "frozen_sources": all(
            file_hash(sensitivity / "source-snapshot" / name) == digest
            for name, digest in config["source_sha256"].items()
        ),
        "opencv_version": _opencv().__version__ == config["opencv_version"],
        "not_promoted": report["navigation_accepted"] is False
        and report["evaluator_file_loaded"] is False,
        "producer_inputs_unchanged": report["sources_and_inputs_unchanged"] is True,
    }
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    original_config, original_report, original_audit = [
        json.loads(input_paths[name].read_bytes()) for name in input_paths
    ]
    checks["original_dense_audit_valid"] = original_audit["valid"] is True
    checks["route_acquisition_identity"] = (
        file_hash(route / "report.json") == original_config["input_report_sha256"]
    )
    checks["evaluation_thresholds_unchanged"] = (
        config["evaluation_thresholds"] == original_config["evaluation_thresholds"]
    )
    checks["complete_pair_set"] = (
        [item["pair"] for item in report["results"]]
        == original_config["pairs"]
        == [item["pair"] for item in original_report["results"]]
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    camera = np.asarray(original_config["camera_intrinsics"])
    motion_config = TwoViewMotionConfig.model_validate(original_config["motion"])
    stability_config = MotionStabilityConfig.model_validate(config["stability"])
    replay = True
    for index, (previous, item) in enumerate(
        zip(original_report["results"], report["results"], strict=True)
    ):
        expected = StableMotionEvidence.model_validate(item["motion"])
        actual = assess_motion_stability(
            DenseCorrespondences.model_validate(previous["correspondences"]),
            camera,
            config=stability_config,
            motion_config=motion_config,
            seed=index,
        )
        replay &= _numeric_equal(
            actual.model_dump(mode="json"), expected.model_dump(mode="json")
        )
    checks["sensitivity_replay"] = bool(replay)
    checks["sources_inputs_unchanged_after_replay"] = (
        file_hash(config_path) == report["config_sha256"]
        and all(
            file_hash(path) == config["input_sha256"][name]
            for name, path in input_paths.items()
        )
        and all(
            file_hash(path) == config["source_sha256"][name]
            for name, path in sources.items()
        )
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # Open privileged motion only after every public replay/control check passes.
    positions = json.loads((route / "evaluator-audit.json").read_bytes())["positions"]
    acquisition = json.loads((route / "report.json").read_bytes())
    checks["evaluator_alignment"] = len(positions) == acquisition[
        "frame_count"
    ] and all(
        0 <= item["pair"][key] < len(positions)
        for item in report["results"]
        for key in ("first", "second")
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    thresholds = config["evaluation_thresholds"]
    scores = []
    for item in report["results"]:
        pair = item["pair"]
        scores.append(
            {
                "pair_id": pair["id"],
                "kind": pair["kind"],
                **score_relative_motion(
                    item["motion"],
                    positions[pair["first"]],
                    positions[pair["second"]],
                    minimum_translation=thresholds["minimum_translation_blocks"],
                    maximum_direction_error=thresholds[
                        "maximum_direction_error_degrees"
                    ],
                    maximum_rotation_error=thresholds["maximum_rotation_error_degrees"],
                ),
            }
        )
    positives = [item for item in scores if item["translated"]]
    controls = [item for item in scores if not item["translated"]]
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "auditor_sha256": file_hash(Path(__file__)),
        "motion_evaluator_sha256": file_hash(
            Path(score_relative_motion.__code__.co_filename)
        ),
        "evaluation": {
            "scope": "development sensitivity; cached dense inputs; not navigation acceptance",
            "positive_pairs": len(positives),
            "correct_positive_pairs": sum(item["correct"] for item in positives),
            "wrong_supported_positive_pairs": sum(
                item["supported"] and not item["correct"] for item in positives
            ),
            "control_pairs": len(controls),
            "correctly_rejected_controls": sum(item["correct"] for item in controls),
            "scores": scores,
            "navigation_accepted": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite motion sensitivity audit")
    result = audit(args.route, args.development, args.sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
