"""Replay frozen pixel refinement before independent relative-motion scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_dense_correspondence import _numeric_equal
from diagnose_motion_stability import file_hash
from probe_refined_motion import infer_pair, runtime_sources

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.visual_geometry import _opencv


def audit(route: Path, development: Path, refined: Path) -> dict:
    config_path, report_path = refined / "config.json", refined / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    sources = runtime_sources()
    inputs = {
        name: development / name
        for name in ("config.json", "report.json", "audit.json")
    }
    checks = {
        "preregistered_config_hash": file_hash(config_path) == report["config_sha256"],
        "input_file_set": set(inputs) == set(config["input_sha256"]),
        "input_hashes": all(
            file_hash(path) == config["input_sha256"].get(name)
            for name, path in inputs.items()
        ),
        "source_file_set": set(sources) == set(config["source_sha256"]),
        "runtime_sources": all(
            file_hash(path) == config["source_sha256"].get(name)
            for name, path in sources.items()
        ),
        "frozen_sources": all(
            file_hash(refined / "source-snapshot" / name) == digest
            for name, digest in config["source_sha256"].items()
        ),
        "opencv_version": _opencv().__version__ == config["opencv_version"],
        "not_promoted": report["navigation_accepted"] is False
        and report["evaluator_file_loaded"] is False,
        "producer_inputs_unchanged": report["sources_and_inputs_unchanged"] is True,
        "route_acquisition_identity": file_hash(route / "report.json")
        == config["route_report_sha256"],
    }
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    original_config, original_report, original_audit = [
        json.loads(inputs[name].read_bytes()) for name in inputs
    ]
    checks["original_dense_audit_valid"] = original_audit["valid"] is True
    checks["camera_motion_thresholds_unchanged"] = all(
        config[key] == original_config[key]
        for key in (
            "camera_intrinsics",
            "motion",
            "evaluation_thresholds",
            "frame_sha256",
        )
    )
    checks["complete_pair_set"] = (
        [item["pair"] for item in report["results"]]
        == original_config["pairs"]
        == [item["pair"] for item in original_report["results"]]
    )
    frame_paths = {
        name: route / "frames" / name for name in original_config["frame_sha256"]
    }
    checks["frame_hashes"] = all(
        file_hash(path) == original_config["frame_sha256"][name]
        for name, path in frame_paths.items()
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    replay = True
    for index, (original, item) in enumerate(
        zip(original_report["results"], report["results"], strict=True)
    ):
        actual = infer_pair(route, original, config, index)
        replay &= _numeric_equal(actual, item)
    checks["refinement_and_motion_replay"] = bool(replay)
    checks["sources_inputs_unchanged_after_replay"] = (
        file_hash(config_path) == report["config_sha256"]
        and all(
            file_hash(path) == config["input_sha256"][name]
            for name, path in inputs.items()
        )
        and all(
            file_hash(path) == config["source_sha256"][name]
            for name, path in sources.items()
        )
        and all(
            file_hash(path) == config["frame_sha256"][name]
            for name, path in frame_paths.items()
        )
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    positions = json.loads((route / "evaluator-audit.json").read_bytes())["positions"]
    acquisition = json.loads((route / "report.json").read_bytes())
    checks["evaluator_alignment"] = len(positions) == acquisition[
        "frame_count"
    ] and all(
        isinstance(item["pair"][key], int) and 0 <= item["pair"][key] < len(positions)
        for item in report["results"]
        for key in ("first", "second")
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    thresholds = config["evaluation_thresholds"]
    scores = []
    for item in report["results"]:
        pair = item["pair"]
        context = {
            "first": positions[pair["first"]],
            "second": positions[pair["second"]],
            "minimum_translation": thresholds["minimum_translation_blocks"],
            "maximum_direction_error": thresholds["maximum_direction_error_degrees"],
            "maximum_rotation_error": thresholds["maximum_rotation_error_degrees"],
        }
        scores.append(
            {
                "pair_id": pair["id"],
                "kind": pair["kind"],
                "base_motion": score_relative_motion(
                    item["motion"]["candidate"], **context
                ),
                "stable_motion": score_relative_motion(item["motion"], **context),
                "retained_matches": len(item["refined"]["retained_source_indices"]),
            }
        )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "auditor_sha256": file_hash(Path(__file__)),
        "motion_evaluator_sha256": file_hash(
            Path(score_relative_motion.__code__.co_filename)
        ),
        "evaluation": {
            "scope": "single-seed development refinement; not navigation acceptance",
            "positive_pairs": sum(
                item["stable_motion"]["translated"] for item in scores
            ),
            "correct_positive_pairs": sum(
                item["stable_motion"]["translated"] and item["stable_motion"]["correct"]
                for item in scores
            ),
            "control_pairs": sum(
                not item["stable_motion"]["translated"] for item in scores
            ),
            "correctly_rejected_controls": sum(
                not item["stable_motion"]["translated"]
                and item["stable_motion"]["correct"]
                for item in scores
            ),
            "scores": scores,
            "navigation_accepted": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite refinement audit")
    result = audit(args.route, args.development, args.refined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
