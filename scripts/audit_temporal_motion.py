"""Independently replay public temporal geometry before privileged scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_dense_correspondence import _numeric_equal
from probe_temporal_motion import (
    chain_indices,
    file_hash,
    frame,
    infer_pair,
    runtime_sources,
    temporal_pairs,
)

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.multiview_motion import MultiViewMotionConfig
from minecraft_scientist.exploration.temporal_tracks import TemporalTrackConfig
from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _opencv,
)


def ledger_checks(route: Path, acquisition: dict, config: dict) -> dict[str, bool]:
    pairs = temporal_pairs(acquisition, config["window_frames"])
    expected_frames = {
        f"frame-{slot:05d}.png" for pair in pairs for slot in chain_indices(pair)
    }
    expected_references = {
        path.name for path in (route / "goal-reference").glob("view-*.png")
    }
    return {
        "public_pair_selection": pairs == config["pairs"],
        "complete_frame_ledger": set(config["frame_sha256"]) == expected_frames,
        "complete_reference_ledger": len(expected_references) >= 8
        and set(config["reference_sha256"]) == expected_references,
    }


def audit(route: Path, experiment: Path, reference_config: Path) -> dict:
    config_path, report_path = experiment / "config.json", experiment / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    acquisition = json.loads((route / "report.json").read_bytes())
    checks = ledger_checks(route, acquisition, config)
    sources = runtime_sources()
    checks["source_file_set"] = set(sources) == set(config["source_sha256"])
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # Capture all replay dependencies, the scoring implementation and own bytes.
    # The producer's source manifest does not need to contain the later auditor.
    sources["_auditor"] = Path(__file__)
    sources["_scorer"] = Path(score_relative_motion.__code__.co_filename)
    sources["_numeric_comparison"] = Path(_numeric_equal.__code__.co_filename)
    initial = {
        path: file_hash(path)
        for path in (
            config_path,
            report_path,
            reference_config,
            route / "report.json",
            *sources.values(),
        )
    }
    frame_paths = {name: route / "frames" / name for name in config["frame_sha256"]}
    reference_paths = {
        name: route / "goal-reference" / name for name in config["reference_sha256"]
    }
    checks.update(
        {
            "preregistered_config_hash": file_hash(config_path)
            == report["config_sha256"],
            "reference_config_identity": file_hash(reference_config)
            == config["reference_config_sha256"],
            "acquisition_identity": file_hash(route / "report.json")
            == config["acquisition_sha256"],
            "acquisition_controls": acquisition["gate"]["passed"] is True,
            "complete_predictions": [item["pair"] for item in report["results"]]
            == config["pairs"],
            "complete_chains": [item["chain_indices"] for item in report["results"]]
            == [list(chain_indices(pair)) for pair in config["pairs"]],
            "runtime_sources": all(
                file_hash(path) == config["source_sha256"].get(name)
                for name, path in sources.items()
                if not name.startswith("_")
            ),
            "frozen_sources": all(
                file_hash(experiment / "source-snapshot" / name) == digest
                for name, digest in config["source_sha256"].items()
            ),
            "frame_hashes": all(
                file_hash(path) == digest == acquisition["frame_sha256"][name]
                for name, path in frame_paths.items()
                for digest in (config["frame_sha256"][name],)
            ),
            "reference_hashes": all(
                file_hash(path) == config["reference_sha256"][name]
                for name, path in reference_paths.items()
            ),
            "inference_completed": report["failure"] is None
            and report["sources_and_inputs_unchanged"] is True,
            "no_navigation_promotion": report["navigation_accepted"] is False
            and report["evaluator_file_loaded"] is False,
        }
    )
    cv2 = _opencv()
    checks["opencv_version"] = cv2.__version__ == config["opencv_version"]
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    reference = json.loads(reference_config.read_bytes())
    checks["thresholds_unchanged"] = (
        reference["evaluation_thresholds"] == config["evaluation_thresholds"]
    )
    checks["reference_inputs_match"] = (
        config["reference_sha256"] == reference["reference_sha256"]
        and config["acquisition_sha256"] == reference["input_report_sha256"]
    )
    temporal = TemporalTrackConfig.model_validate(config["temporal"])
    MultiViewMotionConfig.model_validate(config["multiview"])
    checks["bounded_chains"] = all(
        len(chain_indices(pair)) <= temporal.maximum_frames for pair in config["pairs"]
    )
    images = [frame(reference_paths[name]) for name in sorted(reference_paths)]
    geometry = VisualGeometryConfig()
    width = images[0].shape[1]
    focal, count, mad = _calibrate_focal(
        [
            (points + (0.5, 0.5), descriptors)
            for points, descriptors in [
                _features(image, geometry, cv2) for image in images
            ]
        ],
        width,
        geometry,
        cv2,
    )
    checks["pixel_center_focal_recomputed"] = bool(
        focal is not None
        and np.isclose(focal, config["reference_horizontal_focal_pixels"])
        and count == config["reference_focal_sample_count"]
        and np.isclose(mad, config["reference_focal_relative_mad"])
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    observed = frame(next(iter(frame_paths.values())))
    resize = CameraResizeSpec.model_validate(config["camera_resize"])
    checks["public_resize_geometry"] = resize.observation_size == (
        observed.shape[1],
        observed.shape[0],
    ) and (
        resize.render_size == tuple(acquisition["render_size"])
        if acquisition.get("render_size") is not None
        else config["render_size_source"] == "explicit_client_configuration"
    )
    checks["camera_recomputed"] = bool(
        np.allclose(
            resize.camera(focal * observed.shape[1] / width),
            config["camera_intrinsics"],
        )
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    replay = True
    for index, item in enumerate(report["results"]):
        replay &= _numeric_equal(infer_pair(route, item["pair"], config, index), item)
    checks["tracks_and_multiview_replay"] = bool(replay)
    checks["sources_inputs_unchanged_after_replay"] = (
        all(file_hash(path) == digest for path, digest in initial.items())
        and all(
            file_hash(path) == config["frame_sha256"][name]
            for name, path in frame_paths.items()
        )
        and all(
            file_hash(path) == config["reference_sha256"][name]
            for name, path in reference_paths.items()
        )
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # Only now open privileged player telemetry. Its effects/errors never flow
    # back into point selection, model fitting, policy, or accepted memory.
    positions = json.loads((route / "evaluator-audit.json").read_bytes())["positions"]
    checks["evaluator_alignment"] = len(positions) == acquisition[
        "frame_count"
    ] and all(
        0 <= pair[key] < len(positions)
        for pair in config["pairs"]
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
                "endpoint_motion": score_relative_motion(
                    item["motion"]["candidate"], **context
                ),
                "validated_motion": score_relative_motion(item["motion"], **context),
                "track_count": len(item["tracks"]["track_ids"]),
            }
        )
    positives = [item for item in scores if item["validated_motion"]["translated"]]
    controls = [item for item in scores if not item["validated_motion"]["translated"]]
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "auditor_sha256": initial[Path(__file__)],
        "motion_evaluator_sha256": initial[
            Path(score_relative_motion.__code__.co_filename)
        ],
        "evaluation": {
            "scope": "single-seed development; overlapping windows are not independent trials",
            "positive_pairs": len(positives),
            "correct_endpoint_positive_pairs": sum(
                item["endpoint_motion"]["correct"] for item in positives
            ),
            "correct_validated_positive_pairs": sum(
                item["validated_motion"]["correct"] for item in positives
            ),
            "wrong_supported_validated_positive_pairs": sum(
                item["validated_motion"]["supported"]
                and not item["validated_motion"]["correct"]
                for item in positives
            ),
            "control_pairs": len(controls),
            "correctly_rejected_controls": sum(
                item["validated_motion"]["correct"] for item in controls
            ),
            "scores": scores,
            "navigation_accepted": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite temporal motion audit")
    result = audit(args.route, args.experiment, args.reference_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
