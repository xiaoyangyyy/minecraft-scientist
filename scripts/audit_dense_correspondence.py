"""Independently replay frozen RGB predictions before scoring privileged motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from probe_dense_correspondence import public_pairs

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.dense_correspondence import (
    DenseCorrespondenceConfig,
    DenseCorrespondences,
    DenseRGBMatcher,
    LocalLoFTRBackend,
)
from minecraft_scientist.exploration.two_view_motion import (
    TwoViewMotionConfig,
    estimate_two_view_motion,
)
from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _opencv,
)
from minecraft_scientist.exploration.visual_odometry import triangulate_landmarks


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _numeric_equal(first, second) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _numeric_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            _numeric_equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, float) and isinstance(second, (float, int)):
        return bool(np.isclose(first, second, atol=1e-6, rtol=1e-6))
    return first == second


def input_ledger_checks(
    config: dict, acquisition: dict, route: Path
) -> dict[str, bool]:
    expected_frames = {
        f"frame-{pair[key]:05d}.png"
        for pair in public_pairs(acquisition)
        for key in ("first", "second")
    }
    expected_references = {
        path.name for path in (route / "goal-reference").glob("view-*.png")
    }
    return {
        "complete_frame_ledger": set(config["frame_sha256"]) == expected_frames,
        "complete_reference_ledger": len(expected_references) >= 8
        and set(config["reference_sha256"]) == expected_references,
    }


def runtime_sources() -> dict[str, Path]:
    return {
        Path(path).name: Path(path)
        for path in (
            CameraResizeSpec.camera.__code__.co_filename,
            DenseRGBMatcher.match.__code__.co_filename,
            estimate_two_view_motion.__code__.co_filename,
            _features.__code__.co_filename,
            triangulate_landmarks.__code__.co_filename,
            Path(__file__).with_name("probe_dense_correspondence.py"),
        )
    }


def audit(route: Path, development: Path, checkpoint: Path) -> dict:
    config_path, report_path = development / "config.json", development / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    acquisition = json.loads((route / "report.json").read_bytes())
    source_paths = runtime_sources()
    checks = input_ledger_checks(config, acquisition, route)
    checks["runtime_source_file_set"] = set(source_paths) == set(
        config["source_sha256"]
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    initial_hashes = {
        path: _hash(path)
        for path in (config_path, report_path, Path(__file__), *source_paths.values())
    }
    checks.update(
        {
            "preregistered_config_hash": _hash(config_path) == report["config_sha256"],
            "input_acquisition_hash": _hash(route / "report.json")
            == config["input_report_sha256"],
            "acquisition_passed": acquisition["gate"]["passed"] is True,
            "public_pairs_recomputed": public_pairs(acquisition) == config["pairs"],
            "all_predictions_present": [item["pair"] for item in report["results"]]
            == config["pairs"],
            "inference_completed": report["failure"] is None
            and report["sources_and_inputs_unchanged"] is True,
            "not_promoted_to_navigation": report["navigation_accepted"] is False,
            "runtime_source_file_set": set(source_paths)
            == set(config["source_sha256"]),
            "checkpoint_identity": _hash(checkpoint)
            == config["checkpoint_sha256"]
            == report["backend"]["checkpoint_sha256"],
            "frozen_sources_valid": all(
                _hash(development / "source-snapshot" / name) == digest
                for name, digest in config["source_sha256"].items()
            ),
            "current_sources_match_frozen": all(
                _hash(path) == config["source_sha256"].get(name)
                for name, path in source_paths.items()
            ),
            "input_frame_hashes": all(
                _hash(route / "frames" / name)
                == digest
                == acquisition["frame_sha256"][name]
                for name, digest in config["frame_sha256"].items()
            ),
            "input_reference_hashes": all(
                _hash(route / "goal-reference" / name) == digest
                for name, digest in config["reference_sha256"].items()
            ),
        }
    )
    cv2 = _opencv()
    checks["opencv_version_matches"] = cv2.__version__ == report["opencv_version"]
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    references = [
        _rgb(route / "goal-reference" / name)
        for name in sorted(config["reference_sha256"])
    ]
    geometry_config = VisualGeometryConfig()
    _, reference_width = references[0].shape[:2]
    focal, count, mad = _calibrate_focal(
        [
            (points + (0.5, 0.5), descriptors)
            for points, descriptors in [
                _features(frame, geometry_config, cv2) for frame in references
            ]
        ],
        reference_width,
        geometry_config,
        cv2,
    )
    checks["focal_calibration_recomputed"] = bool(
        focal is not None
        and np.isclose(focal, config["reference_focal_pixels"])
        and count == config["reference_focal_sample_count"]
        and np.isclose(mad, config["reference_focal_relative_mad"])
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    first = _rgb(route / "frames" / f"frame-{config['pairs'][0]['first']:05d}.png")
    height, width = first.shape[:2]
    resize = CameraResizeSpec.model_validate(config["camera_resize"])
    expected_camera = resize.camera(focal * width / reference_width)
    checks["public_resize_dimensions"] = resize.observation_size == (
        width,
        height,
    ) and (
        tuple(acquisition["render_size"]) == resize.render_size
        if acquisition.get("render_size") is not None
        else config["render_size_source"] == "explicit_client_configuration"
    )
    checks["camera_resize_recomputed"] = bool(
        np.allclose(expected_camera, config["camera_intrinsics"])
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    backend = LocalLoFTRBackend(
        checkpoint,
        sha256=config["checkpoint_sha256"],
        threads=report["backend"]["threads"],
    )
    checks["backend_provenance_matches"] = backend.provenance == report["backend"]
    matcher = DenseRGBMatcher(
        backend, config=DenseCorrespondenceConfig.model_validate(config["matcher"])
    )
    replay_valid, geometry_valid = True, True
    for index, item in enumerate(report["results"]):
        pair = item["pair"]
        first = _rgb(route / "frames" / f"frame-{pair['first']:05d}.png")
        second = _rgb(route / "frames" / f"frame-{pair['second']:05d}.png")
        matches = matcher.match(first, second)
        expected = DenseCorrespondences.model_validate(item["correspondences"])
        replay_valid &= _numeric_equal(
            matches.model_dump(mode="json"), expected.model_dump(mode="json")
        )
        motion = estimate_two_view_motion(
            matches,
            expected_camera,
            config=TwoViewMotionConfig.model_validate(config["motion"]),
            seed=index,
        )
        geometry_valid &= _numeric_equal(motion.model_dump(mode="json"), item["motion"])
    checks["dense_model_replay_matches"] = bool(replay_valid)
    checks["geometry_replay_matches"] = bool(geometry_valid)
    checks["sources_and_manifests_unchanged_after_replay"] = all(
        _hash(path) == digest for path, digest in initial_hashes.items()
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # Model outputs are now independently reproduced. Privileged telemetry is
    # opened only here; no error, direction or label can influence model replay.
    positions = json.loads((route / "evaluator-audit.json").read_bytes())["positions"]
    checks["evaluator_frame_alignment"] = len(positions) == acquisition[
        "frame_count"
    ] and all(
        isinstance(pair[key], int) and 0 <= pair[key] < len(positions)
        for pair in config["pairs"]
        for key in ("first", "second")
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    thresholds = config["evaluation_thresholds"]
    scores = []
    for item in report["results"]:
        pair = item["pair"]
        score = score_relative_motion(
            item["motion"],
            positions[pair["first"]],
            positions[pair["second"]],
            minimum_translation=thresholds["minimum_translation_blocks"],
            maximum_direction_error=thresholds["maximum_direction_error_degrees"],
            maximum_rotation_error=thresholds["maximum_rotation_error_degrees"],
        )
        scores.append({"pair_id": pair["id"], "kind": pair["kind"], **score})
    positives = [item for item in scores if item["translated"]]
    controls = [item for item in scores if not item["translated"]]
    angular_errors = [
        item["direction_error_degrees"]
        for item in scores
        if item["direction_error_degrees"] is not None
    ]
    evaluation = {
        "scope": "development-only single-seed pairs; not navigation acceptance",
        "positive_pairs": len(positives),
        "correct_positive_pairs": sum(item["correct"] for item in positives),
        "control_pairs": len(controls),
        "correctly_rejected_controls": sum(item["correct"] for item in controls),
        "supported_direction_error_median": float(np.median(angular_errors))
        if angular_errors
        else None,
        "scores": scores,
        "navigation_accepted": False,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "evaluation": evaluation,
        "auditor_sha256": _hash(Path(__file__)),
        "motion_evaluator_sha256": _hash(
            Path(score_relative_motion.__code__.co_filename)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite dense correspondence audit")
    result = audit(args.route, args.development, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
