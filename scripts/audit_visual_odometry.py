"""Independently replay public RGB evidence before evaluating visual pose errors.

Evaluator coordinates are loaded only AFTER replay and source/data validation.
The similarity fit removes monocular coordinate gauge, not prediction error:
it is fitted on the first eight supported poses and scored on subsequent poses.
This is a development diagnostic, not held-out navigation acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.evaluation.visual_pose import fit_similarity
from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _opencv,
)
from minecraft_scientist.exploration.visual_odometry import (
    RGBVisualOdometry,
    VisualOdometryConfig,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(route: Path, development: Path, *, calibration_count: int = 8) -> dict:
    if calibration_count < 3:
        raise ValueError("at least three calibration poses are required")
    payload = json.loads(development.read_text(encoding="utf-8"))
    acquisition = json.loads((route / "report.json").read_text(encoding="utf-8"))
    checks = {}
    checks["input_report_hash"] = (
        _hash(route / "report.json") == payload["input_report_sha256"]
    )
    checks["acquisition_passed"] = acquisition["gate"]["passed"] is True
    checks["snapshot_hashes"] = all(
        _hash(development.parent / "source-snapshot" / name) == digest
        for name, digest in payload["source_sha256"].items()
    )
    current_sources = (
        Path(RGBVisualOdometry.process.__code__.co_filename),
        Path(_features.__code__.co_filename),
    )
    checks["current_sources_match_snapshot"] = all(
        _hash(path) == payload["source_sha256"][path.name] for path in current_sources
    )
    checks["goal_reference_hashes"] = all(
        _hash(route / "goal-reference" / name) == digest
        for name, digest in payload["goal_reference_sha256"].items()
    )
    checks["goal_reference_file_set"] = set(payload["goal_reference_sha256"]) == {
        path.name for path in (route / "goal-reference").glob("view-*.png")
    }
    cv2 = _opencv()
    checks["opencv_version_matches"] = cv2.__version__ == payload["opencv_version"]
    checks["pose_count_matches"] = len(payload["poses"]) == acquisition["frame_count"]
    checks["reported_counts_recomputed"] = (
        dict(Counter(item["status"] for item in payload["poses"]))
        == payload["status_counts"]
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    panorama = []
    for path in sorted((route / "goal-reference").glob("view-*.png")):
        with Image.open(path) as source_image:
            panorama.append(np.asarray(source_image.convert("RGB")))
    geometry_config = VisualGeometryConfig.model_validate(payload["config"]["features"])
    reference_height, reference_width = panorama[0].shape[:2]
    reference_focal, count, mad = _calibrate_focal(
        [_features(frame, geometry_config, cv2) for frame in panorama],
        reference_width,
        geometry_config,
        cv2,
    )
    checks["focal_calibration_recomputed"] = bool(
        reference_focal is not None
        and np.isclose(
            reference_focal,
            payload.get("reference_focal_pixels", payload["focal_pixels"]),
            atol=1e-7,
        )
        and count == payload["focal_sample_count"]
        and np.isclose(mad, payload["focal_relative_mad"], atol=1e-7)
    )
    if not checks["focal_calibration_recomputed"]:
        return {"valid": False, "checks": checks, "evaluation": None}
    focal = payload["focal_pixels"]
    motion_frames = {
        index
        for item in acquisition["timeline"]
        if item["action"]["kind"] == "move"
        for index in range(item["start_frame"] + 1, item["end_frame"] + 1)
    }
    tracker = None
    replay_valid, frames_valid = True, True
    for index, expected in enumerate(payload["poses"]):
        path = route / "frames" / f"frame-{index:05d}.png"
        with Image.open(path) as source_image:
            frame = np.asarray(source_image.convert("RGB"))
        frames_valid &= _hash(path) == acquisition["frame_sha256"][path.name]
        frames_valid &= (
            hashlib.sha256(frame.tobytes()).hexdigest()
            == acquisition["rgb_sha256"][index]
        )
        if tracker is None:
            height, width = frame.shape[:2]
            camera = np.array(
                payload.get(
                    "camera_intrinsics",
                    ((focal, 0, width / 2), (0, focal, height / 2), (0, 0, 1)),
                )
            )
            expected_camera = np.array(
                (
                    (reference_focal * width / reference_width, 0, width / 2),
                    (0, reference_focal * height / reference_height, height / 2),
                    (0, 0, 1),
                )
            )
            checks["intrinsics_resize_recomputed"] = bool(
                np.allclose(camera, expected_camera, atol=1e-7)
            )
            tracker = RGBVisualOdometry(
                camera, config=VisualOdometryConfig.model_validate(payload["config"])
            )
        actual = tracker.process(
            frame, motion_allowed=index in motion_frames
        ).model_dump(mode="json")
        for key in actual:
            if (
                key == "camera_center"
                and actual[key] is not None
                and expected[key] is not None
            ):
                replay_valid &= bool(
                    np.allclose(actual[key], expected[key], atol=1e-7, rtol=1e-7)
                )
            else:
                replay_valid &= actual[key] == expected[key]
    checks["rgb_hashes_recomputed"] = bool(frames_valid)
    checks["public_pose_replay_matches"] = bool(replay_valid)
    checks["unsupported_poses_are_unknown"] = all(
        item["tracked"] or item["camera_center"] is None for item in payload["poses"]
    )
    checks["landmark_budget_respected"] = all(
        item["map_points"] <= payload["config"]["landmark_budget"]
        for item in payload["poses"]
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # The policy and replay have finished; only this independent evaluation
    # branch opens privileged telemetry. No gauge estimate goes back to policy.
    positions = json.loads(
        (route / "evaluator-audit.json").read_text(encoding="utf-8")
    )["positions"]
    checks["evaluator_frame_alignment"] = len(positions) == len(payload["poses"])
    if not checks["evaluator_frame_alignment"]:
        return {"valid": False, "checks": checks, "evaluation": None}
    supported = [item for item in payload["poses"] if item["tracked"]]
    indices = [item["frame_index"] for item in supported]
    estimate = np.asarray([item["camera_center"] for item in supported]).reshape(-1, 3)
    truth = np.asarray(
        [
            [positions[index][key] for key in ("xpos", "ypos", "zpos")]
            for index in indices
        ]
    ).reshape(-1, 3)
    evaluation = {
        "supported_frame_count": len(indices),
        "frame_count": len(payload["poses"]),
        "supported_fraction": len(indices) / len(payload["poses"]),
        "calibration_count": calibration_count,
        "calibration_frame_indices": indices[:calibration_count],
        "score_frame_indices": indices[calibration_count:],
        "evaluation_scope": "development-only; RGB camera versus player-position telemetry",
        "navigation_accepted": False,
    }
    try:
        if len(indices) <= calibration_count:
            raise ValueError("insufficient supported poses for disjoint scoring")
        scale, rotation, translation = fit_similarity(
            estimate[:calibration_count], truth[:calibration_count]
        )
        predicted = scale * estimate @ rotation.T + translation
        errors = np.linalg.norm(predicted - truth, axis=1)
        evaluation.update(
            {
                "gauge_scale": scale,
                "gauge_rotation": rotation.tolist(),
                "gauge_translation": translation.tolist(),
                "calibration_rmse": float(
                    np.sqrt(np.mean(errors[:calibration_count] ** 2))
                ),
                "score_rmse": float(np.sqrt(np.mean(errors[calibration_count:] ** 2))),
                "score_maximum_error": float(errors[calibration_count:].max()),
                "score_median_error": float(np.median(errors[calibration_count:])),
            }
        )
    except ValueError as error:
        evaluation["gauge_rejection"] = str(error)
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "evaluation": evaluation,
        "auditor_sha256": _hash(Path(__file__)),
        "gauge_evaluator_sha256": _hash(Path(fit_similarity.__code__.co_filename)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite odometry audit")
    result = audit(
        args.route, args.development, calibration_count=args.calibration_count
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
