"""Preregister and infer dense RGB motion candidates without evaluator telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.exploration.dense_correspondence import (
    DenseCorrespondenceConfig,
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
    _matched_indices,
    _opencv,
)
from minecraft_scientist.exploration.visual_odometry import triangulate_landmarks


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def public_pairs(report: dict) -> list[dict]:
    moves = [item for item in report["timeline"] if item["action"]["kind"] == "move"]
    if len(moves) < 4:
        raise ValueError(
            "dense development requires at least four public movement interventions"
        )
    selected = sorted(
        set(range(min(4, len(moves)))) | set(range(max(0, len(moves) - 4), len(moves)))
    )
    pairs = [
        {
            "id": f"move-{index:02d}",
            "kind": "motion_intervention",
            "first": moves[index]["start_frame"],
            "second": moves[index]["end_frame"],
        }
        for index in selected
    ]
    before = moves[0]["start_frame"]
    waits = [
        item
        for item in report["timeline"]
        if item["action"]["kind"] == "wait"
        and item["end_frame"] <= before
        and item["end_frame"] - item["start_frame"] >= 2
    ]
    if waits:
        item = waits[-1]
        pairs.append(
            {
                "id": "wait-control",
                "kind": "wait_control",
                "first": item["start_frame"],
                "second": item["end_frame"],
            }
        )
    for index, item in enumerate(
        [item for item in report["timeline"] if item["action"]["kind"] == "look"][:2]
    ):
        pairs.append(
            {
                "id": f"rotation-{index:02d}",
                "kind": "rotation_control",
                "first": item["start_frame"],
                "second": item["end_frame"],
            }
        )
    for index in (before, report["frame_count"] - 1):
        pairs.append(
            {
                "id": f"identity-{index:03d}",
                "kind": "identity_control",
                "first": index,
                "second": index,
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_bytes = (args.route / "report.json").read_bytes()
    acquisition = json.loads(report_bytes)
    if not acquisition["gate"]["passed"]:
        raise ValueError("dense route acquisition must pass its controls")
    pairs = public_pairs(acquisition)
    match_config, motion_config = DenseCorrespondenceConfig(), TwoViewMotionConfig()
    source_paths = {
        Path(path).name: Path(path)
        for path in (
            DenseRGBMatcher.match.__code__.co_filename,
            estimate_two_view_motion.__code__.co_filename,
            _features.__code__.co_filename,
            triangulate_landmarks.__code__.co_filename,
            __file__,
        )
    }
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    for name, data in source_bytes.items():
        (snapshot / name).write_bytes(data)
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()
    }
    inputs = {
        index: args.route / "frames" / f"frame-{index:05d}.png"
        for item in pairs
        for index in (item["first"], item["second"])
    }
    for path in inputs.values():
        if _hash(path) != acquisition["frame_sha256"][path.name]:
            raise ValueError("dense RGB input hash mismatch")
    cv2 = _opencv()
    geometry_config = VisualGeometryConfig()
    reference_paths = sorted((args.route / "goal-reference").glob("view-*.png"))
    reference = [_rgb(path) for path in reference_paths]
    height, width = reference[0].shape[:2]
    focal, count, mad = _calibrate_focal(
        [_features(frame, geometry_config, cv2) for frame in reference],
        width,
        geometry_config,
        cv2,
    )
    if focal is None:
        raise ValueError("public focal calibration was rejected")
    observation_height, observation_width = _rgb(inputs[pairs[0]["first"]]).shape[:2]
    camera = np.array(
        (
            (focal * observation_width / width, 0, observation_width / 2),
            (0, focal * observation_height / height, observation_height / 2),
            (0, 0, 1),
        )
    )
    config = {
        "pairs": pairs,
        "matcher": match_config.model_dump(mode="json"),
        "motion": motion_config.model_dump(mode="json"),
        "camera_intrinsics": camera.tolist(),
        "reference_focal_pixels": focal,
        "reference_focal_sample_count": count,
        "reference_focal_relative_mad": mad,
        "source_sha256": source_hashes,
        "input_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "frame_sha256": {path.name: _hash(path) for path in inputs.values()},
        "reference_sha256": {path.name: _hash(path) for path in reference_paths},
        "checkpoint_sha256": args.sha256,
        "scope": "development-only; no navigation promotion",
        "evaluation_thresholds": {
            "minimum_translation_blocks": 0.15,
            "maximum_direction_error_degrees": 20.0,
            "maximum_rotation_error_degrees": 5.0,
        },
    }
    config_path = args.output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = _hash(config_path)
    backend = LocalLoFTRBackend(args.checkpoint, sha256=args.sha256)
    matcher = DenseRGBMatcher(backend, config=match_config)
    results, failure = [], None
    try:
        for index, pair in enumerate(pairs):
            first, second = _rgb(inputs[pair["first"]]), _rgb(inputs[pair["second"]])
            matches = matcher.match(first, second)
            motion = estimate_two_view_motion(
                matches, camera, config=motion_config, seed=index
            )
            sift0, sift1 = (
                _features(first, geometry_config, cv2),
                _features(second, geometry_config, cv2),
            )
            sift_count = len(_matched_indices(sift0, sift1, geometry_config, cv2)[0])
            item = {
                "pair": pair,
                "correspondences": matches.model_dump(mode="json"),
                "motion": motion.model_dump(mode="json"),
                "sift_mutual_match_count": sift_count,
            }
            results.append(item)
            print(
                json.dumps(
                    {
                        "pair": pair["id"],
                        "dense": len(matches.points0),
                        "sift": sift_count,
                        "supported": motion.supported,
                        "reason": motion.reason,
                    }
                ),
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 -- preserve failed inference and exit nonzero
        failure = {"type": type(error).__name__, "message": str(error)}
    unchanged = (
        all(
            path.read_bytes() == source_bytes[name]
            for name, path in source_paths.items()
        )
        and _hash(config_path) == config_hash
        and all(
            _hash(path) == config["frame_sha256"][path.name] for path in inputs.values()
        )
        and all(
            _hash(path) == config["reference_sha256"][path.name]
            for path in reference_paths
        )
    )
    payload = {
        "suite": "dense_correspondence_development",
        "evaluator_file_loaded": False,
        "config_sha256": config_hash,
        "backend": backend.provenance,
        "opencv_version": cv2.__version__,
        "results": results,
        "failure": failure,
        "sources_and_inputs_unchanged": unchanged,
        "navigation_accepted": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failure or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
