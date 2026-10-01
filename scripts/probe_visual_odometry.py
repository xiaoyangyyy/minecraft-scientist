"""Reconstruct a captured route from dense public pixels, never evaluator poses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _opencv,
)
from minecraft_scientist.exploration.visual_odometry import RGBVisualOdometry


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite odometry evidence")
    source_paths = tuple(
        Path(path)
        for path in (
            RGBVisualOdometry.process.__code__.co_filename,
            _features.__code__.co_filename,
            __file__,
        )
    )
    source_bytes = {path.name: path.read_bytes() for path in source_paths}
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()
    }
    input_report = (args.run / "report.json").read_bytes()
    report = json.loads(input_report)
    if not report["gate"]["passed"]:
        raise ValueError("dense route acquisition did not pass its gates")
    config = VisualGeometryConfig()
    cv2 = _opencv()
    panorama = [
        _image(path)
        for path in sorted((args.run / "goal-reference").glob("view-*.png"))
    ]
    height, width = panorama[0].shape[:2]
    focal, samples, mad = _calibrate_focal(
        [_features(frame, config, cv2) for frame in panorama], width, config, cv2
    )
    if focal is None:
        raise RuntimeError("rotation-only focal calibration was rejected")
    reference_focal = focal
    reference_size = (width, height)
    first_frame = _image(args.run / "frames" / "frame-00000.png")
    frame_height, frame_width = first_frame.shape[:2]
    focal_x, focal_y = focal * frame_width / width, focal * frame_height / height
    camera = np.asarray(
        ((focal_x, 0, frame_width / 2), (0, focal_y, frame_height / 2), (0, 0, 1))
    )
    tracker = RGBVisualOdometry(camera)
    motion_frames = {
        index
        for item in report["timeline"]
        if item["action"]["kind"] == "move"
        for index in range(item["start_frame"] + 1, item["end_frame"] + 1)
    }
    poses = []
    for index in range(report["frame_count"]):
        path = args.run / "frames" / f"frame-{index:05d}.png"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != report["frame_sha256"][path.name]
        ):
            raise ValueError(f"dense frame hash mismatch: {path.name}")
        frame = _image(path)
        if hashlib.sha256(frame.tobytes()).hexdigest() != report["rgb_sha256"][index]:
            raise ValueError(f"dense RGB hash mismatch: {path.name}")
        result = tracker.process(frame, motion_allowed=index in motion_frames)
        poses.append(result.model_dump(mode="json"))
        if index % 50 == 0:
            print(json.dumps(result.model_dump(mode="json")), flush=True)
    payload = {
        "suite": "visual_odometry_development",
        "evaluator_loaded": False,
        "focal_pixels": focal_x,
        "reference_focal_pixels": reference_focal,
        "calibration_reference_size": reference_size,
        "observation_size": (frame_width, frame_height),
        "camera_intrinsics": camera.tolist(),
        "focal_sample_count": samples,
        "focal_relative_mad": mad,
        "config": tracker.config.model_dump(mode="json"),
        "opencv_version": cv2.__version__,
        "poses": poses,
        "status_counts": dict(Counter(item["status"] for item in poses)),
        "source_sha256": source_hashes,
        "input_report_sha256": hashlib.sha256(input_report).hexdigest(),
        "input_frames_verified": True,
        "goal_reference_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.run / "goal-reference").glob("view-*.png"))
        },
    }
    if any(path.read_bytes() != source_bytes[path.name] for path in source_paths):
        raise RuntimeError("source changed during development run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "source-snapshot"
    snapshot.mkdir(exist_ok=False)
    for name, data in source_bytes.items():
        (snapshot / name).write_bytes(data)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
