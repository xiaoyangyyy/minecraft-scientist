"""Preregister continuous RGB tracks and multi-view motion without telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from probe_dense_correspondence import public_pairs

from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.correspondence_refinement import _crop_gray
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_stability import _pose
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    estimate_multiview_motion,
)
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)
from minecraft_scientist.exploration.two_view_motion import estimate_two_view_motion
from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _opencv,
)
from minecraft_scientist.exploration.visual_odometry import triangulate_landmarks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sources() -> dict[str, Path]:
    return {
        Path(path).name: Path(path)
        for path in (
            CameraResizeSpec.camera.__code__.co_filename,
            _crop_gray.__code__.co_filename,
            DenseCorrespondences._aligned.__code__.co_filename,
            _pose.__code__.co_filename,
            estimate_multiview_motion.__code__.co_filename,
            track_frame_chain.__code__.co_filename,
            estimate_two_view_motion.__code__.co_filename,
            _features.__code__.co_filename,
            triangulate_landmarks.__code__.co_filename,
            public_pairs.__code__.co_filename,
            __file__,
        )
    }


def frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def chain_indices(pair: dict) -> tuple[int, ...]:
    first, second = pair["first"], pair["second"]
    if (
        not isinstance(first, int)
        or not isinstance(second, int)
        or first < 0
        or second < first
    ):
        raise ValueError(
            "public temporal endpoints must be ordered nonnegative indices"
        )
    return (first, first) if first == second else tuple(range(first, second + 1))


def temporal_pairs(acquisition: dict, window_frames: int = 0) -> list[dict]:
    if window_frames != 0 and not 3 <= window_frames <= 32:
        raise ValueError(
            "temporal window must be zero (full intervention) or 3–32 images"
        )
    pairs = []
    for pair in public_pairs(acquisition):
        indices = chain_indices(pair)
        if (
            not window_frames
            or pair["kind"] != "motion_intervention"
            or len(indices) <= window_frames
        ):
            pairs.append(pair)
            continue
        for start in range(len(indices) - window_frames + 1):
            pairs.append(
                {
                    **pair,
                    "id": f"{pair['id']}-window-{start:02d}",
                    "parent_pair_id": pair["id"],
                    "first": indices[start],
                    "second": indices[start + window_frames - 1],
                }
            )
    return pairs


def infer_pair(route: Path, pair: dict, config: dict, index: int) -> dict:
    indices = chain_indices(pair)
    tracks = track_frame_chain(
        [frame(route / "frames" / f"frame-{slot:05d}.png") for slot in indices],
        config=TemporalTrackConfig.model_validate(config["temporal"]),
    )
    motion = estimate_multiview_motion(
        tracks,
        np.asarray(config["camera_intrinsics"]),
        config=MultiViewMotionConfig.model_validate(config["multiview"]),
        seed=config["seed"] + index,
    )
    return {
        "pair": pair,
        "chain_indices": list(indices),
        "tracks": tracks.model_dump(mode="json"),
        "motion": motion.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--render-size", type=int, nargs=2)
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference_config.read_bytes())
    acquisition = json.loads((args.route / "report.json").read_bytes())
    acquisition_hash = file_hash(args.route / "report.json")
    if (
        acquisition["gate"]["passed"] is not True
        or acquisition_hash != reference["input_report_sha256"]
    ):
        raise ValueError("temporal inputs require their controlled acquisition")
    if public_pairs(acquisition) != reference["pairs"]:
        raise ValueError("temporal pairs must match the public baseline selection")
    pairs = temporal_pairs(acquisition, args.window_frames)
    frame_paths = {
        f"frame-{slot:05d}.png": args.route / "frames" / f"frame-{slot:05d}.png"
        for pair in pairs
        for slot in chain_indices(pair)
    }
    frame_hashes = {name: file_hash(path) for name, path in frame_paths.items()}
    if any(
        digest != acquisition["frame_sha256"].get(name)
        for name, digest in frame_hashes.items()
    ):
        raise ValueError("temporal frame hash mismatch")
    reference_paths = {
        path.name: path
        for path in sorted((args.route / "goal-reference").glob("view-*.png"))
    }
    reference_hashes = {name: file_hash(path) for name, path in reference_paths.items()}
    if reference_hashes != reference["reference_sha256"]:
        raise ValueError("public calibration view ledger mismatch")
    cv2 = _opencv()
    images = [frame(path) for path in reference_paths.values()]
    geometry = VisualGeometryConfig()
    _, ref_width = images[0].shape[:2]
    original_features = [_features(image, geometry, cv2) for image in images]
    reference_features = (
        [
            (points + (0.5, 0.5), descriptors)
            for points, descriptors in original_features
        ]
        if "camera_resize" in reference
        else original_features
    )
    focal, count, mad = _calibrate_focal(reference_features, ref_width, geometry, cv2)
    if (
        focal is None
        or not np.isclose(focal, reference["reference_focal_pixels"])
        or count != reference["reference_focal_sample_count"]
        or not np.isclose(mad, reference["reference_focal_relative_mad"])
    ):
        raise ValueError("public focal calibration does not replay")
    height, width = frame(next(iter(frame_paths.values()))).shape[:2]
    # Legacy calibration centers at width/2. Shifting public point coordinates
    # by half a pixel gives the actual pixel-grid center without changing the
    # calibrator or accessing any evaluator measurement.
    shifted_features = [
        (points + (0.5, 0.5), descriptors) for points, descriptors in original_features
    ]
    corrected_focal, corrected_count, corrected_mad = _calibrate_focal(
        shifted_features, ref_width, geometry, cv2
    )
    if corrected_focal is None:
        raise ValueError("pixel-center focal calibration was rejected")
    render_size = args.render_size or acquisition.get("render_size")
    if render_size is None:
        raise ValueError(
            "legacy acquisitions require explicit public --render-size WIDTH HEIGHT"
        )
    if (
        acquisition.get("render_size") is not None
        and list(render_size) != acquisition["render_size"]
    ):
        raise ValueError("explicit render size disagrees with acquisition metadata")
    resize = CameraResizeSpec(
        render_size=tuple(render_size), observation_size=(width, height)
    )
    camera = resize.camera(corrected_focal * width / ref_width).tolist()
    sources = runtime_sources()
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    source_hashes = {name: file_hash(snapshot / name) for name in sources}
    temporal, multiview = TemporalTrackConfig(), MultiViewMotionConfig()
    if any(len(chain_indices(pair)) > temporal.maximum_frames for pair in pairs):
        raise ValueError("public pair exceeds preregistered temporal frame budget")
    reference_config_hash = file_hash(args.reference_config)
    config = {
        "scope": "single-seed temporal development; no navigation promotion",
        "seed": args.seed,
        "window_frames": args.window_frames,
        "pairs": pairs,
        "temporal": temporal.model_dump(mode="json"),
        "multiview": multiview.model_dump(mode="json"),
        "camera_intrinsics": camera,
        "camera_resize": resize.model_dump(mode="json"),
        "render_size_source": "acquisition_report"
        if acquisition.get("render_size") is not None
        else "explicit_client_configuration",
        "reference_horizontal_focal_pixels": corrected_focal,
        "reference_focal_sample_count": corrected_count,
        "reference_focal_relative_mad": corrected_mad,
        "reference_config_sha256": reference_config_hash,
        "acquisition_sha256": acquisition_hash,
        "frame_sha256": frame_hashes,
        "reference_sha256": reference_hashes,
        "source_sha256": source_hashes,
        "opencv_version": cv2.__version__,
        "evaluation_thresholds": reference["evaluation_thresholds"],
    }
    config_path = args.output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = file_hash(config_path)
    results, failure = [], None
    try:
        for index, pair in enumerate(pairs):
            item = infer_pair(args.route, pair, config, index)
            results.append(item)
            print(
                json.dumps(
                    {
                        "pair": pair["id"],
                        "seed_points": item["tracks"]["seed_count"],
                        "survival": item["tracks"]["surviving_counts"],
                        "whole_chain_tracks": len(item["tracks"]["track_ids"]),
                        "endpoint_supported": item["motion"]["candidate"]["supported"],
                        "multiview_supported": item["motion"]["supported"],
                        "reason": item["motion"]["reason"],
                    }
                ),
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 -- preserve failed inference and exit nonzero
        failure = {"type": type(error).__name__, "message": str(error)}
    unchanged = (
        file_hash(config_path) == config_hash
        and file_hash(args.reference_config) == reference_config_hash
        and file_hash(args.route / "report.json") == acquisition_hash
        and all(
            file_hash(path) == source_hashes[name] for name, path in sources.items()
        )
        and all(
            file_hash(path) == frame_hashes[name] for name, path in frame_paths.items()
        )
        and all(
            file_hash(path) == reference_hashes[name]
            for name, path in reference_paths.items()
        )
    )
    report = {
        "config_sha256": config_hash,
        "results": results,
        "failure": failure,
        "sources_and_inputs_unchanged": unchanged,
        "evaluator_file_loaded": False,
        "navigation_accepted": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failure or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
