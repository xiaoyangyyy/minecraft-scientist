"""Public-only integrity and inference helpers for fixed motion diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.correspondence_refinement import _crop_gray
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)
from minecraft_scientist.exploration.motion_stability import _pose
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    estimate_multiview_motion,
    estimate_temporal_endpoint,
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
from minecraft_scientist.minecraft import MineStudioVisualAdapter
from minecraft_scientist.protocols import TransitionRecord


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def capture_sources() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "capture_controlled_rgb_motion.py": root
        / "environments/minestudio/capture_controlled_rgb_motion.py",
        "motion_probe_schedule.py": Path(motion_probe_schedule.__code__.co_filename),
        "minestudio.py": Path(MineStudioVisualAdapter.step.__code__.co_filename),
        "navigation_probe.py": root / "environments/minestudio/navigation_probe.py",
    }


def public_inputs(route: Path) -> dict[str, Path]:
    paths = {
        name: route / name for name in ("config.json", "report.json", "events.jsonl")
    }
    for directory, pattern in (
        ("frames", "frame-*.png"),
        ("goal-reference", "view-*.png"),
        ("source-snapshot", "*.py"),
    ):
        paths.update(
            {
                f"{directory}/{path.name}": path
                for path in sorted((route / directory).glob(pattern))
            }
        )
    return paths


def validate_acquisition(route: Path) -> tuple[MotionProbeConfig, dict, dict]:
    """No evaluator file is opened here, even on invalid or partial acquisition."""
    manifest = json.loads((route / "config.json").read_bytes())
    report = json.loads((route / "report.json").read_bytes())
    config = MotionProbeConfig.model_validate(manifest["config"])
    timeline, pairs = motion_probe_timeline(config), motion_probe_pairs(config)
    sources = capture_sources()
    hashes = manifest["source_sha256"]
    checks = {
        "acquisition_suite": report["suite"] == "controlled_rgb_motion_acquisition",
        "acquisition_passed": report["gate"]["passed"] is True
        and all(report["gate"].values())
        and report["failure"] is None,
        "acquisition_preregistration": digest(route / "config.json")
        == report["config_sha256"],
        "fixed_schedule": manifest["schedule"]
        == [item.model_dump(mode="json") for item in motion_probe_schedule(config)]
        and report["timeline"] == timeline,
        "complete_pair_selection": manifest["pairs"] == report["pairs"] == pairs,
        "public_render_dimensions": manifest["render_size"] == list(config.render_size),
        "capture_source_set": set(hashes) == set(sources) | {"sdk-entry.py"}
        and {path.name for path in (route / "source-snapshot").glob("*.py")}
        == set(hashes),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid public acquisition: {checks}")
    checks["capture_sources_current"] = all(
        digest(path) == hashes[name] for name, path in sources.items()
    )
    checks["capture_snapshots_verified"] = all(
        digest(route / "source-snapshot" / name) == value
        for name, value in hashes.items()
    )
    count = timeline[-1]["end_frame"] + 1
    expected = {f"frame-{slot:05d}.png" for slot in range(count)}
    reference_indices = [
        item["end_frame"]
        for item in timeline
        if item["scheduled"]["phase"] == "reference_settle"
    ]
    reference_names = {f"view-{number:02d}.png" for number in range(12)}
    checks["complete_frame_ledger"] = (
        report["frame_count"] == count
        and len(report["rgb_sha256"]) == count
        and set(report["frame_sha256"]) == expected
        and {path.name for path in (route / "frames").glob("frame-*.png")} == expected
    )
    checks["complete_reference_ledger"] = (
        report["reference_indices"] == reference_indices
        and set(report["reference_sha256"]) == reference_names
        and {path.name for path in (route / "goal-reference").glob("view-*.png")}
        == reference_names
    )
    if not all(checks.values()):
        raise ValueError(f"incomplete or stale public acquisition: {checks}")
    checks["frame_bytes_verified"] = all(
        digest(route / "frames" / name) == value
        for name, value in report["frame_sha256"].items()
    )
    checks["reference_bytes_verified"] = all(
        digest(route / "goal-reference" / name) == value
        and value
        == report["frame_sha256"][f"frame-{reference_indices[number]:05d}.png"]
        for number, (name, value) in enumerate(
            sorted(report["reference_sha256"].items())
        )
    )
    raw_hashes = []
    shapes = []
    for slot in range(count):
        image = rgb(route / "frames" / f"frame-{slot:05d}.png")
        raw_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        shapes.append(image.shape)
    checks["raw_rgb_verified"] = raw_hashes == report["rgb_sha256"] and all(
        shape == (config.observation_size, config.observation_size, 3)
        for shape in shapes
    )
    text = (route / "events.jsonl").read_text(encoding="utf-8")
    records = [
        TransitionRecord.model_validate_json(line) for line in text.splitlines() if line
    ]
    checks["public_events_clean"] = not any(
        token in text
        for token in ("location_stats", "xpos", "ypos", "zpos", '"yaw"', '"pitch"')
    )
    checks["action_frame_alignment"] = len(records) == len(timeline) and len(
        {item.transition_id for item in records}
    ) == len(records)
    if checks["action_frame_alignment"]:
        checks["action_frame_alignment"] = all(
            item.event.duration == action.ticks
            and item.executed_action.kind.value == action.kind
            and item.executed_action.ticks == action.ticks
            and item.intervention_compliance
            and item.event.success
            and item.state_before.agent_state["frame_sha256"]
            == raw_hashes[span["start_frame"]]
            and item.state_after.agent_state["frame_sha256"]
            == raw_hashes[span["end_frame"]]
            for item, action, span in zip(
                records, motion_probe_schedule(config), timeline, strict=True
            )
        )
    checks["scheduled_targets_verified"] = len(records) == len(timeline)
    if checks["scheduled_targets_verified"]:
        for item, scheduled in zip(records, motion_probe_schedule(config), strict=True):
            options = item.state_before.objects
            if scheduled.kind == "move":
                handles = [
                    obj.object_id
                    for obj in options
                    if tuple(kind.value for kind in obj.affordances) == ("move",)
                ]
                expected_target = (
                    handles[scheduled.motion_handle_index]
                    if len(handles) == 4
                    else "invalid"
                )
            elif scheduled.kind == "look":
                handles = [
                    obj.object_id
                    for obj in options
                    if any(kind.value == "look" for kind in obj.affordances)
                    and obj.observable_features.get("screen_row") == 1
                    and obj.observable_features.get("screen_column") == 2
                ]
                expected_target = handles[0] if len(handles) == 1 else "invalid"
            else:
                expected_target = None
            checks["scheduled_targets_verified"] &= (
                item.executed_action.target_id == expected_target
                and item.executed_action.item_id is None
            )
    if not all(checks.values()):
        raise ValueError(f"public acquisition failed verification: {checks}")
    return config, report, checks


def runtime_sources() -> dict[str, Path]:
    paths = (
        CameraResizeSpec.camera.__code__.co_filename,
        _crop_gray.__code__.co_filename,
        DenseCorrespondences._aligned.__code__.co_filename,
        _pose.__code__.co_filename,
        estimate_multiview_motion.__code__.co_filename,
        track_frame_chain.__code__.co_filename,
        estimate_two_view_motion.__code__.co_filename,
        _features.__code__.co_filename,
        triangulate_landmarks.__code__.co_filename,
        motion_probe_schedule.__code__.co_filename,
        MineStudioVisualAdapter.step.__code__.co_filename,
        __file__,
    )
    return {Path(path).name: Path(path) for path in paths} | {
        "probe_controlled_rgb_motion.py": Path(__file__).with_name(
            "probe_controlled_rgb_motion.py"
        )
    }


def inference_config(
    route: Path, acquisition: MotionProbeConfig, report: dict, seed: int = 0
) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("inference seed must be a nonnegative integer")
    cv2 = _opencv()
    geometry = VisualGeometryConfig()
    images = [
        rgb(route / "goal-reference" / f"view-{number:02d}.png") for number in range(12)
    ]
    features = [
        (points + (0.5, 0.5), descriptors)
        for points, descriptors in (_features(image, geometry, cv2) for image in images)
    ]
    focal, count, mad = _calibrate_focal(
        features, acquisition.observation_size, geometry, cv2
    )
    if focal is None:
        raise ValueError("public RGB focal calibration rejected")
    resize = CameraResizeSpec(
        render_size=acquisition.render_size,
        observation_size=(acquisition.observation_size,) * 2,
    )
    return {
        "seed": seed,
        "pairs": report["pairs"],
        "camera_resize": resize.model_dump(mode="json"),
        "camera_intrinsics": resize.camera(focal).tolist(),
        "focal_pixels": focal,
        "focal_sample_count": count,
        "focal_relative_mad": mad,
        "temporal": TemporalTrackConfig().model_dump(mode="json"),
        "multiview": MultiViewMotionConfig().model_dump(mode="json"),
        "opencv_version": cv2.__version__,
        "evaluation_thresholds": {
            "minimum_translation_blocks": 0.15,
            "maximum_direction_error_degrees": 20.0,
            "maximum_rotation_error_degrees": 5.0,
        },
        "scope": "single-seed controlled development; moving/settled share trials; not navigation acceptance",
    }


def infer_pair(route: Path, pair: dict, config: dict, index: int) -> dict:
    first, second = pair["first"], pair["second"]
    indices = (first, first) if first == second else tuple(range(first, second + 1))
    frames = [rgb(route / "frames" / f"frame-{slot:05d}.png") for slot in indices]
    temporal = TemporalTrackConfig.model_validate(config["temporal"])
    multiview = MultiViewMotionConfig.model_validate(config["multiview"])
    camera = np.asarray(config["camera_intrinsics"])
    tracks = track_frame_chain(frames, config=temporal)
    motion = estimate_multiview_motion(
        tracks, camera, config=multiview, seed=config["seed"] + index
    )
    endpoints = track_frame_chain([frames[0], frames[-1]], config=temporal)
    endpoint = estimate_temporal_endpoint(
        endpoints.endpoint_correspondences(),
        camera,
        config=multiview.two_view,
        seed=config["seed"] + index,
    )
    return {
        "pair": pair,
        "chain_indices": list(indices),
        "tracks": tracks.model_dump(mode="json"),
        "motion": motion.model_dump(mode="json"),
        "endpoint_tracks": endpoints.model_dump(mode="json"),
        "endpoint_motion": endpoint.model_dump(mode="json"),
    }
