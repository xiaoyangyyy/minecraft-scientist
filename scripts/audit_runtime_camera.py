"""Independently match native RGB bytes to evaluator-only renderer snapshots.

Camera-base timing is not full bobbing/hurt-camera ground truth. No pose model is
selected by accuracy, no historic evaluator is rewritten, and no navigation is
certified. Fail closed on missing, ambiguous or non-contiguous correspondence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.protocols import TransitionRecord

ROOT = Path(__file__).resolve().parents[1]
CAMERA_KEYS = ("xpos", "ypos", "zpos", "yaw", "pitch")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(path):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("runtime timing image is not native RGB")
        return np.array(image)


def match_snapshots(captures, hashes):
    """Match the entire sequence, not each frame's nearest pose or lowest error."""
    if not hashes or len(captures) > 10000:
        raise ValueError("empty or unbounded timing sequence")
    matches = [
        start
        for start in range(len(captures) - len(hashes) + 1)
        if [item["raw_rgb_sha256"] for item in captures[start : start + len(hashes)]]
        == hashes
    ]
    if len(matches) != 1:
        raise ValueError("native RGB sequence has no unique contiguous camera match")
    start = matches[0]
    aligned = captures[start : start + len(hashes)]
    if any(
        type(item["sequence"]) is not int or type(item["tick_counter"]) is not int
        for item in aligned
    ) or any(
        second["sequence"] != first["sequence"] + 1
        or second["tick_counter"] != first["tick_counter"] + 1
        for first, second in pairwise(aligned)
    ):
        raise ValueError("camera recorder counters are not contiguous")
    return start, aligned


def timing_metrics(aligned, positions):
    """Report both current and previous position errors; never choose a winner."""
    if len(aligned) != len(positions) or not aligned:
        raise ValueError("camera and player sample counts differ")
    telemetry_errors, previous_errors, current_errors, angles, previous_angles = (
        [],
        [],
        [],
        [],
        [],
    )
    moving = 0
    for sample, telemetry in zip(aligned, positions, strict=True):
        if (
            any(
                type(sample[key]) is not bool
                for key in ("sync_integrated_server", "disable_recorder", "bob_view")
            )
            or any(
                type(sample[key]) not in (int, float) or not np.isfinite(sample[key])
                for key in (
                    "partial_ticks",
                    "fov_hand",
                    "previous_fov_hand",
                    "player_fov_modifier",
                )
            )
            or not 0 <= sample["partial_ticks"] <= 1
            or any(
                not 0 < sample[key] < 3
                for key in ("fov_hand", "previous_fov_hand", "player_fov_modifier")
            )
        ):
            raise ValueError("invalid renderer settings or focal modifier")
        for pose in (
            sample["player"],
            sample["previous_player"],
            sample["camera_base"],
            telemetry,
        ):
            if (
                set(pose) != set(CAMERA_KEYS)
                or any(
                    type(pose[key]) not in (int, float) or not np.isfinite(pose[key])
                    for key in CAMERA_KEYS
                )
                or not -90 <= pose["pitch"] <= 90
            ):
                raise ValueError("invalid evaluator camera/player pose")
        if (
            type(sample["eye_height"]) not in (int, float)
            or not 0 < sample["eye_height"] < 3
        ):
            raise ValueError("invalid evaluator eye height")

        def vector(pose):
            return np.array([pose[key] for key in CAMERA_KEYS[:3]])

        player, previous, camera = [
            vector(sample[key]) for key in ("player", "previous_player", "camera_base")
        ]
        eye = np.array((0, sample["eye_height"], 0))
        telemetry_errors.append(
            float(
                np.max(
                    np.abs(
                        np.array([telemetry[key] for key in CAMERA_KEYS])
                        - np.array([sample["player"][key] for key in CAMERA_KEYS])
                    )
                )
            )
        )
        current_errors.append(float(np.linalg.norm(camera - player - eye)))
        previous_errors.append(float(np.linalg.norm(camera - previous - eye)))
        moving += int(np.linalg.norm(player - previous) > 1e-5)
        for target, pose in (
            (angles, sample["player"]),
            (previous_angles, sample["previous_player"]),
        ):
            target.append(
                max(
                    abs((sample["camera_base"]["yaw"] - pose["yaw"] + 180) % 360 - 180),
                    abs(sample["camera_base"]["pitch"] - pose["pitch"]),
                )
            )
    return {
        "samples": len(aligned),
        "nonstationary_samples": moving,
        "max_telemetry_vs_capture_player_component_error": max(telemetry_errors),
        "max_camera_vs_current_player_eye_distance": max(current_errors),
        "max_camera_vs_previous_player_eye_distance": max(previous_errors),
        "max_camera_vs_current_player_angle_error_degrees": max(angles),
        "max_camera_vs_previous_player_angle_error_degrees": max(previous_angles),
        "partial_ticks": sorted({sample["partial_ticks"] for sample in aligned}),
        "sync_integrated_server": sorted(
            {sample["sync_integrated_server"] for sample in aligned}
        ),
        "disable_recorder": sorted({sample["disable_recorder"] for sample in aligned}),
        "bob_view": sorted({sample["bob_view"] for sample in aligned}),
        "fov_hand_range": [
            min(sample["fov_hand"] for sample in aligned),
            max(sample["fov_hand"] for sample in aligned),
        ],
        "previous_fov_hand_range": [
            min(sample["previous_fov_hand"] for sample in aligned),
            max(sample["previous_fov_hand"] for sample in aligned),
        ],
        "player_fov_modifier_range": [
            min(sample["player_fov_modifier"] for sample in aligned),
            max(sample["player_fov_modifier"] for sample in aligned),
        ],
    }


def public_data(directory, engine, *, registration=None):
    """Validate RGB and action receipts without opening any private pose file."""
    config = json.loads((directory / "config.json").read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    # Source-snapshot-only external SDK checks do not pretend to verify a different
    # installation. The project code and supplied engine are verified explicitly.
    project = {
        "capture_runtime_camera.py": ROOT
        / "environments/minestudio/capture_runtime_camera.py",
        "CameraAudit.java": ROOT
        / "environments/minestudio/camera_audit_agent/CameraAudit.java",
        "MANIFEST.MF": ROOT / "environments/minestudio/camera_audit_agent/MANIFEST.MF",
        "probe-recorder.py": ROOT
        / "environments/minestudio/capture_controlled_rgb_motion.py",
        "position-audit.py": ROOT / "environments/minestudio/navigation_probe.py",
    } | {
        "package/" + path.relative_to(ROOT / "src/minecraft_scientist").as_posix(): path
        for path in sorted((ROOT / "src/minecraft_scientist").rglob("*.py"))
    }
    expected_checks = {
        "one_fresh_instrumented_launch",
        "complete_frames_actions_positions",
        "sdk_runtime_unchanged",
        "engine_agent_unchanged",
        "sources_unchanged",
        "config_unchanged",
    }
    # Import only a Python definition file, never MineStudio or a JVM at audit time.
    import importlib.util

    definition = importlib.util.spec_from_file_location(
        "runtime_capture_definition", project["capture_runtime_camera.py"]
    )
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    extra_artifacts = set()
    if registration is None:
        expected_schedule = [list(action) for action in module.SCHEDULE]
        registered_schedule = config["schema"] == "runtime-camera-projection-v2" and (
            config["schedule"] == expected_schedule
            and config["reference_indices"] == list(module.REFERENCE_INDICES)
            and config["trial_intervals"]
            == [list(item) for item in module.TRIAL_INTERVALS]
        )
    else:
        # An explicit source-bound caller must independently replay a dynamic
        # public policy. Default fixed-capture validation is never relaxed.
        registered = registration(directory, config, report, module)
        expected_schedule = registered["schedule"]
        registered_schedule = registered["valid"] is True
        extra_artifacts = set(registered["artifacts"])
        expected_checks |= set(registered["capture_checks"])
    action_cap = 200
    if (
        config.get("schema")
        in (
            "stable-keyframe-camera-projection-v1",
            "micro-stable-keyframe-camera-projection-v1",
            "responsive-stable-keyframe-camera-projection-v1",
            "recovering-stable-keyframe-camera-projection-v1",
            "safe-seed-stable-keyframe-camera-projection-v1",
            "jump-safe-seed-stable-keyframe-camera-projection-v1",
            "long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1",
        )
        and registration is not None
    ):
        from minecraft_scientist.exploration.stable_keyframe_map import (
            StableJumpSafeSeedKeyframePolicy,
            StableKeyframePolicy,
            StableLongWaitJumpKeyframePolicy,
            StableMicroKeyframePolicy,
            StableRecoveringKeyframePolicy,
            StableResponsiveKeyframePolicy,
            StableSafeSeedKeyframePolicy,
        )

        policy_type = (
            StableLongWaitJumpKeyframePolicy
            if config["schema"]
            == "long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1"
            else StableJumpSafeSeedKeyframePolicy
            if config["schema"] == "jump-safe-seed-stable-keyframe-camera-projection-v1"
            else StableSafeSeedKeyframePolicy
            if config["schema"] == "safe-seed-stable-keyframe-camera-projection-v1"
            else StableRecoveringKeyframePolicy
            if config["schema"] == "recovering-stable-keyframe-camera-projection-v1"
            else StableResponsiveKeyframePolicy
            if config["schema"] == "responsive-stable-keyframe-camera-projection-v1"
            else StableMicroKeyframePolicy
            if config["schema"] == "micro-stable-keyframe-camera-projection-v1"
            else StableKeyframePolicy
        )
        if (
            json.dumps(config.get("keyframe_policy"), sort_keys=True, allow_nan=False)
            != json.dumps(
                policy_type().model_dump(mode="json"), sort_keys=True, allow_nan=False
            )
            or registered.get("maximum_actions") != 288
        ):
            raise ValueError(
                "keyframes require their exact separate action registration"
            )
        action_cap = 288
    if not 1 <= len(expected_schedule) <= action_cap:
        raise ValueError("runtime registration exceeded bounded action count")
    count = len(expected_schedule) + 1
    checks = {
        "complete_capture": report["completed"] is True
        and report["failure"] is None
        and report["navigation_accepted"] is False
        and report["frame_count"] == count
        and set(report["checks"]) == expected_checks
        and all(value is True for value in report["checks"].values()),
        "config_identity": digest(directory / "config.json") == report["config_sha256"],
        "registered_schedule": registered_schedule
        and config["render_size"] == [1280, 720]
        and config["observation_size"] == [448, 448],
        "complete_source_snapshots": {
            path.relative_to(directory / "source-snapshot").as_posix(): digest(path)
            for path in (directory / "source-snapshot").rglob("*")
            if path.is_file()
        }
        == config["source_sha256"],
        "current_project_sources": set(config["source_sha256"])
        == set(project) | {"sdk-entry.py", "sdk-launcher.py"}
        and all(
            config["source_sha256"].get(name) == digest(path)
            for name, path in project.items()
        ),
        "engine_agent_identity": digest(engine) == config["engine_sha256"]
        and digest(directory / "camera-audit.jar") == config["agent_sha256"],
    }
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    names = {f"frame-{slot:05d}.png" for slot in range(count)}
    checks["complete_image_sets"] = all(
        {path.name for path in (directory / folder).iterdir()} == names
        for folder in ("frames", "evaluator-native")
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    public_text = (directory / "events.jsonl").read_text(encoding="utf-8")
    checks["public_log_clean"] = not any(
        token in public_text
        for token in (
            "location_stats",
            "xpos",
            "ypos",
            "zpos",
            '"yaw"',
            '"pitch"',
            "camera_base",
        )
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    events = [
        TransitionRecord.model_validate_json(line) for line in public_text.splitlines()
    ]
    public_hashes = []
    for slot in range(count):
        image = load_image(directory / "frames" / f"frame-{slot:05d}.png")
        if image.shape != (448, 448, 3):
            raise ValueError("incorrect public image dimensions")
        public_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())

    def scheduled_target(event, action):
        kind, index = action
        if kind == "wait":
            return event.executed_action.target_id is None
        if kind == "move":
            targets = [
                obj.object_id
                for obj in event.state_before.objects
                if tuple(value.value for value in obj.affordances) == ("move",)
            ]
            return (
                len(targets) == 4 and event.executed_action.target_id == targets[index]
            )
        targets = [
            obj.object_id
            for obj in event.state_before.objects
            if any(value.value == "look" for value in obj.affordances)
            and obj.observable_features.get("screen_row") == 1
            and obj.observable_features.get("screen_column") == 2
        ]
        return len(targets) == 1 and event.executed_action.target_id == targets[0]

    checks["every_public_action_and_frame_receipt"] = len(events) == len(
        expected_schedule
    ) and all(
        event.executed_action.kind.value == expected_schedule[index][0]
        and scheduled_target(event, expected_schedule[index])
        and event.executed_action.ticks == event.event.duration == 1
        and event.intervention_compliance is True
        and event.event.success is True
        and event.controller_failure is None
        and event.state_before.agent_state.get("frame_sha256") == public_hashes[index]
        and event.state_after.agent_state.get("frame_sha256")
        == public_hashes[index + 1]
        for index, event in enumerate(events)
    )
    expected_files = (
        extra_artifacts
        | {
            "events.jsonl",
            "evaluator-positions.json",
            "evaluator-camera.jsonl",
        }
        | {
            f"{folder}/{name}"
            for folder in ("frames", "evaluator-native")
            for name in names
        }
    )
    checks["complete_artifact_ledger"] = (
        set(report["artifact_sha256"]) == expected_files
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    private_names = {"evaluator-positions.json", "evaluator-camera.jsonl"}
    paths = [directory / name for name in expected_files - private_names]
    frozen = {path: digest(path) for path in paths}
    checks["public_artifact_bytes_verified"] = {
        path.relative_to(directory).as_posix(): value for path, value in frozen.items()
    } == {
        name: value
        for name, value in report["artifact_sha256"].items()
        if name not in private_names
    }
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    frozen.update(
        {
            path: digest(path)
            for path in list(project.values())
            + [
                engine,
                directory / "camera-audit.jar",
                directory / "config.json",
                directory / "report.json",
                Path(__file__),
            ]
            + [
                path
                for path in (directory / "source-snapshot").rglob("*")
                if path.is_file()
            ]
        }
    )
    import cv2

    native_hashes = []
    for slot in range(count):
        native = load_image(directory / "evaluator-native" / f"frame-{slot:05d}.png")
        image = load_image(directory / "frames" / f"frame-{slot:05d}.png")
        if native.shape != (720, 1280, 3) or not np.array_equal(
            image, cv2.resize(native, (448, 448), interpolation=cv2.INTER_LINEAR)
        ):
            raise ValueError("SDK resize cannot be independently reproduced")
        native_hashes.append(hashlib.sha256(native[::-1].tobytes()).hexdigest())
    checks["all_public_native_resize_replayed"] = True
    checks["public_bytes_stable"] = all(
        digest(path) == value for path, value in frozen.items()
    )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "config": config,
        "report": report,
        "frozen": frozen,
        "native_hashes": native_hashes,
        "count": count,
    }


def audit(directory, engine, *, registration=None):
    public = (
        public_data(directory, engine)
        if registration is None
        else public_data(directory, engine, registration=registration)
    )
    if not public["valid"]:
        return {"valid": False, "checks": public["checks"], "timing": None}
    checks, config, report = public["checks"], public["config"], public["report"]
    frozen, count, native_hashes = (
        public["frozen"],
        public["count"],
        public["native_hashes"],
    )
    # First private bytes read, after all public observations and receipts pass.
    private_names = ("evaluator-positions.json", "evaluator-camera.jsonl")
    private_hashes = {name: digest(directory / name) for name in private_names}
    checks["private_artifact_bytes_verified"] = all(
        report["artifact_sha256"][name] == value
        for name, value in private_hashes.items()
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "timing": None}
    frozen.update({directory / name: value for name, value in private_hashes.items()})
    private = json.loads((directory / "evaluator-positions.json").read_bytes())
    rows = [
        json.loads(line)
        for line in (directory / "evaluator-camera.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    headers = [row for row in rows if row["kind"] == "instrumented"]
    if (
        len(headers) != 2
        or {json.dumps(row, sort_keys=True) for row in headers}
        != {
            json.dumps(
                {"kind": "instrumented", "target": target, "hooks": 1}, sort_keys=True
            )
            for target in ("recorder", "world_renderer")
        }
        or any(row["kind"] not in ("capture", "instrumented") for row in rows)
    ):
        raise ValueError("instrumentation missing or contains an error")
    if not len(private["positions"]) == len(private["native_rgb_sha256"]) == count:
        raise ValueError("incomplete private sequence")
    if native_hashes != private["native_rgb_sha256"]:
        raise ValueError("native bytes disagree with evaluator ledger")
    start, aligned = match_snapshots(
        [row for row in rows if row["kind"] == "capture"], native_hashes
    )
    if any(
        sample["camera_valid"] is not True
        or sample["third_person"] is not False
        or (sample["width"], sample["height"]) != (1280, 720)
        for sample in aligned
    ):
        raise ValueError("camera is invalid or not first person")
    timing = timing_metrics(aligned, private["positions"])
    from minecraft_scientist.evaluation.relative_motion import camera_rotation
    from minecraft_scientist.evaluation.rendered_camera import decompose_rendered_camera

    rendered = [
        decompose_rendered_camera(
            sample["render"], observation_size=tuple(config["observation_size"])
        )
        for sample in aligned
    ]
    if any(
        second.render_sequence != first.render_sequence + 1
        for first, second in pairwise(rendered)
    ) or any(
        not np.allclose(
            sample["render"]["base_position"],
            [sample["camera_base"][key] for key in CAMERA_KEYS[:3]],
            rtol=0,
            atol=1e-7,
        )
        or sample["render"]["partial_ticks"] != sample["partial_ticks"]
        for sample in aligned
    ):
        raise ValueError(
            "render matrices do not bind consecutive captured camera bases"
        )
    rotation_offsets = []
    for camera, sample in zip(rendered, aligned, strict=True):
        base = sample["camera_base"]
        expected = camera_rotation(base["yaw"], base["pitch"])
        rotation_offsets.append(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(np.array(camera.rotation) @ expected.T) - 1) / 2,
                            -1,
                            1,
                        )
                    )
                )
            )
        )
    intrinsic = np.array([camera.camera for camera in rendered])
    projection_metrics = {
        "maximum_center_offset_blocks": max(
            float(np.linalg.norm(camera.camera_center_offset)) for camera in rendered
        ),
        "maximum_base_rotation_offset_degrees": max(rotation_offsets),
        "maximum_reconstruction_error": max(
            camera.reconstruction_error for camera in rendered
        ),
        "fx_range": [float(intrinsic[:, 0, 0].min()), float(intrinsic[:, 0, 0].max())],
        "fy_range": [float(intrinsic[:, 1, 1].min()), float(intrinsic[:, 1, 1].max())],
        "maximum_absolute_skew_pixels": float(np.abs(intrinsic[:, 0, 1]).max()),
        "principal_point_range": [
            [float(intrinsic[:, axis, 2].min()), float(intrinsic[:, axis, 2].max())]
            for axis in (0, 1)
        ],
    }
    checks["native_pixels_resize_and_unique_contiguous_camera_match"] = True
    checks["reported_telemetry_matches_capture_player"] = (
        timing["max_telemetry_vs_capture_player_component_error"] < 1e-5
    )
    checks["complete_render_matrix_decomposition_and_alignment"] = True
    checks["all_bytes_stable_through_analysis"] = all(
        digest(path) == value for path, value in frozen.items()
    )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "timing": timing,
        "projection_metrics": projection_metrics,
        "rendered_cameras": [camera.model_dump(mode="json") for camera in rendered],
        "matching_capture_start": start,
        "navigation_accepted": False,
        "scope": config["scope"],
        "audit_source_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.output.with_suffix(".source.py")
    if args.output.exists() or snapshot.exists():
        raise FileExistsError("refusing to overwrite runtime camera audit")
    snapshot.write_bytes(Path(__file__).read_bytes())
    result = audit(args.directory, args.engine_jar)
    result["audit_snapshot_sha256"] = digest(snapshot)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": result["valid"], "timing": result["timing"]}))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
