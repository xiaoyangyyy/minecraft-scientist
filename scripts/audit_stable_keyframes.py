"""Replay all public sampled-keyframe decisions before any private pose scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_runtime_camera as runtime
import numpy as np

from minecraft_scientist.evaluation.relative_motion import (
    relative_motion_truth,
    score_relative_motion,
)
from minecraft_scientist.evaluation.rendered_camera import (
    rendered_motion_truth,
    score_rendered_motion,
)
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
)
from minecraft_scientist.exploration.settled_map_control import PublicControlStep
from minecraft_scientist.exploration.stable_keyframe_map import (
    StableKeyframePolicy,
    acquire_stable_keyframe_map,
)
from minecraft_scientist.protocols import TransitionRecord

THRESHOLDS = {
    "minimum_translation": 0.15,
    "maximum_direction_error": 20,
    "maximum_rotation_error": 5,
}


def serializable_truth(truth):
    """Keep evaluator rotations/directions explicit and finite in persisted JSON."""
    rotation = np.asarray(truth["relative_rotation"])
    direction = truth["unit_translation"]
    if (
        not np.isfinite(truth["distance"])
        or not np.isfinite(rotation).all()
        or (direction is not None and not np.isfinite(direction).all())
    ):
        raise ValueError("stability diagnostics cannot persist nonfinite truth")
    return {
        "distance": float(truth["distance"]),
        "relative_rotation": rotation.tolist(),
        "unit_translation": None
        if direction is None
        else np.asarray(direction).tolist(),
    }


def registration(directory, config, report, definition):
    settings = StableKeyframePolicy()
    baseline = [list(item) for item in definition.SCHEDULE[:48]]
    result = {
        "valid": False,
        "schedule": baseline,
        "artifacts": ["keyframe-acquisition.json"],
        "capture_checks": ["keyframe_policy_completed"],
        "maximum_actions": 288,
    }
    count = report.get("frame_count")
    if (
        config.get("schema") != "stable-keyframe-camera-projection-v1"
        or config.get("keyframe_policy") != settings.model_dump(mode="json")
        or config.get("active_seed") != 2000
        or config.get("schedule") != baseline
        or config.get("reference_indices") != list(definition.REFERENCE_INDICES)
        or config.get("trial_intervals") != []
        or any(
            name in config
            for name in (
                "active_policy",
                "settled_policy",
                "recovery_policy",
                "maintenance_policy",
                "gauge_policy",
                "shared_anchor_policy",
                "shared_endpoint_policy",
            )
        )
        or type(count) is not int
        or not 49 <= count <= 289
    ):
        return result
    paths = [
        directory / "config.json",
        directory / "report.json",
        directory / "events.jsonl",
        directory / "keyframe-acquisition.json",
    ]
    paths += [directory / "frames" / f"frame-{slot:05d}.png" for slot in range(count)]
    before = {path: runtime.digest(path) for path in paths}
    images = [runtime.load_image(path) for path in paths[4:]]
    if any(image.shape != (448, 448, 3) for image in images):
        raise ValueError("keyframe replay requires registered public RGB")
    for path in paths[2:]:
        if (
            before[path]
            != report["artifact_sha256"][path.relative_to(directory).as_posix()]
        ):
            raise ValueError("public replay artifact differs from capture hash")
    events = [
        TransitionRecord.model_validate_json(line)
        for line in (directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    saved = json.loads((directory / "keyframe-acquisition.json").read_bytes())
    calibration = calibrate_reference_camera(
        [images[slot] for slot in definition.REFERENCE_INDICES], (1280, 720)
    )
    cursor, requested = 48, list(baseline)

    def execute(kind, handle):
        nonlocal cursor
        if cursor >= len(events) or cursor + 1 >= len(images):
            raise ValueError("public run ended before a keyframe control request")
        event = events[cursor]
        targets = [
            obj.object_id
            for obj in event.state_before.objects
            if tuple(action.value for action in obj.affordances) == ("move",)
        ]
        actual = (
            targets.index(event.executed_action.target_id)
            if event.executed_action.target_id in targets
            else None
        )
        requested.append([kind, handle])
        cursor += 1
        return PublicControlStep(
            images[cursor].copy(),
            event.executed_action.kind.value,
            actual,
            event.intervention_compliance is True
            and event.event.success is True
            and event.controller_failure is None
            and event.event.duration == 1,
            event.executed_action.ticks,
        )

    replayed = {
        "calibration": calibration,
        "acquisition": acquire_stable_keyframe_map(
            images[48].copy(),
            execute,
            calibration["camera_intrinsics"],
            anchor_frame=48,
            seed=config["active_seed"],
            policy=settings,
        ),
    }
    result.update(
        valid=(
            json.dumps(saved, sort_keys=True, allow_nan=False)
            == json.dumps(replayed, sort_keys=True, allow_nan=False)
            and cursor == len(events) == count - 1
            and replayed["acquisition"]["failure"] is None
            and all(runtime.digest(path) == value for path, value in before.items())
        ),
        schedule=requested,
    )
    return result


def audit(directory, engine):
    directory, engine = (
        Path(directory).resolve(strict=True),
        Path(engine).resolve(strict=True),
    )
    report = json.loads((directory / "report.json").read_bytes())
    public_names = set(report["artifact_sha256"]) - {
        "evaluator-camera.jsonl",
        "evaluator-positions.json",
    }
    public_paths = [directory / name for name in public_names]
    if any(not path.resolve().is_relative_to(directory) for path in public_paths):
        raise ValueError("audit artifact cannot escape its capture directory")
    public_paths += [
        directory / "config.json",
        directory / "report.json",
        engine,
        Path(__file__),
        Path(runtime.__file__),
    ]
    public_paths += list((directory / "source-snapshot").rglob("*"))
    public_paths += list((runtime.ROOT / "src/minecraft_scientist").rglob("*.py"))
    frozen = {path: runtime.digest(path) for path in public_paths if path.is_file()}
    try:
        physical = runtime.audit(directory, engine, registration=registration)
    except (KeyError, ValueError, OSError, TypeError) as error:
        return {
            "valid": False,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "camera_audit": None,
            "evaluation": None,
            "stability_diagnostics": None,
            "metric_scale_known": False,
            "navigation_accepted": False,
        }
    invalid = {
        "valid": False,
        "camera_audit": physical,
        "evaluation": None,
        "stability_diagnostics": None,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }
    if not physical["valid"]:
        return invalid
    if not all(runtime.digest(path) == value for path, value in frozen.items()):
        return invalid | {
            "failure": {
                "type": "ValueError",
                "message": "public bytes changed across camera audit",
            }
        }
    private = [
        directory / name
        for name in ("evaluator-camera.jsonl", "evaluator-positions.json")
    ]
    frozen.update({path: runtime.digest(path) for path in private})
    if any(frozen[path] != report["artifact_sha256"][path.name] for path in private):
        return invalid | {
            "failure": {
                "type": "ValueError",
                "message": "private bytes changed after camera audit",
            }
        }
    acquisition = json.loads((directory / "keyframe-acquisition.json").read_bytes())[
        "acquisition"
    ]
    positions = json.loads((directory / "evaluator-positions.json").read_bytes())[
        "positions"
    ]
    cameras = physical["rendered_cameras"]
    stability = [
        {
            "phase": item["phase"],
            "interval": item["interval"],
            "stable_tail": item["stable_tail"],
            "rendered": [
                serializable_truth(rendered_motion_truth(cameras[first], cameras[last]))
                for first, last in item["stable_tail"]
            ],
            "body": [
                serializable_truth(
                    relative_motion_truth(positions[first], positions[last])
                )
                for first, last in item["stable_tail"]
            ],
            "diagnostic_only": True,
        }
        for item in acquisition["settles"]
    ]
    evaluation = None
    if acquisition["future"] is not None:
        first = acquisition["frozen"]["index"]["frame_ordinals"][-1]
        last = acquisition["future"]["absolute_frame"]
        motion = acquisition["motion"]
        predicted = (
            {"supported": False}
            if motion is None
            else {
                "supported": True,
                "relative_rotation": motion["relative_rotation"],
                "unit_translation": motion["unit_translation"],
            }
        )
        evaluation = {
            "interval": [first, last],
            "rendered": score_rendered_motion(
                predicted, cameras[first], cameras[last], **THRESHOLDS
            ),
            "body": score_relative_motion(
                predicted, positions[first], positions[last], **THRESHOLDS
            ),
        }
    checks = {
        "capture_public_policy_independently_replayed": physical["valid"],
        "all_artifacts_match_capture_ledger": all(
            runtime.digest(directory / name) == value
            for name, value in report["artifact_sha256"].items()
        ),
        "bytes_stable_through_scoring": all(
            runtime.digest(path) == value for path, value in frozen.items()
        ),
    }
    if not all(checks.values()):
        return invalid | {
            "checks": checks,
            "failure": {
                "type": "ValueError",
                "message": "late mutation discards all physical scores",
            },
        }
    return {
        "valid": True,
        "checks": checks,
        "camera_audit": physical,
        "policy_terminal": acquisition["terminal"],
        "attempts": len(acquisition["attempts"]),
        "selected_keyframes": sum(item["selected"] for item in acquisition["samples"]),
        "evaluation": evaluation,
        "stability_diagnostics": stability,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.output.with_suffix(".source.py")
    if args.output.exists() or source.exists():
        raise FileExistsError("refusing to overwrite a keyframe audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    try:
        result = audit(args.route, args.engine_jar)
    except (KeyError, ValueError, OSError, TypeError) as error:
        result = {
            "valid": False,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "evaluation": None,
            "navigation_accepted": False,
        }
    result["audit_snapshot_sha256"] = runtime.digest(source)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "valid",
                    "policy_terminal",
                    "attempts",
                    "selected_keyframes",
                    "evaluation",
                    "failure",
                )
            }
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
