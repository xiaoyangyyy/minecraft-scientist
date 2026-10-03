"""Acquire fresh natural RGB using bounded one-tick visibility-aware control.

Private position capture remains in the evaluator wrapper. Probe decisions see
only RGB plus public execution receipts. This is not a navigation acceptance run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import capture_controlled_rgb_motion as base
import plan_active_rgb_motion as planner
from controlled_motion_common import digest
from PIL import Image

from minecraft_scientist.exploration.motion_probe_schedule import motion_probe_schedule
from minecraft_scientist.exploration.multiview_motion import MultiViewMotionConfig
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
    fit_online_landmarks,
)
from minecraft_scientist.exploration.visibility_acquisition import (
    PublicProbeStep,
    VisibilityPolicy,
    acquire_visibility_probe,
)
from minecraft_scientist.minecraft import MineStudioConfig, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


def online_policy():
    return VisibilityPolicy(landmark_protection="first_supported_map_v1")


def online_settings():
    return {
        "schema": "causal-landmark-fit-v1",
        "seed_base": 0,
        "seed_stride": 5,
        "seed_offset": 2,
        "multiview": MultiViewMotionConfig().model_dump(mode="json"),
    }


def target(world, kind, handle=None):
    objects = world.observe().objects
    if kind == "move":
        return [
            item.object_id for item in objects if item.affordances == (ActionKind.MOVE,)
        ][handle]
    if kind == "look":
        return next(
            item.object_id
            for item in objects
            if ActionKind.LOOK in item.affordances
            and item.observable_features.get("screen_row") == 1
            and item.observable_features.get("screen_column") == 2
        )
    return None


def collect(world, frames, config, policy, records, trials):
    """RGB-only callback boundary; this function cannot read evaluator positions."""
    world.reset(config.seed)
    for scheduled in motion_probe_schedule(config)[:25]:
        records.append(
            world.step(
                PrimitiveAction(
                    kind=ActionKind(scheduled.kind),
                    target_id=target(
                        world, scheduled.kind, scheduled.motion_handle_index
                    ),
                    ticks=scheduled.ticks,
                )
            )
        )
        if not records[-1].intervention_compliance or not records[-1].event.success:
            raise RuntimeError("startup/reference control failed")
    references = [
        sum(item.event.duration for item in records[: index + 1])
        for index, action in enumerate(motion_probe_schedule(config)[:25])
        if action.phase == "reference_settle"
    ]
    calibration = calibrate_reference_camera(
        [frames[slot] for slot in references], config.render_size
    )
    for repetition in range(2):
        for pair in config.handle_pairs:
            anchor = len(frames) - 1
            number = len(trials)
            settings = online_settings()
            landmark_seed = (
                settings["seed_base"]
                + number * settings["seed_stride"]
                + settings["seed_offset"]
            )

            def execute(handle):
                object_id = target(world, "move", handle)
                record = world.step(
                    PrimitiveAction(kind=ActionKind.MOVE, target_id=object_id, ticks=1)
                )
                records.append(record)
                correct_action = (
                    record.executed_action.kind == ActionKind.MOVE
                    and record.executed_action.target_id == object_id
                    and record.executed_action.ticks == 1
                    and record.event.duration == 1
                )
                return PublicProbeStep(
                    frames[-1],
                    handle if correct_action else -1,
                    bool(record.intervention_compliance and record.event.success),
                )

            def observe_landmarks(tracked, retained, *, seed=landmark_seed):
                return fit_online_landmarks(
                    tracked,
                    retained,
                    calibration["camera_intrinsics"],
                    seed=seed,
                )

            result = acquire_visibility_probe(
                frames[-1],
                execute,
                pair,
                policy=policy,
                landmark_observer=observe_landmarks,
            )
            trials.append(
                {
                    "repetition": repetition,
                    "handles": list(pair),
                    "anchor_frame": anchor,
                    "end_frame": len(frames) - 1,
                    "terminal": result.terminal,
                    "failure": result.failure,
                    "landmark_seed": landmark_seed,
                    "landmark_attempts": [
                        item.model_dump(mode="json")
                        for item in result.landmark_attempts
                    ],
                    "steps": list(result.steps),
                    "decisions": [
                        item.model_dump(mode="json") for item in result.decisions
                    ],
                    "prefix_tracks": [
                        item.model_dump(mode="json") for item in result.tracks
                    ],
                }
            )
            if result.failure:
                raise RuntimeError(f"visibility probe failed: {result.failure}")
            # An aborted probe remains a failure category, never accepted evidence.
            record = world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=8))
            records.append(record)
            if not record.intervention_compliance or not record.event.success:
                raise RuntimeError("post-probe WAIT control failed")
    records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=8)))
    if not records[-1].intervention_compliance or not records[-1].event.success:
        raise RuntimeError("final WAIT control failed")
    return references, calibration


def run(plan_directory, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite adaptive visibility capture")
    config = planner.validate_plan(plan_directory)  # Public-only prior pixel replay.
    policy = online_policy()
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks.commands import CommandsCallback
    from navigation_probe import PositionAudit, PositionAuditSimulator

    source_paths = (
        planner.sources()
        | base.source_paths()
        | {
            "capture_visibility_rgb.py": Path(__file__),
            "visibility_acquisition.py": Path(
                acquire_visibility_probe.__code__.co_filename
            ),
            "sdk-entry.py": Path(
                __import__(MinecraftSim.__module__, fromlist=["__file__"]).__file__
            ),
        }
    )
    package = (
        Path(MineStudioVisualAdapter.step.__code__.co_filename).resolve().parents[1]
    )
    source_paths.update(
        {
            f"package/{path.relative_to(package).as_posix()}": path
            for path in sorted(package.rglob("*.py"))
        }
    )
    prior = json.loads((plan_directory / "config.json").read_bytes())
    inputs = planner.inputs(
        ROOT / prior["source_acquisition"], ROOT / prior["source_predictions"]
    ) | {
        "plan_config": plan_directory / "config.json",
        "plan_result": plan_directory / "plan.json",
    }
    maximum_frames = (
        57
        + 2
        * len(config.handle_pairs)
        * (policy.maximum_first_ticks + policy.maximum_second_ticks + 8)
        + 8
    )
    if maximum_frames > 225:
        raise ValueError("adaptive capture exceeds registered 225-frame ceiling")
    manifest = {
        "schema": "visibility-rgb-acquisition-v2",
        "policy": policy.model_dump(mode="json"),
        "online_landmarks": online_settings(),
        "world": {
            "seed": config.seed,
            "observation_size": config.observation_size,
            "render_size": list(config.render_size),
            "time": config.time_of_day,
            "weather": config.weather,
        },
        "startup": [
            item.model_dump(mode="json") for item in motion_probe_schedule(config)[:25]
        ],
        "handle_pairs": config.model_dump(mode="json")["handle_pairs"],
        "repetitions": 2,
        "plan_directory": plan_directory.resolve().relative_to(ROOT).as_posix(),
        "maximum_frames": maximum_frames,
        "input_sha256": {name: digest(path) for name, path in inputs.items()},
        "source_sha256": {name: digest(path) for name, path in source_paths.items()},
        "scope": "fresh visibility-aware RGB probe on development seed; no hidden telemetry in policy, no terrain edits, no pose validation or navigation promotion",
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in source_paths.items():
        saved = snapshot / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
    preregistration = output / "config.json"
    preregistration.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(preregistration)
    audit, frames, records, trials = PositionAudit(), [], [], []
    commands = [
        "/gamerule sendCommandFeedback false",
        "/gamerule announceAdvancements false",
        "/gamemode creative",
        "/difficulty peaceful",
        f"/weather {config.weather}",
        f"/time set {config.time_of_day}",
        "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false",
        "/clear @p",
    ]

    def factory(settings):
        sim = MinecraftSim(
            action_type="env",
            seed=settings.seed,
            obs_size=settings.obs_size,
            render_size=settings.render_size,
            callbacks=[CommandsCallback(commands)],
        )
        return base.FrameCapture(PositionAuditSimulator(sim, audit), frames)

    world = MineStudioVisualAdapter(
        base.ProbeRecorder(),
        MineStudioConfig(
            obs_size=(config.observation_size,) * 2, render_size=config.render_size
        ),
        simulator_factory=factory,
    )
    failure, references, calibration = None, [], None
    try:
        references, calibration = collect(
            world, frames, config, policy, records, trials
        )
    except Exception as error:  # noqa: BLE001 -- retain acquisition failures
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            world.close()
        except Exception as error:  # noqa: BLE001 -- cleanup failure invalidates run
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "prior_failure": failure,
            }
    image_directory = output / "frames"
    image_directory.mkdir()
    frame_hashes, rgb_hashes = {}, []
    for slot, frame in enumerate(frames):
        path = image_directory / f"frame-{slot:05d}.png"
        Image.fromarray(frame).save(path)
        frame_hashes[path.name] = digest(path)
        rgb_hashes.append(base.hashlib.sha256(frame.tobytes()).hexdigest())
    (output / "events.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    reference_directory = output / "goal-reference"
    reference_directory.mkdir()
    for number, slot in enumerate(references):
        (reference_directory / f"view-{number:02d}.png").write_bytes(
            (image_directory / f"frame-{slot:05d}.png").read_bytes()
        )
    timeline, cursor = [], 0
    aligned = bool(frames)
    for record in records:
        end = cursor + record.event.duration
        aligned = (
            aligned
            and end < len(frames)
            and record.state_before.agent_state["frame_sha256"] == rgb_hashes[cursor]
            and record.state_after.agent_state["frame_sha256"] == rgb_hashes[end]
            and record.intervention_compliance
            and record.event.success
        )
        timeline.append(
            {
                "start_frame": cursor,
                "end_frame": end,
                "action": record.executed_action.model_dump(mode="json"),
            }
        )
        cursor = end
    gates = {
        "execution_completed": failure is None
        and len(trials) == 2 * len(config.handle_pairs),
        "action_frame_alignment": aligned and cursor + 1 == len(frames),
        "bounded_frames": 1 <= len(frames) <= maximum_frames,
        "telemetry_alignment": len(audit.positions) == len(frames),
        "preregistration_unchanged": digest(preregistration) == frozen,
        "input_sources_unchanged": all(
            digest(path) == manifest["input_sha256"][name]
            for name, path in inputs.items()
        )
        and all(
            digest(path) == manifest["source_sha256"][name]
            for name, path in source_paths.items()
        ),
        "public_log_clean": not any(
            token in (output / "events.jsonl").read_text(encoding="utf-8")
            for token in ("location_stats", "xpos", "ypos", "zpos", '"yaw"', '"pitch"')
        ),
    }
    report = {
        "passed": all(gates.values()),
        "checks": gates,
        "config_sha256": frozen,
        "failure": failure,
        "frame_count": len(frames),
        "frame_sha256": frame_hashes,
        "rgb_sha256": rgb_hashes,
        "reference_indices": references,
        "landmark_calibration": calibration,
        "timeline": timeline,
        "trials": trials,
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "evaluator-audit.json").write_text(
        json.dumps({"positions": audit.positions}, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.plan, args.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "frames": report["frame_count"],
                "failure": report["failure"],
                "terminals": [item["terminal"] for item in report["trials"]],
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
