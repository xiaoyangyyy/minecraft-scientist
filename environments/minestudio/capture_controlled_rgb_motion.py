"""Capture fixed natural-world WAIT/move/settle trials without policy telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)
from minecraft_scientist.minecraft import (
    GridVisualController,
    MineStudioConfig,
    MineStudioVisualAdapter,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")


class ProbeRecorder:
    def __init__(self, *, jump_controls=False):
        if type(jump_controls) is not bool:
            raise ValueError("compound movement registration requires explicit boolean")
        cardinal = ("forward", "back", "left", "right")
        self.grid = GridVisualController(
            rows=3,
            columns=3,
            hotbar_slots=1,
            movement_controls=() if jump_controls else cardinal,
            movement_programs=tuple((control, "jump") for control in cardinal)
            if jump_controls
            else (),
        )

    def perceive(self, frame, info, *, seed):
        if any(
            key in info
            for key in ("location_stats", "xpos", "ypos", "zpos", "yaw", "pitch")
        ):
            raise RuntimeError("privileged motion state reached RGB controller")
        return self.grid.perceive(frame, {}, seed=seed)

    def encode_action(self, action, scene):
        return self.grid.encode_action(action, scene)


class FrameCapture:
    def __init__(self, simulator, frames):
        self.simulator, self.frames = simulator, frames

    def _capture(self, frame):
        if len(self.frames) >= 500:
            raise RuntimeError("controlled image budget exhausted")
        self.frames.append(
            np.asarray(frame["image"] if isinstance(frame, dict) else frame).copy()
        )

    def reset(self):
        frame, info = self.simulator.reset()
        self._capture(frame)
        return frame, info

    def step(self, action):
        frame, reward, terminated, truncated, info = self.simulator.step(action)
        self._capture(frame)
        return frame, reward, terminated, truncated, info

    def noop_action(self):
        return self.simulator.noop_action()

    def close(self):
        self.simulator.close()


def source_paths() -> dict[str, Path]:
    return {
        "capture_controlled_rgb_motion.py": Path(__file__),
        "motion_probe_schedule.py": Path(motion_probe_schedule.__code__.co_filename),
        "minestudio.py": Path(MineStudioVisualAdapter.step.__code__.co_filename),
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--observation-size", type=int, default=448)
    parser.add_argument("--time", choices=("noon", "midnight"), default="noon")
    parser.add_argument("--weather", choices=("clear", "thunder"), default="clear")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = MotionProbeConfig(
        seed=args.seed,
        observation_size=args.observation_size,
        time_of_day=args.time,
        weather=args.weather,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks.commands import CommandsCallback
    from navigation_probe import PositionAudit, PositionAuditSimulator

    sources = source_paths() | {
        "sdk-entry.py": Path(
            __import__(MinecraftSim.__module__, fromlist=["__file__"]).__file__
        )
    }
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, data in source_bytes.items():
        (snapshot / name).write_bytes(data)
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()
    }
    preregistered = {
        "config": config.model_dump(mode="json"),
        "render_size": list(config.render_size),
        "schedule": [
            item.model_dump(mode="json") for item in motion_probe_schedule(config)
        ],
        "pairs": motion_probe_pairs(config),
        "source_sha256": source_hashes,
        "scope": "controlled development at natural spawn; no terrain edits; not navigation acceptance",
    }
    manifest_path = args.output / "config.json"
    manifest_path.write_text(
        json.dumps(preregistered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
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
    audit, frames, records = PositionAudit(), [], []

    def factory(settings):
        sim = MinecraftSim(
            action_type="env",
            seed=settings.seed,
            obs_size=settings.obs_size,
            render_size=settings.render_size,
            callbacks=[CommandsCallback(commands)],
        )
        return FrameCapture(PositionAuditSimulator(sim, audit), frames)

    world = MineStudioVisualAdapter(
        ProbeRecorder(),
        MineStudioConfig(
            obs_size=(config.observation_size,) * 2, render_size=config.render_size
        ),
        simulator_factory=factory,
    )
    failure = None
    try:
        world.reset(config.seed)
        for scheduled in motion_probe_schedule(config):
            target = None
            if scheduled.kind == "move":
                target = [
                    item.object_id
                    for item in world.observe().objects
                    if item.affordances == (ActionKind.MOVE,)
                ][scheduled.motion_handle_index]
            elif scheduled.kind == "look":
                target = next(
                    item.object_id
                    for item in world.observe().objects
                    if ActionKind.LOOK in item.affordances
                    and item.observable_features.get("screen_row") == 1
                    and item.observable_features.get("screen_column") == 2
                )
            records.append(
                world.step(
                    PrimitiveAction(
                        kind=ActionKind(scheduled.kind),
                        target_id=target,
                        ticks=scheduled.ticks,
                    )
                )
            )
    except Exception as error:  # noqa: BLE001 -- retain all acquisition failures
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        world.close()
    directory = args.output / "frames"
    directory.mkdir()
    frame_hashes, rgb_hashes = {}, []
    for slot, frame in enumerate(frames):
        path = directory / f"frame-{slot:05d}.png"
        Image.fromarray(frame).save(path)
        frame_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rgb_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    events = "".join(item.model_dump_json() + "\n" for item in records)
    (args.output / "events.jsonl").write_text(events, encoding="utf-8")
    timeline = motion_probe_timeline(config)
    references = [
        item["end_frame"]
        for item in timeline
        if item["scheduled"]["phase"] == "reference_settle"
    ]
    reference_dir = args.output / "goal-reference"
    reference_dir.mkdir()
    reference_hashes = {}
    for number, slot in enumerate(references):
        if slot < len(frames):
            path = reference_dir / f"view-{number:02d}.png"
            path.write_bytes((directory / f"frame-{slot:05d}.png").read_bytes())
            reference_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    aligned = (
        len(records) == len(timeline) and len(frames) == timeline[-1]["end_frame"] + 1
    )
    if aligned:
        aligned = all(
            item.event.duration == scheduled.ticks
            and item.executed_action.kind.value == scheduled.kind
            and item.intervention_compliance
            and item.event.success
            and item.state_before.agent_state["frame_sha256"]
            == rgb_hashes[span["start_frame"]]
            and item.state_after.agent_state["frame_sha256"]
            == rgb_hashes[span["end_frame"]]
            for item, scheduled, span in zip(
                records, motion_probe_schedule(config), timeline, strict=True
            )
        )
    gate = {
        "execution_completed": failure is None,
        "action_frame_alignment": aligned,
        "telemetry_alignment": len(audit.positions) == len(frames),
        "public_log_clean": not any(
            token in events
            for token in ("location_stats", "xpos", "ypos", "zpos", '"yaw"', '"pitch"')
        ),
        "sources_unchanged": all(
            path.read_bytes() == source_bytes[name] for name, path in sources.items()
        ),
        "preregistration_unchanged": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        == manifest_hash,
    }
    gate["passed"] = all(gate.values())
    report = {
        "suite": "controlled_rgb_motion_acquisition",
        "config_sha256": manifest_hash,
        "frame_count": len(frames),
        "frame_sha256": frame_hashes,
        "rgb_sha256": rgb_hashes,
        "reference_sha256": reference_hashes,
        "timeline": timeline,
        "pairs": motion_probe_pairs(config),
        "reference_indices": references,
        "gate": gate,
        "failure": failure,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "evaluator-audit.json").write_text(
        json.dumps({"positions": audit.positions}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"frame_count": len(frames), "gate": gate, "failure": failure}))
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
