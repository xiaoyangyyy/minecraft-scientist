"""Closed-loop RGB approach to a visual goal through an opaque MOVE handle."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from PIL import Image
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback

from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.minecraft.visual_evidence import central_view_salience
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


class NavigationRecorder:
    def __init__(self):
        self.grid = GridVisualController(
            rows=1,
            columns=1,
            hotbar_slots=1,
            movement_controls=("forward",),
        )
        self.frames = []

    def perceive(self, frame, info, *, seed):
        if any(
            key in info
            for key in ("location_stats", "xpos", "ypos", "zpos", "pitch", "yaw")
        ):
            raise RuntimeError("privileged position telemetry reached RGB perception")
        array = np.asarray(frame).copy()
        self.frames.append(array)
        return self.grid.perceive(array, {}, seed=seed)

    def encode_action(self, action, scene):
        return self.grid.encode_action(action, scene)


class PositionAudit:
    """Evaluator-only telemetry sink, kept separate from the RGB controller."""

    def __init__(self):
        self.positions = []
        self.metadata_keys = set()

    def record(self, frame, info):
        for source in (frame, info):
            if not isinstance(source, dict):
                continue
            self.metadata_keys.update(str(key) for key in source)
            stats = source.get("location_stats")
            if not isinstance(stats, dict):
                continue
            position = {}
            for key in ("xpos", "ypos", "zpos", "pitch", "yaw"):
                if key not in stats:
                    continue
                value = np.asarray(stats[key])
                if value.size == 1 and np.issubdtype(value.dtype, np.number):
                    position[key] = float(value.reshape(-1)[0])
            if position:
                self.positions.append(position)


class PositionAuditSimulator:
    """Intercept privileged telemetry before the adapter sees simulator output."""

    _PRIVILEGED_KEYS = {"location_stats", "xpos", "ypos", "zpos", "pitch", "yaw"}

    def __init__(self, simulator, audit):
        self.simulator = simulator
        self.audit = audit

    def _record(self, frame, info):
        self.audit.record(frame, info)

    @classmethod
    def _public(cls, value):
        if not isinstance(value, dict):
            return value
        return {
            key: item
            for key, item in value.items()
            if str(key) not in cls._PRIVILEGED_KEYS
        }

    def reset(self):
        frame, info = self.simulator.reset()
        self._record(frame, info)
        return self._public(frame), self._public(info)

    def step(self, action):
        frame, reward, terminated, truncated, info = self.simulator.step(action)
        self._record(frame, info)
        return (
            self._public(frame),
            reward,
            terminated,
            truncated,
            self._public(info),
        )

    def noop_action(self):
        return self.simulator.noop_action()

    def close(self):
        self.simulator.close()


def factory(config, audit):
    commands = [
        "/gamerule sendCommandFeedback false",
        "/gamerule announceAdvancements false",
        "/gamerule doTileDrops false",
        "/gamemode survival",
        "/difficulty peaceful",
        "/weather clear",
        "/time set noon",
        "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false",
        "/execute at @p align xyz run fill ~-8 ~-1 ~-4 ~8 ~-1 ~14 minecraft:stone",
        "/execute at @p align xyz run fill ~-8 ~ ~-4 ~8 ~8 ~14 minecraft:air",
        "/execute at @p run kill @e[type=!minecraft:player,distance=..32]",
        "/execute at @p align xyz run fill ~-8 ~ ~10 ~8 ~8 ~10 minecraft:black_concrete",
        "/execute at @p align xyz run fill ~-2 ~1 ~10 ~2 ~5 ~10 minecraft:white_wool",
        "/execute at @p align xyz run tp @p ~0.5 ~ ~0.5 0 0",
        "/clear @p",
    ]
    return PositionAuditSimulator(
        MinecraftSim(
            action_type="env",
            seed=config.seed,
            callbacks=[CommandsCallback(commands)],
        ),
        audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--score-threshold", type=float, default=150.0)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.score_threshold <= 0 or args.move_frames < 1 or args.max_moves < 1:
        parser.error("threshold and movement budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    recorder = NavigationRecorder()
    position_audit = PositionAudit()
    records = []
    trajectory = []
    with MineStudioVisualAdapter(
        recorder, simulator_factory=lambda config: factory(config, position_audit)
    ) as world:
        world.reset(args.seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        move_handle = next(
            item.object_id for item in world.observe().objects
            if item.affordances == (ActionKind.MOVE,)
        )

        def observe_score(move_index):
            score = central_view_salience(recorder.frames[-1])
            trajectory.append({
                "move_index": move_index,
                "score": score,
                "frame_sha256": world.observe().agent_state["frame_sha256"],
            })
            Image.fromarray(recorder.frames[-1]).save(
                args.output / f"frame-{move_index:02d}.png"
            )
            return score

        score = observe_score(0)
        moves = 0
        while score < args.score_threshold and moves < args.max_moves:
            records.append(world.step(PrimitiveAction(
                kind=ActionKind.MOVE,
                target_id=move_handle,
                ticks=args.move_frames,
            )))
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
            moves += 1
            score = observe_score(moves)

    positions = position_audit.positions
    displacement = None
    if positions:
        start, end = positions[0], positions[-1]
        if all(key in start and key in end for key in ("xpos", "zpos")):
            displacement = math.hypot(
                end["xpos"] - start["xpos"],
                end["zpos"] - start["zpos"],
            )
    move_records = [
        record for record in records
        if record.executed_action.kind is ActionKind.MOVE
    ]
    report = {
        "seed": args.seed,
        "backend": "real-minestudio",
        "score_threshold": args.score_threshold,
        "initial_score": trajectory[0]["score"],
        "final_score": trajectory[-1]["score"],
        "moves": moves,
        "move_frames": args.move_frames,
        "stopped_on_visual_threshold": score >= args.score_threshold,
        "all_move_actions_valid": bool(move_records) and all(
            record.intervention_compliance and record.event.success
            for record in move_records
        ),
        "evaluator_displacement": displacement,
        "pixel_only_policy": True,
        "scope": "Straight fixture corridor with one opaque forward handle.",
    }
    (args.output / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2), encoding="utf-8"
    )
    (args.output / "position-audit.json").write_text(
        json.dumps({
            "positions": positions,
            "raw_metadata_keys": sorted(position_audit.metadata_keys),
        }, indent=2), encoding="utf-8"
    )
    (args.output / "events.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    if (
        not report["stopped_on_visual_threshold"]
        or not report["all_move_actions_valid"]
        or displacement is None
        or displacement <= 1.0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
