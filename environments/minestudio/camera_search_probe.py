"""Active RGB camera search over three opaque viewpoints in real MineStudio."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from PIL import Image
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback

from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.minecraft.visual_evidence import (
    central_view_salience,
    hud_safe_view,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


VIEW_COLUMNS = {"negative": 0, "center": 1, "positive": 2}
MARKER_X = {
    "negative": (2, 4),
    "center": (-1, 1),
    "positive": (-4, -2),
}


class ScanRecorder:
    def __init__(self):
        self.grid = GridVisualController(rows=1, columns=3, hotbar_slots=1)
        self.frames = []

    def perceive(self, frame, info, *, seed):
        del info
        array = np.asarray(frame).copy()
        self.frames.append(array)
        return self.grid.perceive(hud_safe_view(array), {}, seed=seed)

    def encode_action(self, action, scene):
        return self.grid.encode_action(action, scene)


def factory(config, *, target_view):
    left, right = MARKER_X[target_view]
    commands = [
        "/gamerule sendCommandFeedback false",
        "/gamerule announceAdvancements false",
        "/gamerule doTileDrops false",
        "/gamemode creative",
        "/weather clear",
        "/time set noon",
        "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false",
        "/difficulty peaceful",
        "/execute at @p align xyz run fill ~-12 ~-1 ~-4 ~12 ~-1 ~12 minecraft:stone",
        "/execute at @p align xyz run fill ~-12 ~ ~-4 ~12 ~8 ~12 minecraft:air",
        "/execute at @p run kill @e[type=!minecraft:player,distance=..32]",
        "/execute at @p align xyz run fill ~-16 ~ ~6 ~16 ~10 ~6 minecraft:black_concrete",
        (
            "/execute at @p align xyz run fill "
            f"~{left} ~1 ~6 ~{right} ~3 ~6 minecraft:white_wool"
        ),
        "/execute at @p align xyz run tp @p ~0.5 ~ ~0.5 0 0",
        "/clear @p",
    ]
    return MinecraftSim(
        action_type="env",
        seed=config.seed,
        callbacks=[CommandsCallback(commands)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-view", choices=tuple(VIEW_COLUMNS), default="negative"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    recorder = ScanRecorder()
    records = []

    def simulator_factory(config):
        return factory(config, target_view=args.target_view)

    with MineStudioVisualAdapter(
        recorder, simulator_factory=simulator_factory
    ) as world:
        world.reset(args.seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))

        def view_id(column):
            return next(
                item.object_id for item in world.observe().objects
                if ActionKind.LOOK in item.affordances
                and item.observable_features.get("screen_column") == column
            )

        opaque_views = {
            name: view_id(column) for name, column in VIEW_COLUMNS.items()
        }
        frames = {}

        def look_and_capture(name):
            records.append(world.step(PrimitiveAction(
                kind=ActionKind.LOOK,
                target_id=opaque_views[name],
            )))
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
            frames[name] = recorder.frames[-1].copy()

        # All scans start and end at yaw zero. Negative and positive are relative
        # one-step turns, while center is an explicit zero-delta LOOK action.
        look_and_capture("center")
        look_and_capture("negative")
        records.append(world.step(PrimitiveAction(
            kind=ActionKind.LOOK, target_id=opaque_views["positive"]
        )))
        records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        look_and_capture("positive")
        records.append(world.step(PrimitiveAction(
            kind=ActionKind.LOOK, target_id=opaque_views["negative"]
        )))

    scores = {
        opaque_views[name]: central_view_salience(frame)
        for name, frame in frames.items()
    }
    predicted_id = max(scores, key=scores.get)
    expected_id = opaque_views[args.target_view]
    look_records = [
        record for record in records
        if record.executed_action.kind is ActionKind.LOOK
    ]
    camera_actions_valid = bool(look_records) and all(
        record.intervention_compliance and record.event.success
        for record in look_records
    )
    report = {
        "seed": args.seed,
        "backend": "real-minestudio",
        "opaque_view_scores": scores,
        "predicted_view_id": predicted_id,
        "correct": predicted_id == expected_id,
        "camera_actions": len(look_records),
        "camera_actions_valid": camera_actions_valid,
        "pixel_only_selection": True,
        "scope": "Three fixture-defined yaw views; brightness-defined marker search.",
    }
    for name, frame in frames.items():
        Image.fromarray(frame).save(args.output / f"view-{name}.png")
    (args.output / "fixture.json").write_text(json.dumps({
        "target_view": args.target_view,
        "expected_view_id": expected_id,
        "view_bindings": opaque_views,
    }, indent=2), encoding="utf-8")
    (args.output / "events.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    if not report["correct"] or not camera_actions_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
