"""Controlled live visual intervention; privileged setup never enters perception."""
import json
import argparse
import os
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from PIL import Image
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback
from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction
from minecraft_scientist.minecraft.visual_evidence import compare_regions, verify_equipment_trace


class PixelRecorder:
    def __init__(self):
        self.grid = GridVisualController(rows=1, columns=1)
        self.frames = []
        self.equipment = []

    def perceive(self, frame, info, *, seed):
        self.frames.append(np.asarray(frame).copy())
        equipped = info.get("equipped_items", {}).get("mainhand", {})
        value = equipped.get("type") if isinstance(equipped, dict) else None
        self.equipment.append(str(value) if value is not None else None)
        # Deliberately discard privileged info, including evaluator labels.
        # Only the fixed target crop enters Agent perception. Full evaluator
        # screenshots may contain recipe toasts and translated item-name HUD text.
        return self.grid.perceive(np.asarray(frame)[80:144, 80:144], {}, seed=seed)

    def encode_action(self, action, scene):
        return self.grid.encode_action(action, scene)


def factory(
    config, *, block="dirt", mode="creative", tool="air", select_slot=None
):
    commands = [
        "/gamerule sendCommandFeedback false",
        "/gamerule announceAdvancements false",
        # Clearing vegetation must not fill the supposedly empty hand.
        "/gamerule doTileDrops false",
        f"/gamemode {mode}",
        "/weather clear",
        "/time set noon",
        "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false",
        # Build in the player's loaded chunk, not an unloaded absolute origin.
        "/execute at @p align xyz run fill ~-5 ~-1 ~-5 ~5 ~-1 ~5 minecraft:stone",
        "/execute at @p align xyz run fill ~-5 ~ ~-5 ~5 ~5 ~5 minecraft:air",
        f"/execute at @p align xyz run setblock ~ ~1 ~2 minecraft:{block}",
        "/execute at @p align xyz run tp @p ~0.5 ~ ~0.5 0 0",
        "/execute at @p run kill @e[type=minecraft:item,distance=..16]",
        "/clear @p",
    ]
    if select_slot is None:
        commands.append(f"/replaceitem entity @p weapon.mainhand minecraft:{tool}")
    elif tool != "air":
        # Minecraft command slots are zero-based; MineStudio hotbar actions are
        # one-based. The Agent must still issue SELECT before the probe.
        commands.append(
            f"/replaceitem entity @p hotbar.{select_slot - 1} minecraft:{tool} 1"
        )
    return MinecraftSim(
        action_type="env", seed=config.seed, callbacks=[CommandsCallback(commands)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", choices=("dirt", "stone", "oak_log"), default="dirt")
    parser.add_argument("--mode", choices=("creative", "survival"), default="creative")
    parser.add_argument("--attack-frames", type=int, default=1)
    parser.add_argument(
        "--tool",
        choices=("air", "wooden_pickaxe", "iron_pickaxe"),
        default="air",
    )
    parser.add_argument(
        "--select-slot",
        type=int,
        choices=range(1, 10),
        help="Place the fixture tool in this slot and select it through the Agent adapter",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.attack_frames <= 200:
        parser.error("attack-frames must be in 1..200")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    recorder = PixelRecorder()
    records = []
    def simulator_factory(config):
        return factory(
            config,
            block=args.block,
            mode=args.mode,
            tool=args.tool,
            select_slot=args.select_slot,
        )

    with MineStudioVisualAdapter(
        recorder, simulator_factory=simulator_factory
    ) as world:
        world.reset(args.seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        equipment_verified_from = 2
        if args.select_slot is not None:
            slot = next(
                item for item in world.observe().objects
                if item.observable_features.get("slot") == args.select_slot
            )
            records.append(
                world.step(
                    PrimitiveAction(kind=ActionKind.SELECT, target_id=slot.object_id)
                )
            )
            equipment_verified_from = len(recorder.equipment) - 1
        baseline = recorder.frames[-1].copy()
        control = recorder.frames[-2].copy()
        action = next(a for a in world.available_actions() if a.kind is ActionKind.INTERACT)
        for _ in range(args.attack_frames):
            transition = world.step(action)
            records.append(transition)
            if transition.observed_delta["terminated"] or transition.observed_delta["truncated"]:
                raise RuntimeError("episode terminated during probe")
        records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        after = recorder.frames[-1]
    for name, frame in (("control", control), ("before", baseline), ("after", after)):
        Image.fromarray(frame).save(output / f"{name}.png")
    # Verify from the settled baseline onward; never give these labels to Agent.
    equipment = verify_equipment_trace(
        recorder.equipment,
        expected=args.tool,
        verified_from_frame=equipment_verified_from,
    )
    (output / "equipment-audit.json").write_text(json.dumps({
        "expected": equipment.expected, "observed": equipment.observed,
        "verified_from_frame": equipment.verified_from_frame, "valid": equipment.valid,
    }, indent=2), encoding="utf-8")
    if not equipment.valid:
        raise RuntimeError("held-item control failed or could not be verified; see equipment-audit.json")
    roi = np.s_[80:144, 80:144, :]
    evidence = compare_regions(control[roi], baseline[roi], after[roi])
    report = {
        "seed": args.seed, "backend": "real-minestudio",
        "mode": args.mode, "attack_frames": args.attack_frames,
        "control_roi_mae": evidence.control_mae,
        "intervention_roi_mae": evidence.intervention_mae,
        "control_stable": evidence.control_stable,
        "appearance_changed": evidence.changed,
        "equipment_audit_passed": equipment.valid,
        "tool_selection_via_adapter": args.select_slot is not None,
        "selected_slot": args.select_slot,
        "scope": "Controlled visual intervention probe, not tool-tier discovery.",
        "pixel_only_perception": True,
    }
    # Privileged fixture metadata is explicitly separate from Agent events.
    (output / "fixture.json").write_text(json.dumps({
        "block": args.block, "mode": args.mode, "seed": args.seed,
        "tool": args.tool,
        "select_slot": args.select_slot,
        "attack_frames": args.attack_frames,
    }, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "events.jsonl").write_text("\n".join(r.model_dump_json() for r in records)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
