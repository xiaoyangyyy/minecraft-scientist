"""Discover an opaque forward control from RGB effects, then approach a goal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from PIL import Image

from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.minecraft.visual_evidence import central_view_salience
from minecraft_scientist.protocols import ActionKind, PrimitiveAction

from navigation_probe import PositionAudit, factory


class DirectionRecorder:
    """RGB-only controller with a seed-permuted set of opaque motion handles."""

    def __init__(self, controls: tuple[str, ...]):
        self.grid = GridVisualController(
            rows=1,
            columns=1,
            hotbar_slots=1,
            movement_controls=controls,
        )
        self.frames: list[np.ndarray] = []

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


def save_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def save_events(path: Path, records) -> None:
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def movement_handles(world) -> list[str]:
    return [
        item.object_id
        for item in world.observe().objects
        if item.affordances == (ActionKind.MOVE,)
    ]


def frame_record(recorder: DirectionRecorder, world, path: Path) -> dict:
    frame = recorder.frames[-1]
    Image.fromarray(frame).save(path)
    return {
        "score": central_view_salience(frame),
        "frame_sha256": world.observe().agent_state["frame_sha256"],
    }


def run_calibration_trial(
    *,
    seed: int,
    controls: tuple[str, ...],
    handle_index: int,
    move_frames: int,
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    recorder = DirectionRecorder(controls)
    position_audit = PositionAudit()
    records = []
    with MineStudioVisualAdapter(
        recorder, simulator_factory=lambda config: factory(config, position_audit)
    ) as world:
        world.reset(seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        handles = movement_handles(world)
        target_id = handles[handle_index]
        before = frame_record(recorder, world, output / "before.png")
        records.append(
            world.step(
                PrimitiveAction(
                    kind=ActionKind.MOVE,
                    target_id=target_id,
                    ticks=move_frames,
                )
            )
        )
        records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
        after = frame_record(recorder, world, output / "after.png")

    save_events(output / "events.jsonl", records)
    public = {
        "handle_id": target_id,
        "before_score": before["score"],
        "after_score": after["score"],
        "score_delta": after["score"] - before["score"],
        "before_frame_sha256": before["frame_sha256"],
        "after_frame_sha256": after["frame_sha256"],
    }
    private = {
        "handle_id": target_id,
        "control": controls[handle_index],
        "positions": position_audit.positions,
        "raw_metadata_keys": sorted(position_audit.metadata_keys),
    }
    return public, private


def select_handle(calibration: list[dict]) -> str:
    """Agent policy: select only from public RGB-derived calibration evidence."""

    return max(calibration, key=lambda item: item["score_delta"])["handle_id"]


def run_approach(
    *,
    seed: int,
    controls: tuple[str, ...],
    selected_handle: str,
    move_frames: int,
    score_threshold: float,
    max_moves: int,
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    recorder = DirectionRecorder(controls)
    position_audit = PositionAudit()
    records = []
    trajectory = []
    with MineStudioVisualAdapter(
        recorder, simulator_factory=lambda config: factory(config, position_audit)
    ) as world:
        world.reset(seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        handles = movement_handles(world)
        if selected_handle not in handles:
            raise RuntimeError("selected opaque handle changed across controlled resets")

        def observe(move_index: int) -> float:
            item = frame_record(
                recorder,
                world,
                output / f"frame-{move_index:02d}.png",
            )
            trajectory.append({"move_index": move_index, **item})
            return item["score"]

        score = observe(0)
        moves = 0
        while score < score_threshold and moves < max_moves:
            records.append(
                world.step(
                    PrimitiveAction(
                        kind=ActionKind.MOVE,
                        target_id=selected_handle,
                        ticks=move_frames,
                    )
                )
            )
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
            moves += 1
            score = observe(moves)

    save_events(output / "events.jsonl", records)
    save_json(output / "trajectory.json", trajectory)
    public = {
        "selected_handle": selected_handle,
        "moves": moves,
        "move_frames": move_frames,
        "score_threshold": score_threshold,
        "initial_score": trajectory[0]["score"],
        "final_score": trajectory[-1]["score"],
        "stopped_on_visual_threshold": score >= score_threshold,
    }
    private = {
        "positions": position_audit.positions,
        "raw_metadata_keys": sorted(position_audit.metadata_keys),
    }
    return public, private


def displacement(positions: list[dict]) -> float | None:
    if not positions:
        return None
    start, end = positions[0], positions[-1]
    if not all(key in start and key in end for key in ("xpos", "zpos")):
        return None
    return float(
        np.hypot(end["xpos"] - start["xpos"], end["zpos"] - start["zpos"])
    )


def goal_progress(positions: list[dict]) -> float | None:
    """Evaluator-only progress toward the fixture plane at positive Z."""

    if not positions:
        return None
    start, end = positions[0], positions[-1]
    if "zpos" not in start or "zpos" not in end:
        return None
    return float(end["zpos"] - start["zpos"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--mapping-seed", type=int)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=150.0)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.move_frames < 1 or args.score_threshold <= 0 or args.max_moves < 1:
        parser.error("movement and score budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    mapping_seed = (
        args.mapping_seed if args.mapping_seed is not None else args.seed + 10_000
    )
    controls = tuple(
        random.Random(mapping_seed).sample(
            ["forward", "back", "left", "right"],
            k=4,
        )
    )
    calibration = []
    private_trials = []
    for index in range(len(controls)):
        public, private = run_calibration_trial(
            seed=args.seed,
            controls=controls,
            handle_index=index,
            move_frames=args.move_frames,
            output=args.output / "calibration" / f"trial-{index}",
        )
        calibration.append(public)
        private_trials.append(private)
    save_json(args.output / "calibration.json", calibration)

    selected_handle = select_handle(calibration)
    approach, private_approach = run_approach(
        seed=args.seed,
        controls=controls,
        selected_handle=selected_handle,
        move_frames=args.move_frames,
        score_threshold=args.score_threshold,
        max_moves=args.max_moves,
        output=args.output / "approach",
    )
    for item in private_trials:
        item["goal_progress"] = goal_progress(item["positions"])
    valid_goal_trials = [
        item for item in private_trials if item["goal_progress"] is not None
    ]
    if len(valid_goal_trials) != len(private_trials):
        raise RuntimeError("evaluator position telemetry is incomplete")
    evaluator_goal_handle = max(
        valid_goal_trials,
        key=lambda item: item["goal_progress"],
    )["handle_id"]
    evaluator_displacement = displacement(private_approach["positions"])
    private_audit = {
        "seed": args.seed,
        "mapping_seed": mapping_seed,
        "mapping_sha256": hashlib.sha256(
            json.dumps(private_trials, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "trials": private_trials,
        "approach": private_approach,
    }
    save_json(args.output / "evaluator-audit.json", private_audit)

    report = {
        "seed": args.seed,
        "mapping_seed": mapping_seed,
        "backend": "real-minestudio",
        "candidate_count": len(calibration),
        "selected_handle": selected_handle,
        "selected_is_goal_directed": (
            selected_handle == evaluator_goal_handle
            and max(item["goal_progress"] for item in valid_goal_trials) > 0
        ),
        "unique_score_winner": sum(
            item["score_delta"]
            == max(candidate["score_delta"] for candidate in calibration)
            for item in calibration
        )
        == 1,
        "evaluator_displacement": evaluator_displacement,
        **approach,
        "pixel_only_selection": True,
        "scope": "Reset-controlled straight corridor with four opaque move handles.",
    }
    save_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    if not all(
        (
            report["selected_is_goal_directed"],
            report["unique_score_winner"],
            report["stopped_on_visual_threshold"],
            evaluator_displacement is not None,
            evaluator_displacement > 5.0,
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
