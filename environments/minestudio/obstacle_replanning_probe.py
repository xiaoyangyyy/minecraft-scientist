"""Calibrate opaque motion from RGB, then replan around a visible obstacle."""

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from direction_discovery_probe import DirectionRecorder, movement_handles
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback
from navigation_probe import PositionAudit, PositionAuditSimulator
from PIL import Image

from minecraft_scientist.minecraft import MineStudioVisualAdapter
from minecraft_scientist.minecraft.visual_evidence import (
    VisualRegionGeometry,
    central_view_salience,
    largest_achromatic_region,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


def save_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def save_events(path: Path, records) -> None:
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def obstacle_factory(
    config,
    audit: PositionAudit,
    *,
    obstacle: bool,
    obstacle_offset: int = 0,
):
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
    ]
    if obstacle:
        left = obstacle_offset - 1
        right = obstacle_offset + 1
        commands.append(
            "/execute at @p align xyz run fill "
            f"~{left} ~ ~3 ~{right} ~ ~4 minecraft:red_concrete"
        )
    commands.extend((
        "/execute at @p align xyz run tp @p ~0.5 ~ ~0.5 0 0",
        "/clear @p",
    ))
    return PositionAuditSimulator(
        MinecraftSim(
            action_type="env",
            seed=config.seed,
            callbacks=[CommandsCallback(commands)],
        ),
        audit,
    )


def _largest_component(mask: np.ndarray) -> VisualRegionGeometry | None:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    largest: list[tuple[int, int]] = []
    for seed_y, seed_x in zip(*np.where(mask), strict=True):
        if seen[seed_y, seed_x]:
            continue
        seen[seed_y, seed_x] = True
        pending = [(int(seed_y), int(seed_x))]
        component: list[tuple[int, int]] = []
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for candidate_y, candidate_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= candidate_y < height
                    and 0 <= candidate_x < width
                    and mask[candidate_y, candidate_x]
                    and not seen[candidate_y, candidate_x]
                ):
                    seen[candidate_y, candidate_x] = True
                    pending.append((candidate_y, candidate_x))
        if len(component) > len(largest):
            largest = component
    if not largest:
        return None
    ys = np.fromiter((item[0] for item in largest), dtype=float)
    xs = np.fromiter((item[1] for item in largest), dtype=float)
    return VisualRegionGeometry(
        pixel_count=len(largest),
        centroid_x=float(xs.mean()),
        centroid_y=float(ys.mean()),
        left=int(xs.min()),
        top=int(ys.min()),
        right=int(xs.max()),
        bottom=int(ys.max()),
    )


def obstacle_pixels(frame: np.ndarray) -> tuple[VisualRegionGeometry | None, int]:
    """Return fixture-color geometry and central occupancy from RGB only."""

    array = np.asarray(frame, dtype=float)
    if array.ndim != 3 or array.shape[2] < 3 or not np.isfinite(array).all():
        raise ValueError("frame must be a finite RGB-like image")
    height, width = array.shape[:2]
    red, green, blue = (array[:, :, channel] for channel in range(3))
    mask = (
        (red > 80)
        & (red > green * 1.45 + 10)
        & (red > blue * 1.45 + 10)
    )
    mask[: height // 5] = False
    mask[height * 4 // 5 :] = False
    mask[:, : width // 12] = False
    mask[:, width * 11 // 12 :] = False
    central = mask[
        height // 4 : height * 3 // 4,
        width * 3 // 8 : width * 5 // 8,
    ]
    return _largest_component(mask), int(central.sum())


def frame_features(frame: np.ndarray) -> dict:
    marker = largest_achromatic_region(frame)
    obstacle, central_obstacle_pixels = obstacle_pixels(frame)
    return {
        "marker": asdict(marker) if marker is not None else None,
        "obstacle": asdict(obstacle) if obstacle is not None else None,
        "central_obstacle_pixels": central_obstacle_pixels,
        "salience": central_view_salience(frame),
    }


def calibrate_handle(
    *,
    seed: int,
    controls: tuple[str, ...],
    handle_index: int,
    move_frames: int,
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    recorder = DirectionRecorder(controls)
    audit = PositionAudit()
    records = []
    with MineStudioVisualAdapter(
        recorder,
        simulator_factory=lambda config: obstacle_factory(config, audit, obstacle=False),
    ) as world:
        world.reset(seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        handles = movement_handles(world)
        handle = handles[handle_index]
        before_frame = recorder.frames[-1]
        before = frame_features(before_frame)
        Image.fromarray(before_frame).save(output / "before.png")
        records.append(world.step(PrimitiveAction(
            kind=ActionKind.MOVE,
            target_id=handle,
            ticks=move_frames,
        )))
        records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
        after_frame = recorder.frames[-1]
        after = frame_features(after_frame)
        Image.fromarray(after_frame).save(output / "after.png")
    if before["marker"] is None or after["marker"] is None:
        raise RuntimeError("calibration marker was not visible")
    save_events(output / "events.jsonl", records)
    public = {
        "handle_id": handle,
        "before": before,
        "after": after,
        "marker_area_delta": (
            after["marker"]["pixel_count"] - before["marker"]["pixel_count"]
        ),
        "marker_x_delta": (
            after["marker"]["centroid_x"] - before["marker"]["centroid_x"]
        ),
    }
    private = {
        "handle_id": handle,
        "control": controls[handle_index],
        "positions": audit.positions,
        "raw_metadata_keys": sorted(audit.metadata_keys),
    }
    return public, private


def classify_handles(calibration: list[dict]) -> dict:
    """Assign functional roles solely from marker area and centroid effects."""

    goal = max(calibration, key=lambda item: item["marker_area_delta"])
    away = min(calibration, key=lambda item: item["marker_area_delta"])
    lateral = [
        item for item in calibration
        if item["handle_id"] not in {goal["handle_id"], away["handle_id"]}
    ]
    if len(lateral) != 2:
        raise RuntimeError("calibration did not leave two lateral candidates")
    negative_shift = min(lateral, key=lambda item: item["marker_x_delta"])
    positive_shift = max(lateral, key=lambda item: item["marker_x_delta"])
    return {
        "goal": goal["handle_id"],
        "away": away["handle_id"],
        "negative_shift": negative_shift["handle_id"],
        "positive_shift": positive_shift["handle_id"],
    }


def run_route(
    *,
    seed: int,
    controls: tuple[str, ...],
    roles: dict,
    move_frames: int,
    central_clear_pixels: int,
    obstacle_passed_pixels: int,
    center_tolerance: float,
    stall_area_delta: int,
    score_threshold: float,
    max_detour_moves: int,
    min_bypass_moves: int,
    max_bypass_moves: int,
    max_recenter_moves: int,
    max_final_moves: int,
    obstacle_offset: int,
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    recorder = DirectionRecorder(controls)
    audit = PositionAudit()
    records = []
    trajectory = []
    private_actions = []
    with MineStudioVisualAdapter(
        recorder,
        simulator_factory=lambda config: obstacle_factory(
            config,
            audit,
            obstacle=True,
            obstacle_offset=obstacle_offset,
        ),
    ) as world:
        world.reset(seed)
        for _ in range(2):
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=10)))
        handles = set(movement_handles(world))
        if set(roles.values()) != handles:
            raise RuntimeError("opaque handles changed between calibration and route")

        def observe(phase: str) -> dict:
            frame = recorder.frames[-1]
            features = frame_features(frame)
            index = len(trajectory)
            item = {
                "observation_index": index,
                "phase": phase,
                "frame_sha256": world.observe().agent_state["frame_sha256"],
                **features,
            }
            trajectory.append(item)
            Image.fromarray(frame).save(output / f"frame-{index:02d}.png")
            return item

        def move(handle: str, phase: str) -> dict:
            records.append(world.step(PrimitiveAction(
                kind=ActionKind.MOVE,
                target_id=handle,
                ticks=move_frames,
            )))
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
            private_actions.append({
                "phase": phase,
                "handle_id": handle,
                "position": audit.positions[-1] if audit.positions else None,
            })
            return observe(phase)

        current = observe("initial")
        marker_center = current["marker"]["centroid_x"]
        initial_obstacle_pixels = current["central_obstacle_pixels"]
        initial_obstacle_area = (
            current["obstacle"]["pixel_count"] if current["obstacle"] else 0
        )
        obstacle_center = current["obstacle"]["centroid_x"]
        detour_role = (
            "positive_shift" if obstacle_center > marker_center else "negative_shift"
        )

        detour_moves = 0
        while (
            current["central_obstacle_pixels"] >= central_clear_pixels
            and detour_moves < max_detour_moves
        ):
            current = move(roles[detour_role], "detour")
            detour_moves += 1

        bypass_moves = 0
        extra_detour_moves = 0
        previous_marker = current["marker"]
        while (
            bypass_moves < min_bypass_moves
            or (
                current["obstacle"] is not None
                and current["obstacle"]["pixel_count"] >= obstacle_passed_pixels
            )
        ) and bypass_moves < max_bypass_moves:
            current = move(roles["goal"], "bypass")
            bypass_moves += 1
            marker = current["marker"]
            marker_stalled = (
                abs(marker["pixel_count"] - previous_marker["pixel_count"])
                <= stall_area_delta
                and abs(marker["centroid_x"] - previous_marker["centroid_x"])
                <= 0.25
                and abs(marker["centroid_y"] - previous_marker["centroid_y"])
                <= 0.25
            )
            if (
                marker_stalled
                and current["obstacle"] is not None
                and current["obstacle"]["pixel_count"] >= obstacle_passed_pixels
                and detour_moves + extra_detour_moves < max_detour_moves
            ):
                current = move(roles[detour_role], "local-replan")
                extra_detour_moves += 1
                marker = current["marker"]
            previous_marker = marker

        obstacle_passed_visually = (
            bypass_moves >= min_bypass_moves
            and (
                current["obstacle"] is None
                or current["obstacle"]["pixel_count"] < obstacle_passed_pixels
            )
        )

        recenter_moves = 0
        while recenter_moves < max_recenter_moves:
            marker_x = current["marker"]["centroid_x"]
            error = marker_x - marker_center
            if abs(error) <= center_tolerance:
                break
            handle = (
                roles["negative_shift"] if error > 0 else roles["positive_shift"]
            )
            current = move(handle, "recenter")
            recenter_moves += 1

        recentered_visually = (
            abs(current["marker"]["centroid_x"] - marker_center)
            <= center_tolerance
        )
        final_moves = 0
        while (
            current["salience"] < score_threshold
            and final_moves < max_final_moves
        ):
            current = move(roles["goal"], "final-approach")
            final_moves += 1

    save_events(output / "events.jsonl", records)
    save_json(output / "trajectory.json", trajectory)
    move_records = [
        record for record in records
        if record.executed_action.kind is ActionKind.MOVE
    ]
    public = {
        "initial_obstacle_pixels": initial_obstacle_pixels,
        "initial_obstacle_area": initial_obstacle_area,
        "detour_role": detour_role,
        "detour_moves": detour_moves,
        "extra_detour_moves": extra_detour_moves,
        "bypass_moves": bypass_moves,
        "recenter_moves": recenter_moves,
        "final_moves": final_moves,
        "path_cleared_visually": (
            trajectory[detour_moves]["central_obstacle_pixels"]
            < central_clear_pixels
        ),
        "obstacle_passed_visually": obstacle_passed_visually,
        "recentered_visually": recentered_visually,
        "goal_reached_visually": current["salience"] >= score_threshold,
        "final_salience": current["salience"],
        "all_move_actions_valid": bool(move_records) and all(
            record.intervention_compliance and record.event.success
            for record in move_records
        ),
        "continuous_route_episode": len({
            record.provenance["episode"] for record in records
        }) == 1,
    }
    private = {
        "positions": audit.positions,
        "actions": private_actions,
        "raw_metadata_keys": sorted(audit.metadata_keys),
    }
    return public, private


def _axis_delta(positions: list[dict], axis: str) -> float | None:
    if not positions or axis not in positions[0] or axis not in positions[-1]:
        return None
    return float(positions[-1][axis] - positions[0][axis])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--mapping-seed", type=int, default=900)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--central-clear-pixels", type=int, default=200)
    parser.add_argument("--obstacle-passed-pixels", type=int, default=80)
    parser.add_argument("--center-tolerance", type=float, default=6.0)
    parser.add_argument("--stall-area-delta", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=150.0)
    parser.add_argument("--max-detour-moves", type=int, default=6)
    parser.add_argument("--min-bypass-moves", type=int, default=7)
    parser.add_argument("--max-bypass-moves", type=int, default=12)
    parser.add_argument("--max-recenter-moves", type=int, default=8)
    parser.add_argument("--max-final-moves", type=int, default=12)
    parser.add_argument("--obstacle-offset", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.move_frames,
        args.central_clear_pixels,
        args.obstacle_passed_pixels,
        args.max_detour_moves,
        args.min_bypass_moves,
        args.max_bypass_moves,
        args.max_recenter_moves,
        args.max_final_moves,
    ) < 1:
        parser.error("budgets and pixel thresholds must be positive")
    if args.min_bypass_moves > args.max_bypass_moves:
        parser.error("minimum bypass moves cannot exceed the bypass budget")
    args.output.mkdir(parents=True, exist_ok=False)

    controls = tuple(random.Random(args.mapping_seed).sample(
        ["forward", "back", "left", "right"], k=4
    ))
    calibration = []
    private_trials = []
    for index in range(4):
        public, private = calibrate_handle(
            seed=args.seed,
            controls=controls,
            handle_index=index,
            move_frames=args.move_frames,
            output=args.output / "calibration" / f"trial-{index}",
        )
        calibration.append(public)
        private_trials.append(private)
    roles = classify_handles(calibration)
    save_json(args.output / "calibration.json", calibration)
    save_json(args.output / "roles.json", roles)

    route, private_route = run_route(
        seed=args.seed,
        controls=controls,
        roles=roles,
        move_frames=args.move_frames,
        central_clear_pixels=args.central_clear_pixels,
        obstacle_passed_pixels=args.obstacle_passed_pixels,
        center_tolerance=args.center_tolerance,
        stall_area_delta=args.stall_area_delta,
        score_threshold=args.score_threshold,
        max_detour_moves=args.max_detour_moves,
        min_bypass_moves=args.min_bypass_moves,
        max_bypass_moves=args.max_bypass_moves,
        max_recenter_moves=args.max_recenter_moves,
        max_final_moves=args.max_final_moves,
        obstacle_offset=args.obstacle_offset,
        output=args.output / "route",
    )

    for item in private_trials:
        item["goal_progress"] = _axis_delta(item["positions"], "zpos")
        item["lateral_progress"] = _axis_delta(item["positions"], "xpos")
    goal_handle = max(private_trials, key=lambda item: item["goal_progress"])[
        "handle_id"
    ]
    positions = private_route["positions"]
    start = positions[0] if positions else {}
    lateral_offsets = [
        abs(item["xpos"] - start["xpos"])
        for item in positions
        if "xpos" in item and "xpos" in start
    ]
    goal_progress = _axis_delta(positions, "zpos")
    final_lateral_offset = (
        abs(positions[-1]["xpos"] - start["xpos"])
        if positions and "xpos" in start and "xpos" in positions[-1]
        else None
    )
    evaluator = {
        "seed": args.seed,
        "mapping_seed": args.mapping_seed,
        "mapping_sha256": hashlib.sha256(
            json.dumps(private_trials, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "trials": private_trials,
        "route": private_route,
        "fixture": {
            "goal_axis": "positive_z",
            "obstacle_center_x_relative_to_start": [
                args.obstacle_offset - 1.5,
                args.obstacle_offset + 1.5,
            ],
            "obstacle_center_z_relative_to_start": [2.5, 4.5],
        },
    }
    save_json(args.output / "evaluator-audit.json", evaluator)

    calibration_valid = all((
        calibration[next(i for i, item in enumerate(calibration)
                         if item["handle_id"] == roles["goal"])]
        ["marker_area_delta"] > 300,
        calibration[next(i for i, item in enumerate(calibration)
                         if item["handle_id"] == roles["away"])]
        ["marker_area_delta"] < -300,
        all(
            abs(next(item for item in calibration if item["handle_id"] == handle)
                ["marker_x_delta"]) > 4
            for handle in (roles["negative_shift"], roles["positive_shift"])
        ),
    ))
    report = {
        "seed": args.seed,
        "mapping_seed": args.mapping_seed,
        "obstacle_offset": args.obstacle_offset,
        "backend": "real-minestudio",
        "roles": roles,
        "calibration_valid": calibration_valid,
        "pixel_goal_role_matches_evaluator": roles["goal"] == goal_handle,
        "initial_obstacle_detected": (
            route["initial_obstacle_pixels"] >= args.central_clear_pixels
            and route["initial_obstacle_area"] >= args.obstacle_passed_pixels
        ),
        **route,
        "evaluator_goal_progress": goal_progress,
        "evaluator_max_lateral_excursion": max(lateral_offsets, default=None),
        "evaluator_final_lateral_offset": final_lateral_offset,
        "evaluator_passed_obstacle_plane": (
            goal_progress is not None and goal_progress > 5.0
        ),
        "pixel_only_policy": True,
        "scope": (
            "Reset-calibrated opaque controls followed by one continuous RGB-only "
            "route around a visible fixture obstacle."
        ),
    }
    save_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    if not all((
        report["calibration_valid"],
        report["pixel_goal_role_matches_evaluator"],
        report["initial_obstacle_detected"],
        report["path_cleared_visually"],
        report["obstacle_passed_visually"],
        report["recentered_visually"],
        report["goal_reached_visually"],
        report["all_move_actions_valid"],
        report["continuous_route_episode"],
        goal_progress is not None and goal_progress > 5.0,
        max(lateral_offsets, default=0.0) > 1.25,
        final_lateral_offset is not None and final_lateral_offset < 0.9,
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
