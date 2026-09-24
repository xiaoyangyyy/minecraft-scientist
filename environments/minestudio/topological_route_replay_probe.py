"""Replay a persisted coordinate-free visual route under new opaque controls."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

from direction_discovery_probe import DirectionRecorder, movement_handles
from navigation_probe import PositionAudit
from obstacle_replanning_probe import (
    _axis_delta,
    calibrate_handle,
    classify_handles,
    frame_features,
    obstacle_factory,
    save_events,
    save_json,
)
from PIL import Image

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft import MineStudioVisualAdapter
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


def landmark_signature(features: dict) -> tuple[int, ...]:
    """Quantize Agent-visible geometry into a stable, nonsemantic landmark."""

    marker = features.get("marker")
    if marker is None:
        raise ValueError("route landmark requires a visible achromatic region")
    obstacle = features.get("obstacle")
    return (
        int(obstacle is not None),
        min(12, int(features["central_obstacle_pixels"]) // 200),
        int(marker["centroid_x"] // 8),
        int(marker["pixel_count"] // 500),
        int(float(features["salience"]) // 10),
    )


def run_replay(
    *,
    seed: int,
    controls: tuple[str, ...],
    roles: dict[str, str],
    topology: VisualTopologicalMap,
    target_id: str,
    move_frames: int,
    score_threshold: float,
    center_tolerance: float,
    obstacle_passed_pixels: int,
    obstacle_offset: int,
    signature_mode: str,
    output: Path,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    recorder = DirectionRecorder(controls)
    audit = PositionAudit()
    records = []
    trajectory = []
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
        if set(roles.values()) != set(movement_handles(world)):
            raise RuntimeError("opaque handles changed between calibration and replay")

        def observe(index: int, role: str | None) -> dict:
            frame = recorder.frames[-1]
            features = frame_features(frame)
            item = {
                "observation_index": index,
                "action_role": role,
                "frame_sha256": world.observe().agent_state["frame_sha256"],
                **features,
            }
            trajectory.append(item)
            Image.fromarray(frame).save(output / f"frame-{index:02d}.png")
            return item

        initial = observe(0, None)
        signature_function = (
            layout_landmark_signature
            if signature_mode == "layout"
            else landmark_signature
        )
        source_signature = signature_function(initial)
        matched_source_id = topology.match(source_signature)
        plan = topology.plan(source_signature, target_id)
        if plan is None:
            raise RuntimeError("persisted map has no confident route for this view")
        handles = plan.bind(roles)
        current = initial
        for index, (role, handle) in enumerate(
            zip(plan.action_roles, handles, strict=True), start=1
        ):
            records.append(world.step(PrimitiveAction(
                kind=ActionKind.MOVE,
                target_id=handle,
                ticks=move_frames,
            )))
            records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2)))
            current = observe(index, role)

    save_events(output / "events.jsonl", records)
    save_json(output / "trajectory.json", trajectory)
    obstacle_areas = [
        item["obstacle"]["pixel_count"] if item["obstacle"] else 0
        for item in trajectory
    ]
    passed_observations = [
        item["observation_index"]
        for item, area in zip(trajectory[1:], obstacle_areas[1:], strict=True)
        if item["action_role"] == "goal" and area < obstacle_passed_pixels
    ]
    move_records = [
        record for record in records
        if record.executed_action.kind is ActionKind.MOVE
    ]
    public = {
        "source_signature": source_signature,
        "matched_source_id": matched_source_id,
        "target_id": target_id,
        "route_ids": plan.route_ids,
        "action_roles": plan.action_roles,
        "program_length": len(plan.action_roles),
        "expected_cost": plan.expected_cost,
        "route_confidence": plan.confidence,
        "initial_obstacle_detected": initial["obstacle"] is not None,
        "obstacle_passed_visually": bool(passed_observations),
        "obstacle_passed_observation": (
            passed_observations[0] if passed_observations else None
        ),
        "recentered_visually": abs(
            current["marker"]["centroid_x"]
            - initial["marker"]["centroid_x"]
        ) <= center_tolerance,
        "goal_reached_visually": current["salience"] >= score_threshold,
        "final_salience": current["salience"],
        "all_move_actions_valid": len(move_records) == len(plan.action_roles)
        and all(
            record.intervention_compliance and record.event.success
            for record in move_records
        ),
        "continuous_replay_episode": len({
            record.provenance["episode"] for record in records
        }) == 1,
        "online_route_exploration_actions": 0,
    }
    private = {
        "positions": audit.positions,
        "raw_metadata_keys": sorted(audit.metadata_keys),
    }
    return public, private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mapping-seed", type=int, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=150.0)
    parser.add_argument("--center-tolerance", type=float, default=6.0)
    parser.add_argument("--obstacle-passed-pixels", type=int, default=80)
    parser.add_argument("--obstacle-offset", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument(
        "--signature-mode", choices=("legacy", "layout"), default="legacy"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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

    topology = VisualTopologicalMap.load(args.map)
    replay, private_replay = run_replay(
        seed=args.seed,
        controls=controls,
        roles=roles,
        topology=topology,
        target_id=args.target_id,
        move_frames=args.move_frames,
        score_threshold=args.score_threshold,
        center_tolerance=args.center_tolerance,
        obstacle_passed_pixels=args.obstacle_passed_pixels,
        obstacle_offset=args.obstacle_offset,
        signature_mode=args.signature_mode,
        output=args.output / "replay",
    )

    for item in private_trials:
        item["goal_progress"] = _axis_delta(item["positions"], "zpos")
        item["lateral_progress"] = _axis_delta(item["positions"], "xpos")
    positions = private_replay["positions"]
    start = positions[0] if positions else {}
    lateral_offsets = [
        abs(item["xpos"] - start["xpos"])
        for item in positions
        if "xpos" in item and "xpos" in start
    ]
    goal_progress = _axis_delta(positions, "zpos")
    final_offset = (
        abs(positions[-1]["xpos"] - start["xpos"])
        if positions and "xpos" in start and "xpos" in positions[-1]
        else None
    )
    evaluator = {
        "seed": args.seed,
        "mapping_seed": args.mapping_seed,
        "obstacle_offset": args.obstacle_offset,
        "signature_mode": args.signature_mode,
        "mapping_sha256": hashlib.sha256(
            json.dumps(private_trials, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "trials": private_trials,
        "replay": private_replay,
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

    goal_handle = max(private_trials, key=lambda item: item["goal_progress"])[
        "handle_id"
    ]
    report = {
        "seed": args.seed,
        "mapping_seed": args.mapping_seed,
        "obstacle_offset": args.obstacle_offset,
        "signature_mode": args.signature_mode,
        "backend": "real-minestudio",
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
        "roles": roles,
        "calibration_valid": all((
            roles["goal"] == goal_handle,
            len(set(roles.values())) == 4,
        )),
        **replay,
        "evaluator_goal_progress": goal_progress,
        "evaluator_max_lateral_excursion": max(lateral_offsets, default=None),
        "evaluator_final_lateral_offset": final_offset,
        "evaluator_passed_obstacle_plane": goal_progress is not None
        and goal_progress > 4.5,
        "coordinate_free_map_replay": True,
        "scope": (
            "Persisted quantized visual landmarks and a learned functional-role "
            "program replayed after independent opaque-control calibration."
        ),
    }
    save_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
