"""Persistent RGB-only multi-place frontier expedition on natural MineStudio terrain.

The policy discovers a reversible anonymous motion program, learns a coordinate-free
A-B-C route under two visual conditions, restarts, routes to a remote unresolved
frontier, discovers D, returns home, then restarts once more and revisits D.  Absolute
position is captured by a separate evaluator only and never reaches policy memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    Session,
    _candidate_headings,
    _capture_scan,
    _discover_scan,
    _move,
    _movement_targets,
    _open_session,
    _orient,
    _orientation_return_error,
    _position_metrics,
    _probe_heading,
    _write_events,
    natural_world_commands,
    view_openness,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    VisualFrontierProgram,
    align_panorama_heading,
)

HEADING_MODALITY_SUFFIX = ".ordered-heading-v1"


def _flatten_views(views: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    if not views or not views[0] or any(len(item) != len(views[0]) for item in views):
        raise ValueError("ordered heading views must be nonempty and equally sized")
    return tuple(value for view in views for value in view)


def _unflatten_views(
    signature: tuple[int, ...], *, period: int, dimensions: int
) -> tuple[tuple[int, ...], ...]:
    if len(signature) != period * dimensions:
        raise ValueError("stored heading reference has an invalid width")
    return tuple(
        tuple(signature[index : index + dimensions])
        for index in range(0, len(signature), dimensions)
    )


def _program(
    *, turn_steps: int, motion_slot: int, repetitions: int, move_frames: int
) -> tuple[str, ...]:
    return VisualFrontierProgram(
        turn_steps=turn_steps,
        motion_slot=motion_slot,
        repetitions=repetitions,
        move_frames=move_frames,
    ).to_roles()


def _segment_metrics(
    positions: list[dict[str, float]], start_index: int
) -> dict[str, float | None]:
    start = max(0, start_index - 1)
    return _position_metrics(positions[start:])


def _execute_visual_program(
    session: Session,
    *,
    roles: tuple[str, ...],
    period: int,
    reference_views: tuple[tuple[int, ...], ...],
    query_views: tuple[tuple[int, ...], ...],
    source_scan_role: str,
    reference_scan_role: str,
) -> dict[str, Any]:
    program = VisualFrontierProgram.from_roles(roles)
    if program.turn_steps >= period:
        raise ValueError("visual frontier turn slot exceeds the learned period")
    alignment = align_panorama_heading(reference_views, query_views)
    _orient(session, alignment.right_turn_steps)
    _orient(session, program.turn_steps)
    targets = _movement_targets(session.world)
    if program.motion_slot >= len(targets):
        raise RuntimeError("persisted anonymous motion slot is unavailable")
    position_start = len(session.audit.positions)
    _move(
        session,
        moves=program.repetitions,
        move_frames=program.move_frames,
        target_id=targets[program.motion_slot],
    )
    return {
        "source_scan_role": source_scan_role,
        "reference_scan_role": reference_scan_role,
        "action_roles": list(roles),
        "alignment": alignment.model_dump(mode="json"),
        "evaluator_metrics": _segment_metrics(
            session.audit.positions, position_start
        ),
    }


def _scan(
    session: Session,
    output: Path,
    scans: dict[str, Any],
    role: str,
    *,
    encoder,
    period: int,
    temporal_samples: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    signature, metadata, views = _capture_scan(
        session,
        output / "scans" / role,
        encoder=encoder,
        period=period,
        temporal_sample_count=temporal_samples,
    )
    scans[role] = metadata
    return signature, views


def _scan_openness(root: Path, role: str, period: int) -> tuple[float, ...]:
    values = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            values.append(view_openness(np.asarray(image.convert("RGB"))))
    return tuple(values)


def _record_resolved_frontier(
    memory: ExplorationMemory,
    source_id: str,
    target_id: str,
    roles: tuple[str, ...],
    *,
    episode_id: int,
    step: int,
    cost: float,
) -> str:
    frontier = memory.register_frontier(
        source_id,
        roles,
        episode_id=episode_id,
        step=step,
        expected_cost=cost,
    )
    memory.record_frontier_outcome(
        frontier.frontier_id,
        episode_id=episode_id,
        step=step + 1,
        success=True,
        cost=cost,
        discovered_place_id=target_id,
    )
    return frontier.frontier_id


def _copy_sources(output: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sources = {
        "natural_visual_expedition_benchmark.py": Path(__file__).resolve(),
        "audit_natural_visual_expedition.py": Path(__file__).with_name(
            "audit_natural_visual_expedition.py"
        ),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
        "frontier_program.py": root
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "memory.py": root / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": root / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": root / "src/minecraft_scientist/minecraft/minestudio.py",
    }
    target = output / "source"
    target.mkdir()
    hashes = {}
    for name, source in sources.items():
        destination = target / name
        shutil.copy2(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


def _recognize(
    memory_path: Path, modality: str, signature: tuple[int, ...]
) -> str | None:
    with ExplorationMemory.open(memory_path) as memory:
        return memory.match_modality_place(modality, signature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--minimum-turn-steps", type=int, default=8)
    parser.add_argument("--maximum-turn-steps", type=int, default=16)
    parser.add_argument("--temporal-samples", type=int, default=3)
    parser.add_argument("--heading-probe-candidates", type=int, default=6)
    parser.add_argument("--frontier-probe-candidates", type=int, default=6)
    parser.add_argument(
        "--reuse-heading-probes",
        type=Path,
        help=(
            "copy a prior run's complete RGB probe evidence and independently "
            "revalidate only its selected intervention"
        ),
    )
    parser.add_argument("--panorama-encoder", choices=tuple(PANORAMA_ENCODERS), default="edge")
    parser.add_argument("--training-time", default="noon")
    parser.add_argument("--training-weather", default="clear")
    parser.add_argument("--calibration-time", default="night")
    parser.add_argument("--calibration-weather", default="rain")
    parser.add_argument("--heldout-time", default="midnight")
    parser.add_argument("--heldout-weather", default="thunder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.moves < 1 or args.move_frames < 1:
        parser.error("movement budgets must be positive")
    if args.temporal_samples < 1 or args.temporal_samples % 2 == 0:
        parser.error("temporal samples must be a positive odd integer")
    if args.minimum_turn_steps < 4 or args.maximum_turn_steps < args.minimum_turn_steps:
        parser.error("invalid camera-period search interval")
    if args.heading_probe_candidates < 1:
        parser.error("heading probe candidate count must be positive")
    if args.frontier_probe_candidates < 1:
        parser.error("frontier probe candidate count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    training_commands = natural_world_commands(
        time_of_day=args.training_time, weather=args.training_weather
    )
    calibration_commands = natural_world_commands(
        time_of_day=args.calibration_time, weather=args.calibration_weather
    )
    heldout_commands = natural_world_commands(
        time_of_day=args.heldout_time, weather=args.heldout_weather
    )
    if len({training_commands, calibration_commands, heldout_commands}) != 3:
        parser.error("training, calibration, and held-out conditions must differ")
    encoder = PANORAMA_ENCODERS[args.panorama_encoder]()
    movement_programs = (("forward",), ("forward", "jump"))
    source_hashes = _copy_sources(args.output)
    sessions: list[Session] = []
    scans: dict[str, Any] = {}
    executions: list[dict[str, Any]] = []

    try:
        discovery = _open_session(args.seed, training_commands, movement_programs)
        sessions.append(discovery)
        period, _anchor, scans["discovery"], _heading, _views = _discover_scan(
            discovery,
            args.output / "scans/discovery",
            encoder=encoder,
            minimum_steps=args.minimum_turn_steps,
            maximum_steps=args.maximum_turn_steps,
        )
        discovery_orientation_error = _orientation_return_error(
            discovery.audit.positions
        )
        movement_target_ids = _movement_targets(discovery.world)
        discovery.world.close()

        openness = tuple(float(item) for item in scans["discovery"]["openness_scores"])
        candidates = _candidate_headings(
            openness, min(args.heading_probe_candidates, period)
        )
        reused_probe_sha256 = None
        if args.reuse_heading_probes is not None:
            reuse_root = args.reuse_heading_probes.resolve()
            prior_path = reuse_root / "heading-probe.json"
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if (
                tuple(int(item) for item in prior["candidate_headings"])
                != candidates
                or tuple(str(item) for item in prior["movement_target_ids"])
                != movement_target_ids
            ):
                parser.error("reused heading probes do not match this RGB discovery")
            probe_results = []
            for item in prior["results"]:
                copied = dict(item)
                copied.pop("session_index", None)
                probe_results.append(copied)
            shutil.copytree(
                reuse_root / "heading-probes", args.output / "heading-probes"
            )
            reused_probe_sha256 = hashlib.sha256(
                prior_path.read_bytes()
            ).hexdigest()
        else:
            probe_results = []
            for heading in candidates:
                for motion_slot, target_id in enumerate(movement_target_ids):
                    probe = _open_session(
                        args.seed, training_commands, movement_programs
                    )
                    sessions.append(probe)
                    result = _probe_heading(
                        probe,
                        encoder=encoder,
                        heading_steps=heading,
                        program_index=motion_slot,
                        movement_target_id=target_id,
                        period=period,
                        moves=args.moves,
                        move_frames=args.move_frames,
                        output=args.output / "heading-probes",
                    )
                    result["session_index"] = len(sessions) - 1
                    probe_results.append(result)
                    probe.world.close()
        selected = max(
            probe_results,
            key=lambda item: (
                float(item["reversibility_score"]),
                float(item["outbound_distance"]),
                -int(item["heading_steps"]),
                -int(item["program_index"]),
            ),
        )
        selected_heading = int(selected["heading_steps"])
        selected_motion_slot = int(selected["program_index"])
        validation_result = None
        if args.reuse_heading_probes is not None:
            validation = _open_session(
                args.seed, training_commands, movement_programs
            )
            sessions.append(validation)
            validation_result = _probe_heading(
                validation,
                encoder=encoder,
                heading_steps=selected_heading,
                program_index=selected_motion_slot,
                movement_target_id=movement_target_ids[selected_motion_slot],
                period=period,
                moves=args.moves,
                move_frames=args.move_frames,
                output=args.output / "selected-probe-validation",
            )
            selected_probe_metrics = _position_metrics(validation.audit.positions)
            validation.world.close()
        else:
            selected_probe_metrics = _position_metrics(
                sessions[int(selected["session_index"])].audit.positions
            )
        heading_probe = {
            "selector": "rgb-reversibility-v2",
            "reused": args.reuse_heading_probes is not None,
            "reused_probe_sha256": reused_probe_sha256,
            "candidate_headings": list(candidates),
            "movement_target_ids": list(movement_target_ids),
            "results": probe_results,
            "selected_heading_steps": selected_heading,
            "selected_motion_slot": selected_motion_slot,
            "selected_evaluator_metrics": selected_probe_metrics,
            "selected_validation": validation_result,
        }
        (args.output / "heading-probe.json").write_text(
            json.dumps(heading_probe, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        roles_ab = _program(
            turn_steps=selected_heading,
            motion_slot=selected_motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
        roles_bc = _program(
            turn_steps=0,
            motion_slot=selected_motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
        roles_reverse = _program(
            turn_steps=period // 2,
            motion_slot=selected_motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )

        training = _open_session(args.seed, training_commands, movement_programs)
        sessions.append(training)
        sig_a1, views_a1 = _scan(
            training, args.output, scans, "train-a1", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            training, roles=roles_ab, period=period, reference_views=views_a1,
            query_views=views_a1, source_scan_role="train-a1",
            reference_scan_role="train-a1"
        ))
        sig_b1, views_b1 = _scan(
            training, args.output, scans, "train-b1", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            training, roles=roles_bc, period=period, reference_views=views_b1,
            query_views=views_b1, source_scan_role="train-b1",
            reference_scan_role="train-b1"
        ))
        sig_c1, views_c1 = _scan(
            training, args.output, scans, "train-c1", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        sig_c2, views_c2 = _scan(
            training, args.output, scans, "train-c2", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            training, roles=roles_reverse, period=period, reference_views=views_c1,
            query_views=views_c2, source_scan_role="train-c2",
            reference_scan_role="train-c1"
        ))
        _sig_b_return, views_b_return = _scan(
            training, args.output, scans, "train-b-return", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            training, roles=roles_reverse, period=period, reference_views=views_b1,
            query_views=views_b_return, source_scan_role="train-b-return",
            reference_scan_role="train-b1"
        ))
        _sig_a_return, _views_a_return = _scan(
            training, args.output, scans, "train-a-return", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        training_metrics = _position_metrics(training.audit.positions)
        training.world.close()

        calibration = _open_session(
            args.seed, calibration_commands, movement_programs
        )
        sessions.append(calibration)
        sig_a_cal, views_a_cal = _scan(
            calibration, args.output, scans, "calibration-a", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            calibration, roles=roles_ab, period=period, reference_views=views_a1,
            query_views=views_a_cal, source_scan_role="calibration-a",
            reference_scan_role="train-a1"
        ))
        sig_b_cal, views_b_cal = _scan(
            calibration, args.output, scans, "calibration-b", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        executions.append(_execute_visual_program(
            calibration, roles=roles_bc, period=period, reference_views=views_b1,
            query_views=views_b_cal, source_scan_role="calibration-b",
            reference_scan_role="train-b1"
        ))
        sig_c_cal, _views_c_cal = _scan(
            calibration, args.output, scans, "calibration-c", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        calibration_metrics = _position_metrics(calibration.audit.positions)
        calibration.world.close()

        c_openness = _scan_openness(args.output, "train-c1", period)
        frontier_candidates = tuple(
            sorted(
                (index for index in range(period) if index != period // 2),
                key=lambda index: (-c_openness[index], index),
            )[: min(args.frontier_probe_candidates, period - 1)]
        )
        frontier_probe_results = []
        for probe_index, candidate in enumerate(frontier_candidates):
            frontier_probe = _open_session(
                args.seed, training_commands, movement_programs
            )
            sessions.append(frontier_probe)
            probe_prefix = f"frontier-probe-{probe_index:02d}"
            _sig_probe_a, probe_views = _scan(
                frontier_probe,
                args.output,
                scans,
                f"{probe_prefix}-a",
                encoder=encoder,
                period=period,
                temporal_samples=1,
            )
            probe_executions = [
                _execute_visual_program(
                    frontier_probe,
                    roles=roles_ab,
                    period=period,
                    reference_views=views_a1,
                    query_views=probe_views,
                    source_scan_role=f"{probe_prefix}-a",
                    reference_scan_role="train-a1",
                )
            ]
            _sig_probe_b, probe_views = _scan(
                frontier_probe,
                args.output,
                scans,
                f"{probe_prefix}-b",
                encoder=encoder,
                period=period,
                temporal_samples=1,
            )
            probe_executions.append(
                _execute_visual_program(
                    frontier_probe,
                    roles=roles_bc,
                    period=period,
                    reference_views=views_b1,
                    query_views=probe_views,
                    source_scan_role=f"{probe_prefix}-b",
                    reference_scan_role="train-b1",
                )
            )
            sig_probe_c, probe_views_c = _scan(
                frontier_probe,
                args.output,
                scans,
                f"{probe_prefix}-c",
                encoder=encoder,
                period=period,
                temporal_samples=1,
            )
            candidate_roles = _program(
                turn_steps=candidate,
                motion_slot=selected_motion_slot,
                repetitions=args.moves,
                move_frames=args.move_frames,
            )
            loop_position_start = len(frontier_probe.audit.positions)
            probe_executions.append(
                _execute_visual_program(
                    frontier_probe,
                    roles=candidate_roles,
                    period=period,
                    reference_views=views_c1,
                    query_views=probe_views_c,
                    source_scan_role=f"{probe_prefix}-c",
                    reference_scan_role="train-c1",
                )
            )
            sig_probe_outbound, probe_views_outbound = _scan(
                frontier_probe,
                args.output,
                scans,
                f"{probe_prefix}-outbound",
                encoder=encoder,
                period=period,
                temporal_samples=1,
            )
            probe_executions.append(
                _execute_visual_program(
                    frontier_probe,
                    roles=roles_reverse,
                    period=period,
                    reference_views=probe_views_outbound,
                    query_views=probe_views_outbound,
                    source_scan_role=f"{probe_prefix}-outbound",
                    reference_scan_role=f"{probe_prefix}-outbound",
                )
            )
            sig_probe_return, _probe_views_return = _scan(
                frontier_probe,
                args.output,
                scans,
                f"{probe_prefix}-return",
                encoder=encoder,
                period=period,
                temporal_samples=1,
            )
            outbound_distance = math.dist(sig_probe_c, sig_probe_outbound)
            return_distance = math.dist(sig_probe_c, sig_probe_return)
            frontier_probe_results.append(
                {
                    "heading_steps": candidate,
                    "openness": c_openness[candidate],
                    "outbound_distance": outbound_distance,
                    "return_distance": return_distance,
                    "reversibility_score": outbound_distance
                    / (return_distance + 1.0),
                    "loop_evaluator_metrics": _segment_metrics(
                        frontier_probe.audit.positions, loop_position_start
                    ),
                    "executions": probe_executions,
                    "session_index": len(sessions) - 1,
                }
            )
            frontier_probe.world.close()
        selected_frontier_probe = max(
            frontier_probe_results,
            key=lambda item: (
                float(item["reversibility_score"]),
                float(item["outbound_distance"]),
                -int(item["heading_steps"]),
            ),
        )
        frontier_heading = int(selected_frontier_probe["heading_steps"])
        frontier_probe = {
            "selector": "remote-rgb-reversibility-v1",
            "candidate_headings": list(frontier_candidates),
            "openness_scores": list(c_openness),
            "results": frontier_probe_results,
            "selected_heading_steps": frontier_heading,
            "selected_evaluator_metrics": selected_frontier_probe[
                "loop_evaluator_metrics"
            ],
        }
        roles_cd = _program(
            turn_steps=frontier_heading,
            motion_slot=selected_motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
        decoy_heading = (selected_heading + 1) % period
        roles_decoy = _program(
            turn_steps=decoy_heading,
            motion_slot=selected_motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )

        memory_path = args.output / "exploration.sqlite3"
        heading_modality = encoder.VERSION + HEADING_MODALITY_SUFFIX
        with ExplorationMemory(memory_path, max_exemplars=6) as memory:
            place_a = memory.observe_place((0,), episode_id=0, step=0)
            place_b = memory.observe_place((1,), episode_id=0, step=1)
            place_c = memory.observe_place((2,), episode_id=0, step=2)
            place_data = (
                (place_a.place_id, sig_a1, views_a1, (sig_a1, sig_a_cal),
                 (sig_b1, sig_b_cal, sig_c1, sig_c_cal)),
                # The reverse traversal is validation evidence, not a calibration
                # label: route drift may place it between B and C.  Treating it as
                # a positive would silently teach the recognizer that drift is B.
                (place_b.place_id, sig_b1, views_b1, (sig_b1, sig_b_cal),
                 (sig_a1, sig_a_cal, sig_c1, sig_c_cal)),
                (place_c.place_id, sig_c1, views_c1, (sig_c1, sig_c2, sig_c_cal),
                 (sig_a1, sig_a_cal, sig_b1, sig_b_cal)),
            )
            for index, (place_id, anchor, heading_views, positives, negatives) in enumerate(place_data):
                memory.record_modality_view(
                    encoder.VERSION, place_id, anchor,
                    episode_id=0, step=10 + index
                )
                memory.calibrate_modality_place(
                    encoder.VERSION, place_id, positives, negatives,
                    episode_id=1, step=10 + index, margin=None
                )
                for prototype_index, prototype in enumerate(positives[1:], start=1):
                    memory.consolidate_modality_view(
                        encoder.VERSION,
                        place_id,
                        prototype,
                        episode_id=1,
                        step=20 + index * 4 + prototype_index,
                    )
                memory.record_modality_view(
                    heading_modality, place_id, _flatten_views(heading_views),
                    episode_id=0, step=40 + index
                )
            route_frontier_ids = {
                "a-b": _record_resolved_frontier(
                    memory, place_a.place_id, place_b.place_id, roles_ab,
                    episode_id=0, step=30, cost=args.moves
                ),
                "b-c": _record_resolved_frontier(
                    memory, place_b.place_id, place_c.place_id, roles_bc,
                    episode_id=0, step=32, cost=args.moves
                ),
            }
            remote_frontier = memory.register_frontier(
                place_c.place_id, roles_cd, episode_id=1, step=40,
                expected_cost=args.moves, uncertainty=1.0, risk=0.0
            )
            memory.register_frontier(
                place_a.place_id, roles_decoy, episode_id=1, step=41,
                expected_cost=args.moves, uncertainty=0.05, risk=10.0
            )
            preregistered_summary = memory.summary()

        calibration_recognition = {
            "a": _recognize(memory_path, encoder.VERSION, sig_a_cal),
            "b": _recognize(memory_path, encoder.VERSION, sig_b_cal),
            "c": _recognize(memory_path, encoder.VERSION, sig_c_cal),
        }

        heldout = _open_session(args.seed, heldout_commands, movement_programs)
        sessions.append(heldout)
        sig_a_hold, views_a_hold = _scan(
            heldout, args.output, scans, "heldout-a", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        heldout_matches = [_recognize(memory_path, encoder.VERSION, sig_a_hold)]
        route_views = views_a_hold
        route_role = "heldout-a"
        with ExplorationMemory.open(memory_path) as memory:
            frontier_plan = memory.plan_next_frontier(place_a.place_id)
        if frontier_plan is None:
            raise RuntimeError("persistent planner found no reachable frontier")
        plan_snapshot = frontier_plan.model_dump(mode="json")
        reference_by_place = {
            place_a.place_id: ("train-a1", views_a1),
            place_b.place_id: ("train-b1", views_b1),
            place_c.place_id: ("train-c1", views_c1),
        }
        for edge_index, ((source_id, target_id), roles) in enumerate(
            zip(
                pairwise(frontier_plan.place_path),
                frontier_plan.travel_action_roles,
                strict=True,
            ),
            start=1,
        ):
            reference_role, reference_views = reference_by_place[source_id]
            executions.append(_execute_visual_program(
                heldout, roles=roles, period=period,
                reference_views=reference_views, query_views=route_views,
                source_scan_role=route_role, reference_scan_role=reference_role
            ))
            role = f"heldout-route-{edge_index}"
            signature, route_views = _scan(
                heldout, args.output, scans, role, encoder=encoder,
                period=period, temporal_samples=args.temporal_samples
            )
            heldout_matches.append(_recognize(memory_path, encoder.VERSION, signature))
            route_role = role
            if heldout_matches[-1] != target_id:
                raise RuntimeError("held-out travel failed visual place recognition")

        executions.append(_execute_visual_program(
            heldout, roles=tuple(frontier_plan.action_roles), period=period,
            reference_views=views_c1, query_views=route_views,
            source_scan_role=route_role, reference_scan_role="train-c1"
        ))
        sig_d1, views_d1 = _scan(
            heldout, args.output, scans, "heldout-d1", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        sig_d2, _views_d2 = _scan(
            heldout, args.output, scans, "heldout-d2", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        d_match_before_learning = _recognize(memory_path, encoder.VERSION, sig_d1)
        frontier_execution_metrics = executions[-1]["evaluator_metrics"]
        if d_match_before_learning is not None:
            raise RuntimeError("selected frontier did not discover a novel visual place")

        with ExplorationMemory.open(memory_path) as memory:
            place_d = memory.observe_place((3,), episode_id=2, step=0)
            memory.record_modality_view(
                encoder.VERSION, place_d.place_id, sig_d1,
                episode_id=2, step=1
            )
            memory.calibrate_modality_place(
                encoder.VERSION, place_d.place_id, (sig_d1, sig_d2),
                (sig_a_hold, sig_b_cal, sig_c_cal),
                episode_id=2, step=2, margin=None
            )
            memory.record_modality_view(
                heading_modality, place_d.place_id, _flatten_views(views_d1),
                episode_id=2, step=3
            )
            memory.record_frontier_outcome(
                remote_frontier.frontier_id, episode_id=2, step=4,
                success=True, cost=args.moves,
                discovered_place_id=place_d.place_id
            )

        heldout_metrics = _position_metrics(heldout.audit.positions)
        heldout.world.close()

        revisit = _open_session(args.seed, heldout_commands, movement_programs)
        sessions.append(revisit)
        sig_revisit_a, revisit_views = _scan(
            revisit, args.output, scans, "revisit-a", encoder=encoder,
            period=period, temporal_samples=args.temporal_samples
        )
        revisit_matches = [_recognize(memory_path, encoder.VERSION, sig_revisit_a)]
        with ExplorationMemory.open(memory_path) as memory:
            revisit_plan = memory.plan_route(place_a.place_id, place_d.place_id)
        if revisit_plan is None:
            raise RuntimeError("persistent memory could not plan a revisit to D")
        revisit_plan_snapshot = revisit_plan.model_dump(mode="json")
        reference_by_place[place_d.place_id] = ("heldout-d1", views_d1)
        route_role = "revisit-a"
        for edge_index, ((source_id, target_id), roles) in enumerate(
            zip(
                pairwise(revisit_plan.place_path),
                revisit_plan.travel_action_roles,
                strict=True,
            ),
            start=1,
        ):
            reference_role, reference_views = reference_by_place[source_id]
            executions.append(_execute_visual_program(
                revisit, roles=roles, period=period,
                reference_views=reference_views, query_views=revisit_views,
                source_scan_role=route_role, reference_scan_role=reference_role
            ))
            role = f"revisit-route-{edge_index}"
            signature, revisit_views = _scan(
                revisit, args.output, scans, role, encoder=encoder,
                period=period, temporal_samples=args.temporal_samples
            )
            match = _recognize(memory_path, encoder.VERSION, signature)
            revisit_matches.append(match)
            route_role = role
            if match != target_id:
                raise RuntimeError("persistent revisit route lost visual localization")
        revisit_metrics = _position_metrics(revisit.audit.positions)
        revisit.world.close()

        with ExplorationMemory.open(memory_path) as memory:
            memory_summary = memory.summary()
            memory_events = tuple(memory.iter_events())
            remote_state = memory.get_frontier(remote_frontier.frontier_id)
        (args.output / "memory-events.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in memory_events) + "\n",
            encoding="utf-8",
        )
        _write_events(args.output / "events.jsonl", sessions)
        public_text = "\n".join(
            (args.output / name).read_text(encoding="utf-8")
            for name in ("events.jsonl", "memory-events.jsonl")
        )
        public_telemetry_clean = not any(
            token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
        )
        action_records = [record for session in sessions for record in session.records]
        actions_compliant = bool(action_records) and all(
            record.intervention_compliance and record.event.success
            for record in action_records
        )
        expected_abc = [place_a.place_id, place_b.place_id, place_c.place_id]
        expected_abcd = [*expected_abc, place_d.place_id]
        gate = {
            "camera_period_even": period % 2 == 0,
            "camera_orientation_restored": discovery_orientation_error is not None
            and discovery_orientation_error <= 1.0,
            "selected_probe_physically_reversible":
                selected_probe_metrics["maximum_excursion"] is not None
                and selected_probe_metrics["maximum_excursion"] >= 2.0
                and selected_probe_metrics["return_error"] is not None
                and selected_probe_metrics["return_error"] <= 1.5,
            "remote_probe_physically_reversible":
                frontier_probe["selected_evaluator_metrics"]["maximum_excursion"]
                is not None
                and frontier_probe["selected_evaluator_metrics"]["maximum_excursion"]
                >= 2.0
                and frontier_probe["selected_evaluator_metrics"]["return_error"]
                is not None
                and frontier_probe["selected_evaluator_metrics"]["return_error"]
                <= 1.5,
            "training_three_place_excursion":
                training_metrics["maximum_excursion"] is not None
                and training_metrics["maximum_excursion"] >= 6.0,
            "condition_shift_calibration_recognized": list(calibration_recognition.values())
            == expected_abc,
            "remote_frontier_planned_after_restart":
                frontier_plan.frontier_id == remote_frontier.frontier_id
                and list(frontier_plan.place_path) == expected_abc,
            "heldout_route_recognized": heldout_matches == expected_abc,
            "frontier_discovered_novel_place": d_match_before_learning is None,
            "frontier_physically_explored":
                frontier_execution_metrics["maximum_excursion"] is not None
                and frontier_execution_metrics["maximum_excursion"] >= 2.0,
            "heldout_four_place_excursion":
                heldout_metrics["maximum_excursion"] is not None
                and heldout_metrics["maximum_excursion"] >= 8.0,
            "persistent_revisit_recognized": revisit_matches == expected_abcd,
            "persistent_revisit_physically_remote":
                revisit_metrics["maximum_excursion"] is not None
                and revisit_metrics["maximum_excursion"] >= 8.0,
            "remote_frontier_resolved": remote_state.status.value == "resolved"
            and remote_state.discovered_place_id == place_d.place_id,
            "memory_integrity": memory_summary["integrity_valid"] is True,
            "public_telemetry_clean": public_telemetry_clean,
            "actions_compliant": actions_compliant,
        }
        evaluator = {
            "sessions": [
                {
                    "positions": session.audit.positions,
                    "raw_metadata_keys": sorted(session.audit.metadata_keys),
                    "startup_attempts": session.startup_attempts,
                }
                for session in sessions
            ],
            "training_session_index": sessions.index(training),
            "calibration_session_index": sessions.index(calibration),
            "heldout_session_index": sessions.index(heldout),
            "revisit_session_index": sessions.index(revisit),
            "training_metrics": training_metrics,
            "calibration_metrics": calibration_metrics,
            "heldout_metrics": heldout_metrics,
            "revisit_metrics": revisit_metrics,
        }
        report = {
            "suite": "natural_visual_expedition",
            "backend": "real-minestudio",
            "seed": args.seed,
            "terrain_modified": False,
            "encoder": encoder.VERSION,
            "descriptor_dimensions": encoder.dimensions,
            "inferred_camera_period": period,
            "temporal_samples": args.temporal_samples,
            "conditions": {
                "training": {"time": args.training_time, "weather": args.training_weather},
                "calibration": {"time": args.calibration_time, "weather": args.calibration_weather},
                "heldout": {"time": args.heldout_time, "weather": args.heldout_weather},
            },
            "selected_heading_steps": selected_heading,
            "selected_motion_slot": selected_motion_slot,
            "frontier_heading_steps": frontier_heading,
            "heading_probe": heading_probe,
            "frontier_probe": frontier_probe,
            "place_ids": {"a": place_a.place_id, "b": place_b.place_id,
                          "c": place_c.place_id, "d": place_d.place_id},
            "route_frontier_ids": route_frontier_ids,
            "remote_frontier_id": remote_frontier.frontier_id,
            "preregistered_memory": preregistered_summary,
            "memory": memory_summary,
            "calibration_recognition": calibration_recognition,
            "heldout_matches": heldout_matches,
            "revisit_matches": revisit_matches,
            "d_match_before_learning": d_match_before_learning,
            "frontier_plan": plan_snapshot,
            "revisit_plan": revisit_plan_snapshot,
            "executions": executions,
            "training_evaluator_metrics": training_metrics,
            "calibration_evaluator_metrics": calibration_metrics,
            "heldout_evaluator_metrics": heldout_metrics,
            "revisit_evaluator_metrics": revisit_metrics,
            "frontier_evaluator_metrics": frontier_execution_metrics,
            "session_startup_attempts": [session.startup_attempts for session in sessions],
            "scans": scans,
            "source_sha256": source_hashes,
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Natural generated terrain; RGB-only cyclic heading localization, "
                "persistent four-place topology, remote frontier planning, and "
                "restart revisit. Evaluator-only coordinates prove movement. This "
                "is a bounded outward expedition without a homing guarantee, not "
                "unrestricted survival, cross-biome SLAM, or a claim of universal "
                "visual invariance."
            ),
        }
        config = {
            "seed": args.seed,
            "moves": args.moves,
            "move_frames": args.move_frames,
            "minimum_turn_steps": args.minimum_turn_steps,
            "maximum_turn_steps": args.maximum_turn_steps,
            "temporal_samples": args.temporal_samples,
            "heading_probe_candidates": args.heading_probe_candidates,
            "frontier_probe_candidates": args.frontier_probe_candidates,
            "reused_heading_probes": args.reuse_heading_probes is not None,
            "reused_probe_sha256": reused_probe_sha256,
            "panorama_encoder": args.panorama_encoder,
            "training_time": args.training_time,
            "training_weather": args.training_weather,
            "calibration_time": args.calibration_time,
            "calibration_weather": args.calibration_weather,
            "heldout_time": args.heldout_time,
            "heldout_weather": args.heldout_weather,
        }
        (args.output / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "evaluator-audit.json").write_text(
            json.dumps(evaluator, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["gate"]["passed"]:
            raise SystemExit(1)
    finally:
        for session in sessions:
            session.world.close()


if __name__ == "__main__":
    main()
