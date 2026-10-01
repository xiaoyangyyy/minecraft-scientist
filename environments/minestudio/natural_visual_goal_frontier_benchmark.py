"""Expand a persistent natural-world route toward a remembered visual goal.

The benchmark consumes an accepted B<->E detour.  Every candidate starts in a
fresh MineStudio session, records the current RGB panorama at A, replays the
persisted A->B->E route, probes one visually open heading from E, and returns.
Selection uses only RGB goal distance, novelty, and visual return recognition.
Coordinates are evaluator-only and may veto promotion after execution, but never
rank candidates.  A novel F (or a direct A revisit) is persisted only after an
independent shadow-validation session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from natural_visual_expedition_benchmark import (
    _execute_visual_program,
    _program,
    _record_resolved_frontier,
    _scan,
    _scan_openness,
)
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    Session,
    _candidate_headings,
    _move,
    _movement_targets,
    _open_session,
    _orient,
    _write_events,
    natural_world_commands,
)

from minecraft_scientist.exploration import (
    ExplorationMemory,
    VisualFrontierProgram,
    VisualHomingProgram,
    align_panorama_heading,
)

RETURN_TOLERANCE_3D = 0.75
MINIMUM_FRONTIER_DISTANCE_3D = 1.0
ReturnProgram = VisualHomingProgram | VisualFrontierProgram


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_tree_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for path in sorted(source.glob("view-*.png")):
        shutil.copy2(path, target / path.name)


def _copy_source_evidence(source: Path, output: Path) -> dict[str, str]:
    target = output / "source-evidence"
    target.mkdir()
    for name in ("report.json", "integrity-audit.json", "exploration.sqlite3"):
        shutil.copy2(source / name, target / name)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    selected = report["selected_result"]
    _copy_tree_files(
        source / "source-evidence" / "scans" / "train-a1",
        target / "scans" / "a-reference",
    )
    _copy_tree_files(
        source / "scans" / str(selected["b_scan_role"]),
        target / "scans" / "b-reference",
    )
    _copy_tree_files(
        source / "scans" / str(selected["e_scan_role"]),
        target / "scans" / "e-reference",
    )
    hashes = {
        path.relative_to(target).as_posix(): _digest(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    (output / "source-evidence-sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashes


def _copy_sources(output: Path) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    sources = {
        "natural_visual_goal_frontier_benchmark.py": Path(__file__).resolve(),
        "audit_natural_visual_goal_frontier.py": Path(__file__).with_name(
            "audit_natural_visual_goal_frontier.py"
        ),
        "natural_visual_detour_benchmark.py": Path(__file__).with_name(
            "natural_visual_detour_benchmark.py"
        ),
        "natural_visual_expedition_benchmark.py": Path(__file__).with_name(
            "natural_visual_expedition_benchmark.py"
        ),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "memory.py": repository / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": repository
        / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": repository
        / "src/minecraft_scientist/minecraft/minestudio.py",
    }
    target = output / "source"
    target.mkdir()
    hashes = {}
    for name, source in sources.items():
        destination = target / name
        shutil.copy2(source, destination)
        hashes[name] = _digest(destination)
    return hashes


def _load_views(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[tuple[int, ...], ...]:
    import numpy as np
    from PIL import Image

    frames = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return tuple(encoder.scene_encoder.encode(frame) for frame in frames)


def _recognize(
    memory_path: Path, modality: str, signature: tuple[int, ...]
) -> str | None:
    with ExplorationMemory.open(memory_path) as memory:
        return memory.match_modality_place(modality, signature)


def _distance_3d(
    positions: list[dict[str, float]], left_index: int, right_index: int
) -> float:
    left, right = positions[left_index], positions[right_index]
    return math.sqrt(
        (float(left["xpos"]) - float(right["xpos"])) ** 2
        + (float(left["ypos"]) - float(right["ypos"])) ** 2
        + (float(left["zpos"]) - float(right["zpos"])) ** 2
    )


def _vertical_drop(
    positions: list[dict[str, float]], start_index: int, end_index: int
) -> float:
    start_y = float(positions[start_index]["ypos"])
    return max(
        start_y - float(item["ypos"])
        for item in positions[start_index : end_index + 1]
    )


def _replay_to_e(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    role_prefix: str,
    encoder: Any,
    modality: str,
    memory_path: Path,
    period: int,
    temporal_samples: int,
    place_ids: tuple[str, str, str],
    reference_views: tuple[tuple[tuple[int, ...], ...], ...],
    route_roles: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    dict[str, Any],
]:
    place_a, place_b, place_e = place_ids
    signature_a, query_views = _scan(
        session,
        output,
        scans,
        f"{role_prefix}-a",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    a_position_index = len(session.audit.positions) - 1
    executions = []
    for edge_index, (source_id, target_id, roles, stored_views) in enumerate(
        (
            (place_a, place_b, route_roles[0], reference_views[0]),
            (place_b, place_e, route_roles[1], reference_views[1]),
        ),
        start=1,
    ):
        source_role = f"{role_prefix}-a" if edge_index == 1 else f"{role_prefix}-b"
        execution = _execute_visual_program(
            session,
            roles=roles,
            period=period,
            reference_views=stored_views,
            query_views=query_views,
            source_scan_role=source_role,
            reference_scan_role=("a-reference" if edge_index == 1 else "b-reference"),
        )
        execution["source_place_id"] = source_id
        execution["target_place_id"] = target_id
        executions.append(execution)
        target_role = f"{role_prefix}-b" if edge_index == 1 else f"{role_prefix}-e"
        signature, query_views = _scan(
            session,
            output,
            scans,
            target_role,
            encoder=encoder,
            period=period,
            temporal_samples=temporal_samples,
        )
    e_match = _recognize(memory_path, modality, signature)
    if e_match != place_e:
        raise RuntimeError(f"{role_prefix} did not visually localize E")
    return signature_a, signature, query_views, {
        "executions": executions,
        "a_position_index": a_position_index,
        "e_position_index": len(session.audit.positions) - 1,
        "e_match": e_match,
    }


def _execute_return_homing(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    role_prefix: str,
    encoder: Any,
    modality: str,
    memory_path: Path,
    period: int,
    temporal_samples: int,
    target_id: str,
    query_views: tuple[tuple[int, ...], ...],
    program: VisualHomingProgram,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    alignment = align_panorama_heading(query_views, query_views)
    _orient(session, alignment.right_turn_steps)
    _orient(session, program.turn_steps)
    targets = _movement_targets(session.world)
    if program.motion_slot >= len(targets):
        raise RuntimeError("persisted return motion slot is unavailable")
    steps = []
    final_signature: tuple[int, ...] | None = None
    success = False
    for repetition in range(1, program.max_repetitions + 1):
        _move(
            session,
            moves=1,
            move_frames=program.move_frames,
            target_id=targets[program.motion_slot],
        )
        role = f"{role_prefix}-return-step-{repetition:02d}"
        signature, _views = _scan(
            session,
            output,
            scans,
            role,
            encoder=encoder,
            period=period,
            temporal_samples=temporal_samples,
        )
        match = _recognize(memory_path, modality, signature)
        steps.append(
            {
                "repetition": repetition,
                "scan_role": role,
                "recognized_place_id": match,
                "evaluator_position_index": len(session.audit.positions) - 1,
            }
        )
        final_signature = signature
        if repetition >= program.min_repetitions and match == target_id:
            success = True
            break
    if final_signature is None:
        raise AssertionError("return homing budget did not execute")
    return {
        "action_roles": list(program.to_roles()),
        "alignment": alignment.model_dump(mode="json"),
        "source_scan_role": f"{role_prefix}-f",
        "reference_scan_role": f"{role_prefix}-f",
        "steps": steps,
        "used_repetitions": len(steps),
        "success": success,
    }, final_signature


def _probe_goal_frontier(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    role_prefix: str,
    encoder: Any,
    modality: str,
    memory_path: Path,
    period: int,
    temporal_samples: int,
    place_ids: tuple[str, str, str],
    reference_views: tuple[tuple[tuple[int, ...], ...], ...],
    route_roles: tuple[tuple[str, ...], tuple[str, ...]],
    candidate_roles: tuple[str, ...],
    return_program: ReturnProgram,
    attempt_return: bool,
) -> dict[str, Any]:
    signature_a, signature_e, views_e, replay = _replay_to_e(
        session,
        output=output,
        scans=scans,
        role_prefix=role_prefix,
        encoder=encoder,
        modality=modality,
        memory_path=memory_path,
        period=period,
        temporal_samples=temporal_samples,
        place_ids=place_ids,
        reference_views=reference_views,
        route_roles=route_roles,
    )
    outbound = _execute_visual_program(
        session,
        roles=candidate_roles,
        period=period,
        reference_views=reference_views[2],
        query_views=views_e,
        source_scan_role=f"{role_prefix}-e",
        reference_scan_role="e-reference",
    )
    signature_f, views_f = _scan(
        session,
        output,
        scans,
        f"{role_prefix}-f",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    f_position_index = len(session.audit.positions) - 1
    outbound_match = _recognize(memory_path, modality, signature_f)
    if not attempt_return:
        signature_return = signature_f
        return_match = outbound_match
        returned = {
            "action_roles": None,
            "alignment": None,
            "source_scan_role": f"{role_prefix}-f",
            "reference_scan_role": None,
            "steps": [],
            "used_repetitions": 0,
            "success": False,
            "skipped_for_directed_frontier": True,
        }
    elif isinstance(return_program, VisualFrontierProgram):
        returned = _execute_visual_program(
            session,
            roles=return_program.to_roles(),
            period=period,
            reference_views=views_f,
            query_views=views_f,
            source_scan_role=f"{role_prefix}-f",
            reference_scan_role=f"{role_prefix}-f",
        )
        signature_return, _views_return = _scan(
            session,
            output,
            scans,
            f"{role_prefix}-return",
            encoder=encoder,
            period=period,
            temporal_samples=temporal_samples,
        )
        return_match = _recognize(memory_path, modality, signature_return)
        returned.update(
            {
                "steps": [
                    {
                        "repetition": return_program.repetitions,
                        "scan_role": f"{role_prefix}-return",
                        "recognized_place_id": return_match,
                        "evaluator_position_index": len(session.audit.positions) - 1,
                    }
                ],
                "used_repetitions": return_program.repetitions,
                "success": return_match == place_ids[2],
            }
        )
    else:
        returned, signature_return = _execute_return_homing(
            session,
            output=output,
            scans=scans,
            role_prefix=role_prefix,
            encoder=encoder,
            modality=modality,
            memory_path=memory_path,
            period=period,
            temporal_samples=temporal_samples,
            target_id=place_ids[2],
            query_views=views_f,
            program=return_program,
        )
    return_position_index = len(session.audit.positions) - 1
    return_match = _recognize(memory_path, modality, signature_return)
    positions = session.audit.positions
    a_index = int(replay["a_position_index"])
    e_index = int(replay["e_position_index"])
    for step in returned["steps"]:
        step["evaluator_return_error_3d"] = _distance_3d(
            positions, e_index, int(step["evaluator_position_index"])
        )
    return {
        "candidate_action_roles": list(candidate_roles),
        "return_action_roles": (
            list(return_program.to_roles()) if attempt_return else None
        ),
        "replay": replay,
        "outbound_execution": outbound,
        "return_execution": returned,
        "a_scan_role": f"{role_prefix}-a",
        "e_scan_role": f"{role_prefix}-e",
        "f_scan_role": f"{role_prefix}-f",
        "return_scan_role": (
            returned["steps"][-1]["scan_role"] if attempt_return else None
        ),
        "outbound_signature": list(signature_f),
        "outbound_match": outbound_match,
        "return_match": return_match,
        "visual_return_recognized": returned["success"],
        "directed_frontier": not attempt_return,
        "goal_visual_distance": math.dist(signature_a, signature_f),
        "source_visual_distance": math.dist(signature_e, signature_f),
        "return_visual_distance": math.dist(signature_e, signature_return),
        "evaluator": {
            "a_position_index": a_index,
            "e_position_index": e_index,
            "f_position_index": f_position_index,
            "return_position_index": return_position_index,
            "source_goal_distance_3d": _distance_3d(positions, e_index, a_index),
            "frontier_goal_distance_3d": _distance_3d(
                positions, f_position_index, a_index
            ),
            "outbound_distance_3d": _distance_3d(
                positions, e_index, f_position_index
            ),
            "return_error_3d": _distance_3d(
                positions, e_index, return_position_index
            ),
            "maximum_vertical_drop": _vertical_drop(
                positions, e_index, return_position_index
            ),
        },
    }


def _persist_frontier(
    memory_path: Path,
    *,
    modality: str,
    place_a: str,
    place_e: str,
    discovery: dict[str, Any],
    validation: dict[str, Any],
    forward_roles: tuple[str, ...],
    reverse_roles: tuple[str, ...] | None,
    directed: bool,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    direct_goal = discovery["outbound_match"] == validation["outbound_match"] == place_a
    with ExplorationMemory.open(memory_path) as memory:
        if direct_goal:
            target_id = place_a
        else:
            target = memory.observe_place(
                (len(memory.places()),), episode_id=30, step=0, uncertainty=0.25
            )
            target_id = target.place_id
            discovery_signature = tuple(discovery["outbound_signature"])
            validation_signature = tuple(validation["outbound_signature"])
            negatives = tuple(
                state.exemplars[0]
                for state in memory.modality_places(modality)
                if state.place_id != target_id
            )
            memory.record_modality_view(
                modality,
                target_id,
                discovery_signature,
                episode_id=30,
                step=1,
            )
            state = memory.calibrate_modality_place(
                modality,
                target_id,
                (discovery_signature, validation_signature),
                negatives,
                episode_id=30,
                step=2,
                margin=None,
            )
            memory.consolidate_modality_view(
                modality,
                target_id,
                validation_signature,
                episode_id=30,
                step=3,
            )
        frontier_ids = {
            "outbound": _record_resolved_frontier(
                memory,
                place_e,
                target_id,
                forward_roles,
                episode_id=30,
                step=4,
                cost=1.0,
            )
        }
        if not directed:
            assert reverse_roles is not None
            frontier_ids["return"] = _record_resolved_frontier(
                memory,
                target_id,
                place_e,
                reverse_roles,
                episode_id=30,
                step=6,
                cost=1.0,
            )
        outward_plan = memory.plan_route(place_e, target_id)
        return_plan = memory.plan_route(target_id, place_e) if not directed else None
        goal_plan = memory.plan_route(target_id, place_a)
        summary = memory.summary()
    if outward_plan is None or (not directed and return_plan is None):
        raise RuntimeError("validated goal frontier is unavailable to the planner")
    return target_id, frontier_ids, {
        "direct_goal_revisit": direct_goal,
        "calibration": (
            None if direct_goal else state.calibration.model_dump(mode="json")
        ),
        "outward": outward_plan.model_dump(mode="json"),
        "return": (
            return_plan.model_dump(mode="json") if return_plan is not None else None
        ),
        "goal": goal_plan.model_dump(mode="json") if goal_plan is not None else None,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--moves", type=int, default=3)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--max-return-steps", type=int, default=5)
    parser.add_argument("--return-move-frames", type=int, default=1)
    parser.add_argument("--fixed-return-turn-steps", type=int)
    parser.add_argument("--allow-directed-frontier", action="store_true")
    parser.add_argument("--temporal-samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_count < 1:
        parser.error("candidate count must be positive")
    if args.moves < 1 or args.move_frames < 1:
        parser.error("movement budgets must be positive")
    if args.max_return_steps < 1 or args.return_move_frames < 1:
        parser.error("return homing budgets must be positive")
    if args.temporal_samples < 1 or args.temporal_samples % 2 == 0:
        parser.error("temporal samples must be a positive odd integer")

    source = args.source_run.resolve()
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    source_audit = json.loads(
        (source / "integrity-audit.json").read_text(encoding="utf-8")
    )
    if (
        source_report.get("suite") != "natural_visual_detour"
        or source_report.get("backend") != "real-minestudio"
        or source_report.get("terrain_modified") is not False
        or source_report.get("gate", {}).get("passed") is not True
        or source_audit.get("valid") is not True
    ):
        parser.error("source run is not an accepted audited natural visual detour")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = _copy_source_evidence(source, args.output)
    source_code_hashes = _copy_sources(args.output)
    evidence = args.output / "source-evidence"
    memory_path = args.output / "exploration.sqlite3"
    shutil.copy2(evidence / "exploration.sqlite3", memory_path)

    seed = int(source_report["seed"])
    period = int(source_report["inferred_camera_period"])
    encoder = PANORAMA_ENCODERS["edge"]()
    modality = encoder.VERSION
    place_a = str(source_report["place_ids"]["a"])
    place_b = str(source_report["place_ids"]["b"])
    place_e = str(source_report["place_ids"]["e"])
    with ExplorationMemory.open(memory_path) as memory:
        route = memory.plan_route(place_a, place_e)
    if route is None or route.place_path != (place_a, place_b, place_e):
        raise RuntimeError("source memory does not contain the accepted A->B->E route")
    route_roles = tuple(route.travel_action_roles)
    if len(route_roles) != 2:
        raise RuntimeError("goal-frontier replay requires exactly two known edges")
    references = tuple(
        _load_views(evidence, role, encoder=encoder, period=period)
        for role in ("a-reference", "b-reference", "e-reference")
    )
    selected_slot = int(source_report["forward_action_roles"][2].split(":", 1)[1])
    excluded_heading = period // 2
    openness = _scan_openness(evidence, "e-reference", period)
    ranked_headings = _candidate_headings(openness, period)
    candidates = tuple(
        heading for heading in ranked_headings if heading != excluded_heading
    )[: args.candidate_count]
    candidate_roles = {
        heading: _program(
            turn_steps=heading,
            motion_slot=selected_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
        for heading in candidates
    }
    if args.fixed_return_turn_steps is not None:
        if not 0 <= args.fixed_return_turn_steps < period:
            parser.error("fixed return turn must be inside the camera period")
        return_program: ReturnProgram = VisualFrontierProgram(
            turn_steps=args.fixed_return_turn_steps,
            motion_slot=selected_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
    else:
        return_program = VisualHomingProgram(
            turn_steps=period // 2,
            motion_slot=selected_slot,
            min_repetitions=1,
            max_repetitions=args.max_return_steps,
            move_frames=args.return_move_frames,
            settle_ticks=0,
        )
    condition = source_report["condition"]
    commands = natural_world_commands(
        time_of_day=str(condition["time"]), weather=str(condition["weather"])
    )
    movement_programs = (("forward",), ("forward", "jump"))
    sessions: list[Session] = []
    scans: dict[str, Any] = {}
    try:
        results = []
        for probe_index, heading in enumerate(candidates):
            session = _open_session(seed, commands, movement_programs)
            sessions.append(session)
            result = _probe_goal_frontier(
                session,
                output=args.output,
                scans=scans,
                role_prefix=f"probe-{probe_index:02d}-h{heading:02d}",
                encoder=encoder,
                modality=modality,
                memory_path=memory_path,
                period=period,
                temporal_samples=args.temporal_samples,
                place_ids=(place_a, place_b, place_e),
                reference_views=references,
                route_roles=route_roles,
                candidate_roles=candidate_roles[heading],
                return_program=return_program,
                attempt_return=not args.allow_directed_frontier,
            )
            result["heading_steps"] = heading
            result["openness"] = openness[heading]
            result["session_index"] = len(sessions) - 1
            results.append(result)
            session.world.close()
        eligible = [
            item
            for item in results
            if item["outbound_match"] in (None, place_a)
            and (
                args.allow_directed_frontier
                or item["visual_return_recognized"]
            )
        ]
        if not eligible:
            failure = {
                "selector": (
                    "rgb-goal-directed-frontier-v1"
                    if args.allow_directed_frontier
                    else "rgb-goal-distance-with-reversibility-v1"
                ),
                "candidate_headings": list(candidates),
                "results": results,
                "reason": "no visually reversible novel/goal frontier from E",
            }
            (args.output / "failed-discovery.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_events(args.output / "events.jsonl", sessions)
            raise RuntimeError(failure["reason"])
        selected = min(
            eligible,
            key=lambda item: (
                float(item["goal_visual_distance"]),
                -float(item["source_visual_distance"]),
                int(item["heading_steps"]),
            ),
        )
        selected_heading = int(selected["heading_steps"])

        validation_session = _open_session(seed, commands, movement_programs)
        sessions.append(validation_session)
        validation = _probe_goal_frontier(
            validation_session,
            output=args.output,
            scans=scans,
            role_prefix="validation",
            encoder=encoder,
            modality=modality,
            memory_path=memory_path,
            period=period,
            temporal_samples=args.temporal_samples,
            place_ids=(place_a, place_b, place_e),
            reference_views=references,
            route_roles=route_roles,
            candidate_roles=candidate_roles[selected_heading],
            return_program=return_program,
            attempt_return=not args.allow_directed_frontier,
        )
        validation["heading_steps"] = selected_heading
        validation["openness"] = openness[selected_heading]
        validation_session.world.close()

        validation_visual = bool(validation["visual_return_recognized"])
        validation_physical = (
            float(validation["evaluator"]["return_error_3d"])
            <= RETURN_TOLERANCE_3D
        )
        validation_progress = (
            float(validation["evaluator"]["frontier_goal_distance_3d"])
            < float(validation["evaluator"]["source_goal_distance_3d"])
        )
        validation_consistent = (
            selected["outbound_match"] == validation["outbound_match"]
        )
        return_valid = (
            args.allow_directed_frontier
            or (validation_visual and validation_physical)
        )
        promoted = return_valid and validation_progress and validation_consistent
        target_id = None
        frontier_ids: dict[str, str] = {}
        persisted = None
        if promoted:
            target_id, frontier_ids, persisted = _persist_frontier(
                memory_path,
                modality=modality,
                place_a=place_a,
                place_e=place_e,
                discovery=selected,
                validation=validation,
                forward_roles=candidate_roles[selected_heading],
                reverse_roles=(
                    tuple(validation["return_action_roles"])
                    if validation["return_action_roles"] is not None
                    else None
                ),
                directed=args.allow_directed_frontier,
            )

        _write_events(args.output / "events.jsonl", sessions)
        public_text = (args.output / "events.jsonl").read_text(encoding="utf-8")
        records = [record for session in sessions for record in session.records]
        gate = {
            "source_run_accepted": True,
            "candidate_selection_rgb_only": True,
            "selected_candidate_novel_or_goal": selected["outbound_match"]
            in (None, place_a),
            "independent_validation_consistent": validation_consistent,
            "selected_frontier_physically_traversed": float(
                selected["evaluator"]["outbound_distance_3d"]
            )
            >= MINIMUM_FRONTIER_DISTANCE_3D,
            "validation_frontier_physically_traversed": float(
                validation["evaluator"]["outbound_distance_3d"]
            )
            >= MINIMUM_FRONTIER_DISTANCE_3D,
            "selected_frontier_reduces_goal_distance": float(
                selected["evaluator"]["frontier_goal_distance_3d"]
            )
            < float(selected["evaluator"]["source_goal_distance_3d"]),
            "validation_frontier_reduces_goal_distance": validation_progress,
            "frontier_promoted_after_validation": bool(
                promoted
                and target_id
                and len(frontier_ids)
                == (1 if args.allow_directed_frontier else 2)
            ),
            "persistent_outbound_plan_available": bool(
                persisted
                and persisted["outward"]["place_path"] == [place_e, target_id]
            ),
            "memory_integrity": bool(
                persisted and persisted["summary"]["integrity_valid"] is True
            ),
            "public_telemetry_clean": not any(
                token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
            ),
            "actions_compliant": bool(records)
            and all(
                record.intervention_compliance and record.event.success
                for record in records
            ),
        }
        if args.allow_directed_frontier:
            gate["directed_frontier_declared"] = bool(
                persisted and persisted["return"] is None
            )
        else:
            gate.update(
                {
                    "selected_candidate_visually_reversible": bool(
                        selected["visual_return_recognized"]
                    ),
                    "independent_validation_visually_reversible": validation_visual,
                    "selected_return_physically_close": float(
                        selected["evaluator"]["return_error_3d"]
                    )
                    <= RETURN_TOLERANCE_3D,
                    "validation_return_physically_close": validation_physical,
                    "persistent_return_plan_available": bool(
                        persisted
                        and persisted["return"]["place_path"]
                        == [target_id, place_e]
                    ),
                }
            )
        report = {
            "suite": "natural_visual_goal_frontier",
            "backend": "real-minestudio",
            "seed": seed,
            "terrain_modified": False,
            "encoder": encoder.VERSION,
            "inferred_camera_period": period,
            "temporal_samples": args.temporal_samples,
            "condition": condition,
            "place_ids": {
                "a": place_a,
                "b": place_b,
                "e": place_e,
                "target": target_id,
            },
            "source_route": route.model_dump(mode="json"),
            "selector": (
                "rgb-goal-directed-frontier-v1"
                if args.allow_directed_frontier
                else "rgb-goal-distance-with-reversibility-v1"
            ),
            "directed_frontier": args.allow_directed_frontier,
            "candidate_headings": list(candidates),
            "excluded_known_return_heading": excluded_heading,
            "openness_scores": list(openness),
            "candidate_results": results,
            "selected_heading_steps": selected_heading,
            "selected_result": selected,
            "validation_result": validation,
            "forward_action_roles": list(candidate_roles[selected_heading]),
            "return_action_roles": (
                None
                if args.allow_directed_frontier
                else list(return_program.to_roles())
            ),
            "frontier_ids": frontier_ids,
            "persisted_frontier": persisted,
            "return_tolerance_3d": RETURN_TOLERANCE_3D,
            "minimum_frontier_distance_3d": MINIMUM_FRONTIER_DISTANCE_3D,
            "scans": scans,
            "source_evidence_sha256": source_hashes,
            "source_sha256": source_code_hashes,
            "session_startup_attempts": [session.startup_attempts for session in sessions],
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Untouched natural seed; goal-directed RGB frontier selection from "
                "persistent E, isolated candidate sessions, fresh-session shadow "
                "validation, and evaluator-only physical progress veto. This expands "
                "one local route toward A; it does not prove unrestricted exploration."
            ),
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
            "selected_session_index": int(selected["session_index"]),
            "validation_session_index": len(sessions) - 1,
        }
        config = {
            "source_run_sha256": _digest(evidence / "report.json"),
            "candidate_count": args.candidate_count,
            "moves": args.moves,
            "move_frames": args.move_frames,
            "max_return_steps": args.max_return_steps,
            "return_move_frames": args.return_move_frames,
            "fixed_return_turn_steps": args.fixed_return_turn_steps,
            "allow_directed_frontier": args.allow_directed_frontier,
            "temporal_samples": args.temporal_samples,
        }
        for name, payload in (
            ("config.json", config),
            ("evaluator-audit.json", evaluator),
            ("report.json", report),
        ):
            (args.output / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["gate"]["passed"]:
            raise SystemExit(1)
    finally:
        for session in sessions:
            session.world.close()


if __name__ == "__main__":
    main()
