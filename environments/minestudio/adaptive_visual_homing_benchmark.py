"""Fail-closed RGB homing on an untouched natural MineStudio world.

This benchmark consumes an independently audited natural visual expedition.  In
two fresh simulator sessions it replays the persisted A-B-C-D outward route, then
validates a configured D-C-B-A reverse prefix. Every reverse edge is executed one
anonymous motion unit at a
time and stops only when the target's frozen calibrated RGB boundary accepts a
fresh panorama.  A budget exhaustion is a failure; no boundary is widened and no
reverse edge is inferred from action symmetry.  Coordinates remain evaluator-only.
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
from natural_visual_expedition_benchmark import (
    _execute_visual_program,
    _record_resolved_frontier,
    _scan,
    _segment_metrics,
)
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    Session,
    _move,
    _movement_targets,
    _open_session,
    _orient,
    _position_metrics,
    _write_events,
    natural_world_commands,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBEdgeCompactSpectrumPanoramaEncoder,
    RGBEdgeDualBandSpectrumPanoramaEncoder,
    RGBEdgeExpandedSpectrumPanoramaEncoder,
    RGBEdgeSpectrumPanoramaEncoder,
    VisualEvidenceHomingProgram,
    VisualFrontierProgram,
    VisualHomingProgram,
    align_panorama_heading,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction

REFERENCE_SCAN_ROLES = ("train-a1", "train-b1", "train-c1", "heldout-d1")
SOURCE_SCAN_ROLES = (
    *REFERENCE_SCAN_ROLES,
    "calibration-a",
    "calibration-b",
    "train-c2",
    "calibration-c",
    "heldout-d2",
)
HOMING_ENCODERS = {
    "spectrum-r4": RGBEdgeSpectrumPanoramaEncoder,
    "spectrum-r3": RGBEdgeCompactSpectrumPanoramaEncoder,
    "spectrum-r1-r6": RGBEdgeDualBandSpectrumPanoramaEncoder,
    "spectrum-r5": RGBEdgeExpandedSpectrumPanoramaEncoder,
}
ARRIVAL_TOLERANCE_3D = 0.75
HomingProgram = VisualHomingProgram | VisualEvidenceHomingProgram


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_source_evidence(source: Path, output: Path) -> dict[str, str]:
    target = output / "source-evidence"
    target.mkdir()
    for name in ("report.json", "integrity-audit.json", "exploration.sqlite3"):
        shutil.copy2(source / name, target / name)
    for role in SOURCE_SCAN_ROLES:
        destination = target / "scans" / role
        destination.mkdir(parents=True)
        for path in sorted((source / "scans" / role).glob("view-*.png")):
            shutil.copy2(path, destination / path.name)
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
        "adaptive_visual_homing_benchmark.py": Path(__file__).resolve(),
        "audit_adaptive_visual_homing.py": Path(__file__).with_name(
            "audit_adaptive_visual_homing.py"
        ),
        "natural_visual_expedition_benchmark.py": Path(__file__).with_name(
            "natural_visual_expedition_benchmark.py"
        ),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "exploration_init.py": repository
        / "src/minecraft_scientist/exploration/__init__.py",
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


def _load_reference_views(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[tuple[int, ...], ...]:
    frames = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return tuple(encoder.scene_encoder.encode(frame) for frame in frames)


def _source_signature(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[int, ...]:
    return encoder.encode_signatures(
        _load_reference_views(root, role, encoder=encoder, period=period)
    )


def _install_homing_modality(
    memory_path: Path,
    evidence: Path,
    *,
    encoder: Any,
    period: int,
    place_ids: tuple[str, ...],
) -> dict[str, Any]:
    roles = {
        place_ids[0]: ("train-a1", "calibration-a"),
        place_ids[1]: ("train-b1", "calibration-b"),
        place_ids[2]: ("train-c1", "train-c2", "calibration-c"),
        place_ids[3]: ("heldout-d1", "heldout-d2"),
    }
    signatures = {
        role: _source_signature(evidence, role, encoder=encoder, period=period)
        for place_roles in roles.values()
        for role in place_roles
    }
    calibration = {}
    with ExplorationMemory.open(memory_path) as memory:
        if memory.modality_places(encoder.VERSION):
            raise RuntimeError("homing modality already exists in source memory")
        for index, (place_id, positive_roles) in enumerate(roles.items()):
            positives = tuple(signatures[role] for role in positive_roles)
            negatives = tuple(
                signatures[role]
                for other_id, other_roles in roles.items()
                if other_id != place_id
                for role in other_roles
            )
            memory.record_modality_view(
                encoder.VERSION,
                place_id,
                positives[0],
                episode_id=3,
                step=index * 10,
            )
            state = memory.calibrate_modality_place(
                encoder.VERSION,
                place_id,
                positives,
                negatives,
                episode_id=3,
                step=index * 10 + 1,
                margin=None,
            )
            for offset, signature in enumerate(positives[1:], start=2):
                memory.consolidate_modality_view(
                    encoder.VERSION,
                    place_id,
                    signature,
                    episode_id=3,
                    step=index * 10 + offset,
                )
            calibration[place_id] = {
                "positive_roles": list(positive_roles),
                "negative_roles": [
                    role
                    for other_id, other_roles in roles.items()
                    if other_id != place_id
                    for role in other_roles
                ],
                "boundary": (
                    state.calibration.model_dump(mode="json")
                    if state.calibration is not None
                    else None
                ),
            }
    return calibration


def _recognition(
    memory_path: Path,
    modality: str,
    target_id: str,
    signature: tuple[int, ...],
) -> tuple[str | None, dict[str, Any]]:
    with ExplorationMemory.open(memory_path) as memory:
        match = memory.match_modality_place(modality, signature)
        score = memory.score_modality_place(modality, target_id, signature)
    return match, score.model_dump(mode="json")


def _parse_homing_edge_spec(raw: str) -> HomingProgram:
    """Parse one preregistered memoryless or evidence-window edge program."""

    base_fields = (
        "turn_steps",
        "motion_slot",
        "min_repetitions",
        "max_repetitions",
        "move_frames",
        "settle_ticks",
    )
    evidence_fields = (*base_fields, "evidence_window", "required_matches")
    parts = tuple(part.strip() for part in raw.split(","))
    if len(parts) not in (len(base_fields), len(evidence_fields)) or any(
        not part.isdecimal() for part in parts
    ):
        raise ValueError(
            "homing edge spec must be "
            "turn,motion,min,max,frames,settle[,evidence_window,required_matches]"
        )
    fields = evidence_fields if len(parts) == len(evidence_fields) else base_fields
    model = (
        VisualEvidenceHomingProgram
        if len(parts) == len(evidence_fields)
        else VisualHomingProgram
    )
    return model.model_validate(
        dict(zip(fields, (int(part) for part in parts), strict=True))
    )


def _position_distance_3d(
    positions: list[dict[str, float]], left_index: int, right_index: int
) -> float:
    left = positions[left_index]
    right = positions[right_index]
    return math.sqrt(
        (float(left["xpos"]) - float(right["xpos"])) ** 2
        + (float(left["ypos"]) - float(right["ypos"])) ** 2
        + (float(left["zpos"]) - float(right["zpos"])) ** 2
    )


def _annotate_arrival_errors(
    session: Session, result: dict[str, Any], place_ids: tuple[str, ...]
) -> None:
    """Add evaluator-only physical errors after policy execution has completed."""

    target_indices = dict(
        zip(place_ids, result["outward_position_indices"], strict=True)
    )
    for edge in result["homing_edges"]:
        edge["evaluator_metrics"]["target_distance_3d"] = _position_distance_3d(
            session.audit.positions,
            int(edge["evaluator_position_index"]),
            int(target_indices[edge["target_place_id"]]),
        )


def _execute_homing_edge(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    memory_path: Path,
    modality: str,
    encoder: Any,
    homing_modality: str,
    homing_encoder: Any,
    period: int,
    temporal_samples: int,
    source_id: str,
    target_id: str,
    source_reference_role: str,
    source_reference_views: tuple[tuple[int, ...], ...],
    query_role: str,
    query_views: tuple[tuple[int, ...], ...],
    program: HomingProgram,
    phase: str,
    edge_index: int,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[tuple[int, ...], ...], str]:
    if program.turn_steps >= period:
        raise ValueError("visual homing turn slot exceeds the learned period")
    alignment = align_panorama_heading(source_reference_views, query_views)
    _orient(session, alignment.right_turn_steps)
    _orient(session, program.turn_steps)
    targets = _movement_targets(session.world)
    if program.motion_slot >= len(targets):
        raise RuntimeError("persisted anonymous homing motion slot is unavailable")
    start_index = len(session.audit.positions)
    steps = []
    final_signature: tuple[int, ...] | None = None
    final_views: tuple[tuple[int, ...], ...] | None = None
    final_role = query_role
    success = False
    target_evidence: list[bool] = []
    for repetition in range(1, program.max_repetitions + 1):
        _move(
            session,
            moves=1,
            move_frames=program.move_frames,
            target_id=targets[program.motion_slot],
        )
        if program.settle_ticks:
            session.records.append(
                session.world.step(
                    PrimitiveAction(kind=ActionKind.WAIT, ticks=program.settle_ticks)
                )
            )
        role = f"{phase}-home-{edge_index}-step-{repetition:02d}"
        signature, views = _scan(
            session,
            output,
            scans,
            role,
            encoder=encoder,
            period=period,
            temporal_samples=temporal_samples,
        )
        homing_signature = _source_signature(
            output, role, encoder=homing_encoder, period=period
        )
        match, score = _recognition(
            memory_path, homing_modality, target_id, homing_signature
        )
        target_evidence.append(match == target_id)
        steps.append(
            {
                "repetition": repetition,
                "scan_role": role,
                "recognized_place_id": match,
                "target_score": score,
                "acceptance_eligible": repetition >= program.min_repetitions,
                "evidence_match_count": program.evidence_match_count(
                    target_evidence
                ),
                "evaluator_position_index": len(session.audit.positions) - 1,
            }
        )
        final_signature, final_views, final_role = signature, views, role
        if program.should_stop(target_evidence):
            success = True
            break
    if final_signature is None or final_views is None:
        raise AssertionError("homing budget did not execute")
    result = {
        "source_place_id": source_id,
        "target_place_id": target_id,
        "source_scan_role": query_role,
        "reference_scan_role": source_reference_role,
        "homing_encoder": homing_encoder.VERSION,
        "action_roles": list(program.to_roles()),
        "alignment": alignment.model_dump(mode="json"),
        "steps": steps,
        "used_repetitions": len(steps),
        "success": success,
        "evaluator_position_index": len(session.audit.positions) - 1,
        "evaluator_metrics": _segment_metrics(session.audit.positions, start_index),
    }
    return result, final_signature, final_views, final_role


def _run_round_trip(
    *,
    phase: str,
    seed: int,
    commands: tuple[str, ...],
    movement_programs: tuple[tuple[str, ...], ...],
    output: Path,
    scans: dict[str, Any],
    memory_path: Path,
    modality: str,
    encoder: Any,
    homing_encoders: tuple[Any, ...],
    period: int,
    temporal_samples: int,
    place_ids: tuple[str, ...],
    reference_roles: dict[str, str],
    reference_views: dict[str, tuple[tuple[int, ...], ...]],
    outward_roles: tuple[tuple[str, ...], ...],
    homing_programs: tuple[HomingProgram, ...],
    homing_edge_count: int,
) -> tuple[Session, dict[str, Any]]:
    session = _open_session(seed, commands, movement_programs)
    signature, route_views = _scan(
        session,
        output,
        scans,
        f"{phase}-a",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    start_match, _score = _recognition(
        memory_path, modality, place_ids[0], signature
    )
    outward_matches = [start_match]
    outward_position_indices = [len(session.audit.positions) - 1]
    outward_executions = []
    route_role = f"{phase}-a"
    for edge_index, ((source_id, target_id), roles) in enumerate(
        zip(pairwise(place_ids), outward_roles, strict=True), start=1
    ):
        execution = _execute_visual_program(
            session,
            roles=roles,
            period=period,
            reference_views=reference_views[source_id],
            query_views=route_views,
            source_scan_role=route_role,
            reference_scan_role=reference_roles[source_id],
        )
        outward_executions.append(execution)
        route_role = f"{phase}-out-{edge_index}"
        signature, route_views = _scan(
            session,
            output,
            scans,
            route_role,
            encoder=encoder,
            period=period,
            temporal_samples=temporal_samples,
        )
        match, _score = _recognition(memory_path, modality, target_id, signature)
        outward_matches.append(match)
        outward_position_indices.append(len(session.audit.positions) - 1)
    if outward_matches[-1] != place_ids[-1]:
        raise RuntimeError(f"{phase} did not visually localize the homing source")

    homing_matches = [outward_matches[-1]]
    homing_edges = []
    reverse_pairs = tuple(pairwise(reversed(place_ids)))[:homing_edge_count]
    for edge_index, (
        (source_id, target_id),
        homing_program,
        homing_encoder,
    ) in enumerate(
        zip(reverse_pairs, homing_programs, homing_encoders, strict=True), start=1
    ):
        homing_modality = homing_encoder.VERSION
        result, signature, route_views, route_role = _execute_homing_edge(
            session,
            output=output,
            scans=scans,
            memory_path=memory_path,
            modality=modality,
            encoder=encoder,
            homing_modality=homing_modality,
            homing_encoder=homing_encoder,
            period=period,
            temporal_samples=temporal_samples,
            source_id=source_id,
            target_id=target_id,
            source_reference_role=reference_roles[source_id],
            source_reference_views=reference_views[source_id],
            query_role=route_role,
            query_views=route_views,
            program=homing_program,
            phase=phase,
            edge_index=edge_index,
        )
        homing_edges.append(result)
        homing_signature = _source_signature(
            output, route_role, encoder=homing_encoder, period=period
        )
        terminal_match, _score = _recognition(
            memory_path, homing_modality, target_id, homing_signature
        )
        match = target_id if result["success"] else terminal_match
        homing_matches.append(match)
        if not result["success"]:
            break
    return session, {
        "outward_matches": outward_matches,
        "outward_position_indices": outward_position_indices,
        "homing_matches": homing_matches,
        "outward_executions": outward_executions,
        "homing_edges": homing_edges,
        "evaluator_metrics": _position_metrics(session.audit.positions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--max-homing-steps", type=int, default=8)
    parser.add_argument("--min-homing-steps", type=int, default=1)
    parser.add_argument(
        "--outward-edge-count",
        type=int,
        default=3,
        help="number of accepted outward edges to replay before homing",
    )
    parser.add_argument("--homing-edge-count", type=int, default=1)
    parser.add_argument(
        "--homing-turn-steps",
        type=int,
        help="override the default learned half-turn for a preregistered probe",
    )
    parser.add_argument(
        "--homing-motion-slot",
        type=int,
        help="override the outward route's anonymous motion slot",
    )
    parser.add_argument(
        "--homing-move-frames",
        type=int,
        help="override the outward route's motion duration for finer visual servoing",
    )
    parser.add_argument("--homing-settle-ticks", type=int, default=10)
    parser.add_argument(
        "--homing-encoder",
        choices=tuple(HOMING_ENCODERS),
        default="spectrum-r4",
        help="versioned arrival-recognition modality",
    )
    parser.add_argument(
        "--homing-edge-encoder",
        action="append",
        choices=tuple(HOMING_ENCODERS),
        default=[],
        help=(
            "repeat once per requested reverse edge to preregister a distinct "
            "versioned arrival modality"
        ),
    )
    parser.add_argument(
        "--homing-edge-spec",
        action="append",
        default=[],
        metavar="TURN,MOTION,MIN,MAX,FRAMES,SETTLE[,WINDOW,REQUIRED]",
        help=(
            "repeat once per requested reverse edge to preregister an edge-specific "
            "closed-loop program; when omitted, the global homing options are reused"
        ),
    )
    parser.add_argument("--temporal-samples", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_homing_steps < 1:
        parser.error("homing budget must be positive")
    if not 1 <= args.min_homing_steps <= args.max_homing_steps:
        parser.error("minimum homing steps must fit the homing budget")
    if not 1 <= args.outward_edge_count <= 3:
        parser.error("outward edge count must be between one and three")
    if not 1 <= args.homing_edge_count <= args.outward_edge_count:
        parser.error("homing edge count must fit the selected outward prefix")
    if args.temporal_samples < 1 or args.temporal_samples % 2 == 0:
        parser.error("temporal samples must be a positive odd integer")
    source = args.source_run.resolve()
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    source_audit = json.loads(
        (source / "integrity-audit.json").read_text(encoding="utf-8")
    )
    if (
        source_report.get("suite") != "natural_visual_expedition"
        or source_report.get("backend") != "real-minestudio"
        or source_report.get("terrain_modified") is not False
        or source_report.get("gate", {}).get("passed") is not True
        or source_audit.get("valid") is not True
    ):
        parser.error("source run is not an accepted audited natural expedition")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = _copy_source_evidence(source, args.output)
    source_code_hashes = _copy_sources(args.output)
    evidence = args.output / "source-evidence"
    memory_path = args.output / "exploration.sqlite3"
    shutil.copy2(evidence / "exploration.sqlite3", memory_path)

    seed = int(source_report["seed"])
    period = int(source_report["inferred_camera_period"])
    if args.homing_turn_steps is not None and not (
        0 <= args.homing_turn_steps < period
    ):
        parser.error("homing turn steps must be inside the learned camera period")
    if args.homing_motion_slot is not None and args.homing_motion_slot < 0:
        parser.error("homing motion slot must be nonnegative")
    if args.homing_move_frames is not None and args.homing_move_frames < 1:
        parser.error("homing move frames must be positive")
    if args.homing_settle_ticks < 0:
        parser.error("homing settle ticks must be nonnegative")
    if not str(source_report["encoder"]).startswith("rgb-edge"):
        parser.error("adaptive homing currently requires edge source evidence")
    encoder = PANORAMA_ENCODERS["edge"]()
    modality = encoder.VERSION
    if args.homing_edge_encoder:
        if len(args.homing_edge_encoder) != args.homing_edge_count:
            parser.error(
                "--homing-edge-encoder must be repeated exactly "
                "--homing-edge-count times"
            )
        homing_encoder_keys = tuple(args.homing_edge_encoder)
    else:
        homing_encoder_keys = (args.homing_encoder,) * args.homing_edge_count
    homing_encoders = tuple(HOMING_ENCODERS[key]() for key in homing_encoder_keys)
    homing_encoder = homing_encoders[0]
    place_ids_by_role = source_report["place_ids"]
    place_ids = tuple(place_ids_by_role[key] for key in "abcd")
    route_place_ids = place_ids[: args.outward_edge_count + 1]
    unique_homing_encoders = {
        encoder.VERSION: encoder for encoder in homing_encoders
    }
    homing_calibrations = {
        version: _install_homing_modality(
            memory_path,
            evidence,
            encoder=edge_encoder,
            period=period,
            place_ids=place_ids,
        )
        for version, edge_encoder in unique_homing_encoders.items()
    }
    homing_calibration = homing_calibrations[homing_encoder.VERSION]
    reference_roles = dict(zip(place_ids, REFERENCE_SCAN_ROLES, strict=True))
    reference_views = {
        place_id: _load_reference_views(
            evidence, role, encoder=encoder, period=period
        )
        for place_id, role in reference_roles.items()
    }
    with ExplorationMemory.open(memory_path) as memory:
        outward_plan = memory.plan_route(place_ids[0], place_ids[-1])
    if outward_plan is None or outward_plan.place_path != place_ids:
        raise RuntimeError("source memory does not contain the accepted outward route")
    route_outward_roles = outward_plan.travel_action_roles[: args.outward_edge_count]

    selected_slot = int(source_report["selected_motion_slot"])
    move_frames = VisualFrontierProgram.from_roles(
        outward_plan.travel_action_roles[0]
    ).move_frames
    homing_program = VisualHomingProgram(
        turn_steps=(
            period // 2
            if args.homing_turn_steps is None
            else args.homing_turn_steps
        ),
        motion_slot=(
            selected_slot
            if args.homing_motion_slot is None
            else args.homing_motion_slot
        ),
        min_repetitions=args.min_homing_steps,
        max_repetitions=args.max_homing_steps,
        move_frames=(
            move_frames
            if args.homing_move_frames is None
            else args.homing_move_frames
        ),
        settle_ticks=args.homing_settle_ticks,
    )
    if args.homing_edge_spec:
        try:
            homing_programs = tuple(
                _parse_homing_edge_spec(raw) for raw in args.homing_edge_spec
            )
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        if len(homing_programs) != args.homing_edge_count:
            parser.error(
                "--homing-edge-spec must be repeated exactly --homing-edge-count times"
            )
    else:
        homing_programs = (homing_program,) * args.homing_edge_count
    if any(program.turn_steps >= period for program in homing_programs):
        parser.error("homing edge turn steps must be inside the learned camera period")
    homing_program = homing_programs[0]
    movement_programs = (("forward",), ("forward", "jump"))
    heldout = source_report["conditions"]["heldout"]
    commands = natural_world_commands(
        time_of_day=str(heldout["time"]), weather=str(heldout["weather"])
    )
    scans: dict[str, Any] = {}
    sessions: list[Session] = []
    try:
        discovery, discovery_result = _run_round_trip(
            phase="discovery",
            seed=seed,
            commands=commands,
            movement_programs=movement_programs,
            output=args.output,
            scans=scans,
            memory_path=memory_path,
            modality=modality,
            encoder=encoder,
            homing_encoders=homing_encoders,
            period=period,
            temporal_samples=args.temporal_samples,
            place_ids=route_place_ids,
            reference_roles=reference_roles,
            reference_views=reference_views,
            outward_roles=route_outward_roles,
            homing_programs=homing_programs,
            homing_edge_count=args.homing_edge_count,
        )
        sessions.append(discovery)
        _annotate_arrival_errors(discovery, discovery_result, route_place_ids)
        expected_home = list(reversed(route_place_ids))[
            : args.homing_edge_count + 1
        ]
        if discovery_result["homing_matches"] != expected_home:
            (args.output / "failed-discovery.json").write_text(
                json.dumps(
                    {
                        "result": discovery_result,
                        "evaluator_positions": discovery.audit.positions,
                        "raw_metadata_keys": sorted(discovery.audit.metadata_keys),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_events(args.output / "events.jsonl", sessions)
            raise RuntimeError("discovery homing did not validate its reverse prefix")
        discovery.world.close()

        validation, validation_result = _run_round_trip(
            phase="validation",
            seed=seed,
            commands=commands,
            movement_programs=movement_programs,
            output=args.output,
            scans=scans,
            memory_path=memory_path,
            modality=modality,
            encoder=encoder,
            homing_encoders=homing_encoders,
            period=period,
            temporal_samples=args.temporal_samples,
            place_ids=route_place_ids,
            reference_roles=reference_roles,
            reference_views=reference_views,
            outward_roles=route_outward_roles,
            homing_programs=homing_programs,
            homing_edge_count=args.homing_edge_count,
        )
        sessions.append(validation)
        _annotate_arrival_errors(validation, validation_result, route_place_ids)
        validation_success = validation_result["homing_matches"] == expected_home
        reverse_frontier_ids = {}
        if validation_success:
            with ExplorationMemory.open(memory_path) as memory:
                reverse_pairs = tuple(pairwise(reversed(route_place_ids)))[
                    : args.homing_edge_count
                ]
                for edge_index, ((source_id, target_id), edge_program) in enumerate(
                    zip(reverse_pairs, homing_programs, strict=True), start=1
                ):
                    reverse_frontier_ids[f"{source_id}->{target_id}"] = (
                        _record_resolved_frontier(
                            memory,
                            source_id,
                            target_id,
                            edge_program.to_roles(),
                            episode_id=4,
                            step=edge_index * 2,
                            cost=float(
                                validation_result["homing_edges"][edge_index - 1][
                                    "used_repetitions"
                                ]
                            ),
                        )
                    )
        validation.world.close()

        with ExplorationMemory.open(memory_path) as memory:
            return_target = route_place_ids[-1 - args.homing_edge_count]
            return_plan = memory.plan_route(route_place_ids[-1], return_target)
            memory_summary = memory.summary()
            memory_events = tuple(memory.iter_events())
        (args.output / "memory-events.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in memory_events)
            + "\n",
            encoding="utf-8",
        )
        _write_events(args.output / "events.jsonl", sessions)
        public_text = "\n".join(
            (args.output / name).read_text(encoding="utf-8")
            for name in ("events.jsonl", "memory-events.jsonl")
        )
        action_records = [record for session in sessions for record in session.records]
        actions_compliant = bool(action_records) and all(
            record.intervention_compliance and record.event.success
            for record in action_records
        )
        expected_outward = list(route_place_ids)
        return_path = tuple(return_plan.place_path) if return_plan is not None else ()
        gate = {
            "source_run_accepted": True,
            "discovery_outward_route_recognized": discovery_result["outward_matches"]
            == expected_outward,
            "discovery_homing_recognized": discovery_result["homing_matches"]
            == expected_home,
            "validation_outward_route_recognized": validation_result[
                "outward_matches"
            ]
            == expected_outward,
            "independent_validation_homing_recognized": validation_result[
                "homing_matches"
            ]
            == expected_home,
            "closed_loop_stopped_on_match_within_budget": all(
                int(edge["used_repetitions"]) <= program.max_repetitions
                for result in (discovery_result, validation_result)
                for edge, program in zip(
                    result["homing_edges"], homing_programs, strict=True
                )
            ),
            "physically_remote_round_trips": all(
                result["evaluator_metrics"]["maximum_excursion"] is not None
                and result["evaluator_metrics"]["maximum_excursion"]
                >= (8.0 if args.outward_edge_count == 3 else 2.0)
                for result in (discovery_result, validation_result)
            ),
            "reverse_edges_physically_traversed": all(
                edge["evaluator_metrics"]["maximum_excursion"] is not None
                and edge["evaluator_metrics"]["maximum_excursion"] >= 2.0
                for result in (discovery_result, validation_result)
                for edge in result["homing_edges"]
            ),
            "reverse_edges_reach_targets": all(
                edge["evaluator_metrics"]["target_distance_3d"]
                <= ARRIVAL_TOLERANCE_3D
                for result in (discovery_result, validation_result)
                for edge in result["homing_edges"]
            ),
            "reverse_edges_promoted_after_validation": len(reverse_frontier_ids)
            == args.homing_edge_count,
            "persistent_return_plan_available": return_path == tuple(expected_home),
            "memory_integrity": memory_summary["integrity_valid"] is True,
            "public_telemetry_clean": not any(
                token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
            ),
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
            "discovery_session_index": 0,
            "validation_session_index": 1,
            "discovery_metrics": discovery_result["evaluator_metrics"],
            "validation_metrics": validation_result["evaluator_metrics"],
        }
        report = {
            "suite": "adaptive_visual_homing",
            "backend": "real-minestudio",
            "seed": seed,
            "terrain_modified": False,
            "encoder": encoder.VERSION,
            "homing_encoder": homing_encoder.VERSION,
            "homing_encoders": [encoder.VERSION for encoder in homing_encoders],
            "homing_edge_count": args.homing_edge_count,
            "outward_edge_count": args.outward_edge_count,
            "arrival_tolerance_3d": ARRIVAL_TOLERANCE_3D,
            "inferred_camera_period": period,
            "temporal_samples": args.temporal_samples,
            "condition": heldout,
            "place_ids": place_ids_by_role,
            "route_place_ids": list(route_place_ids),
            "homing_program": homing_program.model_dump(mode="json"),
            "homing_action_roles": list(homing_program.to_roles()),
            "homing_programs": [
                program.model_dump(mode="json") for program in homing_programs
            ],
            "homing_action_roles_by_edge": [
                list(program.to_roles()) for program in homing_programs
            ],
            "source_reference_roles": reference_roles,
            "homing_calibration": homing_calibration,
            "homing_calibrations": homing_calibrations,
            "outward_plan": outward_plan.model_dump(mode="json"),
            "discovery": discovery_result,
            "validation": validation_result,
            "reverse_frontier_ids": reverse_frontier_ids,
            "return_plan": (
                return_plan.model_dump(mode="json") if return_plan is not None else None
            ),
            "memory": memory_summary,
            "scans": scans,
            "source_evidence_sha256": source_hashes,
            "source_sha256": source_code_hashes,
            "session_startup_attempts": [
                session.startup_attempts for session in sessions
            ],
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Untouched natural seed; RGB-only adaptive stopping on a configured "
                "reverse prefix, persistent validated return topology, and fresh-session shadow "
                "validation. Evaluator-only coordinates prove excursion and return. "
                "Only the requested reverse prefix is promoted; unvalidated reverse "
                "edges remain absent. This is bounded route homing, not cross-biome "
                "SLAM, survival, or "
                "unrestricted open-world exploration."
            ),
        }
        config = {
            "source_run_sha256": _digest(evidence / "report.json"),
            "max_homing_steps": args.max_homing_steps,
            "min_homing_steps": args.min_homing_steps,
            "homing_edge_count": args.homing_edge_count,
            "outward_edge_count": args.outward_edge_count,
            "homing_turn_steps": homing_program.turn_steps,
            "homing_motion_slot": homing_program.motion_slot,
            "homing_move_frames": homing_program.move_frames,
            "homing_settle_ticks": homing_program.settle_ticks,
            "homing_edge_specs": [
                program.model_dump(mode="json") for program in homing_programs
            ],
            "homing_encoder": args.homing_encoder,
            "homing_edge_encoders": list(homing_encoder_keys),
            "temporal_samples": args.temporal_samples,
        }
        (args.output / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "evaluator-audit.json").write_text(
            json.dumps(evaluator, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
