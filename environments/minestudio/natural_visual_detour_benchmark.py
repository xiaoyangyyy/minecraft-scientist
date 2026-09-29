"""Discover and validate a reversible RGB-only detour from a known place.

The benchmark consumes an accepted natural visual expedition, replays its
persisted A->B edge in fresh MineStudio sessions, and probes several visually
open headings from B.  Candidate selection uses only RGB novelty and visual
return recognition.  Simulator coordinates are captured by a separate audit
channel and are consulted only after policy execution.  A B<->E detour is
persisted only after the selected program succeeds again in a fresh session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from adaptive_visual_homing_benchmark import (
    _copy_source_evidence,
    _install_homing_modality,
    _load_reference_views,
    _source_signature,
)
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
    _open_session,
    _position_metrics,
    _write_events,
    natural_world_commands,
)

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBEdgeDualBandSpectrumPanoramaEncoder,
)

RETURN_TOLERANCE_3D = 0.75
RANKED_RECOGNITION_MARGIN = 0.5


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_sources(output: Path) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    sources = {
        "natural_visual_detour_benchmark.py": Path(__file__).resolve(),
        "audit_natural_visual_detour.py": Path(__file__).with_name(
            "audit_natural_visual_detour.py"
        ),
        "adaptive_visual_homing_benchmark.py": Path(__file__).with_name(
            "adaptive_visual_homing_benchmark.py"
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


def _distance_3d(
    positions: list[dict[str, float]], left_index: int, right_index: int
) -> float:
    left = positions[left_index]
    right = positions[right_index]
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


def _recognize(
    memory_path: Path, modality: str, signature: tuple[int, ...]
) -> tuple[str | None, dict[str, Any]]:
    with ExplorationMemory.open(memory_path) as memory:
        strict_match = memory.match_modality_place(modality, signature)
        scores = tuple(
            memory.score_modality_place(modality, state.place_id, signature)
            for state in memory.modality_places(modality)
        )
    ranked = sorted(scores, key=lambda item: (item.nearest_distance, item.place_id))
    if not ranked:
        return None, {"strict_match": strict_match, "ranked": []}
    separation = (
        ranked[1].nearest_distance - ranked[0].nearest_distance
        if len(ranked) > 1
        else math.inf
    )
    ranked_match = (
        ranked[0].place_id
        if separation >= RANKED_RECOGNITION_MARGIN
        else None
    )
    return strict_match or ranked_match, {
        "strict_match": strict_match,
        "ranked_match": ranked_match,
        "nearest_separation": separation,
        "required_separation": RANKED_RECOGNITION_MARGIN,
        "ranked": [item.model_dump(mode="json") for item in ranked],
    }


def _replay_to_b(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    role_prefix: str,
    encoder: Any,
    recognition_encoder: Any,
    memory_path: Path,
    period: int,
    temporal_samples: int,
    place_a: str,
    place_b: str,
    views_a: tuple[tuple[int, ...], ...],
    roles_ab: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], dict[str, Any]]:
    _sig_a, query_views = _scan(
        session,
        output,
        scans,
        f"{role_prefix}-a",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    execution = _execute_visual_program(
        session,
        roles=roles_ab,
        period=period,
        reference_views=views_a,
        query_views=query_views,
        source_scan_role=f"{role_prefix}-a",
        reference_scan_role="train-a1",
    )
    signature, views = _scan(
        session,
        output,
        scans,
        f"{role_prefix}-b",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    recognition_signature = _source_signature(
        output,
        f"{role_prefix}-b",
        encoder=recognition_encoder,
        period=period,
    )
    match, recognition = _recognize(
        memory_path, recognition_encoder.VERSION, recognition_signature
    )
    if match != place_b:
        raise RuntimeError(f"{role_prefix} did not visually localize B")
    execution["source_place_id"] = place_a
    execution["target_place_id"] = place_b
    execution["recognized_place_id"] = match
    execution["recognition"] = recognition
    return signature, views, execution


def _probe_detour(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    role_prefix: str,
    encoder: Any,
    recognition_encoder: Any,
    memory_path: Path,
    period: int,
    temporal_samples: int,
    place_a: str,
    place_b: str,
    views_a: tuple[tuple[int, ...], ...],
    roles_ab: tuple[str, ...],
    candidate_roles: tuple[str, ...],
    reverse_roles: tuple[str, ...],
) -> dict[str, Any]:
    signature_b, views_b, replay = _replay_to_b(
        session,
        output=output,
        scans=scans,
        role_prefix=role_prefix,
        encoder=encoder,
        recognition_encoder=recognition_encoder,
        memory_path=memory_path,
        period=period,
        temporal_samples=temporal_samples,
        place_a=place_a,
        place_b=place_b,
        views_a=views_a,
        roles_ab=roles_ab,
    )
    b_position_index = len(session.audit.positions) - 1
    outbound = _execute_visual_program(
        session,
        roles=candidate_roles,
        period=period,
        reference_views=views_b,
        query_views=views_b,
        source_scan_role=f"{role_prefix}-b",
        reference_scan_role=f"{role_prefix}-b",
    )
    signature_e, views_e = _scan(
        session,
        output,
        scans,
        f"{role_prefix}-e",
        encoder=encoder,
        period=period,
        temporal_samples=temporal_samples,
    )
    e_position_index = len(session.audit.positions) - 1
    returned = _execute_visual_program(
        session,
        roles=reverse_roles,
        period=period,
        reference_views=views_e,
        query_views=views_e,
        source_scan_role=f"{role_prefix}-e",
        reference_scan_role=f"{role_prefix}-e",
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
    return_position_index = len(session.audit.positions) - 1
    return_recognition_signature = _source_signature(
        output,
        f"{role_prefix}-return",
        encoder=recognition_encoder,
        period=period,
    )
    return_match, return_recognition = _recognize(
        memory_path, recognition_encoder.VERSION, return_recognition_signature
    )
    outbound_distance = math.dist(signature_b, signature_e)
    return_distance = math.dist(signature_b, signature_return)
    return {
        "candidate_action_roles": list(candidate_roles),
        "reverse_action_roles": list(reverse_roles),
        "replay_execution": replay,
        "outbound_execution": outbound,
        "return_execution": returned,
        "b_scan_role": f"{role_prefix}-b",
        "e_scan_role": f"{role_prefix}-e",
        "return_scan_role": f"{role_prefix}-return",
        "outbound_signature": list(signature_e),
        "return_match": return_match,
        "return_recognition": return_recognition,
        "visual_return_recognized": return_match == place_b,
        "outbound_distance": outbound_distance,
        "return_distance": return_distance,
        "reversibility_score": outbound_distance / (return_distance + 1.0),
        "evaluator": {
            "b_position_index": b_position_index,
            "e_position_index": e_position_index,
            "return_position_index": return_position_index,
            "outbound_distance_3d": _distance_3d(
                session.audit.positions, b_position_index, e_position_index
            ),
            "return_error_3d": _distance_3d(
                session.audit.positions, b_position_index, return_position_index
            ),
            "maximum_vertical_drop": _vertical_drop(
                session.audit.positions, b_position_index, return_position_index
            ),
            "session_metrics": _position_metrics(session.audit.positions),
        },
    }


def _persist_detour(
    memory_path: Path,
    *,
    modality: str,
    place_b: str,
    discovery: dict[str, Any],
    validation: dict[str, Any],
    forward_roles: tuple[str, ...],
    reverse_roles: tuple[str, ...],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    discovery_signature = tuple(int(item) for item in discovery["outbound_signature"])
    validation_signature = tuple(int(item) for item in validation["outbound_signature"])
    with ExplorationMemory.open(memory_path) as memory:
        existing_count = len(memory.places())
        place_e = memory.observe_place(
            (existing_count,), episode_id=20, step=0, uncertainty=0.25
        )
        negative_signatures = tuple(
            state.exemplars[0]
            for state in memory.modality_places(modality)
            if state.place_id != place_e.place_id
        )
        memory.record_modality_view(
            modality,
            place_e.place_id,
            discovery_signature,
            episode_id=20,
            step=1,
        )
        state = memory.calibrate_modality_place(
            modality,
            place_e.place_id,
            (discovery_signature, validation_signature),
            negative_signatures,
            episode_id=20,
            step=2,
            margin=None,
        )
        memory.consolidate_modality_view(
            modality,
            place_e.place_id,
            validation_signature,
            episode_id=20,
            step=3,
        )
        frontier_ids = {
            "b_to_e": _record_resolved_frontier(
                memory,
                place_b,
                place_e.place_id,
                forward_roles,
                episode_id=20,
                step=4,
                cost=1.0,
            ),
            "e_to_b": _record_resolved_frontier(
                memory,
                place_e.place_id,
                place_b,
                reverse_roles,
                episode_id=20,
                step=6,
                cost=1.0,
            ),
        }
        route_forward = memory.plan_route(place_b, place_e.place_id)
        route_reverse = memory.plan_route(place_e.place_id, place_b)
        summary = memory.summary()
    if route_forward is None or route_reverse is None:
        raise RuntimeError("validated detour was not available to the route planner")
    return place_e.place_id, frontier_ids, {
        "forward": route_forward.model_dump(mode="json"),
        "reverse": route_reverse.model_dump(mode="json"),
        "calibration": state.calibration.model_dump(mode="json"),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--moves", type=int, default=3)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--temporal-samples", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_count < 1:
        parser.error("candidate count must be positive")
    if args.moves < 1 or args.move_frames < 1:
        parser.error("movement budgets must be positive")
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
    if args.candidate_count > period:
        parser.error("candidate count cannot exceed the learned camera period")
    encoder = PANORAMA_ENCODERS["edge"]()
    modality = encoder.VERSION
    recognition_encoder = RGBEdgeDualBandSpectrumPanoramaEncoder()
    place_ids = tuple(source_report["place_ids"][key] for key in "abcd")
    place_a, place_b = place_ids[:2]
    views_a = _load_reference_views(
        evidence, "train-a1", encoder=encoder, period=period
    )
    with ExplorationMemory.open(memory_path) as memory:
        route_ab = memory.plan_route(place_a, place_b)
    if route_ab is None or route_ab.place_path != (place_a, place_b):
        raise RuntimeError("source memory does not contain the accepted A->B edge")
    recognition_calibration = _install_homing_modality(
        memory_path,
        evidence,
        encoder=recognition_encoder,
        period=period,
        place_ids=place_ids,
    )
    roles_ab = route_ab.travel_action_roles[0]
    selected_slot = int(source_report["selected_motion_slot"])
    candidate_roles_by_heading = {
        heading: _program(
            turn_steps=heading,
            motion_slot=selected_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        )
        for heading in range(period)
    }
    reverse_roles = _program(
        turn_steps=period // 2,
        motion_slot=selected_slot,
        repetitions=args.moves,
        move_frames=args.move_frames,
    )
    source_b_signature = _source_signature(
        evidence, "train-b1", encoder=encoder, period=period
    )
    openness = _scan_openness(evidence, "train-b1", period)
    candidates = _candidate_headings(openness, args.candidate_count)
    heldout = source_report["conditions"]["heldout"]
    commands = natural_world_commands(
        time_of_day=str(heldout["time"]), weather=str(heldout["weather"])
    )
    movement_programs = (("forward",), ("forward", "jump"))
    sessions: list[Session] = []
    scans: dict[str, Any] = {}
    try:
        results = []
        for probe_index, heading in enumerate(candidates):
            session = _open_session(seed, commands, movement_programs)
            sessions.append(session)
            result = _probe_detour(
                session,
                output=args.output,
                scans=scans,
                role_prefix=f"probe-{probe_index:02d}-h{heading:02d}",
                encoder=encoder,
                recognition_encoder=recognition_encoder,
                memory_path=memory_path,
                period=period,
                temporal_samples=args.temporal_samples,
                place_a=place_a,
                place_b=place_b,
                views_a=views_a,
                roles_ab=roles_ab,
                candidate_roles=candidate_roles_by_heading[heading],
                reverse_roles=reverse_roles,
            )
            result["heading_steps"] = heading
            result["openness"] = openness[heading]
            result["session_index"] = len(sessions) - 1
            results.append(result)
            session.world.close()

        visually_reversible = [item for item in results if item["visual_return_recognized"]]
        if not visually_reversible:
            failure = {
                "selector": "rgb-novelty-reversibility-v1",
                "candidate_headings": list(candidates),
                "results": results,
                "reason": "no candidate visually returned to B",
            }
            (args.output / "failed-discovery.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_events(args.output / "events.jsonl", sessions)
            raise RuntimeError("no visually reversible B frontier was discovered")
        selected = max(
            visually_reversible,
            key=lambda item: (
                float(item["reversibility_score"]),
                float(item["outbound_distance"]),
                -int(item["heading_steps"]),
            ),
        )
        selected_heading = int(selected["heading_steps"])

        validation_session = _open_session(seed, commands, movement_programs)
        sessions.append(validation_session)
        validation = _probe_detour(
            validation_session,
            output=args.output,
            scans=scans,
            role_prefix="validation",
            encoder=encoder,
            recognition_encoder=recognition_encoder,
            memory_path=memory_path,
            period=period,
            temporal_samples=args.temporal_samples,
            place_a=place_a,
            place_b=place_b,
            views_a=views_a,
            roles_ab=roles_ab,
            candidate_roles=candidate_roles_by_heading[selected_heading],
            reverse_roles=reverse_roles,
        )
        validation["heading_steps"] = selected_heading
        validation["openness"] = openness[selected_heading]
        validation_session.world.close()

        validation_visual_succeeded = bool(validation["visual_return_recognized"])
        validation_audit_succeeded = (
            float(validation["evaluator"]["return_error_3d"])
            <= RETURN_TOLERANCE_3D
        )
        validation_succeeded = (
            validation_visual_succeeded and validation_audit_succeeded
        )
        place_e = None
        frontier_ids: dict[str, str] = {}
        persisted: dict[str, Any] | None = None
        if validation_succeeded:
            place_e, frontier_ids, persisted = _persist_detour(
                memory_path,
                modality=modality,
                place_b=place_b,
                discovery=selected,
                validation=validation,
                forward_roles=candidate_roles_by_heading[selected_heading],
                reverse_roles=reverse_roles,
            )

        _write_events(args.output / "events.jsonl", sessions)
        public_text = (args.output / "events.jsonl").read_text(encoding="utf-8")
        action_records = [record for session in sessions for record in session.records]
        gate = {
            "source_run_accepted": True,
            "candidate_selection_rgb_only": True,
            "selected_candidate_visually_reversible": bool(
                selected["visual_return_recognized"]
            ),
            "independent_validation_visually_reversible": (
                validation_visual_succeeded
            ),
            "selected_frontier_is_novel": float(selected["outbound_distance"]) > 0,
            "validation_frontier_is_novel": float(validation["outbound_distance"]) > 0,
            "selected_return_physically_close": float(
                selected["evaluator"]["return_error_3d"]
            )
            <= RETURN_TOLERANCE_3D,
            "validation_return_physically_close": validation_audit_succeeded,
            "selected_frontier_physically_traversed": float(
                selected["evaluator"]["outbound_distance_3d"]
            )
            >= 1.0,
            "validation_frontier_physically_traversed": float(
                validation["evaluator"]["outbound_distance_3d"]
            )
            >= 1.0,
            "detour_promoted_after_validation": bool(place_e and len(frontier_ids) == 2),
            "persistent_bidirectional_plan_available": bool(
                persisted
                and persisted["forward"]["place_path"] == [place_b, place_e]
                and persisted["reverse"]["place_path"] == [place_e, place_b]
            ),
            "memory_integrity": bool(
                persisted and persisted["summary"]["integrity_valid"] is True
            ),
            "public_telemetry_clean": not any(
                token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
            ),
            "actions_compliant": bool(action_records)
            and all(
                record.intervention_compliance and record.event.success
                for record in action_records
            ),
        }
        report = {
            "suite": "natural_visual_detour",
            "backend": "real-minestudio",
            "seed": seed,
            "terrain_modified": False,
            "encoder": encoder.VERSION,
            "recognition_encoder": recognition_encoder.VERSION,
            "recognition_calibration": recognition_calibration,
            "inferred_camera_period": period,
            "condition": heldout,
            "place_ids": {**source_report["place_ids"], "e": place_e},
            "source_place_id": place_b,
            "source_b_signature_distance": math.dist(
                source_b_signature,
                _source_signature(
                    args.output,
                    str(selected["b_scan_role"]),
                    encoder=encoder,
                    period=period,
                ),
            ),
            "selector": "rgb-novelty-reversibility-v1",
            "candidate_headings": list(candidates),
            "openness_scores": list(openness),
            "candidate_results": results,
            "selected_heading_steps": selected_heading,
            "selected_result": selected,
            "validation_result": validation,
            "forward_action_roles": list(candidate_roles_by_heading[selected_heading]),
            "reverse_action_roles": list(reverse_roles),
            "frontier_ids": frontier_ids,
            "persisted_detour": persisted,
            "scans": scans,
            "return_tolerance_3d": RETURN_TOLERANCE_3D,
            "ranked_recognition_margin": RANKED_RECOGNITION_MARGIN,
            "source_evidence_sha256": source_hashes,
            "source_sha256": source_code_hashes,
            "session_startup_attempts": [session.startup_attempts for session in sessions],
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Untouched natural seed; RGB-only frontier selection at B, fresh-session "
                "shadow validation, and validation-gated persistent B<->E topology. "
                "Evaluator coordinates are posthoc only. This discovers a locally "
                "reversible detour edge; it does not yet prove E->A connectivity."
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
