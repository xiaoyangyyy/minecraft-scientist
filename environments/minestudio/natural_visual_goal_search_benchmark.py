"""Continue an audited visual frontier using any persisted directed route.

Every probe and shadow validation runs in a fresh natural MineStudio world.
Policy inputs are RGB signatures, opaque place identities, and functional action
slots. Evaluator coordinates can veto the already-selected edge but cannot rank
probes. A final independent restart must recognize the tentatively learned node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_expedition_benchmark import (
    _execute_visual_program,
    _record_resolved_frontier,
    _scan,
)
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    Session,
    _candidate_headings,
    _open_session,
    _write_events,
    natural_world_commands,
    view_openness,
)
from PIL import Image

from minecraft_scientist.exploration import ExplorationMemory, VisualFrontierProgram
from minecraft_scientist.exploration.goal_search import (
    VisualGoalCandidate,
    rank_visual_goal_candidates,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _source_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    return {
        name: here / name
        for name in (
            "natural_visual_goal_search_benchmark.py",
            "audit_natural_visual_goal_search.py",
            "natural_visual_expedition_benchmark.py",
            "natural_visual_loop_benchmark.py",
            "navigation_probe.py",
        )
    } | {
        name: repository / "src/minecraft_scientist" / relative
        for name, relative in {
            "goal_search.py": "exploration/goal_search.py",
            "frontier_program.py": "exploration/frontier_program.py",
            "memory.py": "exploration/memory.py",
            "perception.py": "exploration/perception.py",
            "minestudio.py": "minecraft/minestudio.py",
        }.items()
    }


def _copy_views(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source.glob("view-*.png")):
        shutil.copy2(path, destination / path.name)


def _prepare_source(source: Path, output: Path) -> dict[str, Any]:
    report = _load(source / "report.json")
    audit = _load(source / "integrity-audit.json")
    if not (
        report.get("suite")
        in ("natural_visual_goal_frontier", "natural_visual_goal_search")
        and report.get("backend") == "real-minestudio"
        and report.get("terrain_modified") is False
        and report.get("gate", {}).get("passed") is True
        and audit.get("valid") is True
        and all(audit.get("checks", {}).values())
    ):
        raise ValueError(
            "source must be an accepted independently audited visual goal run"
        )
    evidence = output / "source-evidence"
    evidence.mkdir()
    for name in ("report.json", "integrity-audit.json", "exploration.sqlite3"):
        shutil.copy2(source / name, evidence / name)
    if report["suite"] == "natural_visual_goal_frontier":
        origin = str(report["place_ids"]["a"])
        references = {
            str(report["place_ids"][key]): source
            / "source-evidence/scans"
            / f"{key}-reference"
            for key in ("a", "b", "e")
        }
        references[str(report["place_ids"]["target"])] = (
            source / "scans" / report["selected_result"]["f_scan_role"]
        )
        goal_reference = source / "scans" / report["validation_result"]["a_scan_role"]
        goal = origin
    else:
        origin = str(report["origin_place_id"])
        goal = str(report["goal_place_id"])
        references = {
            place: source / "references" / place
            for place in report["reference_place_ids"]
        }
        goal_reference = source / "goal-reference"
    for place_id, directory in references.items():
        _copy_views(directory, output / "references" / place_id)
    _copy_views(goal_reference, output / "goal-reference")
    shutil.copy2(evidence / "exploration.sqlite3", output / "exploration.sqlite3")
    with ExplorationMemory.open(output / "exploration.sqlite3") as memory:
        route = memory.plan_route(origin, str(report["place_ids"]["target"]))
        if route is None:
            raise ValueError("source target has no persisted route from origin")
        for roles in route.travel_action_roles:
            VisualFrontierProgram.from_roles(roles)
    return {
        "seed": int(report["seed"]),
        "period": int(report["inferred_camera_period"]),
        "encoder": report["encoder"],
        "condition": report["condition"],
        "origin": origin,
        "goal": goal,
        "source": str(report["place_ids"]["target"]),
        "route": route.model_dump(mode="json"),
        "reference_place_ids": sorted(references),
    }


def _panorama(
    directory: Path, *, encoder: Any, period: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    frames = []
    for index in range(period):
        with Image.open(directory / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return encoder.encode(frames), tuple(
        encoder.scene_encoder.encode(frame) for frame in frames
    )


def _replay(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    prefix: str,
    spec: dict[str, Any],
    encoder: Any,
    samples: int,
    route: dict[str, Any],
    memory_path: Path,
) -> dict[str, Any]:
    localizations = []
    executions = []
    signature = None
    query_views = None
    for index, place_id in enumerate(route["place_path"]):
        role = f"{prefix}-node-{index:02d}"
        signature, query_views = _scan(
            session,
            output,
            scans,
            role,
            encoder=encoder,
            period=spec["period"],
            temporal_samples=samples,
        )
        with ExplorationMemory.open(memory_path) as memory:
            match = memory.match_modality_place(spec["encoder"], signature)
        localizations.append(
            {
                "place_id": place_id,
                "recognized_place_id": match,
                "scan_role": role,
                "position_index": len(session.audit.positions) - 1,
            }
        )
        if match != place_id:
            raise RuntimeError(
                f"{prefix}: persisted route node {index} failed visual localization"
            )
        if index == len(route["place_path"]) - 1:
            break
        _signature, reference_views = _panorama(
            output / "references" / place_id, encoder=encoder, period=spec["period"]
        )
        executions.append(
            _execute_visual_program(
                session,
                roles=tuple(route["travel_action_roles"][index]),
                period=spec["period"],
                reference_views=reference_views,
                query_views=query_views,
                source_scan_role=role,
                reference_scan_role=f"reference:{place_id}",
            )
        )
    return {"localizations": localizations, "executions": executions}


def _probe(
    session: Session,
    *,
    output: Path,
    scans: dict[str, Any],
    prefix: str,
    spec: dict[str, Any],
    encoder: Any,
    samples: int,
    roles: tuple[str, ...],
    session_index: int,
) -> dict[str, Any]:
    replay = _replay(
        session,
        output=output,
        scans=scans,
        prefix=prefix,
        spec=spec,
        encoder=encoder,
        samples=samples,
        route=spec["route"],
        memory_path=output / "source-evidence/exploration.sqlite3",
    )
    source_role = replay["localizations"][-1]["scan_role"]
    source_signature, source_views = _panorama(
        output / "scans" / source_role, encoder=encoder, period=spec["period"]
    )
    _signature, reference = _panorama(
        output / "references" / spec["source"], encoder=encoder, period=spec["period"]
    )
    execution = _execute_visual_program(
        session,
        roles=roles,
        period=spec["period"],
        reference_views=reference,
        query_views=source_views,
        source_scan_role=source_role,
        reference_scan_role=f"reference:{spec['source']}",
    )
    target_role = f"{prefix}-target"
    signature, _views = _scan(
        session,
        output,
        scans,
        target_role,
        encoder=encoder,
        period=spec["period"],
        temporal_samples=samples,
    )
    with ExplorationMemory.open(
        output / "source-evidence/exploration.sqlite3"
    ) as memory:
        match = memory.match_modality_place(spec["encoder"], signature)
    return {
        "candidate_id": prefix,
        "heading_steps": VisualFrontierProgram.from_roles(roles).turn_steps,
        "signature": list(signature),
        "source_signature": list(source_signature),
        "recognized_place_id": match,
        "session_index": session_index,
        "replay": replay,
        "execution": execution,
        "target_scan_role": target_role,
        "target_position_index": len(session.audit.positions) - 1,
    }


def _physical(
    result: dict[str, Any],
    session: Session,
    *,
    goal_tolerance: float,
    maximum_drop: float,
    minimum_travel: float,
    goal: str,
) -> dict[str, Any]:
    positions = session.audit.positions
    origin_index = int(result["replay"]["localizations"][0]["position_index"])
    source_index = int(result["replay"]["localizations"][-1]["position_index"])
    target_index = int(result["target_position_index"])

    def distance(left_index: int, right_index: int) -> float:
        left, right = positions[left_index], positions[right_index]
        return math.dist(
            (left["xpos"], left["ypos"], left["zpos"]),
            (right["xpos"], right["ypos"], right["zpos"]),
        )

    source_goal = distance(source_index, origin_index)
    target_goal = distance(target_index, origin_index)
    travel = distance(source_index, target_index)
    drop = max(
        positions[source_index]["ypos"] - item["ypos"]
        for item in positions[source_index : target_index + 1]
    )
    return {
        "source_goal_distance_3d": source_goal,
        "target_goal_distance_3d": target_goal,
        "travel_distance_3d": travel,
        "maximum_vertical_drop": drop,
        "accepted": travel >= minimum_travel
        and drop <= maximum_drop
        and target_goal < source_goal
        and (result["recognized_place_id"] != goal or target_goal <= goal_tolerance),
    }


def _tentative_node(
    output: Path,
    spec: dict[str, Any],
    discovery: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    if (
        discovery["recognized_place_id"]
        == validation["recognized_place_id"]
        == spec["goal"]
    ):
        return str(spec["goal"])
    with ExplorationMemory.open(output / "exploration.sqlite3") as memory:
        target = memory.observe_place(
            (len(memory.places()),),
            episode_id=40 + len(memory.places()),
            step=0,
            uncertainty=0.25,
        )
        negatives = tuple(
            exemplar
            for state in memory.modality_places(spec["encoder"])
            for exemplar in state.exemplars
        )
        positive = (tuple(discovery["signature"]), tuple(validation["signature"]))
        memory.record_modality_view(
            spec["encoder"], target.place_id, positive[0], episode_id=40, step=1
        )
        memory.calibrate_modality_place(
            spec["encoder"],
            target.place_id,
            positive,
            negatives,
            episode_id=40,
            step=2,
            margin=None,
        )
        memory.consolidate_modality_view(
            spec["encoder"], target.place_id, positive[1], episode_id=40, step=3
        )
    _copy_views(
        output / "scans" / discovery["target_scan_role"],
        output / "references" / target.place_id,
    )
    return target.place_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--moves", type=int, default=3)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--temporal-samples", type=int, default=3)
    parser.add_argument("--maximum-vertical-drop", type=float, default=4.0)
    parser.add_argument("--minimum-travel", type=float, default=1.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.candidate_count, args.moves, args.move_frames) < 1:
        parser.error("probe budgets must be positive")
    if args.temporal_samples < 1 or args.temporal_samples % 2 == 0:
        parser.error("temporal sample count must be positive and odd")
    if not all(
        math.isfinite(item) and item > 0
        for item in (
            args.maximum_vertical_drop,
            args.minimum_travel,
            args.goal_tolerance,
        )
    ):
        parser.error("physical tolerances must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = _prepare_source(args.source_run.resolve(), args.output)
    config = {
        "candidate_count": args.candidate_count,
        "moves": args.moves,
        "move_frames": args.move_frames,
        "temporal_samples": args.temporal_samples,
        "maximum_vertical_drop": args.maximum_vertical_drop,
        "minimum_travel": args.minimum_travel,
        "goal_tolerance": args.goal_tolerance,
        "source_spec": spec,
    }
    _write(args.output / "config.json", config)
    source_dir = args.output / "source"
    source_dir.mkdir()
    for name, path in _source_paths().items():
        shutil.copy2(path, source_dir / name)
    frozen_sources = _hash_tree(source_dir)
    source_evidence = _hash_tree(args.output / "source-evidence")
    initial_references = _hash_tree(args.output / "references")
    goal_hashes = _hash_tree(args.output / "goal-reference")
    encoder = PANORAMA_ENCODERS["edge"]()
    if spec["encoder"] != encoder.VERSION:
        raise ValueError("unsupported source visual modality")
    goal_signature, _goal_views = _panorama(
        args.output / "goal-reference", encoder=encoder, period=spec["period"]
    )
    source_frames = []
    for index in range(spec["period"]):
        with Image.open(
            args.output / "references" / spec["source"] / f"view-{index:02d}.png"
        ) as image:
            source_frames.append(np.asarray(image.convert("RGB")))
    openness = tuple(view_openness(frame) for frame in source_frames)
    headings = _candidate_headings(openness, min(args.candidate_count, spec["period"]))
    motion_slot = VisualFrontierProgram.from_roles(
        tuple(spec["route"]["travel_action_roles"][-1])
    ).motion_slot
    programs = {
        heading: VisualFrontierProgram(
            turn_steps=heading,
            motion_slot=motion_slot,
            repetitions=args.moves,
            move_frames=args.move_frames,
        ).to_roles()
        for heading in headings
    }
    commands = natural_world_commands(
        time_of_day=spec["condition"]["time"], weather=spec["condition"]["weather"]
    )
    sessions: list[Session] = []
    scans: dict[str, Any] = {}
    report: dict[str, Any] = {
        "suite": "natural_visual_goal_search",
        "backend": "real-minestudio",
        "terrain_modified": False,
        "seed": spec["seed"],
        "condition": spec["condition"],
        "encoder": spec["encoder"],
        "inferred_camera_period": spec["period"],
        "origin_place_id": spec["origin"],
        "goal_place_id": spec["goal"],
        "place_ids": {"source": spec["source"], "target": None},
        "source_route": spec["route"],
        "candidate_headings": list(headings),
        "openness_scores": list(openness),
        "selector": "persisted-rgb-goal-v1",
        "candidate_results": [],
        "source_sha256": frozen_sources,
        "source_evidence_sha256": source_evidence,
        "initial_reference_sha256": initial_references,
        "goal_reference_sha256": goal_hashes,
    }
    evaluator: dict[str, Any] = {"probe_metrics": {}}
    gate: dict[str, bool] = {"source_run_accepted": True}
    try:
        for index, heading in enumerate(headings):
            prefix = f"probe-{index:02d}-h{heading:02d}"
            print(
                f"Probing {prefix} after {len(spec['route']['travel_action_roles'])} persisted edges",
                flush=True,
            )
            session = _open_session(
                spec["seed"], commands, (("forward",), ("forward", "jump"))
            )
            sessions.append(session)
            result = _probe(
                session,
                output=args.output,
                scans=scans,
                prefix=prefix,
                spec=spec,
                encoder=encoder,
                samples=args.temporal_samples,
                roles=programs[heading],
                session_index=len(sessions) - 1,
            )
            report["candidate_results"].append(result)
            session.world.close()
        inputs = [
            VisualGoalCandidate(
                **{
                    key: result[key]
                    for key in (
                        "candidate_id",
                        "heading_steps",
                        "signature",
                        "recognized_place_id",
                    )
                }
            )
            for result in report["candidate_results"]
        ]
        ranked = rank_visual_goal_candidates(
            goal_signature,
            {
                item["candidate_id"]: item["source_signature"]
                for item in report["candidate_results"]
            },
            inputs,
            goal_place_id=spec["goal"],
        )
        report["ranking"] = [item.model_dump(mode="json") for item in ranked]
        if not ranked:
            raise RuntimeError(
                "no novel or goal candidate; persisted memory was not changed"
            )
        winner = ranked[0].candidate.candidate_id
        selected = next(
            item
            for item in report["candidate_results"]
            if item["candidate_id"] == winner
        )
        report["selected_result"] = selected
        report["selected_heading_steps"] = selected["heading_steps"]
        for result in report["candidate_results"]:
            evaluator["probe_metrics"][result["candidate_id"]] = _physical(
                result,
                sessions[result["session_index"]],
                goal_tolerance=args.goal_tolerance,
                maximum_drop=args.maximum_vertical_drop,
                minimum_travel=args.minimum_travel,
                goal=spec["goal"],
            )
        gate["selected_physical_progress"] = evaluator["probe_metrics"][winner][
            "accepted"
        ]
        if not gate["selected_physical_progress"]:
            raise RuntimeError(
                "RGB-selected candidate failed physical veto; no runner-up substitution"
            )
        print(f"Shadow validating heading {selected['heading_steps']}", flush=True)
        session = _open_session(
            spec["seed"], commands, (("forward",), ("forward", "jump"))
        )
        sessions.append(session)
        validation = _probe(
            session,
            output=args.output,
            scans=scans,
            prefix="validation",
            spec=spec,
            encoder=encoder,
            samples=args.temporal_samples,
            roles=programs[selected["heading_steps"]],
            session_index=len(sessions) - 1,
        )
        report["validation_result"] = validation
        evaluator["validation_metrics"] = _physical(
            validation,
            session,
            goal_tolerance=args.goal_tolerance,
            maximum_drop=args.maximum_vertical_drop,
            minimum_travel=args.minimum_travel,
            goal=spec["goal"],
        )
        session.world.close()
        gate["validation_physical_progress"] = evaluator["validation_metrics"][
            "accepted"
        ]
        gate["independent_recognition_consistent"] = (
            validation["recognized_place_id"] == selected["recognized_place_id"]
        )
        if not all(gate.values()):
            raise RuntimeError("shadow validation did not confirm selected frontier")
        target = _tentative_node(args.output, spec, selected, validation)
        report["place_ids"]["target"] = target
        replay_route = {
            "place_path": [*spec["route"]["place_path"], target],
            "travel_action_roles": [
                *spec["route"]["travel_action_roles"],
                list(programs[selected["heading_steps"]]),
            ],
        }
        print("Testing learned node after independent process restart", flush=True)
        session = _open_session(
            spec["seed"], commands, (("forward",), ("forward", "jump"))
        )
        sessions.append(session)
        restart = _replay(
            session,
            output=args.output,
            scans=scans,
            prefix="restart",
            spec=spec,
            encoder=encoder,
            samples=args.temporal_samples,
            route=replay_route,
            memory_path=args.output / "exploration.sqlite3",
        )
        report["restart_result"] = {
            "replay": restart,
            "session_index": len(sessions) - 1,
        }
        session.world.close()
        gate["restart_target_recognized"] = (
            restart["localizations"][-1]["recognized_place_id"] == target
        )
        selected_position = sessions[selected["session_index"]].audit.positions[
            selected["target_position_index"]
        ]
        restart_position = session.audit.positions[
            restart["localizations"][-1]["position_index"]
        ]
        arrival_error = math.dist(
            tuple(selected_position[key] for key in ("xpos", "ypos", "zpos")),
            tuple(restart_position[key] for key in ("xpos", "ypos", "zpos")),
        )
        evaluator["restart_target_error_3d"] = arrival_error
        gate["restart_physical_arrival"] = arrival_error <= args.goal_tolerance
        if not all(gate.values()):
            raise RuntimeError(
                "independent restart did not confirm physical target arrival"
            )
        with ExplorationMemory.open(args.output / "exploration.sqlite3") as memory:
            frontier_id = _record_resolved_frontier(
                memory,
                spec["source"],
                target,
                programs[selected["heading_steps"]],
                episode_id=40,
                step=4,
                cost=1.0,
            )
            route = memory.plan_route(spec["origin"], target)
            goal_route = memory.plan_route(target, spec["goal"])
            report["frontier_id"] = frontier_id
            report["persisted_route"] = route.model_dump(mode="json") if route else None
            report["goal_route"] = (
                goal_route.model_dump(mode="json") if goal_route else None
            )
            report["memory_summary"] = memory.summary()
        gate["persistent_extended_route"] = route is not None and (
            target == spec["goal"]
            or list(route.place_path) == replay_route["place_path"]
        )
        gate["memory_integrity"] = report["memory_summary"]["integrity_valid"]
        report["goal_reached"] = target == spec["goal"]
    except Exception as error:  # noqa: BLE001 -- preserve failed simulator evidence, then exit nonzero
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        gate["workflow_completed"] = False
    finally:
        for session in sessions:
            session.world.close()
        _write_events(args.output / "events.jsonl", sessions)
        public_text = (args.output / "events.jsonl").read_text(encoding="utf-8")
        gate["public_telemetry_clean"] = not any(
            token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
        )
        records = [record for session in sessions for record in session.records]
        gate["actions_compliant"] = bool(records) and all(
            record.intervention_compliance and record.event.success
            for record in records
        )
        report["scans"] = scans
        report["reference_place_ids"] = sorted(
            path.name for path in (args.output / "references").iterdir()
        )
        report["reference_sha256"] = _hash_tree(args.output / "references")
        report["gate"] = {**gate, "passed": all(gate.values())}
        evaluator["sessions"] = [
            {
                "positions": session.audit.positions,
                "raw_metadata_keys": sorted(session.audit.metadata_keys),
                "startup_attempts": session.startup_attempts,
            }
            for session in sessions
        ]
        _write(args.output / "evaluator-audit.json", evaluator)
        _write(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "failure": report.get("failure"),
                "selected_heading": report.get("selected_heading_steps"),
                "place_ids": report["place_ids"],
            },
            sort_keys=True,
        )
    )
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
