"""Recompute arbitrary-route visual goal-search evidence independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    view_openness,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    VisualFrontierProgram,
    align_panorama_heading,
    temporal_median_rgb,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes_valid(root: Path, hashes: dict[str, str]) -> bool:
    return bool(hashes) and all(
        (root / name).is_file() and _digest(root / name) == expected
        for name, expected in hashes.items()
    )


def _signature_digest(signature: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode()
    ).hexdigest()


def _panorama(
    directory: Path, encoder: Any, period: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], list[Any]]:
    frames = []
    for index in range(period):
        with Image.open(directory / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return (
        encoder.encode(frames),
        tuple(encoder.scene_encoder.encode(frame) for frame in frames),
        frames,
    )


def _scan(
    root: Path, role: str, metadata: dict[str, Any], encoder: Any, period: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]:
    directory = root / "scans" / role
    panorama, views, frames = _panorama(directory, encoder, period)
    valid = (
        metadata["view_count"] == period
        and _signature_digest(panorama) == metadata["panorama_sha256"]
        and _hashes_valid(directory, metadata["image_sha256"])
    )
    count = int(metadata["temporal_sample_count"])
    valid = valid and count > 0 and count % 2 == 1
    for index, frame in enumerate(frames):
        samples = []
        role_name = f"view-{index:02d}"
        for sample_index in range(count):
            name = f"sample-{sample_index:02d}.png"
            path = directory / "samples" / role_name / name
            valid = (
                valid
                and _digest(path) == metadata["temporal_sample_sha256"][role_name][name]
            )
            with Image.open(path) as image:
                samples.append(np.asarray(image.convert("RGB")))
        valid = valid and bool(np.array_equal(frame, temporal_median_rgb(samples)))
    return panorama, views, valid


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.dist(
        tuple(left[key] for key in ("xpos", "ypos", "zpos")),
        tuple(right[key] for key in ("xpos", "ypos", "zpos")),
    )


def _physical(
    result: dict[str, Any],
    positions: list[dict[str, float]],
    config: dict[str, Any],
    goal: str,
) -> dict[str, Any]:
    nodes = result["replay"]["localizations"]
    origin = positions[int(nodes[0]["position_index"])]
    start_index = int(nodes[-1]["position_index"])
    end_index = int(result["target_position_index"])
    start, end = positions[start_index], positions[end_index]
    source_goal, target_goal = _distance(start, origin), _distance(end, origin)
    travel = _distance(start, end)
    drop = max(
        start["ypos"] - item["ypos"] for item in positions[start_index : end_index + 1]
    )
    return {
        "source_goal_distance_3d": source_goal,
        "target_goal_distance_3d": target_goal,
        "travel_distance_3d": travel,
        "maximum_vertical_drop": drop,
        "accepted": travel >= config["minimum_travel"]
        and drop <= config["maximum_vertical_drop"]
        and target_goal < source_goal
        and (
            result["recognized_place_id"] != goal
            or target_goal <= config["goal_tolerance"]
        ),
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


def audit(root: Path) -> dict[str, Any]:
    report = _load(root / "report.json")
    config = _load(root / "config.json")
    evaluator = _load(root / "evaluator-audit.json")
    spec = config["source_spec"]
    encoder = PANORAMA_ENCODERS["edge"]()
    period = int(spec["period"])
    signatures = {}
    views = {}
    scans_valid = True
    for role, metadata in report["scans"].items():
        signatures[role], views[role], valid = _scan(
            root, role, metadata, encoder, period
        )
        scans_valid = scans_valid and valid
    references = {}
    for place in report["reference_place_ids"]:
        _signature, reference, _frames = _panorama(
            root / "references" / place, encoder, period
        )
        references[f"reference:{place}"] = reference
    goal_signature, _views, _frames = _panorama(
        root / "goal-reference", encoder, period
    )
    _signature, _source_views, source_frames = _panorama(
        root / "references" / spec["source"], encoder, period
    )
    openness = tuple(view_openness(frame) for frame in source_frames)
    headings = sorted(range(period), key=lambda heading: (-openness[heading], heading))[
        : min(config["candidate_count"], period)
    ]
    candidate_generation = (
        headings == report["candidate_headings"]
        and list(openness) == report["openness_scores"]
    )
    routes_valid = True
    localization_valid = True
    alignments_valid = True
    roles_valid = True
    visual_inputs_valid = True
    metrics_valid = True
    policy_ranking = []
    source_path = root / "source-evidence/exploration.sqlite3"
    with ExplorationMemory.open(source_path) as source_memory:
        source_route = source_memory.plan_route(spec["origin"], spec["source"])
        routes_valid = (
            source_route is not None
            and source_route.model_dump(mode="json")
            == spec["route"]
            == report["source_route"]
        )
        with ExplorationMemory.open(root / "exploration.sqlite3") as final_memory:
            all_results = [*report["candidate_results"], report["validation_result"]]
            for result in all_results:
                replay = result["replay"]
                routes_valid = (
                    routes_valid
                    and [item["place_id"] for item in replay["localizations"]]
                    == spec["route"]["place_path"]
                )
                for node in replay["localizations"]:
                    match = source_memory.match_modality_place(
                        spec["encoder"], signatures[node["scan_role"]]
                    )
                    localization_valid = (
                        localization_valid
                        and match == node["recognized_place_id"] == node["place_id"]
                    )
                source_signature = signatures[replay["localizations"][-1]["scan_role"]]
                target_signature = signatures[result["target_scan_role"]]
                match = source_memory.match_modality_place(
                    spec["encoder"], target_signature
                )
                visual_inputs_valid = visual_inputs_valid and (
                    list(source_signature) == result["source_signature"]
                    and list(target_signature) == result["signature"]
                    and match == result["recognized_place_id"]
                )
                for execution in [*replay["executions"], result["execution"]]:
                    alignment = align_panorama_heading(
                        references[execution["reference_scan_role"]],
                        views[execution["source_scan_role"]],
                    )
                    alignments_valid = (
                        alignments_valid
                        and alignment.model_dump(mode="json") == execution["alignment"]
                    )
                    program = VisualFrontierProgram.from_roles(
                        tuple(execution["action_roles"])
                    )
                    roles_valid = (
                        roles_valid
                        and program.turn_steps < period
                        and not any(
                            token in role
                            for role in execution["action_roles"]
                            for token in ("forward", "jump", "left", "right", "obj_")
                        )
                    )
                positions = evaluator["sessions"][result["session_index"]]["positions"]
                physical = _physical(result, positions, config, spec["goal"])
                if result["candidate_id"] == "validation":
                    metrics_valid = (
                        metrics_valid and physical == evaluator["validation_metrics"]
                    )
                else:
                    metrics_valid = (
                        metrics_valid
                        and physical
                        == evaluator["probe_metrics"][result["candidate_id"]]
                    )
                    if match in (None, spec["goal"]):
                        policy_ranking.append(
                            {
                                "candidate": {
                                    "candidate_id": result["candidate_id"],
                                    "heading_steps": result["heading_steps"],
                                    "signature": list(target_signature),
                                    "recognized_place_id": match,
                                },
                                "goal_visual_distance": math.dist(
                                    goal_signature, target_signature
                                ),
                                "source_visual_distance": math.dist(
                                    source_signature, target_signature
                                ),
                            }
                        )
            policy_ranking.sort(
                key=lambda item: (
                    item["goal_visual_distance"],
                    -item["source_visual_distance"],
                    item["candidate"]["heading_steps"],
                    item["candidate"]["candidate_id"],
                )
            )
            ranking_valid = (
                policy_ranking == report["ranking"]
                and policy_ranking[0]["candidate"]["candidate_id"]
                == report["selected_result"]["candidate_id"]
            )
            target = report["place_ids"]["target"]
            restart = report["restart_result"]
            for node in restart["replay"]["localizations"]:
                match = final_memory.match_modality_place(
                    spec["encoder"], signatures[node["scan_role"]]
                )
                localization_valid = (
                    localization_valid
                    and match == node["recognized_place_id"] == node["place_id"]
                )
            for execution in restart["replay"]["executions"]:
                alignment = align_panorama_heading(
                    references[execution["reference_scan_role"]],
                    views[execution["source_scan_role"]],
                )
                alignments_valid = (
                    alignments_valid
                    and alignment.model_dump(mode="json") == execution["alignment"]
                )
            selected = report["selected_result"]
            selected_position = evaluator["sessions"][selected["session_index"]][
                "positions"
            ][selected["target_position_index"]]
            restart_position = evaluator["sessions"][restart["session_index"]][
                "positions"
            ][restart["replay"]["localizations"][-1]["position_index"]]
            restart_error = _distance(selected_position, restart_position)
            restart_valid = (
                restart_error == evaluator["restart_target_error_3d"]
                and restart_error <= config["goal_tolerance"]
            )
            persistent_route = final_memory.plan_route(spec["origin"], target)
            persistent_goal = final_memory.plan_route(target, spec["goal"])
            routes_valid = (
                routes_valid
                and persistent_route is not None
                and persistent_route.model_dump(mode="json")
                == report["persisted_route"]
            )
            routes_valid = (
                routes_valid
                and (
                    persistent_goal.model_dump(mode="json") if persistent_goal else None
                )
                == report["goal_route"]
            )
            source_transitions = {
                item.model_dump_json() for item in source_memory.transitions()
            }
            final_transitions = {
                item.model_dump_json() for item in final_memory.transitions()
            }
            added = [
                item
                for item in final_memory.transitions()
                if item.model_dump_json() not in source_transitions
            ]
            topology_valid = (
                source_transitions <= final_transitions
                and len(added) == 1
                and (
                    added[0].source_place_id == spec["source"]
                    and added[0].target_place_id == target
                    and list(added[0].action_roles)
                    == report["selected_result"]["execution"]["action_roles"]
                )
            )
            frontier = final_memory.get_frontier(report["frontier_id"])
            topology_valid = topology_valid and frontier.status.value == "resolved"
            memory_valid = (
                final_memory.verify_integrity()
                and final_memory.summary() == report["memory_summary"]
            )
    source_report = _load(root / "source-evidence/report.json")
    source_audit = _load(root / "source-evidence/integrity-audit.json")
    source_accepted = (
        source_report["gate"]["passed"] is True
        and source_audit["valid"] is True
        and all(source_audit["checks"].values())
    )
    source_paths = _source_paths()
    sources_valid = set(source_paths) == set(report["source_sha256"]) and all(
        _digest(source_paths[name]) == expected
        for name, expected in report["source_sha256"].items()
    )
    events = (root / "events.jsonl").read_text(encoding="utf-8")
    public_clean = not any(token in events for token in PRIVILEGED_ARTIFACT_TOKENS)
    checks = {
        "source_evidence_accepted": source_accepted,
        "source_evidence_hashes_valid": _hashes_valid(
            root / "source-evidence", report["source_evidence_sha256"]
        ),
        "source_snapshot_valid": _hashes_valid(
            root / "source", report["source_sha256"]
        ),
        "current_source_matches_frozen": sources_valid,
        "initial_references_immutable": _hashes_valid(
            root / "references", report["initial_reference_sha256"]
        ),
        "references_hashes_valid": _hashes_valid(
            root / "references", report["reference_sha256"]
        ),
        "persistent_goal_immutable": _hashes_valid(
            root / "goal-reference", report["goal_reference_sha256"]
        ),
        "scans_recomputed_from_png": scans_valid,
        "candidate_generation_recomputed": candidate_generation,
        "visual_inputs_recomputed": visual_inputs_valid,
        "rgb_goal_ranking_recomputed": ranking_valid,
        "route_localizations_recomputed": localization_valid,
        "action_alignments_recomputed": alignments_valid,
        "opaque_programs_valid": roles_valid,
        "physical_metrics_recomputed": metrics_valid,
        "independent_restart_arrival_recomputed": restart_valid,
        "persistent_routes_recomputed": routes_valid,
        "exactly_one_directed_transition_added": topology_valid,
        "memory_projection_and_integrity": memory_valid,
        "public_telemetry_clean": public_clean,
        "selected_validation_physical_progress": evaluator["probe_metrics"][
            report["selected_result"]["candidate_id"]
        ]["accepted"]
        and evaluator["validation_metrics"]["accepted"],
        "reported_gate_passed": report["gate"]["passed"] is True
        and all(report["gate"].values()),
    }
    return {
        "suite": "natural_visual_goal_search_audit",
        "checks": checks,
        "valid": all(checks.values()),
        "root": str(root.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.write:
        (args.root / "integrity-audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
