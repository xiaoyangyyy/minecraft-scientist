"""Independent artifact audit for natural_visual_goal_frontier_benchmark.py."""

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
    _candidate_headings,
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


def _signature_digest(signature: tuple[int, ...]) -> str:
    payload = json.dumps(signature, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    return {
        "natural_visual_goal_frontier_benchmark.py": here
        / "natural_visual_goal_frontier_benchmark.py",
        "audit_natural_visual_goal_frontier.py": Path(__file__).resolve(),
        "natural_visual_detour_benchmark.py": here
        / "natural_visual_detour_benchmark.py",
        "natural_visual_expedition_benchmark.py": here
        / "natural_visual_expedition_benchmark.py",
        "natural_visual_loop_benchmark.py": here
        / "natural_visual_loop_benchmark.py",
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "memory.py": repository / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": repository
        / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": repository
        / "src/minecraft_scientist/minecraft/minestudio.py",
    }


def _load_scan(
    root: Path,
    role: str,
    metadata: dict[str, Any],
    *,
    encoder: Any,
    period: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]:
    directory = root / "scans" / role
    frames = []
    valid = int(metadata["view_count"]) == period
    for index in range(period):
        path = directory / f"view-{index:02d}.png"
        valid = valid and path.is_file()
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
        valid = valid and _digest(path) == metadata["image_sha256"][path.name]
    panorama = encoder.encode(frames)
    views = tuple(encoder.scene_encoder.encode(frame) for frame in frames)
    valid = valid and _signature_digest(panorama) == metadata["panorama_sha256"]
    sample_count = int(metadata.get("temporal_sample_count", 1))
    for view_index, frame in enumerate(frames):
        samples = []
        view_name = f"view-{view_index:02d}"
        for sample_index in range(sample_count):
            sample_name = f"sample-{sample_index:02d}.png"
            path = directory / "samples" / view_name / sample_name
            valid = valid and path.is_file()
            valid = valid and _digest(path) == metadata["temporal_sample_sha256"][
                view_name
            ][sample_name]
            with Image.open(path) as image:
                samples.append(np.asarray(image.convert("RGB")))
        valid = valid and bool(np.array_equal(temporal_median_rgb(samples), frame))
    return panorama, views, valid


def _reference_views(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[tuple[int, ...], ...]:
    result = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            result.append(encoder.scene_encoder.encode(np.asarray(image.convert("RGB"))))
    return tuple(result)


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


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=1e-12)


def _roles_are_opaque(roles: tuple[str, ...], period: int) -> bool:
    try:
        program = VisualFrontierProgram.from_roles(roles)
    except ValueError:
        return False
    forbidden = ("forward", "jump", "back", "left", "right", "obj_")
    return program.turn_steps < period and not any(
        token in role for role in roles for token in forbidden
    )


def audit(root: Path) -> dict[str, Any]:
    report = _load(root / "report.json")
    config = _load(root / "config.json")
    evaluator = _load(root / "evaluator-audit.json")
    source_root = root / "source-evidence"
    source_hashes = _load(root / "source-evidence-sha256.json")
    source_evidence_valid = all(
        (source_root / name).is_file()
        and _digest(source_root / name) == expected
        for name, expected in source_hashes.items()
    )
    source_report = _load(source_root / "report.json")
    source_audit = _load(source_root / "integrity-audit.json")
    source_accepted = (
        source_report.get("suite") == "natural_visual_detour"
        and source_report.get("gate", {}).get("passed") is True
        and source_audit.get("valid") is True
        and source_report.get("terrain_modified") is False
    )

    encoder = PANORAMA_ENCODERS["edge"]()
    period = int(report["inferred_camera_period"])
    panoramas: dict[str, tuple[int, ...]] = {}
    views: dict[str, tuple[tuple[int, ...], ...]] = {}
    scans_valid = True
    for role, metadata in report["scans"].items():
        panorama, scan_views, valid = _load_scan(
            root, role, metadata, encoder=encoder, period=period
        )
        panoramas[role] = panorama
        views[role] = scan_views
        scans_valid = scans_valid and valid
    references = {
        role: _reference_views(source_root, role, encoder=encoder, period=period)
        for role in ("a-reference", "b-reference", "e-reference")
    }

    metrics_valid = True
    alignments_valid = True
    roles_valid = True
    localization_valid = True
    recomputed: list[dict[str, Any]] = []
    place_a = str(report["place_ids"]["a"])
    place_e = str(report["place_ids"]["e"])
    target = str(report["place_ids"]["target"])
    with ExplorationMemory.open(root / "exploration.sqlite3") as memory:
        for result in [*report["candidate_results"], report["validation_result"]]:
            replay = result["replay"]
            localization_valid = localization_valid and replay["e_match"] == place_e
            for execution in replay["executions"]:
                roles = tuple(str(item) for item in execution["action_roles"])
                roles_valid = roles_valid and _roles_are_opaque(roles, period)
                alignment = align_panorama_heading(
                    references[str(execution["reference_scan_role"])],
                    views[str(execution["source_scan_role"])],
                )
                alignments_valid = alignments_valid and (
                    alignment.model_dump(mode="json") == execution["alignment"]
                )
            roles = tuple(str(item) for item in result["candidate_action_roles"])
            roles_valid = roles_valid and _roles_are_opaque(roles, period)
            outbound = result["outbound_execution"]
            alignment = align_panorama_heading(
                references[str(outbound["reference_scan_role"])],
                views[str(outbound["source_scan_role"])],
            )
            alignments_valid = alignments_valid and (
                alignment.model_dump(mode="json") == outbound["alignment"]
            )

            a_role = str(result["a_scan_role"])
            e_role = str(result["e_scan_role"])
            f_role = str(result["f_scan_role"])
            goal_distance = math.dist(panoramas[a_role], panoramas[f_role])
            source_distance = math.dist(panoramas[e_role], panoramas[f_role])
            session_index = int(result.get("session_index", evaluator["validation_session_index"]))
            positions = evaluator["sessions"][session_index]["positions"]
            indices = result["evaluator"]
            source_goal = _distance_3d(
                positions, int(indices["e_position_index"]), int(indices["a_position_index"])
            )
            frontier_goal = _distance_3d(
                positions, int(indices["f_position_index"]), int(indices["a_position_index"])
            )
            outbound_3d = _distance_3d(
                positions, int(indices["e_position_index"]), int(indices["f_position_index"])
            )
            return_error = _distance_3d(
                positions,
                int(indices["e_position_index"]),
                int(indices["return_position_index"]),
            )
            drop = _vertical_drop(
                positions,
                int(indices["e_position_index"]),
                int(indices["return_position_index"]),
            )
            metrics_valid = metrics_valid and all(
                _close(left, right)
                for left, right in (
                    (goal_distance, result["goal_visual_distance"]),
                    (source_distance, result["source_visual_distance"]),
                    (source_goal, indices["source_goal_distance_3d"]),
                    (frontier_goal, indices["frontier_goal_distance_3d"]),
                    (outbound_3d, indices["outbound_distance_3d"]),
                    (return_error, indices["return_error_3d"]),
                    (drop, indices["maximum_vertical_drop"]),
                )
            )
            if "session_index" in result:
                recomputed.append(
                    {
                        "heading": int(result["heading_steps"]),
                        "goal": goal_distance,
                        "source": source_distance,
                        "eligible": result["outbound_match"] in (None, place_a),
                    }
                )

        memory_integrity = memory.verify_integrity()
        memory_summary_valid = memory.summary() == report["persisted_frontier"]["summary"]
        outward = memory.plan_route(place_e, target)
        reverse = memory.plan_route(target, place_e)
        goal = memory.plan_route(target, place_a)
        plans_valid = (
            outward is not None
            and outward.model_dump(mode="json")
            == report["persisted_frontier"]["outward"]
            and reverse is None
            and report["persisted_frontier"]["return"] is None
            and goal is None
            and report["persisted_frontier"]["goal"] is None
        )
        transitions = memory.transitions()
        matching_outbound = [
            item
            for item in transitions
            if item.source_place_id == place_e and item.target_place_id == target
        ]
        matching_reverse = [
            item
            for item in transitions
            if item.source_place_id == target and item.target_place_id == place_e
        ]
        directed_topology_valid = len(matching_outbound) == 1 and not matching_reverse
        frontiers_valid = (
            set(report["frontier_ids"]) == {"outbound"}
            and memory.get_frontier(report["frontier_ids"]["outbound"]).status.value
            == "resolved"
        )

    eligible = [item for item in recomputed if item["eligible"]]
    selected = min(
        eligible,
        key=lambda item: (item["goal"], -item["source"], item["heading"]),
    )
    selector_valid = selected["heading"] == int(report["selected_heading_steps"])
    validation = report["validation_result"]
    selected_result = report["selected_result"]
    validation_consistent = (
        selected_result["outbound_match"] == validation["outbound_match"]
        and float(selected_result["evaluator"]["frontier_goal_distance_3d"])
        < float(selected_result["evaluator"]["source_goal_distance_3d"])
        and float(validation["evaluator"]["frontier_goal_distance_3d"])
        < float(validation["evaluator"]["source_goal_distance_3d"])
    )

    reference_frames = []
    for index in range(period):
        with Image.open(
            source_root / "scans" / "e-reference" / f"view-{index:02d}.png"
        ) as image:
            reference_frames.append(np.asarray(image.convert("RGB")))
    openness = tuple(view_openness(frame) for frame in reference_frames)
    ranked = _candidate_headings(openness, period)
    excluded = int(report["excluded_known_return_heading"])
    candidates = tuple(item for item in ranked if item != excluded)[
        : int(config["candidate_count"])
    ]
    candidate_generation_valid = (
        list(candidates) == report["candidate_headings"]
        and all(
            _close(left, right)
            for left, right in zip(openness, report["openness_scores"], strict=True)
        )
    )

    source_snapshot_valid = all(
        (root / "source" / name).is_file()
        and _digest(root / "source" / name) == expected
        for name, expected in report["source_sha256"].items()
    )
    current_paths = _source_paths()
    current_sources_match = set(current_paths) == set(report["source_sha256"]) and all(
        _digest(current_paths[name]) == expected
        for name, expected in report["source_sha256"].items()
    )
    public_text = (root / "events.jsonl").read_text(encoding="utf-8")
    public_clean = not any(token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS)
    evaluator_separate = all(
        any(key in item["raw_metadata_keys"] for key in ("xpos", "location_stats"))
        for item in evaluator["sessions"]
    )
    report_relations_valid = (
        report["suite"] == "natural_visual_goal_frontier"
        and report["backend"] == "real-minestudio"
        and report["terrain_modified"] is False
        and report["directed_frontier"] is True
        and report["selector"] == "rgb-goal-directed-frontier-v1"
        and report["return_action_roles"] is None
        and report["validation_result"]["heading_steps"]
        == report["selected_heading_steps"]
        and report["selected_result"]
        == next(
            item
            for item in report["candidate_results"]
            if item["heading_steps"] == report["selected_heading_steps"]
        )
    )
    checks = {
        "candidate_generation_recomputed": candidate_generation_valid,
        "candidate_metrics_recomputed": metrics_valid,
        "candidate_selection_recomputed": selector_valid,
        "current_source_matches_frozen": current_sources_match,
        "directed_topology_recomputed": directed_topology_valid,
        "evaluator_is_separate": evaluator_separate,
        "execution_alignments_recomputed": alignments_valid,
        "independent_validation_recomputed": validation_consistent,
        "memory_integrity": memory_integrity,
        "memory_summary_valid": memory_summary_valid,
        "opaque_action_roles_valid": roles_valid,
        "persistent_directed_plan_recomputed": plans_valid,
        "public_telemetry_clean": public_clean,
        "report_relations_valid": report_relations_valid,
        "resolved_frontier_valid": frontiers_valid,
        "scans_recomputed_from_png": scans_valid,
        "source_e_localization_recomputed": localization_valid,
        "source_evidence_accepted": source_accepted,
        "source_evidence_hashes_valid": source_evidence_valid,
        "source_snapshot_valid": source_snapshot_valid,
        "reported_gate_passed": report["gate"]["passed"] is True
        and all(report["gate"][key] for key in report["gate"] if key != "passed"),
    }
    return {
        "suite": "natural_visual_goal_frontier_audit",
        "root": str(root.resolve()),
        "checks": checks,
        "valid": all(checks.values()),
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
