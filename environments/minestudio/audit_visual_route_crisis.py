"""Independently audit a real MineStudio visual-route crisis run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from obstacle_replanning_probe import classify_handles, frame_features
from PIL import Image

from minecraft_scientist.cognitive_map import (
    VisualRouteStatus,
    VisualTopologicalMap,
)
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    return (
        [json.loads(line) for line in text.splitlines() if line.strip()],
        text,
    )


def recompute_calibration(directory: Path) -> tuple[list[dict], bool, list[str]]:
    saved = load(directory / "calibration.json")
    recomputed = []
    valid = True
    texts = []
    for index, item in enumerate(saved):
        trial = directory / "calibration" / f"trial-{index}"
        before = frame_features(np.asarray(Image.open(trial / "before.png")))
        after = frame_features(np.asarray(Image.open(trial / "after.png")))
        candidate = {
            "handle_id": item["handle_id"],
            "before": before,
            "after": after,
            "marker_area_delta": (
                after["marker"]["pixel_count"]
                - before["marker"]["pixel_count"]
            ),
            "marker_x_delta": (
                after["marker"]["centroid_x"]
                - before["marker"]["centroid_x"]
            ),
        }
        recomputed.append(candidate)
        events, text = load_events(trial / "events.jsonl")
        texts.append(text)
        moves = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        valid = valid and (
            len(moves) == 1
            and moves[0]["executed_action"]["target_id"] == item["handle_id"]
            and moves[0]["intervention_compliance"] is True
            and moves[0]["event"]["success"] is True
        )
    return recomputed, valid, texts


def recompute_trajectory(directory: Path) -> list[dict]:
    saved = load(directory / "trajectory.json")
    result = []
    for item in saved:
        frame = np.asarray(
            Image.open(directory / f"frame-{item['observation_index']:02d}.png")
        )
        phase = (
            {"phase": item["phase"]}
            if "phase" in item
            else {"action_role": item["action_role"]}
        )
        result.append({
            "observation_index": item["observation_index"],
            **phase,
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            **frame_features(frame),
        })
    return result


def path_metrics(
    positions: list[dict], obstacle_offset: int
) -> tuple[float | None, float | None, float | None, bool]:
    if not positions:
        return None, None, None, False
    start = positions[0]
    relative = [
        (item["xpos"] - start["xpos"], item["zpos"] - start["zpos"])
        for item in positions
        if all(key in item and key in start for key in ("xpos", "zpos"))
    ]
    offsets = [abs(x) for x, _ in relative]
    left, right = obstacle_offset - 1.5, obstacle_offset + 1.5
    avoids = bool(relative) and not any(
        left < x < right and 2.5 < z < 4.5 for x, z in relative
    )
    return (
        relative[-1][1] if relative else None,
        max(offsets, default=None),
        offsets[-1] if offsets else None,
        avoids,
    )


def public_has_no_leak(parts: list[str]) -> bool:
    lowered = "\n".join(parts).lower()
    forbidden = (
        "location_stats",
        "xpos",
        "ypos",
        "zpos",
        '"control"',
        "positive_z",
        "obstacle_center_x_relative_to_start",
    )
    return not any(token in lowered for token in forbidden)


def source_audit(
    directory: Path,
    expected: dict,
    settings: dict,
) -> dict:
    report = load(directory / "report.json")
    saved_calibration = load(directory / "calibration.json")
    saved_roles = load(directory / "roles.json")
    recomputed_calibration, events_valid, calibration_texts = (
        recompute_calibration(directory)
    )
    trajectory = load(directory / "route" / "trajectory.json")
    recomputed = recompute_trajectory(directory / "route")
    route_events, route_text = load_events(directory / "route" / "events.jsonl")
    inverse = {handle: role for role, handle in saved_roles.items()}
    moves = [
        event for event in route_events
        if event["executed_action"]["kind"] == "move"
    ]
    program = tuple(
        inverse[event["executed_action"]["target_id"]] for event in moves
    )
    evaluator = load(directory / "evaluator-audit.json")
    metrics = path_metrics(
        evaluator["route"]["positions"], expected["obstacle_offset"]
    )
    progress, excursion, final_offset, avoids = metrics
    public_parts = [
        (directory / name).read_text(encoding="utf-8")
        for name in ("report.json", "calibration.json", "roles.json")
    ] + calibration_texts + [route_text, json.dumps(trajectory)]
    return {
        "seed_and_mapping_match": (
            report["seed"] == expected["seed"]
            and report["mapping_seed"] == expected["mapping_seed"]
            and report["obstacle_offset"] == expected["obstacle_offset"]
        ),
        "report_hash_matches": digest(directory / "report.json")
        == expected["report_sha256"],
        "calibration_features_recomputed": saved_calibration
        == recomputed_calibration,
        "roles_recomputed_from_pixels": saved_roles
        == classify_handles(recomputed_calibration),
        "calibration_events_valid": events_valid,
        "trajectory_features_recomputed": trajectory == recomputed,
        "signatures_recomputed": (
            list(layout_landmark_signature(recomputed[0]))
            == expected["source_signature"]
            and list(layout_landmark_signature(recomputed[-1]))
            == expected["target_signature"]
        ),
        "program_recomputed": list(program) == expected["program"],
        "route_events_valid": bool(moves)
        and all(
            event["intervention_compliance"] is True
            and event["event"]["success"] is True
            for event in moves
        ),
        "metrics_match_report": (
            progress == report["evaluator_goal_progress"]
            and excursion == report["evaluator_max_lateral_excursion"]
            and final_offset == report["evaluator_final_lateral_offset"]
        ),
        "metrics_in_range": (
            progress is not None
            and settings["minimum_goal_progress"] <= progress
            <= settings["maximum_goal_progress"]
            and excursion is not None
            and settings["minimum_lateral_excursion"] <= excursion
            <= settings["maximum_lateral_excursion"]
            and final_offset is not None
            and final_offset <= settings["maximum_final_lateral_offset"]
            and avoids
        ),
        "public_artifacts_have_no_leak": public_has_no_leak(public_parts),
        "privileged_positions_are_evaluator_only": "xpos" in json.dumps(evaluator)
        and public_has_no_leak(public_parts),
    }


def replay_audit(
    directory: Path,
    expected: dict,
    expected_map: Path,
    expected_route_id: str | list[str],
    settings: dict,
    *,
    should_succeed: bool,
) -> dict:
    report = load(directory / "report.json")
    saved_calibration = load(directory / "calibration.json")
    roles = load(directory / "roles.json")
    recomputed_calibration, events_valid, calibration_texts = (
        recompute_calibration(directory)
    )
    trajectory = load(directory / "replay" / "trajectory.json")
    recomputed = recompute_trajectory(directory / "replay")
    events, event_text = load_events(directory / "replay" / "events.jsonl")
    moves = [
        event for event in events
        if event["executed_action"]["kind"] == "move"
    ]
    bound_handles = [
        roles[item["action_role"]]
        for item in recomputed[1:]
    ]
    evaluator = load(directory / "evaluator-audit.json")
    progress, excursion, final_offset, avoids = path_metrics(
        evaluator["replay"]["positions"], expected["obstacle_offset"]
    )
    metric_success = (
        progress is not None
        and settings["minimum_goal_progress"] <= progress
        <= settings["maximum_goal_progress"]
        and excursion is not None
        and settings["minimum_lateral_excursion"] <= excursion
        <= settings["maximum_lateral_excursion"]
        and final_offset is not None
        and final_offset <= settings["maximum_final_lateral_offset"]
        and avoids
        and report["obstacle_passed_visually"]
        and report["goal_reached_visually"]
    )
    public_parts = [
        (directory / name).read_text(encoding="utf-8")
        for name in ("report.json", "calibration.json", "roles.json")
    ] + calibration_texts + [event_text, json.dumps(trajectory)]
    expected_route_ids = (
        [expected_route_id]
        if isinstance(expected_route_id, str)
        else expected_route_id
    )
    return {
        "seed_and_mapping_match": (
            report["seed"] == expected["seed"]
            and report["mapping_seed"] == expected["mapping_seed"]
            and report["obstacle_offset"] == expected["obstacle_offset"]
        ),
        "map_hash_matches": report["map_sha256"] == digest(expected_map),
        "calibration_features_recomputed": saved_calibration
        == recomputed_calibration,
        "roles_recomputed_from_pixels": roles
        == classify_handles(recomputed_calibration),
        "calibration_events_valid": events_valid,
        "trajectory_features_recomputed": trajectory == recomputed,
        "signature_recomputed": list(layout_landmark_signature(recomputed[0]))
        == expected["source_signature"],
        "expected_route_selected": report["route_ids"] == expected_route_ids,
        "program_rebound_to_current_handles": (
            len(moves) == len(bound_handles)
            and [event["executed_action"]["target_id"] for event in moves]
            == bound_handles
        ),
        "route_events_valid": bool(moves)
        and all(
            event["intervention_compliance"] is True
            and event["event"]["success"] is True
            for event in moves
        ),
        "metrics_match_report": (
            progress == report["evaluator_goal_progress"]
            and excursion == report["evaluator_max_lateral_excursion"]
            and final_offset == report["evaluator_final_lateral_offset"]
        ),
        "outcome_matches_preregistration": metric_success is should_succeed,
        "control_valid_despite_failed_outcome": should_succeed
        or (
            report["calibration_valid"]
            and report["all_move_actions_valid"]
            and report["continuous_replay_episode"]
            and not metric_success
        ),
        "public_artifacts_have_no_leak": public_has_no_leak(public_parts),
        "privileged_positions_are_evaluator_only": "xpos" in json.dumps(evaluator)
        and public_has_no_leak(public_parts),
    }


def all_true(item: dict) -> bool:
    return all(value is True for value in item.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run
    prereg = load(root / "preregistered.json")
    report = load(root / "report.json")
    manifest = load(root / "runtime-manifest.json")
    settings = prereg["settings"]

    snapshot_checks = []
    for relative, expected_hash in manifest["source_sha256"].items():
        path = root / "source-snapshot" / relative
        actual = digest(path) if path.exists() else None
        snapshot_checks.append({
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "valid": actual == expected_hash,
        })

    prechange_path = root / "map-prechange.json"
    challenged_path = root / "map-challenged.json"
    candidate_path = root / "map-candidate.json"
    final_path = root / "visual-topology.json"
    prechange = VisualTopologicalMap.load(prechange_path)
    challenged = VisualTopologicalMap.load(challenged_path)
    candidate = VisualTopologicalMap.load(candidate_path)
    final = VisualTopologicalMap.load(final_path)
    old_id = report["old_route"]["route_id"]
    replacement_id = report["replacement_route"]["route_id"]
    source_signature, changed_signature = (
        tuple(item) for item in report["source_and_mutation_signatures"]
    )
    target_id = report["old_route"]["target_id"]
    pre_plan = prechange.plan(source_signature, target_id)
    challenged_plan = challenged.plan(changed_signature, target_id)
    candidate_plan = candidate.plan(changed_signature, target_id)
    final_plan = final.plan(changed_signature, target_id)
    lifecycle = {
        "prechange_route_active_and_confident": (
            prechange.get_route(old_id).status is VisualRouteStatus.ACTIVE
            and prechange.get_route(old_id).trials == 2
            and prechange.get_route(old_id).successes == 2
            and prechange.get_route(old_id).confidence == 0.75
            and pre_plan is not None
            and pre_plan.route_ids == (old_id,)
        ),
        "failure_challenged_and_excluded_route": (
            challenged.get_route(old_id).status
            is VisualRouteStatus.CHALLENGED
            and challenged.get_route(old_id).trials == 3
            and challenged.get_route(old_id).successes == 2
            and challenged.get_route(old_id).failure_streak == 1
            and challenged_plan is None
        ),
        "candidate_selected_while_old_remained_challenged": (
            candidate.get_route(old_id).status
            is VisualRouteStatus.CHALLENGED
            and candidate.get_route(replacement_id).status
            is VisualRouteStatus.ACTIVE
            and candidate_plan is not None
            and candidate_plan.route_ids == (replacement_id,)
        ),
        "final_supersession_links_are_reciprocal": (
            final.get_route(old_id).status is VisualRouteStatus.SUPERSEDED
            and final.get_route(old_id).superseded_by == replacement_id
            and final.get_route(replacement_id).status
            is VisualRouteStatus.ACTIVE
            and final.get_route(replacement_id).parent_route_id == old_id
        ),
        "final_plan_uses_only_replacement": final_plan is not None
        and final_plan.route_ids == (replacement_id,),
        "format_and_hash_match_report": (
            report["map_format_version"] == VisualTopologicalMap.FORMAT_VERSION
            and digest(final_path) == report["map_sha256"]
        ),
    }

    source_results = [
        source_audit(root / f"source-{index}", expected, settings)
        for index, expected in enumerate(report["source_results"])
    ]
    failed = replay_audit(
        root / "failed-replay",
        report["failed_replay"],
        prechange_path,
        old_id,
        settings,
        should_succeed=False,
    )
    repair_expected = {
        **report["repair"],
        "source_signature": report["failed_replay"]["source_signature"],
        "target_signature": report["source_results"][0]["target_signature"],
    }
    repair = source_audit(root / "repair-source", repair_expected, settings)
    shadow = replay_audit(
        root / "shadow-validation",
        report["shadow_validation"],
        candidate_path,
        replacement_id,
        settings,
        should_succeed=True,
    )
    audit = {
        "run": str(root),
        "auditor_sha256": digest(Path(__file__)),
        "preregistration_intact": digest(root / "preregistered.json")
        == report["preregistration_sha256"],
        "runtime_manifest_intact": digest(root / "runtime-manifest.json")
        == report["runtime_manifest_sha256"],
        "source_snapshot": snapshot_checks,
        "source_snapshot_complete": bool(snapshot_checks)
        and all(item["valid"] for item in snapshot_checks),
        "lifecycle": lifecycle,
        "lifecycle_valid": all_true(lifecycle),
        "sources": source_results,
        "source_success_rate": sum(all_true(item) for item in source_results)
        / len(source_results),
        "failed_replay": failed,
        "failed_replay_valid": all_true(failed),
        "repair": repair,
        "repair_valid": all_true(repair),
        "shadow_validation": shadow,
        "shadow_validation_valid": all_true(shadow),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["runtime_manifest_intact"],
        audit["source_snapshot_complete"],
        audit["lifecycle_valid"],
        audit["source_success_rate"] == 1.0,
        audit["failed_replay_valid"],
        audit["repair_valid"],
        audit["shadow_validation_valid"],
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
