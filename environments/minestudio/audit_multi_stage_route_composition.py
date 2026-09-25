"""Independently audit real two-edge visual route composition."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_visual_route_crisis import (
    all_true,
    digest,
    load,
    recompute_trajectory,
    replay_audit,
    source_audit,
)

from minecraft_scientist.cognitive_map import (
    VisualRouteStatus,
    VisualTopologicalMap,
)
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run
    prereg = load(root / "preregistered.json")
    report = load(root / "report.json")
    runtime = load(root / "runtime-manifest.json")
    source_manifest = load(root / "source-artifact-manifest.json")
    upstream_audit = load(root / "upstream" / "integrity-audit.json")
    settings = prereg["settings"]

    snapshot_checks = []
    for relative, expected_hash in runtime["source_sha256"].items():
        path = root / "source-snapshot" / relative
        actual = digest(path) if path.exists() else None
        snapshot_checks.append({
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "valid": actual == expected_hash,
        })
    copied_source_checks = {
        name: tree_manifest(root / name) == expected
        for name, expected in source_manifest.items()
    }

    map_path = root / "visual-topology.json"
    topology = VisualTopologicalMap.load(map_path)
    start_signature = tuple(report["start_signature"])
    midpoint_signature = tuple(report["midpoint_signature"])
    target_signature = tuple(report["target_signature"])
    start_id = topology.node_id_for(start_signature)
    midpoint_id = topology.node_id_for(midpoint_signature)
    target_id = topology.node_id_for(target_signature)
    first_id, second_id = report["route_ids"]
    first = topology.get_route(first_id)
    second = topology.get_route(second_id)
    plan = topology.plan(start_signature, target_id)
    reverse = topology.plan(target_signature, start_id)
    challenged = VisualTopologicalMap.load(map_path)
    challenged.challenge_route(
        second_id,
        reason="independent negative-control reconstruction",
    )
    blocked = challenged.plan(start_signature, target_id)
    map_checks = {
        "format_hash_and_shape_match": (
            report["map_format_version"] == VisualTopologicalMap.FORMAT_VERSION
            and digest(map_path) == report["map_sha256"]
            and topology.graph.number_of_nodes() == 3
            and topology.graph.number_of_edges() == 2
            and report["map_node_count"] == 3
            and report["map_edge_count"] == 2
        ),
        "three_landmark_ids_recomputed": (
            len({start_id, midpoint_id, target_id}) == 3
            and midpoint_id == report["midpoint_id"]
            and target_id == report["target_id"]
        ),
        "first_edge_connects_start_to_midpoint": (
            first.source_id == start_id
            and first.target_id == midpoint_id
            and first.status is VisualRouteStatus.ACTIVE
            and list(first.action_roles) == report["first_edge_program"]
        ),
        "second_edge_connects_midpoint_to_goal": (
            second.source_id == midpoint_id
            and second.target_id == target_id
            and second.status is VisualRouteStatus.ACTIVE
            and list(second.action_roles) == report["second_edge_program"]
        ),
        "no_direct_start_to_goal_edge": (
            not topology.graph.has_edge(start_id, target_id)
            and report["no_direct_start_to_goal_edge"] is True
        ),
        "two_edge_plan_recomputed": (
            plan is not None
            and plan.route_ids == (first_id, second_id)
            and list(plan.action_roles) == report["composed_program"]
        ),
        "directed_reverse_route_rejected": reverse is None
        and report["reverse_route_rejected"] is True,
        "challenged_second_edge_blocks_plan": blocked is None
        and report["challenged_second_edge_blocks_plan"] is True,
    }

    source_results = [
        source_audit(root / f"source-{index}", expected, settings)
        for index, expected in enumerate(report["source_results"])
    ]
    source_roles = [
        load(root / f"source-{index}" / "roles.json")
        for index in range(2)
    ]
    source_independence = {
        "source_control_bindings_distinct": source_roles[0] != source_roles[1]
        and report["source_control_bindings_distinct"] is True,
        "first_edge_comes_from_source_zero": (
            list(first.action_roles)
            == report["source_results"][0]["program"][: report["split_index"]]
        ),
        "second_edge_comes_from_source_one": (
            list(second.action_roles)
            == report["source_results"][1]["program"][report["split_index"] :]
        ),
    }

    replay_directory = root / "replay"
    replay_checks = replay_audit(
        replay_directory,
        report["replay"],
        map_path,
        report["route_ids"],
        settings,
        should_succeed=True,
    )
    replay_report = load(replay_directory / "report.json")
    recomputed = recompute_trajectory(replay_directory / "replay")
    split_index = report["split_index"]
    midpoint_observed = layout_landmark_signature(recomputed[split_index])
    target_observed = layout_landmark_signature(recomputed[-1])
    boundaries = replay_report["boundary_matches"]
    boundary_checks = {
        "replanning_mode_enabled": replay_report["replan_at_landmarks"] is True,
        "exactly_one_midpoint_replan": replay_report["landmark_replans"] == 1
        and report["replay"]["landmark_replans"] == 1,
        "two_route_boundaries_recorded": len(boundaries) == 2,
        "midpoint_signature_recomputed": midpoint_observed == midpoint_signature
        and boundaries[0]["signature"] == list(midpoint_signature),
        "midpoint_id_matched_before_second_edge": (
            boundaries[0]["route_id"] == first_id
            and boundaries[0]["expected_landmark_id"] == midpoint_id
            and boundaries[0]["matched_landmark_id"] == midpoint_id
        ),
        "target_signature_recomputed": target_observed == target_signature
        and boundaries[1]["signature"] == list(target_signature),
        "target_id_matched_after_second_edge": (
            boundaries[1]["route_id"] == second_id
            and boundaries[1]["expected_landmark_id"] == target_id
            and boundaries[1]["matched_landmark_id"] == target_id
        ),
        "no_online_route_exploration": (
            replay_report["online_route_exploration_actions"] == 0
            and report["replay"]["online_route_exploration_actions"] == 0
        ),
    }

    audit = {
        "run": str(root),
        "auditor_sha256": digest(Path(__file__)),
        "preregistration_intact": digest(root / "preregistered.json")
        == report["preregistration_sha256"],
        "runtime_manifest_intact": digest(root / "runtime-manifest.json")
        == report["runtime_manifest_sha256"],
        "source_artifact_manifest_intact": digest(
            root / "source-artifact-manifest.json"
        ) == report["source_artifact_manifest_sha256"],
        "upstream_audit_intact_and_valid": (
            upstream_audit.get("valid") is True
            and digest(root / "upstream" / "integrity-audit.json")
            == report["upstream_audit_sha256"]
            == prereg["upstream_audit_sha256"]
        ),
        "source_snapshot": snapshot_checks,
        "source_snapshot_complete": bool(snapshot_checks)
        and all(item["valid"] for item in snapshot_checks),
        "copied_source_artifacts": copied_source_checks,
        "copied_source_artifacts_intact": all(copied_source_checks.values()),
        "map": map_checks,
        "map_valid": all_true(map_checks),
        "sources": source_results,
        "source_success_rate": sum(all_true(item) for item in source_results)
        / len(source_results),
        "source_independence": source_independence,
        "source_independence_valid": all_true(source_independence),
        "replay": replay_checks,
        "replay_valid": all_true(replay_checks),
        "boundaries": boundary_checks,
        "boundaries_valid": all_true(boundary_checks),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["runtime_manifest_intact"],
        audit["source_artifact_manifest_intact"],
        audit["upstream_audit_intact_and_valid"],
        audit["source_snapshot_complete"],
        audit["copied_source_artifacts_intact"],
        audit["map_valid"],
        audit["source_success_rate"] == prereg["required_source_success_rate"],
        audit["source_independence_valid"],
        audit["replay_valid"],
        audit["boundaries_valid"],
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
