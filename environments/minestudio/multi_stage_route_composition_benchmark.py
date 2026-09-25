"""Compose two learned visual edges with held-out landmark replanning."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from visual_route_crisis_benchmark import (
    digest,
    load,
    replay_assessment,
    route_program,
    run_replay,
    save,
)

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature

ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument(
        "--source-run",
        type=Path,
        default=ROOT / "runs/visual-route-crisis-seed140-map1400",
        help="Audited route-crisis run supplying two independent source episodes",
    )
    parser.add_argument("--replay-seed", type=int, default=143)
    parser.add_argument("--replay-mapping-seed", type=int, default=1415)
    parser.add_argument("--split-index", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        f"runs/multi-stage-route-seed{args.replay_seed}"
        f"-map{args.replay_mapping_seed}"
    )
    output.mkdir(parents=True, exist_ok=False)

    upstream_audit_path = args.source_run / "integrity-audit-v2.json"
    if not upstream_audit_path.exists():
        upstream_audit_path = args.source_run / "integrity-audit.json"
    upstream_audit = load(upstream_audit_path)
    upstream_report = load(args.source_run / "report.json")
    upstream_prereg = load(args.source_run / "preregistered.json")
    if upstream_audit.get("valid") is not True:
        raise RuntimeError("source run has no valid independent integrity audit")
    if upstream_audit.get("source_success_rate") != 1.0:
        raise RuntimeError("source run did not validate both source episodes")
    if len(upstream_report.get("source_results", ())) != 2:
        raise RuntimeError("source run must contain exactly two source episodes")

    settings = dict(upstream_prereg["settings"])
    replay_case = {
        "seed": args.replay_seed,
        "mapping_seed": args.replay_mapping_seed,
        "obstacle_offset": -1,
    }
    preregistration = {
        "upstream_run_name": args.source_run.name,
        "upstream_audit_sha256": digest(upstream_audit_path),
        "source_cases": upstream_prereg["source_cases"],
        "replay_case": replay_case,
        "split_index": args.split_index,
        "settings": settings,
        "policy": (
            "learn source-to-midpoint and midpoint-to-goal edges from separate "
            "opaque-control episodes; require a two-edge plan; after the first "
            "edge, re-identify its target from RGB and invoke the planner again "
            "before executing the second edge"
        ),
        "required_source_success_rate": 1.0,
        "required_replay_success_rate": 1.0,
        "required_route_edges": 2,
        "required_landmark_replans": 1,
    }
    save(output / "preregistered.json", preregistration)

    source_files = (
        "environments/minestudio/multi_stage_route_composition_benchmark.py",
        "environments/minestudio/audit_multi_stage_route_composition.py",
        "environments/minestudio/visual_route_crisis_benchmark.py",
        "environments/minestudio/audit_visual_route_crisis.py",
        "environments/minestudio/topological_route_replay_probe.py",
        "environments/minestudio/obstacle_replanning_probe.py",
        "environments/minestudio/direction_discovery_probe.py",
        "environments/minestudio/navigation_probe.py",
        "src/minecraft_scientist/cognitive_map/visual_topology.py",
        "src/minecraft_scientist/minecraft/minestudio.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    save(output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("minestudio", "networkx", "numpy", "pydantic", "pillow")
        },
        "source_sha256": source_hashes,
    })
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    copied_sources = []
    upstream_copy = output / "upstream"
    upstream_copy.mkdir()
    shutil.copy2(upstream_audit_path, upstream_copy / "integrity-audit.json")
    shutil.copy2(args.source_run / "report.json", upstream_copy / "report.json")
    shutil.copy2(
        args.source_run / "preregistered.json",
        upstream_copy / "preregistered.json",
    )
    for index in range(2):
        source = args.source_run / f"source-{index}"
        destination = output / f"source-{index}"
        shutil.copytree(source, destination)
        copied_sources.append(destination)
    source_artifact_manifest = {
        f"source-{index}": tree_manifest(directory)
        for index, directory in enumerate(copied_sources)
    }
    save(output / "source-artifact-manifest.json", source_artifact_manifest)

    trajectories = [
        load(directory / "route" / "trajectory.json")
        for directory in copied_sources
    ]
    programs = [route_program(directory) for directory in copied_sources]
    if not 0 < args.split_index < len(programs[0]):
        raise ValueError("split index must be inside the source route program")
    if len(programs[0]) != len(programs[1]):
        raise RuntimeError("source programs have different lengths")
    signatures = [
        (
            layout_landmark_signature(trajectory[0]),
            layout_landmark_signature(trajectory[args.split_index]),
            layout_landmark_signature(trajectory[-1]),
        )
        for trajectory in trajectories
    ]
    if signatures[0] != signatures[1]:
        raise RuntimeError("independent sources did not agree on route landmarks")
    start_signature, midpoint_signature, target_signature = signatures[0]
    if len({start_signature, midpoint_signature, target_signature}) != 3:
        raise RuntimeError("route stages did not form three distinct landmarks")

    topology = VisualTopologicalMap(
        match_tolerance=0.75,
        ambiguity_margin=0.25,
    )
    first_route_id = topology.record_route(
        start_signature,
        midpoint_signature,
        programs[0][:args.split_index],
        success=True,
        cost=args.split_index,
    )
    second_route_id = topology.record_route(
        midpoint_signature,
        target_signature,
        programs[1][args.split_index:],
        success=True,
        cost=len(programs[1]) - args.split_index,
    )
    target_id = topology.node_id_for(target_signature)
    midpoint_id = topology.node_id_for(midpoint_signature)
    map_path = output / "visual-topology.json"
    topology.save(map_path)
    plan = topology.plan(start_signature, target_id)
    if plan is None or plan.route_ids != (first_route_id, second_route_id):
        raise RuntimeError("map did not compose the two preregistered route edges")
    if plan.action_roles != programs[0]:
        raise RuntimeError("composed route did not recover the full source program")
    if topology.plan(target_signature, topology.node_id_for(start_signature)):
        raise RuntimeError("directed map unexpectedly produced a reverse route")
    challenged = VisualTopologicalMap.load(map_path)
    challenged.challenge_route(
        second_route_id,
        reason="negative control removes the required second edge",
    )
    if challenged.plan(start_signature, target_id) is not None:
        raise RuntimeError("planner ignored a missing/challenged second edge")

    replay_directory = output / "replay"
    run_replay(
        replay_case,
        replay_directory,
        settings,
        map_path,
        target_id,
        replan_at_landmarks=True,
    )
    assessment = replay_assessment(replay_directory, settings)
    replay_report = load(replay_directory / "report.json")
    if not assessment["successful"]:
        raise RuntimeError("held-out composed route failed")
    if replay_report["route_ids"] != [first_route_id, second_route_id]:
        raise RuntimeError("held-out replay did not execute both route edges")
    if replay_report["landmark_replans"] != 1:
        raise RuntimeError("held-out replay did not replan at the midpoint")
    if len(replay_report["boundary_matches"]) != 2 or not all(
        item["matched_landmark_id"] == item["expected_landmark_id"]
        for item in replay_report["boundary_matches"]
    ):
        raise RuntimeError("held-out replay missed a route boundary landmark")
    if replay_report["boundary_matches"][0]["matched_landmark_id"] != midpoint_id:
        raise RuntimeError("first route edge did not terminate at the midpoint")

    report = {
        "preregistration_sha256": digest(output / "preregistered.json"),
        "runtime_manifest_sha256": digest(output / "runtime-manifest.json"),
        "source_artifact_manifest_sha256": digest(
            output / "source-artifact-manifest.json"
        ),
        "upstream_audit_sha256": digest(
            upstream_copy / "integrity-audit.json"
        ),
        "map_sha256": digest(map_path),
        "map_format_version": topology.FORMAT_VERSION,
        "map_node_count": topology.graph.number_of_nodes(),
        "map_edge_count": topology.graph.number_of_edges(),
        "start_signature": start_signature,
        "midpoint_signature": midpoint_signature,
        "target_signature": target_signature,
        "midpoint_id": midpoint_id,
        "target_id": target_id,
        "split_index": args.split_index,
        "route_ids": [first_route_id, second_route_id],
        "first_edge_program": programs[0][:args.split_index],
        "second_edge_program": programs[1][args.split_index:],
        "composed_program": plan.action_roles,
        "source_results": upstream_report["source_results"],
        "source_success_rate": 1.0,
        "source_control_bindings_distinct": (
            load(copied_sources[0] / "roles.json")
            != load(copied_sources[1] / "roles.json")
        ),
        "no_direct_start_to_goal_edge": not topology.graph.has_edge(
            topology.node_id_for(start_signature), target_id
        ),
        "reverse_route_rejected": True,
        "challenged_second_edge_blocks_plan": True,
        "replay": {
            **replay_case,
            **assessment,
            "route_ids": replay_report["route_ids"],
            "boundary_matches": replay_report["boundary_matches"],
            "landmark_replans": replay_report["landmark_replans"],
            "online_route_exploration_actions": replay_report[
                "online_route_exploration_actions"
            ],
        },
        "replay_success_rate": 1.0,
        "scope": (
            "A three-landmark, two-edge route derived from two independently "
            "permuted source episodes and replayed in real MineStudio with RGB "
            "boundary recognition and replanning at the midpoint. This is one "
            "controlled fixture path, not arbitrary graph navigation or SLAM."
        ),
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
