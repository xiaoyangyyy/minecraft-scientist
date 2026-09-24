"""Learn two visual-layout routes, select held-out branches, and reject unknowns."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def move_program(directory: Path) -> tuple[str, ...]:
    roles = load(directory / "roles.json")
    inverse = {handle: role for role, handle in roles.items()}
    events = [
        json.loads(line)
        for line in (directory / "route" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return tuple(
        inverse[event["executed_action"]["target_id"]]
        for event in events
        if event["executed_action"]["kind"] == "move"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--mapping-seed", type=int, default=1300)
    parser.add_argument("--mapping-seed-stride", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mapping_seed_stride == 0:
        parser.error("mapping seed stride must be nonzero")
    output = args.output or ROOT / f"runs/layout-route-branching-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/layout_route_branching_benchmark.py",
        "environments/minestudio/audit_layout_route_branching.py",
        "environments/minestudio/topological_route_replay_probe.py",
        "environments/minestudio/obstacle_replanning_probe.py",
        "environments/minestudio/direction_discovery_probe.py",
        "environments/minestudio/navigation_probe.py",
        "src/minecraft_scientist/cognitive_map/visual_topology.py",
        "src/minecraft_scientist/protocols/models.py",
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

    source_cases = [
        {
            "case": index,
            "seed": args.seed + index,
            "mapping_seed": args.mapping_seed + index * args.mapping_seed_stride,
            "obstacle_offset": offset,
        }
        for index, offset in enumerate((-1, 1))
    ]
    replay_cases = [
        {
            "case": index,
            "seed": args.seed + 2 + index,
            "mapping_seed": args.mapping_seed
            + (index + 2) * args.mapping_seed_stride,
            "obstacle_offset": offset,
        }
        for index, offset in enumerate((1, -1))
    ]
    settings = {
        "candidate_count": 4,
        "move_frames": 4,
        "match_tolerance": 0.75,
        "ambiguity_margin": 0.25,
        "central_clear_pixels": 200,
        "obstacle_passed_pixels": 80,
        "center_tolerance": 6.0,
        "stall_area_delta": 20,
        "score_threshold": 150.0,
        "max_detour_moves": 6,
        "min_bypass_moves": 7,
        "max_bypass_moves": 12,
        "max_recenter_moves": 8,
        "max_final_moves": 12,
        "minimum_goal_progress": 6.0,
        "maximum_goal_progress": 9.5,
        "minimum_lateral_excursion": 1.5,
        "maximum_lateral_excursion": 4.5,
        "maximum_final_lateral_offset": 0.9,
        "minimum_role_index_variety": 2,
    }
    save(output / "preregistered.json", {
        "source_cases": source_cases,
        "replay_cases": replay_cases,
        "settings": settings,
        "policy": (
            "learn separate functional-role programs for left- and right-offset "
            "central obstacles; select by a quantized coordinate-free visual "
            "landmark; fail closed on an unrecognized intermediate signature"
        ),
        "required_source_success_rate": 1.0,
        "required_replay_success_rate": 1.0,
        "required_unknown_rejections": 1,
    })
    preregistration_sha256 = digest(output / "preregistered.json")

    source_results = []
    topology = VisualTopologicalMap(
        match_tolerance=settings["match_tolerance"],
        ambiguity_margin=settings["ambiguity_margin"],
    )
    target_signature = None
    for case in source_cases:
        directory = output / f"source-{case['case']}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("obstacle_replanning_probe.py")),
            "--seed", str(case["seed"]),
            "--mapping-seed", str(case["mapping_seed"]),
            "--obstacle-offset", str(case["obstacle_offset"]),
            "--move-frames", str(settings["move_frames"]),
            "--central-clear-pixels", str(settings["central_clear_pixels"]),
            "--obstacle-passed-pixels", str(settings["obstacle_passed_pixels"]),
            "--center-tolerance", str(settings["center_tolerance"]),
            "--stall-area-delta", str(settings["stall_area_delta"]),
            "--score-threshold", str(settings["score_threshold"]),
            "--max-detour-moves", str(settings["max_detour_moves"]),
            "--min-bypass-moves", str(settings["min_bypass_moves"]),
            "--max-bypass-moves", str(settings["max_bypass_moves"]),
            "--max-recenter-moves", str(settings["max_recenter_moves"]),
            "--max-final-moves", str(settings["max_final_moves"]),
            "--output", str(directory),
        ]
        subprocess.run(command, check=True, timeout=900)
        report = load(directory / "report.json")
        trajectory = load(directory / "route" / "trajectory.json")
        program = move_program(directory)
        source_signature = layout_landmark_signature(trajectory[0])
        candidate_target = layout_landmark_signature(trajectory[-1])
        if target_signature is None:
            target_signature = candidate_target
        elif candidate_target != target_signature:
            raise RuntimeError("source layouts did not converge to one goal landmark")
        successful = all((
            report["calibration_valid"],
            report["initial_obstacle_detected"],
            report["path_cleared_visually"],
            report["obstacle_passed_visually"],
            report["recentered_visually"],
            report["goal_reached_visually"],
            report["all_move_actions_valid"],
            report["continuous_route_episode"],
            settings["minimum_goal_progress"]
            <= report["evaluator_goal_progress"]
            <= settings["maximum_goal_progress"],
            settings["minimum_lateral_excursion"]
            <= report["evaluator_max_lateral_excursion"]
            <= settings["maximum_lateral_excursion"],
            report["evaluator_final_lateral_offset"]
            <= settings["maximum_final_lateral_offset"],
        ))
        if not successful:
            raise RuntimeError(f"source layout failed: {directory}")
        route_id = topology.record_route(
            source_signature,
            candidate_target,
            program,
            success=True,
            cost=len(program),
        )
        source_results.append({
            **case,
            "successful": successful,
            "source_signature": source_signature,
            "target_signature": candidate_target,
            "source_landmark_id": topology.node_id_for(source_signature),
            "route_id": route_id,
            "program": program,
            "detour_role": report["detour_role"],
            "report_sha256": digest(directory / "report.json"),
            "trajectory_sha256": digest(directory / "route" / "trajectory.json"),
            "events_sha256": digest(directory / "route" / "events.jsonl"),
            "evaluator_sha256": digest(directory / "evaluator-audit.json"),
        })

    assert target_signature is not None
    if len({item["source_landmark_id"] for item in source_results}) != 2:
        raise RuntimeError("layout signatures collapsed into one source landmark")
    if len({item["detour_role"] for item in source_results}) != 2:
        raise RuntimeError("source layouts did not learn distinct detour roles")
    target_id = topology.node_id_for(target_signature)
    map_path = output / "visual-topology.json"
    topology.save(map_path)

    left, right = sorted(
        (tuple(item["source_signature"]) for item in source_results),
        key=lambda signature: signature[1],
    )
    if right[1] - left[1] < 2:
        raise RuntimeError("source layout bins are not separated enough for a negative control")
    unknown_signature = list(left)
    unknown_signature[1] = (left[1] + right[1]) // 2
    unknown_signature = tuple(unknown_signature)
    unknown_plan = topology.plan(unknown_signature, target_id)
    if unknown_plan is not None:
        raise RuntimeError("unrecognized intermediate layout was not rejected")

    replay_results = []
    source_by_offset = {
        item["obstacle_offset"]: item for item in source_results
    }
    for case in replay_cases:
        directory = output / f"replay-{case['case']}"
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("topological_route_replay_probe.py")),
            "--seed", str(case["seed"]),
            "--mapping-seed", str(case["mapping_seed"]),
            "--obstacle-offset", str(case["obstacle_offset"]),
            "--signature-mode", "layout",
            "--map", str(map_path),
            "--target-id", target_id,
            "--move-frames", str(settings["move_frames"]),
            "--score-threshold", str(settings["score_threshold"]),
            "--center-tolerance", str(settings["center_tolerance"]),
            "--obstacle-passed-pixels", str(settings["obstacle_passed_pixels"]),
            "--output", str(directory),
        ], check=True, timeout=900)
        report = load(directory / "report.json")
        expected = source_by_offset[case["obstacle_offset"]]
        progress = report["evaluator_goal_progress"]
        excursion = report["evaluator_max_lateral_excursion"]
        final_offset = report["evaluator_final_lateral_offset"]
        successful = all((
            report["calibration_valid"],
            report["matched_source_id"] == expected["source_landmark_id"],
            report["route_ids"] == [expected["route_id"]],
            report["action_roles"] == list(expected["program"]),
            report["initial_obstacle_detected"],
            report["obstacle_passed_visually"],
            report["recentered_visually"],
            report["goal_reached_visually"],
            report["all_move_actions_valid"],
            report["continuous_replay_episode"],
            report["online_route_exploration_actions"] == 0,
            settings["minimum_goal_progress"] <= progress
            <= settings["maximum_goal_progress"],
            settings["minimum_lateral_excursion"] <= excursion
            <= settings["maximum_lateral_excursion"],
            final_offset <= settings["maximum_final_lateral_offset"],
        ))
        if not successful:
            raise RuntimeError(f"held-out layout replay failed: {directory}")
        calibration = load(directory / "calibration.json")
        roles = load(directory / "roles.json")
        role_indices = {
            role: next(
                index for index, item in enumerate(calibration)
                if item["handle_id"] == handle
            )
            for role, handle in roles.items()
        }
        replay_results.append({
            **case,
            "successful": successful,
            "selected_source_landmark_id": report["matched_source_id"],
            "selected_route_id": report["route_ids"][0],
            "role_indices": role_indices,
            "report_sha256": digest(directory / "report.json"),
            "calibration_sha256": digest(directory / "calibration.json"),
            "roles_sha256": digest(directory / "roles.json"),
            "trajectory_sha256": digest(directory / "replay" / "trajectory.json"),
            "events_sha256": digest(directory / "replay" / "events.jsonl"),
            "evaluator_sha256": digest(directory / "evaluator-audit.json"),
            "goal_progress": progress,
            "maximum_lateral_excursion": excursion,
            "final_lateral_offset": final_offset,
        })

    source_success_rate = sum(
        item["successful"] for item in source_results
    ) / len(source_results)
    replay_success_rate = sum(
        item["successful"] for item in replay_results
    ) / len(replay_results)
    role_index_variety = len({
        tuple(sorted(item["role_indices"].items())) for item in replay_results
    })
    report = {
        "preregistration_sha256": preregistration_sha256,
        "map_sha256": digest(map_path),
        "map_format_version": topology.FORMAT_VERSION,
        "target_landmark_id": target_id,
        "sources": source_results,
        "replays": replay_results,
        "unknown_signature": unknown_signature,
        "unknown_signature_rejected": unknown_plan is None,
        "source_success_rate": source_success_rate,
        "replay_success_rate": replay_success_rate,
        "route_branch_count": len(source_results),
        "role_index_variety": role_index_variety,
        "online_route_exploration_actions": 0,
        "scope": (
            "Two color-coded offset-obstacle layouts under fixed camera and "
            "reset-calibrated opaque controls; one synthetic between-layout "
            "visual-signature rejection control."
        ),
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if not all((
        source_success_rate == 1.0,
        replay_success_rate == 1.0,
        unknown_plan is None,
        role_index_variety >= settings["minimum_role_index_variety"],
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
