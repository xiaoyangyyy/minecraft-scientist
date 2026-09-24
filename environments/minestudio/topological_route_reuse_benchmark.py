"""Learn one visual topological route and reuse it in held-out episodes."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from topological_route_replay_probe import landmark_signature

from minecraft_scientist.cognitive_map import VisualTopologicalMap

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mapping-seed", type=int, default=1000)
    parser.add_argument("--mapping-seed-stride", type=int, default=3)
    parser.add_argument("--replay-cases", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replay_cases < 2:
        parser.error("at least two held-out replay cases are required")
    if args.mapping_seed_stride == 0:
        parser.error("mapping seed stride must be nonzero")
    output = args.output or ROOT / f"runs/topological-route-reuse-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/topological_route_reuse_benchmark.py",
        "environments/minestudio/topological_route_replay_probe.py",
        "environments/minestudio/audit_topological_route_reuse.py",
        "environments/minestudio/obstacle_replanning_probe.py",
        "environments/minestudio/direction_discovery_probe.py",
        "environments/minestudio/navigation_probe.py",
        "src/minecraft_scientist/cognitive_map/visual_topology.py",
        "src/minecraft_scientist/protocols/models.py",
        "src/minecraft_scientist/minecraft/minestudio.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {
        name: digest(ROOT / name) for name in source_files
    }
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

    source_case = {"seed": args.seed, "mapping_seed": args.mapping_seed}
    replay_cases = [
        {
            "seed": args.seed + index + 1,
            "mapping_seed": args.mapping_seed
            + (index + 1) * args.mapping_seed_stride,
        }
        for index in range(args.replay_cases)
    ]
    settings = {
        "candidate_count": 4,
        "move_frames": 4,
        "match_tolerance": 2.0,
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
        "minimum_lateral_excursion": 2.0,
        "maximum_lateral_excursion": 4.5,
        "maximum_final_lateral_offset": 0.9,
        "minimum_role_index_variety": 2,
    }
    save(output / "preregistered.json", {
        "source_case": source_case,
        "replay_cases": replay_cases,
        "settings": settings,
        "policy": (
            "learn one continuous RGB obstacle-detour route; encode only "
            "quantized appearances and functional action roles; reload the map "
            "and replay its program in independently permuted held-out episodes"
        ),
        "required_replay_success_rate": 1.0,
    })
    preregistration_sha256 = digest(output / "preregistered.json")

    source_directory = output / "source-learning"
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("obstacle_replanning_probe.py")),
        "--seed", str(source_case["seed"]),
        "--mapping-seed", str(source_case["mapping_seed"]),
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
        "--output", str(source_directory),
    ], check=True, timeout=900)

    source_report = json.loads(
        (source_directory / "report.json").read_text(encoding="utf-8")
    )
    if not all((
        source_report["calibration_valid"],
        source_report["goal_reached_visually"],
        source_report["evaluator_passed_obstacle_plane"],
        source_report["evaluator_final_lateral_offset"]
        <= settings["maximum_final_lateral_offset"],
    )):
        raise RuntimeError("source route did not satisfy the preregistered gate")
    roles = json.loads(
        (source_directory / "roles.json").read_text(encoding="utf-8")
    )
    inverse_roles = {handle: role for role, handle in roles.items()}
    move_events = [
        event for event in load_events(source_directory / "route" / "events.jsonl")
        if event["executed_action"]["kind"] == "move"
    ]
    action_roles = tuple(
        inverse_roles[event["executed_action"]["target_id"]]
        for event in move_events
    )
    trajectory = json.loads(
        (source_directory / "route" / "trajectory.json").read_text(
            encoding="utf-8"
        )
    )
    source_signature = landmark_signature(trajectory[0])
    target_signature = landmark_signature(trajectory[-1])
    topology = VisualTopologicalMap(
        match_tolerance=settings["match_tolerance"]
    )
    route_id = topology.record_route(
        source_signature,
        target_signature,
        action_roles,
        success=True,
        cost=len(action_roles),
    )
    target_id = topology.node_id_for(target_signature)
    map_path = output / "visual-topology.json"
    topology.save(map_path)

    replay_results = []
    for index, case in enumerate(replay_cases):
        directory = output / f"replay-{index}"
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("topological_route_replay_probe.py")),
            "--seed", str(case["seed"]),
            "--mapping-seed", str(case["mapping_seed"]),
            "--map", str(map_path),
            "--target-id", target_id,
            "--move-frames", str(settings["move_frames"]),
            "--score-threshold", str(settings["score_threshold"]),
            "--center-tolerance", str(settings["center_tolerance"]),
            "--obstacle-passed-pixels", str(settings["obstacle_passed_pixels"]),
            "--output", str(directory),
        ], check=True, timeout=900)
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        progress = report["evaluator_goal_progress"]
        excursion = report["evaluator_max_lateral_excursion"]
        final_offset = report["evaluator_final_lateral_offset"]
        successful = all((
            report["calibration_valid"],
            report["matched_source_id"] == topology.node_id_for(source_signature),
            report["route_ids"] == [route_id],
            report["action_roles"] == list(action_roles),
            report["initial_obstacle_detected"],
            report["obstacle_passed_visually"],
            report["recentered_visually"],
            report["goal_reached_visually"],
            report["all_move_actions_valid"],
            report["continuous_replay_episode"],
            report["online_route_exploration_actions"] == 0,
            settings["minimum_goal_progress"]
            <= progress <= settings["maximum_goal_progress"],
            settings["minimum_lateral_excursion"]
            <= excursion <= settings["maximum_lateral_excursion"],
            final_offset <= settings["maximum_final_lateral_offset"],
        ))
        if not successful:
            raise RuntimeError(f"held-out topological replay failed: {directory}")
        calibration = json.loads(
            (directory / "calibration.json").read_text(encoding="utf-8")
        )
        replay_roles = json.loads(
            (directory / "roles.json").read_text(encoding="utf-8")
        )
        role_indices = {
            role: next(
                item_index for item_index, item in enumerate(calibration)
                if item["handle_id"] == handle
            )
            for role, handle in replay_roles.items()
        }
        replay_results.append({
            "case": index,
            **case,
            "successful": successful,
            "role_indices": role_indices,
            "report_sha256": digest(report_path),
            "calibration_sha256": digest(directory / "calibration.json"),
            "roles_sha256": digest(directory / "roles.json"),
            "trajectory_sha256": digest(directory / "replay" / "trajectory.json"),
            "events_sha256": digest(directory / "replay" / "events.jsonl"),
            "evaluator_sha256": digest(directory / "evaluator-audit.json"),
            "goal_progress": progress,
            "maximum_lateral_excursion": excursion,
            "final_lateral_offset": final_offset,
        })

    success_rate = sum(item["successful"] for item in replay_results) / len(
        replay_results
    )
    role_index_variety = len({
        tuple(sorted(item["role_indices"].items())) for item in replay_results
    })
    report = {
        "preregistration_sha256": preregistration_sha256,
        "source": {
            "seed": source_case["seed"],
            "mapping_seed": source_case["mapping_seed"],
            "report_sha256": digest(source_directory / "report.json"),
            "events_sha256": digest(source_directory / "route" / "events.jsonl"),
            "trajectory_sha256": digest(
                source_directory / "route" / "trajectory.json"
            ),
            "program_length": len(action_roles),
        },
        "map_sha256": digest(map_path),
        "source_landmark_id": topology.node_id_for(source_signature),
        "target_landmark_id": target_id,
        "route_id": route_id,
        "replays": replay_results,
        "replay_success_rate": success_rate,
        "role_index_variety": role_index_variety,
        "map_reuse_count": len(replay_results),
        "online_route_exploration_actions": 0,
        "coordinate_free_visual_topology": True,
        "scope": (
            "One fixture route transferred across held-out world seeds and opaque "
            "control permutations after per-episode visual motor calibration."
        ),
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if (
        success_rate != 1.0
        or role_index_variety < settings["minimum_role_index_variety"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
