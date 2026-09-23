"""Run a preregistered real-MineStudio obstacle-replanning suite."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--mapping-seed", type=int, default=910)
    parser.add_argument("--mapping-seed-stride", type=int, default=3)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 2:
        parser.error("at least two independently permuted cases are required")
    if args.mapping_seed_stride == 0:
        parser.error("mapping seed stride must be nonzero")
    output = args.output or ROOT / f"runs/obstacle-replanning-suite-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/obstacle_replanning_benchmark.py",
        "environments/minestudio/obstacle_replanning_probe.py",
        "environments/minestudio/audit_obstacle_replanning.py",
        "environments/minestudio/direction_discovery_probe.py",
        "environments/minestudio/navigation_probe.py",
        "src/minecraft_scientist/protocols/models.py",
        "src/minecraft_scientist/minecraft/minestudio.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    save(output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("minestudio", "numpy", "pydantic", "pillow")
        },
        "source_sha256": source_hashes,
    })
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    cases = [
        {
            "seed": args.seed + index,
            "mapping_seed": args.mapping_seed + index * args.mapping_seed_stride,
        }
        for index in range(args.cases)
    ]
    settings = {
        "candidate_count": 4,
        "move_frames": 4,
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
        "cases": cases,
        "settings": settings,
        "policy": (
            "classify four opaque controls from reset-identical marker geometry; "
            "in one continuous execution episode detect central fixture-color "
            "occupancy, detour until clear, satisfy a calibrated seven-move bypass "
            "safety gate, recenter from marker centroid, and stop on salience"
        ),
        "required_success_rate": 1.0,
    })
    preregistration_sha256 = digest(output / "preregistered.json")

    results = []
    for index, case in enumerate(cases):
        directory = output / f"case-{index}"
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("obstacle_replanning_probe.py")),
            "--seed", str(case["seed"]),
            "--mapping-seed", str(case["mapping_seed"]),
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
        ], check=True, timeout=900)
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        calibration_path = directory / "calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        role_indices = {
            role: next(
                item_index
                for item_index, item in enumerate(calibration)
                if item["handle_id"] == handle
            )
            for role, handle in report["roles"].items()
        }
        progress = report["evaluator_goal_progress"]
        excursion = report["evaluator_max_lateral_excursion"]
        final_offset = report["evaluator_final_lateral_offset"]
        successful = all((
            report["calibration_valid"],
            report["pixel_goal_role_matches_evaluator"],
            report["initial_obstacle_detected"],
            report["path_cleared_visually"],
            report["obstacle_passed_visually"],
            report["recentered_visually"],
            report["goal_reached_visually"],
            report["all_move_actions_valid"],
            report["continuous_route_episode"],
            report["evaluator_passed_obstacle_plane"],
            settings["minimum_goal_progress"]
            <= progress <= settings["maximum_goal_progress"],
            settings["minimum_lateral_excursion"]
            <= excursion <= settings["maximum_lateral_excursion"],
            final_offset <= settings["maximum_final_lateral_offset"],
        ))
        if not successful:
            raise RuntimeError(f"obstacle-replanning case failed: {directory}")
        results.append({
            "case": index,
            "seed": case["seed"],
            "mapping_seed": case["mapping_seed"],
            "report_sha256": digest(report_path),
            "calibration_sha256": digest(calibration_path),
            "roles_sha256": digest(directory / "roles.json"),
            "trajectory_sha256": digest(directory / "route" / "trajectory.json"),
            "events_sha256": digest(directory / "route" / "events.jsonl"),
            "evaluator_sha256": digest(directory / "evaluator-audit.json"),
            "successful": successful,
            "role_indices": role_indices,
            "goal_progress": progress,
            "maximum_lateral_excursion": excursion,
            "final_lateral_offset": final_offset,
        })

    success_rate = sum(item["successful"] for item in results) / len(results)
    role_index_variety = len({
        tuple(sorted(item["role_indices"].items())) for item in results
    })
    report = {
        "preregistration_sha256": preregistration_sha256,
        "cases": results,
        "success_rate": success_rate,
        "role_index_variety": role_index_variety,
        "pixel_only_obstacle_replanning": True,
        "scope": (
            "Visible fixture-color obstacle, reset-calibrated motor effects, and "
            "one continuous route episode per case."
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
