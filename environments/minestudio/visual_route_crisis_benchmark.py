"""Trigger and repair a visual-route crisis in the real MineStudio fixture."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from minecraft_scientist.cognitive_map import (
    VisualRouteStatus,
    VisualTopologicalMap,
)
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_program(directory: Path) -> tuple[str, ...]:
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


def successful_source(directory: Path, settings: dict) -> bool:
    report = load(directory / "report.json")
    evaluator = load(directory / "evaluator-audit.json")
    progress, excursion, final_offset, avoids = path_metrics(
        evaluator["route"]["positions"], report["obstacle_offset"]
    )
    return all((
        report["calibration_valid"],
        report["initial_obstacle_detected"],
        report["path_cleared_visually"],
        report["obstacle_passed_visually"],
        report["recentered_visually"],
        report["goal_reached_visually"],
        report["all_move_actions_valid"],
        report["continuous_route_episode"],
        progress is not None
        and settings["minimum_goal_progress"] <= progress
        <= settings["maximum_goal_progress"],
        excursion is not None
        and settings["minimum_lateral_excursion"] <= excursion
        <= settings["maximum_lateral_excursion"],
        final_offset is not None
        and final_offset <= settings["maximum_final_lateral_offset"],
        avoids,
    ))


def replay_assessment(directory: Path, settings: dict) -> dict:
    report = load(directory / "report.json")
    evaluator = load(directory / "evaluator-audit.json")
    progress, excursion, final_offset, avoids = path_metrics(
        evaluator["replay"]["positions"], report["obstacle_offset"]
    )
    successful = all((
        report["calibration_valid"],
        report["initial_obstacle_detected"],
        report["obstacle_passed_visually"],
        report["recentered_visually"],
        report["goal_reached_visually"],
        report["all_move_actions_valid"],
        report["continuous_replay_episode"],
        progress is not None
        and settings["minimum_goal_progress"] <= progress
        <= settings["maximum_goal_progress"],
        excursion is not None
        and settings["minimum_lateral_excursion"] <= excursion
        <= settings["maximum_lateral_excursion"],
        final_offset is not None
        and final_offset <= settings["maximum_final_lateral_offset"],
        avoids,
    ))
    return {
        "successful": successful,
        "goal_progress": progress,
        "maximum_lateral_excursion": excursion,
        "final_lateral_offset": final_offset,
        "path_avoids_obstacle": avoids,
        "matched_source_id": report["matched_source_id"],
        "selected_route_ids": report["route_ids"],
        "source_signature": report["source_signature"],
    }


def run_source(case: dict, output: Path, settings: dict) -> None:
    subprocess.run([
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
        "--output", str(output),
    ], check=True, timeout=900)


def run_replay(
    case: dict,
    output: Path,
    settings: dict,
    map_path: Path,
    target_id: str,
) -> None:
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
        "--output", str(output),
    ], check=True, timeout=900)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=140)
    parser.add_argument("--mapping-seed", type=int, default=1400)
    parser.add_argument("--mapping-seed-stride", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mapping_seed_stride == 0:
        parser.error("mapping seed stride must be nonzero")
    output = args.output or ROOT / f"runs/visual-route-crisis-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    settings = {
        "move_frames": 4,
        "match_tolerance": 8.25,
        "ambiguity_margin": 0.0,
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
    }
    source_cases = [
        {
            "seed": args.seed,
            "mapping_seed": args.mapping_seed
            + index * args.mapping_seed_stride,
            "obstacle_offset": -1,
        }
        for index in range(2)
    ]
    failed_case = {
        "seed": args.seed + 2,
        "mapping_seed": args.mapping_seed + 2 * args.mapping_seed_stride,
        "obstacle_offset": 1,
    }
    repair_case = {
        "seed": args.seed + 1,
        "mapping_seed": args.mapping_seed + 3 * args.mapping_seed_stride,
        "obstacle_offset": 1,
    }
    validation_case = {
        "seed": args.seed + 2,
        "mapping_seed": args.mapping_seed + 4 * args.mapping_seed_stride,
        "obstacle_offset": 1,
    }
    preregistration = {
        "source_cases": source_cases,
        "failed_replay_case": failed_case,
        "repair_case": repair_case,
        "shadow_validation_case": validation_case,
        "settings": settings,
        "policy": (
            "merge two visually close layout signatures to expose an alias; "
            "challenge the high-confidence active route only after a failed "
            "held-out replay; learn a distinct repair and supersede the old "
            "route only after an independent successful shadow replay"
        ),
        "required_source_successes": 2,
        "required_failed_replays": 1,
        "required_shadow_successes": 1,
    }
    save(output / "preregistered.json", preregistration)

    source_files = (
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
    hashes = {name: digest(ROOT / name) for name in source_files}
    save(output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("minestudio", "networkx", "numpy", "pydantic", "pillow")
        },
        "source_sha256": hashes,
    })
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    topology = VisualTopologicalMap(
        match_tolerance=settings["match_tolerance"],
        ambiguity_margin=settings["ambiguity_margin"],
    )
    source_results = []
    old_route_id = None
    target_signature = None
    source_signature = None
    for index, case in enumerate(source_cases):
        directory = output / f"source-{index}"
        run_source(case, directory, settings)
        if not successful_source(directory, settings):
            raise RuntimeError(f"source route failed: {directory}")
        trajectory = load(directory / "route" / "trajectory.json")
        candidate_source = layout_landmark_signature(trajectory[0])
        candidate_target = layout_landmark_signature(trajectory[-1])
        program = route_program(directory)
        if source_signature is None:
            source_signature = candidate_source
            target_signature = candidate_target
        elif (candidate_source, candidate_target) != (
            source_signature,
            target_signature,
        ):
            raise RuntimeError("source route landmarks were not stable")
        route_id = topology.record_route(
            candidate_source,
            candidate_target,
            program,
            success=True,
            cost=len(program),
        )
        if old_route_id is None:
            old_route_id = route_id
        elif route_id != old_route_id:
            raise RuntimeError("source route program was not stable")
        source_results.append({
            **case,
            "successful": True,
            "source_signature": candidate_source,
            "target_signature": candidate_target,
            "program": program,
            "route_id": route_id,
            "report_sha256": digest(directory / "report.json"),
        })

    assert old_route_id is not None
    assert source_signature is not None
    assert target_signature is not None
    if topology.get_route(old_route_id).confidence < 0.75:
        raise RuntimeError("source route did not reach preregistered confidence")
    target_id = topology.node_id_for(target_signature)
    prechange_map = output / "map-prechange.json"
    topology.save(prechange_map)

    failed_directory = output / "failed-replay"
    run_replay(failed_case, failed_directory, settings, prechange_map, target_id)
    failed_assessment = replay_assessment(failed_directory, settings)
    if failed_assessment["successful"]:
        raise RuntimeError("mutated layout did not produce a route counterexample")
    changed_signature = tuple(failed_assessment["source_signature"])
    if topology.match(changed_signature) != topology.node_id_for(source_signature):
        raise RuntimeError("changed layout did not exercise the aliased landmark")
    challenged = topology.record_outcome(
        old_route_id,
        success=False,
        failure_reason="held-out visual replay failed after layout mutation",
    )
    if challenged.status is not VisualRouteStatus.CHALLENGED:
        raise RuntimeError("high-confidence failure did not challenge the route")
    challenged_map = output / "map-challenged.json"
    topology.save(challenged_map)
    if topology.plan(changed_signature, target_id) is not None:
        raise RuntimeError("challenged route remained available for planning")

    repair_directory = output / "repair-source"
    run_source(repair_case, repair_directory, settings)
    if not successful_source(repair_directory, settings):
        raise RuntimeError("online repair route failed")
    repair_trajectory = load(repair_directory / "route" / "trajectory.json")
    repair_source = layout_landmark_signature(repair_trajectory[0])
    repair_target = layout_landmark_signature(repair_trajectory[-1])
    repair_program = route_program(repair_directory)
    if topology.match(repair_source) != topology.node_id_for(source_signature):
        raise RuntimeError("repair did not attach to the aliased source landmark")
    if repair_target != target_signature:
        raise RuntimeError("repair did not reach the existing goal landmark")
    replacement_id = topology.record_route(
        repair_source,
        repair_target,
        repair_program,
        success=True,
        cost=len(repair_program),
    )
    if replacement_id == old_route_id:
        raise RuntimeError("repair did not create a distinct route program")
    candidate_map = output / "map-candidate.json"
    topology.save(candidate_map)

    validation_directory = output / "shadow-validation"
    run_replay(
        validation_case,
        validation_directory,
        settings,
        candidate_map,
        target_id,
    )
    validation = replay_assessment(validation_directory, settings)
    if not validation["successful"]:
        raise RuntimeError("replacement failed independent shadow validation")
    if validation["selected_route_ids"] != [replacement_id]:
        raise RuntimeError("shadow validation did not select the replacement")
    topology.record_outcome(
        replacement_id,
        success=True,
        cost=len(repair_program),
    )
    superseded, replacement = topology.supersede_route(
        old_route_id,
        replacement_id,
        reason="replacement passed independent visual shadow validation",
    )
    final_map = output / "visual-topology.json"
    topology.save(final_map)
    final_plan = topology.plan(changed_signature, target_id)
    if final_plan is None or final_plan.route_ids != (replacement_id,):
        raise RuntimeError("final map did not promote the replacement route")

    report = {
        "preregistration_sha256": digest(output / "preregistered.json"),
        "runtime_manifest_sha256": digest(output / "runtime-manifest.json"),
        "map_sha256": digest(final_map),
        "map_format_version": topology.FORMAT_VERSION,
        "source_results": source_results,
        "source_route_confidence_before_failure": 0.75,
        "failed_replay": {**failed_case, **failed_assessment},
        "failure_detected": not failed_assessment["successful"],
        "challenged_route_excluded": True,
        "repair": {
            **repair_case,
            "successful": True,
            "program": repair_program,
            "route_id": replacement_id,
            "report_sha256": digest(repair_directory / "report.json"),
        },
        "shadow_validation": {**validation_case, **validation},
        "old_route": superseded.model_dump(mode="json"),
        "replacement_route": replacement.model_dump(mode="json"),
        "final_selected_route_ids": final_plan.route_ids,
        "source_and_mutation_signatures": [source_signature, changed_signature],
        "signature_alias_distance": sum(
            (left - right) ** 2
            for left, right in zip(source_signature, changed_signature, strict=True)
        ) ** 0.5,
        "online_repair_experiments": 1,
        "scope": (
            "Real MineStudio RGB evidence in two color-coded mirror layouts; an "
            "intentional coarse landmark tolerance creates one perceptual alias "
            "to exercise route challenge, repair, shadow validation and immutable "
            "supersession. This is not natural-scene change detection or SLAM."
        ),
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
