"""Independently audit persisted visual-topology learning and held-out reuse."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from obstacle_replanning_probe import classify_handles, frame_features
from PIL import Image
from topological_route_replay_probe import landmark_signature

from minecraft_scientist.cognitive_map import VisualTopologicalMap


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    return [
        json.loads(line) for line in text.splitlines() if line.strip()
    ], text


def close(left, right, *, tolerance: float = 1e-9) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance=tolerance)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0)
    return left == right


def recompute_trajectory(directory: Path, trajectory: list[dict]) -> list[dict]:
    result = []
    for item in trajectory:
        frame_path = directory / f"frame-{item['observation_index']:02d}.png"
        frame = np.asarray(Image.open(frame_path))
        role_field = (
            {"phase": item["phase"]}
            if "phase" in item
            else {"action_role": item["action_role"]}
        )
        result.append({
            "observation_index": item["observation_index"],
            **role_field,
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            **frame_features(frame),
        })
    return result


def position_delta(positions: list[dict], axis: str) -> float | None:
    if not positions or axis not in positions[0] or axis not in positions[-1]:
        return None
    return float(positions[-1][axis] - positions[0][axis])


def path_metrics(positions: list[dict]) -> tuple[float | None, float | None, float | None, bool]:
    if not positions:
        return None, None, None, False
    start = positions[0]
    relative = [
        (item["xpos"] - start["xpos"], item["zpos"] - start["zpos"])
        for item in positions
        if all(key in item and key in start for key in ("xpos", "zpos"))
    ]
    offsets = [abs(x) for x, _ in relative]
    avoids = bool(relative) and not any(
        -1.5 < x < 1.5 and 2.5 < z < 4.5 for x, z in relative
    )
    return (
        position_delta(positions, "zpos"),
        max(offsets, default=None),
        offsets[-1] if offsets else None,
        avoids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    prereg_path = root / "preregistered.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    suite = json.loads((root / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    settings = prereg["settings"]

    source_snapshot = []
    for relative, expected in sorted(manifest["source_sha256"].items()):
        path = root / "source-snapshot" / relative
        actual = digest(path) if path.is_file() else None
        source_snapshot.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": actual == expected,
        })

    map_path = root / "visual-topology.json"
    topology = VisualTopologicalMap.load(map_path)
    map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    source_directory = root / "source-learning"
    source_report_path = source_directory / "report.json"
    source_events_path = source_directory / "route" / "events.jsonl"
    source_trajectory_path = source_directory / "route" / "trajectory.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_roles = json.loads(
        (source_directory / "roles.json").read_text(encoding="utf-8")
    )
    source_events, source_event_text = load_events(source_events_path)
    source_trajectory = json.loads(
        source_trajectory_path.read_text(encoding="utf-8")
    )
    source_moves = [
        event for event in source_events
        if event.get("executed_action", {}).get("kind") == "move"
    ]
    inverse_source_roles = {
        handle: role for role, handle in source_roles.items()
    }
    derived_program = tuple(
        inverse_source_roles[event["executed_action"]["target_id"]]
        for event in source_moves
    )
    source_signature = landmark_signature(source_trajectory[0])
    target_signature = landmark_signature(source_trajectory[-1])
    source_id = topology.node_id_for(source_signature)
    target_id = topology.node_id_for(target_signature)
    learned_plan = topology.plan(source_signature, target_id)
    source_positions = json.loads(
        (source_directory / "evaluator-audit.json").read_text(encoding="utf-8")
    )["route"]["positions"]
    source_progress, source_excursion, source_offset, source_avoids = path_metrics(
        source_positions
    )
    source_public = "\n".join((
        source_report_path.read_text(encoding="utf-8"),
        (source_directory / "roles.json").read_text(encoding="utf-8"),
        source_event_text,
        source_trajectory_path.read_text(encoding="utf-8"),
    ))
    forbidden_tokens = (
        "location_stats", "xpos", "ypos", "zpos", '"control"',
        ': "forward"', ': "back"', ': "left"', ': "right"',
    )
    source_audit = {
        "seed_matches_preregistration": source_report["seed"]
        == prereg["source_case"]["seed"],
        "mapping_seed_matches_preregistration": source_report["mapping_seed"]
        == prereg["source_case"]["mapping_seed"],
        "report_hash_matches_suite": digest(source_report_path)
        == suite["source"]["report_sha256"],
        "events_hash_matches_suite": digest(source_events_path)
        == suite["source"]["events_sha256"],
        "trajectory_hash_matches_suite": digest(source_trajectory_path)
        == suite["source"]["trajectory_sha256"],
        "trajectory_features_recomputed": close(
            source_trajectory,
            recompute_trajectory(source_directory / "route", source_trajectory),
        ),
        "program_derived_from_successful_move_events": bool(source_moves)
        and all(
            event["intervention_compliance"] is True
            and event["event"]["success"] is True
            and event["executed_action"]["ticks"] == settings["move_frames"]
            for event in source_moves
        ),
        "map_program_matches_source_events": learned_plan is not None
        and learned_plan.action_roles == derived_program,
        "landmark_ids_recomputed": source_id == suite["source_landmark_id"]
        and target_id == suite["target_landmark_id"],
        "public_artifacts_have_no_coordinate_or_direction_leak": not any(
            token in source_public for token in forbidden_tokens
        ),
        "map_contains_no_source_motor_handles": not any(
            handle in map_path.read_text(encoding="utf-8")
            for handle in source_roles.values()
        ),
        "source_path_avoids_obstacle_volume": source_avoids,
        "source_metrics_in_range": all((
            source_progress is not None
            and settings["minimum_goal_progress"] <= source_progress
            <= settings["maximum_goal_progress"],
            source_excursion is not None
            and settings["minimum_lateral_excursion"] <= source_excursion
            <= settings["maximum_lateral_excursion"],
            source_offset is not None
            and source_offset <= settings["maximum_final_lateral_offset"],
        )),
    }

    replay_audits = []
    for index, expected_case in enumerate(prereg["replay_cases"]):
        directory = root / f"replay-{index}"
        suite_case = suite["replays"][index]
        report_path = directory / "report.json"
        calibration_path = directory / "calibration.json"
        roles_path = directory / "roles.json"
        trajectory_path = directory / "replay" / "trajectory.json"
        events_path = directory / "replay" / "events.jsonl"
        evaluator_path = directory / "evaluator-audit.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        events, event_text = load_events(events_path)
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))

        recomputed_calibration = []
        calibration_events_valid = True
        public_parts = [
            report_path.read_text(encoding="utf-8"),
            calibration_path.read_text(encoding="utf-8"),
            roles_path.read_text(encoding="utf-8"),
            trajectory_path.read_text(encoding="utf-8"),
            event_text,
            map_path.read_text(encoding="utf-8"),
        ]
        for trial_index, item in enumerate(calibration):
            trial = directory / "calibration" / f"trial-{trial_index}"
            before = frame_features(np.asarray(Image.open(trial / "before.png")))
            after = frame_features(np.asarray(Image.open(trial / "after.png")))
            recomputed_calibration.append({
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
            })
            trial_events, trial_text = load_events(trial / "events.jsonl")
            public_parts.append(trial_text)
            moves = [
                event for event in trial_events
                if event.get("executed_action", {}).get("kind") == "move"
            ]
            calibration_events_valid = calibration_events_valid and (
                len(moves) == 1
                and moves[0]["executed_action"]["target_id"] == item["handle_id"]
                and moves[0]["executed_action"]["ticks"] == settings["move_frames"]
                and moves[0]["intervention_compliance"] is True
                and moves[0]["event"]["success"] is True
            )

        initial_signature = landmark_signature(trajectory[0])
        visually_passed = [
            item["observation_index"]
            for item in trajectory[1:]
            if item["action_role"] == "goal"
            and (
                item["obstacle"] is None
                or item["obstacle"]["pixel_count"]
                < settings["obstacle_passed_pixels"]
            )
        ]
        plan = topology.plan(initial_signature, target_id)
        expected_handles = plan.bind(roles) if plan is not None else ()
        move_events = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        actual_handles = tuple(
            event["executed_action"]["target_id"] for event in move_events
        )
        actual_roles = tuple(item["action_role"] for item in trajectory[1:])
        positions = evaluator["replay"]["positions"]
        progress, excursion, final_offset, avoids = path_metrics(positions)
        public_text = "\n".join(public_parts)
        expected_fixture = {
            "goal_axis": "positive_z",
            "obstacle_center_x_relative_to_start": [-1.5, 1.5],
            "obstacle_center_z_relative_to_start": [2.5, 4.5],
        }
        replay_audits.append({
            "case": index,
            "seed_matches_preregistration": report["seed"]
            == expected_case["seed"],
            "mapping_seed_matches_preregistration": report["mapping_seed"]
            == expected_case["mapping_seed"] == evaluator["mapping_seed"],
            "hashes_match_suite": all((
                digest(report_path) == suite_case["report_sha256"],
                digest(calibration_path) == suite_case["calibration_sha256"],
                digest(roles_path) == suite_case["roles_sha256"],
                digest(trajectory_path) == suite_case["trajectory_sha256"],
                digest(events_path) == suite_case["events_sha256"],
                digest(evaluator_path) == suite_case["evaluator_sha256"],
            )),
            "map_hash_matches": report["map_sha256"] == suite["map_sha256"]
            == digest(map_path),
            "calibration_features_recomputed": close(
                calibration, recomputed_calibration
            ),
            "roles_recomputed_from_pixels": roles
            == classify_handles(recomputed_calibration),
            "calibration_events_valid": calibration_events_valid,
            "trajectory_features_recomputed": close(
                trajectory,
                recompute_trajectory(directory / "replay", trajectory),
            ),
            "source_landmark_matched": topology.match(initial_signature)
            == source_id == report["matched_source_id"],
            "persisted_program_selected": plan is not None
            and plan.action_roles == derived_program
            and tuple(report["action_roles"]) == derived_program,
            "program_rebound_to_current_handles": actual_handles
            == expected_handles,
            "trajectory_roles_match_program": actual_roles == derived_program,
            "temporal_obstacle_pass_recomputed": bool(visually_passed)
            and report["obstacle_passed_visually"] is True
            and report["obstacle_passed_observation"] == visually_passed[0],
            "route_events_valid": len(move_events) == len(derived_program)
            and all(
                event["intervention_compliance"] is True
                and event["event"]["success"] is True
                and event["executed_action"]["ticks"] == settings["move_frames"]
                for event in move_events
            ),
            "continuous_replay_episode": {
                event["provenance"]["episode"] for event in events
            } == {1},
            "no_online_route_exploration": report[
                "online_route_exploration_actions"
            ] == suite["online_route_exploration_actions"] == 0,
            "public_artifacts_have_no_coordinate_or_direction_leak": not any(
                token in public_text for token in forbidden_tokens
            ),
            "fixture_bounds_match": evaluator["fixture"] == expected_fixture,
            "privileged_positions_are_evaluator_only": bool(positions)
            and "location_stats" in evaluator["replay"]["raw_metadata_keys"],
            "evaluator_path_avoids_obstacle_volume": avoids,
            "goal_progress_matches_report": progress is not None
            and math.isclose(
                progress, report["evaluator_goal_progress"], abs_tol=1e-9
            ),
            "goal_progress_in_range": progress is not None
            and settings["minimum_goal_progress"] <= progress
            <= settings["maximum_goal_progress"],
            "lateral_excursion_matches_report": excursion is not None
            and math.isclose(
                excursion,
                report["evaluator_max_lateral_excursion"],
                abs_tol=1e-9,
            ),
            "lateral_excursion_in_range": excursion is not None
            and settings["minimum_lateral_excursion"] <= excursion
            <= settings["maximum_lateral_excursion"],
            "final_offset_matches_report": final_offset is not None
            and math.isclose(
                final_offset,
                report["evaluator_final_lateral_offset"],
                abs_tol=1e-9,
            ),
            "final_offset_in_range": final_offset is not None
            and final_offset <= settings["maximum_final_lateral_offset"],
        })

    ignored = {"case"}
    source_valid = all(value is True for value in source_audit.values())
    replay_success_rate = sum(
        all(value is True for key, value in item.items() if key not in ignored)
        for item in replay_audits
    ) / len(replay_audits)
    role_mappings = [
        tuple(sorted(item["role_indices"].items())) for item in suite["replays"]
    ]
    audit = {
        "run": str(root),
        "preregistration_intact": digest(prereg_path)
        == suite["preregistration_sha256"],
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot)
        and all(item["valid"] for item in source_snapshot),
        "map_schema_strictly_loaded": map_payload["format_version"]
        in {1, VisualTopologicalMap.FORMAT_VERSION},
        "map_hash_matches_suite": digest(map_path) == suite["map_sha256"],
        "source": source_audit,
        "source_valid": source_valid,
        "replays": replay_audits,
        "replay_success_rate": replay_success_rate,
        "role_index_variety": len(set(role_mappings)),
        "role_index_variety_valid": len(set(role_mappings))
        >= settings["minimum_role_index_variety"],
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
        audit["map_schema_strictly_loaded"],
        audit["map_hash_matches_suite"],
        audit["source_valid"],
        audit["replay_success_rate"]
        == prereg["required_replay_success_rate"],
        audit["role_index_variety_valid"],
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
