"""Independently audit visual-layout branching, replay, and unknown rejection."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from obstacle_replanning_probe import classify_handles, frame_features
from PIL import Image

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], text


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
        action_field = (
            {"phase": item["phase"]}
            if "phase" in item
            else {"action_role": item["action_role"]}
        )
        result.append({
            "observation_index": item["observation_index"],
            **action_field,
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            **frame_features(frame),
        })
    return result


def path_metrics(
    positions: list[dict], obstacle_offset: int
) -> tuple[float | None, float | None, float | None, bool, bool]:
    if not positions:
        return None, None, None, False, False
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
    detours_before = any(abs(x) >= 1.5 and z < 2.5 for x, z in relative)
    progress = relative[-1][1] if relative else None
    return (
        progress,
        max(offsets, default=None),
        offsets[-1] if offsets else None,
        avoids,
        detours_before,
    )


def recompute_calibration(directory: Path, calibration: list[dict]) -> tuple[list[dict], bool, list[str]]:
    result = []
    valid = True
    texts = []
    for index, item in enumerate(calibration):
        trial = directory / "calibration" / f"trial-{index}"
        before = frame_features(np.asarray(Image.open(trial / "before.png")))
        after = frame_features(np.asarray(Image.open(trial / "after.png")))
        result.append({
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
    return result, valid, texts


def public_has_no_leak(parts: list[str]) -> bool:
    text = "\n".join(parts)
    forbidden = (
        "location_stats", "xpos", "ypos", "zpos", '"control"',
        ': "forward"', ': "back"', ': "left"', ': "right"',
        "obstacle_offset",
    )
    return not any(token in text for token in forbidden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    prereg_path = root / "preregistered.json"
    prereg = load(prereg_path)
    suite = load(root / "report.json")
    manifest = load(root / "runtime-manifest.json")
    settings = prereg["settings"]

    snapshots = []
    for relative, expected in sorted(manifest["source_sha256"].items()):
        path = root / "source-snapshot" / relative
        actual = digest(path) if path.is_file() else None
        snapshots.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": actual == expected,
        })

    map_path = root / "visual-topology.json"
    map_text = map_path.read_text(encoding="utf-8")
    map_payload = json.loads(map_text)
    topology = VisualTopologicalMap.load(map_path)
    target_id = suite["target_landmark_id"]

    source_audits = []
    source_programs = {}
    source_signatures = []
    source_handles = []
    for expected, recorded in zip(
        prereg["source_cases"], suite["sources"], strict=True
    ):
        index = expected["case"]
        directory = root / f"source-{index}"
        report_path = directory / "report.json"
        trajectory_path = directory / "route" / "trajectory.json"
        events_path = directory / "route" / "events.jsonl"
        evaluator_path = directory / "evaluator-audit.json"
        roles_path = directory / "roles.json"
        report = load(report_path)
        trajectory = load(trajectory_path)
        events, event_text = load_events(events_path)
        evaluator = load(evaluator_path)
        roles = load(roles_path)
        source_handles.extend(roles.values())
        inverse = {handle: role for role, handle in roles.items()}
        moves = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        program = tuple(
            inverse[event["executed_action"]["target_id"]] for event in moves
        )
        source_signature = layout_landmark_signature(trajectory[0])
        goal_signature = layout_landmark_signature(trajectory[-1])
        source_signatures.append(source_signature)
        source_programs[expected["obstacle_offset"]] = program
        plan = topology.plan(source_signature, target_id)
        positions = evaluator["route"]["positions"]
        progress, excursion, final_offset, avoids, detours_before = path_metrics(
            positions, expected["obstacle_offset"]
        )
        expected_fixture = {
            "goal_axis": "positive_z",
            "obstacle_center_x_relative_to_start": [
                expected["obstacle_offset"] - 1.5,
                expected["obstacle_offset"] + 1.5,
            ],
            "obstacle_center_z_relative_to_start": [2.5, 4.5],
        }
        public_parts = [
            trajectory_path.read_text(encoding="utf-8"),
            event_text,
            roles_path.read_text(encoding="utf-8"),
            (directory / "calibration.json").read_text(encoding="utf-8"),
            map_text,
        ]
        source_audits.append({
            "case": index,
            "seed_matches_preregistration": report["seed"] == expected["seed"],
            "mapping_seed_matches_preregistration": report["mapping_seed"]
            == expected["mapping_seed"] == evaluator["mapping_seed"],
            "layout_matches_preregistration": report["obstacle_offset"]
            == expected["obstacle_offset"],
            "hashes_match_suite": all((
                digest(report_path) == recorded["report_sha256"],
                digest(trajectory_path) == recorded["trajectory_sha256"],
                digest(events_path) == recorded["events_sha256"],
                digest(evaluator_path) == recorded["evaluator_sha256"],
            )),
            "trajectory_features_recomputed": close(
                trajectory,
                recompute_trajectory(directory / "route", trajectory),
            ),
            "program_derived_from_valid_events": bool(moves)
            and all(
                event["executed_action"]["ticks"] == settings["move_frames"]
                and event["intervention_compliance"] is True
                and event["event"]["success"] is True
                for event in moves
            ),
            "signature_and_landmark_recomputed": list(source_signature)
            == recorded["source_signature"]
            and list(goal_signature) == recorded["target_signature"]
            and topology.node_id_for(source_signature)
            == recorded["source_landmark_id"],
            "map_selects_source_program": plan is not None
            and plan.route_ids == (recorded["route_id"],)
            and plan.action_roles == program == tuple(recorded["program"]),
            "detour_role_matches_program": bool(program)
            and program[0] == report["detour_role"]
            == recorded["detour_role"],
            "public_agent_artifacts_have_no_leak": public_has_no_leak(public_parts),
            "fixture_bounds_match": evaluator["fixture"] == expected_fixture,
            "privileged_positions_are_evaluator_only": bool(positions)
            and "location_stats" in evaluator["route"]["raw_metadata_keys"],
            "path_avoids_obstacle_volume": avoids,
            "detour_occurs_before_obstacle_plane": detours_before,
            "metrics_match_report": all((
                progress is not None and math.isclose(
                    progress, report["evaluator_goal_progress"], abs_tol=1e-9
                ),
                excursion is not None and math.isclose(
                    excursion,
                    report["evaluator_max_lateral_excursion"],
                    abs_tol=1e-9,
                ),
                final_offset is not None and math.isclose(
                    final_offset,
                    report["evaluator_final_lateral_offset"],
                    abs_tol=1e-9,
                ),
            )),
            "metrics_in_range": all((
                progress is not None
                and settings["minimum_goal_progress"] <= progress
                <= settings["maximum_goal_progress"],
                excursion is not None
                and settings["minimum_lateral_excursion"] <= excursion
                <= settings["maximum_lateral_excursion"],
                final_offset is not None
                and final_offset <= settings["maximum_final_lateral_offset"],
            )),
        })

    replay_audits = []
    source_by_offset = {
        item["obstacle_offset"]: item for item in suite["sources"]
    }
    for expected, recorded in zip(
        prereg["replay_cases"], suite["replays"], strict=True
    ):
        index = expected["case"]
        directory = root / f"replay-{index}"
        report_path = directory / "report.json"
        calibration_path = directory / "calibration.json"
        roles_path = directory / "roles.json"
        trajectory_path = directory / "replay" / "trajectory.json"
        events_path = directory / "replay" / "events.jsonl"
        evaluator_path = directory / "evaluator-audit.json"
        report = load(report_path)
        calibration = load(calibration_path)
        roles = load(roles_path)
        trajectory = load(trajectory_path)
        events, event_text = load_events(events_path)
        evaluator = load(evaluator_path)
        expected_source = source_by_offset[expected["obstacle_offset"]]
        recomputed_calibration, calibration_valid, calibration_texts = (
            recompute_calibration(directory, calibration)
        )
        source_signature = layout_landmark_signature(trajectory[0])
        plan = topology.plan(source_signature, target_id)
        expected_handles = plan.bind(roles) if plan is not None else ()
        moves = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        actual_handles = tuple(
            event["executed_action"]["target_id"] for event in moves
        )
        actual_roles = tuple(item["action_role"] for item in trajectory[1:])
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
        positions = evaluator["replay"]["positions"]
        progress, excursion, final_offset, avoids, detours_before = path_metrics(
            positions, expected["obstacle_offset"]
        )
        public_parts = [
            calibration_path.read_text(encoding="utf-8"),
            roles_path.read_text(encoding="utf-8"),
            trajectory_path.read_text(encoding="utf-8"),
            event_text,
            *calibration_texts,
            map_text,
        ]
        replay_audits.append({
            "case": index,
            "seed_matches_preregistration": report["seed"] == expected["seed"],
            "mapping_seed_matches_preregistration": report["mapping_seed"]
            == expected["mapping_seed"] == evaluator["mapping_seed"],
            "layout_and_signature_mode_match": report["obstacle_offset"]
            == expected["obstacle_offset"]
            and report["signature_mode"] == "layout",
            "hashes_match_suite": all((
                digest(report_path) == recorded["report_sha256"],
                digest(calibration_path) == recorded["calibration_sha256"],
                digest(roles_path) == recorded["roles_sha256"],
                digest(trajectory_path) == recorded["trajectory_sha256"],
                digest(events_path) == recorded["events_sha256"],
                digest(evaluator_path) == recorded["evaluator_sha256"],
            )),
            "map_hash_matches": report["map_sha256"] == suite["map_sha256"]
            == digest(map_path),
            "calibration_features_recomputed": close(
                calibration, recomputed_calibration
            ),
            "roles_recomputed_from_pixels": roles
            == classify_handles(recomputed_calibration),
            "calibration_events_valid": calibration_valid,
            "trajectory_features_recomputed": close(
                trajectory,
                recompute_trajectory(directory / "replay", trajectory),
            ),
            "correct_layout_branch_selected": plan is not None
            and topology.match(source_signature)
            == expected_source["source_landmark_id"]
            == report["matched_source_id"]
            == recorded["selected_source_landmark_id"]
            and plan.route_ids == (expected_source["route_id"],)
            and report["route_ids"] == [expected_source["route_id"]]
            and recorded["selected_route_id"] == expected_source["route_id"],
            "program_rebound_to_heldout_handles": plan is not None
            and plan.action_roles == source_programs[expected["obstacle_offset"]]
            == actual_roles
            and actual_handles == expected_handles,
            "temporal_obstacle_pass_recomputed": bool(visually_passed)
            and report["obstacle_passed_visually"] is True
            and report["obstacle_passed_observation"] == visually_passed[0],
            "route_events_valid": len(moves) == len(actual_roles)
            and all(
                event["executed_action"]["ticks"] == settings["move_frames"]
                and event["intervention_compliance"] is True
                and event["event"]["success"] is True
                for event in moves
            ),
            "continuous_replay_episode": {
                event["provenance"]["episode"] for event in events
            } == {1},
            "no_online_route_exploration": report[
                "online_route_exploration_actions"
            ] == suite["online_route_exploration_actions"] == 0,
            "public_agent_artifacts_have_no_leak": public_has_no_leak(public_parts),
            "privileged_positions_are_evaluator_only": bool(positions)
            and "location_stats" in evaluator["replay"]["raw_metadata_keys"],
            "path_avoids_obstacle_volume": avoids,
            "detour_occurs_before_obstacle_plane": detours_before,
            "metrics_match_report": all((
                progress is not None and math.isclose(
                    progress, report["evaluator_goal_progress"], abs_tol=1e-9
                ),
                excursion is not None and math.isclose(
                    excursion,
                    report["evaluator_max_lateral_excursion"],
                    abs_tol=1e-9,
                ),
                final_offset is not None and math.isclose(
                    final_offset,
                    report["evaluator_final_lateral_offset"],
                    abs_tol=1e-9,
                ),
            )),
            "metrics_in_range": all((
                progress is not None
                and settings["minimum_goal_progress"] <= progress
                <= settings["maximum_goal_progress"],
                excursion is not None
                and settings["minimum_lateral_excursion"] <= excursion
                <= settings["maximum_lateral_excursion"],
                final_offset is not None
                and final_offset <= settings["maximum_final_lateral_offset"],
            )),
        })

    left, right = sorted(source_signatures, key=lambda signature: signature[1])
    unknown = list(left)
    unknown[1] = (left[1] + right[1]) // 2
    unknown = tuple(unknown)
    unknown_rejected = topology.plan(unknown, target_id) is None
    ignored = {"case"}
    source_success_rate = sum(
        all(value is True for key, value in item.items() if key not in ignored)
        for item in source_audits
    ) / len(source_audits)
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
        "source_snapshot": snapshots,
        "source_snapshot_complete": bool(snapshots)
        and all(item["valid"] for item in snapshots),
        "map_format_and_shape_valid": map_payload["format_version"]
        == suite["map_format_version"]
        and map_payload["format_version"] in {
            2,
            VisualTopologicalMap.FORMAT_VERSION,
        }
        and topology.graph.number_of_nodes() == 3
        and topology.graph.number_of_edges() == 2,
        "map_hash_matches_suite": digest(map_path) == suite["map_sha256"],
        "map_contains_no_raw_source_handles": not any(
            handle in map_text for handle in source_handles
        ),
        "source_detour_roles_are_distinct": len({
            item["detour_role"] for item in suite["sources"]
        }) == 2,
        "sources": source_audits,
        "source_success_rate": source_success_rate,
        "replays": replay_audits,
        "replay_success_rate": replay_success_rate,
        "unknown_signature_recomputed": list(unknown)
        == suite["unknown_signature"],
        "unknown_signature_rejected": unknown_rejected
        and suite["unknown_signature_rejected"] is True,
        "role_index_variety": len(set(role_mappings)),
        "role_index_variety_valid": len(set(role_mappings))
        >= settings["minimum_role_index_variety"],
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
        audit["map_format_and_shape_valid"],
        audit["map_hash_matches_suite"],
        audit["map_contains_no_raw_source_handles"],
        audit["source_detour_roles_are_distinct"],
        audit["source_success_rate"] == prereg["required_source_success_rate"],
        audit["replay_success_rate"] == prereg["required_replay_success_rate"],
        audit["unknown_signature_rejected"],
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
