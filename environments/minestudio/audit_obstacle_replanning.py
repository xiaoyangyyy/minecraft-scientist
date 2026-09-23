"""Independently audit real-MineStudio RGB obstacle-replanning evidence."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from obstacle_replanning_probe import classify_handles, obstacle_pixels
from PIL import Image

from minecraft_scientist.minecraft.visual_evidence import (
    central_view_salience,
    largest_achromatic_region,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], text


def position_delta(positions: list[dict], axis: str) -> float | None:
    if not positions or axis not in positions[0] or axis not in positions[-1]:
        return None
    return float(positions[-1][axis] - positions[0][axis])


def recompute_features(path: Path) -> dict:
    frame = np.asarray(Image.open(path))
    marker = largest_achromatic_region(frame)
    obstacle, central_pixels = obstacle_pixels(frame)
    return {
        "marker": asdict(marker) if marker is not None else None,
        "obstacle": asdict(obstacle) if obstacle is not None else None,
        "central_obstacle_pixels": central_pixels,
        "salience": central_view_salience(frame),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    prereg_path = root / "preregistered.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    suite_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
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

    cases = []
    for index, expected_case in enumerate(prereg["cases"]):
        directory = root / f"case-{index}"
        report_path = directory / "report.json"
        calibration_path = directory / "calibration.json"
        roles_path = directory / "roles.json"
        trajectory_path = directory / "route" / "trajectory.json"
        events_path = directory / "route" / "events.jsonl"
        evaluator_path = directory / "evaluator-audit.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
        events, event_text = load_events(events_path)
        suite_case = suite_report["cases"][index]

        recomputed_calibration = []
        calibration_events_valid = True
        public_text_parts = [
            calibration_path.read_text(encoding="utf-8"),
            roles_path.read_text(encoding="utf-8"),
            trajectory_path.read_text(encoding="utf-8"),
            event_text,
            report_path.read_text(encoding="utf-8"),
        ]
        for trial_index, item in enumerate(calibration):
            trial = directory / "calibration" / f"trial-{trial_index}"
            before = recompute_features(trial / "before.png")
            after = recompute_features(trial / "after.png")
            recomputed = {
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
            recomputed_calibration.append(recomputed)
            trial_events, trial_text = load_events(trial / "events.jsonl")
            public_text_parts.append(trial_text)
            moves = [
                event for event in trial_events
                if event.get("executed_action", {}).get("kind") == "move"
            ]
            calibration_events_valid = calibration_events_valid and (
                len(moves) == 1
                and moves[0]["executed_action"]["target_id"] == item["handle_id"]
                and moves[0]["executed_action"]["ticks"]
                == settings["move_frames"]
                and moves[0].get("intervention_compliance") is True
                and moves[0].get("event", {}).get("success") is True
            )

        trajectory_recomputed = []
        for item in trajectory:
            frame_path = (
                directory
                / "route"
                / f"frame-{item['observation_index']:02d}.png"
            )
            frame = np.asarray(Image.open(frame_path))
            features = recompute_features(frame_path)
            trajectory_recomputed.append({
                "observation_index": item["observation_index"],
                "phase": item["phase"],
                "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                **features,
            })

        move_events = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        expected_move_count = sum(
            report[key]
            for key in (
                "detour_moves",
                "extra_detour_moves",
                "bypass_moves",
                "recenter_moves",
                "final_moves",
            )
        )
        trajectory_phases = [item["phase"] for item in trajectory[1:]]
        event_targets = [event["executed_action"]["target_id"] for event in move_events]
        phase_targets_valid = len(trajectory_phases) == len(event_targets) and all(
            (
                phase in {"detour", "local-replan"}
                and target == roles["negative_shift"]
            )
            or (phase in {"bypass", "final-approach"} and target == roles["goal"])
            or (
                phase == "recenter"
                and target in {roles["negative_shift"], roles["positive_shift"]}
            )
            for phase, target in zip(trajectory_phases, event_targets, strict=True)
        )
        route_events_valid = (
            len(move_events) == expected_move_count
            and all(
                event["executed_action"]["ticks"] == settings["move_frames"]
                and event.get("intervention_compliance") is True
                and event.get("event", {}).get("success") is True
                for event in move_events
            )
            and phase_targets_valid
        )
        route_episode_ids = {
            event.get("provenance", {}).get("episode") for event in events
        }

        private_trials = evaluator["trials"]
        controls_are_permutation = {
            item["control"] for item in private_trials
        } == {"forward", "back", "left", "right"}
        private_progress = {
            item["handle_id"]: position_delta(item["positions"], "zpos")
            for item in private_trials
        }
        numeric_progress = {
            handle: value for handle, value in private_progress.items()
            if value is not None
        }
        max_progress = max(numeric_progress.values(), default=float("-inf"))
        evaluator_goal_handles = [
            handle for handle, value in numeric_progress.items()
            if value == max_progress
        ]

        positions = evaluator["route"]["positions"]
        start = positions[0]
        goal_progress = position_delta(positions, "zpos")
        lateral_offsets = [
            abs(item["xpos"] - start["xpos"])
            for item in positions
            if "xpos" in item and "xpos" in start
        ]
        max_excursion = max(lateral_offsets, default=None)
        final_offset = lateral_offsets[-1] if lateral_offsets else None
        relative_path = [
            {
                "x": item["xpos"] - start["xpos"],
                "z": item["zpos"] - start["zpos"],
            }
            for item in positions
            if all(key in item and key in start for key in ("xpos", "zpos"))
        ]
        expected_fixture = {
            "goal_axis": "positive_z",
            "obstacle_center_x_relative_to_start": [-1.5, 1.5],
            "obstacle_center_z_relative_to_start": [2.5, 4.5],
        }
        path_avoids_obstacle_volume = bool(relative_path) and not any(
            -1.5 < point["x"] < 1.5 and 2.5 < point["z"] < 4.5
            for point in relative_path
        )
        detour_before_obstacle_plane = any(
            abs(point["x"]) >= settings["minimum_lateral_excursion"]
            and point["z"] < 2.5
            for point in relative_path
        )

        detour_end_index = report["detour_moves"]
        bypass_end_index = (
            detour_end_index
            + report["bypass_moves"]
            + report["extra_detour_moves"]
        )
        recenter_end_index = bypass_end_index + report["recenter_moves"]
        initial = trajectory[0]
        detour_end = trajectory[detour_end_index]
        bypass_end = trajectory[bypass_end_index]
        recenter_end = trajectory[recenter_end_index]
        obstacle_area_at_bypass_end = (
            bypass_end["obstacle"]["pixel_count"]
            if bypass_end["obstacle"] is not None
            else 0
        )
        visual_policy_stops_valid = all((
            initial["central_obstacle_pixels"]
            >= settings["central_clear_pixels"],
            detour_end["central_obstacle_pixels"]
            < settings["central_clear_pixels"],
            report["bypass_moves"] >= settings["min_bypass_moves"],
            obstacle_area_at_bypass_end < settings["obstacle_passed_pixels"],
            abs(
                recenter_end["marker"]["centroid_x"]
                - initial["marker"]["centroid_x"]
            ) <= settings["center_tolerance"],
            trajectory[-1]["salience"] >= settings["score_threshold"],
        ))

        forbidden_public_tokens = (
            "location_stats",
            "xpos",
            "ypos",
            "zpos",
            '"control"',
            ': "forward"',
            ':"forward"',
            ': "back"',
            ':"back"',
            ': "left"',
            ':"left"',
            ': "right"',
            ':"right"',
        )
        public_text = "\n".join(public_text_parts)
        privileged_sources_present = (
            all(
                "location_stats" in item["raw_metadata_keys"]
                and bool(item["positions"])
                for item in private_trials
            )
            and "location_stats" in evaluator["route"]["raw_metadata_keys"]
            and bool(positions)
        )

        cases.append({
            "case": index,
            "seed_matches_preregistration": report["seed"]
            == expected_case["seed"],
            "mapping_seed_matches_preregistration": report["mapping_seed"]
            == expected_case["mapping_seed"]
            == evaluator["mapping_seed"],
            "report_hash_matches_suite": digest(report_path)
            == suite_case["report_sha256"],
            "calibration_hash_matches_suite": digest(calibration_path)
            == suite_case["calibration_sha256"],
            "roles_hash_matches_suite": digest(roles_path)
            == suite_case["roles_sha256"],
            "trajectory_hash_matches_suite": digest(trajectory_path)
            == suite_case["trajectory_sha256"],
            "events_hash_matches_suite": digest(events_path)
            == suite_case["events_sha256"],
            "evaluator_hash_matches_suite": digest(evaluator_path)
            == suite_case["evaluator_sha256"],
            "candidate_count_matches": len(calibration)
            == settings["candidate_count"],
            "calibration_features_recomputed": close(
                calibration, recomputed_calibration
            ),
            "roles_recomputed_from_pixels": roles
            == classify_handles(recomputed_calibration),
            "calibration_events_valid": calibration_events_valid,
            "trajectory_features_recomputed": close(
                trajectory, trajectory_recomputed
            ),
            "route_events_valid": route_events_valid,
            "continuous_route_episode": route_episode_ids == {1},
            "controls_are_private_permutation": controls_are_permutation,
            "evaluator_goal_direction_unique_positive": (
                len(evaluator_goal_handles) == 1 and max_progress > 0
            ),
            "pixel_goal_role_matches_evaluator": roles["goal"]
            == (evaluator_goal_handles[0] if evaluator_goal_handles else None),
            "privileged_position_sources_present": privileged_sources_present,
            "fixture_bounds_match": evaluator["fixture"] == expected_fixture,
            "public_artifacts_have_no_semantic_or_coordinate_leak": not any(
                token in public_text for token in forbidden_public_tokens
            ),
            "visual_policy_stops_valid": visual_policy_stops_valid,
            "detour_occurs_before_obstacle_plane": detour_before_obstacle_plane,
            "evaluator_path_avoids_obstacle_volume": path_avoids_obstacle_volume,
            "goal_progress_recomputed": goal_progress,
            "goal_progress_matches_report": goal_progress is not None
            and math.isclose(
                goal_progress,
                report["evaluator_goal_progress"],
                abs_tol=1e-9,
            ),
            "goal_progress_in_range": goal_progress is not None
            and settings["minimum_goal_progress"]
            <= goal_progress <= settings["maximum_goal_progress"],
            "lateral_excursion_matches_report": max_excursion is not None
            and math.isclose(
                max_excursion,
                report["evaluator_max_lateral_excursion"],
                abs_tol=1e-9,
            ),
            "lateral_excursion_in_range": max_excursion is not None
            and settings["minimum_lateral_excursion"]
            <= max_excursion <= settings["maximum_lateral_excursion"],
            "final_offset_matches_report": final_offset is not None
            and math.isclose(
                final_offset,
                report["evaluator_final_lateral_offset"],
                abs_tol=1e-9,
            ),
            "final_offset_in_range": final_offset is not None
            and final_offset <= settings["maximum_final_lateral_offset"],
        })

    role_mappings = [
        tuple(sorted(item["role_indices"].items()))
        for item in suite_report["cases"]
    ]
    audit = {
        "run": str(root),
        "preregistration_intact": digest(prereg_path)
        == suite_report["preregistration_sha256"],
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot)
        and all(item["valid"] for item in source_snapshot),
        "role_index_variety": len(set(role_mappings)),
        "role_index_variety_valid": len(set(role_mappings))
        >= settings["minimum_role_index_variety"],
        "cases": cases,
    }
    ignored = {"case", "goal_progress_recomputed"}
    required = tuple(key for key in cases[0] if key not in ignored)
    audit["success_rate"] = sum(
        all(case[field] is True for field in required) for case in cases
    ) / len(cases)
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
        audit["role_index_variety_valid"],
        audit["success_rate"] == prereg["required_success_rate"],
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
