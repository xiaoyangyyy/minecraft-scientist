"""Independently audit opaque-direction discovery in real MineStudio runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.minecraft.visual_evidence import central_view_salience


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], text


def position_delta(positions: list[dict], axis: str) -> float | None:
    if not positions or axis not in positions[0] or axis not in positions[-1]:
        return None
    return float(positions[-1][axis] - positions[0][axis])


def planar_displacement(positions: list[dict]) -> float | None:
    dx = position_delta(positions, "xpos")
    dz = position_delta(positions, "zpos")
    if dx is None or dz is None:
        return None
    return math.hypot(dx, dz)


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
        source_snapshot.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "valid": actual == expected,
            }
        )

    cases = []
    for index, expected_case in enumerate(prereg["cases"]):
        directory = root / f"case-{index}"
        report_path = directory / "report.json"
        calibration_path = directory / "calibration.json"
        evaluator_path = directory / "evaluator-audit.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))

        recomputed_calibration = []
        public_text_parts = [
            calibration_path.read_text(encoding="utf-8"),
            report_path.read_text(encoding="utf-8"),
        ]
        trial_events_valid = True
        for trial_index, item in enumerate(calibration):
            trial_dir = directory / "calibration" / f"trial-{trial_index}"
            before_score = central_view_salience(
                np.asarray(Image.open(trial_dir / "before.png"))
            )
            after_score = central_view_salience(
                np.asarray(Image.open(trial_dir / "after.png"))
            )
            events, event_text = load_events(trial_dir / "events.jsonl")
            public_text_parts.append(event_text)
            move_events = [
                event
                for event in events
                if event.get("executed_action", {}).get("kind") == "move"
            ]
            trial_events_valid = trial_events_valid and (
                len(move_events) == 1
                and move_events[0].get("executed_action", {}).get("target_id")
                == item["handle_id"]
                and move_events[0].get("executed_action", {}).get("ticks")
                == settings["move_frames"]
                and move_events[0].get("intervention_compliance") is True
                and move_events[0].get("event", {}).get("success") is True
            )
            recomputed_calibration.append(
                {
                    "handle_id": item["handle_id"],
                    "before": before_score,
                    "after": after_score,
                    "delta": after_score - before_score,
                }
            )

        calibration_scores_match = all(
            abs(recomputed["before"] - recorded["before_score"]) < 1e-9
            and abs(recomputed["after"] - recorded["after_score"]) < 1e-9
            and abs(recomputed["delta"] - recorded["score_delta"]) < 1e-9
            for recomputed, recorded in zip(
                recomputed_calibration, calibration, strict=True
            )
        )
        winning_delta = max(item["delta"] for item in recomputed_calibration)
        pixel_winners = [
            item["handle_id"]
            for item in recomputed_calibration
            if item["delta"] == winning_delta
        ]

        private_trials = evaluator["trials"]
        private_by_handle = {item["handle_id"]: item for item in private_trials}
        goal_progress_matches = all(
            item.get("goal_progress") is not None
            and abs(
                item["goal_progress"]
                - position_delta(item["positions"], "zpos")
            )
            < 1e-9
            for item in private_trials
        )
        evaluator_progress = {
            item["handle_id"]: position_delta(item["positions"], "zpos")
            for item in private_trials
        }
        numeric_progress = {
            handle_id: progress
            for handle_id, progress in evaluator_progress.items()
            if progress is not None
        }
        maximum_goal_progress = max(numeric_progress.values(), default=float("-inf"))
        evaluator_goal_handles = [
            handle_id
            for handle_id, progress in numeric_progress.items()
            if progress == maximum_goal_progress
        ]
        controls_are_permutation = {
            item["control"] for item in private_trials
        } == {"forward", "back", "left", "right"}
        public_handles_match_private = (
            set(private_by_handle)
            == {item["handle_id"] for item in calibration}
        )
        privileged_sources_present = all(
            "location_stats" in item["raw_metadata_keys"]
            and bool(item["positions"])
            for item in private_trials
        ) and (
            "location_stats" in evaluator["approach"]["raw_metadata_keys"]
            and bool(evaluator["approach"]["positions"])
        )

        approach_dir = directory / "approach"
        trajectory_path = approach_dir / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        public_text_parts.append(trajectory_path.read_text(encoding="utf-8"))
        approach_events, approach_event_text = load_events(
            approach_dir / "events.jsonl"
        )
        public_text_parts.append(approach_event_text)
        approach_scores = [
            central_view_salience(
                np.asarray(
                    Image.open(approach_dir / f"frame-{item['move_index']:02d}.png")
                )
            )
            for item in trajectory
        ]
        approach_scores_match = all(
            abs(score - item["score"]) < 1e-9
            for score, item in zip(approach_scores, trajectory, strict=True)
        )
        first_threshold_crossing = (
            bool(approach_scores)
            and approach_scores[-1] >= settings["score_threshold"]
            and all(
                score < settings["score_threshold"]
                for score in approach_scores[:-1]
            )
        )
        approach_moves = [
            event
            for event in approach_events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        approach_events_valid = (
            bool(approach_moves)
            and len(approach_moves) == report["moves"] == len(trajectory) - 1
            and all(
                event.get("executed_action", {}).get("target_id")
                == report["selected_handle"]
                and event.get("executed_action", {}).get("ticks")
                == settings["move_frames"]
                and event.get("intervention_compliance") is True
                and event.get("event", {}).get("success") is True
                for event in approach_moves
            )
        )
        displacement = planar_displacement(evaluator["approach"]["positions"])
        forbidden_public_tokens = (
            "location_stats",
            "xpos",
            "ypos",
            "zpos",
            '"control"',
            '"forward"',
            '"back"',
            '"left"',
            '"right"',
        )
        public_text = "\n".join(public_text_parts)

        cases.append(
            {
                "case": index,
                "seed_matches_preregistration": report["seed"]
                == expected_case["seed"],
                "mapping_seed_matches_preregistration": report["mapping_seed"]
                == expected_case["mapping_seed"]
                == evaluator["mapping_seed"],
                "report_hash_matches_suite": digest(report_path)
                == suite_report["cases"][index]["report_sha256"],
                "calibration_hash_matches_suite": digest(calibration_path)
                == suite_report["cases"][index]["calibration_sha256"],
                "evaluator_hash_matches_suite": digest(evaluator_path)
                == suite_report["cases"][index]["evaluator_sha256"],
                "candidate_count_matches": len(calibration)
                == settings["candidate_count"],
                "calibration_scores_recomputed": calibration_scores_match,
                "unique_pixel_winner_selected": pixel_winners
                == [report["selected_handle"]],
                "trial_events_valid": trial_events_valid,
                "controls_are_private_permutation": controls_are_permutation,
                "public_handles_match_private": public_handles_match_private,
                "goal_progress_recomputed": goal_progress_matches,
                "evaluator_goal_direction_unique_positive": (
                    len(numeric_progress) == len(private_trials)
                    and len(evaluator_goal_handles) == 1
                    and maximum_goal_progress > 0
                ),
                "pixel_winner_is_evaluator_goal_direction": report["selected_handle"]
                == (evaluator_goal_handles[0] if evaluator_goal_handles else None),
                "privileged_position_sources_present": privileged_sources_present,
                "public_artifacts_have_no_semantic_or_coordinate_leak": not any(
                    token in public_text for token in forbidden_public_tokens
                ),
                "approach_scores_recomputed": approach_scores_match,
                "first_threshold_crossing": first_threshold_crossing,
                "approach_events_valid": approach_events_valid,
                "displacement_recomputed": displacement,
                "displacement_matches_report": displacement is not None
                and abs(displacement - report["evaluator_displacement"]) < 1e-9,
                "displacement_in_preregistered_range": displacement is not None
                and settings["minimum_displacement"]
                <= displacement
                <= settings["maximum_displacement"],
            }
        )

    selected_indices = [
        suite_report["cases"][index]["selected_handle_index"]
        for index in range(len(cases))
    ]
    audit = {
        "run": str(root),
        "preregistration_intact": digest(prereg_path)
        == suite_report["preregistration_sha256"],
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot)
        and all(item["valid"] for item in source_snapshot),
        "selected_index_variety": len(set(selected_indices)),
        "selected_index_variety_valid": len(set(selected_indices))
        >= settings["minimum_selected_index_variety"],
        "cases": cases,
    }
    required = tuple(
        key
        for key in cases[0]
        if key not in {"case", "displacement_recomputed"}
    )
    audit["success_rate"] = sum(
        all(case[field] is True for field in required) for case in cases
    ) / len(cases)
    audit["valid"] = all(
        (
            audit["preregistration_intact"],
            audit["source_snapshot_complete"],
            audit["selected_index_variety_valid"],
            audit["success_rate"] == prereg["required_success_rate"],
        )
    )
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
