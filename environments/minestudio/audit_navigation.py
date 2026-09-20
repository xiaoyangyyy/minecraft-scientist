"""Independently audit a preregistered real MineStudio navigation suite."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    prereg_path = root / "preregistered.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
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
        case_report_path = directory / "report.json"
        case_report = json.loads(case_report_path.read_text(encoding="utf-8"))
        trajectory = json.loads(
            (directory / "trajectory.json").read_text(encoding="utf-8")
        )
        position_audit = json.loads(
            (directory / "position-audit.json").read_text(encoding="utf-8")
        )
        event_text = (directory / "events.jsonl").read_text(encoding="utf-8")
        events = [json.loads(line) for line in event_text.splitlines() if line.strip()]
        move_events = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "move"
        ]
        recomputed_scores = [
            central_view_salience(np.asarray(Image.open(
                directory / f"frame-{item['move_index']:02d}.png"
            )))
            for item in trajectory
        ]
        scores_match = all(
            abs(score - item["score"]) < 1e-9
            for score, item in zip(recomputed_scores, trajectory, strict=True)
        )
        first_threshold_crossing = (
            bool(recomputed_scores)
            and recomputed_scores[-1] >= settings["score_threshold"]
            and all(
                score < settings["score_threshold"]
                for score in recomputed_scores[:-1]
            )
        )
        positions = position_audit["positions"]
        displacement = None
        if positions and all(
            key in positions[0] and key in positions[-1]
            for key in ("xpos", "zpos")
        ):
            displacement = math.hypot(
                positions[-1]["xpos"] - positions[0]["xpos"],
                positions[-1]["zpos"] - positions[0]["zpos"],
            )
        cases.append({
            "case": index,
            "seed_matches_preregistration": case_report["seed"] == expected_case["seed"],
            "report_hash_matches_suite": (
                digest(case_report_path) == report["cases"][index]["report_sha256"]
            ),
            "scores_recomputed_exactly": scores_match,
            "first_threshold_crossing": first_threshold_crossing,
            "move_count_matches_trajectory": (
                len(move_events) == case_report["moves"] == len(trajectory) - 1
            ),
            "all_move_actions_valid": bool(move_events)
            and all(
                (
                    event.get("executed_action", {}).get("ticks")
                    == settings["move_frames"]
                    and event.get("intervention_compliance") is True
                    and event.get("event", {}).get("success") is True
                )
                for event in move_events
            ),
            "agent_event_log_has_no_coordinates": not any(
                token in event_text
                for token in ("location_stats", "xpos", "ypos", "zpos")
            ),
            "privileged_position_source_recorded": (
                "location_stats" in position_audit["raw_metadata_keys"]
                and bool(position_audit["positions"])
            ),
            "displacement_recomputed": displacement,
            "displacement_matches_report": (
                displacement is not None
                and abs(displacement - case_report["evaluator_displacement"]) < 1e-9
            ),
            "displacement_in_preregistered_range": (
                displacement is not None
                and settings["minimum_displacement"] <= displacement
                <= settings["maximum_displacement"]
            ),
        })

    audit = {
        "run": str(root),
        "preregistration_intact": digest(prereg_path) == report["preregistration_sha256"],
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot) and all(
            item["valid"] for item in source_snapshot
        ),
        "cases": cases,
    }
    required = (
        "seed_matches_preregistration",
        "report_hash_matches_suite",
        "scores_recomputed_exactly",
        "first_threshold_crossing",
        "move_count_matches_trajectory",
        "all_move_actions_valid",
        "agent_event_log_has_no_coordinates",
        "privileged_position_source_recorded",
        "displacement_matches_report",
        "displacement_in_preregistered_range",
    )
    audit["success_rate"] = sum(
        all(case[field] for field in required) for case in cases
    ) / len(cases)
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
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
