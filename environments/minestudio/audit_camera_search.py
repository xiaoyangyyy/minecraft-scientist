"""Independently audit a preregistered real MineStudio camera-search suite."""
import argparse
import hashlib
import json
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
    preregistration = root / "preregistered.json"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    preregistered = json.loads(preregistration.read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "runtime-manifest.json").read_text(encoding="utf-8")
    )

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
    for index, expected_case in enumerate(preregistered["cases"]):
        directory = root / f"case-{index}"
        fixture = json.loads((directory / "fixture.json").read_text(encoding="utf-8"))
        case_report = json.loads(
            (directory / "report.json").read_text(encoding="utf-8")
        )
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        looks = [
            event for event in events
            if event.get("executed_action", {}).get("kind") == "look"
        ]
        recomputed_scores = {
            fixture["view_bindings"][name]: central_view_salience(
                np.asarray(Image.open(directory / f"view-{name}.png"))
            )
            for name in preregistered["view_set"]
        }
        predicted = max(recomputed_scores, key=recomputed_scores.get)
        scores_match = all(
            abs(recomputed_scores[key] - case_report["opaque_view_scores"][key])
            < 1e-9
            for key in recomputed_scores
        )
        cases.append({
            "case": index,
            "report_hash_matches_suite": (
                digest(directory / "report.json")
                == report["cases"][index]["report_sha256"]
            ),
            "fixture_matches_preregistration": (
                fixture["target_view"] == expected_case["target_view"]
                and case_report["seed"] == expected_case["seed"]
            ),
            "opaque_scores_only": all(
                key.startswith("obj_") for key in recomputed_scores
            ),
            "scores_recomputed_exactly": scores_match,
            "prediction_recomputed": predicted,
            "prediction_matches_report": predicted == case_report["predicted_view_id"],
            "prediction_correct": predicted == fixture["expected_view_id"],
            "successful_look_actions": len(looks),
            "all_look_actions_valid": all(
                event.get("intervention_compliance") is True
                and event.get("event", {}).get("success") is True
                for event in looks
            ),
        })

    expected_looks = preregistered["required_successful_look_actions_per_case"]
    audit = {
        "run": str(root),
        "preregistration_intact": (
            digest(preregistration) == report["preregistration_sha256"]
        ),
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot) and all(
            item["valid"] for item in source_snapshot
        ),
        "cases": cases,
        "accuracy": sum(case["prediction_correct"] for case in cases) / len(cases),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
        audit["accuracy"] == preregistered["required_accuracy"],
        all(
            case["fixture_matches_preregistration"]
            and case["report_hash_matches_suite"]
            and case["opaque_scores_only"]
            and case["scores_recomputed_exactly"]
            and case["prediction_matches_report"]
            and case["successful_look_actions"] == expected_looks
            and case["all_look_actions_valid"]
            for case in cases
        ),
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
