"""Recompute reference/direction searches completely before private pose scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_active_rgb_motion as baseline
import pose_direction_common as common
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest

from minecraft_scientist.exploration.pose_direction_search import DirectionSearchConfig


def audit(route, experiment, diagnosis):
    checks = {}
    try:
        config = json.loads((diagnosis / "config.json").read_bytes())
        predicted = json.loads((diagnosis / "report.json").read_bytes())
        paths, sources = common.inputs(route, experiment), common.sources()
        checks = {
            "diagnosis_completed": predicted["completed"] is True
            and predicted["failure"] is None
            and predicted["inputs_sources_unchanged"] is True,
            "no_navigation_promotion": predicted["navigation_accepted"] is False,
            "diagnosis_manifest_identity": predicted["config_sha256"]
            == digest(diagnosis / "config.json"),
            "registered_search": {
                key: value
                for key, value in config.items()
                if key not in ("input_sha256", "source_sha256")
            }
            == common.configuration(route, experiment),
            "complete_public_inputs": config["input_sha256"]
            == {name: digest(path) for name, path in paths.items()},
            "current_search_sources": config["source_sha256"]
            == {name: digest(path) for name, path in sources.items()},
            "search_source_snapshots": {
                path.name for path in (diagnosis / "source-snapshot").glob("*.py")
            }
            == set(sources)
            and all(
                digest(diagnosis / "source-snapshot" / name) == value
                for name, value in config["source_sha256"].items()
            ),
        }
        if not all(checks.values()):
            return {
                "valid": False,
                "checks": checks,
                "evaluation": None,
                "failure": None,
            }
        frozen = {
            path: digest(path)
            for path in list(paths.values())
            + list(sources.values())
            + [diagnosis / "config.json", diagnosis / "report.json"]
            + list((diagnosis / "source-snapshot").glob("*.py"))
        }
        replay = common.replay(route, experiment, DirectionSearchConfig())
        checks["all_public_searches_replayed"] = _numeric_equal(
            replay, predicted["results"]
        )
        checks["public_bytes_stable_before_private_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        if not all(checks.values()):
            return {
                "valid": False,
                "checks": checks,
                "evaluation": None,
                "failure": None,
            }
        scored = baseline.audit(route, experiment)
        checks.update(
            {"baseline/" + name: value for name, value in scored["checks"].items()}
        )
        checks["baseline_valid"] = scored["valid"]
        if not all(checks.values()):
            return {
                "valid": False,
                "checks": checks,
                "evaluation": None,
                "failure": scored["failure"],
            }
        samples, groups = [], {}
        for diagnostic, original in zip(
            replay, scored["evaluation"]["scores"], strict=True
        ):
            if (
                diagnostic["pair"] != original["pair"]
                or diagnostic["selected_offset"] != original["selected_offset"]
            ):
                raise ValueError("direction search/baseline interval differs")
            selected = original["selected_local"]
            ambiguous = diagnostic["direction_search"]["ambiguity_detected"]
            group = groups.setdefault(
                original["pair"]["group"],
                {
                    "trials": 0,
                    "selected": 0,
                    "reference_failed": 0,
                    "original_correct": 0,
                    "original_wrong": 0,
                    "ambiguity_vetoed_correct": 0,
                    "ambiguity_vetoed_wrong": 0,
                    "remaining_correct": 0,
                    "remaining_wrong": 0,
                },
            )
            group["trials"] += 1
            if selected is not None:
                group["selected"] += 1
                group["reference_failed"] += int(
                    not diagnostic["direction_search"][
                        "reference_matches_original_checks"
                    ]
                )
                kind = "correct" if selected["correct"] else "wrong"
                group["original_" + kind] += 1
                group[("ambiguity_vetoed_" if ambiguous else "remaining_") + kind] += 1
            samples.append(
                {
                    "pair": original["pair"],
                    "selected_offset": original["selected_offset"],
                    "original_score": selected,
                    "ambiguity_detected": ambiguous,
                    "reference_matches_original_checks": diagnostic["direction_search"][
                        "reference_matches_original_checks"
                    ],
                    "old_grid_ambiguity_detected": diagnostic["original_competition"][
                        "ambiguity_detected"
                    ],
                }
            )
        checks["public_bytes_stable_through_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        checks["baseline_evaluator_identity"] = (
            digest(route / "evaluator-audit.json") == scored["evaluator_sha256"]
        )
        valid = all(checks.values())
        return {
            "valid": valid,
            "checks": checks,
            "failure": None,
            "evaluation": None
            if not valid
            else {
                "groups": groups,
                "samples": samples,
                "navigation_accepted": False,
                "scope": "development-only incomplete fitted family; optional veto never replaces/rescues a pose; no witness is not uniqueness",
            },
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "valid": False,
            "checks": checks,
            "evaluation": None,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("route", "experiment", "diagnosis"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite direction-search audit")
    result = audit(args.route, args.experiment, args.diagnosis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "failure": result["failure"],
                "groups": None
                if result["evaluation"] is None
                else result["evaluation"]["groups"],
            }
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
