"""Recompute every public switched-view comparison before any private scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import action_switch_common as common
import audit_active_rgb_motion as baseline
import numpy as np
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest

from minecraft_scientist.exploration.action_switch_motion import ActionSwitchConfig


def _physical_switch(positions, plan):
    if not plan["available"]:
        return None

    def position(slot):
        return np.array([positions[slot][key] for key in ("xpos", "ypos", "zpos")])

    original = position(plan["candidate_end_frame"]) - position(plan["anchor_frame"])
    later = position(plan["target_frames"][-1]) - position(plan["target_frames"][0])
    distances = tuple(float(np.linalg.norm(item)) for item in (original, later))
    angle = None
    if min(distances) >= 0.15:
        angle = math.degrees(
            math.acos(float(np.clip(original @ later / np.prod(distances), -1, 1)))
        )
    return {
        "displacements_blocks": distances,
        "direction_difference_degrees": angle,
        "effective_noncollinear": angle is not None and 45 <= angle <= 135,
    }


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
            "registered_switched_views": {
                key: value
                for key, value in config.items()
                if key not in ("input_sha256", "source_sha256")
            }
            == common.configuration(route, experiment),
            "complete_public_inputs": config["input_sha256"]
            == {name: digest(path) for name, path in paths.items()},
            "current_switched_view_sources": config["source_sha256"]
            == {name: digest(path) for name, path in sources.items()},
            "switched_view_snapshots": {
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
        replay = common.replay(route, experiment, ActionSwitchConfig())
        checks["all_public_switched_views_replayed"] = _numeric_equal(
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
        # Only after complete public recomputation and baseline scoring may this
        # evaluator inspect actual physical diversity. It never selects views.
        evaluator = route / "evaluator-audit.json"
        positions = json.loads(evaluator.read_bytes())["positions"]
        if digest(evaluator) != scored["evaluator_sha256"]:
            raise ValueError("private telemetry changed before diversity scoring")
        samples, groups = [], {}
        for diagnostic, original in zip(
            replay, scored["evaluation"]["scores"], strict=True
        ):
            if (
                diagnostic["pair"] != original["pair"]
                or diagnostic["selected_offset"] != original["selected_offset"]
            ):
                raise ValueError("switched-view/baseline interval differs")
            selected, evidence = original["selected_local"], diagnostic["action_switch"]
            reference = evidence["reference_future"]
            # Diagnostic-only removal: no failed reference/unknown may be rescued,
            # no alternative pose becomes executable, no uniqueness is asserted.
            retained = (
                reference is not None
                and reference["status"] == "consistent"
                and evidence["comparison"]
                not in ("ambiguity_survives", "incomplete_comparison")
            )
            group = groups.setdefault(
                original["pair"]["group"],
                {
                    "trials": 0,
                    "selected": 0,
                    "original_correct": 0,
                    "original_wrong": 0,
                    "retained_correct": 0,
                    "retained_wrong": 0,
                    "comparisons": {},
                },
            )
            group["trials"] += 1
            category = evidence["comparison"]
            group["comparisons"][category] = group["comparisons"].get(category, 0) + 1
            if selected is not None:
                group["selected"] += 1
                kind = "correct" if selected["correct"] else "wrong"
                group["original_" + kind] += 1
                group["retained_" + kind] += int(retained)
            samples.append(
                {
                    "pair": original["pair"],
                    "selected_offset": original["selected_offset"],
                    "original_score": selected,
                    "comparison": category,
                    "diagnostic_retained": retained,
                    "physical_switch": _physical_switch(positions, evidence["plan"]),
                    "additional_observation_count": len(diagnostic["future_indices"])
                    - (diagnostic["selected_offset"] + 1)
                    if diagnostic["future_indices"]
                    else 0,
                }
            )
        checks["public_bytes_stable_through_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        checks["baseline_evaluator_identity"] = (
            digest(evaluator) == scored["evaluator_sha256"]
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
                "scope": "development-only temporal holdout at fixed action switch; longer view horizon is not matched observation cost; no witness is not uniqueness; no pose replacement or navigation",
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
        raise FileExistsError("refusing to overwrite switched-view audit")
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
