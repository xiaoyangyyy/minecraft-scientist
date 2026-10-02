"""Replay all compound prefixes, then independently measure realized horizontal turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compound_motion_common import compound_pipeline
from controlled_motion_common import digest


def _audit(route, experiment):
    import audit_keyframe_rgb_motion as base

    initial = {
        path: digest(path)
        for path in (
            Path(__file__),
            experiment / "config.json",
            experiment / "report.json",
        )
    }
    with compound_pipeline(base):
        result = base.audit(route, experiment)
    if not result["valid"]:
        return result
    # Base replay has completed every public calculation and separately scored
    # poses. Only now compute the additional physical intervention characterization.
    evaluator = route / "evaluator-audit.json"
    initial[evaluator] = result["evaluator_sha256"]
    result["checks"]["compound_evaluator_identity"] = (
        digest(evaluator) == initial[evaluator]
    )
    if not result["checks"]["compound_evaluator_identity"]:
        result["valid"] = False
        result["evaluation"] = None
        return result
    positions = json.loads(evaluator.read_bytes())["positions"]
    config = json.loads((experiment / "config.json").read_bytes())
    pairs = config["pairs"]
    thresholds = config["physical_turn_thresholds"]
    turns = []
    for pair in [pair for pair in pairs if pair["group"] == "leg_one"]:
        number = pair["id"].rsplit("-", 1)[1]
        second = next(item for item in pairs if item["id"] == f"leg_two-{number}")
        vectors = [
            np.array(
                (
                    positions[item["second"]]["xpos"]
                    - positions[item["first"]]["xpos"],
                    positions[item["second"]]["zpos"]
                    - positions[item["first"]]["zpos"],
                )
            )
            for item in (pair, second)
        ]
        lengths = [float(np.linalg.norm(vector)) for vector in vectors]
        angle = (
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            vectors[0] @ vectors[1] / (lengths[0] * lengths[1]), -1, 1
                        )
                    )
                )
            )
            if min(lengths) >= thresholds["minimum_leg_horizontal_blocks"]
            else None
        )
        turns.append(
            {
                "trial": number,
                "leg_horizontal_blocks": lengths,
                "angle_degrees": angle,
                "observed_noncollinear": angle is not None
                and thresholds["minimum_turn_degrees"]
                <= angle
                <= thresholds["maximum_turn_degrees"],
            }
        )
    result["checks"]["compound_sources_inputs_stable"] = all(
        digest(path) == value for path, value in initial.items()
    )
    result["valid"] = all(result["checks"].values())
    result["compound_auditor_sha256"] = initial[Path(__file__)]
    if not result["valid"]:
        result["evaluation"] = None
        return result
    result["evaluation"]["physical_turns"] = turns
    result["evaluation"]["observed_noncollinear_trials"] = sum(
        item["observed_noncollinear"] for item in turns
    )
    return result


def audit(route, experiment):
    try:
        return _audit(route, experiment)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return {
            "valid": False,
            "checks": {"compound_audit_completed": False},
            "evaluation": None,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite compound motion audit")
    result = audit(args.route, args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "checks": result["checks"],
                "groups": None
                if not result["valid"]
                else result["evaluation"]["groups"],
                "observed_noncollinear_trials": None
                if not result["valid"]
                else result["evaluation"]["observed_noncollinear_trials"],
            },
            sort_keys=True,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
