"""Independently replay public planning/capture/prefixes, then score actual motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import plan_active_rgb_motion as planner
from active_motion_pipeline import active_pipeline
from controlled_motion_common import digest


def audit(route, experiment):
    import audit_keyframe_rgb_motion as base

    try:
        profile = route.with_name(route.name + ".active-profile")
        manifest = json.loads((profile / "config.json").read_bytes())
        plan = planner.ROOT / manifest["plan_directory"]
        planned = json.loads((plan / "config.json").read_bytes())
        previous = planner.inputs(
            planner.ROOT / planned["source_acquisition"],
            planner.ROOT / planned["source_predictions"],
        )
        paths = (
            list(previous.values())
            + list(planner.sources().values())
            + [
                profile / "config.json",
                profile / "report.json",
                plan / "config.json",
                plan / "plan.json",
                Path(__file__),
            ]
        )
        frozen = {path: digest(path) for path in paths}
        with active_pipeline(base):
            result = base.audit(route, experiment)
        if not result["valid"]:
            return result
        evaluator = route / "evaluator-audit.json"
        frozen[evaluator] = result["evaluator_sha256"]
        result["checks"]["active_evaluator_identity"] = (
            digest(evaluator) == frozen[evaluator]
        )
        if not result["checks"]["active_evaluator_identity"]:
            result.update(valid=False, evaluation=None)
            return result
        positions = json.loads(evaluator.read_bytes())["positions"]
        config = json.loads((experiment / "config.json").read_bytes())
        turns = []
        for pair in [item for item in config["pairs"] if item["group"] == "leg_one"]:
            number = pair["id"].rsplit("-", 1)[1]
            second = next(
                item for item in config["pairs"] if item["id"] == f"leg_two-{number}"
            )
            vectors = [
                np.array(
                    [
                        positions[item["second"]][key] - positions[item["first"]][key]
                        for key in ("xpos", "zpos")
                    ]
                )
                for item in (pair, second)
            ]
            lengths = [float(np.linalg.norm(vector)) for vector in vectors]
            thresholds = config["physical_turn_thresholds"]
            angle = (
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                vectors[0] @ vectors[1] / (lengths[0] * lengths[1]),
                                -1,
                                1,
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
        result["checks"]["active_prior_profile_sources_stable"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        result["valid"] = all(result["checks"].values())
        result["active_auditor_sha256"] = frozen[Path(__file__)]
        if not result["valid"]:
            result["evaluation"] = None
        else:
            result["evaluation"].update(
                physical_turns=turns,
                observed_noncollinear_trials=sum(
                    item["observed_noncollinear"] for item in turns
                ),
            )
        return result
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return {
            "valid": False,
            "checks": {"active_audit_execution": False},
            "evaluation": None,
            "failure": str(error),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite active motion audit")
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
            },
            sort_keys=True,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
