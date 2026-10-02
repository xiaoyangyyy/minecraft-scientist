"""Replay every public prefix/shadow result before separately scoring the veto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_motion_common import digest, public_inputs
from shadow_motion_common import runtime_sources, shadow_pipeline

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.keyframe_motion import KeyframeSelection
from minecraft_scientist.exploration.shadow_motion import ShadowMotionEvidence


def _audit(route, experiment):
    import audit_keyframe_rgb_motion as base

    frozen = {
        path: digest(path)
        for path in list(public_inputs(route).values())
        + list(runtime_sources().values())
        + [
            Path(__file__),
            experiment / "config.json",
            experiment / "report.json",
        ]
    }
    with shadow_pipeline(base):
        result = base.audit(route, experiment)
    if not result["valid"]:
        return result
    # Base independently reconstructs ALL prefix and shadow computations before
    # opening telemetry. No private value can choose a prefix or validation pose.
    evaluator = route / "evaluator-audit.json"
    frozen[evaluator] = result["evaluator_sha256"]
    result["checks"]["shadow_evaluator_identity"] = (
        digest(evaluator) == frozen[evaluator]
    )
    if not result["checks"]["shadow_evaluator_identity"]:
        result.update(valid=False, evaluation=None)
        return result
    positions = json.loads(evaluator.read_bytes())["positions"]
    config = json.loads((experiment / "config.json").read_bytes())
    predictions = json.loads((experiment / "report.json").read_bytes())["results"]
    thresholds = config["evaluation_thresholds"]
    scored = []
    candidates_match = True
    for item in predictions:
        selection = KeyframeSelection.model_validate(item["selection"])
        shadow = ShadowMotionEvidence.model_validate(item["shadow"])
        candidates_match &= shadow.candidate == selection.selected()
        selected = selection.selected()
        score = (
            None
            if selected is None
            else score_relative_motion(
                shadow.model_dump(),
                positions[item["pair"]["first"]],
                positions[item["pair"]["first"] + selected.end_offset],
                minimum_translation=thresholds["minimum_translation_blocks"],
                maximum_direction_error=thresholds["maximum_direction_error_degrees"],
                maximum_rotation_error=thresholds["maximum_rotation_error_degrees"],
            )
        )
        scored.append(
            {
                "pair": item["pair"],
                "selected_offset": selection.selected_offset,
                "shadow_supported": shadow.supported,
                "shadow_reason": shadow.reason,
                "shadow_local": score,
            }
        )
    groups = {}
    for group in sorted({item["pair"]["group"] for item in scored}):
        items = [item for item in scored if item["pair"]["group"] == group]
        local = [
            item["shadow_local"] for item in items if item["shadow_local"] is not None
        ]
        groups[group] = {
            "trials": len(items),
            "latched_candidates": len(local),
            "shadow_accepted": sum(item["supported"] for item in local),
            "correct_shadow_movements": sum(
                item["supported"] and item["translated"] and item["correct"]
                for item in local
            ),
            "wrong_shadow_movements": sum(
                item["supported"] and item["translated"] and not item["correct"]
                for item in local
            ),
            "false_shadow_translations": sum(
                item["supported"] and not item["translated"] for item in local
            ),
        }
    result["checks"].update(
        shadow_candidate_never_reselected=candidates_match,
        shadow_sources_inputs_stable=all(
            digest(path) == value for path, value in frozen.items()
        ),
    )
    result["valid"] = all(result["checks"].values())
    result["shadow_auditor_sha256"] = frozen[Path(__file__)]
    if not result["valid"]:
        result["evaluation"] = None
        return result
    result["evaluation"].update(shadow_groups=groups, shadow_scores=scored)
    return result


def audit(route, experiment):
    try:
        return _audit(route, experiment)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "valid": False,
            "checks": {"shadow_audit_completed": False},
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
        raise FileExistsError("refusing to overwrite shadow motion audit")
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
                else result["evaluation"]["shadow_groups"],
            },
            sort_keys=True,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
