"""Replay every public prefix before separately evaluating local and terminal poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, public_inputs, validate_acquisition
from keyframe_motion_common import configuration, infer_pair, runtime_sources

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeMotionConfig,
    KeyframeSelection,
)


def audit(route: Path, experiment: Path):
    checks = {}

    def reject(error=None):
        return {"valid": False, "checks": checks, "evaluation": None, "failure": error}

    try:
        acquisition, report, checks = validate_acquisition(route)
        config = json.loads((experiment / "config.json").read_bytes())
        predictions = json.loads((experiment / "report.json").read_bytes())
        sources, inputs = runtime_sources(), public_inputs(route)
        checks.update(
            {
                "registered_selector": config.get("selector")
                == "first_supported_chronological_prefix",
                "runtime_source_set": set(config["source_sha256"]) == set(sources)
                and {
                    path.name for path in (experiment / "source-snapshot").glob("*.py")
                }
                == set(sources),
                "complete_input_set": set(config["input_sha256"]) == set(inputs),
                "preregistered_config": digest(experiment / "config.json")
                == predictions["config_sha256"],
                "prediction_completed": predictions["suite"]
                == "chronological_keyframe_development"
                and predictions["completed"] is True
                and predictions["input_sources_unchanged"] is True
                and predictions["failure"] is None,
                "complete_pairs": [item["pair"] for item in predictions["results"]]
                == report["pairs"],
                "no_navigation_promotion": predictions["navigation_accepted"] is False,
            }
        )
        if not all(checks.values()):
            return reject()
        checks.update(
            {
                "input_bytes_verified": all(
                    digest(path) == config["input_sha256"][name]
                    for name, path in inputs.items()
                ),
                "current_sources_verified": all(
                    digest(path) == config["source_sha256"][name]
                    for name, path in sources.items()
                ),
                "source_snapshots_verified": all(
                    digest(experiment / "source-snapshot" / name) == value
                    for name, value in config["source_sha256"].items()
                ),
            }
        )
        if not all(checks.values()):
            return reject()
        expected = configuration(route, acquisition, report, config["seed"])
        checks["public_calibration_configuration"] = _numeric_equal(
            {
                key: value
                for key, value in config.items()
                if key not in ("input_sha256", "source_sha256")
            },
            expected,
        )
        if not all(checks.values()):
            return reject()
        policy = KeyframeMotionConfig.model_validate(config["keyframe"])
        selections = [
            KeyframeSelection.model_validate(item["selection"])
            for item in predictions["results"]
        ]
        checks["complete_prefixes_first_supported"] = all(
            selection.terminal_offset == pair["second"] - pair["first"]
            and tuple(item.end_offset for item in selection.attempts)
            == policy.offsets(selection.terminal_offset)
            for selection, pair in zip(selections, report["pairs"], strict=True)
        )
        if not all(checks.values()):
            return reject()
        observed = {
            path: digest(path)
            for path in list(inputs.values())
            + list(sources.values())
            + [
                experiment / "config.json",
                experiment / "report.json",
                Path(__file__),
                Path(_numeric_equal.__code__.co_filename),
                Path(score_relative_motion.__code__.co_filename),
            ]
        }
        replay = [
            infer_pair(route, pair, config, index)
            for index, pair in enumerate(report["pairs"])
        ]
        checks["all_public_prefixes_replayed"] = _numeric_equal(
            replay, predictions["results"]
        )
        checks["sources_inputs_stable"] = all(
            digest(path) == value for path, value in observed.items()
        )
        if not all(checks.values()):
            return reject()
        # No telemetry is opened before all inputs, declarations and EVERY prefix
        # have been independently reconstructed, including failed alternatives.
        evaluator_path = route / "evaluator-audit.json"
        evaluator_hash = digest(evaluator_path)
        positions = json.loads(evaluator_path.read_bytes())["positions"]
        checks["evaluator_alignment"] = len(positions) == report["frame_count"] and all(
            isinstance(position, dict)
            and all(
                type(position.get(key)) in (int, float) and math.isfinite(position[key])
                for key in ("xpos", "ypos", "zpos", "yaw", "pitch")
            )
            for position in positions
        )
        if not all(checks.values()):
            return reject()
        thresholds = config["evaluation_thresholds"]

        def score(motion, first, second):
            return score_relative_motion(
                motion.model_dump(),
                positions[first],
                positions[second],
                minimum_translation=thresholds["minimum_translation_blocks"],
                maximum_direction_error=thresholds["maximum_direction_error_degrees"],
                maximum_rotation_error=thresholds["maximum_rotation_error_degrees"],
            )

        scored = []
        for pair, selection in zip(report["pairs"], selections, strict=True):
            selected = selection.selected()
            scored.append(
                {
                    "pair": pair,
                    "selected_offset": selection.selected_offset,
                    "selected_local": None
                    if selected is None
                    else score(
                        selected.motion,
                        pair["first"],
                        pair["first"] + selected.end_offset,
                    ),
                    "terminal_baseline": score(
                        selection.attempts[-1].motion, pair["first"], pair["second"]
                    ),
                    "all_prefixes": [
                        {
                            "end_offset": attempt.end_offset,
                            "score": score(
                                attempt.motion,
                                pair["first"],
                                pair["first"] + attempt.end_offset,
                            ),
                        }
                        for attempt in selection.attempts
                    ],
                }
            )
        groups = {}
        for group in sorted({item["pair"]["group"] for item in scored}):
            items = [item for item in scored if item["pair"]["group"] == group]
            local = [
                item["selected_local"]
                for item in items
                if item["selected_local"] is not None
            ]
            terminal = [item["terminal_baseline"] for item in items]
            groups[group] = {
                "trials": len(items),
                "selected_prefixes": len(local),
                "correct_selected_local_movements": sum(
                    item["translated"] and item["correct"] for item in local
                ),
                "wrong_selected_local_movements": sum(
                    item["translated"] and not item["correct"] for item in local
                ),
                "false_selected_local_translations": sum(
                    not item["translated"] for item in local
                ),
                "terminal_actual_movements": sum(
                    item["translated"] for item in terminal
                ),
                "correct_terminal_baseline_movements": sum(
                    item["translated"] and item["correct"] for item in terminal
                ),
                "terminal_actual_controls": sum(
                    not item["translated"] for item in terminal
                ),
                "correct_terminal_baseline_controls": sum(
                    not item["translated"] and item["correct"] for item in terminal
                ),
                "prefix_evaluations": sum(len(item["all_prefixes"]) for item in items),
            }
        checks["all_evaluation_inputs_stable"] = digest(
            evaluator_path
        ) == evaluator_hash and all(
            digest(path) == value for path, value in observed.items()
        )
        if not all(checks.values()):
            return reject()
        return {
            "valid": True,
            "checks": checks,
            "failure": None,
            "auditor_sha256": observed[Path(__file__)],
            "evaluator_sha256": evaluator_hash,
            "motion_evaluator_sha256": observed[
                Path(score_relative_motion.__code__.co_filename)
            ],
            "evaluation": {
                "groups": groups,
                "scores": scored,
                "navigation_accepted": False,
                "scope": config["scope"],
            },
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        checks["audit_execution"] = False
        return reject(str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite keyframe audit")
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
                "evaluation": None
                if result["evaluation"] is None
                else result["evaluation"]["groups"],
            },
            sort_keys=True,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
