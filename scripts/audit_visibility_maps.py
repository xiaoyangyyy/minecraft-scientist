"""Replay every public adaptive control and map before private motion scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import visibility_capture_common as common
from audit_action_switch import _physical_switch
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeSelection,
)


def _control_turn(positions, trial):
    switch = next(
        (item for item in trial["decisions"] if item["instruction"] == "switch"), None
    )
    if switch is None:
        return None
    slots = (
        trial["anchor_frame"],
        trial["anchor_frame"] + switch["tick"],
        trial["end_frame"],
    )
    points = [
        np.array([positions[slot][key] for key in ("xpos", "ypos", "zpos")])
        for slot in slots
    ]
    left, right = points[1] - points[0], points[2] - points[1]
    lengths = tuple(float(np.linalg.norm(item)) for item in (left, right))
    angle = (
        None
        if min(lengths) < 0.15
        else math.degrees(
            math.acos(float(np.clip(left @ right / np.prod(lengths), -1, 1)))
        )
    )
    return {
        "displacements_blocks": lengths,
        "angle_degrees": angle,
        "effective_noncollinear": angle is not None and 45 <= angle <= 135,
    }


def audit(route, experiment):
    checks = {}

    def reject(error=None):
        return {"valid": False, "checks": checks, "evaluation": None, "failure": error}

    try:
        config = json.loads((experiment / "config.json").read_bytes())
        predicted = json.loads((experiment / "report.json").read_bytes())
        paths, sources = common.inputs(route), common.sources()
        checks = {
            "completed_prediction": predicted["completed"] is True
            and predicted["failure"] is None
            and predicted["input_sources_unchanged"] is True,
            "no_navigation_promotion": predicted["navigation_accepted"] is False,
            "configuration_identity": predicted["config_sha256"]
            == digest(experiment / "config.json"),
            "complete_current_public_inputs": config["input_sha256"]
            == {name: digest(path) for name, path in paths.items()},
            "complete_current_runtime_sources": config["source_sha256"]
            == {name: digest(path) for name, path in sources.items()},
            "complete_source_snapshots": {
                path.relative_to(experiment / "source-snapshot").as_posix()
                for path in (experiment / "source-snapshot").rglob("*.py")
            }
            == set(sources)
            and all(
                digest(experiment / "source-snapshot" / name) == value
                for name, value in config["source_sha256"].items()
            ),
        }
        if not all(checks.values()):
            return reject()
        frozen = {
            path: digest(path)
            for path in list(paths.values())
            + list(sources.values())
            + [
                experiment / "config.json",
                experiment / "report.json",
                Path(__file__),
                Path(_physical_switch.__code__.co_filename),
            ]
            + list((experiment / "source-snapshot").rglob("*.py"))
        }
        acquisition, raw, _, acquisition_checks = common.validate_capture(route)
        checks.update(
            {"capture/" + name: value for name, value in acquisition_checks.items()}
        )
        checks["same_public_acquisition_replay"] = (
            acquisition_checks == predicted["public_acquisition_checks"]
        )
        checks["registered_public_calibration_policy"] = _numeric_equal(
            {
                key: value
                for key, value in config.items()
                if key not in ("source_sha256", "input_sha256")
            },
            common.configuration(route, acquisition, raw, config["seed"]),
        )
        checks["all_declared_pairs_retained"] = (
            [item["pair"] for item in predicted["results"]]
            == config["pairs"]
            == common.pairs(raw)
        )
        if not all(checks.values()):
            return reject()
        replay = common.infer(route, raw, config)
        checks["all_public_maps_prefixes_and_future_pixels_replayed"] = _numeric_equal(
            replay, predicted["results"]
        )
        checks["public_bytes_stable_before_private_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        if not all(checks.values()):
            return reject()
        # Privileged information is first read only after complete public replay.
        evaluator = route / "evaluator-audit.json"
        evaluator_digest = digest(evaluator)
        positions = json.loads(evaluator.read_bytes())["positions"]
        checks["evaluator_alignment_and_finite_values"] = len(positions) == raw[
            "frame_count"
        ] and all(
            isinstance(point, dict)
            and all(
                type(point.get(key)) in (int, float) and math.isfinite(point[key])
                for key in ("xpos", "ypos", "zpos", "yaw", "pitch")
            )
            and -90 <= point["pitch"] <= 90
            for point in positions
        )
        if not all(checks.values()):
            return reject()
        thresholds = config["evaluation_thresholds"]

        def score(motion, first, second):
            return score_relative_motion(
                motion.model_dump(mode="json"),
                positions[first],
                positions[second],
                minimum_translation=thresholds["minimum_translation_blocks"],
                maximum_direction_error=thresholds["maximum_direction_error_degrees"],
                maximum_rotation_error=thresholds["maximum_rotation_error_degrees"],
            )

        samples, groups = [], {}
        for record in replay:
            pair, saved = record["pair"], record["selection"]
            selected_score, terminal, prefix_scores = None, None, []
            selected_offset = None
            if saved is not None:
                selection = KeyframeSelection.model_validate(saved)
                if tuple(
                    item.end_offset for item in selection.attempts
                ) != common.expected_offsets(pair, raw):
                    raise ValueError(
                        "missing chronological prefix or changed first-supported latch"
                    )
                selected, selected_offset = (
                    selection.selected(),
                    selection.selected_offset,
                )
                selected_score = (
                    None
                    if selected is None
                    else score(
                        selected.motion,
                        pair["first"],
                        pair["first"] + selected.end_offset,
                    )
                )
                terminal = score(
                    selection.attempts[-1].motion, pair["first"], pair["second"]
                )
                prefix_scores = [
                    {
                        "offset": item.end_offset,
                        "score": score(
                            item.motion, pair["first"], pair["first"] + item.end_offset
                        ),
                    }
                    for item in selection.attempts
                ]
            future = record["action_switch"]
            retained = (
                future is not None
                and future["reference_future"] is not None
                and future["reference_future"]["status"] == "consistent"
                and future["comparison"]
                not in ("ambiguity_survives", "incomplete_comparison")
            )
            group = groups.setdefault(
                pair["group"],
                {
                    "trials": 0,
                    "selected": 0,
                    "selected_correct": 0,
                    "selected_wrong": 0,
                    "terminal_correct_moving": 0,
                    "terminal_wrong_supported": 0,
                    "stationary_controls": 0,
                    "stationary_false_translations": 0,
                    "retained_correct": 0,
                    "retained_wrong": 0,
                },
            )
            group["trials"] += 1
            if selected_score is not None:
                group["selected"] += 1
                kind = "correct" if selected_score["correct"] else "wrong"
                group["selected_" + kind] += 1
                group["retained_" + kind] += int(retained)
            if terminal is not None:
                group["terminal_correct_moving"] += int(
                    terminal["correct"]
                    and terminal["translated"]
                    and terminal["supported"]
                )
                group["terminal_wrong_supported"] += int(
                    terminal["supported"] and not terminal["correct"]
                )
                if pair["group"].endswith("control") and not terminal["translated"]:
                    group["stationary_controls"] += 1
                    group["stationary_false_translations"] += int(terminal["supported"])
            samples.append(
                {
                    "pair": pair,
                    "selected_offset": selected_offset,
                    "selected_local": selected_score,
                    "terminal": terminal,
                    "all_prefixes": prefix_scores,
                    "comparison": None if future is None else future["comparison"],
                    "diagnostic_retained": bool(retained),
                    "shadow_view_diversity": None
                    if future is None
                    else _physical_switch(positions, future["plan"]),
                }
            )
        physical_trials = [_control_turn(positions, trial) for trial in raw["trials"]]
        checks["public_bytes_stable_through_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
        checks["evaluator_bytes_stable_through_scoring"] = (
            digest(evaluator) == evaluator_digest
        )
        valid = all(checks.values())
        return {
            "valid": valid,
            "checks": checks,
            "failure": None,
            "evaluator_sha256": evaluator_digest,
            "evaluation": None
            if not valid
            else {
                "groups": groups,
                "samples": samples,
                "physical_trials": physical_trials,
                "frame_count": raw["frame_count"],
                "navigation_accepted": False,
                "scope": "single development seed adaptive control; first-leg frozen map shadow and all original trial/control denominators; not matched-path or held-out navigation success",
            },
        }
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        StopIteration,
    ) as error:
        return reject({"type": type(error).__name__, "message": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite adaptive-map audit")
    result = audit(args.route, args.experiment)
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
