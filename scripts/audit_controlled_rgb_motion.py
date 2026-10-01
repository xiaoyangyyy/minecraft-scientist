"""Replay controlled public RGB evidence, then separately score physical motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_dense_correspondence import _numeric_equal as _equivalent
from controlled_motion_common import (
    digest,
    infer_pair,
    inference_config,
    public_inputs,
    runtime_sources,
    validate_acquisition,
)

from minecraft_scientist.evaluation.relative_motion import score_relative_motion


def audit(route: Path, experiment: Path) -> dict:
    try:
        acquisition, report, checks = validate_acquisition(route)
    except (ValueError, KeyError, OSError, TypeError) as error:
        return {
            "valid": False,
            "checks": {"public_acquisition": False},
            "failure": str(error),
            "evaluation": None,
        }
    config = json.loads((experiment / "config.json").read_bytes())
    predictions = json.loads((experiment / "report.json").read_bytes())
    sources, inputs = runtime_sources(), public_inputs(route)
    checks.update(
        {
            "runtime_source_set": set(config["source_sha256"]) == set(sources)
            and {path.name for path in (experiment / "source-snapshot").glob("*.py")}
            == set(sources),
            "complete_input_set": set(config["input_sha256"]) == set(inputs),
            "preregistered_config": digest(experiment / "config.json")
            == predictions["config_sha256"],
            "prediction_completed": predictions["completed"] is True
            and predictions["input_sources_unchanged"] is True
            and predictions["failure"] is None,
            "all_predictions": [item["pair"] for item in predictions["results"]]
            == report["pairs"],
            "no_navigation_promotion": predictions["navigation_accepted"] is False,
        }
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    checks["input_bytes_verified"] = all(
        digest(path) == config["input_sha256"][name] for name, path in inputs.items()
    )
    checks["current_sources_verified"] = all(
        digest(path) == config["source_sha256"][name] for name, path in sources.items()
    )
    checks["source_snapshots_verified"] = all(
        digest(experiment / "source-snapshot" / name) == value
        for name, value in config["source_sha256"].items()
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    initial = {
        path: digest(path)
        for path in list(inputs.values())
        + list(sources.values())
        + [
            experiment / "config.json",
            experiment / "report.json",
            Path(__file__),
            Path(_equivalent.__code__.co_filename),
            Path(score_relative_motion.__code__.co_filename),
        ]
    }
    try:
        expected = inference_config(route, acquisition, report, config["seed"])
    except ValueError:
        checks["public_calibration_recomputed"] = False
        return {"valid": False, "checks": checks, "evaluation": None}
    checks["public_calibration_recomputed"] = _equivalent(
        {
            key: value
            for key, value in config.items()
            if key not in ("input_sha256", "source_sha256")
        },
        expected,
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    replay = [
        infer_pair(route, pair, config, index)
        for index, pair in enumerate(report["pairs"])
    ]
    checks["all_geometry_replayed"] = _equivalent(replay, predictions["results"])
    checks["sources_inputs_stable"] = all(
        digest(path) == value for path, value in initial.items()
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # First evaluator read occurs only after every public calculation has passed.
    positions = json.loads((route / "evaluator-audit.json").read_bytes())["positions"]
    checks["evaluator_alignment"] = len(positions) == report["frame_count"] and all(
        isinstance(position, dict)
        and all(
            type(position.get(key)) in (int, float) and math.isfinite(position[key])
            for key in ("xpos", "ypos", "zpos", "yaw", "pitch")
        )
        for position in positions
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    thresholds = config["evaluation_thresholds"]
    scores = []
    for item in predictions["results"]:
        pair = item["pair"]
        context = {
            "first": positions[pair["first"]],
            "second": positions[pair["second"]],
            "minimum_translation": thresholds["minimum_translation_blocks"],
            "maximum_direction_error": thresholds["maximum_direction_error_degrees"],
            "maximum_rotation_error": thresholds["maximum_rotation_error_degrees"],
        }
        scores.append(
            {
                "pair": pair,
                "chain_tracks": len(item["tracks"]["track_ids"]),
                "endpoint_tracks": len(item["endpoint_tracks"]["track_ids"]),
                "validated": score_relative_motion(item["motion"], **context),
                "endpoint": score_relative_motion(item["endpoint_motion"], **context),
            }
        )
    groups = {}
    for group in sorted({item["pair"]["group"] for item in scores}):
        items = [item for item in scores if item["pair"]["group"] == group]
        positive = [item for item in items if item["validated"]["translated"]]
        controls = [item for item in items if not item["validated"]["translated"]]
        groups[group] = {
            "trials": len(items),
            "actual_movement_trials": len(positive),
            "correct_validated_movements": sum(
                item["validated"]["correct"] for item in positive
            ),
            "wrong_supported_validated_movements": sum(
                item["validated"]["supported"] and not item["validated"]["correct"]
                for item in positive
            ),
            "correct_endpoint_movements": sum(
                item["endpoint"]["correct"] for item in positive
            ),
            "actual_controls": len(controls),
            "correct_validated_controls": sum(
                item["validated"]["correct"] for item in controls
            ),
            "correct_endpoint_controls": sum(
                item["endpoint"]["correct"] for item in controls
            ),
        }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "auditor_sha256": initial[Path(__file__)],
        "motion_evaluator_sha256": initial[
            Path(score_relative_motion.__code__.co_filename)
        ],
        "evaluation": {
            "groups": groups,
            "scores": scores,
            "navigation_accepted": False,
            "scope": config["scope"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite controlled motion audit")
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
