"""Replay a complete light/weather factorial without treating pairs as seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_controlled_rgb_motion import _equivalent, audit, score_relative_motion
from controlled_motion_common import (
    capture_sources,
    digest,
    public_inputs,
    runtime_sources,
)

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


def summarize(cells: list[dict]) -> dict:
    if len(cells) != 4:
        raise ValueError(
            "condition diagnosis requires all four independent acquisitions"
        )
    configs = [MotionProbeConfig.model_validate(cell["config"]) for cell in cells]
    keys = [(item.time_of_day, item.weather) for item in configs]
    if set(keys) != {
        (time, weather)
        for time in ("noon", "midnight")
        for weather in ("clear", "thunder")
    }:
        raise ValueError("require exactly one acquisition in each light/weather cell")
    comparable = [
        item.model_dump(exclude={"time_of_day", "weather"}) for item in configs
    ]
    if any(item != comparable[0] for item in comparable):
        raise ValueError("condition cells must share seed, camera and action budgets")
    if any(
        cell["audit"]["valid"] is not True
        or not all(value is True for value in cell["audit"]["checks"].values())
        for cell in cells
    ):
        raise ValueError("a failed component audit cannot become condition evidence")
    reference = cells[0]["positions"]
    fields = ("xpos", "ypos", "zpos", "yaw", "pitch")
    reference_array = np.array(
        [[point[key] for key in fields] for point in reference], dtype=float
    )
    differences = []
    for cell in cells:
        observed = np.array(
            [[point[key] for key in fields] for point in cell["positions"]], dtype=float
        )
        if (
            observed.shape != reference_array.shape
            or observed.shape[0] < 2
            or not np.isfinite(observed).all()
            or not np.isfinite(reference_array).all()
        ):
            raise ValueError("all evaluator trajectories must be finite and aligned")
        differences.append(float(np.max(np.abs(observed - reference_array))))
    results = []
    for config, cell, difference in zip(configs, cells, differences, strict=True):
        counts = [
            len(item["tracks"]["track_ids"])
            for item in cell["predictions"]
            if item["pair"]["group"] == "settled"
        ]
        if len(counts) != 8:
            raise ValueError("every settled trial must be retained")
        results.append(
            {
                "time_of_day": config.time_of_day,
                "weather": config.weather,
                "groups": cell["audit"]["evaluation"]["groups"],
                "settled_track_counts": counts,
                "settled_track_median": float(np.median(counts)),
                "horizontal_focal_pixels": cell["inference_config"]["focal_pixels"],
                "maximum_trajectory_component_difference": difference,
            }
        )
    return {
        "scope": "one natural seed, one session per factorial cell; overlapping trials are not independent seed samples",
        "seed": configs[0].seed,
        "matched_trajectories": max(differences) <= 1e-5,
        "maximum_trajectory_component_difference": max(differences),
        "sessions_per_cell": 1,
        "cells": sorted(
            results, key=lambda item: (item["time_of_day"], item["weather"])
        ),
        "navigation_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("ACQUISITION", "INFERENCE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite condition audit")
    if (
        len(args.case) != 4
        or len({tuple(path.resolve() for path in pair) for pair in args.case}) != 4
        or len({pair[0].resolve() for pair in args.case}) != 4
    ):
        raise ValueError("four distinct acquisition/inference cases are required")
    sources = (
        runtime_sources()
        | capture_sources()
        | {
            "condition_auditor.py": Path(__file__),
            "motion_auditor.py": Path(audit.__code__.co_filename),
            "motion_evaluator.py": Path(score_relative_motion.__code__.co_filename),
            "numeric_comparison.py": Path(_equivalent.__code__.co_filename),
        }
    )
    frozen = {path: digest(path) for path in sources.values()}
    inputs = {}
    for number, (route, experiment) in enumerate(args.case):
        files = public_inputs(route) | {
            "evaluator-audit.json": route / "evaluator-audit.json",
            "inference/config.json": experiment / "config.json",
            "inference/report.json": experiment / "report.json",
        }
        inputs.update(
            {
                f"case-{number}/{name}": {
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                }
                for name, path in files.items()
            }
        )
        frozen.update({path: digest(path) for path in files.values()})
    cells = []
    for route, experiment in args.case:
        result = audit(route, experiment)
        if not result["valid"]:
            raise ValueError(f"failed component audit: {route}: {result['checks']}")
        cells.append(
            {
                "config": json.loads((route / "config.json").read_bytes())["config"],
                "positions": json.loads((route / "evaluator-audit.json").read_bytes())[
                    "positions"
                ],
                "predictions": json.loads((experiment / "report.json").read_bytes())[
                    "results"
                ],
                "inference_config": json.loads(
                    (experiment / "config.json").read_bytes()
                ),
                "audit": result,
            }
        )
        print(
            json.dumps(
                {
                    "case": str(route),
                    "valid": True,
                    "replayed_checks": len(result["checks"]),
                }
            ),
            flush=True,
        )
    evaluation = summarize(cells)
    checks = {
        "complete_factorial": True,
        "component_replays_valid": True,
        "matched_public_configuration": True,
        "sources_inputs_unchanged": all(
            digest(path) == value for path, value in frozen.items()
        ),
        "physical_trajectories_matched": evaluation["matched_trajectories"],
        "no_navigation_promotion": evaluation["navigation_accepted"] is False,
    }
    output = {
        "valid": all(checks.values()),
        "checks": checks,
        "input_sha256": inputs,
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"valid": output["valid"], "checks": checks, "evaluation": evaluation},
            sort_keys=True,
        )
    )
    if not output["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
