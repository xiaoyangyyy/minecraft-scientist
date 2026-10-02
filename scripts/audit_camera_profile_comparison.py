"""Replay both explicit camera-profile cases before comparing public motion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_camera_profile import audit as profile_audit
from audit_controlled_rgb_motion import _equivalent
from audit_controlled_rgb_motion import audit as motion_audit
from controlled_motion_common import digest, public_inputs, rgb, runtime_sources

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


def summarize(cells):
    if len(cells) != 2 or {cell["profile_config"]["enabled"] for cell in cells} != {
        True,
        False,
    }:
        raise ValueError("require one enabled and one disabled camera acquisition")
    ordered = sorted(cells, key=lambda cell: not cell["profile_config"]["enabled"])
    if any(
        type(cell["profile_config"]["enabled"]) is not bool
        or cell["profile_audit"]["valid"] is not True
        or cell["motion_audit"]["valid"] is not True
        or not all(
            value is True
            for audit in (cell["profile_audit"], cell["motion_audit"])
            for value in audit["checks"].values()
        )
        for cell in ordered
    ):
        raise ValueError("invalid component evidence cannot become camera comparison")
    configs = [MotionProbeConfig.model_validate(cell["config"]) for cell in ordered]
    if configs[0] != configs[1]:
        raise ValueError("camera cases must share world, rendering and action budgets")
    profiles = [cell["profile_config"] for cell in ordered]
    if any(
        profiles[0][key] != profiles[1][key]
        for key in (
            "original_runtime_sha256",
            "original_options_sha256",
            "source_sha256",
        )
    ):
        raise ValueError("camera cases must start from identical runtime and source")
    ignored = {
        "camera_intrinsics",
        "focal_pixels",
        "focal_sample_count",
        "focal_relative_mad",
        "input_sha256",
    }
    inference = [
        {
            key: value
            for key, value in cell["inference_config"].items()
            if key not in ignored
        }
        for cell in ordered
    ]
    if "variant" in inference[0] or not _equivalent(inference[0], inference[1]):
        raise ValueError("paired camera ablation requires the same baseline inference")
    fields = ("xpos", "ypos", "zpos", "yaw", "pitch")
    trajectories = [
        np.array([[point[key] for key in fields] for point in cell["positions"]])
        for cell in ordered
    ]
    if (
        trajectories[0].shape != trajectories[1].shape
        or trajectories[0].shape != (201, 5)
        or any(not np.isfinite(points).all() for points in trajectories)
    ):
        raise ValueError("require complete finite aligned physical trajectories")
    difference = float(np.max(np.abs(trajectories[0] - trajectories[1])))
    results = []
    for cell in ordered:
        counts = [
            len(item["tracks"]["track_ids"])
            for item in cell["predictions"]
            if item["pair"]["group"] == "settled"
        ]
        if len(counts) != 8:
            raise ValueError("every settled trial must remain in the comparison")
        results.append(
            {
                "bobbing_enabled": cell["profile_config"]["enabled"],
                "groups": cell["motion_audit"]["evaluation"]["groups"],
                "settled_track_counts": counts,
                "settled_track_median": float(np.median(counts)),
                "focal_pixels": cell["inference_config"]["focal_pixels"],
            }
        )
    return {
        "scope": "single development seed; one session per explicit camera profile; not statistical generalization or navigation acceptance",
        "seed": configs[0].seed,
        "sessions_per_setting": 1,
        "maximum_trajectory_component_difference": difference,
        "matched_trajectories": difference <= 1e-5,
        "cells": results,
        "navigation_accepted": False,
    }


def compare_pixels(first, second):
    differences = []
    for slot in range(201):
        images = [
            rgb(route / "frames" / f"frame-{slot:05d}.png") for route in (first, second)
        ]
        if images[0].shape != images[1].shape:
            raise ValueError("camera comparison requires the same public RGB shape")
        height = images[0].shape[0]
        # Fixed band crop reduces HUD contamination; it is not a complete hand
        # segmentation, so cross-session differences remain only descriptive.
        crops = [image[int(height * 0.08) : int(height * 0.75)] for image in images]
        differences.append(
            float(np.abs(crops[0].astype(float) - crops[1].astype(float)).mean())
        )
    return {
        "viewport_fraction": [0.08, 0.75],
        "frame_mean_absolute_rgb_difference": differences,
        "median_mean_absolute_rgb_difference": float(np.median(differences)),
        "nonidentical_frames": sum(value > 0 for value in differences),
        "scope": "descriptive cross-session pixel change; not proof of actual engine flag or causal motion improvement",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        nargs=3,
        action="append",
        required=True,
        metavar=("PROFILE", "ACQUISITION", "INFERENCE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite camera comparison")
    if len(args.case) != 2 or any(
        len({case[index].resolve() for case in args.case}) != 2 for index in range(3)
    ):
        raise ValueError(
            "two distinct profile/acquisition/inference cases are required"
        )
    sources = runtime_sources() | {
        "camera_comparison.py": Path(__file__),
        "profile_auditor.py": Path(profile_audit.__code__.co_filename),
        "motion_auditor.py": Path(motion_audit.__code__.co_filename),
        "numeric_comparison.py": Path(_equivalent.__code__.co_filename),
    }
    inputs = {}
    for index, (profile, route, experiment) in enumerate(args.case):
        paths = public_inputs(route) | {
            "evaluator-audit.json": route / "evaluator-audit.json",
            "inference/config.json": experiment / "config.json",
            "inference/report.json": experiment / "report.json",
        }
        paths.update(
            {
                f"profile/{path.relative_to(profile).as_posix()}": path
                for path in profile.rglob("*")
                if path.is_file()
            }
        )
        paths.update(
            {
                f"inference/source-snapshot/{path.name}": path
                for path in (experiment / "source-snapshot").glob("*.py")
            }
        )
        inputs.update({f"case-{index}/{name}": path for name, path in paths.items()})
    frozen = {
        path: digest(path) for path in list(inputs.values()) + list(sources.values())
    }
    cells = []
    for profile, route, experiment in args.case:
        camera_result = profile_audit(profile, route)
        if not camera_result["valid"]:
            raise ValueError(f"camera profile failed replay: {camera_result['checks']}")
        motion_result = motion_audit(route, experiment)
        if not motion_result["valid"]:
            raise ValueError(f"public motion failed replay: {motion_result['checks']}")
        cells.append(
            {
                "config": json.loads((route / "config.json").read_bytes())["config"],
                "profile_config": json.loads((profile / "config.json").read_bytes()),
                "profile_audit": camera_result,
                "motion_audit": motion_result,
                "inference_config": json.loads(
                    (experiment / "config.json").read_bytes()
                ),
                "predictions": json.loads((experiment / "report.json").read_bytes())[
                    "results"
                ],
                "positions": json.loads((route / "evaluator-audit.json").read_bytes())[
                    "positions"
                ],
            }
        )
        print(
            json.dumps(
                {
                    "case": str(route),
                    "profile_checks": len(camera_result["checks"]),
                    "motion_checks": len(motion_result["checks"]),
                }
            ),
            flush=True,
        )
    evaluation = summarize(cells)
    pixel_comparison = compare_pixels(args.case[0][1], args.case[1][1])
    checks = {
        "both_component_replays_valid": True,
        "paired_single_setting_ablation": True,
        "physical_trajectories_matched": evaluation["matched_trajectories"],
        "all_public_frames_compared": len(
            pixel_comparison["frame_mean_absolute_rgb_difference"]
        )
        == 201,
        "sources_inputs_unchanged": all(
            digest(path) == value for path, value in frozen.items()
        ),
        "no_navigation_promotion": evaluation["navigation_accepted"] is False,
    }
    output = {
        "valid": all(checks.values()),
        "checks": checks,
        "input_sha256": {
            name: {"path": str(path.resolve()), "sha256": frozen[path]}
            for name, path in inputs.items()
        },
        "source_sha256": {name: frozen[path] for name, path in sources.items()},
        "evaluation": evaluation,
        "pixel_comparison": pixel_comparison,
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
