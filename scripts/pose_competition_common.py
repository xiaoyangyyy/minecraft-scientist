"""Source-bound public replay for a bounded competing-map diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import active_motion_pipeline as active
import keyframe_motion_common as keyframes
import numpy as np
import plan_active_rgb_motion as planner
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, public_inputs

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.keyframe_motion import KeyframeSelection
from minecraft_scientist.exploration.pose_competition import (
    PoseCompetitionConfig,
    inspect_pose_competition,
)


def sources():
    return active.runtime_sources() | {
        "pose_competition.py": Path(inspect_pose_competition.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "inspect_pose_competition.py": Path(__file__).with_name(
            "inspect_pose_competition.py"
        ),
        "audit_pose_competition.py": Path(__file__).with_name(
            "audit_pose_competition.py"
        ),
        "audit_active_rgb_motion.py": Path(__file__).with_name(
            "audit_active_rgb_motion.py"
        ),
        "audit_keyframe_rgb_motion.py": Path(__file__).with_name(
            "audit_keyframe_rgb_motion.py"
        ),
        "relative_motion.py": Path(score_relative_motion.__code__.co_filename),
        "audit_dense_correspondence.py": Path(_numeric_equal.__code__.co_filename),
    }


def inputs(route, experiment):
    profile = route.with_name(route.name + ".active-profile")
    manifest = json.loads((profile / "config.json").read_bytes())
    plan = planner.ROOT / manifest["plan_directory"]
    previous = json.loads((plan / "config.json").read_bytes())
    paths = list(public_inputs(route).values()) + list(
        planner.inputs(
            planner.ROOT / previous["source_acquisition"],
            planner.ROOT / previous["source_predictions"],
        ).values()
    )
    for directory, result in (
        (experiment, "report.json"),
        (profile, "report.json"),
        (plan, "plan.json"),
    ):
        paths.extend([directory / "config.json", directory / result])
        paths.extend(sorted((directory / "source-snapshot").glob("*.py")))
    return {path.resolve().relative_to(planner.ROOT).as_posix(): path for path in paths}


def replay(route, experiment, policy, *, results=None):
    """Reconstruct the old public evidence and all new witnesses; no private reads."""
    acquisition, raw, _ = active.validate_acquisition(route)
    config = json.loads((experiment / "config.json").read_bytes())
    predicted = json.loads((experiment / "report.json").read_bytes())
    expected = active.configuration(route, acquisition, raw, config["seed"])
    runtime, public = active.runtime_sources(), public_inputs(route)
    if (
        predicted["completed"] is not True
        or predicted["input_sources_unchanged"] is not True
        or predicted["failure"] is not None
        or predicted["navigation_accepted"] is not False
        or predicted["suite"] != "chronological_keyframe_development"
        or predicted["config_sha256"] != digest(experiment / "config.json")
        or not _numeric_equal(
            {
                key: value
                for key, value in config.items()
                if key not in ("source_sha256", "input_sha256")
            },
            expected,
        )
        or config["source_sha256"]
        != {name: digest(path) for name, path in runtime.items()}
        or config["input_sha256"]
        != {name: digest(path) for name, path in public.items()}
        or {path.name for path in (experiment / "source-snapshot").glob("*.py")}
        != set(runtime)
        or any(
            digest(experiment / "source-snapshot" / name) != value
            for name, value in config["source_sha256"].items()
        )
        or [item["pair"] for item in predicted["results"]] != raw["pairs"]
        or config["seed"] + len(raw["pairs"]) >= 2**31 - 1024
        or config["multiview"] != policy.geometry.model_dump(mode="json")
    ):
        raise ValueError(
            "invalid frozen public active predictions or competition geometry"
        )
    results = [] if results is None else results
    for index, pair in enumerate(raw["pairs"]):
        inferred = keyframes.infer_pair(route, pair, config, index)
        if not _numeric_equal(inferred, predicted["results"][index]):
            raise ValueError("original public prefix replay differs")
        selection = KeyframeSelection.model_validate(inferred["selection"])
        competition = inspect_pose_competition(
            selection.selected(),
            np.asarray(config["camera_intrinsics"]),
            seed=config["seed"] + index,
            config=policy,
        )
        results.append(
            {
                "pair": pair,
                "selected_offset": selection.selected_offset,
                "competition": competition.model_dump(mode="json"),
            }
        )
    return results


def configuration(route, experiment):
    return {
        "schema": "public-pose-competition-v1",
        "acquisition": route.resolve().relative_to(planner.ROOT).as_posix(),
        "prior_predictions": experiment.resolve().relative_to(planner.ROOT).as_posix(),
        "policy": PoseCompetitionConfig().model_dump(mode="json"),
        "scope": "development diagnosis; incomplete competing-map family, not unique pose or navigation acceptance",
    }
