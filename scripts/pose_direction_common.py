"""Public-only baseline replay followed by a fixed direction-constrained search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pose_competition_common as original

from minecraft_scientist.exploration.keyframe_motion import KeyframeSelection
from minecraft_scientist.exploration.pose_competition import PoseCompetitionConfig
from minecraft_scientist.exploration.pose_direction_search import (
    DirectionSearchConfig,
    inspect_direction_search,
)

inputs = original.inputs


def sources():
    return original.sources() | {
        "pose_direction_search.py": Path(inspect_direction_search.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "probe_pose_direction.py": Path(__file__).with_name("probe_pose_direction.py"),
        "audit_pose_direction.py": Path(__file__).with_name("audit_pose_direction.py"),
    }


def configuration(route, experiment):
    return {
        **original.configuration(route, experiment),
        "schema": "public-pose-direction-search-v1",
        "policy": DirectionSearchConfig().model_dump(mode="json"),
        "scope": "development-only fixed direction family with reference sanity replay; no witness is not uniqueness; no pose replacement or navigation",
    }


def replay(route, experiment, policy, *, results=None):
    if policy.geometry != PoseCompetitionConfig().geometry:
        raise ValueError("direction search must preserve original fixed geometry")
    # Every original RGB prefix and bounded-map trial is recomputed, not trusted
    # from serialized predictions. No private telemetry is read by this replay.
    baseline = original.replay(route, experiment, PoseCompetitionConfig())
    config = json.loads((experiment / "config.json").read_bytes())
    predicted = json.loads((experiment / "report.json").read_bytes())
    results = [] if results is None else results
    for index, (prior, record) in enumerate(
        zip(baseline, predicted["results"], strict=True)
    ):
        selection = KeyframeSelection.model_validate(record["selection"])
        if (
            prior["pair"] != record["pair"]
            or prior["selected_offset"] != selection.selected_offset
        ):
            raise ValueError("replayed pair or selected interval differs")
        evidence = inspect_direction_search(
            selection.selected(),
            np.asarray(config["camera_intrinsics"]),
            seed=config["seed"] + index,
            config=policy,
        )
        results.append(
            {
                "pair": record["pair"],
                "selected_offset": selection.selected_offset,
                "original_competition": prior["competition"],
                "direction_search": evidence.model_dump(mode="json"),
            }
        )
    return results
