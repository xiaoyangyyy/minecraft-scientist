"""Public prefix replay, then fixed switched-action views of frozen hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pose_competition_common as original
from controlled_motion_common import rgb

from minecraft_scientist.exploration.action_switch_motion import (
    ActionSwitchConfig,
    inspect_action_switch,
    plan_action_switch,
)
from minecraft_scientist.exploration.compound_motion import (
    CompoundProbeConfig,
    compound_timeline,
)
from minecraft_scientist.exploration.keyframe_motion import KeyframeSelection
from minecraft_scientist.exploration.pose_competition import PoseCompetitionConfig
from minecraft_scientist.exploration.pose_direction_search import (
    inspect_direction_search,
)
from minecraft_scientist.exploration.temporal_tracks import track_frame_chain

inputs = original.inputs


def sources():
    return original.sources() | {
        "action_switch_motion.py": Path(inspect_action_switch.__code__.co_filename),
        "pose_direction_search.py": Path(inspect_direction_search.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "probe_action_switch.py": Path(__file__).with_name("probe_action_switch.py"),
        "audit_action_switch.py": Path(__file__).with_name("audit_action_switch.py"),
    }


def configuration(route, experiment):
    return {
        **original.configuration(route, experiment),
        "schema": "public-action-switch-maps-v1",
        "policy": ActionSwitchConfig().model_dump(mode="json"),
        "scope": "development temporal holdout; midpoint/end of next different control; original fit <=32 frames, future tracking <=64; no guarantee of physical diversity, uniqueness, pose replacement or navigation",
    }


def replay(route, experiment, policy, *, results=None):
    if policy.search.geometry != PoseCompetitionConfig().geometry:
        raise ValueError(
            "switched-action validation cannot change original geometry gates"
        )
    baseline = original.replay(route, experiment, PoseCompetitionConfig())
    config = json.loads((experiment / "config.json").read_bytes())
    predicted = json.loads((experiment / "report.json").read_bytes())
    acquisition = json.loads((route / "config.json").read_bytes())
    raw = json.loads((route / "report.json").read_bytes())
    timeline = [
        {
            "start_frame": item["start_frame"],
            "end_frame": item["end_frame"],
            "kind": item["scheduled"]["kind"],
            "motion_handle": item["scheduled"]["motion_handle_index"],
        }
        for item in compound_timeline(
            CompoundProbeConfig.model_validate(acquisition["config"])
        )
    ]
    results = [] if results is None else results
    for index, (prior, record) in enumerate(
        zip(baseline, predicted["results"], strict=True)
    ):
        selection = KeyframeSelection.model_validate(record["selection"])
        if (
            prior["pair"] != record["pair"]
            or prior["selected_offset"] != selection.selected_offset
        ):
            raise ValueError("replayed original interval differs")
        selected, camera, seed = (
            selection.selected(),
            np.asarray(config["camera_intrinsics"]),
            config["seed"] + index,
        )
        plan = plan_action_switch(
            selected,
            timeline,
            anchor_frame=record["pair"]["first"],
            frame_count=raw["frame_count"],
            config=policy,
        )
        # Freeze each hypothesis using its prefix-only data before extending its
        # tracks. Other original diagnostic prefixes are replayed but never used
        # to change the chronological first-supported latch or this fit.
        prefit = inspect_direction_search(
            selected, camera, seed=seed, config=policy.search
        )
        frozen = prefit.model_dump_json()
        indices, extended = [], None
        if plan.available and prefit.reference_matches_original_checks:
            indices = list(range(plan.anchor_frame, plan.target_frames[-1] + 1))
            frames = [
                rgb(route / "frames" / f"frame-{slot:05d}.png") for slot in indices
            ]
            extended = track_frame_chain(frames, config=policy.temporal)
        if prefit.model_dump_json() != frozen:
            raise ValueError("future tracking mutated frozen map evidence")
        evidence = inspect_action_switch(
            selected,
            extended,
            plan,
            camera,
            seed=seed,
            config=policy,
            frozen_search=prefit,
        )
        results.append(
            {
                "pair": record["pair"],
                "selected_offset": selection.selected_offset,
                "original_competition": prior["competition"],
                "future_indices": indices,
                "future_tracks": None
                if extended is None
                else extended.model_dump(mode="json"),
                "action_switch": evidence.model_dump(mode="json"),
            }
        )
    return results
