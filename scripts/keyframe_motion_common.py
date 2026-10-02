"""Public frozen-input pipeline for chronological local-keyframe experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from controlled_motion_common import inference_config as baseline_config
from controlled_motion_common import rgb
from controlled_motion_common import runtime_sources as baseline_sources

from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeMotionConfig,
    infer_keyframe_prefixes,
)


def runtime_sources():
    return baseline_sources() | {
        "keyframe_motion.py": Path(infer_keyframe_prefixes.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "probe_keyframe_rgb_motion.py": Path(__file__).with_name(
            "probe_keyframe_rgb_motion.py"
        ),
    }


def configuration(route, acquisition, report, seed):
    policy = KeyframeMotionConfig()
    if (
        type(seed) is not int
        or seed < 0
        or seed + len(report["pairs"]) - 1 >= 2**31 - policy.multiview.stability.trials
    ):
        raise ValueError(
            "keyframe seed must leave room for all pair and sensitivity seeds"
        )
    config = baseline_config(route, acquisition, report, seed)
    # The existing camera, tracking, PnP, stability and evaluator thresholds are
    # not altered by this temporal-window ablation.
    if config["temporal"] != policy.temporal.model_dump(mode="json") or config[
        "multiview"
    ] != policy.multiview.model_dump(mode="json"):
        raise ValueError("keyframe ablation cannot silently change baseline thresholds")
    config.update(
        selector="first_supported_chronological_prefix",
        keyframe=policy.model_dump(mode="json"),
        scope=config["scope"]
        + "; selected local pose is not terminal displacement or navigation acceptance",
    )
    return config


def infer_pair(route, pair, config, index):
    first, second = pair["first"], pair["second"]
    frames = [
        rgb(route / "frames" / f"frame-{slot:05d}.png")
        for slot in range(first, second + 1)
    ]
    selection = infer_keyframe_prefixes(
        frames,
        np.asarray(config["camera_intrinsics"]),
        config=KeyframeMotionConfig.model_validate(config["keyframe"]),
        seed=config["seed"] + index,
    )
    return {"pair": pair, "selection": selection.model_dump(mode="json")}
