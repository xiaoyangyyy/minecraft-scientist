"""Source-bound compound prefix inference with a fixed subsequent-image veto."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import compound_motion_common as compounds
import controlled_motion_common as canonical
import keyframe_motion_common as keyframes
import numpy as np

from minecraft_scientist.exploration.keyframe_motion import KeyframeSelection
from minecraft_scientist.exploration.shadow_motion import (
    ShadowMotionConfig,
    validate_shadow_motion,
)
from minecraft_scientist.exploration.temporal_tracks import track_frame_chain


def runtime_sources():
    return compounds.runtime_sources() | {
        "shadow_motion.py": Path(validate_shadow_motion.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "probe_shadow_rgb_motion.py": Path(__file__).with_name(
            "probe_shadow_rgb_motion.py"
        ),
    }


def configuration(route, acquisition, report, seed):
    config = compounds.configuration(route, acquisition, report, seed)
    policy = ShadowMotionConfig()
    if config["temporal"] != policy.temporal.model_dump(mode="json") or config[
        "multiview"
    ] != policy.geometry.model_dump(mode="json"):
        raise ValueError("shadow validation cannot change the baseline geometry gates")
    if seed + len(report["pairs"]) - 1 >= 2**31 - policy.temporal.maximum_frames:
        raise ValueError("shadow seed must reserve all future frame seeds")
    config.update(
        shadow=policy.model_dump(mode="json"), frame_count=report["frame_count"]
    )
    config["scope"] += (
        "; fixed two-future-image veto; no reselection or final-pose transport"
    )
    return config


def infer_pair(route, pair, config, index):
    result = keyframes.infer_pair(route, pair, config, index)
    selected = KeyframeSelection.model_validate(result["selection"]).selected()
    policy = ShadowMotionConfig.model_validate(config["shadow"])
    indices, tracks = [], None
    if selected is not None:
        end = pair["first"] + selected.end_offset + policy.future_frames
        if (
            end < config["frame_count"]
            and selected.end_offset + policy.future_frames
            < policy.temporal.maximum_frames
        ):
            indices = list(range(pair["first"], end + 1))
            frames = [
                canonical.rgb(route / "frames" / f"frame-{slot:05d}.png")
                for slot in indices
            ]
            tracks = track_frame_chain(frames, config=policy.temporal)
    shadow = validate_shadow_motion(
        selected,
        tracks,
        np.asarray(config["camera_intrinsics"]),
        config=policy,
        seed=config["seed"] + index,
    )
    result.update(
        shadow_indices=indices,
        shadow_tracks=None if tracks is None else tracks.model_dump(mode="json"),
        shadow=shadow.model_dump(mode="json"),
    )
    return result


@contextmanager
def shadow_pipeline(module):
    originals = {
        name: getattr(module, name)
        for name in (
            "validate_acquisition",
            "configuration",
            "runtime_sources",
            "infer_pair",
        )
    }
    expected = {
        "validate_acquisition": canonical.validate_acquisition,
        "configuration": keyframes.configuration,
        "runtime_sources": keyframes.runtime_sources,
        "infer_pair": keyframes.infer_pair,
    }
    if any(originals[name] is not value for name, value in expected.items()):
        raise ValueError("shadow pipeline requires canonical unmodified functions")
    replacements = {
        "validate_acquisition": compounds.validate_acquisition,
        "configuration": configuration,
        "runtime_sources": runtime_sources,
        "infer_pair": infer_pair,
    }
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
