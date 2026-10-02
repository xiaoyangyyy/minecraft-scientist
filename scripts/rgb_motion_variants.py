"""Registered public-only ablations over the same controlled RGB acquisition."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from controlled_motion_common import inference_config as baseline_config
from controlled_motion_common import rgb
from controlled_motion_common import runtime_sources as baseline_sources

from minecraft_scientist.exploration.consensus_motion import estimate_consensus_endpoint
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    estimate_multiview_motion,
    estimate_temporal_endpoint,
)
from minecraft_scientist.exploration.stationary_context import (
    StationaryContextConfig,
    stationary_background_seeds,
)
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)

VARIANTS = ("static-context", "consensus", "static-consensus")


def runtime_sources():
    return baseline_sources() | {
        "stationary_context.py": Path(stationary_background_seeds.__code__.co_filename),
        "consensus_motion.py": Path(estimate_consensus_endpoint.__code__.co_filename),
        Path(__file__).name: Path(__file__),
    }


def configuration(route, acquisition, report, seed, *, variant):
    if variant not in VARIANTS:
        raise ValueError("unregistered RGB motion variant")
    config = baseline_config(route, acquisition, report, seed)
    config.update(
        variant=variant,
        static_context=StationaryContextConfig().model_dump(mode="json"),
        endpoint_solver="conditioned_consensus_before_pose"
        if variant in ("consensus", "static-consensus")
        else "legacy_refit_after_pose",
    )
    config["scope"] += f"; preregistered {variant} development ablation"
    return config


def context_indices(pair, timeline, count):
    first = pair["first"]
    indices = tuple(range(first - count + 1, first + 1))
    if indices[0] < 0 or any(
        item["scheduled"]["kind"] in ("move", "look")
        and item["start_frame"] < first
        and item["end_frame"] > indices[0]
        for item in timeline
    ):
        raise ValueError(
            "stationary context requires bounded past images without planned camera/body motion"
        )
    return indices


def infer_pair(route, pair, config, index):
    # Pair endpoints/schedule are independently checked before this function runs.
    import json

    first, second = pair["first"], pair["second"]
    indices = (first, first) if first == second else tuple(range(first, second + 1))
    frames = [rgb(route / "frames" / f"frame-{slot:05d}.png") for slot in indices]
    static = config["variant"] in ("static-context", "static-consensus")
    consensus = config["variant"] in ("consensus", "static-consensus")
    selected, context, seeds = None, (), None
    if static:
        stationary = StationaryContextConfig.model_validate(config["static_context"])
        timeline = json.loads((route / "report.json").read_bytes())["timeline"]
        context = context_indices(pair, timeline, stationary.context_frames)
        selected = stationary_background_seeds(
            [rgb(route / "frames" / f"frame-{slot:05d}.png") for slot in context],
            config=stationary,
        )
        seeds = selected.pixels
    temporal, multiview = (
        TemporalTrackConfig.model_validate(config["temporal"]),
        MultiViewMotionConfig.model_validate(config["multiview"]),
    )
    camera = np.asarray(config["camera_intrinsics"])
    tracks = track_frame_chain(frames, config=temporal, seed_pixels=seeds)
    estimator = estimate_consensus_endpoint if consensus else estimate_temporal_endpoint
    motion = estimate_multiview_motion(
        tracks,
        camera,
        config=multiview,
        seed=config["seed"] + index,
        endpoint_estimator=estimator,
    )
    endpoints = track_frame_chain(
        [frames[0], frames[-1]], config=temporal, seed_pixels=seeds
    )
    endpoint = estimator(
        endpoints.endpoint_correspondences(),
        camera,
        config=multiview.two_view,
        seed=config["seed"] + index,
    )
    return {
        "pair": pair,
        "chain_indices": list(indices),
        "context_indices": list(context),
        "stationary_seeds": None
        if selected is None
        else selected.model_dump(mode="json"),
        "tracks": tracks.model_dump(mode="json"),
        "motion": motion.model_dump(mode="json"),
        "endpoint_tracks": endpoints.model_dump(mode="json"),
        "endpoint_motion": endpoint.model_dump(mode="json"),
    }


def backend(variant):
    if variant not in VARIANTS:
        raise ValueError("unregistered RGB motion variant")
    return SimpleNamespace(
        runtime_sources=runtime_sources,
        inference_config=partial(configuration, variant=variant),
        infer_pair=infer_pair,
    )
