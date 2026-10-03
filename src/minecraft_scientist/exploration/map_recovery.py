"""Public-image stability recovery with explicit, unstitched map epochs.

Stable optical flow does not certify player rest or arrival. Recovery creates a
new past-only map; it never replenishes or retrospectively accepts the old map.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .active_causal_map import ActiveMapPolicy, acquire_causal_map
from .joint_multiview_map import _camera
from .settled_map_control import (
    PublicControlStep,
    SettledControlPolicy,
    run_settled_controls,
)
from .temporal_tracks import TemporalTrackConfig, track_frame_chain
from .visibility_acquisition import PublicProbeStep
from .visual_geometry import _cells, _frame


class MapRecoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_wait_ticks: int = Field(default=8, ge=8, le=32, strict=True)
    maximum_wait_ticks: int = Field(default=32, ge=8, le=32, strict=True)
    stable_prefixes: int = Field(default=3, ge=3, le=8, strict=True)
    minimum_tracks: int = Field(default=32, ge=32, le=800, strict=True)
    minimum_cells: int = Field(default=4, ge=4, le=16, strict=True)
    minimum_survival: FiniteFloat = Field(default=0.5, ge=0.5, le=1)
    median_flow_pixels: FiniteFloat = Field(default=0.2, gt=0, le=0.2)
    percentile95_flow_pixels: FiniteFloat = Field(default=0.5, gt=0, le=0.5)
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=3)
    acquisition: ActiveMapPolicy = ActiveMapPolicy(minimum_ticks=8)
    continuation: SettledControlPolicy = SettledControlPolicy()

    @model_validator(mode="after")
    def _budgets(self):
        if (
            self.minimum_wait_ticks > self.maximum_wait_ticks
            or self.temporal.maximum_frames != 3
        ):
            raise ValueError("recovery requires ordered waits and pair-local tracking")
        return self


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def recover_public_map(parent, last_frame, execute, *, policy=None):
    """One bounded recovery epoch after failed acquisition/settled continuation.

    The parent contains only the public acquisition/settled results. All WAIT
    comparisons are new pair-local track namespaces. A subsequent acquisition
    starts at the actually consumed stable image, with new seed and lineage.
    Local gauges are explicitly not aligned, so no global route is certified.
    """
    policy = MapRecoveryPolicy.model_validate(
        policy.model_dump() if isinstance(policy, MapRecoveryPolicy) else policy or {}
    )
    old = parent["acquisition"]
    stopped = parent["settled"]
    frozen_parent = _digest(parent)
    base = {
        "policy": policy.model_dump(mode="json"),
        "parent_sha256": frozen_parent,
        "wait_steps": [],
        "flow_prefixes": [],
        "acquisition": None,
        "settled": None,
        "failure": None,
        "terminal": "not_started",
        "visual_stability_only": True,
        "global_gauge_aligned": False,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }

    def finish():
        if _digest(parent) != frozen_parent:
            raise ValueError("recovery changed its immutable parent")
        return base

    if old["failure"] is not None or stopped["failure"] is not None:
        raise ValueError("failed control/tracking receipts cannot initiate recovery")
    if stopped["terminal"] == "completed":
        return finish()
    if stopped["terminal"] not in (
        "not_started",
        "support_aborted",
        "settle_exhausted",
    ):
        raise ValueError("recovery requires an explicit supported abort category")
    if stopped["terminal"] == "not_started" and old["terminal"] not in (
        "exhausted",
        "rejected",
        "support_aborted",
    ):
        raise ValueError("unstarted continuation requires an unavailable initial map")
    camera = _camera(parent["calibration"]["camera_intrinsics"])
    if type(old["seed"]) is not int or not 0 <= old["seed"] < 2**31 - 517:
        raise ValueError("recovery requires a bounded parent seed")
    count = stopped.get("evidence_frames", old["evidence_frames"])
    if type(count) is not int or count < 1 or type(old["anchor_frame"]) is not int:
        raise ValueError("recovery needs a public image endpoint")
    anchor = old["anchor_frame"] + count - 1
    if stopped["terminal"] != "not_started" and (
        not stopped["prefixes"] or stopped["prefixes"][-1]["absolute_frame"] != anchor
    ):
        raise ValueError(
            "recovery endpoint differs from the last executed public prefix"
        )
    image = _frame(last_frame).copy()
    base.update(anchor_frame=anchor, track_namespace=f"recovery-{anchor}")
    stable_count = 0
    for tick in range(1, policy.maximum_wait_ticks + 1):
        step = {
            "absolute_frame": anchor + tick,
            "compliant": False,
            "executed_kind": None,
            "executed_handle": None,
            "ticks": None,
        }
        base["wait_steps"].append(step)
        try:
            receipt = execute("wait", None)
            if not isinstance(receipt, PublicControlStep):
                raise TypeError("recovery requires a public RGB/action receipt")
            step.update(
                executed_kind=receipt.executed_kind
                if type(receipt.executed_kind) is str
                else None,
                executed_handle=receipt.executed_handle
                if type(receipt.executed_handle) is int
                else None,
                ticks=receipt.ticks if type(receipt.ticks) is int else None,
                compliant=receipt.compliant
                if type(receipt.compliant) is bool
                else False,
            )
            if (
                receipt.executed_kind != "wait"
                or type(receipt.executed_kind) is not str
                or receipt.executed_handle is not None
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or receipt.compliant is not True
            ):
                raise ValueError("failed recovery WAIT cannot enter image evidence")
            current = _frame(receipt.frame).copy()
            if current.shape != image.shape:
                raise ValueError("recovery WAIT changed image shape")
        except Exception as error:  # noqa: BLE001 -- preserve failed public receipt
            base.update(
                terminal="control_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return finish()
        try:
            tracks = track_frame_chain([image, current], config=policy.temporal)
            if tracks.frame_count != 2 or tracks.image_shape != image.shape[:2]:
                raise ValueError(
                    "recovery tracker changed the pair-local image contract"
                )
            pixels = np.asarray(tracks.pixels, dtype=float)
            flow = (
                np.linalg.norm(pixels[1] - pixels[0], axis=1)
                if tracks.track_ids
                else np.empty(0)
            )
            cells = _cells(pixels[-1], *tracks.image_shape) if len(flow) else 0
            median = float(np.median(flow)) if len(flow) else None
            p95 = float(np.percentile(flow, 95)) if len(flow) else None
            stable = (
                len(flow) >= policy.minimum_tracks
                and cells >= policy.minimum_cells
                and len(flow) / max(tracks.seed_count, 1) >= policy.minimum_survival
                and median <= policy.median_flow_pixels
                and p95 <= policy.percentile95_flow_pixels
            )
            base["flow_prefixes"].append(
                {
                    "interval": [anchor + tick - 1, anchor + tick],
                    "track_namespace": f"recovery-{anchor}/pair-{tick}",
                    "first_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "last_rgb_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                    "tracks": tracks.model_dump(mode="json"),
                    "cells": cells,
                    "median_flow_pixels": median,
                    "percentile95_flow_pixels": p95,
                    "stable": bool(stable),
                }
            )
            image = current
            stable_count = stable_count + 1 if stable else 0
        except Exception as error:  # noqa: BLE001 -- retain executed evidence, fail closed
            base.update(
                terminal="tracking_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return finish()
        if tick >= policy.minimum_wait_ticks and stable_count >= policy.stable_prefixes:
            break
    else:
        base["terminal"] = "wait_exhausted"
        return finish()
    next_anchor = anchor + len(base["wait_steps"])
    new_frames = [image.copy()]

    def acquire(handle):
        receipt = execute("move", handle)
        if not isinstance(receipt, PublicControlStep):
            raise TypeError("reacquisition requires a public control receipt")
        compliant = (
            receipt.executed_kind == "move"
            and type(receipt.executed_kind) is str
            and type(receipt.executed_handle) is int
            and receipt.executed_handle == handle
            and receipt.compliant is True
        )
        if compliant and type(receipt.ticks) is int and receipt.ticks == 1:
            new_frames.append(_frame(receipt.frame).copy())
        return PublicProbeStep(
            receipt.frame, receipt.executed_handle, compliant, receipt.ticks
        )

    base.update(
        new_anchor_frame=next_anchor,
        seed=old["seed"] + 5,
        new_track_namespace=f"map-{next_anchor}",
    )
    base["acquisition"] = acquire_causal_map(
        image,
        acquire,
        camera,
        anchor_frame=next_anchor,
        seed=base["seed"],
        policy=policy.acquisition,
    )
    child = base["acquisition"]
    if child["failure"] is not None:
        base.update(terminal="reacquisition_failed", failure=child["failure"])
    elif child["terminal"] != "validated":
        base["terminal"] = "map_unavailable"
    else:
        base["settled"] = run_settled_controls(
            child, new_frames, execute, policy=policy.continuation
        )
        following = base["settled"]
        base.update(
            terminal="completed"
            if following["terminal"] == "completed"
            else "continuation_aborted",
            failure=following["failure"],
        )
    return finish()
