"""Counterexample-trained shared map and publicly settled endpoint experiment.

Renderer/player telemetry is not an input. Optical stability is not player rest;
only an independent evaluator may check that distinction after complete replay.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .causal_map_window import CausalWindowPolicy, PrefixFuture, _consensus, _relative
from .joint_multiview_map import validate_joint_future
from .map_recovery import _digest
from .second_leg_shadow import _append_tracks, retain_tracks
from .settled_map_control import PublicControlStep, _center
from .shared_anchor_map import (
    SharedCrisisFitPolicy,
    original_shared_root,
    refine_shared_crisis_map,
)
from .temporal_tracks import TemporalTrackConfig, TemporalTracks, track_frame_chain
from .visual_geometry import _cells, _frame


class SharedEndpointPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fitting: SharedCrisisFitPolicy = SharedCrisisFitPolicy()
    move_ticks: Literal[2] = 2
    minimum_wait_ticks: int = Field(default=8, ge=8, le=16, strict=True)
    maximum_wait_ticks: int = Field(default=16, ge=8, le=16, strict=True)
    stable_prefixes: Literal[3] = 3
    minimum_tracks: int = Field(default=32, ge=32, le=800, strict=True)
    minimum_cells: int = Field(default=4, ge=4, le=16, strict=True)
    minimum_survival: FiniteFloat = Field(default=0.5, ge=0.5, le=1)
    median_flow_pixels: FiniteFloat = Field(default=0.2, gt=0, le=0.2)
    percentile95_flow_pixels: FiniteFloat = Field(default=0.5, gt=0, le=0.5)
    maximum_total_actions: Literal[200] = 200
    minimum_translation_ratio: Literal[0.01] = 0.01
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=64)

    @model_validator(mode="after")
    def _bounded(self):
        if (
            self.minimum_wait_ticks > self.maximum_wait_ticks
            or self.temporal != TemporalTrackConfig(maximum_frames=64)
        ):
            raise ValueError(
                "endpoint trials require ordered waits and unchanged bounded tracking"
            )
        return self


def _visible_flow(first, second, settings, *, seed_pixels=None):
    seed_arguments = {} if seed_pixels is None else {"seed_pixels": seed_pixels}
    tracks = track_frame_chain(
        [first, second], config=TemporalTrackConfig(maximum_frames=3), **seed_arguments
    )
    if tracks.frame_count != 2 or tracks.image_shape != first.shape[:2]:
        raise ValueError(
            "endpoint flow must refer to exactly the new public image pair"
        )
    pixels = np.asarray(tracks.pixels)
    flow = (
        np.linalg.norm(pixels[1] - pixels[0], axis=1)
        if tracks.track_ids
        else np.empty(0)
    )
    cells = _cells(pixels[-1], *tracks.image_shape) if len(flow) else 0
    median = float(np.median(flow)) if len(flow) else None
    p95 = float(np.percentile(flow, 95)) if len(flow) else None
    stable = bool(
        len(flow) >= settings.minimum_tracks
        and cells >= settings.minimum_cells
        and len(flow) / max(tracks.seed_count, 1) >= settings.minimum_survival
        and median <= settings.median_flow_pixels
        and p95 <= settings.percentile95_flow_pixels
    )
    return {
        "tracks": tracks.model_dump(mode="json"),
        "stable": stable,
        "median_flow_pixels": median,
        "percentile95_flow_pixels": p95,
        "cells": cells,
    }


def run_shared_endpoint_trial(parent, frames, execute, *, policy=None):
    """Freeze a sixteen-view revision; WAIT, distinct MOVE, WAIT, then shadow test.

    Old failure/acceptance records are immutable. Initial endpoint validation freezes
    at its own prefix, before movement; later surviving identities cannot restamp it.
    The trial does not stop waiting merely because its frozen map loses support.
    """
    settings = SharedEndpointPolicy.model_validate(
        policy.model_dump()
        if isinstance(policy, SharedEndpointPolicy)
        else policy or {}
    )
    parent_sha = _digest(parent)
    result = {
        "policy": settings.model_dump(mode="json"),
        "parent_sha256": parent_sha,
        "frozen": None,
        "frozen_sha256": None,
        "steps": [],
        "prefixes": [],
        "flow_prefixes": [],
        "initial_settle": None,
        "final_settle": None,
        "initial_endpoint": None,
        "final_endpoint": None,
        "motion": None,
        "status": "not_started",
        "terminal": "not_started",
        "failure": None,
        "visual_stability_only": True,
        "metric_scale_known": False,
        "global_gauge_aligned": False,
        "navigation_accepted": False,
    }

    def finish():
        if _digest(parent) != parent_sha:
            raise ValueError("endpoint trial changed its immutable parent")
        if (
            result["frozen"] is not None
            and _digest(result["frozen"]) != result["frozen_sha256"]
        ):
            raise ValueError("future controls changed a frozen endpoint map")
        if (
            result["initial_endpoint"] is not None
            and _digest(result["initial_endpoint"]) != result["initial_endpoint_sha256"]
        ):
            raise ValueError("later controls changed the frozen initial endpoint")
        return result

    if parent["failure"] is not None or any(
        parent[key] is not False
        for key in ("navigation_accepted", "metric_scale_known", "global_gauge_aligned")
    ):
        raise ValueError(
            "endpoint trials require public, failure-free local-map evidence"
        )
    versions = parent["shared_anchor_versions"]
    if not versions:
        return finish()
    if len(versions) != 1 or parent["terminal"] != "completed":
        raise ValueError(
            "endpoint trial requires the completed registered two-epoch parent"
        )
    version = versions[0]
    if version["frozen"] is None:
        result["terminal"] = "history_budget_exhausted"
        return finish()
    if version["status"] not in ("rejected", "shadow_validated"):
        raise ValueError(
            "endpoint trial needs a terminal shared counterexample or validated version"
        )
    root = parent["epochs"][version["parent_epoch"]]["frozen"]
    frozen_root, original, originals = original_shared_root(root["snapshot"])
    if (
        _digest(frozen_root) != root["snapshot_sha256"]
        or root["snapshot_sha256"] != version["original_snapshot_sha256"]
    ):
        raise ValueError("endpoint reconstruction cannot change original lineage")
    anchor = frozen_root["anchor_frame"]
    last = version["decision"]["absolute_frame"]
    if type(last) is not int or not anchor < last <= settings.maximum_total_actions:
        raise ValueError("endpoint trial needs bounded actual parent image endpoints")
    images = [_frame(image).copy() for image in frames]
    if len(images) != last + 1 or any(
        image.shape != images[0].shape for image in images
    ):
        raise ValueError(
            "endpoint trial requires exactly the consumed complete public episode"
        )
    images = images[anchor:]
    if [
        hashlib.sha256(image.tobytes()).hexdigest()
        for image in images[: original.frame_count]
    ] != root["rgb_sha256"]:
        raise ValueError("endpoint trial RGB differs from its immutable original root")
    previous = TemporalTracks.model_validate(
        version["terminal_snapshot"]["prefixes"][-1]
    )
    if (
        previous.frame_count != len(images)
        or len(images) > settings.fitting.maximum_consumed_frames
    ):
        raise ValueError("endpoint revision exceeds its separate consumed-frame budget")
    retained = set(previous.track_ids)
    expected = retain_tracks(
        track_frame_chain(images, config=settings.temporal), retained
    )
    if expected.model_dump() != previous.model_dump():
        raise ValueError(
            "endpoint initial history does not reproduce its protected track prefix"
        )
    models = tuple(
        refine_shared_crisis_map(item, original, previous, policy=settings.fitting)
        for item in originals
    )
    required = tuple(item.bootstrap_slot for item in originals if item.supported)
    result.update(
        status="proposed",
        update_kind="crisis_refit"
        if version["status"] == "rejected"
        else "maintenance_refit",
        frozen={
            "anchor_frame": anchor,
            "original_snapshot_sha256": root["snapshot_sha256"],
            "previous_version_sha256": _digest(version),
            "past": previous.model_dump(mode="json"),
            "maps": [item.model_dump(mode="json") for item in models],
            "required_bootstraps": list(required),
            "rgb_sha256": [
                hashlib.sha256(image.tobytes()).hexdigest() for image in images
            ],
            "metric_scale_known": False,
            "global_gauge_aligned": False,
            "navigation_accepted": False,
        },
    )
    result["frozen_sha256"] = _digest(result["frozen"])
    past = previous
    counts = {
        int(handle): count for handle, count in parent["final_handle_counts"].items()
    }
    if set(counts) != set(range(4)) or any(
        type(count) is not int or count < 0 for count in counts.values()
    ):
        raise ValueError("endpoint trial requires four public opaque handle counts")
    old_handle = parent["steps"][-1]["requested_handle"]
    handle = min(
        (item for item in counts if item != old_handle),
        key=lambda item: (counts[item], item),
    )
    result["validation_handle"] = handle

    def advance(kind, requested_handle, phase):
        nonlocal previous, retained, last
        if last >= settings.maximum_total_actions:
            result.update(terminal="budget_exhausted", status="rejected")
            return False
        step = {
            "phase": phase,
            "requested_kind": kind,
            "requested_handle": requested_handle,
            "executed_kind": None,
            "executed_handle": None,
            "compliant": False,
            "ticks": None,
            "action_response_frame": last + 1,
        }
        result["steps"].append(step)
        try:
            receipt = execute(kind, requested_handle)
            if not isinstance(receipt, PublicControlStep):
                raise TypeError("endpoint trial requires a public RGB/action receipt")
            step.update(
                executed_kind=receipt.executed_kind
                if type(receipt.executed_kind) is str
                else None,
                executed_handle=receipt.executed_handle
                if type(receipt.executed_handle) is int
                else None,
                compliant=receipt.compliant
                if type(receipt.compliant) is bool
                else False,
                ticks=receipt.ticks if type(receipt.ticks) is int else None,
            )
            if (
                type(receipt.executed_kind) is not str
                or receipt.executed_kind != kind
                or receipt.executed_handle != requested_handle
                or (kind == "move" and type(receipt.executed_handle) is not int)
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or receipt.compliant is not True
            ):
                raise ValueError("failed endpoint controls cannot enter image evidence")
            image = _frame(receipt.frame).copy()
            if image.shape != images[0].shape:
                raise ValueError("endpoint control changed RGB shape")
        except Exception as error:  # noqa: BLE001 -- preserve partial actual controls
            result.update(
                terminal="control_failed",
                status="rejected",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False
        last += 1
        images.append(image)
        try:
            tracked = track_frame_chain(images, config=settings.temporal)
            retained = _append_tracks(previous, tracked, retained)
            previous = retain_tracks(tracked, retained)
            result["prefixes"].append(
                {
                    "absolute_frame": last,
                    "input_sha256": hashlib.sha256(
                        previous.model_dump_json().encode()
                    ).hexdigest(),
                    "retained_ids": list(previous.track_ids),
                    "surviving_counts": list(previous.surviving_counts),
                }
            )
        except Exception as error:  # noqa: BLE001 -- retain executed frame/failure
            result.update(
                terminal="tracking_failed",
                status="rejected",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False
        return True

    def settle(phase):
        record = {
            "start_frame": last,
            "end_frame": last,
            "wait_ticks": 0,
            "settled": False,
        }
        result["initial_settle" if phase == "initial_wait" else "final_settle"] = record
        consecutive = 0
        for tick in range(1, settings.maximum_wait_ticks + 1):
            if not advance("wait", None, phase):
                return False
            record.update(end_frame=last, wait_ticks=tick)
            try:
                flow = _visible_flow(images[-2], images[-1], settings)
            except Exception as error:  # noqa: BLE001 -- retain failed public flow
                result.update(
                    terminal="tracking_failed",
                    status="rejected",
                    failure={"type": type(error).__name__, "message": str(error)},
                )
                return False
            result["flow_prefixes"].append(
                flow
                | {
                    "phase": phase,
                    "interval": [last - 1, last],
                    "track_namespace": f"{phase}/pair-{last}",
                }
            )
            consecutive = consecutive + 1 if flow["stable"] else 0
            if (
                tick >= settings.minimum_wait_ticks
                and consecutive >= settings.stable_prefixes
            ):
                record["settled"] = True
                return True
        result.update(terminal=f"{phase}_exhausted", status="rejected")
        return False

    def endpoint():
        return {
            "absolute_frame": last,
            "input_sha256": hashlib.sha256(
                previous.model_dump_json().encode()
            ).hexdigest(),
            "candidates": [
                {
                    "bootstrap_slot": model.bootstrap_slot,
                    "future": validate_joint_future(
                        model, past, previous, target_slots=(previous.frame_count - 1,)
                    ),
                }
                for model in models
            ],
        }

    if not settle("initial_wait"):
        return finish()
    try:
        result["initial_endpoint"] = endpoint()
    except Exception as error:  # noqa: BLE001 -- retain controls before failed endpoint fit
        result.update(
            terminal="tracking_failed",
            status="rejected",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        return finish()
    result["initial_endpoint_sha256"] = _digest(result["initial_endpoint"])
    for _ in range(settings.move_ticks):
        if not advance("move", handle, "new_control"):
            return finish()
    if not settle("final_wait"):
        return finish()
    try:
        result["final_endpoint"] = endpoint()
    except Exception as error:  # noqa: BLE001 -- retain complete failed-control experiment
        result.update(
            terminal="tracking_failed",
            status="rejected",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        return finish()
    relatives, motion_ratios = [], []
    for model, first, second in zip(
        models,
        result["initial_endpoint"]["candidates"],
        result["final_endpoint"]["candidates"],
        strict=True,
    ):
        before, after = (
            PrefixFuture.model_validate(first["future"]),
            PrefixFuture.model_validate(second["future"]),
        )
        if (
            model.bootstrap_slot in required
            and model.supported
            and before.status == after.status == "consistent"
        ):
            relatives.append(_relative(before.poses[0], after.poses[0]))
            motion_ratios.append(
                {
                    "bootstrap_slot": model.bootstrap_slot,
                    "translation_ratio": float(
                        np.linalg.norm(
                            _center(after.poses[0]) - _center(before.poses[0])
                        )
                        / max(
                            float(np.linalg.norm(_center(pose))) for pose in model.poses
                        )
                    ),
                }
            )
    direction, rotation_error, rotation, translation = _consensus(
        relatives, CausalWindowPolicy()
    )
    supported = (
        len(required) >= 2
        and len(relatives) == len(required)
        and translation is not None
        and all(
            item["translation_ratio"] > settings.minimum_translation_ratio
            for item in motion_ratios
        )
    )
    result.update(
        terminal="validated" if supported else "rejected",
        status="shadow_validated" if supported else "rejected",
        motion={
            "supported": supported,
            "reason": "all original models and publicly settled endpoint rays agree"
            if supported
            else "original models lack endpoint support, disagree, or predict insufficient motion; no subset rescue",
            "relative_rotation": rotation if supported else None,
            "unit_translation": translation if supported else None,
            "maximum_direction_disagreement": direction,
            "maximum_rotation_disagreement": rotation_error,
            "endpoint_motion_ratios": motion_ratios,
            "metric_scale_known": False,
            "navigation_accepted": False,
        },
    )
    return finish()
