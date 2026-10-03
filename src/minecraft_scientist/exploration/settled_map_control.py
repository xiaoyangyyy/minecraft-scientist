"""Frozen public-map controls with visible settling, never current-player poses.

Frame ordinals identify returned images, not player-position timestamps. Settling
is a public geometric test; evaluator coordinates cannot choose its endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    model_validator,
)

from .causal_map_window import CausalWindowPolicy, PrefixFuture, _consensus, _relative
from .joint_multiview_map import JointMapEvidence, validate_joint_future
from .second_leg_shadow import _append_tracks, retain_tracks
from .temporal_tracks import TemporalTrackConfig, TemporalTracks, track_frame_chain
from .visual_geometry import _frame


class SettledControlPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    handles: tuple[StrictInt, StrictInt, StrictInt] = (0, 1, 2)
    move_ticks: int = Field(default=4, ge=4, le=4, strict=True)
    minimum_wait_ticks: int = Field(default=3, ge=3, le=8, strict=True)
    maximum_wait_ticks: int = Field(default=8, ge=3, le=8, strict=True)
    stable_prefixes: int = Field(default=2, ge=2, le=2, strict=True)
    maximum_translation_ratio: FiniteFloat = Field(default=0.01, gt=0, le=0.01)
    maximum_rotation_degrees: FiniteFloat = Field(default=0.2, gt=0, le=0.2)
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=64)

    @model_validator(mode="after")
    def _bounded(self):
        if (
            self.minimum_wait_ticks > self.maximum_wait_ticks
            or self.temporal.maximum_frames != 64
            or self.handles != (0, 1, 2)
        ):
            raise ValueError(
                "settled controls require ordered waits and bounded original tracks"
            )
        return self


@dataclass(frozen=True)
class PublicControlStep:
    frame: np.ndarray
    executed_kind: Literal["move", "wait"]
    executed_handle: int | None
    compliant: bool
    ticks: int = 1


def _center(pose):
    return -np.array(pose.rotation).T @ np.array(pose.translation)


def _rotation_angle(first, second):
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    (
                        np.trace(np.array(first.rotation) @ np.array(second.rotation).T)
                        - 1
                    )
                    / 2,
                    -1,
                    1,
                )
            )
        )
    )


def run_settled_controls(acquisition, frames, execute, *, policy=None):
    """Continue a validated image map: WAIT, then three MOVE/WAIT experiments.

    Maps and original partitions freeze once; no fitting, reseeding, replenishment
    or private endpoint selection. Failed controls cannot enter evidential frames.
    This controls acquisition, not goal navigation or metric localization.
    """
    policy = SettledControlPolicy.model_validate(
        policy.model_dump()
        if isinstance(policy, SettledControlPolicy)
        else policy or {}
    )
    base = {
        "policy": policy.model_dump(mode="json"),
        "steps": [],
        "prefixes": [],
        "trials": [],
        "initial_settle": None,
        "failure": None,
        "terminal": "not_started",
        "navigation_accepted": False,
        "metric_scale_known": False,
        "pose_domain": "returned_image_camera; not current-player pose",
    }
    if acquisition["terminal"] != "validated":
        return base
    snapshot = acquisition["snapshot"]
    window = CausalWindowPolicy.model_validate(snapshot["policy"])
    past = TemporalTracks.model_validate(snapshot["past"])
    maps = tuple(JointMapEvidence.model_validate(item) for item in snapshot["maps"])
    images = [_frame(image).copy() for image in frames]
    if (
        len(images) != acquisition["evidence_frames"]
        or len(images) != snapshot["prefixes"][-1]["frame_count"]
        or any(image.shape != images[0].shape for image in images)
        or len(images) + 8 + 3 * (4 + 8) > policy.temporal.maximum_frames
    ):
        raise ValueError(
            "settled session needs the complete bounded public acquisition prefix"
        )
    previous = TemporalTracks.model_validate(snapshot["prefixes"][-1])
    retained = set(previous.track_ids)
    # Public recomputation verifies the supplied initial checkpoint, not labels.
    expected = track_frame_chain(images, config=policy.temporal)
    if retain_tracks(expected, retained).model_dump() != previous.model_dump():
        raise ValueError(
            "settled initial image prefix does not reproduce frozen tracks"
        )
    initial = [
        PrefixFuture.model_validate(
            validate_joint_future(
                fitted, past, previous, target_slots=(previous.frame_count - 1,)
            )
        )
        for fitted in maps
    ]
    if [item.model_dump(mode="json") for item in initial] != [
        item["future"] for item in snapshot["decisions"][-1]["candidates"]
    ]:
        raise ValueError("settled initial future checkpoint was forged")
    poses = {
        fitted.bootstrap_slot: checked.poses[0]
        for fitted, checked in zip(maps, initial, strict=True)
        if fitted.supported and checked.status == "consistent"
    }
    if len(poses) < window.minimum_models:
        raise ValueError(
            "settled session requires the validated initial model population"
        )
    scales = {
        fitted.bootstrap_slot: max(
            np.linalg.norm(_center(pose)) for pose in fitted.poses
        )
        for fitted in maps
        if fitted.supported
    }
    if any(not np.isfinite(scale) or scale <= 1e-9 for scale in scales.values()):
        raise ValueError("settled maps need a nondegenerate past-only scale reference")
    frozen_maps = tuple(item.model_dump_json() for item in maps)
    anchor = acquisition["anchor_frame"]
    terminal, failure = "completed", None

    def advance(kind, handle):
        nonlocal previous, retained, poses, terminal, failure
        step = {
            "kind": kind,
            "requested_handle": handle,
            "executed_kind": None,
            "executed_handle": None,
            "ticks": None,
            "compliant": False,
            "action_response_frame": anchor + len(images),
        }
        base["steps"].append(step)
        try:
            receipt = execute(kind, handle)
            if not isinstance(receipt, PublicControlStep):
                raise TypeError("settled controls require a public RGB/action receipt")
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
                receipt.executed_kind != kind
                or type(receipt.executed_kind) is not str
                or receipt.executed_handle != handle
                or (kind == "move" and type(receipt.executed_handle) is not int)
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or type(receipt.compliant) is not bool
                or not receipt.compliant
            ):
                raise ValueError("failed settled control cannot enter pose evidence")
            image = _frame(receipt.frame).copy()
            if image.shape != images[0].shape:
                raise ValueError("settled control changed image shape")
        except Exception as error:  # noqa: BLE001 -- retain failed controls
            terminal, failure = (
                "control_failed",
                {"type": type(error).__name__, "message": str(error)},
            )
            return False
        images.append(image)
        try:
            current = track_frame_chain(images, config=policy.temporal)
            if current.frame_count != len(images):
                raise ValueError("settled tracking changed its executed frame count")
            retained = _append_tracks(previous, current, retained)
            current = retain_tracks(current, retained)
            checks = [
                PrefixFuture.model_validate(
                    validate_joint_future(
                        fitted, past, current, target_slots=(current.frame_count - 1,)
                    )
                )
                for fitted in maps
            ]
            current_poses = {
                fitted.bootstrap_slot: checked.poses[0]
                for fitted, checked in zip(maps, checks, strict=True)
                if fitted.supported and checked.status == "consistent"
            }
            common = set(poses) & set(current_poses)
            ratios = [
                float(
                    np.linalg.norm(_center(current_poses[slot]) - _center(poses[slot]))
                    / scales[slot]
                )
                for slot in sorted(common)
            ]
            angles = [
                _rotation_angle(current_poses[slot], poses[slot])
                for slot in sorted(common)
            ]
            record = {
                "absolute_frame": anchor + current.frame_count - 1,
                "tracks": current.model_dump(mode="json"),
                "candidates": [checked.model_dump(mode="json") for checked in checks],
                "common_pose_slots": sorted(common),
                "translation_ratios": ratios,
                "rotation_degrees": angles,
                "stable": len(common) >= window.minimum_models
                and len(common) == len(current_poses) == len(poses)
                and max(ratios, default=float("inf"))
                <= policy.maximum_translation_ratio
                and max(angles, default=float("inf"))
                <= policy.maximum_rotation_degrees,
            }
            base["prefixes"].append(record)
            previous, poses = current, current_poses
            if tuple(item.model_dump_json() for item in maps) != frozen_maps:
                raise ValueError("settled controls changed frozen maps")
            if len(poses) < window.minimum_models:
                terminal = "support_aborted"
                return False
        except Exception as error:  # noqa: BLE001 -- retain partial tracking evidence
            terminal, failure = (
                "tracking_failed",
                {"type": type(error).__name__, "message": str(error)},
            )
            return False
        return True

    def settle():
        nonlocal terminal
        start, stable = anchor + len(images) - 1, 0
        for tick in range(1, policy.maximum_wait_ticks + 1):
            if not advance("wait", None):
                return {
                    "start_frame": start,
                    "end_frame": anchor + len(images) - 1,
                    "wait_ticks": tick,
                    "settled": False,
                }
            stable = stable + 1 if base["prefixes"][-1]["stable"] else 0
            if tick >= policy.minimum_wait_ticks and stable >= policy.stable_prefixes:
                return {
                    "start_frame": start,
                    "end_frame": anchor + len(images) - 1,
                    "wait_ticks": tick,
                    "settled": True,
                }
        terminal = "settle_exhausted"
        return {
            "start_frame": start,
            "end_frame": anchor + len(images) - 1,
            "wait_ticks": policy.maximum_wait_ticks,
            "settled": False,
        }

    base["initial_settle"] = settle()
    if base["initial_settle"]["settled"]:
        for handle in policy.handles:
            before = dict(poses)
            start = anchor + len(images) - 1
            trial = {
                "handle": handle,
                "start_frame": start,
                "move_end_frame": None,
                "settle": None,
                "end_frame": None,
                "motion": None,
            }
            base["trials"].append(trial)
            for _ in range(policy.move_ticks):
                if not advance("move", handle):
                    break
            trial["end_frame"] = anchor + len(images) - 1
            if terminal != "completed":
                break
            trial["move_end_frame"] = anchor + len(images) - 1
            trial["settle"] = settle()
            trial["end_frame"] = anchor + len(images) - 1
            if not trial["settle"]["settled"]:
                break
            relative = [
                _relative(before[slot], poses[slot])
                for slot in sorted(before.keys() & poses.keys())
            ]
            direction, angle, rotation, shift = _consensus(relative, window)
            trial["motion"] = {
                "supported": shift is not None,
                "relative_rotation": rotation,
                "unit_translation": shift,
                "maximum_direction_disagreement": direction,
                "maximum_rotation_disagreement": angle,
                "metric_scale_known": False,
                "navigation_accepted": False,
            }
    base.update(
        terminal=terminal,
        failure=failure,
        evidence_frames=len(images),
        frozen_maps_unchanged=tuple(item.model_dump_json() for item in maps)
        == frozen_maps,
    )
    return base
