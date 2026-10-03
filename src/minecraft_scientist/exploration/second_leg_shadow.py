"""Freeze completed second-leg maps before a bounded distinct-control holdout.

Only public RGB and compliant one-tick receipts enter acquisition. The first
supported prefix is latched; terminal fits remain diagnostics. No physical pose
or uniqueness claim follows from these internal held-out pixel checks.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .action_switch_motion import ActionSwitchConfig, ActionSwitchPlan
from .keyframe_motion import KeyframeSelection
from .online_landmark_probe import fit_online_landmarks
from .pose_direction_search import DirectionSearchEvidence, inspect_direction_search
from .temporal_tracks import TemporalTracks, track_frame_chain
from .visibility_acquisition import PublicProbeStep
from .visual_geometry import _cells, _frame


class SecondLegShadowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["second-leg-shadow-v1"] = "second-leg-shadow-v1"
    ticks: Literal[4] = 4
    seed_base: StrictInt = 1000
    seed_stride: Literal[5] = 5
    seed_offset: Literal[2] = 2
    handle_rule: Literal["lowest_distinct_from_both_v1"] = (
        "lowest_distinct_from_both_v1"
    )
    future: ActionSwitchConfig = ActionSwitchConfig()

    @model_validator(mode="after")
    def _fixed(self):
        if self.seed_base != 1000 or self.future != ActionSwitchConfig():
            raise ValueError(
                "shadow policy cannot change registered seeds or geometry gates"
            )
        return self

    def seed(self, number):
        if type(number) is not int or not 0 <= number < 8:
            raise ValueError("shadow seed schedule covers all eight registered trials")
        return self.seed_base + self.seed_stride * number + self.seed_offset


class SecondLegFreeze(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, lt=2**31 - 128, strict=True)
    selection: KeyframeSelection
    prefix_tracks: tuple[TemporalTracks, ...]
    retained_ids: tuple[StrictInt, ...]
    search: DirectionSearchEvidence

    @model_validator(mode="after")
    def _complete(self):
        terminal = self.selection.terminal_offset
        if (
            not 2 <= terminal <= 4
            or tuple(item.end_offset for item in self.selection.attempts)
            != tuple(range(2, terminal + 1))
            or tuple(item.frame_count for item in self.prefix_tracks)
            != tuple(range(2, terminal + 2))
            or tuple(sorted(set(self.retained_ids))) != self.retained_ids
        ):
            raise ValueError(
                "freeze must retain every completed second-leg prefix and chronological fit"
            )
        retained = set(self.prefix_tracks[0].track_ids)
        for tracks in self.prefix_tracks[1:]:
            retained &= set(tracks.track_ids)
        if tuple(sorted(retained)) != self.retained_ids:
            raise ValueError("freeze cannot replenish previously lost identities")
        selected = self.selection.selected()
        if (selected is None) != (self.search.reference is None):
            raise ValueError("frozen search must agree with the first-supported latch")
        return self

    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


def retain_tracks(tracks, identities):
    keep = [
        index
        for index, identity in enumerate(tracks.track_ids)
        if identity in identities
    ]
    return TemporalTracks.model_validate(
        tracks.model_dump()
        | {
            "track_ids": tuple(tracks.track_ids[index] for index in keep),
            "pixels": tuple(
                tuple(frame[index] for index in keep) for frame in tracks.pixels
            ),
            "minimum_patch_correlations": tuple(
                tracks.minimum_patch_correlations[index] for index in keep
            ),
            "chain_cycle_errors": tuple(
                tracks.chain_cycle_errors[index] for index in keep
            ),
        }
    )


def _append_tracks(previous, current, retained):
    if previous is not None:
        if (
            current.frame_count != previous.frame_count + 1
            or current.image_shape != previous.image_shape
            or current.seed_count != previous.seed_count
            or current.surviving_counts[0] != previous.surviving_counts[0]
        ):
            raise ValueError(
                "one-tick tracking must preserve the original second-leg anchor"
            )
        retained = retained & set(current.track_ids)
        before = {identity: index for index, identity in enumerate(previous.track_ids)}
        after = {identity: index for index, identity in enumerate(current.track_ids)}
        for identity in retained:
            if not np.allclose(
                np.asarray(previous.pixels)[:, before[identity]],
                np.asarray(current.pixels)[:-1, after[identity]],
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError("shadow tracking changed frozen prefix pixels")
    else:
        retained = set(current.track_ids)
    return retained


def prepare_second_leg(frames, camera, *, seed, policy=None):
    policy = SecondLegShadowPolicy.model_validate(
        policy.model_dump()
        if isinstance(policy, SecondLegShadowPolicy)
        else policy or {}
    )
    if type(seed) is not int or not 0 <= seed < 2**31 - 128:
        raise ValueError(
            "shadow fitting seed must leave room for signed validation seeds"
        )
    if not 3 <= len(frames) <= 5:
        raise ValueError(
            "freeze requires the actual completed two-to-four-tick second leg"
        )
    prefixes, attempts, retained = [], [], None
    for end in range(1, len(frames)):
        tracked = track_frame_chain(frames[: end + 1], config=policy.future.temporal)
        retained = _append_tracks(prefixes[-1] if prefixes else None, tracked, retained)
        prefixes.append(tracked)
        if end >= 2:
            attempts.append(fit_online_landmarks(tracked, retained, camera, seed=seed))
    selected = next(
        (item.end_offset for item in attempts if item.motion.supported), None
    )
    selection = KeyframeSelection(
        terminal_offset=len(frames) - 1,
        attempts=tuple(attempts),
        selected_offset=selected,
    )
    search = inspect_direction_search(
        selection.selected(), camera, seed=seed, config=policy.future.search
    )
    return SecondLegFreeze(
        seed=seed,
        selection=selection,
        prefix_tracks=tuple(prefixes),
        retained_ids=tuple(sorted(retained)),
        search=search,
    )


def choose_shadow_handle(handles, handle_count):
    if (
        len(handles) != 2
        or handles[0] == handles[1]
        or type(handle_count) is not int
        or not 3 <= handle_count <= 32
        or any(
            type(item) is not int or not 0 <= item < handle_count for item in handles
        )
    ):
        raise ValueError(
            "shadow needs two known distinct handles and a public third handle"
        )
    return next(index for index in range(handle_count) if index not in handles)


def _support(tracks, retained, candidate, policy):
    settings = policy.future.search.geometry
    if candidate is None:
        identities = tuple(sorted(retained))
        minimum = settings.minimum_tracks
    else:
        train = set(candidate.motion.training_track_ids) & retained
        valid = set(candidate.motion.validation_track_ids) & retained
        if (
            len(train) < settings.minimum_training_inliers
            or len(valid) < settings.minimum_validation_tracks
        ):
            return False
        identities = tuple(sorted(valid))
        minimum = settings.minimum_tracks
        if len(train) + len(valid) < minimum:
            return False
    if candidate is None and len(identities) < minimum:
        return False
    pixels = np.asarray(tracks.pixels, dtype=float).reshape(tracks.frame_count, -1, 2)
    slots = [tracks.track_ids.index(identity) for identity in identities]
    return bool(
        slots
        and _cells(pixels[-1, slots], *tracks.image_shape)
        >= settings.minimum_validation_cells
    )


def acquire_second_leg_shadow(
    leg_frames, execute, handles, camera, *, seed, handle_count=4, policy=None
):
    """Freeze all fitting before the first new control; never retry or refit."""
    policy = SecondLegShadowPolicy.model_validate(
        policy.model_dump()
        if isinstance(policy, SecondLegShadowPolicy)
        else policy or {}
    )
    handle = choose_shadow_handle(handles, handle_count)
    frames = []
    for frame in leg_frames:
        image = _frame(frame).copy()
        image.setflags(write=False)
        frames.append(image)
    frozen = prepare_second_leg(frames, camera, seed=seed, policy=policy)
    digest = frozen.digest()
    candidate = frozen.selection.selected()
    retained = set(frozen.retained_ids)
    previous = frozen.prefix_tracks[-1]
    steps, prefixes, retentions = [], [], []
    terminal, failure = "support_aborted", None
    if _support(previous, retained, candidate, policy):
        for _ in range(policy.ticks):
            receipt_data = {
                "requested_handle": handle,
                "executed_handle": None,
                "ticks": None,
                "compliant": False,
            }
            steps.append(receipt_data)
            try:
                receipt = execute(handle)
                if not isinstance(receipt, PublicProbeStep):
                    raise TypeError(
                        "shadow callback requires an RGB-only execution receipt"
                    )
                receipt_data.update(
                    executed_handle=receipt.executed_handle,
                    ticks=receipt.ticks,
                    compliant=receipt.compliant,
                )
                image = _frame(receipt.frame).copy()
                image.setflags(write=False)
                frames.append(image)
                if (
                    type(receipt.executed_handle) is not int
                    or receipt.executed_handle != handle
                    or type(receipt.ticks) is not int
                    or receipt.ticks != 1
                    or type(receipt.compliant) is not bool
                    or not receipt.compliant
                ):
                    raise ValueError("failed shadow control cannot become map evidence")
            except Exception as error:  # noqa: BLE001 -- retain failures without retries
                terminal, failure = (
                    "control_failed",
                    {"type": type(error).__name__, "message": str(error)},
                )
                break
            try:
                current = track_frame_chain(frames, config=policy.future.temporal)
                retained = _append_tracks(previous, current, retained)
                previous = current
                prefixes.append(current)
                retentions.append(tuple(sorted(retained)))
                if frozen.digest() != digest:
                    raise ValueError("future control mutated the frozen map/search")
                if not _support(current, retained, candidate, policy):
                    break
            except Exception as error:  # noqa: BLE001 -- preserve partial tracking evidence
                terminal, failure = (
                    "tracking_failed",
                    {"type": type(error).__name__, "message": str(error)},
                )
                break
        else:
            terminal = "completed"
    switch = frozen.selection.terminal_offset
    targets = (switch + 2, switch + policy.ticks)
    available = terminal == "completed" and candidate is not None
    plan = ActionSwitchPlan(
        anchor_frame=0,
        candidate_end_frame=None if candidate is None else candidate.end_offset,
        available=available,
        reason="fixed new-control midpoint/end views; physical diversity unknown"
        if available
        else "no frozen candidate or incomplete shadow support",
        current_handle=handles[1],
        next_handle=handle,
        action_start_frame=switch,
        action_end_frame=switch + policy.ticks,
        requested_frames=targets,
        target_frames=targets if available else (),
    )
    return {
        "terminal": terminal,
        "failure": failure,
        "handle": handle,
        "freeze": frozen.model_dump(mode="json"),
        "freeze_sha256": digest,
        "steps": steps,
        "prefix_tracks": [item.model_dump(mode="json") for item in prefixes],
        "retained_ids": [list(item) for item in retentions],
        "plan": plan.model_dump(mode="json"),
    }
