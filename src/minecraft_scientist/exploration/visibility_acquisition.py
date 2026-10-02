"""Causal one-tick acquisition decisions from public, identity-preserving tracks.

This chooses when to request different controls, not whether a pose is correct.
Visibility extrapolation is an experimental heuristic, not a calibrated survival
probability or a physical direction estimate. No failed identity is replenished.
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

from .temporal_tracks import TemporalTrackConfig, TemporalTracks, track_frame_chain
from .visual_geometry import _cells, _frame


class VisibilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_leg_ticks: int = Field(default=2, ge=2, le=8, strict=True)
    maximum_first_ticks: int = Field(default=6, ge=2, le=12, strict=True)
    maximum_second_ticks: int = Field(default=4, ge=2, le=12, strict=True)
    minimum_tracks: int = Field(default=32, ge=32, le=800, strict=True)
    minimum_cells: int = Field(default=2, ge=2, le=16, strict=True)
    retention_trigger: FiniteFloat = Field(default=0.7, gt=0, lt=1)
    projected_edge_trigger: FiniteFloat = Field(default=0.25, gt=0, le=1)
    reference_edge_margin: FiniteFloat = Field(default=12, gt=0, le=40)
    prefix_pixel_tolerance: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)
    temporal: TemporalTrackConfig = TemporalTrackConfig()

    @model_validator(mode="after")
    def _bounded(self):
        if (
            min(self.maximum_first_ticks, self.maximum_second_ticks)
            < self.minimum_leg_ticks
            or self.maximum_first_ticks + self.maximum_second_ticks + 1
            > self.temporal.maximum_frames
        ):
            raise ValueError("both legs must fit the declared causal tracking budget")
        return self


class VisibilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tick: int = Field(ge=1, strict=True)
    phase: Literal["first", "second"]
    instruction: Literal["continue", "switch", "finish", "abort"]
    reason: str
    next_handle: StrictInt | None
    retained_ids: tuple[StrictInt, ...]
    initial_track_count: int = Field(ge=0, strict=True)
    retention_fraction: FiniteFloat = Field(ge=0, le=1)
    projected_edge_fraction: FiniteFloat = Field(ge=0, le=1)
    occupied_cells: int = Field(ge=0, le=16, strict=True)
    endpoint_tick: int | None = Field(default=None, ge=2, strict=True)
    holdout_ticks: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def _aligned(self):
        if (
            tuple(sorted(set(self.retained_ids))) != self.retained_ids
            or any(item < 0 for item in self.retained_ids)
            or len(self.retained_ids) > self.initial_track_count
            or not np.isclose(
                self.retention_fraction,
                len(self.retained_ids) / self.initial_track_count
                if self.initial_track_count
                else 0.0,
                atol=1e-12,
                rtol=0,
            )
            or (self.next_handle is None) != (self.instruction in ("finish", "abort"))
            or (self.next_handle is not None and self.next_handle < 0)
            or (self.phase == "second" and self.endpoint_tick is None)
            or (self.endpoint_tick is not None and self.endpoint_tick > self.tick)
            or (self.instruction == "finish" and self.phase != "second")
            or (
                self.instruction == "switch"
                and (self.phase != "first" or self.endpoint_tick != self.tick)
            )
        ):
            raise ValueError(
                "decision must preserve IDs and executable phase/action alignment"
            )
        expected = (
            (self.endpoint_tick + 1, self.tick)
            if self.instruction == "finish" and self.endpoint_tick is not None
            else ()
        )
        if self.holdout_ticks != expected or (expected and expected[0] >= expected[1]):
            raise ValueError(
                "only completed acquisition exposes two distinct future views"
            )
        return self


class VisibilityController:
    """A bounded probe, driven after every executed one-tick public action.

    The caller must honor ``next_handle`` before submitting the next prefix.
    Execution compliance is checked by the acquisition runner, not inferred from
    pixels here. ``finish`` means enough views requested, never accepted motion.
    Restarting/reseeding requires a new controller/anchor and a separate record.
    """

    def __init__(
        self, handles: tuple[int, int], *, policy: VisibilityPolicy | None = None
    ):
        if (
            len(handles) != 2
            or any(type(item) is not int or item < 0 for item in handles)
            or handles[0] == handles[1]
        ):
            raise ValueError("probe requires two distinct opaque public handle indices")
        self.handles = tuple(handles)
        self.policy = policy or VisibilityPolicy()
        self.next_handle = self.handles[0]
        self.endpoint_tick = None
        self._previous = None
        self._retained = None
        self._initial_count = 0
        self._decisions = []

    @property
    def decisions(self):
        return tuple(self._decisions)

    def observe(self, tracks: TemporalTracks) -> VisibilityDecision:
        if self.next_handle is None:
            raise ValueError("terminal probes cannot consume more pixels")
        tracks = TemporalTracks.model_validate(
            tracks.model_dump() if isinstance(tracks, TemporalTracks) else tracks
        )
        expected = 2 if self._previous is None else self._previous.frame_count + 1
        if tracks.frame_count != expected:
            raise ValueError(
                "each causal observation must add exactly one consecutive frame"
            )
        if tracks.frame_count > self.policy.temporal.maximum_frames:
            raise ValueError("causal probe exceeded declared tracking budget")
        current = set(tracks.track_ids)
        if self._previous is None:
            retained = current
            # Count the anchor's valid detector points, not only those surviving
            # the first executed action. Otherwise early loss is normalized away.
            initial_count = tracks.surviving_counts[0]
        else:
            previous = self._previous
            if (
                tracks.image_shape != previous.image_shape
                or tracks.seed_count != previous.seed_count
                or tracks.surviving_counts[0] != previous.surviving_counts[0]
            ):
                raise ValueError(
                    "one anchor must retain image shape and detector seed identity"
                )
            # Prefix recomputation can change reverse-cycle membership, but a
            # lost identity never returns to this probe's evidential population.
            retained = self._retained & current
            old_index = {
                identity: index for index, identity in enumerate(previous.track_ids)
            }
            new_index = {
                identity: index for index, identity in enumerate(tracks.track_ids)
            }
            for identity in retained:
                old = np.asarray(previous.pixels)[:, old_index[identity]]
                new = np.asarray(tracks.pixels)[:-1, new_index[identity]]
                if np.max(np.abs(old - new)) > self.policy.prefix_pixel_tolerance:
                    raise ValueError("future tracking changed frozen prefix pixels")
            initial_count = self._initial_count
        ids = tuple(sorted(retained))
        indices = [tracks.track_ids.index(identity) for identity in ids]
        pixels = np.asarray(tracks.pixels, dtype=float).reshape(
            tracks.frame_count, -1, 2
        )
        current_pixels = pixels[-1, indices]
        projected = current_pixels + current_pixels - pixels[-2, indices]
        height, width = tracks.image_shape
        # Same public central ROI used by temporal tracking; no HUD/terrain labels.
        margin = self.policy.reference_edge_margin * width / 224
        bounds = np.array((int(width * 0.18), int(height * 0.2)))
        upper = np.array((int(width * 0.82), int(height * 0.76)))
        endangered = (
            (projected < bounds + margin) | (projected >= upper - margin)
        ).any(axis=1)
        edge_fraction = float(endangered.mean()) if ids else 1.0
        retention = len(ids) / initial_count if initial_count else 0.0
        cells = _cells(current_pixels, height, width) if ids else 0
        tick = tracks.frame_count - 1
        phase = "first" if self.endpoint_tick is None else "second"
        risk = (
            retention <= self.policy.retention_trigger
            or edge_fraction >= self.policy.projected_edge_trigger
        )
        insufficient = (
            len(ids) < self.policy.minimum_tracks or cells < self.policy.minimum_cells
        )
        endpoint = self.endpoint_tick
        handle, instruction, reason = (
            self.next_handle,
            "continue",
            "retain visible original identities",
        )
        if insufficient:
            handle, instruction, reason = (
                None,
                "abort",
                "insufficient original track support; not a pose contradiction",
            )
        elif (
            phase == "first"
            and tick >= self.policy.minimum_leg_ticks
            and (risk or tick >= self.policy.maximum_first_ticks)
        ):
            handle, instruction = self.handles[1], "switch"
            endpoint = tick
            reason = (
                "projected visibility/retention risk"
                if risk
                else "first-leg budget reached"
            )
        elif (
            phase == "second"
            and tick - endpoint >= self.policy.minimum_leg_ticks
            and (risk or tick - endpoint >= self.policy.maximum_second_ticks)
        ):
            handle, instruction = None, "finish"
            reason = "bounded future views acquired; map validation remains separate"
        decision = VisibilityDecision(
            tick=tick,
            phase=phase,
            instruction=instruction,
            reason=reason,
            next_handle=handle,
            retained_ids=ids,
            initial_track_count=initial_count,
            retention_fraction=retention,
            projected_edge_fraction=edge_fraction,
            occupied_cells=cells,
            endpoint_tick=endpoint,
            holdout_ticks=(endpoint + 1, tick) if instruction == "finish" else (),
        )
        # Failed schema/prefix checks above leave the controller unchanged.
        self._previous, self._retained, self._initial_count = (
            tracks,
            retained,
            initial_count,
        )
        self.endpoint_tick, self.next_handle = endpoint, handle
        self._decisions.append(decision)
        return decision


@dataclass(frozen=True)
class PublicProbeStep:
    """Public execution receipt; simulator info/telemetry is not part of this API."""

    frame: np.ndarray
    executed_handle: int
    compliant: bool
    ticks: int = 1


@dataclass(frozen=True)
class VisibilityProbeRun:
    frames: tuple[np.ndarray, ...]
    steps: tuple[dict, ...]
    tracks: tuple[TemporalTracks, ...]
    decisions: tuple[VisibilityDecision, ...]
    terminal: Literal["finished", "aborted", "control_failed", "tracking_failed"]
    failure: dict | None


def acquire_visibility_probe(initial_frame, execute, handles, *, policy=None):
    """Execute causal requests against an RGB-only callback, preserving failures.

    The callback executes exactly the requested opaque handle for one tick and
    returns its public compliance receipt. It must not return simulator ``info``.
    An unsuccessful action's image is retained but cannot become track evidence.
    No retry, alternate identity, pose substitution or unbounded loop is allowed.
    """
    controller = VisibilityController(handles, policy=policy)

    def public_copy(frame):
        copied = _frame(frame).copy()
        copied.setflags(write=False)
        return copied

    frames, steps, prefixes = [public_copy(initial_frame)], [], []
    failure, terminal = None, "tracking_failed"
    for _ in range(
        controller.policy.maximum_first_ticks + controller.policy.maximum_second_ticks
    ):
        handle = controller.next_handle
        receipt = None
        attempt = {
            "requested_handle": handle,
            "executed_handle": None,
            "ticks": None,
            "compliant": False,
        }
        steps.append(attempt)
        try:
            receipt = execute(handle)
            if not isinstance(receipt, PublicProbeStep):
                raise TypeError(
                    "callback must return an RGB-only public execution receipt"
                )
            attempt.update(
                executed_handle=receipt.executed_handle,
                ticks=receipt.ticks,
                compliant=receipt.compliant,
            )
            frames.append(public_copy(receipt.frame))
            if (
                type(receipt.executed_handle) is not int
                or receipt.executed_handle != handle
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or type(receipt.compliant) is not bool
                or not receipt.compliant
            ):
                raise ValueError(
                    "failed or mismatched one-tick control cannot become evidence"
                )
        except Exception as error:  # noqa: BLE001 -- retain failed control evidence
            terminal = "control_failed"
            failure = {"type": type(error).__name__, "message": str(error)}
            break
        try:
            tracked = track_frame_chain(frames, config=controller.policy.temporal)
            decision = controller.observe(tracked)
            prefixes.append(tracked)
        except Exception as error:  # noqa: BLE001 -- preserve partial prefix failure
            failure = {"type": type(error).__name__, "message": str(error)}
            break
        if decision.instruction in ("finish", "abort"):
            terminal = "finished" if decision.instruction == "finish" else "aborted"
            break
    else:
        failure = {
            "type": "BudgetExhausted",
            "message": "controller did not terminate within declared action budget",
        }
    return VisibilityProbeRun(
        tuple(frames),
        tuple(steps),
        tuple(prefixes),
        controller.decisions,
        terminal,
        failure,
    )
