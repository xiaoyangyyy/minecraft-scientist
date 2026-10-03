"""Bounded RGB-only acquisition with a first-eligible past-map latch.

Action handles carry no world directions. Eligibility only requests an independent
two-tick control test; it is not metric localization or navigation acceptance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .causal_map_window import CausalMapWindow, CausalWindowPolicy
from .joint_multiview_map import _camera
from .second_leg_shadow import _append_tracks, retain_tracks
from .temporal_tracks import TemporalTrackConfig, track_frame_chain
from .visibility_acquisition import PublicProbeStep
from .visual_geometry import _frame


class ActiveMapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    control_strategy: Literal["past_support", "fixed_cap"] = "past_support"
    minimum_ticks: int = Field(default=3, ge=3, le=10, strict=True)
    first_control_ticks: int = Field(default=6, ge=2, le=10, strict=True)
    maximum_ticks: int = Field(default=10, ge=3, le=11, strict=True)
    handles: tuple[StrictInt, StrictInt] = (0, 2)
    handle_count: int = Field(default=4, ge=3, le=32, strict=True)
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=64)
    window: CausalWindowPolicy = CausalWindowPolicy()

    @model_validator(mode="after")
    def _budgets(self):
        if (
            self.minimum_ticks > self.maximum_ticks
            or self.first_control_ticks >= self.maximum_ticks
            or self.maximum_ticks + 1 > self.window.fitting.maximum_frames
            or self.maximum_ticks + self.window.horizon + 1
            > self.temporal.maximum_frames
            or len(set(self.handles)) != 2
            or any(
                type(item) is not int or not 0 <= item < self.handle_count
                for item in self.handles
            )
        ):
            raise ValueError(
                "active acquisition requires distinct handles and bounded views"
            )
        return self


def acquire_causal_map(
    initial_frame, execute, camera, *, anchor_frame, seed=2000, policy=None
):
    """Choose subsequent controls only from compliant public image prefixes.

    ``execute`` receives an opaque handle index and must return PublicProbeStep.
    A failed receipt is retained but cannot be appended to evidential frames.
    Every bootstrap/prefix is retained, no future/private pose selects a winner.
    """
    policy = ActiveMapPolicy.model_validate(
        policy.model_dump() if isinstance(policy, ActiveMapPolicy) else policy or {}
    )
    if type(anchor_frame) is not int or anchor_frame < 0:
        raise ValueError("active acquisition needs a nonnegative frame anchor")
    # Validate all camera/seed arguments before dispatching even one action.
    camera = _camera(camera)
    if type(seed) is not int or not 0 <= seed < 2**31 - 512:
        raise ValueError("active acquisition needs a bounded seed")
    frames = [_frame(initial_frame).copy()]
    previous, retained, controller = None, None, None
    steps, prefixes, attempts = [], [], []
    decision, failure, terminal = None, None, "exhausted"
    next_acquisition_handle, first_control_end_tick = policy.handles[0], None
    control_decisions = []
    future_handle = next(
        item for item in range(policy.handle_count) if item not in policy.handles
    )
    for tick in range(1, policy.maximum_ticks + policy.window.horizon + 1):
        validating = controller is not None
        handle = future_handle if validating else next_acquisition_handle
        receipt_data = {
            "requested_handle": handle,
            "executed_handle": None,
            "compliant": False,
            "ticks": None,
            "phase": "validation" if validating else "acquisition",
        }
        steps.append(receipt_data)
        try:
            receipt = execute(handle)
            if not isinstance(receipt, PublicProbeStep):
                raise TypeError("active map requires an RGB-only public receipt")
            receipt_data.update(
                executed_handle=receipt.executed_handle
                if type(receipt.executed_handle) is int
                else None,
                compliant=receipt.compliant
                if type(receipt.compliant) is bool
                else False,
                ticks=receipt.ticks if type(receipt.ticks) is int else None,
            )
            if (
                type(receipt.executed_handle) is not int
                or type(receipt.compliant) is not bool
                or type(receipt.ticks) is not int
            ):
                receipt_data["invalid_receipt_types"] = {
                    "executed_handle": type(receipt.executed_handle).__name__,
                    "compliant": type(receipt.compliant).__name__,
                    "ticks": type(receipt.ticks).__name__,
                }
            if (
                type(receipt.executed_handle) is not int
                or receipt.executed_handle != handle
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or type(receipt.compliant) is not bool
                or not receipt.compliant
            ):
                raise ValueError("failed control cannot enter map evidence")
            image = _frame(receipt.frame).copy()
            if image.shape != frames[0].shape:
                raise ValueError("active frame changed its acquisition shape")
        except Exception as error:  # noqa: BLE001 -- preserve failed controls
            terminal, failure = (
                "control_failed",
                {"type": type(error).__name__, "message": str(error)},
            )
            break
        frames.append(image)
        try:
            current = track_frame_chain(frames, config=policy.temporal)
            if current.frame_count != len(frames):
                raise ValueError(
                    "tracking must preserve exactly the executed image prefix"
                )
            retained = _append_tracks(previous, current, retained)
            previous = retain_tracks(current, retained)
            prefixes.append(previous.model_dump(mode="json"))
            if validating:
                decision = controller.advance(previous)
                if decision.instruction != "collect":
                    terminal = (
                        "validated" if decision.instruction == "refresh" else "rejected"
                    )
                    break
            else:
                # A lost original population cannot recover by moving longer.
                if len(retained) < policy.window.fitting.geometry.minimum_tracks:
                    terminal = "support_aborted"
                    control_decisions.append(
                        {
                            "absolute_frame": anchor_frame + tick,
                            "instruction": "abort",
                            "next_handle": None,
                            "reason": "original track support lost; no replenishment",
                            "supported_maps": None,
                            "retained_tracks": len(retained),
                        }
                    )
                    break
                supported = None
                if tick >= policy.minimum_ticks:
                    candidate = CausalMapWindow(
                        previous,
                        camera,
                        anchor_frame=anchor_frame,
                        seed=seed,
                        policy=policy.window,
                    )
                    snapshot = candidate.snapshot()
                    supported = sum(item["supported"] for item in snapshot["maps"])
                    eligible = supported >= policy.window.minimum_models
                    attempts.append(
                        {
                            "absolute_frame": anchor_frame + tick,
                            "eligible": eligible,
                            "snapshot": snapshot,
                            "snapshot_sha256": candidate.digest(),
                        }
                    )
                    if eligible:
                        controller = candidate
                instruction, reason = "collect", "continue bounded acquisition"
                if controller is not None:
                    instruction, reason = (
                        "validate",
                        "first eligible past prefix frozen before new control",
                    )
                    next_handle = future_handle
                elif tick == policy.maximum_ticks:
                    instruction, reason, next_handle = (
                        "exhausted",
                        "past-map acquisition budget exhausted",
                        None,
                    )
                elif first_control_end_tick is None and (
                    tick >= policy.first_control_ticks
                    or (
                        policy.control_strategy == "past_support"
                        and supported is not None
                        and supported < policy.window.minimum_models
                    )
                ):
                    first_control_end_tick = tick
                    next_acquisition_handle = policy.handles[1]
                    instruction = "switch"
                    reason = (
                        "insufficient past-map support; request different public control"
                        if tick < policy.first_control_ticks
                        else "first-control cap reached"
                    )
                    next_handle = next_acquisition_handle
                else:
                    next_handle = next_acquisition_handle
                control_decisions.append(
                    {
                        "absolute_frame": anchor_frame + tick,
                        "instruction": instruction,
                        "next_handle": next_handle,
                        "reason": reason,
                        "supported_maps": supported,
                        "retained_tracks": len(retained),
                    }
                )
                if instruction == "exhausted":
                    break
        except Exception as error:  # noqa: BLE001 -- preserve estimator failures
            terminal, failure = (
                "tracking_failed",
                {"type": type(error).__name__, "message": str(error)},
            )
            break
    return {
        "policy": policy.model_dump(mode="json"),
        "anchor_frame": anchor_frame,
        "seed": seed,
        "terminal": terminal,
        "failure": failure,
        "steps": steps,
        "prefixes": prefixes,
        "acquisition_attempts": attempts,
        "fit_count": sum(len(item["snapshot"]["maps"]) for item in attempts),
        "evidence_frames": len(frames),
        "future_handle": future_handle,
        "control_decisions": control_decisions,
        "first_control_end_tick": first_control_end_tick,
        "snapshot": None if controller is None else controller.snapshot(),
        "snapshot_sha256": None if controller is None else controller.digest(),
        "controller_decision": None
        if decision is None
        else decision.model_dump(mode="json"),
        "navigation_accepted": False,
        "metric_scale_known": False,
    }
