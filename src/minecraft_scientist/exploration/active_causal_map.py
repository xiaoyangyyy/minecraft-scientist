"""Bounded RGB-only acquisition with a first-eligible past-map latch.

Action handles carry no world directions. Eligibility only requests an independent
two-tick control test; it is not metric localization or navigation acceptance.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .causal_map_window import CausalMapWindow, CausalWindowPolicy
from .joint_multiview_map import _camera
from .second_leg_shadow import _append_tracks, retain_tracks
from .temporal_tracks import TemporalTrackConfig, track_frame_chain
from .visibility_acquisition import PublicProbeStep
from .visual_geometry import _frame


class ActiveMapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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
    future_handle = next(
        item for item in range(policy.handle_count) if item not in policy.handles
    )
    for tick in range(1, policy.maximum_ticks + policy.window.horizon + 1):
        validating = controller is not None
        handle = (
            future_handle
            if validating
            else policy.handles[int(tick > policy.first_control_ticks)]
        )
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
                executed_handle=receipt.executed_handle,
                compliant=receipt.compliant,
                ticks=receipt.ticks,
            )
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
                    break
                if tick >= policy.minimum_ticks:
                    candidate = CausalMapWindow(
                        previous,
                        camera,
                        anchor_frame=anchor_frame,
                        seed=seed,
                        policy=policy.window,
                    )
                    snapshot = candidate.snapshot()
                    eligible = (
                        sum(item["supported"] for item in snapshot["maps"])
                        >= policy.window.minimum_models
                    )
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
                    elif tick == policy.maximum_ticks:
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
        "snapshot": None if controller is None else controller.snapshot(),
        "snapshot_sha256": None if controller is None else controller.digest(),
        "controller_decision": None
        if decision is None
        else decision.model_dump(mode="json"),
        "navigation_accepted": False,
        "metric_scale_known": False,
    }
