"""Test frozen map explanations after a public control switch, never replace poses.

View selection uses declared action intervals, not physical direction or truth.
Different handles do not guarantee different physical motion. This diagnostic
distinguishes unavailable comparisons from failed withheld-pixel predictions.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .keyframe_motion import KeyframeAttempt
from .motion_stability import _pose
from .multiview_motion import IntermediateFrameCheck
from .pose_direction_search import (
    DirectionSearchConfig,
    DirectionSearchEvidence,
    _fit_rotation,
    inspect_direction_search,
)
from .temporal_tracks import TemporalTrackConfig, TemporalTracks
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class PublicActionInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(gt=0, strict=True)
    kind: Literal["move", "wait", "look"]
    motion_handle: StrictInt | None = None

    @model_validator(mode="after")
    def _valid(self):
        if self.end_frame <= self.start_frame or (
            (self.kind == "move") != (self.motion_handle is not None)
        ):
            raise ValueError(
                "action intervals need positive duration and aligned handles"
            )
        if self.motion_handle is not None and self.motion_handle < 0:
            raise ValueError("public handle indices cannot be negative")
        return self


class ActionSwitchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search: DirectionSearchConfig = DirectionSearchConfig()
    # Baseline fitting remains <=32 frames. Only new tracking's declared horizon
    # grows, to cover the next distinct action without re-fitting the frozen map.
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=64)
    prefix_pixel_tolerance: float = Field(default=1e-6, gt=0, le=1e-3)


class ActionSwitchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_frame: int = Field(ge=0, strict=True)
    candidate_end_frame: int | None = Field(default=None, ge=0, strict=True)
    available: bool
    reason: str
    current_handle: StrictInt | None = None
    next_handle: StrictInt | None = None
    action_start_frame: int | None = Field(default=None, ge=0, strict=True)
    action_end_frame: int | None = Field(default=None, gt=0, strict=True)
    target_frames: tuple[StrictInt, ...] = ()
    requested_frames: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def _complete(self):
        if (
            self.candidate_end_frame is not None
            and self.candidate_end_frame < self.anchor_frame
        ):
            raise ValueError("candidate cannot end before its anchor")
        if self.available:
            if (
                self.candidate_end_frame is None
                or self.current_handle is None
                or self.next_handle is None
                or self.current_handle == self.next_handle
                or self.action_start_frame is None
                or self.action_end_frame is None
                or self.action_start_frame < self.candidate_end_frame
                or self.target_frames != self.requested_frames
                or len(self.target_frames) != 2
                or not self.candidate_end_frame
                < self.target_frames[0]
                < self.target_frames[1]
                or self.target_frames[0] <= self.action_start_frame
                or self.target_frames[1] != self.action_end_frame
            ):
                raise ValueError(
                    "available holdout needs two frames after a distinct action switch"
                )
        elif self.target_frames:
            raise ValueError("unavailable plans cannot expose usable holdout frames")
        return self


def plan_action_switch(
    candidate: KeyframeAttempt | None,
    timeline,
    *,
    anchor_frame,
    frame_count,
    config=None,
):
    config = config or ActionSwitchConfig()
    if (
        type(anchor_frame) is not int
        or anchor_frame < 0
        or type(frame_count) is not int
        or frame_count <= anchor_frame
    ):
        raise ValueError("holdout planning needs a real bounded capture anchor")
    intervals = tuple(PublicActionInterval.model_validate(item) for item in timeline)
    if (
        not intervals
        or intervals[0].start_frame != 0
        or intervals[-1].end_frame != frame_count - 1
        or any(
            left.end_frame != right.start_frame for left, right in pairwise(intervals)
        )
    ):
        raise ValueError(
            "public action timeline must cover the complete capture consecutively"
        )
    base = {"anchor_frame": anchor_frame}
    if candidate is None or not candidate.motion.supported:
        return ActionSwitchPlan(
            available=False, reason="no supported frozen candidate", **base
        )
    endpoint = anchor_frame + candidate.end_offset
    if endpoint >= frame_count:
        raise ValueError("candidate lies outside the real capture")
    base["candidate_end_frame"] = endpoint
    past_moves = [
        item
        for item in intervals
        if item.kind == "move" and item.start_frame < endpoint
    ]
    if not past_moves:
        return ActionSwitchPlan(
            available=False, reason="no preceding public movement handle", **base
        )
    handle = past_moves[-1].motion_handle
    base["current_handle"] = handle
    following = next(
        (
            item
            for item in intervals
            if item.kind == "move"
            and item.start_frame >= endpoint
            and item.motion_handle != handle
        ),
        None,
    )
    if following is None:
        return ActionSwitchPlan(
            available=False, reason="no subsequent distinct public movement", **base
        )
    middle = following.start_frame + max(
        1, (following.end_frame - following.start_frame) // 2
    )
    targets = (middle, following.end_frame)
    base.update(
        next_handle=following.motion_handle,
        action_start_frame=following.start_frame,
        action_end_frame=following.end_frame,
        requested_frames=targets,
    )
    if (
        middle == following.end_frame
        or targets[-1] - anchor_frame >= config.temporal.maximum_frames
    ):
        return ActionSwitchPlan(
            available=False,
            reason="distinct-action views exceed the fixed tracking horizon",
            **base,
        )
    return ActionSwitchPlan(
        available=True,
        reason="fixed midpoint/end views of next different control; physical diversity unknown",
        target_frames=targets,
        **base,
    )


class FrozenFutureCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["consistent", "inconsistent", "unavailable"]
    reason: str
    training_track_ids: tuple[int, ...] = ()
    validation_track_ids: tuple[int, ...] = ()
    frame_checks: tuple[IntermediateFrameCheck, ...] = ()

    @model_validator(mode="after")
    def _complete(self):
        train, valid = self.training_track_ids, self.validation_track_ids
        if (
            len(set(train)) != len(train)
            or len(set(valid)) != len(valid)
            or set(train) & set(valid)
        ):
            raise ValueError("future fitting/validation IDs must remain disjoint")
        if self.status == "unavailable":
            if self.frame_checks:
                raise ValueError("unavailable views cannot claim pixel contradictions")
        elif len(self.frame_checks) != 2 or (self.status == "consistent") != all(
            item.supported for item in self.frame_checks
        ):
            raise ValueError("future result must match both recorded frame checks")
        return self


class AlternativeFutureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_index: int = Field(ge=0, strict=True)
    prefit_plausible: bool
    future: FrozenFutureCheck | None = None

    @model_validator(mode="after")
    def _eligible(self):
        if not self.prefit_plausible and self.future is not None:
            raise ValueError(
                "failed original alternatives cannot consume holdout pixels"
            )
        return self


Comparison = Literal[
    "no_candidate",
    "unavailable",
    "reference_inconsistent",
    "ambiguity_survives",
    "tested_alternatives_rejected",
    "incomplete_comparison",
    "no_prefit_witness",
]


def _comparison(plan, search, reference, alternatives):
    if search.reference is None:
        return "no_candidate"
    if (
        not plan.available
        or not search.reference_matches_original_checks
        or reference is None
        or reference.status == "unavailable"
    ):
        return "unavailable"
    if reference.status == "inconsistent":
        return "reference_inconsistent"
    eligible = [item.future for item in alternatives if item.prefit_plausible]
    if not eligible:
        return "no_prefit_witness"
    if any(item is not None and item.status == "consistent" for item in eligible):
        return "ambiguity_survives"
    if any(item is None or item.status == "unavailable" for item in eligible):
        return "incomplete_comparison"
    return "tested_alternatives_rejected"


class ActionSwitchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ActionSwitchConfig
    plan: ActionSwitchPlan
    search: DirectionSearchEvidence
    comparison: Comparison
    reference_future: FrozenFutureCheck | None = None
    alternatives: tuple[AlternativeFutureRecord, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        if self.search.config != self.config.search:
            raise ValueError(
                "future comparisons cannot change the frozen search policy"
            )
        if self.search.reference is None and self.plan.available:
            raise ValueError("an absent candidate cannot have usable future views")
        if tuple(item.trial_index for item in self.alternatives) != tuple(
            range(len(self.search.trials))
        ):
            raise ValueError("retain the complete original alternative family")
        for record, trial in zip(self.alternatives, self.search.trials, strict=True):
            if record.prefit_plausible != trial.plausible:
                raise ValueError("holdout views cannot change prefit eligibility")
        if self.comparison != _comparison(
            self.plan, self.search, self.reference_future, self.alternatives
        ):
            raise ValueError(
                "comparison must follow the frozen reference/alternative evidence"
            )
        checks = [
            item
            for item in (self.reference_future,)
            + tuple(record.future for record in self.alternatives)
            if item is not None
        ]
        if not self.plan.available and checks:
            raise ValueError("unavailable plans cannot consume future evidence")
        for item in checks:
            if not set(item.training_track_ids) <= set(
                self.search.training_track_ids
            ) or not set(item.validation_track_ids) <= set(
                self.search.validation_track_ids
            ):
                raise ValueError("future IDs cannot be invented or reassigned")
            if (
                item.frame_checks
                and tuple(
                    check.frame_slot + self.plan.anchor_frame
                    for check in item.frame_checks
                )
                != self.plan.target_frames
            ):
                raise ValueError(
                    "only preregistered action-switch views may be checked"
                )
            if self.reference_future is not None and (
                item.training_track_ids,
                item.validation_track_ids,
            ) != (
                self.reference_future.training_track_ids,
                self.reference_future.validation_track_ids,
            ):
                raise ValueError(
                    "all maps must be compared on identical surviving partitions"
                )
        return self


def _future_map(
    candidate, extended, plan, camera, rotation, translation, *, seed, config
):
    base = {}

    def unavailable(reason):
        return FrozenFutureCheck(status="unavailable", reason=reason, **base)

    if (
        extended is None
        or extended.frame_count != plan.target_frames[-1] - plan.anchor_frame + 1
        or extended.frame_count > config.temporal.maximum_frames
    ):
        return unavailable("missing complete bounded future tracking chain")
    old = candidate.tracks
    if extended.image_shape != old.image_shape or extended.seed_count != old.seed_count:
        raise ValueError("future chain must share the real original anchor")
    old_indices = {identity: index for index, identity in enumerate(old.track_ids)}
    new_indices = {identity: index for index, identity in enumerate(extended.track_ids)}
    partitions = [
        tuple(identity for identity in ids if identity in new_indices)
        for ids in (
            candidate.motion.training_track_ids,
            candidate.motion.validation_track_ids,
        )
    ]
    training_ids, validation_ids = partitions
    base.update(training_track_ids=training_ids, validation_track_ids=validation_ids)
    identities = training_ids + validation_ids
    if not identities:
        return unavailable("no original identities survive the switched action")
    before = np.array([old_indices[item] for item in identities])
    after = np.array([new_indices[item] for item in identities])
    old_pixels, pixels = np.asarray(old.pixels), np.asarray(extended.pixels)
    if not np.allclose(
        old_pixels[:, before],
        pixels[: candidate.end_offset + 1, after],
        rtol=0,
        atol=config.prefix_pixel_tolerance,
    ):
        return unavailable("future tracking changed frozen prefix pixels")
    settings = config.search.geometry
    if (
        len(identities) < settings.minimum_tracks
        or len(training_ids) < settings.minimum_training_inliers
        or len(validation_ids) < settings.minimum_validation_tracks
    ):
        return unavailable("insufficient surviving original fitting/validation points")
    cv2, scale = _opencv(), old.image_shape[1] / 224
    points, valid = triangulate_landmarks(
        old_pixels[0, before],
        old_pixels[-1, before],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=settings.two_view.minimum_parallax_degrees,
            reprojection_pixels=settings.two_view.reference_reprojection_pixels * scale,
        ),
        cv2=cv2,
    )
    if not valid.all():
        return unavailable(
            "frozen map no longer retains its declared original identities"
        )
    train = np.arange(len(training_ids))
    valid_ids = np.arange(len(training_ids), len(identities))
    checks = []
    for absolute in plan.target_frames:
        slot = absolute - plan.anchor_frame
        # Same seed and same fitting IDs for every frozen map; no RNG search.
        cv2.setRNGSeed(seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            points[train],
            pixels[slot, after[train]],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=settings.reference_pnp_pixels * scale,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        count = 0 if inliers is None else len(inliers)
        values = {
            "frame_slot": slot,
            "training_inliers": count,
            "validation_tracks": len(validation_ids),
        }
        if not success or count < settings.minimum_training_inliers:
            checks.append(
                IntermediateFrameCheck(
                    supported=False,
                    reason="switched-view PnP lacks fitting support",
                    **values,
                )
            )
            continue
        selected = train[inliers.ravel()]
        vector, shift = cv2.solvePnPRefineLM(
            points[selected], pixels[slot, after[selected]], camera, None, vector, shift
        )
        if not np.isfinite(vector).all() or not np.isfinite(shift).all():
            checks.append(
                IntermediateFrameCheck(
                    supported=False, reason="switched-view PnP is nonfinite", **values
                )
            )
            continue
        projected_camera = (
            points[valid_ids] @ cv2.Rodrigues(vector)[0].T + shift.ravel()
        )
        projected = projected_camera @ camera.T
        errors = np.linalg.norm(
            projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)
            - pixels[slot, after[valid_ids]],
            axis=1,
        )
        positive = (projected_camera[:, 2] > 0) & np.isfinite(errors)
        passing = positive & (errors <= settings.reference_validation_pixels * scale)
        fraction = float(passing.mean())
        error95 = float(np.quantile(errors[positive], 0.95)) if positive.any() else None
        cells = _cells(pixels[slot, after[valid_ids[passing]]], *old.image_shape)
        supported = bool(
            positive.all()
            and fraction >= settings.minimum_validation_acceptance
            and error95 is not None
            and error95 <= settings.reference_validation_pixels * scale
            and cells >= settings.minimum_validation_cells
        )
        checks.append(
            IntermediateFrameCheck(
                supported=supported,
                reason="switched withheld pixels consistent"
                if supported
                else "switched withheld pixels inconsistent",
                validation_acceptance=fraction,
                validation_error_95_pixels=error95,
                validation_cells=cells,
                **values,
            )
        )
    return FrozenFutureCheck(
        status="consistent"
        if all(item.supported for item in checks)
        else "inconsistent",
        reason="fixed switched-action pixel tests; not ground-truth accuracy",
        frame_checks=tuple(checks),
        **base,
    )


def _alternative_pose(candidate, camera, trial, config):
    rotation, translation = _pose(candidate.motion.candidate)
    center = -rotation.T @ translation
    tangent = np.eye(3)[int(np.argmin(np.abs(center)))]
    tangent -= center * (tangent @ center)
    tangent /= np.linalg.norm(tangent)
    perpendicular = np.cross(center, tangent)
    theta, phi = np.radians(
        (trial.requested_direction_offset_degrees, trial.tangent_azimuth_degrees)
    )
    direction = math.cos(theta) * center + math.sin(theta) * (
        math.cos(phi) * tangent + math.sin(phi) * perpendicular
    )
    slots = [
        candidate.tracks.track_ids.index(item)
        for item in candidate.motion.training_track_ids
    ]
    pixels = np.asarray(candidate.tracks.pixels)
    first, second = [
        np.column_stack((frame[slots], np.ones(len(slots))))
        for frame in (pixels[0], pixels[-1])
    ]
    threshold = (
        config.geometry.two_view.reference_ransac_pixels
        * candidate.tracks.image_shape[1]
        / 224
    )
    changed, initial, final, steps, adjustment = _fit_rotation(
        first, second, camera, rotation, direction, config, threshold, _opencv()
    )
    # No future pixels enter this reconstruction. A changed fit invalidates it.
    if steps != trial.accepted_steps or not np.allclose(
        (initial, final, adjustment),
        (
            trial.training_objective_initial,
            trial.training_objective_final,
            trial.rotation_adjustment_degrees,
        ),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("frozen training-only alternative fit changed")
    return changed, -changed @ direction


def inspect_action_switch(
    candidate: KeyframeAttempt | None,
    extended: TemporalTracks | None,
    plan,
    camera,
    *,
    seed=0,
    config=None,
    frozen_search=None,
):
    config = config or ActionSwitchConfig()
    search = inspect_direction_search(
        candidate, camera, seed=seed, config=config.search
    )
    if frozen_search is not None and frozen_search != search:
        raise ValueError("declared frozen search differs from training-only replay")
    reference, alternatives = None, []
    if (
        candidate is not None
        and plan.candidate_end_frame != plan.anchor_frame + candidate.end_offset
    ):
        raise ValueError("action-switch plan and frozen candidate endpoints differ")
    if plan.available and search.reference_matches_original_checks:
        rotation, translation = _pose(candidate.motion.candidate)
        reference = _future_map(
            candidate,
            extended,
            plan,
            np.asarray(camera),
            rotation,
            translation,
            seed=seed,
            config=config,
        )
    for index, trial in enumerate(search.trials):
        future = None
        if trial.plausible and reference is not None:
            rotation, translation = _alternative_pose(
                candidate, np.asarray(camera), trial, config.search
            )
            future = _future_map(
                candidate,
                extended,
                plan,
                np.asarray(camera),
                rotation,
                translation,
                seed=seed,
                config=config,
            )
        alternatives.append(
            AlternativeFutureRecord(
                trial_index=index, prefit_plausible=trial.plausible, future=future
            )
        )
    return ActionSwitchEvidence(
        config=config,
        plan=plan,
        search=search,
        comparison=_comparison(plan, search, reference, alternatives),
        reference_future=reference,
        alternatives=tuple(alternatives),
    )
