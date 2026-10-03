"""Short-horizon public-RGB map validation with explicit refresh boundaries.

A consensus is an initialization-stability check, not independent observations,
metric localization, loop closure or navigation acceptance. Future support is
evaluated at its actual prefix; later track survival cannot rewrite earlier gates.
"""

from __future__ import annotations

import hashlib
import json
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

from .joint_multiview_map import (
    JointFrameCheck,
    JointMapConfig,
    JointMapEvidence,
    JointPose,
    fit_joint_map,
    validate_joint_future,
)
from .second_leg_shadow import retain_tracks
from .temporal_tracks import TemporalTracks


class CausalWindowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    horizon: int = Field(default=2, ge=1, le=4, strict=True)
    minimum_models: int = Field(default=2, ge=2, le=14, strict=True)
    maximum_direction_disagreement: FiniteFloat = Field(default=10, gt=0, le=20)
    maximum_rotation_disagreement: FiniteFloat = Field(default=2, gt=0, le=5)
    maximum_windows: int = Field(default=16, ge=1, le=64, strict=True)
    seed_stride: Literal[5] = 5
    fitting: JointMapConfig = JointMapConfig()


class PrefixFuture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["consistent", "inconsistent", "unavailable"]
    reason: str
    frame_checks: tuple[JointFrameCheck, ...]
    poses: tuple[JointPose | None, ...]
    training_ids: tuple[StrictInt, ...] = ()
    validation_ids: tuple[StrictInt, ...] = ()
    navigation_accepted: Literal[False] = False

    @model_validator(mode="after")
    def _single_prefix(self):
        if len(self.frame_checks) != len(self.poses) or len(self.poses) > 1:
            raise ValueError("causal validation must contain at most one current view")
        if self.status == "consistent" and (
            len(self.poses) != 1
            or self.poses[0] is None
            or not self.frame_checks[0].supported
        ):
            raise ValueError("consistent prefix needs a supported current-view pose")
        return self


class PrefixCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bootstrap_slot: int = Field(ge=2, strict=True)
    future: PrefixFuture


class CausalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    absolute_frame: int = Field(ge=0, strict=True)
    elapsed_ticks: int = Field(ge=1, le=4, strict=True)
    instruction: Literal["collect", "refresh", "reject"]
    reason: str
    retained_tracks: int = Field(ge=0, le=2000, strict=True)
    candidates: tuple[PrefixCandidate, ...] = ()
    maximum_direction_disagreement: FiniteFloat | None = None
    maximum_rotation_disagreement: FiniteFloat | None = None
    relative_rotation: (
        tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None
    ) = None
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    metric_scale_known: Literal[False] = False
    navigation_accepted: Literal[False] = False

    @model_validator(mode="after")
    def _motion_contract(self):
        if self.instruction == "refresh":
            rotation = np.asarray(self.relative_rotation, dtype=float)
            shift = np.asarray(self.unit_translation, dtype=float)
            if (
                rotation.shape != (3, 3)
                or shift.shape != (3,)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
                or not np.isclose(np.linalg.norm(shift), 1, atol=1e-6)
            ):
                raise ValueError(
                    "refresh requires proper relative rotation and unit translation"
                )
        elif self.relative_rotation is not None or self.unit_translation is not None:
            raise ValueError(
                "collect/reject decisions cannot contain an accepted motion"
            )
        return self


def _relative(first, second):
    rotation = np.array(second.rotation) @ np.array(first.rotation).T
    translation = np.array(second.translation) - rotation @ np.array(first.translation)
    norm = np.linalg.norm(translation)
    return rotation, translation / norm if norm > 1e-9 else None


def _consensus(poses, policy):
    if len(poses) < policy.minimum_models or any(shift is None for _, shift in poses):
        return None, None, None, None
    rotations = np.array([rotation for rotation, _ in poses])
    shifts = np.array([shift for _, shift in poses])
    direction = float(np.degrees(np.arccos(np.clip(shifts @ shifts.T, -1, 1))).max())
    angle = max(
        float(np.degrees(np.arccos(np.clip((np.trace(left @ right.T) - 1) / 2, -1, 1))))
        for left in rotations
        for right in rotations
    )
    if (
        direction > policy.maximum_direction_disagreement
        or angle > policy.maximum_rotation_disagreement
    ):
        return direction, angle, None, None
    shift = shifts.mean(axis=0)
    shift /= np.linalg.norm(shift)
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    rotation = u @ np.diag((1, 1, np.linalg.det(u @ vt))) @ vt
    return direction, angle, tuple(map(tuple, rotation)), tuple(shift)


class CausalMapWindow:
    """One frozen past window, one chronological horizon, then explicit refresh.

    The input is public TemporalTracks, not world state or camera telemetry. All
    bootstrap hypotheses are preserved. Successful different initializations use
    correlated evidence, so their agreement is never called statistical confidence.
    """

    def __init__(
        self,
        past,
        camera,
        *,
        anchor_frame,
        seed=2000,
        policy=None,
        window_index=0,
        parent_sha256=None,
    ):
        self._policy = CausalWindowPolicy.model_validate(
            policy.model_dump()
            if isinstance(policy, CausalWindowPolicy)
            else policy or {}
        )
        if (
            type(anchor_frame) is not int
            or anchor_frame < 0
            or type(seed) is not int
            or not 0 <= seed < 2**31 - 512
        ):
            raise ValueError("causal window needs bounded nonnegative frame/seed")
        if (
            type(window_index) is not int
            or not 0 <= window_index < self._policy.maximum_windows
        ):
            raise ValueError("causal refresh window budget exceeded")
        if (window_index == 0) != (parent_sha256 is None) or (
            parent_sha256 is not None
            and (
                len(parent_sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in parent_sha256
                )
            )
        ):
            raise ValueError("causal refresh lineage is invalid")
        self._past = TemporalTracks.model_validate(past.model_dump())
        self._camera = np.array(camera, dtype=float, copy=True)
        self._anchor, self._seed = anchor_frame, seed
        self._window_index, self._parent = window_index, parent_sha256
        # Every candidate freezes here, before advance can see a new track prefix.
        self._maps = tuple(
            JointMapEvidence.model_validate(
                fit_joint_map(
                    self._past,
                    self._camera,
                    bootstrap_slot=slot,
                    seed=seed,
                    config=self._policy.fitting,
                ).model_dump()
            )
            for slot in range(2, self._past.frame_count)
        )
        if not self._maps:
            raise ValueError("causal window needs at least three past frames")
        self._previous = self._past
        self._retained = set(self._past.track_ids)
        self._decisions = []
        self._prefixes = []

    @property
    def decisions(self):
        return tuple(self._decisions)

    def snapshot(self):
        # A fresh serialized copy; callers cannot mutate stored arrays/records.
        return {
            "policy": self._policy.model_dump(mode="json"),
            "anchor_frame": self._anchor,
            "seed": self._seed,
            "window_index": self._window_index,
            "parent_sha256": self._parent,
            "past": self._past.model_dump(mode="json"),
            "maps": [item.model_dump(mode="json") for item in self._maps],
            "prefixes": [item.model_dump(mode="json") for item in self._prefixes],
            "decisions": [item.model_dump(mode="json") for item in self._decisions],
            "navigation_accepted": False,
            "metric_scale_known": False,
        }

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.snapshot(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def advance(self, extended):
        if self._decisions and self._decisions[-1].instruction != "collect":
            raise ValueError(
                "terminal causal window must explicitly refresh before more input"
            )
        extended = TemporalTracks.model_validate(extended.model_dump())
        previous = self._previous
        if (
            extended.frame_count != previous.frame_count + 1
            or extended.image_shape != previous.image_shape
            or extended.seed_count != previous.seed_count
            or extended.surviving_counts[0] != previous.surviving_counts[0]
        ):
            raise ValueError(
                "causal validation needs exactly the next unchanged-anchor prefix"
            )
        retained = self._retained & set(extended.track_ids)
        before = {identity: slot for slot, identity in enumerate(previous.track_ids)}
        after = {identity: slot for slot, identity in enumerate(extended.track_ids)}
        if any(
            not np.allclose(
                np.array(previous.pixels)[:, before[identity]],
                np.array(extended.pixels)[:-1, after[identity]],
                rtol=0,
                atol=1e-6,
            )
            for identity in retained
        ):
            raise ValueError("causal prefix cannot revise previously observed pixels")
        # Intersection prevents a failed identity from returning at a later prefix.
        current = retain_tracks(extended, retained)
        elapsed = current.frame_count - self._past.frame_count
        common = {
            "absolute_frame": self._anchor + current.frame_count - 1,
            "elapsed_ticks": elapsed,
            "retained_tracks": len(retained),
        }
        if elapsed < self._policy.horizon:
            decision = CausalDecision(
                instruction="collect",
                reason="registered horizon not yet observed",
                **common,
            )
        else:
            candidates, poses = [], []
            for fitted in self._maps:
                future = PrefixFuture.model_validate(
                    validate_joint_future(
                        fitted,
                        self._past,
                        current,
                        target_slots=(current.frame_count - 1,),
                    )
                )
                if any(
                    check.frame_slot != current.frame_count - 1
                    for check in future.frame_checks
                ) or any(
                    pose is not None and pose.frame_slot != current.frame_count - 1
                    for pose in future.poses
                ):
                    raise ValueError(
                        "causal future test must refer only to the current prefix endpoint"
                    )
                candidates.append(
                    PrefixCandidate(bootstrap_slot=fitted.bootstrap_slot, future=future)
                )
                if fitted.supported and future.status == "consistent":
                    poses.append(_relative(fitted.poses[-1], future.poses[0]))
            direction, angle, rotation, translation = _consensus(poses, self._policy)
            decision = CausalDecision(
                instruction="refresh" if translation is not None else "reject",
                reason="current-prefix pixel checks and all passing initializations agree; refresh required"
                if translation is not None
                else "insufficient or disagreeing current-prefix models; no motion accepted",
                candidates=tuple(candidates),
                maximum_direction_disagreement=direction,
                maximum_rotation_disagreement=angle,
                relative_rotation=rotation,
                unit_translation=translation,
                **common,
            )
        # Invalid observations leave the previous state byte-for-byte untouched.
        self._previous, self._retained = current, retained
        self._prefixes.append(current)
        self._decisions.append(decision)
        return decision

    def refresh(self, past, *, anchor_frame):
        if not self._decisions or self._decisions[-1].instruction == "collect":
            raise ValueError("cannot refresh before a horizon decision")
        if (
            type(anchor_frame) is not int
            or anchor_frame <= self._anchor
            or anchor_frame + past.frame_count - 1 != self._decisions[-1].absolute_frame
        ):
            raise ValueError(
                "refreshed past must end at exactly the current observation"
            )
        return CausalMapWindow(
            past,
            self._camera,
            anchor_frame=anchor_frame,
            seed=self._seed + self._policy.seed_stride,
            policy=self._policy,
            window_index=self._window_index + 1,
            parent_sha256=self.digest(),
        )
