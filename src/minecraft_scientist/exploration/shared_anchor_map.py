"""Versioned shared-landmark reconstruction, never a private gauge rescue.

Only consumed training rays can revise a proposed map. Original withheld points
are retriangulated from the original past views with the new training-only poses;
neither the consumed extension nor subsequent controls can fit those points.
The original root, partitions and every failed candidate remain immutable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from .causal_map_window import CausalDecision, CausalMapWindow, CausalWindowPolicy
from .joint_multiview_map import (
    JointMapConfig,
    JointMapEvidence,
    JointPose,
    _bundle,
    _camera,
    _check,
    _triangulate_many,
)
from .temporal_tracks import TemporalTracks
from .visual_geometry import _opencv


class SharedAnchorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    horizon: Literal[2] = 2
    maximum_consumed_frames: Literal[14] = 14
    minimum_models: Literal[2] = 2
    fitting: JointMapConfig = JointMapConfig(maximum_frames=14)

    @model_validator(mode="after")
    def _registered(self):
        if self.fitting != JointMapConfig(maximum_frames=14):
            raise ValueError(
                "shared reconstruction cannot relax original geometry gates"
            )
        return self


def _sha(model):
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def refine_shared_map(parent, original, consumed, *, policy=None):
    """Fit a new model in the original anchor gauge without new landmark IDs.

    All original training/withheld memberships are retained or lost, never resplit.
    The frame-zero pose and unit original bootstrap fix the monocular gauge. This
    revises estimates, not old evidence, and does not calibrate a world/block scale.
    """
    settings = SharedAnchorPolicy.model_validate(
        policy.model_dump() if isinstance(policy, SharedAnchorPolicy) else policy or {}
    )
    parent = JointMapEvidence.model_validate(parent.model_dump())
    original, consumed = [
        TemporalTracks.model_validate(value.model_dump())
        for value in (original, consumed)
    ]
    if (
        parent.input_sha256 != _sha(original)
        or parent.fitting_frames != original.frame_count
        or parent.config != JointMapConfig()
        or not original.frame_count
        < consumed.frame_count
        <= settings.maximum_consumed_frames
        or consumed.image_shape != original.image_shape
        or consumed.seed_count != original.seed_count
        or consumed.surviving_counts[: original.frame_count]
        != original.surviving_counts
        or not set(consumed.track_ids) <= set(original.track_ids)
        or not set(parent.reserved_training_ids + parent.validation_ids)
        <= set(original.track_ids)
    ):
        raise ValueError(
            "shared reconstruction needs an exact unchanged-anchor consumed extension"
        )
    before = {identity: slot for slot, identity in enumerate(original.track_ids)}
    after = {identity: slot for slot, identity in enumerate(consumed.track_ids)}
    pixels = np.asarray(consumed.pixels)
    original_pixels = np.asarray(original.pixels)
    if any(
        not np.allclose(
            original_pixels[:, before[identity]],
            pixels[: original.frame_count, after[identity]],
            atol=1e-6,
            rtol=0,
        )
        for identity in consumed.track_ids
    ):
        raise ValueError("shared reconstruction cannot revise original pixels")
    train_keep = [
        i for i, identity in enumerate(parent.training_ids) if identity in after
    ]
    train_ids = tuple(parent.training_ids[i] for i in train_keep)
    valid_ids = tuple(
        identity for identity in parent.validation_ids if identity in after
    )
    base = {
        "config": settings.fitting,
        "seed": parent.seed,
        "camera": parent.camera,
        "bootstrap_slot": parent.bootstrap_slot,
        "fitting_frames": consumed.frame_count,
        "input_sha256": _sha(consumed),
        "reserved_training_ids": tuple(
            i for i in parent.reserved_training_ids if i in after
        ),
        "validation_ids": valid_ids,
    }

    def reject(reason):
        return JointMapEvidence(supported=False, reason=reason, **base)

    if not parent.supported:
        return reject("unsupported original map retained; no new bootstrap rescue")
    if (
        not np.allclose(parent.poses[0].rotation, np.eye(3), rtol=0, atol=1e-6)
        or not np.allclose(parent.poses[0].translation, 0, rtol=0, atol=1e-6)
        or not np.isclose(
            np.linalg.norm(parent.poses[parent.bootstrap_slot].translation),
            1,
            rtol=0,
            atol=1e-6,
        )
    ):
        raise ValueError(
            "shared reconstruction requires the original origin/unit bootstrap gauge"
        )
    geometry = settings.fitting.geometry
    if (
        len(train_ids) < geometry.minimum_training_inliers
        or len(valid_ids) < geometry.minimum_validation_tracks
        or len(train_ids) + len(valid_ids) < geometry.minimum_tracks
    ):
        return reject("insufficient surviving original partitions; no replenishment")
    points = np.asarray(parent.training_points)[train_keep].copy()
    rotations = np.repeat(np.eye(3)[None], consumed.frame_count, axis=0)
    translations = np.zeros((consumed.frame_count, 3))
    rotations[: original.frame_count] = [pose.rotation for pose in parent.poses]
    translations[: original.frame_count] = [pose.translation for pose in parent.poses]
    train_slots = [after[identity] for identity in train_ids]
    valid_slots = [after[identity] for identity in valid_ids]
    camera, cv2 = _camera(parent.camera), _opencv()
    for slot in range(original.frame_count, consumed.frame_count):
        cv2.setRNGSeed(parent.seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            points,
            pixels[slot, train_slots],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=geometry.reference_pnp_pixels
            * consumed.image_shape[1]
            / 224,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < geometry.minimum_training_inliers
        ):
            return reject("consumed training-only pose initialization lacks support")
        selected = inliers.ravel()
        vector, shift = cv2.solvePnPRefineLM(
            points[selected],
            pixels[slot, np.asarray(train_slots)[selected]],
            camera,
            None,
            vector,
            shift,
        )
        rotations[slot], translations[slot] = cv2.Rodrigues(vector)[0], shift.ravel()
    fitted = _bundle(
        points,
        rotations,
        translations,
        pixels[:, train_slots],
        camera,
        parent.bootstrap_slot,
        settings.fitting,
        cv2,
        pixel_scale=consumed.image_shape[1] / 224,
    )
    if fitted is None:
        return reject("shared training reconstruction has invalid initialization")
    points, rotations, translations, history = fitted
    # Extension holdouts never fit a point. New-control pixels are not present.
    withheld = _triangulate_many(
        pixels[: original.frame_count, valid_slots],
        rotations[: original.frame_count],
        translations[: original.frame_count],
        camera,
    )
    if withheld is None or not np.isfinite(withheld).all():
        return reject("original-view withheld triangulation is degenerate")
    checks = tuple(
        _check(
            slot,
            points,
            withheld,
            pixels[slot, train_slots],
            pixels[slot, valid_slots],
            rotations[slot],
            translations[slot],
            camera,
            consumed.image_shape,
            geometry,
        )
        for slot in range(consumed.frame_count)
    )
    supported = all(check.supported for check in checks)
    return JointMapEvidence(
        **base,
        supported=supported,
        reason="shared-anchor proposal passes original and consumed withheld rays; new controls pending"
        if supported
        else "shared-anchor proposal fails independent withheld rays",
        training_ids=train_ids,
        training_points=tuple(map(tuple, points)),
        validation_points=tuple(map(tuple, withheld)),
        training_cost_history=history,
        frame_checks=checks,
        poses=tuple(
            JointPose(
                frame_slot=slot,
                rotation=tuple(map(tuple, rotation)),
                translation=tuple(translation),
            )
            for slot, (rotation, translation) in enumerate(
                zip(rotations, translations, strict=True)
            )
        ),
    )


class SharedAnchorWindow(CausalMapWindow):
    """A new frozen version with subsequent-control shadow validation.

    All originally supported models must pass, not just a passing subset. Base
    prefix handling permanently intersects identities and never refits on futures.
    """

    def __init__(self, parent_snapshot, consumed, *, policy=None):
        self._shared_policy = SharedAnchorPolicy.model_validate(
            policy.model_dump()
            if isinstance(policy, SharedAnchorPolicy)
            else policy or {}
        )
        frozen = json.loads(json.dumps(parent_snapshot, allow_nan=False))
        original = TemporalTracks.model_validate(frozen["past"])
        models = tuple(JointMapEvidence.model_validate(item) for item in frozen["maps"])
        anchor, seed = frozen["anchor_frame"], frozen["seed"]
        if (
            type(anchor) is not int
            or anchor < 0
            or type(seed) is not int
            or not 0 <= seed < 2**31 - 512
            or frozen["prefixes"]
            or frozen["decisions"]
            or frozen["navigation_accepted"] is not False
            or frozen["metric_scale_known"] is not False
            or frozen["policy"] != CausalWindowPolicy().model_dump(mode="json")
            or tuple(item.bootstrap_slot for item in models)
            != tuple(range(2, original.frame_count))
            or any(item.camera != models[0].camera for item in models)
        ):
            raise ValueError(
                "shared version requires every original frozen bootstrap, not a future/private root"
            )
        self._parent_snapshot = frozen
        self._parent_digest = _json_digest(frozen)
        self._required = tuple(item.bootstrap_slot for item in models if item.supported)
        self._policy = CausalWindowPolicy(fitting=self._shared_policy.fitting)
        self._past = TemporalTracks.model_validate(consumed.model_dump())
        self._maps = tuple(
            refine_shared_map(item, original, self._past, policy=self._shared_policy)
            for item in models
        )
        self._camera = _camera(models[0].camera)
        self._anchor, self._seed = anchor, seed
        self._window_index, self._parent = 0, self._parent_digest
        self._previous, self._retained = self._past, set(self._past.track_ids)
        self._decisions, self._prefixes = [], []

    def snapshot(self):
        return super().snapshot() | {
            "shared_anchor_policy": self._shared_policy.model_dump(mode="json"),
            "original_snapshot_sha256": self._parent_digest,
            "required_bootstraps": list(self._required),
            "original_fitting_frames": self._parent_snapshot["past"]["frame_count"],
            "shared_anchor": True,
            "global_gauge_aligned": False,
        }

    def advance(self, extended):
        # Cap fitting+future at the existing maximum joint geometry extent.
        if (
            extended.frame_count
            > self._shared_policy.maximum_consumed_frames + self._shared_policy.horizon
        ):
            raise ValueError("shared anchor future exceeds registered 16-frame budget")
        decision = super().advance(extended)
        if decision.instruction != "collect":
            supported = {item.bootstrap_slot for item in self._maps if item.supported}
            consistent = {
                item.bootstrap_slot
                for item in decision.candidates
                if item.future.status == "consistent"
            }
            if (
                len(self._required) < self._shared_policy.minimum_models
                or not set(self._required) <= supported & consistent
            ):
                decision = CausalDecision.model_validate(
                    decision.model_dump()
                    | {
                        "instruction": "reject",
                        "reason": "one or more original supported models fail; no passing-subset rescue",
                        "relative_rotation": None,
                        "unit_translation": None,
                    }
                )
                self._decisions[-1] = decision
        return decision

    def refresh(self, past, *, anchor_frame):
        raise ValueError(
            "shared versions require explicit protected-partition reconstruction, not redetection refresh"
        )
