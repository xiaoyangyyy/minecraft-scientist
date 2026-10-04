"""Training-only local similarities with withheld cross-namespace ray checks.

Public map consistency is not unique reconstruction, metric localization or
navigation. Every originally supported bootstrap pair is retained and tested.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .joint_multiview_map import JointMapEvidence, JointPose, _project
from .temporal_tracks import TemporalTracks
from .visual_geometry import _cells


class GaugeBridgePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_overlap_frames: int = Field(default=3, ge=3, le=12, strict=True)
    minimum_training_pairs: int = Field(default=12, ge=12, le=256, strict=True)
    minimum_validation_pairs: int = Field(default=8, ge=8, le=256, strict=True)
    minimum_total_pairs: int = Field(default=32, ge=32, le=512, strict=True)
    minimum_validation_cells: int = Field(default=2, ge=2, le=8, strict=True)
    minimum_models: int = Field(default=2, ge=2, le=10, strict=True)
    reference_matching_pixels: FiniteFloat = Field(default=0.25, gt=0, le=0.25)
    reference_path_pixels: FiniteFloat = Field(default=0.5, gt=0, le=0.5)
    reference_matching_margin_pixels: FiniteFloat = Field(default=0.1, ge=0.1, le=1)
    reference_validation_pixels: FiniteFloat = Field(default=1.5, gt=0, le=1.5)
    minimum_validation_acceptance: FiniteFloat = Field(default=0.8, ge=0.8, le=1)
    maximum_relative_point_error: FiniteFloat = Field(default=0.1, gt=0, le=0.1)
    maximum_rotation_error_degrees: FiniteFloat = Field(default=2, gt=0, le=2)
    maximum_center_error: FiniteFloat = Field(default=0.1, gt=0, le=0.1)
    minimum_spatial_ratio: FiniteFloat = Field(default=0.01, ge=0.01, le=1)


class GaugeSimilarity(BaseModel):
    """Child to parent: X_parent = scale * rotation * X_child + translation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scale: FiniteFloat = Field(gt=0)
    rotation: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    metric_scale_known: Literal[False] = False

    @model_validator(mode="after")
    def _proper_rotation(self):
        matrix = np.asarray(self.rotation)
        if (
            matrix.shape != (3, 3)
            or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6)
        ):
            raise ValueError("gauge similarity needs a proper finite rotation")
        return self

    def points(self, value):
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("gauge points must be a finite N by 3 array")
        return self.scale * points @ np.asarray(self.rotation).T + self.translation

    def inverse(self):
        rotation = np.asarray(self.rotation).T
        return GaugeSimilarity(
            scale=1 / self.scale,
            rotation=tuple(map(tuple, rotation)),
            translation=tuple(-rotation @ np.asarray(self.translation) / self.scale),
        )

    def pose(self, value):
        """Express a child's world-to-camera pose in the parent coordinates."""
        pose = JointPose.model_validate(value)
        rotation = np.asarray(pose.rotation) @ np.asarray(self.rotation).T
        return JointPose(
            frame_slot=pose.frame_slot,
            rotation=tuple(map(tuple, rotation)),
            translation=tuple(
                self.scale * np.asarray(pose.translation)
                - rotation @ np.asarray(self.translation)
            ),
        )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _frozen(value):
    snapshot = json.loads(json.dumps(value, allow_nan=False))
    past = TemporalTracks.model_validate(snapshot["past"])
    anchor = snapshot["anchor_frame"]
    if (
        type(anchor) is not int
        or anchor < 0
        or not 3 <= past.frame_count <= 12
        or snapshot["prefixes"] != []
        or snapshot["decisions"] != []
        or snapshot["navigation_accepted"] is not False
        or snapshot["metric_scale_known"] is not False
    ):
        raise ValueError("gauge registration requires immutable past-only local roots")
    maps = tuple(JointMapEvidence.model_validate(item) for item in snapshot["maps"])
    if tuple(item.bootstrap_slot for item in maps) != tuple(range(2, past.frame_count)):
        raise ValueError("gauge registration must retain every past bootstrap")
    input_hash = hashlib.sha256(past.model_dump_json().encode()).hexdigest()
    if any(
        item.input_sha256 != input_hash
        or item.fitting_frames != past.frame_count
        or not np.allclose(item.camera, maps[0].camera, rtol=0, atol=1e-9)
        or not set(item.training_ids + item.validation_ids) <= set(past.track_ids)
        or (
            item.supported
            and (
                not np.allclose(item.poses[0].rotation, np.eye(3), rtol=0, atol=1e-6)
                or not np.allclose(
                    item.poses[0].translation, (0, 0, 0), rtol=0, atol=1e-6
                )
            )
        )
        for item in maps
    ):
        raise ValueError("gauge map does not match its declared past observations")
    return snapshot, past, maps


def _similarity(child, parent, settings):
    cmean, pmean = child.mean(axis=0), parent.mean(axis=0)
    centered_c, centered_p = child - cmean, parent - pmean
    cvalues, pvalues = [
        np.linalg.svd(points, compute_uv=False) for points in (centered_c, centered_p)
    ]
    if any(
        values[0] <= 1e-9 or values[1] / values[0] < settings.minimum_spatial_ratio
        for values in (cvalues, pvalues)
    ):
        raise ValueError("training correspondence geometry is coincident or collinear")
    u, singular, vt = np.linalg.svd(centered_p.T @ centered_c)
    signs = np.array((1.0, 1.0, np.linalg.det(u @ vt)))
    rotation = u @ np.diag(signs) @ vt
    scale = float((singular * signs).sum() / np.square(centered_c).sum())
    return GaugeSimilarity(
        scale=scale,
        rotation=tuple(map(tuple, rotation)),
        translation=tuple(pmean - scale * rotation @ cmean),
    ), float(np.sqrt(np.square(centered_p).sum(axis=1).mean()))


def _pair_check(parent, child, pmodel, cmodel, pairs, slots, settings):
    pids, cids = parent.track_ids, child.track_ids
    training = [
        (pids[i], cids[j])
        for i, j in pairs
        if pids[i] in pmodel.training_ids and cids[j] in cmodel.training_ids
    ]
    validation = [
        (pids[i], cids[j])
        for i, j in pairs
        if pids[i] in pmodel.validation_ids and cids[j] in cmodel.validation_ids
    ]
    result = {
        "parent_bootstrap": pmodel.bootstrap_slot,
        "child_bootstrap": cmodel.bootstrap_slot,
        "training_pairs": training,
        "validation_pairs": validation,
        "similarity": None,
        "supported": False,
        "reason": "insufficient disjoint original training/withheld correspondences",
        "checks": [],
    }
    if (
        len(training) < settings.minimum_training_pairs
        or len(validation) < settings.minimum_validation_pairs
        or len(training) + len(validation) < settings.minimum_total_pairs
    ):
        return result
    ptrain, ctrain, pvalid, cvalid = [
        dict(zip(ids, points, strict=True))
        for ids, points in (
            (pmodel.training_ids, pmodel.training_points),
            (cmodel.training_ids, cmodel.training_points),
            (pmodel.validation_ids, pmodel.validation_points),
            (cmodel.validation_ids, cmodel.validation_points),
        )
    ]
    x, y = [
        np.asarray([mapping[pair[column]] for pair in training])
        for mapping, column in ((ctrain, 1), (ptrain, 0))
    ]
    try:
        similarity, spread = _similarity(x, y, settings)
    except ValueError as error:
        return result | {"reason": str(error)}
    # The fitted parameters exist before the first withheld point is consumed.
    result["similarity"] = similarity.model_dump(mode="json")
    training_error = np.linalg.norm(similarity.points(x) - y, axis=1) / spread
    vp, vc = [
        np.asarray([mapping[pair[column]] for pair in validation])
        for mapping, column in ((pvalid, 0), (cvalid, 1))
    ]
    validation_error = np.linalg.norm(similarity.points(vc) - vp, axis=1) / spread
    result.update(
        training_relative_error_95=float(np.quantile(training_error, 0.95)),
        validation_relative_error_95=float(np.quantile(validation_error, 0.95)),
        training_spread=spread,
    )
    pix_p, pix_c = np.asarray(parent.pixels), np.asarray(child.pixels)
    idx_p = [pids.index(pair[0]) for pair in validation]
    idx_c = [cids.index(pair[1]) for pair in validation]
    pixel_scale = parent.image_shape[1] / 224
    inverse = similarity.inverse()
    for absolute, ps, cs in slots:
        pp, cp = pmodel.poses[ps], cmodel.poses[cs]
        mapped_pose = similarity.pose(cp)
        rotation_error = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (
                            np.trace(
                                np.asarray(mapped_pose.rotation)
                                @ np.asarray(pp.rotation).T
                            )
                            - 1
                        )
                        / 2,
                        -1,
                        1,
                    )
                )
            )
        )
        centers = [
            -np.asarray(pose.rotation).T @ np.asarray(pose.translation)
            for pose in (mapped_pose, pp)
        ]
        checks = []
        for transformed, observed, pose, model in (
            (similarity.points(vc), pix_p[ps, idx_p], pp, pmodel),
            (inverse.points(vp), pix_c[cs, idx_c], cp, cmodel),
        ):
            predicted, camera_points = _project(
                transformed,
                np.asarray(pose.rotation),
                np.asarray(pose.translation),
                np.asarray(model.camera),
            )
            error = np.linalg.norm(predicted - observed, axis=1)
            positive = camera_points[:, 2] > 1e-9
            passing = positive & (
                error <= settings.reference_validation_pixels * pixel_scale
            )
            checks.append(
                {
                    "positive_depth": bool(positive.all()),
                    "acceptance": float(passing.mean()),
                    "error_95_pixels": float(np.quantile(error, 0.95)),
                    "validation_cells": _cells(observed[passing], *parent.image_shape),
                }
            )
        center_error = float(np.linalg.norm(centers[0] - centers[1]))
        supported = (
            rotation_error <= settings.maximum_rotation_error_degrees
            and center_error <= settings.maximum_center_error
            and all(
                check["positive_depth"]
                and check["acceptance"] >= settings.minimum_validation_acceptance
                and check["error_95_pixels"]
                <= settings.reference_validation_pixels * pixel_scale
                and check["validation_cells"] >= settings.minimum_validation_cells
                for check in checks
            )
        )
        result["checks"].append(
            {
                "absolute_frame": absolute,
                "rotation_error_degrees": rotation_error,
                "center_error_parent_bootstrap_units": center_error,
                "parent_rays": checks[0],
                "child_rays": checks[1],
                "supported": supported,
            }
        )
    result["supported"] = (
        result["training_relative_error_95"] <= settings.maximum_relative_point_error
        and result["validation_relative_error_95"]
        <= settings.maximum_relative_point_error
        and all(check["supported"] for check in result["checks"])
    )
    result["reason"] = (
        "disjoint withheld geometry and overlap poses consistent"
        if result["supported"]
        else "local similarity fails withheld geometry or overlap poses"
    )
    return result


def register_map_gauges(
    parent,
    child,
    *,
    parent_images,
    child_images,
    parent_namespace,
    child_namespace,
    policy=None,
):
    """Freeze a public child-to-parent link before any child future observation."""
    settings = GaugeBridgePolicy.model_validate(
        policy.model_dump() if isinstance(policy, GaugeBridgePolicy) else policy or {}
    )
    original_hashes = [_digest(value) for value in (parent, child)]
    p, past_p, pmaps = _frozen(parent)
    c, past_c, cmaps = _frozen(child)
    if (
        type(parent_namespace) is not str
        or not parent_namespace
        or type(child_namespace) is not str
        or not child_namespace
        or parent_namespace == child_namespace
        or c["anchor_frame"] <= p["anchor_frame"]
        or past_p.image_shape != past_c.image_shape
        or not np.allclose(pmaps[0].camera, cmaps[0].camera, rtol=0, atol=1e-9)
    ):
        raise ValueError(
            "gauge link needs distinct chronological calibrated namespaces"
        )
    for images, tracks in ((parent_images, past_p), (child_images, past_c)):
        if len(images) != tracks.frame_count or any(
            type(value) is not str
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in images
        ):
            raise ValueError("gauge link must bind every consumed RGB hash")
    result = {
        "policy": settings.model_dump(mode="json"),
        "parent_sha256": original_hashes[0],
        "child_sha256": original_hashes[1],
        "parent_namespace": parent_namespace,
        "child_namespace": child_namespace,
        "supported": False,
        "reason": "insufficient common past image interval",
        "overlap_frames": [],
        "track_pairs": [],
        "map_pairs": [],
        "reference_pair": None,
        "similarity": None,
        "global_gauge_aligned": False,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }
    start = c["anchor_frame"]
    stop = min(p["anchor_frame"] + past_p.frame_count, start + past_c.frame_count)
    slots = [
        (absolute, absolute - p["anchor_frame"], absolute - start)
        for absolute in range(start, stop)
    ]
    result["overlap_frames"] = [absolute for absolute, _, _ in slots]
    if any(parent_images[ps] != child_images[cs] for _, ps, cs in slots):
        raise ValueError("same absolute image slot has different RGB hashes")
    if len(slots) < settings.minimum_overlap_frames:
        return result
    x = np.asarray(past_p.pixels)[slots[0][1] : slots[-1][1] + 1]
    y = np.asarray(past_c.pixels)[: len(slots)]
    pairs = []
    if len(past_p.track_ids) and len(past_c.track_ids):
        distance = np.linalg.norm(x[0, :, None] - y[0, None, :], axis=-1)
        nearest = distance.argmin(axis=1)
        reverse = distance.argmin(axis=0)
        scale = past_p.image_shape[1] / 224
        for i, j in enumerate(nearest):
            # Mutuality must remain true at every common frame, not only the seed.
            if (
                reverse[j] != i
                or distance[i, j] > settings.reference_matching_pixels * scale
            ):
                continue
            path_error = np.linalg.norm(x[:, i] - y[:, j], axis=-1)
            if np.max(path_error) > settings.reference_path_pixels * scale:
                continue
            ambiguous = False
            for slot in range(len(slots)):
                for errors in (
                    np.linalg.norm(y[slot] - x[slot, i], axis=1),
                    np.linalg.norm(x[slot] - y[slot, j], axis=1),
                ):
                    if (
                        len(errors) > 1
                        and np.diff(np.partition(errors, 1)[:2])[0]
                        <= settings.reference_matching_margin_pixels * scale
                    ):
                        ambiguous = True
            if ambiguous:
                continue
            if any(
                np.argmin(np.linalg.norm(y[slot] - x[slot, i], axis=1)) != j
                or np.argmin(np.linalg.norm(x[slot] - y[slot, j], axis=1)) != i
                for slot in range(len(slots))
            ):
                continue
            pairs.append((i, int(j)))
    result["track_pairs"] = [
        [past_p.track_ids[i], past_c.track_ids[j]] for i, j in pairs
    ]
    supported_p = [model for model in pmaps if model.supported]
    supported_c = [model for model in cmaps if model.supported]
    result["map_pairs"] = [
        _pair_check(past_p, past_c, pm, cm, pairs, slots, settings)
        for pm in supported_p
        for cm in supported_c
    ]
    if (
        len(supported_p) >= settings.minimum_models
        and len(supported_c) >= settings.minimum_models
    ):
        # The first ordered supported pair is declared regardless of its future
        # score; a rejected pair can never be replaced by a passing runner-up.
        reference = result["map_pairs"][0]
        result["reference_pair"] = [
            reference["parent_bootstrap"],
            reference["child_bootstrap"],
        ]
        result["supported"] = all(pair["supported"] for pair in result["map_pairs"])
        if result["supported"]:
            result["similarity"] = reference["similarity"]
        result["reason"] = (
            "all original supported bootstrap pairs pass; local link only"
            if result["supported"]
            else "at least one original bootstrap pair rejects; no gauge link accepted"
        )
    else:
        result["reason"] = "insufficient originally supported maps in one namespace"
    if [_digest(value) for value in (parent, child)] != original_hashes:
        raise ValueError("gauge registration changed immutable input maps")
    return result
