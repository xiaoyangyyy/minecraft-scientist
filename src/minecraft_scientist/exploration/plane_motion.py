"""Homography motion proposals with explicit branch competition and held-out rays.

This development estimator never resolves a plane's two-view ambiguity by branch
order, small rotation, an intended action, or private poses. A fixed training split
fits the plane; disjoint off-plane selection and shadow rays must discriminate
all competing poses. Accepted endpoints still require the external multi-frame
and sensitivity gates before navigation, and are not metric motion estimates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .camera_intrinsics import resize_intrinsics
from .consensus_motion import estimate_consensus_endpoint
from .dense_correspondence import DenseCorrespondences
from .two_view_motion import TwoViewMotionConfig, TwoViewMotionEvidence
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class PlaneMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_training_points: int = Field(default=20, ge=8)
    minimum_partition_points: int = Field(default=8, ge=8)
    minimum_plane_inliers: int = Field(default=12, ge=8)
    minimum_plane_fraction: FiniteFloat = Field(default=0.5, gt=0, le=1)
    minimum_off_plane_points: int = Field(default=8, ge=8)
    off_plane_threshold_multiple: FiniteFloat = Field(default=3, gt=1)
    minimum_acceptance: FiniteFloat = Field(default=0.8, gt=0, le=1)
    minimum_loss_margin: FiniteFloat = Field(default=0.25, gt=0)
    equivalent_direction_degrees: FiniteFloat = Field(default=5, gt=0, lt=20)
    equivalent_rotation_degrees: FiniteFloat = Field(default=1, gt=0, lt=3)


class BranchCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    branch: int = Field(ge=0)
    family: Literal["homography", "essential_consensus"] = "homography"
    relative_rotation: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    selection_loss: FiniteFloat = Field(ge=0)
    shadow_loss: FiniteFloat = Field(ge=0)
    selection_acceptance: FiniteFloat = Field(ge=0, le=1)
    shadow_acceptance: FiniteFloat = Field(ge=0, le=1)


class PlaneMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: TwoViewMotionEvidence
    training_ids: tuple[int, ...] = ()
    selection_ids: tuple[int, ...] = ()
    shadow_ids: tuple[int, ...] = ()
    selection_off_plane_ids: tuple[int, ...] = ()
    shadow_off_plane_ids: tuple[int, ...] = ()
    branches: tuple[BranchCheck, ...] = ()
    selected_branch: int | None = Field(default=None, ge=0)
    selection_margin: FiniteFloat | None = None
    shadow_margin: FiniteFloat | None = None

    @model_validator(mode="after")
    def _partitions(self):
        groups = (self.training_ids, self.selection_ids, self.shadow_ids)
        flat = sum(groups, ())
        if flat and (
            len(set(flat)) != len(flat)
            or set(flat) != set(range(self.endpoint.matches))
        ):
            raise ValueError(
                "plane fitting, selection and shadow IDs must partition rays"
            )
        for subset, parent in (
            (self.selection_off_plane_ids, self.selection_ids),
            (self.shadow_off_plane_ids, self.shadow_ids),
        ):
            if len(set(subset)) != len(subset) or not set(subset) <= set(parent):
                raise ValueError(
                    "off-plane IDs must belong to their independent partition"
                )
        branches = {item.branch: item for item in self.branches}
        if len(branches) != len(self.branches):
            raise ValueError("duplicate plane branch")
        if self.selected_branch is not None and self.selected_branch not in branches:
            raise ValueError("selected plane branch must have diagnostic evidence")
        if self.endpoint.supported:
            branch = branches.get(self.selected_branch)
            if (
                branch is None
                or branch.relative_rotation != self.endpoint.relative_rotation
                or branch.unit_translation != self.endpoint.unit_translation
                or not self.selection_off_plane_ids
                or not self.shadow_off_plane_ids
            ):
                raise ValueError(
                    "supported plane motion requires independently checked branch"
                )
        return self


def _symmetric_plane_errors(first, second, homography):
    errors = []
    for source, target, matrix in (
        (first, second, homography),
        (second, first, np.linalg.inv(homography)),
    ):
        transformed = np.column_stack((source, np.ones(len(source)))) @ matrix.T
        scale = transformed[:, 2:]
        valid = np.abs(scale[:, 0]) > 1e-12
        projected = transformed[:, :2] / np.where(valid[:, None], scale, 1)
        error = np.linalg.norm(projected - target, axis=1)
        error[~valid] = np.inf
        errors.append(error)
    return np.maximum(*errors)


def _branch_geometry(first, second, rotation, translation, camera, config, cv2):
    tx, ty, tz = translation
    essential = np.array(((0, -tz, ty), (tz, 0, -tx), (-ty, tx, 0))) @ rotation
    inverse = np.linalg.inv(camera)
    fundamental = inverse.T @ essential @ inverse
    h0, h1 = [np.column_stack((p, np.ones(len(p)))) for p in (first, second)]
    line1, line0 = h0 @ fundamental.T, h1 @ fundamental
    numerator = np.sum(h1 * line1, axis=1) ** 2
    denominator = np.sum(line1[:, :2] ** 2 + line0[:, :2] ** 2, axis=1)
    residual = np.sqrt(numerator / np.maximum(denominator, 1e-15))
    # Competition retains low-parallax alternatives. Discarding such a branch
    # early could turn an ambiguous plane into an arbitrary "unique" solution.
    loose = VisualOdometryConfig(
        minimum_parallax_degrees=1e-5,
        reprojection_pixels=config.reference_reprojection_pixels,
    )
    points, positive = triangulate_landmarks(
        first,
        second,
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=loose,
        cv2=cv2,
    )
    center = -rotation.T @ translation
    ray0, ray1 = points.copy(), points - center
    ray0 /= np.maximum(np.linalg.norm(ray0, axis=1, keepdims=True), 1e-12)
    ray1 /= np.maximum(np.linalg.norm(ray1, axis=1, keepdims=True), 1e-12)
    angles = np.degrees(np.arccos(np.clip(np.sum(ray0 * ray1, axis=1), -1, 1)))
    valid = (
        positive
        & np.isfinite(residual)
        & (residual <= config.reference_ransac_pixels)
        & (angles >= config.minimum_parallax_degrees)
    )
    loss = np.minimum((residual / config.reference_ransac_pixels) ** 2, 9)
    loss = np.nan_to_num(loss, nan=9, posinf=9, neginf=9) + 9 * ~positive
    return loss, valid, positive, angles


def _equivalent(first, second, config):
    rotation0, translation0 = first
    rotation1, translation1 = second
    direction = np.degrees(np.arccos(np.clip(translation0 @ translation1, -1, 1)))
    rotation = np.degrees(
        np.arccos(np.clip((np.trace(rotation0 @ rotation1.T) - 1) / 2, -1, 1))
    )
    return (
        direction <= config.equivalent_direction_degrees
        and rotation <= config.equivalent_rotation_degrees
    )


def analyze_plane_motion(
    matches: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: TwoViewMotionConfig,
    plane_config: PlaneMotionConfig | None = None,
    seed: int = 0,
) -> PlaneMotionEvidence:
    plane = plane_config or PlaneMotionConfig()
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError(
            "plane-motion seed must be a nonnegative signed 32-bit integer"
        )
    if matches.image_shape0 != matches.image_shape1:
        raise ValueError("plane motion requires same-size images")
    height, width = matches.image_shape0
    camera = resize_intrinsics(np.asarray(camera), (width, height), (width, height))
    if not np.isfinite(list(config.model_dump().values())).all():
        raise ValueError("plane-motion acceptance parameters must be finite")
    first, second = [
        np.asarray(p).reshape(-1, 2) for p in (matches.points0, matches.points1)
    ]
    base, details = {"matches": len(first)}, {}

    def reject(reason):
        return PlaneMotionEvidence(
            endpoint=TwoViewMotionEvidence(supported=False, reason=reason, **base),
            **details,
        )

    minimum = max(
        config.minimum_matches,
        plane.minimum_training_points + 2 * plane.minimum_partition_points,
    )
    if len(first) < minimum:
        return reject("insufficient rays for independent plane branch selection")
    permutation = np.random.default_rng(seed).permutation(len(first))
    partition_count = min(
        max(plane.minimum_partition_points, len(first) // 4),
        (len(first) - plane.minimum_training_points) // 2,
    )
    shadow, selection, training = [
        np.sort(part)
        for part in (
            permutation[:partition_count],
            permutation[partition_count : 2 * partition_count],
            permutation[2 * partition_count :],
        )
    ]
    details.update(
        training_ids=tuple(training),
        selection_ids=tuple(selection),
        shadow_ids=tuple(shadow),
    )
    cv2 = _opencv()
    threshold = config.reference_ransac_pixels * width / 224
    cv2.setRNGSeed(seed)
    homography, mask = cv2.findHomography(
        first[training], second[training], cv2.RANSAC, threshold
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return reject("training homography unavailable")
    if abs(np.linalg.det(homography)) <= 1e-12:
        return reject("training homography is singular")
    train_inliers = mask.ravel() > 0
    if (
        train_inliers.sum() < plane.minimum_plane_inliers
        or train_inliers.mean() < plane.minimum_plane_fraction
        or min(
            _cells(first[training][train_inliers], height, width),
            _cells(second[training][train_inliers], height, width),
        )
        < config.minimum_spatial_cells
    ):
        return reject("training plane lacks count, fraction or spatial support")
    errors = _symmetric_plane_errors(first, second, homography)
    base["homography_inliers"] = int((errors <= threshold).sum())
    calibrated = np.linalg.inv(camera) @ homography @ camera
    singular = np.linalg.svd(calibrated, compute_uv=False)
    spread = float(
        (singular.max() - singular.min()) / max(float(np.median(singular)), 1e-9)
    )
    base["rotation_singular_spread"] = spread
    if (
        base["homography_inliers"] / len(first) >= config.pure_rotation_support_fraction
        and spread <= config.pure_rotation_singular_spread
    ):
        return reject("pure rotation or stationary ambiguity")
    off = errors > threshold * plane.off_plane_threshold_multiple
    selection_off, shadow_off = selection[off[selection]], shadow[off[shadow]]
    details.update(
        selection_off_plane_ids=tuple(selection_off),
        shadow_off_plane_ids=tuple(shadow_off),
    )
    _, rotations, translations, _ = cv2.decomposeHomographyMat(homography, camera)
    poses, checks, geometries = {}, [], {}
    # Native reference-pixel configuration is converted once for triangulation.
    scaled_config = config.model_copy(
        update={
            "reference_ransac_pixels": threshold,
            "reference_reprojection_pixels": config.reference_reprojection_pixels
            * width
            / 224,
        }
    )
    proposals = [
        ("homography", r, t) for r, t in zip(rotations, translations, strict=True)
    ]
    # A general-scene competitor is fit on exactly the same training rays, not
    # on selection/shadow evidence. Preserve *all* plane branches as rivals.
    trained_matches = DenseCorrespondences(
        points0=tuple(map(tuple, first[training])),
        points1=tuple(map(tuple, second[training])),
        confidence=tuple(np.asarray(matches.confidence)[training]),
        image_shape0=matches.image_shape0,
        image_shape1=matches.image_shape1,
        raw_match_count=len(training),
    )
    general = estimate_consensus_endpoint(
        trained_matches, camera, config=config, seed=seed
    )
    if general.supported:
        proposals.append(
            (
                "essential_consensus",
                np.asarray(general.relative_rotation),
                np.asarray(general.unit_translation),
            )
        )
    for index, (family, rotation, translation) in enumerate(proposals):
        norm = np.linalg.norm(translation)
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        translation = translation.ravel() / norm
        poses[index] = rotation, translation
        loss, valid, positive, angles = _branch_geometry(
            first, second, rotation, translation, camera, scaled_config, cv2
        )
        geometries[index] = valid, positive, angles
        checks.append(
            BranchCheck(
                branch=index,
                family=family,
                relative_rotation=tuple(map(tuple, rotation)),
                unit_translation=tuple(translation),
                selection_loss=float(loss[selection_off].mean())
                if len(selection_off)
                else 18,
                shadow_loss=float(loss[shadow_off].mean()) if len(shadow_off) else 18,
                selection_acceptance=float(valid[selection_off].mean())
                if len(selection_off)
                else 0,
                shadow_acceptance=float(valid[shadow_off].mean())
                if len(shadow_off)
                else 0,
            )
        )
    details["branches"] = tuple(checks)
    if (
        len(selection_off) < plane.minimum_off_plane_points
        or len(shadow_off) < plane.minimum_off_plane_points
    ):
        return reject("plane ambiguity: insufficient independent off-plane rays")
    if not checks:
        return reject("no finite homography translation branches")
    # Select once using selection rays. Shadow rays can veto, never reselect.
    winner = min(checks, key=lambda item: (item.selection_loss, item.branch))
    competitors = [
        item
        for item in checks
        if not _equivalent(poses[winner.branch], poses[item.branch], plane)
    ]
    selection_margin = min(
        (item.selection_loss - winner.selection_loss for item in competitors), default=0
    )
    shadow_margin = min(
        (item.shadow_loss - winner.shadow_loss for item in competitors), default=0
    )
    details.update(
        selected_branch=winner.branch,
        selection_margin=selection_margin,
        shadow_margin=shadow_margin,
    )
    if (
        winner.selection_acceptance < plane.minimum_acceptance
        or selection_margin < plane.minimum_loss_margin
    ):
        return reject("plane branches not distinguished by selection rays")
    if (
        winner.shadow_acceptance < plane.minimum_acceptance
        or shadow_margin < plane.minimum_loss_margin
    ):
        return reject("selected plane branch fails independent shadow rays")
    valid, positive, angles = geometries[winner.branch]
    cells = min(
        _cells(first[valid], height, width), _cells(second[valid], height, width)
    )
    base.update(
        positive_inliers=int(positive.sum()),
        triangulated_inliers=int(valid.sum()),
        spatial_cells=cells,
    )
    if (
        valid.sum() < config.minimum_positive_inliers
        or cells < config.minimum_spatial_cells
    ):
        return reject("selected plane branch lacks positive-depth parallax or coverage")
    return PlaneMotionEvidence(
        endpoint=TwoViewMotionEvidence(
            supported=True,
            reason="independently disambiguated plane branch; external accuracy unvalidated",
            relative_rotation=winner.relative_rotation,
            unit_translation=winner.unit_translation,
            median_parallax_degrees=float(np.median(angles[valid])),
            **base,
        ),
        **details,
    )


def estimate_plane_endpoint(matches, camera, *, config, seed=0, plane_config=None):
    """Endpoint hook; intermediate PnP and sensitivity still run in the caller."""
    return analyze_plane_motion(
        matches, camera, config=config, seed=seed, plane_config=plane_config
    ).endpoint
