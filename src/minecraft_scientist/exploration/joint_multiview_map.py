"""Bounded training-only bundle adjustment of public temporal rays.

An internally consistent local minimum is not a unique or navigation-ready map.
All poses are monocular/unscaled; future pixels never enter the past-only fitter.
"""

from __future__ import annotations

import hashlib
import math
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

from .dense_correspondence import DenseCorrespondences
from .multiview_motion import (
    IntermediateFrameCheck,
    MultiViewMotionConfig,
    estimate_temporal_endpoint,
)
from .temporal_tracks import TemporalTracks
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks

Vector = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Matrix = tuple[Vector, Vector, Vector]


class JointMapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_frames: int = Field(default=12, ge=3, le=16, strict=True)
    track_budget: int = Field(default=256, ge=32, le=512, strict=True)
    iterations: int = Field(default=30, ge=1, le=100, strict=True)
    initial_damping: FiniteFloat = Field(default=0.001, gt=0, le=1)
    reference_huber_pixels: FiniteFloat = Field(default=1, gt=0, le=4)
    survival_rule: Literal["original-partitions-minimums-v1"] = (
        "original-partitions-minimums-v1"
    )
    geometry: MultiViewMotionConfig = MultiViewMotionConfig()

    @model_validator(mode="after")
    def _budget(self):
        if self.track_budget < self.geometry.minimum_tracks:
            raise ValueError("joint-map budget cannot reduce validation support")
        return self


class JointPose(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_slot: int = Field(ge=0, strict=True)
    rotation: Matrix
    translation: Vector

    @model_validator(mode="after")
    def _rotation(self):
        rotation = np.asarray(self.rotation)
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
            raise ValueError("joint pose requires a proper rotation")
        return self


class JointFrameCheck(IntermediateFrameCheck):
    frame_slot: int = Field(ge=0, strict=True)


class JointMapEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    navigation_accepted: Literal[False] = False
    config: JointMapConfig
    seed: int = Field(ge=0, lt=2**31 - 128, strict=True)
    camera: Matrix
    bootstrap_slot: int = Field(ge=2, strict=True)
    fitting_frames: int = Field(ge=3, le=16, strict=True)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reserved_training_ids: tuple[StrictInt, ...] = ()
    validation_ids: tuple[StrictInt, ...] = ()
    training_ids: tuple[StrictInt, ...] = ()
    training_points: tuple[Vector, ...] = ()
    validation_points: tuple[Vector, ...] = ()
    poses: tuple[JointPose, ...] = ()
    training_cost_history: tuple[FiniteFloat, ...] = ()
    frame_checks: tuple[JointFrameCheck, ...] = ()

    @model_validator(mode="after")
    def _complete(self):
        _camera(self.camera)
        train, reserved, valid = map(
            set, (self.training_ids, self.reserved_training_ids, self.validation_ids)
        )
        if (
            any(
                len(set(ids)) != len(ids)
                for ids in (
                    self.training_ids,
                    self.reserved_training_ids,
                    self.validation_ids,
                )
            )
            or reserved & valid
            or any(identity < 0 for identity in reserved | valid)
            or not train <= reserved
            or len(self.training_points) != len(train)
            or self.bootstrap_slot >= self.fitting_frames
            or self.fitting_frames > self.config.maximum_frames
            or any(
                right > left + 1e-8
                for left, right in zip(
                    self.training_cost_history, self.training_cost_history[1:]
                )
            )
        ):
            raise ValueError("joint-map partitions, budget or cost history invalid")
        if self.poses and (
            tuple(pose.frame_slot for pose in self.poses)
            != tuple(range(self.fitting_frames))
            or len(self.validation_points) != len(valid)
            or len(self.frame_checks) != self.fitting_frames
            or tuple(check.frame_slot for check in self.frame_checks)
            != tuple(range(self.fitting_frames))
        ):
            raise ValueError("joint map must retain all past poses and withheld checks")
        if self.supported and (
            not self.poses
            or not all(check.supported for check in self.frame_checks)
            or not self.training_cost_history
            or len(train) < self.config.geometry.minimum_training_inliers
            or len(valid) < self.config.geometry.minimum_validation_tracks
            or len(train) + len(valid) < self.config.geometry.minimum_tracks
        ):
            raise ValueError("joint-map support requires all independent point checks")
        return self


def _camera(value):
    camera = np.array(value, dtype=float, copy=True)
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(camera[2], (0, 0, 1))
        or not np.isclose(camera[0, 1], 0)
        or not np.isclose(camera[1, 0], 0)
    ):
        raise ValueError("joint map requires zero-skew finite pinhole intrinsics")
    return camera


def _project(points, rotation, translation, camera):
    transformed = points @ rotation.T + translation
    depth = transformed[:, 2]
    predicted = transformed @ camera.T
    return predicted[:, :2] / np.maximum(depth[:, None], 1e-12), transformed


def _skew(points):
    output = np.zeros((len(points), 3, 3))
    x, y, z = points.T
    output[:, 0, 1], output[:, 0, 2] = -z, y
    output[:, 1, 0], output[:, 1, 2] = z, -x
    output[:, 2, 0], output[:, 2, 1] = -y, x
    return output


def _linearize(points, rotation, translation, observed, camera, delta):
    predicted, transformed = _project(points, rotation, translation, camera)
    errors = predicted - observed
    norms = np.linalg.norm(errors, axis=1)
    weights = np.sqrt(np.minimum(1, delta / np.maximum(norms, 1e-12)))
    x, y, z = transformed.T
    projection = np.zeros((len(points), 2, 3))
    projection[:, 0, 0] = camera[0, 0] / z
    projection[:, 1, 1] = camera[1, 1] / z
    projection[:, 0, 2] = -camera[0, 0] * x / z**2
    projection[:, 1, 2] = -camera[1, 1] * y / z**2
    pose = np.concatenate((projection @ -_skew(transformed), projection), axis=2)
    point = projection @ rotation
    return (
        pose * weights[:, None, None],
        point * weights[:, None, None],
        errors * weights[:, None],
    )


def _cost(points, rotations, translations, pixels, camera, delta):
    cost = 0.0
    for rotation, translation, observed in zip(
        rotations, translations, pixels, strict=True
    ):
        predicted, transformed = _project(points, rotation, translation, camera)
        if not np.isfinite(predicted).all() or np.any(transformed[:, 2] <= 1e-9):
            return float("inf")
        norms = np.linalg.norm(predicted - observed, axis=1)
        cost += float(
            np.where(
                norms <= delta, 0.5 * norms**2, delta * (norms - 0.5 * delta)
            ).sum()
        )
    return cost


def _bundle(
    points,
    rotations,
    translations,
    pixels,
    camera,
    bootstrap,
    config,
    cv2,
    *,
    pixel_scale=1.0,
):
    """Huber IRLS, damped Schur solve, fixed origin and normalized scale gauge."""
    points, rotations, translations = [
        np.array(item, copy=True) for item in (points, rotations, translations)
    ]
    frames, count = len(rotations), len(points)
    dimension = 6 * (frames - 1)
    delta = config.reference_huber_pixels * pixel_scale
    damping = config.initial_damping
    cost = _cost(points, rotations, translations, pixels, camera, delta)
    if not math.isfinite(cost):
        return None
    history = [cost]
    for _ in range(config.iterations):
        a, b, c = (
            np.zeros((dimension, dimension)),
            np.zeros((count, dimension, 3)),
            np.zeros((count, 3, 3)),
        )
        g, h = np.zeros(dimension), np.zeros((count, 3))
        for slot in range(frames):
            pose, point, residual = _linearize(
                points, rotations[slot], translations[slot], pixels[slot], camera, delta
            )
            c += np.einsum("nki,nkj->nij", point, point)
            h += np.einsum("nki,nk->ni", point, residual)
            if slot:
                span = slice(6 * (slot - 1), 6 * slot)
                a[span, span] += np.einsum("nki,nkj->ij", pose, pose)
                b[:, span] = np.einsum("nki,nkj->nij", pose, point)
                g[span] = np.einsum("nki,nk->i", pose, residual)
        a += np.diag(damping * np.maximum(np.diag(a), 1e-6))
        c += (
            damping
            * np.eye(3)[None]
            * np.maximum(np.diagonal(c, axis1=1, axis2=2), 1e-6)[:, None]
        )
        try:
            inverse = np.linalg.inv(c)
            reduced = a - np.einsum("ndi,nij,nej->de", b, inverse, b, optimize=True)
            rhs = -g + np.einsum("ndi,nij,nj->d", b, inverse, h, optimize=True)
            pose_step = np.linalg.solve(reduced, rhs).reshape(frames - 1, 6)
            point_step = -np.einsum(
                "nij,nj->ni", inverse, h + np.einsum("ndi,d->ni", b, pose_step.ravel())
            )
        except np.linalg.LinAlgError:
            damping *= 10
            continue
        if not np.isfinite(pose_step).all() or not np.isfinite(point_step).all():
            damping *= 10
            continue
        # Limit an iteration's rotational excursion; every trial still needs
        # strictly lower robust cost and positive depth at every observation.
        factor = min(
            1.0, 0.2 / max(float(np.linalg.norm(pose_step[:, :3], axis=1).max()), 1e-12)
        )
        trial_r, trial_t = rotations.copy(), translations.copy()
        for slot, step in enumerate(pose_step, 1):
            change = cv2.Rodrigues(step[:3] * factor)[0]
            trial_r[slot] = change @ rotations[slot]
            trial_t[slot] = change @ translations[slot] + step[3:] * factor
        trial_points = points + point_step * factor
        gauge = float(np.linalg.norm(trial_t[bootstrap]))
        if not math.isfinite(gauge) or gauge < 1e-9:
            damping *= 10
            continue
        trial_points /= gauge
        trial_t /= gauge
        trial_cost = _cost(trial_points, trial_r, trial_t, pixels, camera, delta)
        if trial_cost < cost:
            improvement = cost - trial_cost
            points, rotations, translations, cost = (
                trial_points,
                trial_r,
                trial_t,
                trial_cost,
            )
            history.append(cost)
            damping = max(damping / 3, 1e-9)
            if improvement < 1e-8 * max(history[0], 1):
                break
        else:
            damping *= 10
            if damping > 1e12:
                break
    return points, rotations, translations, tuple(history)


def _triangulate_many(pixels, rotations, translations, camera):
    """Fixed-pose DLT: withheld rays may fit points, never camera poses."""
    rows = []
    for observed, rotation, translation in zip(
        pixels, rotations, translations, strict=True
    ):
        projection = camera @ np.column_stack((rotation, translation))
        rows.extend(
            (
                observed[:, 0, None] * projection[2] - projection[0],
                observed[:, 1, None] * projection[2] - projection[1],
            )
        )
    design = np.stack(rows, axis=1)
    _, _, basis = np.linalg.svd(design, full_matrices=False)
    homogeneous = basis[:, -1]
    if np.any(np.abs(homogeneous[:, 3]) < 1e-12):
        return None
    return homogeneous[:, :3] / homogeneous[:, 3, None]


def _check(
    slot,
    train_points,
    valid_points,
    train_pixels,
    valid_pixels,
    rotation,
    translation,
    camera,
    shape,
    geometry,
):
    scale = shape[1] / 224
    train_predicted, train_camera = _project(
        train_points, rotation, translation, camera
    )
    train_error = np.linalg.norm(train_predicted - train_pixels, axis=1)
    count = int(
        (
            (train_camera[:, 2] > 0)
            & (train_error <= geometry.reference_pnp_pixels * scale)
        ).sum()
    )
    predicted, transformed = _project(valid_points, rotation, translation, camera)
    errors = np.linalg.norm(predicted - valid_pixels, axis=1)
    positive = (transformed[:, 2] > 0) & np.isfinite(errors)
    passing = positive & (errors <= geometry.reference_validation_pixels * scale)
    error95 = float(np.quantile(errors[positive], 0.95)) if positive.any() else None
    cells = _cells(valid_pixels[passing], *shape)
    supported = bool(
        count >= geometry.minimum_training_inliers
        and positive.all()
        and float(passing.mean()) >= geometry.minimum_validation_acceptance
        and error95 is not None
        and error95 <= geometry.reference_validation_pixels * scale
        and cells >= geometry.minimum_validation_cells
    )
    return JointFrameCheck(
        frame_slot=slot,
        supported=supported,
        reason="fixed withheld-point rays consistent"
        if supported
        else "fixed withheld-point rays inconsistent",
        training_inliers=count,
        validation_tracks=len(valid_points),
        validation_acceptance=float(passing.mean()),
        validation_error_95_pixels=error95,
        validation_cells=cells,
    )


def fit_joint_map(tracks, camera, *, bootstrap_slot, seed=0, config=None):
    """Past-only RGB fitting. Split raw IDs before any geometric estimation."""
    config = JointMapConfig.model_validate(
        (config.model_dump() if isinstance(config, JointMapConfig) else config) or {}
    )
    tracks = TemporalTracks.model_validate(tracks.model_dump())
    camera = _camera(camera)
    if type(seed) is not int or not 0 <= seed < 2**31 - 128:
        raise ValueError("joint map seed outside deterministic signed schedule")
    if (
        type(bootstrap_slot) is not int
        or not 2 <= bootstrap_slot < tracks.frame_count <= config.maximum_frames
    ):
        raise ValueError("joint map needs a past bootstrap and bounded complete prefix")
    base = {
        "config": config,
        "seed": seed,
        "camera": tuple(map(tuple, camera)),
        "bootstrap_slot": bootstrap_slot,
        "fitting_frames": tracks.frame_count,
        "input_sha256": hashlib.sha256(tracks.model_dump_json().encode()).hexdigest(),
    }

    def reject(reason, **kwargs):
        return JointMapEvidence(supported=False, reason=reason, **(base | kwargs))

    if len(tracks.track_ids) < config.geometry.minimum_tracks:
        return reject("insufficient common past tracks")
    # Sorting IDs makes storage order irrelevant. Selection depends on IDs/seed,
    # not epipolar residuals, validation rays or an evaluator's pose.
    order = np.argsort(tracks.track_ids)
    chosen = np.random.default_rng(seed).permutation(order)[: config.track_budget]
    validation_count = max(
        config.geometry.minimum_validation_tracks,
        math.ceil(len(chosen) * config.geometry.validation_fraction),
    )
    valid, train = [
        np.array(sorted(indices, key=lambda index: tracks.track_ids[index]))
        for indices in (chosen[:validation_count], chosen[validation_count:])
    ]
    base.update(
        reserved_training_ids=tuple(tracks.track_ids[index] for index in train),
        validation_ids=tuple(tracks.track_ids[index] for index in valid),
    )
    pixels = np.asarray(tracks.pixels, dtype=float)
    matches = DenseCorrespondences(
        points0=tuple(map(tuple, pixels[0, train])),
        points1=tuple(map(tuple, pixels[bootstrap_slot, train])),
        confidence=(1.0,) * len(train),
        image_shape0=tracks.image_shape,
        image_shape1=tracks.image_shape,
        raw_match_count=len(train),
    )
    bootstrap = estimate_temporal_endpoint(
        matches, camera, config=config.geometry.two_view, seed=seed
    )
    if not bootstrap.supported:
        return reject("training-only bootstrap unsupported: " + bootstrap.reason)
    cv2 = _opencv()
    rotation, translation = (
        np.array(bootstrap.relative_rotation),
        np.array(bootstrap.unit_translation),
    )
    points, retained = triangulate_landmarks(
        pixels[0, train],
        pixels[bootstrap_slot, train],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=config.geometry.two_view.minimum_parallax_degrees,
            reprojection_pixels=config.geometry.two_view.reference_reprojection_pixels
            * tracks.image_shape[1]
            / 224,
        ),
        cv2=cv2,
    )
    train, points = train[retained], points[retained]
    if (
        len(train) < config.geometry.minimum_training_inliers
        or len(train) + len(valid) < config.geometry.minimum_tracks
    ):
        return reject("insufficient original training landmarks after bootstrap")
    rotations, translations = (
        np.repeat(np.eye(3)[None], tracks.frame_count, axis=0),
        np.zeros((tracks.frame_count, 3)),
    )
    rotations[bootstrap_slot], translations[bootstrap_slot] = rotation, translation
    for slot in range(1, tracks.frame_count):
        if slot == bootstrap_slot:
            continue
        cv2.setRNGSeed(seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            points,
            pixels[slot, train],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=config.geometry.reference_pnp_pixels
            * tracks.image_shape[1]
            / 224,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < config.geometry.minimum_training_inliers
        ):
            return reject("training-only pose initialization lacks support")
        selected = inliers.ravel()
        vector, shift = cv2.solvePnPRefineLM(
            points[selected], pixels[slot, train[selected]], camera, None, vector, shift
        )
        rotations[slot], translations[slot] = cv2.Rodrigues(vector)[0], shift.ravel()
    fitted = _bundle(
        points,
        rotations,
        translations,
        pixels[:, train],
        camera,
        bootstrap_slot,
        config,
        cv2,
        pixel_scale=tracks.image_shape[1] / 224,
    )
    if fitted is None:
        return reject("joint fitting has nonfinite or behind-camera initialization")
    points, rotations, translations, history = fitted
    # Validation pixels are first consumed geometrically after pose optimization.
    withheld = _triangulate_many(pixels[:, valid], rotations, translations, camera)
    if withheld is None or not np.isfinite(withheld).all():
        return reject("withheld triangulation is degenerate")
    checks = tuple(
        _check(
            slot,
            points,
            withheld,
            pixels[slot, train],
            pixels[slot, valid],
            rotations[slot],
            translations[slot],
            camera,
            tracks.image_shape,
            config.geometry,
        )
        for slot in range(tracks.frame_count)
    )
    supported = all(check.supported for check in checks)
    return JointMapEvidence(
        supported=supported,
        reason="joint past map internally consistent; uniqueness/accuracy unvalidated"
        if supported
        else "joint past map fails withheld-ray validation",
        **base,
        training_ids=tuple(tracks.track_ids[index] for index in train),
        training_points=tuple(map(tuple, points)),
        validation_points=tuple(map(tuple, withheld)),
        training_cost_history=history,
        frame_checks=checks,
        poses=tuple(
            JointPose(
                frame_slot=slot,
                rotation=tuple(map(tuple, rotations[slot])),
                translation=tuple(translations[slot]),
            )
            for slot in range(tracks.frame_count)
        ),
    )


def validate_joint_future(candidate, past, extended, *, target_slots):
    """Frozen points + new training-only PnP; never rerun bundle adjustment."""
    candidate = JointMapEvidence.model_validate(candidate.model_dump())
    past, extended = [
        TemporalTracks.model_validate(item.model_dump()) for item in (past, extended)
    ]
    if (
        hashlib.sha256(past.model_dump_json().encode()).hexdigest()
        != candidate.input_sha256
    ):
        raise ValueError("joint future must use exactly the frozen fitting tracks")
    if (
        not target_slots
        or any(
            type(slot) is not int
            or not candidate.fitting_frames <= slot < extended.frame_count
            for slot in target_slots
        )
        or tuple(sorted(set(target_slots))) != tuple(target_slots)
    ):
        raise ValueError(
            "joint future targets must be distinct chronological new views"
        )
    if (
        extended.image_shape != past.image_shape
        or extended.seed_count != past.seed_count
    ):
        raise ValueError("joint future anchor changed")
    before, after = (
        {identity: index for index, identity in enumerate(item.track_ids)}
        for item in (past, extended)
    )
    retained = set(before) & set(after)
    before_pixels, after_pixels = np.asarray(past.pixels), np.asarray(extended.pixels)
    if any(
        not np.allclose(
            before_pixels[:, before[identity]],
            after_pixels[: past.frame_count, after[identity]],
            rtol=0,
            atol=1e-6,
        )
        for identity in retained
    ):
        raise ValueError("joint future revised past pixels")
    base = {
        "status": "unavailable",
        "reason": "no supported frozen joint map",
        "frame_checks": [],
        "poses": [],
        "navigation_accepted": False,
    }
    if not candidate.supported:
        return base
    train_keep = [
        index
        for index, identity in enumerate(candidate.training_ids)
        if identity in retained
    ]
    valid_keep = [
        index
        for index, identity in enumerate(candidate.validation_ids)
        if identity in retained
    ]
    training_ids = tuple(candidate.training_ids[index] for index in train_keep)
    validation_ids = tuple(candidate.validation_ids[index] for index in valid_keep)
    base.update(training_ids=list(training_ids), validation_ids=list(validation_ids))
    geometry = candidate.config.geometry
    if (
        len(training_ids) < geometry.minimum_training_inliers
        or len(validation_ids) < geometry.minimum_validation_tracks
        or len(training_ids) + len(validation_ids) < geometry.minimum_tracks
    ):
        return base | {
            "reason": "insufficient surviving original fitting/withheld support; no replacement"
        }
    camera, cv2 = _camera(candidate.camera), _opencv()
    train_points, valid_points = (
        np.array(candidate.training_points)[train_keep],
        np.array(candidate.validation_points)[valid_keep],
    )
    pixels = np.asarray(extended.pixels)
    train_slots, valid_slots = (
        [after[identity] for identity in ids] for ids in (training_ids, validation_ids)
    )
    checks, poses = [], []
    for slot in target_slots:
        cv2.setRNGSeed(candidate.seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            train_points,
            pixels[slot, train_slots],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=geometry.reference_pnp_pixels * past.image_shape[1] / 224,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < geometry.minimum_training_inliers
        ):
            checks.append(
                JointFrameCheck(
                    frame_slot=slot,
                    supported=False,
                    reason="joint future training PnP lacks support",
                    validation_tracks=len(valid_slots),
                ).model_dump(mode="json")
            )
            poses.append(None)
            continue
        ids = inliers.ravel()
        vector, shift = cv2.solvePnPRefineLM(
            train_points[ids],
            pixels[slot, np.array(train_slots)[ids]],
            camera,
            None,
            vector,
            shift,
        )
        rotation, translation = cv2.Rodrigues(vector)[0], shift.ravel()
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return base | {"reason": "joint future training PnP nonfinite"}
        check = _check(
            slot,
            train_points,
            valid_points,
            pixels[slot, train_slots],
            pixels[slot, valid_slots],
            rotation,
            translation,
            camera,
            past.image_shape,
            geometry,
        )
        checks.append(check.model_dump(mode="json"))
        poses.append(
            JointPose(
                frame_slot=slot,
                rotation=tuple(map(tuple, rotation)),
                translation=tuple(translation),
            ).model_dump(mode="json")
        )
    return base | {
        "status": "consistent"
        if all(check["supported"] for check in checks)
        else "inconsistent",
        "reason": "frozen joint map new-control pixel tests; accuracy/uniqueness unvalidated",
        "frame_checks": checks,
        "poses": poses,
    }
