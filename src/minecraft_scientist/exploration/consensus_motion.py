"""Conditioned consensus refit before pose rejection, without truth or gate relaxation."""

from __future__ import annotations

import numpy as np

from .camera_intrinsics import resize_intrinsics
from .dense_correspondence import DenseCorrespondences
from .two_view_motion import TwoViewMotionConfig, TwoViewMotionEvidence
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


def _conditioned_eight_point(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray | None:
    """Hartley-condition calibrated rays, uncondition, then enforce essential rank."""
    normalized, transforms = [], []
    for points in (first, second):
        center = points[:, :2].mean(axis=0)
        radius = np.linalg.norm(points[:, :2] - center, axis=1).mean()
        if not np.isfinite(radius) or radius <= 1e-9:
            return None
        scale = np.sqrt(2) / radius
        transform = np.array(
            ((scale, 0, -scale * center[0]), (0, scale, -scale * center[1]), (0, 0, 1))
        )
        transforms.append(transform)
        normalized.append(points @ transform.T)
    x0, y0 = normalized[0][:, :2].T
    x1, y1 = normalized[1][:, :2].T
    design = np.column_stack(
        (x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, np.ones(len(first)))
    )
    _, singular, basis = np.linalg.svd(design, full_matrices=True)
    if len(singular) < 8 or singular[-2] / max(singular[0], 1e-15) <= 1e-9:
        return None
    fitted = transforms[1].T @ basis[-1].reshape(3, 3) @ transforms[0]
    left, singular, right = np.linalg.svd(fitted)
    average = (singular[0] + singular[1]) / 2
    return left @ np.diag((average, average, 0)) @ right


def estimate_consensus_endpoint(
    matches: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: TwoViewMotionConfig,
    seed: int = 0,
) -> TwoViewMotionEvidence:
    """Refit the epipolar consensus *before* evaluating its physical pose.

    The RANSAC inlier field is public image evidence even if the initial minimal
    pose has poor cheirality/parallax. Rotation ambiguity is rejected first;
    count, depth, coverage, reprojection and parallax gates are unchanged.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("consensus seed must be a nonnegative signed 32-bit integer")
    if matches.image_shape0 != matches.image_shape1:
        raise ValueError("consensus endpoint requires same-size images")
    height, width = matches.image_shape0
    camera = resize_intrinsics(np.asarray(camera), (width, height), (width, height))
    parameters = np.array(
        (
            config.minimum_parallax_degrees,
            config.reference_ransac_pixels,
            config.reference_reprojection_pixels,
            config.pure_rotation_support_fraction,
            config.pure_rotation_singular_spread,
        )
    )
    if not np.isfinite(parameters).all():
        raise ValueError("consensus acceptance parameters must be finite")
    first, second = (
        np.asarray(matches.points0).reshape(-1, 2),
        np.asarray(matches.points1).reshape(-1, 2),
    )
    base = {"matches": len(first)}

    def reject(reason):
        return TwoViewMotionEvidence(supported=False, reason=reason, **base)

    if len(first) < config.minimum_matches:
        return reject("insufficient correspondences")
    base["spatial_cells"] = min(
        _cells(first, height, width), _cells(second, height, width)
    )
    if base["spatial_cells"] < config.minimum_spatial_cells:
        return reject("insufficient spatial coverage")
    cv2 = _opencv()
    threshold = config.reference_ransac_pixels * width / 224
    cv2.setRNGSeed(seed)
    homography, mask = cv2.findHomography(first, second, cv2.RANSAC, threshold)
    base["homography_inliers"] = 0 if mask is None else int(mask.sum())
    inverse = np.linalg.inv(camera)
    if homography is not None and np.isfinite(homography).all():
        singular = np.linalg.svd(inverse @ homography @ camera, compute_uv=False)
        spread = float(
            (singular.max() - singular.min()) / max(float(np.median(singular)), 1e-9)
        )
        base["rotation_singular_spread"] = spread
        if (
            base["homography_inliers"] / len(first)
            >= config.pure_rotation_support_fraction
            and spread <= config.pure_rotation_singular_spread
        ):
            return reject("pure rotation or stationary ambiguity")
    cv2.setRNGSeed(seed)
    essential, mask = cv2.findEssentialMat(
        first, second, camera, method=cv2.RANSAC, prob=0.999, threshold=threshold
    )
    if essential is None or essential.shape != (3, 3) or mask is None:
        return reject("essential model unavailable")
    homogeneous = [
        np.column_stack((points, np.ones(len(first)))) for points in (first, second)
    ]
    normalized = [points @ inverse.T for points in homogeneous]

    def inliers(matrix):
        fundamental = inverse.T @ matrix @ inverse
        line1, line0 = homogeneous[0] @ fundamental.T, homogeneous[1] @ fundamental
        numerator = np.sum(homogeneous[1] * line1, axis=1) ** 2
        denominator = np.sum(line1[:, :2] ** 2 + line0[:, :2] ** 2, axis=1)
        errors = np.sqrt(numerator / np.maximum(denominator, 1e-15))
        return np.isfinite(errors) & (errors <= threshold)

    selected = (mask.ravel() > 0) & inliers(essential)
    for _ in range(3):
        if int(selected.sum()) < config.minimum_matches:
            return reject("consensus refinement lacks epipolar support")
        fitted = _conditioned_eight_point(
            normalized[0][selected], normalized[1][selected]
        )
        if fitted is None:
            return reject("consensus design is degenerate")
        essential, selected = fitted, inliers(fitted)
    count, rotation, translation, positive = cv2.recoverPose(
        essential, first, second, camera, mask=selected.astype(np.uint8).reshape(-1, 1)
    )
    base["positive_inliers"] = int(count)
    points, valid = triangulate_landmarks(
        first,
        second,
        np.eye(3),
        np.zeros(3),
        rotation,
        translation.ravel(),
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=config.minimum_parallax_degrees,
            reprojection_pixels=config.reference_reprojection_pixels * width / 224,
        ),
        cv2=cv2,
    )
    valid &= positive.ravel() > 0
    cells = min(
        _cells(first[valid], height, width), _cells(second[valid], height, width)
    )
    base.update(triangulated_inliers=int(valid.sum()), spatial_cells=cells)
    if (
        count < config.minimum_positive_inliers
        or valid.sum() < config.minimum_positive_inliers
        or cells < config.minimum_spatial_cells
    ):
        return reject("consensus triangulation lacks depth, parallax or coverage")
    center = -rotation.T @ translation.ravel()
    rays0 = points[valid] / np.linalg.norm(points[valid], axis=1, keepdims=True)
    rays1 = points[valid] - center
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(rays0 * rays1, axis=1), -1, 1)))
    return TwoViewMotionEvidence(
        supported=True,
        reason="consensus-refitted motion; external accuracy unvalidated",
        relative_rotation=tuple(map(tuple, rotation)),
        unit_translation=tuple(translation.ravel()),
        median_parallax_degrees=float(np.median(angles)),
        **base,
    )
