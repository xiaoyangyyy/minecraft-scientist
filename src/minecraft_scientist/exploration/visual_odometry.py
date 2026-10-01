"""Bounded monocular landmark tracking from public RGB and action timing."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .visual_geometry import (
    VisualGeometryConfig,
    _cells,
    _features,
    _frame,
    _matched_indices,
    _opencv,
)


class VisualOdometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_map_points: int = Field(default=15, ge=8)
    minimum_tracking_inliers: int = Field(default=12, ge=6)
    minimum_parallax_degrees: float = Field(default=0.5, gt=0, lt=30)
    reprojection_pixels: float = Field(default=2.0, gt=0)
    landmark_budget: int = Field(default=2000, ge=32)
    features: VisualGeometryConfig = VisualGeometryConfig()
    minimum_bootstrap_observations: int = Field(default=4, ge=2)
    flow_round_trip_pixels: float = Field(default=1.0, gt=0)
    flow_patch_error: float = Field(default=30.0, gt=0)
    maximum_tracking_gap_frames: int = Field(default=3, ge=1)
    seed_flow_with_homography: bool = True

    @model_validator(mode="after")
    def _valid_budget(self) -> VisualOdometryConfig:
        if (
            max(self.minimum_map_points, self.minimum_tracking_inliers)
            > self.landmark_budget
        ):
            raise ValueError("landmark budget is below required pose support")
        if self.minimum_map_points > self.features.feature_budget:
            raise ValueError("feature budget is below required bootstrap support")
        return self


class VisualPoseEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_index: int = Field(ge=0)
    status: str
    initialized: bool
    tracked: bool
    camera_center: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    inliers: int = Field(default=0, ge=0)
    map_points: int = Field(default=0, ge=0)
    origin_frame_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _valid_pose(self) -> VisualPoseEvidence:
        if self.tracked != (self.camera_center is not None):
            raise ValueError("only supported poses may contain a camera center")
        if self.tracked and (not self.initialized or self.origin_frame_index is None):
            raise ValueError("supported poses require an initialized visual origin")
        if (
            self.origin_frame_index is not None
            and self.origin_frame_index > self.frame_index
        ):
            raise ValueError("visual origin cannot occur after the current frame")
        return self


def _tracking_gray(frame: np.ndarray, cv2: Any) -> np.ndarray:
    rgb = _frame(frame)
    height, width = rgb.shape[:2]
    x0, x1 = int(width * 0.18), int(width * 0.82)
    y0, y1 = int(height * 0.2), int(height * 0.76)
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    crop = cv2.medianBlur(crop, 3)
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    gray = np.zeros((height, width), dtype=np.uint8)
    gray[y0:y1, x0:x1] = crop
    return gray


def triangulate_landmarks(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_rotation: np.ndarray,
    first_translation: np.ndarray,
    second_rotation: np.ndarray,
    second_translation: np.ndarray,
    camera: np.ndarray,
    *,
    config: VisualOdometryConfig,
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject behind-camera, low-parallax and high-reprojection 3D points."""
    if len(first_points) == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool)
    first_projection = camera @ np.column_stack((first_rotation, first_translation))
    second_projection = camera @ np.column_stack((second_rotation, second_translation))
    homogeneous = cv2.triangulatePoints(
        first_projection, second_projection, first_points.T, second_points.T
    )
    valid_scale = np.abs(homogeneous[3]) > 1e-9
    points = (homogeneous[:3] / np.where(valid_scale, homogeneous[3], 1)).T
    first_camera = points @ first_rotation.T + first_translation.reshape(1, 3)
    second_camera = points @ second_rotation.T + second_translation.reshape(1, 3)
    valid = (
        valid_scale
        & np.isfinite(points).all(axis=1)
        & (first_camera[:, 2] > 0)
        & (second_camera[:, 2] > 0)
    )
    for predicted, observed in (
        (first_camera, first_points),
        (second_camera, second_points),
    ):
        projected = predicted @ camera.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
        valid &= (
            np.linalg.norm(projected - observed, axis=1) <= config.reprojection_pixels
        )
    first_center = -first_rotation.T @ first_translation.reshape(3)
    second_center = -second_rotation.T @ second_translation.reshape(3)
    first_rays, second_rays = points - first_center, points - second_center
    first_rays /= np.maximum(np.linalg.norm(first_rays, axis=1, keepdims=True), 1e-9)
    second_rays /= np.maximum(np.linalg.norm(second_rays, axis=1, keepdims=True), 1e-9)
    cosine = np.sum(first_rays * second_rays, axis=1)
    valid &= cosine <= np.cos(np.radians(config.minimum_parallax_degrees))
    return points, valid


class RGBVisualOdometry:
    """Camera poses and bounded landmarks in one unscaled visual coordinate frame.

    A retained pre-motion anchor and recurrent pixel tracks may initialize the
    map through essential geometry and triangulation. Subsequent
    poses require 3D/2D PnP support; an unsupported frame is explicitly lost and
    never integrated as if it had been observed. Simulator poses are not inputs.
    """

    def __init__(
        self, camera: np.ndarray, *, config: VisualOdometryConfig | None = None
    ) -> None:
        self.config = config or VisualOdometryConfig()
        self.cv2 = _opencv()
        self.camera = np.asarray(camera, dtype=np.float64)
        if self.camera.shape != (3, 3) or not np.isfinite(self.camera).all():
            raise ValueError("visual odometry requires a finite 3x3 camera matrix")
        if (
            self.camera[0, 0] <= 0
            or self.camera[1, 1] <= 0
            or not np.allclose(self.camera[2], (0, 0, 1))
            or abs(np.linalg.det(self.camera)) < 1e-9
        ):
            raise ValueError("visual odometry requires nonsingular pinhole intrinsics")
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.points = np.empty((0, 3), dtype=np.float64)
        self.descriptors = None
        self.previous_features = None
        self.previous_rotation = self.rotation.copy()
        self.previous_translation = self.translation.copy()
        self.frame_index = -1
        self.initialized = False
        self.previous_gray = None
        self.active_points = np.empty((0, 3))
        self.active_pixels = np.empty((0, 2))
        self.anchor_features = None
        self.anchor_pixels = np.empty((0, 2))
        self.anchor_indices = np.empty(0, dtype=np.int64)
        self.anchor_gray = None
        self.bootstrap_observations = 0
        self.motion_started = False
        self.origin_frame_index = None
        self.frame_shape = None
        self.flow_homography = None
        self.last_tracked_frame = None

    def _flow(
        self, first_gray: np.ndarray, second_gray: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return np.empty((0, 2)), np.zeros(0, dtype=bool)
        height, width = second_gray.shape
        x0, x1 = int(width * 0.18), int(width * 0.82)
        y0, y1 = int(height * 0.2), int(height * 0.76)
        offset = np.array((x0, y0), dtype=np.float32)
        # Build pyramids inside the world crop. A masked full-frame pyramid
        # creates artificial stationary edges that bias coarse optical flow.
        first_crop = np.ascontiguousarray(first_gray[y0:y1, x0:x1])
        second_crop = np.ascontiguousarray(second_gray[y0:y1, x0:x1])
        source = np.asarray(points - offset, dtype=np.float32).reshape(-1, 1, 2)
        options = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        initial = None
        if self.flow_homography is not None:
            initial = (
                self.cv2.perspectiveTransform(source + offset, self.flow_homography)
                - offset
            )
            if np.isfinite(initial).all():
                options["flags"] = self.cv2.OPTFLOW_USE_INITIAL_FLOW
            else:
                initial = None
        target, status, error = self.cv2.calcOpticalFlowPyrLK(
            first_crop, second_crop, source, initial, **options
        )
        if target is None or status is None or error is None:
            return np.zeros_like(points), np.zeros(len(points), dtype=bool)
        backward, reverse_status, _ = self.cv2.calcOpticalFlowPyrLK(
            second_crop,
            first_crop,
            target,
            source.copy() if initial is not None else None,
            **options,
        )
        if backward is None or reverse_status is None:
            return target.reshape(-1, 2) + offset, np.zeros(len(points), dtype=bool)
        pixels = target.reshape(-1, 2).astype(np.float64) + offset
        valid = (
            (status.ravel() > 0)
            & (reverse_status.ravel() > 0)
            & np.isfinite(pixels).all(axis=1)
            & (error.ravel() <= self.config.flow_patch_error)
            & (
                np.linalg.norm(backward.reshape(-1, 2) + offset - points, axis=1)
                <= self.config.flow_round_trip_pixels
            )
            & (pixels[:, 0] >= width * 0.18 + 3)
            & (pixels[:, 0] < width * 0.82 - 3)
            & (pixels[:, 1] >= height * 0.2 + 3)
            & (pixels[:, 1] < height * 0.76 - 3)
        )
        return pixels, valid

    def _set_anchor(self, features: tuple[np.ndarray, Any], gray: np.ndarray) -> None:
        self.anchor_features = features
        self.anchor_pixels = features[0].copy()
        self.anchor_indices = np.arange(len(features[0]))
        self.anchor_gray = gray
        self.bootstrap_observations = 0
        self.origin_frame_index = self.frame_index

    def _append_landmarks(self, points: np.ndarray, descriptors: Any) -> None:
        if len(points) == 0 or descriptors is None:
            return
        self.points = np.concatenate((self.points, points))[
            -self.config.landmark_budget :
        ]
        self.descriptors = (
            descriptors.copy()
            if self.descriptors is None
            else np.concatenate((self.descriptors, descriptors))
        )
        self.descriptors = self.descriptors[-self.config.landmark_budget :]

    def _map_matches(
        self, features: tuple[np.ndarray, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels, descriptors = features
        if descriptors is None or self.descriptors is None or len(pixels) < 2:
            return np.empty((0, 3)), np.empty((0, 2))
        matcher = self.cv2.BFMatcher(self.cv2.NORM_L2)
        matches = matcher.knnMatch(descriptors, self.descriptors, k=2)
        good = [
            pair[0]
            for pair in matches
            if len(pair) == 2
            and pair[0].distance
            < self.config.features.ratio_threshold * pair[1].distance
        ]
        # A landmark cannot be assigned to multiple current pixels.
        unique = {}
        for match in sorted(
            good, key=lambda item: (item.distance, item.queryIdx, item.trainIdx)
        ):
            unique.setdefault(match.trainIdx, match)
        return (
            np.asarray(
                [self.points[item.trainIdx] for item in unique.values()]
            ).reshape(-1, 3),
            np.asarray([pixels[item.queryIdx] for item in unique.values()]).reshape(
                -1, 2
            ),
        )

    def process(self, frame: np.ndarray, *, motion_allowed: bool) -> VisualPoseEvidence:
        array = _frame(frame)
        if self.frame_shape is not None and array.shape != self.frame_shape:
            raise ValueError("visual odometry frame dimensions changed")
        self.frame_shape = array.shape
        features = _features(frame, self.config.features, self.cv2)
        gray = _tracking_gray(frame, self.cv2)
        self.frame_index += 1
        self.flow_homography = None
        if self.previous_features is None:
            self.previous_features = features
            self.previous_gray = gray
            self._set_anchor(features, gray)
            return self._result("uninitialized", False, 0)
        if not self.initialized:
            if not self.motion_started and not motion_allowed:
                self._set_anchor(features, gray)
                return self._result("awaiting_motion", False, 0)
            self.motion_started = True
            flowed, valid = self._flow(self.anchor_gray, gray, self.anchor_pixels)
            self.anchor_pixels = flowed[valid]
            self.anchor_indices = self.anchor_indices[valid]
            self.anchor_gray = gray
            self.bootstrap_observations += 1
            inliers = self._initialize_from_anchor(features)
            if (
                not self.initialized
                and len(self.anchor_indices) < self.config.minimum_map_points
            ):
                # A new submap origin is explicit; no unknown displacement is
                # ever composed into a previously accepted metric pose.
                self._set_anchor(features, gray)
            self.previous_features = features
            self.previous_gray = gray
            self.previous_rotation = self.rotation.copy()
            self.previous_translation = self.translation.copy()
            if self.initialized:
                self.last_tracked_frame = self.frame_index
            return self._result(
                "initialized" if self.initialized else "awaiting_parallax",
                self.initialized,
                inliers,
            )
        if (
            self.frame_index - self.last_tracked_frame
            > self.config.maximum_tracking_gap_frames
        ):
            # Repeated terrain descriptors are not independent relocalization
            # evidence. A disconnected submap must not silently inherit a pose.
            return self._result("lost_temporal_continuity", False, 0)
        if self.config.seed_flow_with_homography:
            first_ids, second_ids = _matched_indices(
                self.previous_features, features, self.config.features, self.cv2
            )
            if len(first_ids) >= self.config.features.minimum_matches:
                self.cv2.setRNGSeed(self.frame_index)
                homography, support = self.cv2.findHomography(
                    self.previous_features[0][first_ids],
                    features[0][second_ids],
                    self.cv2.RANSAC,
                    self.config.reprojection_pixels,
                )
                if (
                    homography is not None
                    and np.isfinite(homography).all()
                    and support is not None
                    and int(support.sum()) >= self.config.features.minimum_inliers
                    and _cells(
                        features[0][second_ids[support.ravel() > 0]], *gray.shape
                    )
                    >= self.config.features.minimum_cells
                ):
                    self.flow_homography = homography
        flowed, valid = self._flow(self.previous_gray, gray, self.active_pixels)
        points, pixels = self.active_points[valid], flowed[valid]
        map_points, map_pixels = self._map_matches(features)
        if len(map_points):
            novel = np.ones(len(map_points), dtype=bool)
            if len(points):
                novel &= ~np.any(
                    np.linalg.norm(map_pixels[:, None] - pixels[None], axis=2) < 2,
                    axis=1,
                )
                novel &= ~np.any(
                    np.linalg.norm(map_points[:, None] - points[None], axis=2) < 1e-8,
                    axis=1,
                )
            points = np.concatenate((points, map_points[novel]))
            pixels = np.concatenate((pixels, map_pixels[novel]))
        if len(points) < self.config.minimum_tracking_inliers:
            return self._result("lost_insufficient_landmarks", False, len(points))
        self.cv2.setRNGSeed(self.frame_index)
        solved, vector, translation, indices = self.cv2.solvePnPRansac(
            points,
            pixels,
            self.camera,
            None,
            iterationsCount=1000,
            reprojectionError=self.config.reprojection_pixels,
            confidence=0.999,
            flags=self.cv2.SOLVEPNP_EPNP,
        )
        if (
            not solved
            or indices is None
            or len(indices) < self.config.minimum_tracking_inliers
        ):
            return self._result(
                "lost_pnp_support", False, 0 if indices is None else len(indices)
            )
        chosen = indices.ravel()
        vector, translation = self.cv2.solvePnPRefineLM(
            points[chosen], pixels[chosen], self.camera, None, vector, translation
        )
        rotation = self.cv2.Rodrigues(vector)[0]
        projected, _ = self.cv2.projectPoints(
            points[chosen], vector, translation, self.camera, None
        )
        depths = (points[chosen] @ rotation.T + translation.ravel())[:, 2]
        supported = (
            np.isfinite(projected.reshape(-1, 2)).all(axis=1)
            & (depths > 0)
            & (
                np.linalg.norm(projected.reshape(-1, 2) - pixels[chosen], axis=1)
                <= self.config.reprojection_pixels
            )
        )
        chosen = chosen[supported]
        if (
            len(chosen) < self.config.minimum_tracking_inliers
            or _cells(pixels[chosen], *gray.shape) < self.config.features.minimum_cells
        ):
            return self._result("lost_refined_support", False, len(chosen))
        previous_flow, previous_valid = self._flow(
            self.previous_gray, gray, self.previous_features[0]
        )
        previous_indices = np.flatnonzero(previous_valid)
        previous_pixels = self.previous_features[0][previous_indices]
        current_pixels = previous_flow[previous_valid]
        triangulated, valid = triangulate_landmarks(
            previous_pixels,
            current_pixels,
            self.previous_rotation,
            self.previous_translation,
            rotation,
            translation.ravel(),
            self.camera,
            config=self.config,
            cv2=self.cv2,
        )
        if len(pixels):
            # Do not duplicate landmarks already matched in this frame. Keeping
            # identical descriptors twice would make every future ratio test fail.
            existing = np.any(
                np.linalg.norm(current_pixels[:, None, :] - pixels[None, :, :], axis=2)
                < 2.0,
                axis=1,
            )
            valid &= ~existing
        if np.any(valid) and features[1] is not None:
            self._append_landmarks(
                triangulated[valid], self.previous_features[1][previous_indices[valid]]
            )
        self.active_points = np.concatenate((points[chosen], triangulated[valid]))[
            -self.config.landmark_budget :
        ]
        self.active_pixels = np.concatenate((pixels[chosen], current_pixels[valid]))[
            -self.config.landmark_budget :
        ]
        self.rotation, self.translation = rotation, translation.ravel()
        self.previous_features = features
        self.previous_gray = gray
        self.previous_rotation = rotation.copy()
        self.previous_translation = translation.ravel().copy()
        self.last_tracked_frame = self.frame_index
        return self._result("tracked", True, len(chosen))

    def _initialize_from_anchor(self, features: tuple[np.ndarray, Any]) -> int:
        # Optical flow alone can follow a moving rain streak across unrelated
        # textures. Independently re-detected descriptors must confirm anchor
        # identity at the accumulated baseline before a landmark is created.
        first_ids, second_ids = _matched_indices(
            self.anchor_features, features, self.config.features, self.cv2
        )
        confirmed = {
            int(first): int(second)
            for first, second in zip(first_ids, second_ids, strict=True)
        }
        eligible = np.asarray(
            [
                int(identity) in confirmed
                and np.linalg.norm(pixel - features[0][confirmed[int(identity)]]) < 2.0
                for identity, pixel in zip(
                    self.anchor_indices, self.anchor_pixels, strict=True
                )
            ],
            dtype=bool,
        )
        anchor_indices = self.anchor_indices[eligible]
        first = self.anchor_features[0][anchor_indices]
        second = self.anchor_pixels[eligible]
        if (
            len(first) < self.config.features.minimum_matches
            or self.bootstrap_observations < self.config.minimum_bootstrap_observations
        ):
            return 0
        self.cv2.setRNGSeed(self.frame_index)
        if self.motion_started:
            essential, mask = self.cv2.findEssentialMat(
                first,
                second,
                self.camera,
                method=self.cv2.RANSAC,
                prob=0.999,
                threshold=self.config.features.ransac_pixels,
            )
            if essential is not None and essential.shape == (3, 3):
                count, relative_rotation, translation, positive = self.cv2.recoverPose(
                    essential, first, second, self.camera, mask=mask
                )
                if count >= self.config.minimum_map_points:
                    rotation = relative_rotation
                    points, valid = triangulate_landmarks(
                        first,
                        second,
                        np.eye(3),
                        np.zeros(3),
                        rotation,
                        translation.ravel(),
                        self.camera,
                        config=self.config,
                        cv2=self.cv2,
                    )
                    valid &= positive.ravel() > 0
                    height, width = self.frame_shape[:2]
                    if (
                        int(valid.sum()) >= self.config.minimum_map_points
                        and _cells(second[valid], height, width)
                        >= self.config.features.minimum_cells
                    ):
                        self._append_landmarks(
                            points[valid],
                            self.anchor_features[1][anchor_indices[valid]],
                        )
                        self.active_points, self.active_pixels = (
                            points[valid],
                            second[valid].copy(),
                        )
                        self.rotation, self.translation = rotation, translation.ravel()
                        self.initialized = True
                        return int(valid.sum())
            return 0
        return 0

    def _result(self, status: str, tracked: bool, inliers: int) -> VisualPoseEvidence:
        center = -self.rotation.T @ self.translation
        return VisualPoseEvidence(
            frame_index=self.frame_index,
            status=status,
            initialized=self.initialized,
            tracked=tracked,
            inliers=inliers,
            map_points=len(self.points),
            origin_frame_index=self.origin_frame_index,
            camera_center=tuple(float(value) for value in center) if tracked else None,
        )
