"""Past-only fitting for experimental landmark protection, not accepted navigation."""

from __future__ import annotations

import numpy as np

from .camera_intrinsics import CameraResizeSpec
from .keyframe_motion import KeyframeAttempt
from .multiview_motion import MultiViewMotionConfig, estimate_multiview_motion
from .temporal_tracks import TemporalTracks
from .visual_geometry import (
    VisualGeometryConfig,
    _calibrate_focal,
    _features,
    _frame,
    _opencv,
)


def calibrate_reference_camera(images, render_size):
    """Use exactly the twelve completed public yaw views, before probe acquisition."""
    if len(images) != 12:
        raise ValueError("online fitting needs twelve public startup reference views")
    images = [_frame(image) for image in images]
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("public reference dimensions must agree")
    height, width = images[0].shape[:2]
    cv2, geometry = _opencv(), VisualGeometryConfig()
    features = [
        (points + (0.5, 0.5), descriptors)
        for points, descriptors in (_features(image, geometry, cv2) for image in images)
    ]
    focal, count, mad = _calibrate_focal(features, width, geometry, cv2)
    if focal is None:
        raise ValueError("public RGB focal calibration rejected")
    resize = CameraResizeSpec(render_size=render_size, observation_size=(width, height))
    return {
        "camera_resize": resize.model_dump(mode="json"),
        "camera_intrinsics": resize.camera(focal).tolist(),
        "focal_pixels": focal,
        "focal_sample_count": count,
        "focal_relative_mad": mad,
        "opencv_version": cv2.__version__,
    }


def fit_online_landmarks(tracks, retained_ids, camera, *, seed=0):
    """One current prefix; no future images, reseeding or accepted-map replacement."""
    tracks = TemporalTracks.model_validate(tracks.model_dump())
    if retained_ids is not None:
        keep = [
            i for i, identity in enumerate(tracks.track_ids) if identity in retained_ids
        ]
        tracks = TemporalTracks.model_validate(
            tracks.model_dump()
            | {
                "track_ids": tuple(tracks.track_ids[i] for i in keep),
                "pixels": tuple(
                    tuple(frame[i] for i in keep) for frame in tracks.pixels
                ),
                "minimum_patch_correlations": tuple(
                    tracks.minimum_patch_correlations[i] for i in keep
                ),
                "chain_cycle_errors": tuple(tracks.chain_cycle_errors[i] for i in keep),
            }
        )
    motion = estimate_multiview_motion(
        tracks, np.asarray(camera), config=MultiViewMotionConfig(), seed=seed
    )
    return KeyframeAttempt(
        end_offset=tracks.frame_count - 1, tracks=tracks, motion=motion
    )
