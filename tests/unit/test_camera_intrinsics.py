import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.camera_intrinsics import (
    CameraResizeSpec,
    resize_intrinsics,
)


def test_wide_native_render_to_square_observation_has_unequal_focal_axes():
    spec = CameraResizeSpec(render_size=(640, 360), observation_size=(224, 224))
    camera = spec.camera(90.0)
    assert camera[0, 0] == pytest.approx(90)
    assert camera[1, 1] == pytest.approx(160)
    np.testing.assert_allclose(camera[:2, 2], (111.5, 111.5), atol=1e-12)
    assert camera[1, 1] / camera[0, 0] == pytest.approx(16 / 9)


def test_aspect_preserving_resize_keeps_equal_native_focal_axes():
    camera = CameraResizeSpec(
        render_size=(640, 360), observation_size=(320, 180)
    ).camera(180)
    np.testing.assert_allclose(np.diag(camera), (180, 180, 1))
    assert camera[0, 2] == 159.5 and camera[1, 2] == 89.5


@pytest.mark.parametrize("target", [(224, 224), (448, 448), (320, 180)])
def test_edge_and_grid_coordinates_produce_identical_rays(target):
    grid = CameraResizeSpec(render_size=(1280, 720), observation_size=target).camera(
        180
    )
    edge = CameraResizeSpec(
        render_size=(1280, 720),
        observation_size=target,
        principal_point="pixel_edge_center",
    ).camera(180)
    np.testing.assert_allclose(edge[:2, 2], np.array(target) / 2)
    points = np.array(((60.0, 70.0, 1), (100.0, 120.0, 1)))
    shifted = points.copy()
    shifted[:, :2] += 0.5
    np.testing.assert_allclose(
        points @ np.linalg.inv(grid).T, shifted @ np.linalg.inv(edge).T, atol=1e-12
    )


def test_edge_origin_full_frame_resize_is_pure_scale():
    camera = np.array(((260.0, 0, 320), (0, 260.0, 180), (0, 0, 1)))
    resized = resize_intrinsics(
        camera, (640, 360), (448, 448), principal_point="pixel_edge_center"
    )
    np.testing.assert_allclose(resized, np.diag((448 / 640, 448 / 360, 1)) @ camera)


def test_raw_resolution_doubling_preserves_normalized_image_rays():
    small = CameraResizeSpec(
        render_size=(640, 360), observation_size=(224, 224)
    ).camera(90)
    large = CameraResizeSpec(
        render_size=(1280, 720), observation_size=(448, 448)
    ).camera(180)
    pixels = np.array(((80.0, 90.0, 1), (140.0, 160.0, 1)))
    resized_pixels = pixels.copy()
    resized_pixels[:, :2] = (pixels[:, :2] + 0.5) * 2 - 0.5
    np.testing.assert_allclose(
        pixels @ np.linalg.inv(small).T,
        resized_pixels @ np.linalg.inv(large).T,
        atol=1e-12,
    )


def test_resize_projects_exactly_as_pixel_center_transform():
    original = np.array(((260.0, 0, 319.5), (0, 260.0, 179.5), (0, 0, 1)))
    updated = resize_intrinsics(original, (640, 360), (224, 224))
    xyz = np.array(((1.0, 0.5, 3), (-1, 0.2, 4)))
    native = xyz @ original.T
    native = native[:, :2] / native[:, 2:]
    predicted = xyz @ updated.T
    predicted = predicted[:, :2] / predicted[:, 2:]
    expected = (native + 0.5) * (np.array((224, 224)) / (640, 360)) - 0.5
    np.testing.assert_allclose(predicted, expected, atol=1e-12)


@pytest.mark.parametrize(
    "changes",
    [
        {"render_size": (0, 360)},
        {"render_size": (640.5, 360)},
        {"transform": "crop"},
        {"xpos": 1},
        {"fov": 70},
    ],
)
def test_public_resize_schema_rejects_unknown_and_privileged_geometry(changes):
    with pytest.raises(ValidationError):
        CameraResizeSpec.model_validate(
            {"render_size": (640, 360), "observation_size": (224, 224)} | changes
        )


@pytest.mark.parametrize("focal", [0, -1, float("nan"), float("inf")])
def test_invalid_horizontal_focal_cannot_construct_an_executable_camera(focal):
    with pytest.raises(ValidationError):
        CameraResizeSpec(render_size=(640, 360), observation_size=(224, 224)).camera(
            focal
        )


def test_invalid_source_camera_is_rejected_without_mutation():
    for camera in (
        np.eye(2),
        np.full((3, 3), np.nan),
        np.zeros((3, 3)),
        np.array(((90, 1, 112), (0, 160, 112), (0, 0, 1))),
    ):
        with pytest.raises(ValueError):
            resize_intrinsics(camera, (640, 360), (224, 224))
