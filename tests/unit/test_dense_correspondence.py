import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.dense_correspondence import (
    DenseCorrespondenceConfig,
    DenseCorrespondences,
    DenseRGBMatcher,
    LocalLoFTRBackend,
    filter_dense_matches,
    prepare_dense_view,
)


def test_dense_preprocessing_excludes_hud_before_equalization_and_resizing():
    rng = np.random.default_rng(50)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    altered = rng.integers(0, 256, image.shape, dtype=np.uint8)
    altered[44:170, 40:183] = image[44:170, 40:183]
    first, second = (
        prepare_dense_view(image, DenseCorrespondenceConfig()),
        prepare_dense_view(altered, DenseCorrespondenceConfig()),
    )
    np.testing.assert_array_equal(first.gray, second.gray)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert (
        first.gray.dtype == np.float32
        and 0 <= first.gray.min() <= first.gray.max() <= 1
    )
    assert all(dimension % 8 == 0 for dimension in first.gray.shape)
    assert first.original_shape == (224, 224)


def test_dense_coordinate_restoration_accounts_for_crop_and_pixel_centers():
    view = prepare_dense_view(
        np.zeros((224, 224, 3)), DenseCorrespondenceConfig(working_dimension=128)
    )
    scale = np.array(view.resized_size) / np.array(view.crop_size)
    points = np.array(((60, 60), (100, 100)), dtype=float)
    resized = (points - view.offset + 0.5) * scale - 0.5
    np.testing.assert_allclose(view.restore_pixels(resized), points, atol=1e-12)


def test_padding_and_nonfinite_or_low_confidence_matches_are_not_observations():
    config = DenseCorrespondenceConfig(working_dimension=128)
    view = prepare_dense_view(np.zeros((224, 224, 3)), config)
    points = np.array(
        (
            (20, 20),
            (30, 30),
            (view.resized_size[0] + 1, 40),
            (float("nan"), 20),
            (40, 40),
        ),
        dtype=float,
    )
    output = filter_dense_matches(
        {
            "keypoints0": points,
            "keypoints1": points,
            "confidence": np.array((0.9, 0.1, 0.9, 0.9, 1.5)),
        },
        view,
        view,
        config,
    )
    assert len(output.points0) == 1 and output.raw_match_count == 5
    assert output.confidence == (0.9,)


def test_dense_matching_deduplicates_both_views_and_is_order_independent():
    config = DenseCorrespondenceConfig(working_dimension=128)
    view = prepare_dense_view(np.zeros((224, 224, 3)), config)
    first = np.array(((20, 20), (20, 20), (40, 40), (50, 50)))
    second = np.array(((20, 20), (30, 30), (20, 20), (50, 50)))
    confidence = np.array((0.95, 0.9, 0.8, 0.7))
    first_result = filter_dense_matches(
        {"keypoints0": first, "keypoints1": second, "confidence": confidence},
        view,
        view,
        config,
    )
    reverse = filter_dense_matches(
        {
            "keypoints0": first[::-1],
            "keypoints1": second[::-1],
            "confidence": confidence[::-1],
        },
        view,
        view,
        config,
    )
    assert first_result == reverse and len(first_result.points0) == 2


def test_dense_correspondence_budget_is_enforced():
    config = DenseCorrespondenceConfig(working_dimension=320, maximum_matches=16)
    view = prepare_dense_view(np.zeros((224, 224, 3)), config)
    points = np.array(
        [(x, y) for x in range(20, 300, 20) for y in range(20, 250, 20)], dtype=float
    )
    result = filter_dense_matches(
        {
            "keypoints0": points,
            "keypoints1": points,
            "confidence": np.linspace(0.6, 0.9, len(points)),
        },
        view,
        view,
        config,
    )
    assert len(result.points0) == 16


def test_backend_receives_only_cropped_public_views_and_does_not_mutate_inputs():
    def backend(left, right):
        assert (
            left.gray.shape == right.gray.shape and left.mask.shape == left.gray.shape
        )
        return {
            "keypoints0": np.array(((20, 20),)),
            "keypoints1": np.array(((25, 20),)),
            "confidence": np.array((0.9,)),
        }

    image = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    original = image.copy()
    result = DenseRGBMatcher(backend).match(image, image)
    assert len(result.points0) == 1
    np.testing.assert_array_equal(image, original)


def test_dense_schema_rejects_mismatched_or_privileged_fields():
    with pytest.raises(ValidationError):
        DenseCorrespondenceConfig(xpos=1)
    values = {
        "points0": ((50, 50),),
        "points1": ((55, 50),),
        "confidence": (0.9,),
        "image_shape0": (224, 224),
        "image_shape1": (224, 224),
        "raw_match_count": 1,
    }
    with pytest.raises(ValidationError):
        DenseCorrespondences(**values, camera_position=(0, 0, 0))
    for changed in (
        {"points1": ()},
        {"confidence": (float("nan"),)},
        {"points0": ((300, 1),)},
        {"raw_match_count": 0},
    ):
        with pytest.raises(ValidationError):
            DenseCorrespondences.model_validate(values | changed)


def test_checkpoint_identity_is_checked_before_importing_heavy_runtime(tmp_path):
    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        LocalLoFTRBackend(path, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        LocalLoFTRBackend(path, sha256="invalid")


def test_dense_output_shape_errors_are_explicit():
    config = DenseCorrespondenceConfig()
    view = prepare_dense_view(np.zeros((224, 224, 3)), config)
    with pytest.raises(ValueError, match="dimensions differ"):
        filter_dense_matches(
            {
                "keypoints0": np.zeros((3, 2)),
                "keypoints1": np.zeros((2, 2)),
                "confidence": np.zeros(3),
            },
            view,
            view,
            config,
        )
