import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_stability import (
    MotionSensitivityTrial,
    MotionStabilityConfig,
    StableMotionEvidence,
    assess_motion_stability,
)
from minecraft_scientist.exploration.two_view_motion import (
    TwoViewMotionConfig,
    TwoViewMotionEvidence,
)

CAMERA = np.array(((90.0, 0, 112), (0, 90.0, 112), (0, 0, 1)))


def _matches(translation=(-0.7, 0.05, -0.15), planar=False):
    rng = np.random.default_rng(101)
    xyz = np.column_stack(
        (
            rng.uniform(-2.5, 2.5, 240),
            rng.uniform(-2.5, 2.5, 240),
            np.full(240, 6.0) if planar else rng.uniform(3.5, 9, 240),
        )
    )
    rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0]
    first = xyz @ CAMERA.T
    second = (xyz @ rotation.T + translation) @ CAMERA.T
    first = first[:, :2] / first[:, 2:]
    second = second[:, :2] / second[:, 2:]
    return DenseCorrespondences(
        points0=tuple(map(tuple, first)),
        points1=tuple(map(tuple, second)),
        confidence=(0.95,) * len(xyz),
        image_shape0=(224, 224),
        image_shape1=(224, 224),
        raw_match_count=len(xyz),
    )


def test_nonplanar_motion_survives_small_pixel_and_focal_uncertainty():
    matches = _matches()
    original_camera = CAMERA.copy()
    # Precisely projected points need a correspondingly tight fitting threshold.
    # A loose pixel RANSAC threshold admits inaccurate minimal fits even here;
    # passing that loose fit is not an invariant of the sensitivity checker.
    result = assess_motion_stability(
        matches,
        CAMERA,
        motion_config=TwoViewMotionConfig(reference_ransac_pixels=0.3),
        seed=11,
    )
    assert result.supported and result.supported_fraction >= 0.8
    assert result.direction_deviation_95_degrees < 20
    assert result.unit_translation == result.candidate.unit_translation
    np.testing.assert_array_equal(CAMERA, original_camera)


def test_default_fit_is_stable_for_exact_nonplanar_correspondences():
    result = assess_motion_stability(
        _matches(),
        CAMERA,
        config=MotionStabilityConfig(reference_pixel_noise=0, relative_focal_noise=0),
        seed=11,
    )
    assert result.supported and result.supported_fraction == 1
    # Minimal-set RANSAC need not select the exact same essential solution even
    # without noise. Assert the declared stability bound, not exact pose equality.
    assert result.rotation_deviation_95_degrees <= 3
    direction = np.array((-0.7, 0.05, -0.15))
    direction /= np.linalg.norm(direction)
    assert np.array(result.unit_translation) @ direction > np.cos(np.radians(20))


def test_stability_replays_exactly_for_same_seed():
    matches = _matches()
    config = MotionStabilityConfig(trials=4)
    assert assess_motion_stability(
        matches, CAMERA, config=config, seed=7
    ) == assess_motion_stability(matches, CAMERA, config=config, seed=7)


def test_unstable_planar_motion_is_not_promoted():
    result = assess_motion_stability(_matches(planar=True), CAMERA)
    assert not result.supported
    assert result.unit_translation is None and result.relative_rotation is None


def test_stationary_or_rotation_control_has_no_executable_motion():
    result = assess_motion_stability(_matches(translation=(0, 0, 0)), CAMERA)
    assert not result.supported and result.trials == ()
    assert result.unit_translation is None


def test_no_supported_trials_is_a_valid_rejection(monkeypatch):
    import minecraft_scientist.exploration.motion_stability as module

    original = module.estimate_two_view_motion
    count = 0

    def fit(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return original(*args, **kwargs)
        return TwoViewMotionEvidence(
            supported=False, reason="injected insufficient support", matches=0
        )

    monkeypatch.setattr(module, "estimate_two_view_motion", fit)
    result = assess_motion_stability(
        _matches(), CAMERA, config=MotionStabilityConfig(trials=4)
    )
    assert not result.supported and result.supported_fraction == 0
    assert len(result.trials) == 4 and result.direction_deviation_95_degrees is None


def test_stability_schemas_reject_nonfinite_privileged_and_inconsistent_records():
    for changed in (
        {"xpos": 1},
        {"trials": 0},
        {"reference_pixel_noise": float("nan")},
        {"relative_focal_noise": float("inf")},
    ):
        with pytest.raises(ValidationError):
            MotionStabilityConfig(**changed)
    with pytest.raises(ValidationError):
        MotionSensitivityTrial(
            supported=False, reason="no fit", direction_deviation_degrees=0
        )
    with pytest.raises(ValidationError):
        MotionSensitivityTrial(supported=True, reason="fit")
    candidate = TwoViewMotionEvidence(supported=False, reason="no fit", matches=0)
    with pytest.raises(ValidationError):
        StableMotionEvidence(
            supported=False, reason="no fit", candidate=candidate, supported_fraction=1
        )
    with pytest.raises(ValidationError):
        StableMotionEvidence(
            supported=False,
            reason="no fit",
            candidate=candidate,
            unit_translation=(1, 0, 0),
        )


def test_malformed_supported_candidate_is_not_executable(monkeypatch):
    import minecraft_scientist.exploration.motion_stability as module

    malformed = TwoViewMotionEvidence(
        supported=True,
        reason="bad injected backend",
        matches=100,
        relative_rotation=((1, 0, 0),),
        unit_translation=(0, 0, 0),
    )
    monkeypatch.setattr(
        module, "estimate_two_view_motion", lambda *args, **kwargs: malformed
    )
    with pytest.raises(ValueError, match="proper rotation"):
        assess_motion_stability(_matches(), CAMERA)
