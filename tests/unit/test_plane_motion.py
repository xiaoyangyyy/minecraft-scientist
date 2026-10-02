import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    estimate_multiview_motion,
)
from minecraft_scientist.exploration.plane_motion import (
    PlaneMotionConfig,
    PlaneMotionEvidence,
    analyze_plane_motion,
    estimate_plane_endpoint,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = CameraResizeSpec(render_size=(640, 360), observation_size=(224, 224)).camera(
    90
)
CONFIG = MultiViewMotionConfig()
TRANSLATION = np.array((-0.7, 0.03, -0.12))


def _tracks(
    *, planar=False, rotation_only=False, stationary=False, scale=1, wide_plane=False
):
    rng = np.random.default_rng(213)
    count = 400
    depth = np.full(count, 7.0)
    if not planar:
        depth[280:] = rng.uniform(3, 5, count - 280)
    xyz = np.column_stack(
        (rng.uniform(-1.8, 1.8, count), rng.uniform(-1.4, 1.4, count), depth)
    )
    if wide_plane:
        xyz[:280, 0] = rng.uniform(-3.5, 3.5, 280)
        xyz[:280, 1] = rng.uniform(-2.5, 2.5, 280)
    pixels = []
    camera = CAMERA.copy()
    camera[:2, :2] *= scale
    camera[:2, 2] = (camera[:2, 2] + 0.5) * scale - 0.5
    for alpha in np.linspace(0, 1, 5):
        rotation = (
            np.eye(3)
            if stationary
            else cv2.Rodrigues(np.array((0.01, 0.03, -0.01)) * alpha)[0]
        )
        translation = (
            np.zeros(3) if stationary or rotation_only else TRANSLATION * alpha
        )
        projected = (xyz @ rotation.T + translation) @ camera.T
        pixels.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    return TemporalTracks(
        image_shape=(224 * scale, 224 * scale),
        frame_count=5,
        seed_count=count,
        surviving_counts=(count,) * 5,
        track_ids=tuple(range(count)),
        pixels=tuple(pixels),
        minimum_patch_correlations=(0.95,) * count,
        chain_cycle_errors=(0.0,) * count,
    ), camera


def test_independent_nonplanar_rays_disambiguate_plane_motion_and_replay():
    tracks, camera = _tracks()
    matches = tracks.endpoint_correspondences()
    result = analyze_plane_motion(matches, camera, config=CONFIG.two_view)
    assert result.endpoint.supported, result.endpoint.reason
    assert (
        np.array(result.endpoint.unit_translation)
        @ (TRANSLATION / np.linalg.norm(TRANSLATION))
        > 0.999
    )
    assert sum(branch.family == "homography" for branch in result.branches) == 4
    assert result.selection_margin >= 0.25 and result.shadow_margin >= 0.25
    assert (
        len(result.selection_off_plane_ids) >= 8
        and len(result.shadow_off_plane_ids) >= 8
    )
    assert set(result.training_ids).isdisjoint(result.selection_ids)
    assert set(result.training_ids).isdisjoint(result.shadow_ids)
    assert set(result.selection_ids).isdisjoint(result.shadow_ids)
    assert result == analyze_plane_motion(matches, camera, config=CONFIG.two_view)


@pytest.mark.parametrize(
    "change", [{"planar": True}, {"rotation_only": True}, {"stationary": True}]
)
def test_planar_rotation_and_stationary_pairs_never_choose_arbitrary_branch(change):
    tracks, camera = _tracks(**change)
    result = analyze_plane_motion(
        tracks.endpoint_correspondences(), camera, config=CONFIG.two_view
    )
    assert not result.endpoint.supported and result.endpoint.unit_translation is None
    if change.get("planar"):
        assert len(result.branches) == 4
        assert "off-plane" in result.endpoint.reason


def test_shadow_disagreement_vetoes_fixed_selection_instead_of_reselecting():
    tracks, camera = _tracks()
    matches = tracks.endpoint_correspondences()
    accepted = analyze_plane_motion(matches, camera, config=CONFIG.two_view)
    values = matches.model_dump()
    second = np.asarray(values["points1"]).copy()
    # Fitting and selection are unchanged; independent shadow now contradicts
    # the chosen branch. No pose or private truth is supplied to the estimator.
    second[list(accepted.shadow_ids)] = np.asarray(matches.points0)[
        list(accepted.shadow_ids)
    ]
    values["points1"] = tuple(map(tuple, second))
    rejected = analyze_plane_motion(
        DenseCorrespondences.model_validate(values), camera, config=CONFIG.two_view
    )
    assert not rejected.endpoint.supported
    assert rejected.selected_branch == accepted.selected_branch
    assert rejected.selection_margin == accepted.selection_margin
    assert "shadow" in rejected.endpoint.reason


def test_plane_estimator_retains_all_multiview_and_sensitivity_requirements():
    tracks, camera = _tracks(wide_plane=True)
    result = estimate_multiview_motion(
        tracks, camera, config=CONFIG, endpoint_estimator=estimate_plane_endpoint
    )
    assert result.supported, result.reason
    assert result.sensitivity.supported and len(result.sensitivity.trials) == 16
    assert all(check.supported for check in result.frame_checks)
    assert set(result.training_track_ids).isdisjoint(result.validation_track_ids)


def test_narrow_plane_cannot_bypass_pixel_perturbation_checks():
    tracks, camera = _tracks()
    result = estimate_multiview_motion(
        tracks, camera, config=CONFIG, endpoint_estimator=estimate_plane_endpoint
    )
    assert result.candidate.supported
    assert all(check.supported for check in result.frame_checks)
    assert len(result.sensitivity.trials) == 16
    # This narrow-plane case is borderline under OpenCV 4.8 versus 4.13.
    # Numerical agreement is not required; respecting the same measured gates
    # is. The separate wide-plane positive must pass on both runtimes.
    stability = CONFIG.stability
    observed = (
        result.sensitivity.supported_fraction >= stability.minimum_supported_fraction
        and result.sensitivity.direction_deviation_95_degrees is not None
        and result.sensitivity.direction_deviation_95_degrees
        <= stability.maximum_direction_deviation
        and result.sensitivity.rotation_deviation_95_degrees is not None
        and result.sensitivity.rotation_deviation_95_degrees
        <= stability.maximum_rotation_deviation
    )
    assert result.supported == result.sensitivity.supported == observed
    if observed:
        assert (
            np.array(result.unit_translation)
            @ (TRANSLATION / np.linalg.norm(TRANSLATION))
            > 0.99
        )
    else:
        assert result.unit_translation is None and result.relative_rotation is None


def test_resized_intrinsics_and_thresholds_preserve_direction():
    tracks, camera = _tracks(scale=2)
    result = analyze_plane_motion(
        tracks.endpoint_correspondences(), camera, config=CONFIG.two_view
    )
    assert result.endpoint.supported
    assert (
        np.array(result.endpoint.unit_translation)
        @ (TRANSLATION / np.linalg.norm(TRANSLATION))
        > 0.999
    )


def test_unrelated_matches_and_insufficient_partitions_reject():
    tracks, camera = _tracks()
    values = tracks.endpoint_correspondences().model_dump()
    values["points1"] = tuple(
        map(tuple, np.random.default_rng(9).uniform(40, 180, (400, 2)))
    )
    result = analyze_plane_motion(
        DenseCorrespondences.model_validate(values), camera, config=CONFIG.two_view
    )
    assert not result.endpoint.supported
    for key in ("points0", "points1", "confidence"):
        values[key] = values[key][:35]
    values["raw_match_count"] = 35
    result = analyze_plane_motion(
        DenseCorrespondences.model_validate(values), camera, config=CONFIG.two_view
    )
    assert "independent" in result.endpoint.reason


def test_partition_and_branch_evidence_cannot_be_forged():
    tracks, camera = _tracks()
    result = analyze_plane_motion(
        tracks.endpoint_correspondences(), camera, config=CONFIG.two_view
    )
    for change in (
        {"shadow_ids": result.selection_ids},
        {"shadow_off_plane_ids": result.training_ids[:8]},
        {"selected_branch": 999},
        {"branches": result.branches + result.branches[:1]},
    ):
        with pytest.raises(ValueError):
            PlaneMotionEvidence.model_validate(result.model_dump() | change)


def test_invalid_inputs_reject_before_cv_and_do_not_mutate(monkeypatch):
    tracks, camera = _tracks()
    matches = tracks.endpoint_correspondences()
    original = camera.copy()
    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *args, **kwargs: pytest.fail("invalid input reached fit"),
    )
    for seed in (-1, True, 2**31):
        with pytest.raises(ValueError, match="seed"):
            analyze_plane_motion(matches, camera, config=CONFIG.two_view, seed=seed)
    with pytest.raises(ValueError):
        analyze_plane_motion(matches, camera * np.nan, config=CONFIG.two_view)
    with pytest.raises(ValueError, match="finite"):
        analyze_plane_motion(
            matches,
            camera,
            config=CONFIG.two_view.model_copy(
                update={"reference_ransac_pixels": np.inf}
            ),
        )
    with pytest.raises(ValueError):
        PlaneMotionConfig(minimum_loss_margin=np.nan)
    np.testing.assert_array_equal(camera, original)


def test_both_model_families_fit_only_training_rays(monkeypatch):
    from minecraft_scientist.exploration import plane_motion

    tracks, camera = _tracks()
    matches = tracks.endpoint_correspondences()
    permutation = np.random.default_rng(0).permutation(len(matches.points0))
    expected = np.asarray(matches.points0)[np.sort(permutation[200:])]
    original_h, original_e = (
        cv2.findHomography,
        plane_motion.estimate_consensus_endpoint,
    )
    seen = []

    def fit_plane(first, second, *args, **kwargs):
        np.testing.assert_array_equal(first, expected)
        seen.append("homography")
        return original_h(first, second, *args, **kwargs)

    def fit_general(training, *args, **kwargs):
        np.testing.assert_array_equal(training.points0, expected)
        seen.append("essential")
        return original_e(training, *args, **kwargs)

    monkeypatch.setattr(cv2, "findHomography", fit_plane)
    monkeypatch.setattr(plane_motion, "estimate_consensus_endpoint", fit_general)
    result = analyze_plane_motion(matches, camera, config=CONFIG.two_view)
    assert result.endpoint.supported and "essential" in seen and "homography" in seen


def test_custom_training_budget_is_not_consumed_by_holdout_fraction():
    tracks, camera = _tracks()
    result = analyze_plane_motion(
        tracks.endpoint_correspondences(),
        camera,
        config=CONFIG.two_view,
        plane_config=PlaneMotionConfig(minimum_training_points=350),
    )
    assert len(result.training_ids) == 350
    assert len(result.selection_ids) == len(result.shadow_ids) == 25


def test_reversing_homography_branch_order_does_not_change_direction(monkeypatch):
    tracks, camera = _tracks()
    matches = tracks.endpoint_correspondences()
    baseline = analyze_plane_motion(matches, camera, config=CONFIG.two_view)
    original = cv2.decomposeHomographyMat

    def reversed_branches(*args):
        count, rotations, translations, normals = original(*args)
        return count, rotations[::-1], translations[::-1], normals[::-1]

    monkeypatch.setattr(cv2, "decomposeHomographyMat", reversed_branches)
    result = analyze_plane_motion(matches, camera, config=CONFIG.two_view)
    assert result.endpoint.supported
    np.testing.assert_allclose(
        result.endpoint.unit_translation, baseline.endpoint.unit_translation, atol=1e-10
    )
