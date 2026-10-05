import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="optional safe-seed geometry")

from test_joint_multiview_map import CAMERA, fixture_tracks, prefix
from test_stable_keyframe_map import public_scene as public_scene  # noqa: PLC0414

from minecraft_scientist.exploration import stable_keyframe_map as stable
from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.shared_map_endpoint import (
    SharedEndpointPolicy,
    _visible_flow,
)
from minecraft_scientist.exploration.temporal_tracks import diagnose_frame_chain


def acquire(first, execute, **kwargs):
    return stable.acquire_stable_keyframe_map(
        first,
        execute,
        CAMERA,
        anchor_frame=48,
        seed=13,
        micro_controls=True,
        responsive_controls=True,
        recovery_controls=True,
        safe_seed_controls=True,
        **kwargs,
    )


@pytest.mark.parametrize("size", [224, 448])
def test_detector_matches_validated_research_and_all_first_seeds_are_trackable(size):
    gray = cv2.GaussianBlur(
        np.random.default_rng(23).integers(0, 256, (size, size), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    first = np.repeat(gray[:, :, None], 3, axis=2)
    original = first.copy()
    seeds = safe_seeds(first)
    spec = importlib.util.spec_from_file_location(
        "reference_safe_seeds",
        Path(__file__).resolve().parents[2]
        / "research/settled_keyframes/safe_seed_detector.py",
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert np.array_equal(seeds, reference.safe_seeds(first))
    diagnosis = diagnose_frame_chain([first, first.copy()], seed_pixels=seeds)
    assert 32 < len(seeds) <= 800
    assert (
        diagnosis.tracks.seed_count
        == diagnosis.tracks.surviving_counts[0]
        == len(seeds)
    )
    assert not any(item.reason == "seed_roi" for item in diagnosis.rejections)
    assert np.array_equal(first, original)
    flow = _visible_flow(first, first.copy(), SharedEndpointPolicy(), seed_pixels=seeds)
    assert flow["stable"] is True and flow["tracks"]["seed_count"] == len(seeds)


def test_future_texture_loss_cannot_resurrect_or_relabel_first_seed_ids():
    first = np.random.default_rng(24).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    seeds = safe_seeds(first)
    diagnosis = diagnose_frame_chain([first, np.zeros_like(first)], seed_pixels=seeds)
    assert not diagnosis.tracks.track_ids
    rejected = [
        identity for item in diagnosis.rejections for identity in item.track_ids
    ]
    assert len(rejected) == len(set(rejected)) == len(seeds)
    assert set(rejected) == set(range(len(seeds)))
    assert np.array_equal(seeds, safe_seeds(first))


def test_untextured_actual_pixels_use_exact_recovery_budget_without_map():
    first = np.zeros((224, 224, 3), dtype=np.uint8)
    from minecraft_scientist.exploration.settled_map_control import PublicControlStep

    calls = []

    def execute(kind, handle):
        calls.append((kind, handle))
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    result = acquire(first, execute)
    assert result["terminal"] == "initial_wait_exhausted" and len(calls) == 48
    assert (
        len(result["failed_wait_windows"]) == 3
        and len(result["recovery_wait_windows"]) == 2
    )
    assert not result["epochs"] and not result["samples"] and not result["attempts"]
    assert all(
        item["tracks"]["seed_count"] == 0 and item["seed_pixels"] == []
        for item in result["flow_prefixes"]
    )


@pytest.fixture
def seeded_scene(public_scene, monkeypatch):
    """Mock perception isolates protocol; real synthetic 3-D fitting/validation."""
    first, execute, calls = public_scene
    tracks, _, _ = fixture_tracks(frames=8, noise=0.02)
    seeds = np.asarray(tracks.pixels[0], dtype=np.float32)
    detector_inputs, tracked_inputs = [], []
    flow = stable._visible_flow

    def detect(image):
        detector_inputs.append(int(image[0, 0, 0]))
        return seeds.copy()

    def seeded_flow(first, last, settings, *, seed_pixels):
        assert np.array_equal(seed_pixels, seeds)
        return flow(first, last, settings) | {
            "tracks": {"track_ids": list(range(150)), "seed_count": 150}
        }

    def tracked(frames, *, config, seed_pixels):
        tracked_inputs.append((len(calls), seed_pixels, seed_pixels.copy()))
        assert not seed_pixels.flags.writeable
        return prefix(tracks, len(frames))

    monkeypatch.setattr(stable, "safe_seeds", detect)
    monkeypatch.setattr(stable, "_visible_flow", seeded_flow)
    monkeypatch.setattr(stable, "track_frame_chain", tracked)
    return first, execute, calls, seeds, detector_inputs, tracked_inputs


def test_first_epoch_seeds_freeze_before_fit_and_remain_identical_for_unseen_future(
    seeded_scene,
):
    first, execute, calls, seeds, detector_inputs, tracked_inputs = seeded_scene
    result = acquire(first, execute)
    assert result["terminal"] == "validated" and len(calls) == 63
    assert (
        result["seed_detection_registration"] == "first-image-only-safe-support-mask-v1"
    )
    assert result["policy"] == stable.StableSafeSeedKeyframePolicy().model_dump(
        mode="json"
    )
    assert [entry[0] for entry in tracked_inputs] == [53, 63]
    assert tracked_inputs[0][1] is tracked_inputs[1][1]
    assert all(np.array_equal(entry[2], seeds) for entry in tracked_inputs)
    epoch_detector = result["epochs"][0]["detector"]
    assert epoch_detector["seed_image_frame"] == 56 and np.array_equal(
        epoch_detector["seed_pixels"], seeds
    )
    assert (
        epoch_detector["seed_image_sha256"]
        == hashlib.sha256(np.full_like(first, 8).tobytes()).hexdigest()
    )
    assert result["frozen"]["detector"] == epoch_detector
    assert result["frozen"]["index"]["namespace"] == "safe-sparse-keyframes-56"
    assert result["future"]["index"]["namespace"] == "safe-sparse-keyframes-56"
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert result["motion"]["frame_ordinals"] == [101, 111]
    assert not result["navigation_accepted"]
    assert (
        63 not in detector_inputs
    )  # final future image cannot determine first-map seeds


@pytest.mark.parametrize("stage", ["flow", "past", "future"])
def test_changed_seed_denominator_is_a_geometry_failure_not_success(
    seeded_scene, monkeypatch, stage
):
    first, execute, _, _, _, _ = seeded_scene
    if stage == "flow":
        flow = stable._visible_flow

        def bad_flow(*args, **kwargs):
            measured = copy.deepcopy(flow(*args, **kwargs))
            measured["tracks"]["seed_count"] -= 1
            return measured

        monkeypatch.setattr(stable, "_visible_flow", bad_flow)
    else:
        tracked = stable.track_frame_chain

        def bad_tracks(frames, **kwargs):
            result = tracked(frames, **kwargs)
            return (
                result.model_copy(update={"seed_count": 151})
                if len(frames) == (6 if stage == "past" else 7)
                else result
            )

        monkeypatch.setattr(stable, "track_frame_chain", bad_tracks)
    result = acquire(first, execute)
    assert result["terminal"] == "geometry_failed" and result["motion"] is None
    assert (
        result["failure"]["stage"]
        == {"flow": "stability", "past": "past_fitting", "future": "future_validation"}[
            stage
        ]
    )
    if stage == "future":
        assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert not result["navigation_accepted"]


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_safe_registration_is_explicit_boolean_before_controls(public_scene, flag):
    first, execute, calls = public_scene
    with pytest.raises(ValueError, match="explicit boolean"):
        stable.acquire_stable_keyframe_map(
            first, execute, CAMERA, anchor_frame=48, safe_seed_controls=flag
        )
    assert calls == []


def test_safe_policy_cannot_disguise_recovery_or_change_tracking_gates(public_scene):
    first, execute, calls = public_scene
    with pytest.raises(ValueError, match="recovering keyframe"):
        stable.acquire_stable_keyframe_map(
            first, execute, CAMERA, anchor_frame=48, safe_seed_controls=True
        )
    with pytest.raises(ValidationError):
        stable.acquire_stable_keyframe_map(
            first,
            execute,
            CAMERA,
            anchor_frame=48,
            micro_controls=True,
            responsive_controls=True,
            recovery_controls=True,
            policy=stable.StableSafeSeedKeyframePolicy(),
        )
    assert calls == []
    with pytest.raises(ValidationError):
        stable.StableSafeSeedKeyframePolicy(seed_detection="future-supported-ids")
    with pytest.raises(ValidationError, match="registered geometry"):
        stable.StableSafeSeedKeyframePolicy(
            temporal={"maximum_frames": 13, "minimum_patch_correlation": 0.5}
        )
