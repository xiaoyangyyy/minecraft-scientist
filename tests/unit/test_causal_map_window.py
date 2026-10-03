import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import causal_map_window as causal
from minecraft_scientist.exploration.joint_multiview_map import JointMapEvidence
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = np.array(((90.0, 0, 112), (0, 120.0, 112), (0, 0, 1)))


def tracks(frames, *, ids=tuple(range(32)), offset=0):
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=frames,
        seed_count=32,
        surviving_counts=(32,) * frames,
        track_ids=ids,
        pixels=tuple(
            tuple(
                (40.5 + identity + frame + offset, 60.5 + identity) for identity in ids
            )
            for frame in range(frames)
        ),
        minimum_patch_correlations=(1,) * len(ids),
        chain_cycle_errors=(0,) * len(ids),
    )


@pytest.fixture
def geometry(monkeypatch):
    trace = []

    def fit(past, camera, *, bootstrap_slot, seed, config):
        trace.append(("fit", bootstrap_slot))
        return JointMapEvidence(
            supported=True,
            reason="synthetic unit mock; not geometric proof",
            config=config,
            seed=seed,
            camera=tuple(map(tuple, camera)),
            bootstrap_slot=bootstrap_slot,
            fitting_frames=past.frame_count,
            input_sha256=hashlib.sha256(past.model_dump_json().encode()).hexdigest(),
            reserved_training_ids=tuple(range(24)),
            training_ids=tuple(range(24)),
            validation_ids=tuple(range(24, 32)),
            training_points=((0.0, 0.0, 5.0),) * 24,
            validation_points=((0.0, 0.0, 6.0),) * 8,
            poses=tuple(
                {
                    "frame_slot": slot,
                    "rotation": np.eye(3).tolist(),
                    "translation": [-float(slot), 0.0, 0.0],
                }
                for slot in range(past.frame_count)
            ),
            training_cost_history=(1.0,),
            frame_checks=tuple(
                {
                    "frame_slot": slot,
                    "supported": True,
                    "reason": "unit mock",
                    "training_inliers": 24,
                    "validation_tracks": 8,
                }
                for slot in range(past.frame_count)
            ),
        )

    def future(fitted, past, current, *, target_slots):
        trace.append(("future", current.frame_count))
        assert target_slots == (current.frame_count - 1,)
        rotation = np.eye(3)
        shift = rotation @ np.array(fitted.poses[-1].translation) + (0, 0, -1)
        return {
            "status": "consistent",
            "reason": "unit mock",
            "navigation_accepted": False,
            "training_ids": list(fitted.training_ids),
            "validation_ids": list(fitted.validation_ids),
            "frame_checks": [
                {
                    "frame_slot": target_slots[0],
                    "supported": True,
                    "reason": "unit mock",
                    "training_inliers": 24,
                    "validation_tracks": 8,
                }
            ],
            "poses": [
                {
                    "frame_slot": target_slots[0],
                    "rotation": rotation.tolist(),
                    "translation": shift.tolist(),
                }
            ],
        }

    monkeypatch.setattr(causal, "fit_joint_map", fit)
    monkeypatch.setattr(causal, "validate_joint_future", future)
    return trace, future


def window():
    return causal.CausalMapWindow(tracks(5), CAMERA, anchor_frame=48)


def test_horizon_uses_only_current_prefix_then_requires_explicit_refresh(geometry):
    trace, _ = geometry
    controller = window()
    assert trace == [("fit", 2), ("fit", 3), ("fit", 4)]
    assert controller.advance(tracks(6)).instruction == "collect"
    assert not any(kind == "future" for kind, _ in trace)
    ready = controller.advance(tracks(7))
    assert ready.instruction == "refresh" and ready.absolute_frame == 54
    assert [slot for kind, slot in trace if kind == "future"] == [7, 7, 7]
    assert ready.unit_translation == pytest.approx((0, 0, -1))
    assert not ready.navigation_accepted and not ready.metric_scale_known
    frozen = controller.digest()
    with pytest.raises(ValueError, match="terminal"):
        controller.advance(tracks(8))
    assert controller.digest() == frozen
    child = controller.refresh(tracks(5, offset=2), anchor_frame=50)
    assert child.snapshot()["parent_sha256"] == frozen
    assert child.snapshot()["seed"] == 2005
    assert child.snapshot()["window_index"] == 1
    assert controller.digest() == frozen


def test_later_track_loss_cannot_rewrite_an_earlier_decision(geometry):
    first, second = window(), window()
    for controller in (first, second):
        controller.advance(tracks(6))
        controller.advance(tracks(7))
    assert first.snapshot() == second.snapshot()
    # Later input belongs to a new scope and cannot be inserted into either gate.
    with pytest.raises(ValueError, match="terminal"):
        first.advance(tracks(8, ids=tuple(range(16))))
    assert first.snapshot() == second.snapshot()


@pytest.mark.parametrize("kind", ["gap", "pixels", "seed", "anchor"])
def test_invalid_prefix_leaves_every_frozen_byte_unchanged(geometry, kind):
    controller = window()
    before = controller.digest()
    extended = tracks(6)
    data = extended.model_dump()
    if kind == "gap":
        extended = tracks(7)
    elif kind == "pixels":
        pixels = np.array(data["pixels"])
        pixels[0, 0, 0] += 0.01
        extended = TemporalTracks.model_validate(
            data | {"pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)}
        )
    elif kind == "seed":
        extended = TemporalTracks.model_validate(data | {"seed_count": 33})
    else:
        extended = TemporalTracks.model_validate(
            data
            | {
                "surviving_counts": (31,) * 6,
                "track_ids": tuple(range(31)),
                "pixels": tuple(frame[:31] for frame in data["pixels"]),
                "minimum_patch_correlations": (1,) * 31,
                "chain_cycle_errors": (0,) * 31,
            }
        )
    with pytest.raises(ValueError):
        controller.advance(extended)
    assert controller.digest() == before


def test_original_track_loss_cannot_resurrect(geometry):
    controller = window()
    assert controller.advance(tracks(6, ids=tuple(range(1, 32)))).retained_tracks == 31
    decision = controller.advance(tracks(7))
    assert decision.retained_tracks == 31
    assert 0 not in controller.snapshot()["prefixes"][-1]["track_ids"]


@pytest.mark.parametrize("mode", ["direction", "rotation", "single", "none"])
def test_disagreement_or_insufficient_models_cannot_select_a_winner(
    geometry, monkeypatch, mode
):
    _, base = geometry

    def future(fitted, past, current, **kwargs):
        result = base(fitted, past, current, **kwargs)
        if mode == "none" or (mode == "single" and fitted.bootstrap_slot != 2):
            return result | {"status": "unavailable", "frame_checks": [], "poses": []}
        if fitted.bootstrap_slot == 2 and mode in ("direction", "rotation"):
            if mode == "direction":
                shift = np.array(fitted.poses[-1].translation) + (1, 0, 0)
                result["poses"][0]["translation"] = shift.tolist()
            else:
                angle = np.deg2rad(10)
                rotation = np.array(
                    (
                        (np.cos(angle), -np.sin(angle), 0),
                        (np.sin(angle), np.cos(angle), 0),
                        (0, 0, 1),
                    )
                )
                result["poses"][0]["rotation"] = rotation.tolist()
                result["poses"][0]["translation"] = (
                    rotation @ np.array(fitted.poses[-1].translation) + (0, 0, -1)
                ).tolist()
        return result

    monkeypatch.setattr(causal, "validate_joint_future", future)
    controller = window()
    controller.advance(tracks(6))
    decision = controller.advance(tracks(7))
    assert decision.instruction == "reject" and decision.unit_translation is None
    assert len(decision.candidates) == 3


def test_snapshot_mutation_cannot_change_state(geometry):
    controller = window()
    before = controller.digest()
    snapshot = controller.snapshot()
    snapshot["maps"][0]["poses"][0]["translation"][0] = 99
    assert controller.digest() == before


def test_refresh_cannot_use_earlier_or_unseen_frames_or_exceed_budget(geometry):
    controller = causal.CausalMapWindow(
        tracks(5),
        CAMERA,
        anchor_frame=48,
        policy=causal.CausalWindowPolicy(maximum_windows=1),
    )
    with pytest.raises(ValueError, match="before"):
        controller.refresh(tracks(5), anchor_frame=50)
    controller.advance(tracks(6))
    controller.advance(tracks(7))
    before = controller.digest()
    for anchor in (48, 49, 51):
        with pytest.raises(ValueError):
            controller.refresh(tracks(5), anchor_frame=anchor)
    with pytest.raises(ValueError, match="budget"):
        controller.refresh(tracks(5, offset=2), anchor_frame=50)
    assert controller.digest() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon": True},
        {"horizon": 5},
        {"maximum_windows": 65},
        {"maximum_direction_disagreement": 21},
        {"minimum_models": 1},
    ],
)
def test_policy_is_bounded_and_does_not_weaken_geometry(changes):
    with pytest.raises(ValidationError):
        causal.CausalWindowPolicy.model_validate(changes)


def test_real_training_only_maps_validate_the_exact_current_horizon():
    pytest.importorskip("cv2")
    from test_joint_multiview_map import CAMERA as camera
    from test_joint_multiview_map import fixture_tracks, prefix

    extended, _, _ = fixture_tracks(frames=9)
    controller = causal.CausalMapWindow(
        prefix(extended, 7), camera, anchor_frame=10, seed=13
    )
    assert controller.advance(prefix(extended, 8)).instruction == "collect"
    decision = controller.advance(extended)
    assert decision.instruction == "refresh"
    assert decision.absolute_frame == 18 and len(decision.candidates) == 5
    assert not decision.navigation_accepted
