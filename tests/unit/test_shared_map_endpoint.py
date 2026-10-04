import copy

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("cv2", reason="requires optional geometry extra")

from test_joint_multiview_map import CAMERA as REFERENCE_CAMERA
from test_joint_multiview_map import fixture_tracks

from minecraft_scientist.exploration import rolling_map_control as rolling
from minecraft_scientist.exploration import shared_map_endpoint as endpoint
from minecraft_scientist.exploration.joint_multiview_map import _project
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.settled_map_control import PublicControlStep
from minecraft_scientist.exploration.shared_anchor_map import refine_shared_crisis_map
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = REFERENCE_CAMERA.copy()
CAMERA[:2, 2] += 112  # Wider synthetic sensor keeps the lateral test in view.


@pytest.fixture(scope="module")
def scene():
    tracks, rotations, translations = fixture_tracks(frames=16, noise=0.03)
    rng = np.random.default_rng(411)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 150), rng.uniform(-1.6, 1.6, 150), rng.uniform(4, 9, 150))
    )
    pixels = [tuple((x + 112, y + 112) for x, y in row) for row in tracks.pixels]
    rotation, center = rotations[-1], -rotations[-1].T @ translations[-1]
    for slot in range(16, 50):
        amount = 0 if slot < 24 else min(slot - 23, 2)
        shift = -rotation @ (center + np.array((0.08 * amount, 0, 0)))
        observed, _ = _project(xyz, rotation, shift, CAMERA)
        pixels.append(tuple(map(tuple, observed)))
    extended = TemporalTracks.model_validate(
        tracks.model_dump()
        | {
            "pixels": tuple(pixels),
            "image_shape": (448, 448),
            "frame_count": 50,
            "surviving_counts": (150,) * 50,
        }
    )
    return extended, [np.full((448, 448, 3), i, dtype=np.uint8) for i in range(50)]


@pytest.fixture
def session(scene, monkeypatch):
    tracks, images = scene

    def track(frames, **kwargs):
        indices = [int(image[0, 0, 0]) for image in frames]
        return TemporalTracks.model_validate(
            tracks.model_dump()
            | {
                "frame_count": len(indices),
                "pixels": tuple(tracks.pixels[i] for i in indices),
                "surviving_counts": (150,) * len(indices),
            }
        )

    monkeypatch.setattr(rolling, "track_frame_chain", track)
    monkeypatch.setattr(endpoint, "track_frame_chain", track)
    parent = {
        "terminal": "map_unavailable",
        "failure": None,
        "seed": 13,
        "acquisition": {
            "anchor_frame": 0,
            "evidence_frames": 12,
            "steps": [{"requested_handle": i} for i in [0] * 6 + [2] * 4 + [1] * 2],
        },
        "settled": None,
    }
    old_steps = []

    def execute_old(kind, handle):
        old_steps.append((kind, handle))
        return PublicControlStep(images[11 + len(old_steps)], kind, handle, True)

    maintained = rolling.maintain_public_maps(
        parent, images[:12], execute_old, CAMERA, gauge_policy={}, shared_policy={}
    )
    assert maintained["terminal"] == "completed" and maintained["failure"] is None
    assert maintained["shared_anchor_versions"][0]["status"] == "shadow_validated"
    requests = []

    def execute(kind, handle):
        requests.append((kind, handle))
        return PublicControlStep(images[15 + len(requests)], kind, handle, True)

    return maintained, images[:16], execute, requests, track


def test_counterexample_fit_and_both_settled_endpoints_are_frozen_and_executable(
    session,
):
    parent, images, execute, requests, _ = session
    old = copy.deepcopy(parent)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert result["terminal"] == "validated" and result["status"] == "shadow_validated"
    assert result["failure"] is None and parent == old
    assert len(requests) == 18
    assert requests[:8] == [("wait", None)] * 8
    assert requests[8:10] == [("move", result["validation_handle"])] * 2
    assert requests[10:] == [("wait", None)] * 8
    assert result["validation_handle"] != parent["steps"][-1]["requested_handle"]
    assert result["initial_endpoint"]["absolute_frame"] == 23
    assert result["final_endpoint"]["absolute_frame"] == 33
    assert result["initial_settle"]["settled"] and result["final_settle"]["settled"]
    assert result["frozen"]["past"]["frame_count"] == 16
    assert result["frozen_sha256"] == endpoint._digest(result["frozen"])
    assert result["initial_endpoint_sha256"] == endpoint._digest(
        result["initial_endpoint"]
    )
    assert len(result["frozen"]["maps"]) == 10
    assert result["motion"]["supported"] and not result["navigation_accepted"]
    assert not result["metric_scale_known"] and not result["global_gauge_aligned"]


def test_earlier_failure_is_preserved_and_labels_only_the_update_kind(session):
    parent, images, execute, _, _ = session
    # Synthetic status fixture; real journals must independently replay.
    parent["shared_anchor_versions"][0]["status"] = "rejected"
    old = copy.deepcopy(parent)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert result["update_kind"] == "crisis_refit" and result["terminal"] == "validated"
    assert parent == old and parent["shared_anchor_versions"][0]["status"] == "rejected"


def test_consumed_counterexample_holdouts_cannot_fit_poses_or_their_own_points(session):
    parent, _, _, _, _ = session
    root = parent["epochs"][0]["frozen"]["snapshot"]
    _, original, models = endpoint.original_shared_root(root)
    consumed = TemporalTracks.model_validate(
        parent["shared_anchor_versions"][0]["terminal_snapshot"]["prefixes"][-1]
    )
    model = next(model for model in models if model.supported)
    good = refine_shared_crisis_map(model, original, consumed)
    pixels = np.array(consumed.pixels)
    index = {identity: i for i, identity in enumerate(consumed.track_ids)}
    pixels[-1, [index[identity] for identity in model.validation_ids]] += (12, -9)
    corrupted = TemporalTracks.model_validate(
        consumed.model_dump()
        | {"pixels": tuple(tuple(map(tuple, row)) for row in pixels)}
    )
    bad = refine_shared_crisis_map(model, original, corrupted)
    assert good.supported and not bad.supported
    assert good.poses == bad.poses and good.training_points == bad.training_points
    assert good.validation_points == bad.validation_points


def test_missing_map_support_does_not_abort_visible_waiting_or_rescue_a_subset(
    session, monkeypatch
):
    parent, images, execute, requests, _ = session
    fit = endpoint.refine_shared_crisis_map
    first = parent["shared_anchor_versions"][0]["frozen"]["required_bootstraps"][0]

    def unsupported(model, *args, **kwargs):
        result = fit(model, *args, **kwargs)
        return (
            result.model_copy(update={"supported": False})
            if model.bootstrap_slot == first
            else result
        )

    monkeypatch.setattr(endpoint, "refine_shared_crisis_map", unsupported)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert len(requests) == 18 and result["final_settle"]["settled"]
    assert result["terminal"] == "rejected" and not result["motion"]["supported"]
    assert len(result["initial_endpoint"]["candidates"]) == 10


def test_initial_endpoint_keeps_its_own_survivors_when_a_later_identity_is_lost(
    session, monkeypatch
):
    parent, images, execute, _, track = session
    identity = parent["shared_anchor_versions"][0]["frozen"]["maps"][-1][
        "validation_ids"
    ][0]

    def dropout(frames, **kwargs):
        result = track(frames, **kwargs)
        if len(frames) > 2 and int(frames[-1][0, 0, 0]) >= 24:
            return retain_tracks(result, set(result.track_ids) - {identity})
        return result

    monkeypatch.setattr(endpoint, "track_frame_chain", dropout)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    before = result["initial_endpoint"]["candidates"][-1]["future"]
    after = result["final_endpoint"]["candidates"][-1]["future"]
    assert (
        identity in before["validation_ids"] and identity not in after["validation_ids"]
    )
    assert all(
        identity not in item["retained_ids"]
        for item in result["prefixes"]
        if item["absolute_frame"] >= 24
    )


def test_future_controls_cannot_refit_the_sixteen_view_proposal(session, monkeypatch):
    parent, images, execute, _, _ = session
    fit = endpoint.refine_shared_crisis_map
    seen = []

    def observed(*args, **kwargs):
        seen.append(len(requests))
        return fit(*args, **kwargs)

    requests = []
    monkeypatch.setattr(endpoint, "refine_shared_crisis_map", observed)

    def guarded(kind, handle):
        requests.append(kind)
        assert len(seen) == 10 and set(seen) == {0}
        return execute(kind, handle)

    assert (
        endpoint.run_shared_endpoint_trial(parent, images, guarded)["terminal"]
        == "validated"
    )


def test_wait_exhaustion_keeps_every_prefix_without_moving(session, monkeypatch):
    parent, images, execute, requests, _ = session
    flow = endpoint._visible_flow

    def moving(*args):
        return flow(*args) | {"stable": False}

    monkeypatch.setattr(endpoint, "_visible_flow", moving)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert len(requests) == 16 and all(kind == "wait" for kind, _ in requests)
    assert result["terminal"] == "initial_wait_exhausted"
    assert result["initial_endpoint"] is None and result["motion"] is None
    assert len(result["flow_prefixes"]) == len(result["prefixes"]) == 16


@pytest.mark.parametrize(
    "case", ["kind", "handle", "ticks", "compliant", "shape", "receipt"]
)
def test_failed_receipt_never_enters_endpoint_evidence(session, case):
    parent, images, _, _, _ = session

    def failed(*args):
        if case == "receipt":
            return {"frame": images[-1]}
        return PublicControlStep(
            np.zeros((64, 64, 3), dtype=np.uint8) if case == "shape" else images[-1],
            "move" if case == "kind" else "wait",
            0 if case == "handle" else None,
            case != "compliant",
            2 if case == "ticks" else 1,
        )

    result = endpoint.run_shared_endpoint_trial(parent, images, failed)
    assert result["terminal"] == "control_failed" and result["failure"] is not None
    assert len(result["steps"]) == 1 and result["prefixes"] == []


def test_parent_mutation_during_control_refuses(session):
    parent, images, execute, _, _ = session

    def mutate(*args):
        parent["terminal"] = "forged"
        return execute(*args)

    with pytest.raises(ValueError, match="immutable parent"):
        endpoint.run_shared_endpoint_trial(parent, images, mutate)


@pytest.mark.parametrize("target", ["map", "initial_endpoint"])
def test_future_controls_cannot_mutate_frozen_evidence(session, monkeypatch, target):
    parent, images, execute, _, _ = session
    digest = endpoint._digest
    captured = {}

    def observe(value):
        if isinstance(value, dict):
            if "required_bootstraps" in value and "previous_version_sha256" in value:
                captured["map"] = value
            elif "candidates" in value and "absolute_frame" in value:
                captured["initial_endpoint"] = value
        return digest(value)

    monkeypatch.setattr(endpoint, "_digest", observe)

    def mutate(kind, handle):
        if kind == "move":
            captured[target]["forged"] = True
        return execute(kind, handle)

    with pytest.raises(ValueError, match="frozen endpoint map|frozen initial endpoint"):
        endpoint.run_shared_endpoint_trial(parent, images, mutate)


def test_stationary_endpoints_cannot_promote_numerical_motion(session, monkeypatch):
    parent, images, execute, requests, track = session

    def stationary(frames, **kwargs):
        remapped = [
            np.full_like(image, min(int(image[0, 0, 0]), 15)) for image in frames
        ]
        return track(remapped, **kwargs)

    monkeypatch.setattr(endpoint, "track_frame_chain", stationary)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert len(requests) == 18
    assert result["initial_settle"]["settled"] and result["final_settle"]["settled"]
    assert result["terminal"] == "rejected" and not result["motion"]["supported"]
    assert all(
        item["translation_ratio"]
        <= endpoint.SharedEndpointPolicy().minimum_translation_ratio
        for item in result["motion"]["endpoint_motion_ratios"]
    )


def test_failed_initial_endpoint_preserves_completed_wait_controls(
    session, monkeypatch
):
    parent, images, execute, requests, _ = session

    def fail(*args, **kwargs):
        raise ValueError("synthetic endpoint solve failure")

    monkeypatch.setattr(endpoint, "validate_joint_future", fail)
    result = endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert requests == [("wait", None)] * 8
    assert len(result["steps"]) == len(result["prefixes"]) == 8
    assert result["initial_settle"]["settled"]
    assert result["terminal"] == "tracking_failed"
    assert result["failure"]["message"] == "synthetic endpoint solve failure"
    assert result["initial_endpoint"] is None and result["motion"] is None


@pytest.mark.parametrize(
    "case", ["frames", "rgb", "lineage", "prefix", "private", "failed"]
)
def test_invalid_initial_checkpoint_refuses_before_controls(session, case):
    parent, images, execute, requests, _ = session
    if case == "frames":
        images = images[:-1]
    elif case == "rgb":
        images = [image.copy() for image in images]
        images[0][0, 0] = 200
    elif case == "lineage":
        parent["shared_anchor_versions"][0]["original_snapshot_sha256"] = "0" * 64
    elif case == "prefix":
        saved = parent["shared_anchor_versions"][0]["terminal_snapshot"]["prefixes"][-1]
        saved["pixels"][0][0][0] += 0.01
    elif case == "private":
        parent["metric_scale_known"] = True
    else:
        parent["failure"] = {"type": "failed control"}
    with pytest.raises(ValueError):
        endpoint.run_shared_endpoint_trial(parent, images, execute)
    assert requests == []


def test_no_shared_root_does_not_invent_new_controls_or_version():
    parent = {
        "failure": None,
        "navigation_accepted": False,
        "metric_scale_known": False,
        "global_gauge_aligned": False,
        "shared_anchor_versions": [],
    }
    result = endpoint.run_shared_endpoint_trial(
        parent, [], lambda *args: pytest.fail("no root")
    )
    assert result["terminal"] == "not_started" and result["frozen"] is None


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_wait_ticks": 16, "maximum_wait_ticks": 8},
        {"percentile95_flow_pixels": 1},
        {"temporal": {"maximum_frames": 65}},
        {"fitting": {"fitting": {"maximum_frames": 16, "reference_huber_pixels": 2}}},
    ],
)
def test_endpoint_policy_cannot_relax_gates_or_change_tracker(policy):
    with pytest.raises(ValidationError):
        endpoint.SharedEndpointPolicy.model_validate(policy)
