import copy

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("cv2", reason="requires optional geometry extra")

from test_joint_multiview_map import CAMERA, fixture_tracks, prefix

from minecraft_scientist.exploration import rolling_map_control as rolling
from minecraft_scientist.exploration import shared_anchor_map as shared
from minecraft_scientist.exploration.causal_map_window import CausalMapWindow
from minecraft_scientist.exploration.joint_multiview_map import fit_joint_map
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.settled_map_control import PublicControlStep
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


@pytest.fixture(scope="module")
def scene():
    tracks, rotations, translations = fixture_tracks(frames=11, noise=0.03)
    past = prefix(tracks, 7)
    root = CausalMapWindow(past, CAMERA, anchor_frame=50, seed=13).snapshot()
    assert sum(item["supported"] for item in root["maps"]) >= 2
    return tracks, rotations, translations, past, root


def altered(tracks, pixels):
    return TemporalTracks.model_validate(
        tracks.model_dump()
        | {"pixels": tuple(tuple(map(tuple, row)) for row in pixels)}
    )


def test_shared_landmarks_recover_new_version_and_independent_bent_future(scene):
    tracks, rotations, translations, _, root = scene
    saved = copy.deepcopy(root)
    window = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    frozen = window.snapshot()
    assert frozen["shared_anchor"] and not frozen["global_gauge_aligned"]
    assert frozen["anchor_frame"] == 50
    assert frozen["original_fitting_frames"] == 7
    assert len(frozen["maps"]) == len(root["maps"])
    assert window.advance(prefix(tracks, 10)).instruction == "collect"
    decision = window.advance(tracks)
    assert decision.instruction == "refresh"
    for original, revised in zip(root["maps"], frozen["maps"], strict=True):
        assert revised["reserved_training_ids"] == original["reserved_training_ids"]
        assert revised["validation_ids"] == original["validation_ids"]
        if revised["supported"]:
            scale = np.linalg.norm(translations[original["bootstrap_slot"]])
            np.testing.assert_allclose(revised["poses"][0]["rotation"], np.eye(3))
            np.testing.assert_allclose(revised["poses"][0]["translation"], (0, 0, 0))
            np.testing.assert_allclose(
                revised["poses"][-1]["rotation"], rotations[8], atol=0.003
            )
            np.testing.assert_allclose(
                revised["poses"][-1]["translation"], translations[8] / scale, atol=0.03
            )
    assert root == saved
    assert window.snapshot()["maps"] == frozen["maps"]
    assert window.snapshot()["past"] == frozen["past"]
    assert not decision.navigation_accepted and not decision.metric_scale_known


def test_consumed_withheld_rays_never_change_training_or_fit_withheld_points(
    scene, monkeypatch
):
    tracks, _, _, past, _ = scene
    parent = fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    consumed = prefix(tracks, 9)
    lengths = []
    triangulate = shared._triangulate_many

    def observe(pixels, *args):
        lengths.append(len(pixels))
        return triangulate(pixels, *args)

    monkeypatch.setattr(shared, "_triangulate_many", observe)
    good = shared.refine_shared_map(parent, past, consumed)
    pixels = np.array(consumed.pixels)
    pixels[8, list(parent.validation_ids)] += (12, -9)
    bad = shared.refine_shared_map(parent, past, altered(consumed, pixels))
    assert good.supported and not bad.supported
    assert good.poses == bad.poses
    assert good.training_points == bad.training_points
    assert good.validation_points == bad.validation_points
    assert good.training_cost_history == bad.training_cost_history
    assert lengths == [past.frame_count, past.frame_count]


def test_future_pixels_do_not_refit_the_proposal(scene, monkeypatch):
    tracks, _, _, _, root = scene
    window = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    frozen = window.snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("future cannot refit training or withheld points")

    monkeypatch.setattr(shared, "_bundle", forbidden)
    monkeypatch.setattr(shared, "_triangulate_many", forbidden)
    window.advance(prefix(tracks, 10))
    assert window.advance(tracks).instruction == "refresh"
    assert window.snapshot()["maps"] == frozen["maps"]


def test_no_passing_subset_can_replace_a_failed_original_supported_model(
    scene, monkeypatch
):
    tracks, _, _, _, root = scene
    required = [item["bootstrap_slot"] for item in root["maps"] if item["supported"]]
    refine = shared.refine_shared_map

    def fail_one(model, *args, **kwargs):
        result = refine(model, *args, **kwargs)
        if model.bootstrap_slot == required[0]:
            return result.model_copy(
                update={"supported": False, "reason": "retained counterexample"}
            )
        return result

    monkeypatch.setattr(shared, "refine_shared_map", fail_one)
    window = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    window.advance(prefix(tracks, 10))
    decision = window.advance(tracks)
    assert decision.instruction == "reject" and decision.unit_translation is None
    assert "no passing-subset rescue" in decision.reason
    assert len(decision.candidates) == len(root["maps"])


def test_lost_original_partitions_cannot_be_replenished(scene):
    tracks, _, _, past, _ = scene
    parent = fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    consumed = retain_tracks(
        prefix(tracks, 9), set(tracks.track_ids) - set(parent.validation_ids[7:])
    )
    result = shared.refine_shared_map(parent, past, consumed)
    assert not result.supported and len(result.validation_ids) == 7
    assert result.reserved_training_ids == parent.reserved_training_ids
    assert "no replenishment" in result.reason


def test_lost_identity_cannot_return_in_a_later_future(scene):
    tracks, _, _, _, root = scene
    window = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    window.advance(retain_tracks(prefix(tracks, 10), set(tracks.track_ids) - {0}))
    window.advance(tracks)
    assert all(
        0 not in ids
        for row in window.snapshot()["decisions"][-1]["candidates"]
        for ids in (row["future"]["training_ids"], row["future"]["validation_ids"])
    )


def test_corrupt_future_holdouts_reject_without_adapting_training_pose(scene):
    tracks, _, _, _, root = scene
    good = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    bad = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    good.advance(prefix(tracks, 10))
    bad.advance(prefix(tracks, 10))
    good_decision = good.advance(tracks)
    pixels = np.array(tracks.pixels)
    valid_ids = root["maps"][0]["validation_ids"]
    pixels[-1, valid_ids] += (12, -9)
    bad_decision = bad.advance(altered(tracks, pixels))
    assert (
        good_decision.instruction == "refresh" and bad_decision.instruction == "reject"
    )
    assert [item.future.poses for item in good_decision.candidates] == [
        item.future.poses for item in bad_decision.candidates
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "future",
        "missing_model",
        "private",
        "metric",
        "negative_anchor",
        "relaxed_policy",
    ],
)
def test_invalid_parent_root_refuses(scene, fault):
    tracks, _, _, _, root = scene
    forged = copy.deepcopy(root)
    if fault == "future":
        forged["prefixes"] = [tracks.model_dump(mode="json")]
    elif fault == "missing_model":
        forged["maps"].pop()
    elif fault == "private":
        forged["navigation_accepted"] = True
    elif fault == "metric":
        forged["metric_scale_known"] = True
    elif fault == "negative_anchor":
        forged["anchor_frame"] = -1
    else:
        forged["policy"]["fitting"]["geometry"]["reference_validation_pixels"] = 3
    with pytest.raises(ValueError, match="original frozen"):
        shared.SharedAnchorWindow(forged, prefix(tracks, 9))


@pytest.mark.parametrize(
    "fault", ["pixels", "seed", "survival", "no_extension", "excess"]
)
def test_invalid_consumed_history_refuses(scene, fault):
    tracks, _, _, past, _ = scene
    parent = fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    consumed = prefix(tracks, 9)
    if fault == "pixels":
        pixels = np.array(consumed.pixels)
        pixels[2, 0, 0] += 0.01
        consumed = altered(consumed, pixels)
    elif fault == "seed":
        consumed = consumed.model_copy(update={"seed_count": 151})
    elif fault == "survival":
        consumed = consumed.model_copy(update={"surviving_counts": (149,) * 9})
        consumed = retain_tracks(consumed, set(consumed.track_ids) - {0})
    elif fault == "no_extension":
        consumed = past
    else:
        consumed, _, _ = fixture_tracks(frames=15)
    with pytest.raises(ValueError, match="unchanged-anchor|original pixels"):
        shared.refine_shared_map(parent, past, consumed)


def test_redetection_refresh_is_not_silent_repartition(scene):
    tracks, _, _, _, root = scene
    window = shared.SharedAnchorWindow(root, prefix(tracks, 9))
    with pytest.raises(ValueError, match="protected-partition"):
        window.refresh(prefix(tracks, 9), anchor_frame=52)


def test_original_geometry_cannot_be_relaxed():
    with pytest.raises(ValidationError, match="cannot relax"):
        shared.SharedAnchorPolicy(
            fitting={"maximum_frames": 14, "reference_huber_pixels": 2}
        )


@pytest.fixture
def rolling_scene(monkeypatch):
    tracks, _, _ = fixture_tracks(frames=16, noise=0.03)
    images = [np.full((224, 224, 3), i, dtype=np.uint8) for i in range(16)]
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
    requests = []

    def track(frames, **kwargs):
        first = int(frames[0][0, 0, 0])
        data = tracks.model_dump()
        return TemporalTracks.model_validate(
            data
            | {
                "pixels": data["pixels"][first : first + len(frames)],
                "frame_count": len(frames),
                "surviving_counts": (150,) * len(frames),
            }
        )

    def execute(kind, handle):
        requests.append((kind, handle))
        return PublicControlStep(images[11 + len(requests)], kind, handle, True)

    monkeypatch.setattr(rolling, "track_frame_chain", track)
    return parent, images[:12], execute, requests


def test_rolling_shared_version_is_frozen_before_identical_new_control_budget(
    rolling_scene, monkeypatch
):
    parent, images, execute, requests = rolling_scene
    original = copy.deepcopy(parent)
    ctor = rolling.SharedAnchorWindow
    snapshots = []

    def observe(root, consumed, **kwargs):
        assert len(requests) == 2 and consumed.frame_count == 14
        controller = ctor(root, consumed, **kwargs)
        snapshots.append(controller.snapshot())
        return controller

    monkeypatch.setattr(rolling, "SharedAnchorWindow", observe)
    result = rolling.maintain_public_maps(
        parent, images, execute, CAMERA, gauge_policy={}, shared_policy={}
    )
    assert result["failure"] is None and result["terminal"] == "completed"
    assert len(requests) == 4 and parent == original
    version = result["shared_anchor_versions"][0]
    assert version["frozen"] == snapshots[0]
    assert version["frozen"]["decisions"] == []
    assert version["frozen"]["anchor_frame"] == 0
    assert len(version["rgb_sha256"]) == 14
    assert version["status"] == "shadow_validated"
    assert result["shadow_validated_shared_versions"] == 1
    assert (
        version["original_snapshot_sha256"]
        == result["epochs"][0]["frozen"]["snapshot_sha256"]
    )
    assert version["frozen"]["maps"] == version["terminal_snapshot"]["maps"]
    assert len(version["future_prefixes"]) == 2
    assert not result["global_gauge_aligned"] and not result["navigation_accepted"]


def test_shared_proposal_cannot_bypass_gauge_failure_record(rolling_scene, monkeypatch):
    parent, images, execute, _ = rolling_scene
    monkeypatch.setattr(
        rolling,
        "register_map_gauges",
        lambda *args, **kwargs: {
            "supported": False,
            "reason": "retained unit counterexample",
        },
    )
    result = rolling.maintain_public_maps(
        parent, images, execute, CAMERA, gauge_policy={}, shared_policy={}
    )
    assert result["shadow_validated_shared_versions"] == 1
    assert result["supported_local_gauge_links"] == 0
    assert (
        result["gauge_bridges"][0]["evidence"]["reason"]
        == "retained unit counterexample"
    )


def test_shared_policy_requires_retained_independent_roots_and_gauge_journal(
    rolling_scene,
):
    parent, images, execute, requests = rolling_scene
    with pytest.raises(ValueError, match="retained gauge"):
        rolling.maintain_public_maps(parent, images, execute, CAMERA, shared_policy={})
    assert requests == []


def test_future_mutation_of_frozen_shared_version_refuses(rolling_scene, monkeypatch):
    parent, images, execute, requests = rolling_scene
    ctor = rolling.SharedAnchorWindow
    leaked = []

    def observe(*args, **kwargs):
        controller = ctor(*args, **kwargs)
        snapshot = controller.snapshot()
        controller.snapshot = lambda: snapshot
        leaked.append(snapshot)
        return controller

    def mutate(kind, handle):
        if leaked:
            leaked[0]["metric_scale_known"] = True
        return execute(kind, handle)

    monkeypatch.setattr(rolling, "SharedAnchorWindow", observe)
    with pytest.raises(ValueError, match="frozen shared-anchor"):
        rolling.maintain_public_maps(
            parent, images, mutate, CAMERA, gauge_policy={}, shared_policy={}
        )
    assert len(requests) == 4
