import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL.Image")

import safe_raw_fit_probe as probe
from test_safe_bridge_probe import bind_public, images, save

from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.stable_keyframe_map import (
    StableSafeSeedKeyframePolicy,
)
from minecraft_scientist.exploration.temporal_tracks import diagnose_frame_chain

pytest_plugins = ("test_safe_bridge_probe",)


@pytest.fixture
def rejected_proposal(monkeypatch):
    """Mock fitting isolates evidence/clock boundaries, not native geometry proof."""
    frames = images(11, size=224)
    slots = [0, 2, 4, 6, 8, 10]
    selected = [frames[slot] for slot in slots]
    settings = StableSafeSeedKeyframePolicy()
    seeds = safe_seeds(frames[0])
    past = diagnose_frame_chain(
        selected, config=settings.temporal, seed_pixels=seeds
    ).tracks.model_dump(mode="json")
    calls = []

    def fit(tracks, camera, *, bootstrap_slot, seed, config):
        assert config == settings.fitting and seed == 19
        assert tracks.frame_count == 6 and tracks.seed_count == len(seeds)
        calls.append((bootstrap_slot, tracks))
        data = {
            "bootstrap_slot": bootstrap_slot,
            "supported": False,
            "reason": "test-only mock rejection",
        }
        return SimpleNamespace(model_dump=lambda **kwargs: data)

    monkeypatch.setattr(probe, "fit_joint_map", fit)
    proposal = {
        "index": {
            "frame_ordinals": slots,
            "rgb_sha256": [
                hashlib.sha256(frame.tobytes()).hexdigest() for frame in selected
            ],
            "tracks": past,
        },
        "past": past,
        "frozen_at_frame": 10,
        "detector": {
            "registration": settings.seed_detection,
            "seed_image_frame": 0,
            "seed_image_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
            "seed_pixels": seeds.tolist(),
        },
        "maps": [
            {
                "bootstrap_slot": slot,
                "supported": False,
                "reason": "test-only mock rejection",
            }
            for slot in range(2, 6)
        ],
        "required_bootstraps": [],
    }
    return proposal, frames, settings, calls


def compare(fixture):
    proposal, frames, settings, _ = fixture
    return probe.compare_proposal(
        proposal,
        frames,
        ((100, 0, 112), (0, 100, 112), (0, 0, 1)),
        19,
        settings.model_dump(mode="json"),
    )


def test_complete_prefix_preserves_seed_ids_clock_and_every_bootstrap(
    rejected_proposal,
):
    proposal, frames, _, calls = rejected_proposal
    old = json.dumps(proposal, sort_keys=True)
    images_before = [frame.copy() for frame in frames]
    result = compare(rejected_proposal)
    assert result["frame_ordinals"] == [0, 2, 4, 6, 8, 10]
    assert result["raw_diagnosis"]["tracks"]["frame_count"] == 11
    assert result["projected"]["frame_count"] == 6
    assert (
        result["projected"]["track_ids"]
        == result["raw_diagnosis"]["tracks"]["track_ids"]
    )
    assert [slot for slot, _ in calls] == [2, 3, 4, 5] * 2
    assert result["all_original_models_replayed"]
    assert not result["raw_past_consensus_candidate"]
    assert not result["future_validated"] and not result["navigation_accepted"]
    assert not result["original_proposal_replaced"]
    assert json.dumps(proposal, sort_keys=True) == old
    assert all(np.array_equal(a, b) for a, b in zip(frames, images_before, strict=True))


def test_failed_unselected_middle_is_not_rescued_by_projection(rejected_proposal):
    rejected_proposal[1][1] = np.zeros_like(rejected_proposal[1][1])
    result = compare(rejected_proposal)
    assert result["projected"]["track_ids"] == []
    assert (
        result["projected"]["seed_count"]
        == result["raw_diagnosis"]["tracks"]["seed_count"]
    )
    assert len(rejected_proposal[3]) == 8


@pytest.mark.parametrize(
    "case",
    [
        "short_views",
        "wrong_end",
        "seed_pixels",
        "seed_frame",
        "seed_hash",
        "rgb_hash",
        "past_tracks",
        "model",
        "required_bootstraps",
    ],
)
def test_original_evidence_and_clock_cannot_be_changed(rejected_proposal, case):
    proposal = rejected_proposal[0]
    if case == "short_views":
        proposal["index"]["frame_ordinals"] = [0, 2, 4]
    elif case == "wrong_end":
        proposal["frozen_at_frame"] = 11
    elif case == "seed_pixels":
        proposal["detector"]["seed_pixels"][0][0] += 1
    elif case == "seed_frame":
        proposal["detector"]["seed_image_frame"] = 10
    elif case == "seed_hash":
        proposal["detector"]["seed_image_sha256"] = "0" * 64
    elif case == "rgb_hash":
        proposal["index"]["rgb_sha256"][-1] = "0" * 64
    elif case == "past_tracks":
        proposal["past"]["seed_count"] += 1
    elif case == "model":
        proposal["maps"][0]["supported"] = True
    else:
        proposal["required_bootstraps"] = [2]
    with pytest.raises(ValueError, match="original|replay|bounded"):
        compare(rejected_proposal)


def test_raw_tracker_cannot_change_seed_denominator(rejected_proposal, monkeypatch):
    diagnose = probe.diagnose_frame_chain

    def broken(frames, **kwargs):
        result = diagnose(frames, **kwargs)
        if len(frames) == 11:
            result = result.model_copy(
                update={
                    "tracks": result.tracks.model_copy(
                        update={"seed_count": result.tracks.seed_count + 1}
                    )
                }
            )
        return result

    monkeypatch.setattr(probe, "diagnose_frame_chain", broken)
    with pytest.raises(ValueError, match="frozen seed denominator"):
        compare(rejected_proposal)


@pytest.fixture
def no_proposal_capture(public_capture):
    path = public_capture / "keyframe-acquisition.json"
    saved = json.loads(path.read_bytes())
    policy = StableSafeSeedKeyframePolicy().model_dump(mode="json")
    saved["acquisition"].update(
        policy=policy, seed=19, failure=None, frozen=None, future=None, attempts=[]
    )
    saved["calibration"] = {
        "camera_intrinsics": [[200, 0, 224], [0, 200, 224], [0, 0, 1]]
    }
    save(path, saved)
    config = json.loads((public_capture / "config.json").read_bytes())
    config["keyframe_policy"] = policy
    save(public_capture / "config.json", config)
    bind_public(public_capture)
    return public_capture


def test_no_original_proposal_stays_empty_and_never_reads_evaluator(
    no_proposal_capture, tmp_path, monkeypatch
):
    read_bytes = Path.read_bytes

    def guard(path):
        if path.name.startswith("evaluator-"):
            raise AssertionError("accessed private evaluator file")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    output = tmp_path / "no-proposal"
    result = probe.probe(no_proposal_capture, output)
    assert all(result["checks"].values()) and result["attempts"] == []
    assert not result["future_validated"] and not result["navigation_accepted"]
    for name, digest in result["config"]["source_sha256"].items():
        assert probe.digest(output / "source-snapshot" / name) == digest
    with pytest.raises(FileExistsError):
        probe.probe(no_proposal_capture, output)


@pytest.mark.parametrize("case", ["audit", "future", "capture_hash", "policy"])
def test_bad_input_rejects_before_output_allocation(
    no_proposal_capture, tmp_path, case
):
    if case == "audit":
        path = no_proposal_capture / "audit.json"
        saved = json.loads(path.read_bytes())
        saved["valid"] = False
        save(path, saved)
    elif case == "capture_hash":
        path = no_proposal_capture / "report.json"
        saved = json.loads(path.read_bytes())
        saved["config_sha256"] = "0" * 64
        save(path, saved)
    else:
        path = no_proposal_capture / "keyframe-acquisition.json"
        saved = json.loads(path.read_bytes())
        if case == "future":
            saved["acquisition"]["future"] = {"forbidden": True}
        else:
            saved["acquisition"]["policy"]["fitting"]["iterations"] = 31
        save(path, saved)
        bind_public(no_proposal_capture)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        probe.probe(no_proposal_capture, output)
    assert not output.exists()
