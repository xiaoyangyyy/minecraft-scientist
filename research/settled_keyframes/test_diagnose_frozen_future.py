import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

import diagnose_frozen_future as probe

from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)


def acquisition():
    past = [56, 60, 64, 68, 72, 76]
    return {
        "anchor_frame": 48,
        "terminal": "rejected",
        "frozen": {"index": {"frame_ordinals": past}, "frozen_at_frame": 76},
        "future": {"absolute_frame": 80, "index": {"frame_ordinals": past + [80]}},
        "steps": [
            {
                "absolute_frame": slot,
                "compliant": True,
                "ticks": 1,
                "requested_kind": "wait",
                "executed_kind": "wait",
                "requested_handle": None,
                "executed_handle": None,
            }
            for slot in range(49, 81)
        ],
    }


def test_complete_schedule_keeps_every_future_and_all_original_past():
    item = acquisition()
    before = copy.deepcopy(item)
    past, sparse, bridge = probe.schedules(item)
    assert sparse == past + [80]
    assert bridge == past + [77, 78, 79, 80]
    assert item == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item["steps"].pop(0),
        lambda item: item["steps"][-1].update(compliant=False),
        lambda item: item["steps"][-1].update(ticks=True),
        lambda item: item["steps"][-1].update(absolute_frame=80.0),
        lambda item: item["steps"][-1].update(
            requested_kind="teleport", executed_kind="teleport"
        ),
        lambda item: item["steps"][-1].update(requested_handle=1, executed_handle=1),
        lambda item: item["steps"][-1].update(
            requested_kind="move",
            executed_kind="move",
            requested_handle=True,
            executed_handle=True,
        ),
        lambda item: item["steps"][-1].update(executed_handle=1),
        lambda item: item["frozen"].update(frozen_at_frame=72),
        lambda item: item["frozen"]["index"]["frame_ordinals"].reverse(),
        lambda item: item["future"]["index"]["frame_ordinals"].pop(0),
        lambda item: item.update(anchor_frame=True),
        lambda item: item.update(frozen=None),
    ],
)
def test_incomplete_or_reselected_evidence_is_rejected(mutation):
    item = acquisition()
    mutation(item)
    with pytest.raises(ValueError):
        probe.schedules(item)


def static_evidence():
    item = acquisition()
    gray = cv2.GaussianBlur(
        np.random.default_rng(44).integers(0, 256, (448, 448), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    seeds = safe_seeds(rgb)
    config = TemporalTrackConfig(maximum_frames=128)
    past, sparse, _ = probe.schedules(item)
    item["frozen"]["detector"] = {
        "registration": "first-image-only-safe-support-mask-v1",
        "seed_image_frame": past[0],
        "seed_image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "seed_pixels": seeds.tolist(),
    }
    for key, slots in (("frozen", past), ("future", sparse)):
        item[key]["index"]["tracks"] = track_frame_chain(
            [rgb for _ in slots],
            config=config,
            seed_pixels=seeds,
        ).model_dump(mode="json")
    return item, rgb


def test_static_pixels_never_invent_a_motion_or_replace_original_result():
    item, rgb = static_evidence()
    before = copy.deepcopy(item)
    result = probe.analyze(item, lambda slot: rgb.copy())
    assert item == before
    assert result["original_terminal"] == "rejected"
    assert result["original_future_replayed"] is True
    assert result["complete_future_bridge"]["seed_count"] == len(
        item["frozen"]["detector"]["seed_pixels"]
    )
    assert result["original_result_replaced"] is False
    assert result["map_fitted"] is False
    assert result["navigation_accepted"] is False
    item["frozen"]["detector"]["seed_pixels"].pop()
    with pytest.raises(ValueError, match="denominator"):
        probe.analyze(item, lambda slot: rgb)


def test_refuses_output_inside_original_capture(tmp_path):
    route = tmp_path / "capture"
    route.mkdir()
    with pytest.raises(FileExistsError):
        probe.probe(route, route / "diagnostic")


def test_complete_bridge_cannot_be_cropped_to_fit_budget():
    item = acquisition()
    item["future"]["absolute_frame"] = 210
    item["future"]["index"]["frame_ordinals"][-1] = 210
    item["steps"].extend(
        item["steps"][-1] | {"absolute_frame": slot} for slot in range(81, 211)
    )
    with pytest.raises(ValueError, match="bounded diagnostic budget"):
        probe.schedules(item)


def save_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def mock_capture(tmp_path):
    """Synthetic ledger tests, explicitly not independent native runtime proof."""
    route = tmp_path / "mock-capture"
    (route / "frames").mkdir(parents=True)
    item, rgb = static_evidence()
    config = {"schema": probe.SCHEMA, "source_sha256": {}}
    for module in probe.DEPENDENCIES:
        source = Path(module.__file__)
        name = "package/exploration/" + source.name
        snapshot = route / "source-snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        config["source_sha256"][name] = probe.digest(source)
    save_json(route / "config.json", config)
    save_json(route / "keyframe-acquisition.json", {"acquisition": item})
    _, _, bridge = probe.schedules(item)
    for slot in bridge:
        Image.fromarray(rgb).save(route / "frames" / f"frame-{slot:05d}.png")
    save_json(
        route / "report.json",
        {
            "completed": True,
            "failure": None,
            "config_sha256": probe.digest(route / "config.json"),
            "artifact_sha256": {
                path.relative_to(route).as_posix(): probe.digest(path)
                for path in [
                    route / "keyframe-acquisition.json",
                    *(route / "frames").glob("*.png"),
                ]
            },
            "checks": {"synthetic_test_binding_only": True},
        },
    )
    (route / "audit.source.py").write_text("# Synthetic mock attestation only\n")
    save_json(
        route / "audit.json",
        {
            "valid": True,
            "checks": {"synthetic_test_binding_only": True},
            "audit_snapshot_sha256": probe.digest(route / "audit.source.py"),
            "evaluation": {"private_numeric_sentinel": 123456789},
        },
    )
    return route


def test_probe_binds_sources_inputs_and_does_not_consume_private_numbers(
    mock_capture, tmp_path
):
    before = {
        path: probe.digest(path) for path in mock_capture.rglob("*") if path.is_file()
    }
    output = tmp_path / "diagnostic"
    result = probe.probe(mock_capture, output)
    assert result["passed"] is True
    assert result["original_result_replaced"] is False
    assert result["future_validation_accepted"] is False
    assert result["navigation_accepted"] is False
    assert "private_numeric_sentinel" not in json.dumps(result)
    assert "123456789" not in json.dumps(result)
    assert before == {path: probe.digest(path) for path in before}
    for name, value in result["source_sha256"].items():
        assert probe.digest(output / name) == value
    with pytest.raises(FileExistsError):
        probe.probe(mock_capture, output)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "keyframe-acquisition.json",
        "audit.source.py",
        "source-snapshot/package/exploration/temporal_tracks.py",
        "frames/frame-00080.png",
    ],
)
def test_probe_rejects_changed_bound_bytes(mock_capture, tmp_path, filename):
    path = mock_capture / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        probe.probe(mock_capture, tmp_path / "diagnostic")


def test_probe_detects_input_change_during_analysis(
    mock_capture, tmp_path, monkeypatch
):
    original = probe.analyze

    def mutate_after_tracking(item, frame):
        result = original(item, frame)
        path = mock_capture / "config.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(probe, "analyze", mutate_after_tracking)
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="changed during analysis"):
        probe.probe(mock_capture, output)
    report = json.loads((output / "report.json").read_bytes())
    assert report["passed"] is False
    assert report["checks"]["public_inputs_unchanged"] is False
    assert report["navigation_accepted"] is False


def test_prefix_geometry_retains_all_prefixes_and_failed_bootstraps(monkeypatch):
    """Dispatch/selection double, not valid native geometry evidence."""
    item, rgb = static_evidence()
    item["frozen"].update(
        maps=[
            {"bootstrap_slot": slot, "supported": slot != 2} for slot in (2, 3, 4, 5)
        ],
        required_bootstraps=[3, 4, 5],
    )
    monkeypatch.setattr(
        probe.JointMapEvidence, "model_validate", lambda value: SimpleNamespace(**value)
    )
    calls = []

    def validate(model, past, extended, *, target_slots):
        calls.append((model.bootstrap_slot, target_slots))
        return {"status": "inconsistent" if model.bootstrap_slot == 4 else "consistent"}

    monkeypatch.setattr(probe, "validate_joint_future", validate)
    before = copy.deepcopy(item)
    rows = probe.prefix_geometry(item, lambda slot: rgb)
    assert item == before
    assert [row["absolute_frame"] for row in rows] == [77, 78, 79, 80]
    assert len(calls) == 16
    for row in rows:
        assert [part["bootstrap_slot"] for part in row["candidates"]] == [2, 3, 4, 5]
        assert row["all_required_pixel_tests_consistent"] is False
        assert row["prefix_selected_for_control"] is False
        assert row["future_validation_accepted"] is False
        assert row["navigation_accepted"] is False
    item["frozen"]["required_bootstraps"] = [3, 5]
    with pytest.raises(ValueError, match="alternative frozen bootstrap"):
        probe.prefix_geometry(item, lambda slot: rgb)


def test_prefix_geometry_flag_is_strict(tmp_path):
    route = tmp_path / "capture"
    route.mkdir()
    with pytest.raises(ValueError, match="explicit boolean"):
        probe.probe(route, tmp_path / "diagnostic", all_future_prefixes=1)
