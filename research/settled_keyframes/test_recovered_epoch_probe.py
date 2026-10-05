import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

import recovered_epoch_probe as probe

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    diagnose_frame_chain,
)


def save(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def public_capture(tmp_path):
    """Synthetic mock ledger, not native-runtime or independent-audit evidence."""
    route = tmp_path / "mock-capture"
    (route / "frames").mkdir(parents=True)
    gray = cv2.GaussianBlur(
        np.random.default_rng(19).integers(0, 256, (448, 448), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    first = np.repeat(gray[:, :, None], 3, axis=2)
    last = cv2.warpAffine(first, np.array(((1.0, 0, 1.2), (0, 1, -0.3))), (448, 448))
    for slot, frame in enumerate((first, last)):
        Image.fromarray(frame).save(route / "frames" / f"frame-{slot:05d}.png")
    tracks = diagnose_frame_chain(
        [first, last], config=TemporalTrackConfig(maximum_frames=3)
    ).tracks
    save(
        route / "config.json",
        {"schema": "recovering-stable-keyframe-camera-projection-v1"},
    )
    save(
        route / "keyframe-acquisition.json",
        {
            "acquisition": {
                "epochs": [{"epoch_id": 0}, {"epoch_id": 1, "anchor_frame": 0}],
                "samples": [
                    {
                        "epoch_id": 1,
                        "trial": 9,
                        "absolute_frame": 1,
                        "selected": False,
                        "correspondence_supported": True,
                        "novelty": {"tracks": tracks.model_dump(mode="json")},
                    }
                ],
            }
        },
    )
    save(
        route / "report.json",
        {
            "completed": True,
            "failure": None,
            "checks": {"mock_only": True},
            "config_sha256": probe.digest(route / "config.json"),
            "artifact_sha256": {
                path.relative_to(route).as_posix(): probe.digest(path)
                for path in route.rglob("*")
                if path.is_file()
            },
        },
    )
    (route / "audit.source.py").write_bytes(b"test-only mock audit source")
    save(
        route / "audit.json",
        {
            "valid": True,
            "checks": {"mock_only": True},
            "audit_snapshot_sha256": probe.digest(route / "audit.source.py"),
        },
    )
    return route


def test_public_diagnosis_replays_exact_sample_and_never_reads_private(
    public_capture, tmp_path, monkeypatch
):
    before = {
        path: probe.digest(path) for path in public_capture.rglob("*") if path.is_file()
    }
    read_bytes = Path.read_bytes

    def guard(path):
        if path.name.startswith("evaluator-"):
            raise AssertionError("public-only diagnosis accessed private evidence")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    output = tmp_path / "new-diagnosis"
    result = probe.probe(public_capture, output)
    assert all(result["checks"].values()) and len(result["records"]) == 1
    record = result["records"][0]
    assert record["direct_tracks_replayed"] and record["interval"] == [0, 1]
    assert len(record["comparisons"]) == 2 and not result["navigation_accepted"]
    assert all(probe.digest(path) == value for path, value in before.items())
    assert (output / "probe.source.py").read_bytes() == Path(
        probe.__file__
    ).read_bytes()
    with pytest.raises(FileExistsError):
        probe.probe(public_capture, output)


@pytest.mark.parametrize(
    "case", ["invalid_audit", "audit_source", "public_hash", "config_hash"]
)
def test_invalid_bound_input_rejects_before_output_allocation(
    public_capture, tmp_path, case
):
    name = "audit.json" if case in ("invalid_audit", "audit_source") else "report.json"
    path = public_capture / name
    data = json.loads(path.read_bytes())
    if case == "invalid_audit":
        data["valid"] = False
    elif case == "audit_source":
        data["audit_snapshot_sha256"] = "0" * 64
    elif case == "public_hash":
        data["artifact_sha256"]["keyframe-acquisition.json"] = "0" * 64
    else:
        data["config_sha256"] = "0" * 64
    save(path, data)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="intact independently audited"):
        probe.probe(public_capture, output)
    assert not output.exists()


def test_recomputed_direct_tracks_cannot_replace_original_sample(
    public_capture, tmp_path
):
    path = public_capture / "keyframe-acquisition.json"
    public = json.loads(path.read_bytes())
    public["acquisition"]["samples"][0]["novelty"]["tracks"]["seed_count"] += 1
    save(path, public)
    report = json.loads((public_capture / "report.json").read_bytes())
    report["artifact_sha256"][path.name] = probe.digest(path)
    save(public_capture / "report.json", report)
    output = tmp_path / "failed-diagnosis"
    with pytest.raises(ValueError, match="replay original sample"):
        probe.probe(public_capture, output)
    assert (output / "probe.source.py").exists() and not (
        output / "report.json"
    ).exists()


def test_late_public_mutation_discards_success_report(
    public_capture, tmp_path, monkeypatch
):
    compare = probe.compare

    def mutate(frames):
        result = compare(frames)
        path = public_capture / "frames/frame-00001.png"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(probe, "compare", mutate)
    output = tmp_path / "mutated-diagnosis"
    with pytest.raises(ValueError, match="input bytes changed"):
        probe.probe(public_capture, output)
    assert not (output / "report.json").exists()


def test_safe_detector_is_a_separate_first_image_namespace_not_old_id_filter(
    public_capture, tmp_path
):
    from safe_seed_detector import safe_seeds

    before = {
        path: probe.digest(path) for path in public_capture.rglob("*") if path.is_file()
    }
    result = probe.probe(
        public_capture, tmp_path / "safe-probe", safe_seed_controls=True
    )
    record = result["records"][0]
    masked = record["safe_seed_probe"]
    assert not masked["original_sample_replaced"] and not masked["map_fitted"]
    assert (
        not masked["navigation_accepted"] and not result["config"]["detector_promotion"]
    )
    with Image.open(public_capture / "frames/frame-00000.png") as first:
        seeds = safe_seeds(np.array(first))
    assert np.array_equal(masked["seed_pixels"], seeds)
    tracked = masked["diagnosis"]["tracks"]
    assert tracked["seed_count"] == len(seeds) == tracked["surviving_counts"][0]
    assert not any(
        event["reason"] == "seed_roi" for event in masked["diagnosis"]["rejections"]
    )
    assert record["direct_tracks_replayed"]
    assert all(probe.digest(path) == value for path, value in before.items())


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_untyped_seed_protocol_cannot_allocate_output(public_capture, tmp_path, flag):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="explicit boolean"):
        probe.probe(public_capture, output, safe_seed_controls=flag)
    assert not output.exists()
