import hashlib
import importlib
import json
import marshal
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="optional public-pixel diagnosis")
Image = pytest.importorskip("PIL.Image", reason="optional public-pixel diagnosis")

from minecraft_scientist.exploration.temporal_tracks import track_frame_chain
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
    VisibilityPolicy,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    script = importlib.import_module("diagnose_visibility_track_losses")
    route = tmp_path / "capture"
    frame_dir = route / "frames"
    frame_dir.mkdir(parents=True)
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    for index in range(17):
        Image.fromarray(pixels).save(frame_dir / f"frame-{index:05d}.png")
    policy = VisibilityPolicy(landmark_protection="first_supported_map_v1")
    pairs = [[0, 1], [0, 2], [0, 3], [1, 0]]
    source = Path(track_frame_chain.__code__.co_filename)
    snapshot = route / "source-snapshot/package/exploration/temporal_tracks.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    manifest = {
        "schema": "visibility-rgb-acquisition-v2",
        "policy": policy.model_dump(mode="json"),
        "handle_pairs": pairs,
        "source_sha256": {
            "package/exploration/temporal_tracks.py": script.digest(snapshot)
        },
    }
    save(route / "config.json", manifest)
    tracks = track_frame_chain([pixels, pixels])
    trials = []
    for number in range(8):
        handles = pairs[number % 4]
        decision = VisibilityController(tuple(handles), policy=policy).observe(tracks)
        trials.append(
            {
                "anchor_frame": number * 2,
                "end_frame": number * 2 + 1,
                "repetition": number // 4,
                "handles": handles,
                "decisions": [decision.model_dump(mode="json")],
                "prefix_tracks": [tracks.model_dump(mode="json")],
                "landmark_attempts": [],
            }
        )
    raw = {
        "failure": None,
        "passed": True,
        "config_sha256": script.digest(route / "config.json"),
        "checks": {
            name: True
            for name in (
                "execution_completed",
                "action_frame_alignment",
                "bounded_frames",
                "telemetry_alignment",
                "preregistration_unchanged",
                "input_sources_unchanged",
                "public_log_clean",
            )
        },
        "frame_count": 17,
        "trials": trials,
        "frame_sha256": {
            path.name: script.digest(path) for path in frame_dir.iterdir()
        },
        "rgb_sha256": [hashlib.sha256(pixels.tobytes()).hexdigest()] * 17,
    }
    save(route / "report.json", raw)
    (route / "events.jsonl").write_text("", encoding="utf-8")
    return script, route, raw


def test_diagnosis_replays_all_original_trials_without_private_inputs(
    capture, monkeypatch
):
    script, route, _ = capture
    read = Path.read_bytes

    def guarded(path):
        if "evaluator" in path.name:
            raise AssertionError("diagnosis cannot consume private pose scores")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    before = {
        path.relative_to(route).as_posix(): read(path)
        for path in route.rglob("*")
        if path.is_file()
    }
    result = script.diagnose(route)
    assert {
        path.relative_to(route).as_posix(): read(path)
        for path in route.rglob("*")
        if path.is_file()
    } == before
    assert result["prefix_count"] == 8 and len(result["trials"]) == 8
    assert (
        result["historical_tracks_unchanged"]
        and result["historical_decisions_replayed"]
    )
    assert not result["navigation_accepted"]


@pytest.mark.parametrize(
    "change",
    (
        "png",
        "snapshot",
        "trial",
        "order",
        "schedule",
        "tracks",
        "decision",
        "rgb",
        "gate",
    ),
)
def test_tampered_public_evidence_cannot_become_a_diagnosis(capture, change):
    script, route, raw = capture
    if change == "png":
        (route / "frames/frame-00000.png").write_bytes(b"wrong image")
    elif change == "snapshot":
        (route / "source-snapshot/package/exploration/temporal_tracks.py").write_bytes(
            b"wrong source"
        )
    elif change == "trial":
        raw["trials"].pop()
    elif change == "order":
        raw["trials"][1]["anchor_frame"] = 0
    elif change == "schedule":
        raw["trials"][0]["decisions"][0]["tick"] = 2
    elif change == "tracks":
        raw["trials"][0]["prefix_tracks"][0]["seed_count"] = 1
    elif change == "decision":
        raw["trials"][0]["decisions"][0]["projected_edge_fraction"] = 0.1
    elif change == "rgb":
        raw["rgb_sha256"][0] = "0" * 64
    elif change == "gate":
        raw["checks"]["public_log_clean"] = False
    save(route / "report.json", raw)
    with pytest.raises(ValueError):
        script.diagnose(route)


def test_forged_bytecode_cache_is_neither_executed_nor_changed(capture):
    script, route, _ = capture
    snapshot = route / "source-snapshot/package/exploration/temporal_tracks.py"
    cache_dir = snapshot.parent / "__pycache__"
    cache_dir.mkdir()
    cache = cache_dir / f"temporal_tracks.{sys.implementation.cache_tag}.pyc"
    # A valid matching timestamp/size header would fool a normal source loader.
    payload = (
        importlib.util.MAGIC_NUMBER
        + struct.pack("<III", 0, int(snapshot.stat().st_mtime), snapshot.stat().st_size)
        + marshal.dumps(
            compile(
                "raise RuntimeError('forged bytecode executed')", str(snapshot), "exec"
            )
        )
    )
    cache.write_bytes(payload)
    result = script.diagnose(route)
    assert result["historical_tracks_unchanged"]
    assert cache.read_bytes() == payload
