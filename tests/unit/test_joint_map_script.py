import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="requires optional joint RGB geometry")
pytest.importorskip("PIL.Image", reason="requires optional public RGB pipeline")

from test_visibility_maps_pipeline import adaptive_capture, shadow_capture  # noqa: F401


@pytest.fixture
def capture(request, monkeypatch):
    _, route = request.getfixturevalue("shadow_capture")
    script = importlib.import_module("probe_joint_maps")
    raw = json.loads((route / "report.json").read_bytes())
    monkeypatch.setattr(
        script,
        "calibrate_reference_camera",
        lambda images, size: raw["landmark_calibration"],
    )
    return script, route


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_all_eight_original_aborts_retained_and_public_inference_never_reads_private(
    capture, monkeypatch
):
    script, route = capture
    read = Path.read_bytes

    def guarded(path):
        assert path.name != "evaluator-audit.json"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    output = route.parent / "joint-predictions"
    result = script.run(route, output)
    assert result["completed"] and result["input_sources_unchanged"]
    assert [record["trial"] for record in result["results"]] == list(range(8))
    assert all(
        record["source_aborted"] and not record["candidates"]
        for record in result["results"]
    )
    with pytest.raises(FileExistsError):
        script.run(route, output)


def test_independent_public_replay_completes_before_first_privileged_read(
    capture, monkeypatch
):
    script, route = capture
    output = route.parent / "joint-predictions"
    assert script.run(route, output)["completed"]
    read, infer = Path.read_bytes, script.infer
    replayed = []

    def replay(*args):
        results = infer(*args)
        replayed.extend(results)
        return results

    def guarded(path):
        if path.name == "evaluator-audit.json":
            assert len(replayed) == 8
        return read(path)

    monkeypatch.setattr(script, "infer", replay)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = script.audit(route, output)
    assert result["valid"] and len(result["evaluation"]) == 8
    assert not result["navigation_accepted"] and all(result["checks"].values())


@pytest.mark.parametrize(
    "change",
    [
        "missing_frame",
        "rgb",
        "checks",
        "shadow_boundary",
        "trial_count",
        "snapshot",
        "private_public_log",
    ],
)
def test_forged_historical_public_inputs_reject_without_private_reads(
    capture, monkeypatch, change
):
    script, route = capture
    raw = json.loads((route / "report.json").read_bytes())
    if change == "missing_frame":
        (route / "frames/frame-00001.png").unlink()
    elif change == "rgb":
        raw["rgb_sha256"][1] = "0" * 64
    elif change == "checks":
        raw["checks"] = {}
    elif change == "shadow_boundary":
        raw["trials"][0]["second_leg_shadow"]["freeze_frame"] += 1
    elif change == "trial_count":
        raw["trials"].pop()
    elif change == "snapshot":
        snapshot = next((route / "source-snapshot").rglob("*.py"))
        snapshot.write_bytes(b"invalid historical source\n")
    else:
        with (route / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('"xpos": 1\n')
    save(route / "report.json", raw)
    read = Path.read_bytes

    def guarded(path):
        assert path.name != "evaluator-audit.json"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = script.run(route, route.parent / "joint-predictions")
    assert not result["completed"] and result["failure"]


@pytest.mark.parametrize(
    "change",
    ["omit_trial", "invent_map", "policy", "snapshot", "source_set", "input_set"],
)
def test_changed_prediction_or_provenance_prevents_private_scoring(
    capture, monkeypatch, change
):
    script, route = capture
    output = route.parent / "joint-predictions"
    assert script.run(route, output)["completed"]
    config = json.loads((output / "config.json").read_bytes())
    report = json.loads((output / "report.json").read_bytes())
    if change == "omit_trial":
        report["results"].pop()
    elif change == "invent_map":
        report["results"][0]["candidates"].append(
            {"map": {"supported": True}, "future": None}
        )
    elif change == "policy":
        config["seeds"][0] += 1
    elif change == "snapshot":
        next((output / "source-snapshot").rglob("*.py")).write_bytes(b"forged source\n")
    elif change == "source_set":
        config["source_sha256"].pop(next(iter(config["source_sha256"])))
    else:
        config["input_sha256"].pop(next(iter(config["input_sha256"])))
    save(output / "config.json", config)
    report["config_sha256"] = script.digest(output / "config.json")
    save(output / "report.json", report)
    read = Path.read_bytes

    def guarded(path):
        assert path.name != "evaluator-audit.json"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert not script.audit(route, output)["valid"]


def test_scoring_mutation_invalidates_completed_audit(capture, monkeypatch):
    script, route = capture
    output = route.parent / "joint-predictions"
    assert script.run(route, output)["completed"]
    reading = Path.read_bytes
    changed = []

    def guarded(path):
        data = reading(path)
        if path.name == "evaluator-audit.json" and not changed:
            changed.append(True)
            (route / "events.jsonl").write_text("corrupt\n", encoding="utf-8")
        return data

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = script.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_relative_pose_composes_world_to_camera_transforms(capture):
    import numpy as np
    from test_joint_multiview_map import fixture_tracks

    script, _ = capture
    _, rotations, translations = fixture_tracks()
    first = {"rotation": rotations[3].tolist(), "translation": translations[3].tolist()}
    second = {
        "rotation": rotations[9].tolist(),
        "translation": translations[9].tolist(),
    }
    relative = script.relative_pose(first, second)
    point = np.array((1, 0.2, 6))
    first_camera = rotations[3] @ point + translations[3]
    second_camera = rotations[9] @ point + translations[9]
    np.testing.assert_allclose(
        np.array(relative["rotation"]) @ first_camera + relative["translation"],
        second_camera,
    )


def test_every_bootstrap_is_frozen_before_any_future_tracking(capture, monkeypatch):
    import numpy as np
    from test_joint_multiview_map import CAMERA, fixture_tracks, prefix

    script, route = capture
    full, _, _ = fixture_tracks(frames=11)
    trials = [
        {
            "anchor_frame": 20 * number,
            "end_frame": 20 * number + 6,
            "terminal": "finished",
            "second_leg_shadow": {
                "terminal": "completed",
                "end_frame": 20 * number + 10,
            },
        }
        for number in range(8)
    ]
    images = [np.full((1, 1, 3), number, dtype=np.uint8) for number in range(151)]
    monkeypatch.setattr(
        script, "public_data", lambda route: ({"trials": trials}, images, CAMERA)
    )
    calls = []
    fit = script.fit_joint_map

    def fitter(*args, **kwargs):
        calls.append(("fit", kwargs["bootstrap_slot"]))
        return fit(*args, **kwargs)

    def tracking(frames, config):
        number = int(frames[0][0, 0, 0]) // 20
        if len(frames) > 7:
            assert calls.count(("fit", 2)) == number + 1
            assert calls[-5:] == [("fit", offset) for offset in range(2, 7)]
        return prefix(full, len(frames))

    monkeypatch.setattr(script, "fit_joint_map", fitter)
    monkeypatch.setattr(script, "track_frame_chain", tracking)
    results = script.infer(route)
    assert len(results) == 8
    assert all(len(record["candidates"]) == 5 for record in results)
    assert all(
        len(record["past_prefixes"]) == 6 and len(record["future_prefixes"]) == 4
        for record in results
    )
