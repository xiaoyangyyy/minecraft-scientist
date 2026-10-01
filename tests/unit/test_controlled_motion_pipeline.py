import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image", reason="requires optional visual extra")

from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    common = importlib.import_module("controlled_motion_common")
    auditor = importlib.import_module("audit_controlled_rgb_motion")
    spec = importlib.util.spec_from_file_location(
        "controlled_capture_fixture",
        root / "environments/minestudio/capture_controlled_rgb_motion.py",
    )
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return common, auditor, capture


@pytest.fixture
def acquisition(tmp_path, modules):
    common, _, _ = modules
    config = MotionProbeConfig(observation_size=224)
    timeline = motion_probe_timeline(config)
    directory = tmp_path / "route"
    directory.mkdir()
    snapshot = directory / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for name, path in common.capture_sources().items():
        (snapshot / name).write_bytes(path.read_bytes())
        hashes[name] = common.digest(path)
    (snapshot / "sdk-entry.py").write_bytes(b"fixture SDK, not a simulator result")
    hashes["sdk-entry.py"] = common.digest(snapshot / "sdk-entry.py")
    manifest = {
        "config": config.model_dump(mode="json"),
        "render_size": list(config.render_size),
        "schedule": [
            item.model_dump(mode="json") for item in motion_probe_schedule(config)
        ],
        "pairs": motion_probe_pairs(config),
        "source_sha256": hashes,
    }
    (directory / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    frames = directory / "frames"
    frames.mkdir()
    array = np.zeros((224, 224, 3), dtype=np.uint8)
    png = frames / "frame-00000.png"
    Image.fromarray(array).save(png)
    image_bytes = png.read_bytes()
    count = timeline[-1]["end_frame"] + 1
    for slot in range(1, count):
        (frames / f"frame-{slot:05d}.png").write_bytes(image_bytes)
    image_hash, raw_hash = (
        common.digest(png),
        hashlib.sha256(array.tobytes()).hexdigest(),
    )
    references = directory / "goal-reference"
    references.mkdir()
    for index in range(12):
        (references / f"view-{index:02d}.png").write_bytes(image_bytes)
    objects = tuple(
        ObjectObservation(object_id=f"obj_{number}", affordances=(ActionKind.MOVE,))
        for number in range(4)
    ) + (
        ObjectObservation(
            object_id="cell",
            affordances=(ActionKind.LOOK,),
            observable_features={"screen_row": 1, "screen_column": 2},
        ),
    )
    observation = Observation(
        episode_id=0,
        step=0,
        inventory={},
        objects=objects,
        agent_state={"frame_sha256": raw_hash},
    )
    records = []
    for number, scheduled in enumerate(motion_probe_schedule(config)):
        target = (
            f"obj_{scheduled.motion_handle_index}"
            if scheduled.kind == "move"
            else "cell"
            if scheduled.kind == "look"
            else None
        )
        records.append(
            TransitionRecord(
                transition_id=f"event_{number}",
                context={},
                state_before=observation,
                state_after=observation,
                executed_action=PrimitiveAction(
                    kind=scheduled.kind, ticks=scheduled.ticks, target_id=target
                ),
                intervention_compliance=True,
                observed_delta={},
                event=WorldEvent(event_type="fixture", duration=scheduled.ticks),
            )
        )
    (directory / "events.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in records), encoding="utf-8"
    )
    report = {
        "suite": "controlled_rgb_motion_acquisition",
        "config_sha256": common.digest(directory / "config.json"),
        "frame_count": count,
        "frame_sha256": {f"frame-{slot:05d}.png": image_hash for slot in range(count)},
        "rgb_sha256": [raw_hash] * count,
        "reference_sha256": {f"view-{slot:02d}.png": image_hash for slot in range(12)},
        "reference_indices": [
            item["end_frame"]
            for item in timeline
            if item["scheduled"]["phase"] == "reference_settle"
        ],
        "timeline": timeline,
        "pairs": motion_probe_pairs(config),
        "gate": {"passed": True},
        "failure": None,
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory, report


def test_complete_public_acquisition_verifies_without_evaluator(modules, acquisition):
    common, _, _ = modules
    directory, _ = acquisition
    config, _, checks = common.validate_acquisition(directory)
    assert config.observation_size == 224 and all(checks.values())
    assert "evaluator-audit.json" not in common.public_inputs(directory)
    assert not (directory / "evaluator-audit.json").exists()


def test_missing_intermediate_frame_ledger_stops_before_rgb_decode(
    modules, acquisition, monkeypatch
):
    common, _, _ = modules
    directory, report = acquisition
    report["frame_sha256"].pop("frame-00070.png")
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        common,
        "rgb",
        lambda path: pytest.fail("incomplete ledger entered RGB decoding"),
    )
    with pytest.raises(ValueError, match="incomplete"):
        common.validate_acquisition(directory)


def test_selected_motion_pairs_cannot_be_silently_dropped(modules, acquisition):
    common, _, _ = modules
    directory, report = acquisition
    report["pairs"].pop(0)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid public acquisition"):
        common.validate_acquisition(directory)


def test_opaque_move_handle_is_checked_against_preregistered_schedule(
    modules, acquisition
):
    common, _, _ = modules
    directory, _ = acquisition
    path = directory / "events.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[25]["executed_action"]["target_id"] = "obj_3"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="scheduled_targets_verified"):
        common.validate_acquisition(directory)


def test_privileged_log_field_is_rejected(modules, acquisition):
    common, _, _ = modules
    directory, _ = acquisition
    path = directory / "events.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["context"]["xpos"] = 1
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="public_events_clean"):
        common.validate_acquisition(directory)


def test_failed_acquisition_never_reaches_geometry_or_evaluator(
    modules, acquisition, monkeypatch
):
    _, auditor, _ = modules
    directory, report = acquisition
    report["gate"]["passed"] = False
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        auditor,
        "infer_pair",
        lambda *args: pytest.fail("failed acquisition entered geometry"),
    )
    monkeypatch.setattr(
        auditor,
        "score_relative_motion",
        lambda *args, **kwargs: pytest.fail("failed acquisition entered evaluator"),
    )
    result = auditor.audit(directory, directory / "nonexistent-experiment")
    assert result["valid"] is False and result["evaluation"] is None


def test_focal_failure_is_an_explicit_stop_not_a_default_guess(
    modules, acquisition, monkeypatch
):
    common, _, _ = modules
    directory, report = acquisition
    monkeypatch.setattr(common, "_opencv", lambda: object())
    monkeypatch.setattr(common, "_features", lambda *args: (np.empty((0, 2)), None))
    monkeypatch.setattr(common, "_calibrate_focal", lambda *args: (None, 0, None))
    with pytest.raises(ValueError, match="calibration rejected"):
        common.inference_config(
            directory, MotionProbeConfig(observation_size=224), report
        )


def test_frame_capture_is_bounded_and_copies_public_rgb(modules):
    _, _, capture = modules
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    frames = []
    wrapped = capture.FrameCapture(object(), frames)
    wrapped._capture({"image": source})
    source[:] = 255
    assert np.max(frames[0]) == 0
    frames[:] = [frames[0]] * 500
    with pytest.raises(RuntimeError, match="budget"):
        wrapped._capture(source)
    with pytest.raises(RuntimeError, match="privileged"):
        capture.ProbeRecorder().perceive(source, {"location_stats": {}}, seed=0)


def _auditable_experiment(modules, acquisition, monkeypatch):
    common, auditor, _ = modules
    route, acquisition_report = acquisition
    experiment = route.parent / "experiment"
    experiment.mkdir()
    sources = common.runtime_sources()
    snapshot = experiment / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    base = {
        "seed": 0,
        "pairs": acquisition_report["pairs"],
        "evaluation_thresholds": {
            "minimum_translation_blocks": 0.15,
            "maximum_direction_error_degrees": 20.0,
            "maximum_rotation_error_degrees": 5.0,
        },
        "scope": "unit fixture; no real inference",
    }
    config = base | {
        "input_sha256": {
            name: common.digest(path)
            for name, path in common.public_inputs(route).items()
        },
        "source_sha256": {name: common.digest(path) for name, path in sources.items()},
    }
    (experiment / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def infer(*args):
        pair = args[1]
        return {
            "pair": pair,
            "tracks": {"track_ids": []},
            "endpoint_tracks": {"track_ids": []},
            "motion": {"supported": False},
            "endpoint_motion": {"supported": False},
        }

    predictions = {
        "config_sha256": common.digest(experiment / "config.json"),
        "completed": True,
        "input_sources_unchanged": True,
        "failure": None,
        "navigation_accepted": False,
        "results": [infer(route, pair) for pair in base["pairs"]],
    }
    (experiment / "report.json").write_text(json.dumps(predictions), encoding="utf-8")
    positions = [
        {"xpos": 0, "ypos": 0, "zpos": 0, "yaw": 0, "pitch": 0}
    ] * acquisition_report["frame_count"]
    (route / "evaluator-audit.json").write_text(
        json.dumps({"positions": positions}), encoding="utf-8"
    )
    monkeypatch.setattr(auditor, "inference_config", lambda *args: base)
    monkeypatch.setattr(auditor, "infer_pair", infer)
    return route, experiment, config, predictions, positions


def _forbid_scoring(monkeypatch, auditor):
    def guard(*args, **kwargs):
        pytest.fail("invalid evidence entered scoring")

    # Pytest assertion-rewrite caches are shared by Windows and WSL. Keep the
    # intentionally hashed mock's filename native to this executing runtime.
    guard.__code__ = guard.__code__.replace(co_filename=str(Path(__file__).resolve()))
    monkeypatch.setattr(auditor, "score_relative_motion", guard)


def test_complete_replay_can_score_all_trials_without_filtering(
    modules, acquisition, monkeypatch
):
    _, auditor, _ = modules
    route, experiment, *_ = _auditable_experiment(modules, acquisition, monkeypatch)
    result = auditor.audit(route, experiment)
    assert result["valid"] is True and len(result["evaluation"]["scores"]) == 30
    assert result["evaluation"]["navigation_accepted"] is False
    # This fixture is stationary even for MOVE intent; intent cannot make truth.
    assert result["evaluation"]["groups"]["moving"]["actual_controls"] == 8


def test_nonfinite_or_misaligned_telemetry_stops_before_scoring(
    modules, acquisition, monkeypatch
):
    _, auditor, _ = modules
    route, experiment, _, _, positions = _auditable_experiment(
        modules, acquisition, monkeypatch
    )
    _forbid_scoring(monkeypatch, auditor)
    for changed in (
        positions[:-1],
        [dict(item, xpos=float("nan")) for item in positions],
    ):
        (route / "evaluator-audit.json").write_text(
            json.dumps({"positions": changed}), encoding="utf-8"
        )
        result = auditor.audit(route, experiment)
        assert (
            result["valid"] is False
            and result["checks"]["evaluator_alignment"] is False
        )
        assert result["evaluation"] is None


def test_forged_geometry_replay_cannot_reach_evaluator(
    modules, acquisition, monkeypatch
):
    _, auditor, _ = modules
    route, experiment, _, predictions, _ = _auditable_experiment(
        modules, acquisition, monkeypatch
    )
    predictions["results"][0]["endpoint_motion"]["supported"] = True
    (experiment / "report.json").write_text(json.dumps(predictions), encoding="utf-8")
    _forbid_scoring(monkeypatch, auditor)
    result = auditor.audit(route, experiment)
    assert (
        result["valid"] is False and result["checks"]["all_geometry_replayed"] is False
    )
    assert result["evaluation"] is None


def test_missing_input_manifest_stops_before_geometry(
    modules, acquisition, monkeypatch
):
    common, auditor, _ = modules
    route, experiment, config, predictions, _ = _auditable_experiment(
        modules, acquisition, monkeypatch
    )
    config["input_sha256"].pop("frames/frame-00070.png")
    (experiment / "config.json").write_text(json.dumps(config), encoding="utf-8")
    predictions["config_sha256"] = common.digest(experiment / "config.json")
    (experiment / "report.json").write_text(json.dumps(predictions), encoding="utf-8")
    monkeypatch.setattr(
        auditor,
        "infer_pair",
        lambda *args: pytest.fail("incomplete input manifest entered geometry"),
    )
    result = auditor.audit(route, experiment)
    assert result["valid"] is False and result["checks"]["complete_input_set"] is False
