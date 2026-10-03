import copy
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="optional native camera image replay")
Image = pytest.importorskip("PIL.Image", reason="optional runtime camera capture")

from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    script = importlib.import_module("audit_runtime_camera")
    spec = importlib.util.spec_from_file_location(
        "capture_runtime_test_definition",
        ROOT / "environments/minestudio/capture_runtime_camera.py",
    )
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return script, capture


def sample(index):
    player = {
        "xpos": float(index),
        "ypos": 10.0,
        "zpos": 0.0,
        "yaw": 30.0,
        "pitch": 0.0,
    }
    previous = player | {"xpos": float(index - 1)}
    return {
        "kind": "capture",
        "sequence": index + 4,
        "tick_counter": index + 9,
        "width": 1280,
        "height": 720,
        "raw_rgb_sha256": f"hash-{index}",
        "partial_ticks": 0.0,
        "eye_height": 1.62,
        "sync_integrated_server": True,
        "disable_recorder": True,
        "bob_view": True,
        "fov_hand": 1.0,
        "previous_fov_hand": 1.0,
        "player_fov_modifier": 1.0,
        "camera_valid": True,
        "third_person": False,
        "player": player,
        "previous_player": previous,
        "camera_base": previous | {"ypos": 11.62, "yaw": 30.0},
        "render": {
            "sequence": index + 1,
            "partial_ticks": 0.0,
            "model_view": np.diag((-1, 1, -1, 1)).tolist(),
            "projection": [
                [0.8, 0, 0, 0],
                [0, 1.4, 0, 0],
                [0, 0, -1.001, -0.1],
                [0, 0, -1, 0],
            ],
            "base_position": [previous["xpos"], 11.62, 0],
            "shader_pack_loaded": False,
            "shadow_pass": False,
        },
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def diagnostic(scripts, tmp_path):
    script, capture = scripts
    route = tmp_path / "timing"
    route.mkdir()
    engine = tmp_path / "test-engine.jar"
    engine.write_bytes(b"test-only mock engine; not runtime proof")
    (route / "camera-audit.jar").write_bytes(b"test-only mock agent")
    sources = {
        "capture_runtime_camera.py": ROOT
        / "environments/minestudio/capture_runtime_camera.py",
        "CameraAudit.java": ROOT
        / "environments/minestudio/camera_audit_agent/CameraAudit.java",
        "MANIFEST.MF": ROOT / "environments/minestudio/camera_audit_agent/MANIFEST.MF",
        "probe-recorder.py": ROOT
        / "environments/minestudio/capture_controlled_rgb_motion.py",
        "position-audit.py": ROOT / "environments/minestudio/navigation_probe.py",
    } | {
        "package/" + path.relative_to(ROOT / "src/minecraft_scientist").as_posix(): path
        for path in sorted((ROOT / "src/minecraft_scientist").rglob("*.py"))
    }
    for name, path in sources.items():
        target = route / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name in ("sdk-entry.py", "sdk-launcher.py"):
        (route / "source-snapshot" / name).write_bytes(b"test-only SDK snapshot")
    config = {
        "schema": "runtime-camera-projection-v2",
        "seed": 201,
        "schedule": capture.SCHEDULE,
        "reference_indices": capture.REFERENCE_INDICES,
        "trial_intervals": capture.TRIAL_INTERVALS,
        "render_size": [1280, 720],
        "observation_size": [448, 448],
        "engine_sha256": script.digest(engine),
        "agent_sha256": script.digest(route / "camera-audit.jar"),
        "scope": "unit fixture; not live evidence",
        "source_sha256": {
            path.relative_to(route / "source-snapshot").as_posix(): script.digest(path)
            for path in (route / "source-snapshot").rglob("*")
            if path.is_file()
        },
    }
    save(route / "config.json", config)
    objects = tuple(
        ObjectObservation(
            object_id=f"anonymous-{index}", affordances=(ActionKind.MOVE,)
        )
        for index in range(4)
    ) + (
        ObjectObservation(
            object_id="anonymous-camera",
            affordances=(ActionKind.LOOK,),
            observable_features={"screen_row": 1, "screen_column": 2},
        ),
    )
    captures, native_hashes, public_hashes, positions = [], [], [], []
    for index in range(len(capture.SCHEDULE) + 1):
        native = np.full((720, 1280, 3), index, dtype=np.uint8)
        native[0] = index + 5
        public = cv2.resize(native, (448, 448), interpolation=cv2.INTER_LINEAR)
        for folder, data in (("frames", public), ("evaluator-native", native)):
            (route / folder).mkdir(exist_ok=True)
            Image.fromarray(data).save(route / folder / f"frame-{index:05d}.png")
        native_hashes.append(hashlib.sha256(native[::-1].tobytes()).hexdigest())
        public_hashes.append(hashlib.sha256(public.tobytes()).hexdigest())
        captures.append(sample(index) | {"raw_rgb_sha256": native_hashes[-1]})
        positions.append(sample(index)["player"])
    events = []
    for index, (kind, handle) in enumerate(capture.SCHEDULE):
        action = PrimitiveAction(
            kind=ActionKind(kind),
            ticks=1,
            target_id=f"anonymous-{handle}"
            if kind == "move"
            else "anonymous-camera"
            if kind == "look"
            else None,
        )
        event = WorldEvent(event_type="fixture", duration=1)
        states = [
            Observation(
                episode_id=0,
                step=index + offset,
                inventory={},
                agent_state={"frame_sha256": public_hashes[index + offset]},
                objects=objects,
            )
            for offset in (0, 1)
        ]
        events.append(
            TransitionRecord(
                transition_id=f"timing-{index}",
                context={},
                state_before=states[0],
                state_after=states[1],
                executed_action=action,
                intervention_compliance=True,
                observed_delta={},
                event=event,
            )
        )
    (route / "events.jsonl").write_text(
        "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
    )
    save(
        route / "evaluator-positions.json",
        {"positions": positions, "native_rgb_sha256": native_hashes},
    )
    (route / "evaluator-camera.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"kind": "instrumented", "target": "recorder", "hooks": 1},
                {"kind": "instrumented", "target": "world_renderer", "hooks": 1},
            ]
            + captures
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "completed": True,
        "failure": None,
        "navigation_accepted": False,
        "frame_count": len(public_hashes),
        "config_sha256": script.digest(route / "config.json"),
        "checks": {
            key: True
            for key in (
                "one_fresh_instrumented_launch",
                "complete_frames_actions_positions",
                "sdk_runtime_unchanged",
                "engine_agent_unchanged",
                "sources_unchanged",
                "config_unchanged",
            )
        },
        "artifact_sha256": {
            path.relative_to(route).as_posix(): script.digest(path)
            for path in list((route / "frames").glob("*.png"))
            + list((route / "evaluator-native").glob("*.png"))
            + [
                route / name
                for name in (
                    "events.jsonl",
                    "evaluator-positions.json",
                    "evaluator-camera.jsonl",
                )
            ]
        },
    }
    save(route / "report.json", report)
    return script, route, engine


def test_unique_whole_sequence_not_per_frame_nearest_pose(scripts):
    script, _ = scripts
    captures = [sample(index) for index in range(7)]
    start, aligned = script.match_snapshots(
        captures, [f"hash-{index}" for index in range(2, 6)]
    )
    assert start == 2 and aligned == captures[2:6]


@pytest.mark.parametrize("case", ["missing", "ambiguous", "gapped", "counter", "bool"])
def test_bad_or_ambiguous_camera_correspondence_rejected(scripts, case):
    script, _ = scripts
    captures, hashes = (
        [sample(index) for index in range(4)],
        [f"hash-{index}" for index in range(4)],
    )
    if case == "missing":
        hashes[1] = "missing"
    elif case == "ambiguous":
        captures += copy.deepcopy(captures)
    elif case == "gapped":
        captures.insert(2, sample(22))
    elif case == "counter":
        captures[2]["tick_counter"] += 1
    else:
        captures[2]["sequence"] = True
    with pytest.raises(ValueError):
        script.match_snapshots(captures, hashes)


def test_metrics_keep_camera_translation_and_orientation_separate(scripts):
    script, _ = scripts
    record = sample(3)
    record["previous_player"]["yaw"] = 0
    result = script.timing_metrics([record], [record["player"]])
    assert result["max_camera_vs_current_player_eye_distance"] == pytest.approx(1)
    assert result["max_camera_vs_previous_player_eye_distance"] < 1e-10
    assert result["max_camera_vs_current_player_angle_error_degrees"] == 0
    assert result["max_camera_vs_previous_player_angle_error_degrees"] == 30


@pytest.mark.parametrize(
    "key,value",
    [
        ("eye_height", float("nan")),
        ("partial_ticks", True),
        ("partial_ticks", 2),
        ("bob_view", 1),
        ("fov_hand", float("nan")),
    ],
)
def test_invalid_renderer_numbers_rejected(scripts, key, value):
    script, _ = scripts
    record = sample(3) | {key: value}
    with pytest.raises(ValueError):
        script.timing_metrics([record], [record["player"]])


def test_full_independent_audit_reproduces_pixels_and_matches_camera(diagnostic):
    script, route, engine = diagnostic
    result = script.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["matching_capture_start"] == 0
    assert result["timing"]["samples"] == 106
    assert result["timing"]["max_telemetry_vs_capture_player_component_error"] == 0
    assert not result["navigation_accepted"]


def private_guard(monkeypatch):
    read, text = Path.read_bytes, Path.read_text

    def guard(path):
        if path.name in ("evaluator-camera.jsonl", "evaluator-positions.json"):
            raise AssertionError("public inference opened private bytes")
        return read(path)

    def guard_text(path, *args, **kwargs):
        if path.name in ("evaluator-camera.jsonl", "evaluator-positions.json"):
            raise AssertionError("public inference opened private text")
        return text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guard)
    monkeypatch.setattr(Path, "read_text", guard_text)


def test_public_camera_validation_never_reads_private_bytes(diagnostic, monkeypatch):
    script, route, engine = diagnostic
    private_guard(monkeypatch)
    public = script.public_data(route, engine)
    assert public["valid"] and public["count"] == 106
    assert all(
        path.name not in ("evaluator-camera.jsonl", "evaluator-positions.json")
        for path in public["frozen"]
    )


@pytest.fixture
def rendered_probe(diagnostic, monkeypatch):
    from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

    _, route, engine = diagnostic
    probe = importlib.import_module("probe_rendered_motion")
    trace = []
    monkeypatch.setattr(
        probe,
        "calibrate_reference_camera",
        lambda images, size: {
            "camera_intrinsics": [[180, 0, 224], [0, 320, 224], [0, 0, 1]]
        },
    )

    def track(images, **kwargs):
        trace.append(("track", len(images)))
        return TemporalTracks(
            image_shape=(448, 448),
            frame_count=len(images),
            seed_count=0,
            surviving_counts=(0,) * len(images),
            track_ids=(),
            pixels=((),) * len(images),
            minimum_patch_correlations=(),
            chain_cycle_errors=(),
        )

    fit = probe.fit_joint_map

    def logged_fit(*args, **kwargs):
        trace.append(("fit", kwargs["bootstrap_slot"]))
        return fit(*args, **kwargs)

    monkeypatch.setattr(probe, "track_frame_chain", track)
    monkeypatch.setattr(probe, "fit_joint_map", logged_fit)
    return probe, route, engine, trace


def test_all_maps_frozen_before_future_and_no_private_inputs(
    rendered_probe, monkeypatch
):
    probe, route, engine, trace = rendered_probe
    private_guard(monkeypatch)
    result = probe.infer(route, engine)
    assert len(result["trials"]) == 2
    assert sum(kind == "fit" for kind, _ in trace) == 18
    first_future = next(
        index
        for index, (kind, count) in enumerate(trace)
        if kind == "track" and count > 11
    )
    assert all(kind != "fit" for kind, _ in trace[first_future:])
    for trial in result["trials"]:
        assert [item["map"]["bootstrap_slot"] for item in trial["candidates"]] == list(
            range(2, 11)
        )
        assert all(not item["map"]["supported"] for item in trial["candidates"])
    assert not result["navigation_accepted"]


def test_rendered_audit_replays_public_fit_before_any_private_access(
    rendered_probe, tmp_path, monkeypatch
):
    probe, route, engine, trace = rendered_probe
    experiment = tmp_path / "prediction"
    assert probe.run(route, engine, experiment)["completed"]
    trace.clear()
    actual_audit = probe.runtime.audit

    def audited(*args):
        assert sum(kind == "fit" for kind, _ in trace) == 18
        return actual_audit(*args)

    monkeypatch.setattr(probe.runtime, "audit", audited)
    result = probe.audit(route, engine, experiment)
    assert result["valid"] and all(result["checks"].values())
    assert sum(len(item["candidates"]) for item in result["evaluation"]) == 18
    assert all(
        candidate["past_motion"]["rendered"]["status"] == "unsupported_motion"
        for trial in result["evaluation"]
        for candidate in trial["candidates"]
    )


def test_forged_prediction_cannot_reach_renderer_truth(
    rendered_probe, tmp_path, monkeypatch
):
    probe, route, engine, _ = rendered_probe
    experiment = tmp_path / "prediction"
    probe.run(route, engine, experiment)
    result = json.loads((experiment / "report.json").read_bytes())
    result["prediction"]["trials"][0]["candidates"].pop()
    save(experiment / "report.json", result)
    private_guard(monkeypatch)
    result = probe.audit(route, engine, experiment)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["entire_public_inference_replayed"]


@pytest.mark.parametrize("name", ["evaluator-positions.json", "evaluator-camera.jsonl"])
def test_private_mutation_between_alignment_and_scoring_is_rejected(
    rendered_probe, tmp_path, monkeypatch, name
):
    probe, route, engine, _ = rendered_probe
    experiment = tmp_path / "prediction"
    probe.run(route, engine, experiment)
    original = probe.runtime.audit

    def mutate(*args):
        result = original(*args)
        path = route / name
        if name == "evaluator-positions.json":
            positions = json.loads(path.read_bytes())
            positions["positions"][0]["xpos"] += 0.01
            save(path, positions)
        else:
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(probe.runtime, "audit", mutate)
    result = probe.audit(route, engine, experiment)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["all_bytes_stable_through_scoring"]


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("available", [True, False])
def test_rejected_future_pose_diagnostics_never_become_policy_acceptance(
    scripts, accepted, available
):
    probe = importlib.import_module("probe_rendered_motion")
    past = {"rotation": np.eye(3).tolist(), "translation": [0, 0, -1]}
    later = (
        {"rotation": np.eye(3).tolist(), "translation": [0, 0, -2]}
        if available
        else None
    )
    future = {
        "frame_checks": [{"supported": accepted, "frame_slot": 12}],
        "poses": [later],
    }

    def score(pose, first, second):
        return {
            "supported": pose is not None,
            "pose": pose,
            "interval": [first, second],
        }

    record = probe.evaluate_future(past, future, 48, 58, score)[0]
    assert record["from_anchor"]["supported"] == (accepted and available)
    assert record["diagnostic_pose_from_anchor"]["supported"] == available
    assert not record["diagnostic_is_policy_acceptance"]
    assert record["diagnostic_pose_from_frozen_endpoint"]["interval"] == [58, 60]
    if available:
        assert record["diagnostic_pose_from_frozen_endpoint"]["pose"][
            "translation"
        ] == [0, 0, -1]


@pytest.fixture
def causal_probe(rendered_probe, monkeypatch):
    common, route, engine, _ = rendered_probe
    probe = importlib.import_module("probe_causal_windows")
    trace = []
    track = common.track_frame_chain

    def traced(images, **kwargs):
        trace.append((int(images[0][224, 224, 0]), int(images[-1][224, 224, 0])))
        return track(images, **kwargs)

    monkeypatch.setattr(probe, "track_frame_chain", traced)
    monkeypatch.setattr(
        probe, "calibrate_reference_camera", common.calibrate_reference_camera
    )
    return probe, route, engine, trace


def test_causal_replay_refreshes_before_later_input_and_never_reads_private(
    causal_probe, monkeypatch
):
    probe, route, engine, trace = causal_probe
    private_guard(monkeypatch)
    predicted = probe.infer(route, engine)
    assert probe.registered(predicted)
    assert sum(len(item["candidates"]) for item in predicted["trials"]) == 36
    assert [
        item["controller_decision"]["absolute_frame"] for item in predicted["trials"]
    ] == [60, 62, 91, 93]
    assert trace.index((50, 60)) < trace.index((50, 61)) < trace.index((50, 62))
    assert trace.index((81, 91)) < trace.index((81, 92)) < trace.index((81, 93))
    for first, second in ((0, 1), (2, 3)):
        assert (
            predicted["trials"][second]["snapshot"]["parent_sha256"]
            == predicted["trials"][first]["snapshot_sha256"]
        )
    assert all(
        item["controller_decision"]["instruction"] == "reject"
        for item in predicted["trials"]
    )


def test_causal_audit_retains_rejected_windows_with_separate_physical_scores(
    causal_probe, tmp_path
):
    probe, route, engine, _ = causal_probe
    experiment = tmp_path / "causal"
    assert probe.run(route, engine, experiment)["completed"]
    result = probe.audit(route, engine, experiment)
    assert result["valid"] and len(result["evaluation"]) == 4
    assert all(
        item["controller_gate"] == "reject"
        and not item["controller_motion"]["rendered"]["supported"]
        for item in result["evaluation"]
    )
    assert result["scope"] == probe.SCOPE
    assert not result["navigation_accepted"]


def test_causal_window_denominator_forgery_cannot_reach_truth(
    causal_probe, tmp_path, monkeypatch
):
    probe, route, engine, _ = causal_probe
    experiment = tmp_path / "causal"
    probe.run(route, engine, experiment)
    report = json.loads((experiment / "report.json").read_bytes())
    report["prediction"]["trials"].pop()
    save(experiment / "report.json", report)
    private_guard(monkeypatch)
    result = probe.audit(route, engine, experiment)
    assert not result["valid"] and result["evaluation"] is None


@pytest.fixture
def active_capture(diagnostic, scripts, monkeypatch):
    runtime, route, engine = diagnostic
    _, definition = scripts
    active = importlib.import_module("audit_active_causal_map")
    calibration = {
        "camera_intrinsics": [[180.0, 0, 224], [0, 320.0, 224], [0, 0, 1]],
        "scope": "unit-only public calibration mock",
    }
    monkeypatch.setattr(active, "calibrate_reference_camera", lambda *args: calibration)
    config = json.loads((route / "config.json").read_bytes())
    config.update(
        schema="active-causal-camera-projection-v2",
        schedule=[list(item) for item in definition.SCHEDULE[:48]],
        trial_intervals=[],
        active_seed=2000,
        active_policy=active.ActiveMapPolicy().model_dump(mode="json"),
    )
    save(route / "config.json", config)
    lines = (route / "events.jsonl").read_text(encoding="utf-8").splitlines()
    (route / "events.jsonl").write_text("\n".join(lines[:49]) + "\n", encoding="utf-8")
    for folder in ("frames", "evaluator-native"):
        for path in (route / folder).glob("*.png"):
            if int(path.stem.split("-")[-1]) >= 50:
                path.unlink()  # test-owned temporary fixture only
    private = json.loads((route / "evaluator-positions.json").read_bytes())
    save(
        route / "evaluator-positions.json",
        {key: value[:50] for key, value in private.items()},
    )
    rows = (route / "evaluator-camera.jsonl").read_text(encoding="utf-8").splitlines()
    (route / "evaluator-camera.jsonl").write_text(
        "\n".join(rows[:52]) + "\n", encoding="utf-8"
    )

    def execute(handle):
        assert handle == 0
        return active.PublicProbeStep(
            runtime.load_image(route / "frames/frame-00049.png"), 0, True
        )

    acquired = active.acquire_causal_map(
        runtime.load_image(route / "frames/frame-00048.png"),
        execute,
        calibration["camera_intrinsics"],
        anchor_frame=48,
    )
    assert (
        acquired["terminal"] == "support_aborted"
    )  # blank public fixture, not real runtime proof
    save(
        route / "active-acquisition.json",
        {"calibration": calibration, "acquisition": acquired},
    )
    report = json.loads((route / "report.json").read_bytes())
    report.update(frame_count=50, config_sha256=runtime.digest(route / "config.json"))
    report["checks"]["active_policy_completed"] = True
    report["artifact_sha256"] = {
        name: runtime.digest(route / name)
        for name in report["artifact_sha256"]
        if (route / name).exists()
    }
    report["artifact_sha256"]["active-acquisition.json"] = runtime.digest(
        route / "active-acquisition.json"
    )
    save(route / "report.json", report)
    return active, route, engine


def test_active_policy_replays_every_request_without_reading_truth(
    active_capture, monkeypatch
):
    active, route, engine = active_capture
    private_guard(monkeypatch)
    public = active.runtime.public_data(route, engine, registration=active.registration)
    assert public["valid"] and public["count"] == 50
    # A dynamic run cannot sneak through the original fixed registration.
    assert not active.runtime.public_data(route, engine)["valid"]


def test_active_camera_audit_retains_aborted_trajectory_without_accuracy_claim(
    active_capture,
):
    active, route, engine = active_capture
    result = active.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["policy_terminal"] == "support_aborted"
    assert result["executed_acquisition_actions"] == 1 and result["fit_count"] == 0
    assert result["evaluation"] is None and not result["navigation_accepted"]


def test_forged_active_gate_cannot_reach_private_evaluator(active_capture, monkeypatch):
    active, route, engine = active_capture
    extra = json.loads((route / "active-acquisition.json").read_bytes())
    extra["acquisition"]["terminal"] = "validated"
    save(route / "active-acquisition.json", extra)
    report = json.loads((route / "report.json").read_bytes())
    report["artifact_sha256"]["active-acquisition.json"] = active.runtime.digest(
        route / "active-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    result = active.audit(route, engine)
    assert not result["valid"]
    assert not result["camera_audit"]["checks"]["registered_schedule"]


@pytest.mark.parametrize(
    "name", ["active-acquisition.json", "evaluator-positions.json", "report.json"]
)
def test_active_bytes_changed_after_nested_audit_discard_scores(
    active_capture, monkeypatch, name
):
    active, route, engine = active_capture
    original = active.runtime.audit

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = route / name
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(active.runtime, "audit", mutate)
    result = active.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None


def test_mutation_during_analysis_invalidates_audit(diagnostic, monkeypatch):
    script, route, engine = diagnostic
    analyze = script.timing_metrics

    def mutate(*args):
        result = analyze(*args)
        (route / "config.json").write_bytes(b"changed during analysis")
        return result

    monkeypatch.setattr(script, "timing_metrics", mutate)
    result = script.audit(route, engine)
    assert not result["valid"]
    assert not result["checks"]["all_bytes_stable_through_analysis"]


@pytest.mark.parametrize("count", [0, 1])
def test_missing_private_sample_cannot_hide_behind_equal_counts(scripts, count):
    script, _ = scripts
    with pytest.raises(ValueError, match="counts differ"):
        script.timing_metrics([sample(1), sample(2)], [sample(1)["player"]] * count)


@pytest.mark.parametrize("case", ["source", "policy", "public", "ledger"])
def test_forgery_rejected_before_private_json_reads(diagnostic, monkeypatch, case):
    script, route, engine = diagnostic
    if case == "source":
        (route / "source-snapshot/CameraAudit.java").write_bytes(b"forged source")
    elif case == "policy":
        config = json.loads((route / "config.json").read_bytes())
        config["schedule"][0] = ["move", 0]
        save(route / "config.json", config)
    elif case == "public":
        path = route / "events.jsonl"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"frame_sha256":', '"xpos":', 1),
            encoding="utf-8",
        )
    else:
        report = json.loads((route / "report.json").read_bytes())
        report["artifact_sha256"].pop("evaluator-camera.jsonl")
        save(route / "report.json", report)
    read = Path.read_bytes
    text = Path.read_text

    def guarded_read(path):
        if path.name in ("evaluator-camera.jsonl", "evaluator-positions.json"):
            raise AssertionError("invalid public evidence reached private JSON")
        return read(path)

    def guarded_text(path, *args, **kwargs):
        if path.name in ("evaluator-camera.jsonl", "evaluator-positions.json"):
            raise AssertionError("invalid public evidence reached private JSON")
        return text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    result = script.audit(route, engine)
    assert not result["valid"] and result["timing"] is None


@pytest.mark.parametrize("previous", [None, "-Xmx512m"])
def test_isolated_agent_restores_launcher_and_environment(
    scripts, tmp_path, monkeypatch, previous
):
    _, capture = scripts
    if previous is None:
        monkeypatch.delenv("JAVA_TOOL_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("JAVA_TOOL_OPTIONS", previous)
    jar = tmp_path / "agent.jar"
    jar.write_bytes(b"test agent")
    seen = []

    class Instance:
        def _launch_minecraft(self):
            seen.append(os.environ["JAVA_TOOL_OPTIONS"])
            return 13

    original = Instance._launch_minecraft
    with capture.isolated_agent_launch(
        Instance, jar, tmp_path / "new-camera.jsonl"
    ) as launches:
        assert Instance()._launch_minecraft() == 13 and launches == [True]
        assert os.environ.get("JAVA_TOOL_OPTIONS") == previous
        with pytest.raises(RuntimeError, match="exactly one"):
            Instance()._launch_minecraft()
    assert Instance._launch_minecraft is original
    assert "-javaagent:" in seen[0] and os.environ.get("JAVA_TOOL_OPTIONS") == previous


def test_failed_jvm_launch_restores_environment_and_class(
    scripts, tmp_path, monkeypatch
):
    _, capture = scripts
    monkeypatch.delenv("JAVA_TOOL_OPTIONS", raising=False)
    jar = tmp_path / "agent.jar"
    jar.write_bytes(b"test agent")

    class Instance:
        def _launch_minecraft(self):
            raise RuntimeError("injected launch failure")

    original = Instance._launch_minecraft
    with (
        pytest.raises(RuntimeError, match="injected"),
        capture.isolated_agent_launch(Instance, jar, tmp_path / "new-camera.jsonl"),
    ):
        Instance()._launch_minecraft()
    assert (
        Instance._launch_minecraft is original and "JAVA_TOOL_OPTIONS" not in os.environ
    )


def test_cli_preserves_auditor_source_and_refuses_overwrite(
    scripts, tmp_path, monkeypatch
):
    script, _ = scripts
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        script, "audit", lambda *args: {"valid": True, "timing": {"fixture": True}}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script.__file__,
            str(tmp_path),
            "--engine-jar",
            str(tmp_path / "unused.jar"),
            "--output",
            str(output),
        ],
    )
    script.main()
    snapshot = output.with_suffix(".source.py")
    assert snapshot.read_bytes() == Path(script.__file__).read_bytes()
    assert json.loads(output.read_bytes())["audit_snapshot_sha256"] == script.digest(
        snapshot
    )
    with pytest.raises(FileExistsError):
        script.main()


def test_existing_snapshot_is_not_overwritten_even_without_report(
    scripts, tmp_path, monkeypatch
):
    script, _ = scripts
    output = tmp_path / "audit.json"
    snapshot = output.with_suffix(".source.py")
    snapshot.write_bytes(b"preserved old auditor")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script.__file__,
            str(tmp_path),
            "--engine-jar",
            str(tmp_path / "unused.jar"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        script.main()
    assert snapshot.read_bytes() == b"preserved old auditor"
