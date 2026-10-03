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
        "schema": "runtime-camera-timing-v1",
        "seed": 201,
        "schedule": capture.SCHEDULE,
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
            json.dumps(row) for row in [{"kind": "instrumented", "hooks": 1}] + captures
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
    assert result["timing"]["samples"] == 66
    assert result["timing"]["max_telemetry_vs_capture_player_component_error"] == 0
    assert not result["navigation_accepted"]


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
