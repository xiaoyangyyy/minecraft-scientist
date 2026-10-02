# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
import test_controlled_motion_pipeline as fixtures
from test_compound_motion_pipeline import compound_acquisition, compound_experiment
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline

from minecraft_scientist.exploration.compound_motion import compound_bindings


@pytest.fixture
def active_experiment(compound_experiment, pipeline, modules, monkeypatch, tmp_path):
    canonical, _, _ = modules
    _, producer, _ = pipeline
    planner = importlib.import_module("plan_active_rgb_motion")
    capture = importlib.import_module("active_capture_common")
    active = importlib.import_module("active_motion_pipeline")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    plan = tmp_path / "plan"
    assert planner.build_plan(*compound_experiment, plan)["completed"]
    requested = planner.validate_plan(plan)
    parent = tmp_path / "fresh"
    parent.mkdir()
    monkeypatch.setattr(
        fixtures, "capture_sources", canonical.capture_sources, raising=False
    )
    with (
        compound_bindings(
            fixtures, source_function="capture_sources", wrapper=capture.wrapper_path()
        ),
        capture.raw_bindings(canonical, source_function="capture_sources"),
        monkeypatch.context() as local,
    ):
        local.setattr(fixtures, "MotionProbeConfig", lambda **kwargs: requested)
        route, report = fixtures.acquisition.__wrapped__(parent, modules)
    with capture.raw_bindings(canonical, source_function="capture_sources"):
        sources = canonical.capture_sources()
    profile = route.with_name(route.name + ".active-profile")
    profile.mkdir()
    snapshot = profile / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    manifest = {
        "plan_directory": plan.relative_to(tmp_path).as_posix(),
        "plan_sha256": {
            "plan_config": canonical.digest(plan / "config.json"),
            "plan_result": canonical.digest(plan / "plan.json"),
        },
        "capture_config": requested.model_dump(mode="json"),
        "source_sha256": {name: canonical.digest(path) for name, path in sources.items()},
    }
    (profile / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    (profile / "report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": dict.fromkeys(active.CHECK_NAMES, True),
                "failure": None,
                "config_sha256": canonical.digest(profile / "config.json"),
                "constructor_calls": [requested.model_dump(mode="json")],
                "capture_config_sha256": canonical.digest(route / "config.json"),
                "capture_report_sha256": canonical.digest(route / "report.json"),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "active"
    with active.active_pipeline(producer):
        result = producer.run(route, output)
    assert result["completed"] and len(result["results"]) == 46
    # Stationary synthetic telemetry verifies ordering, never live performance.
    (route / "evaluator-audit.json").write_text(
        json.dumps(
            {
                "positions": [{"xpos": 0, "ypos": 0, "zpos": 0, "yaw": 0, "pitch": 0}]
                * report["frame_count"]
            }
        ),
        encoding="utf-8",
    )
    return route, output, compound_experiment[0]


def test_active_replays_past_plan_and_all_fresh_prefixes_before_private_read(
    active_experiment, pipeline, monkeypatch
):
    route, output, previous = active_experiment
    _, _, base = pipeline
    planner = importlib.import_module("plan_active_rgb_motion")
    auditor = importlib.import_module("audit_active_rgb_motion")
    replay, planned, private = [], [], []
    original_infer, original_plan, original_read = (
        base.infer_pair,
        planner.replay_prior,
        Path.read_bytes,
    )

    def inference(*args):
        result = original_infer(*args)
        replay.append(result)
        return result

    def planning(*args):
        result = original_plan(*args)
        planned.append(args)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert path.parent == route and path.parent != previous
            assert len(replay) == 46 and len(planned) == 1
            private.append(path)
        return original_read(path)

    monkeypatch.setattr(base, "infer_pair", inference)
    monkeypatch.setattr(planner, "replay_prior", planning)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert result["valid"] and private and len(result["checks"]) == 42
    assert result["evaluation"]["observed_noncollinear_trials"] == 0


@pytest.mark.parametrize("tamper", ["plan", "source", "constructor"])
def test_active_profile_tampering_rejects_before_any_private_read(
    active_experiment, monkeypatch, tamper
):
    route, output, _ = active_experiment
    profile = route.with_name(route.name + ".active-profile")
    if tamper == "source":
        (profile / "source-snapshot" / "active_parallax.py").write_text("tampered")
    else:
        path = profile / ("config.json" if tamper == "plan" else "report.json")
        content = json.loads(path.read_bytes())
        if tamper == "plan":
            content["plan_sha256"]["plan_result"] = "0" * 64
        else:
            content["constructor_calls"] = []
        path.write_text(json.dumps(content), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_active_rgb_motion").audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_active_prior_bytes_stay_frozen_through_final_private_scoring(
    active_experiment, pipeline, monkeypatch
):
    route, output, previous = active_experiment
    _, _, base = pipeline
    original = base.audit

    def changed(*args):
        result = original(*args)
        assert result["valid"]
        (previous / "frames/frame-00000.png").write_bytes(b"changed after replay")
        return result

    monkeypatch.setattr(base, "audit", changed)
    result = importlib.import_module("audit_active_rgb_motion").audit(route, output)
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["active_prior_profile_sources_stable"] is False


def test_active_pipeline_and_source_factory_restore_on_nested_failure(modules):
    canonical, _, _ = modules
    active = importlib.import_module("active_motion_pipeline")
    keyframes = importlib.import_module("keyframe_motion_common")
    capture = importlib.import_module("active_capture_common")
    module = SimpleNamespace(
        validate_acquisition=canonical.validate_acquisition,
        configuration=keyframes.configuration,
        runtime_sources=keyframes.runtime_sources,
    )
    before = vars(module).copy()
    with pytest.raises(RuntimeError), active.active_pipeline(module):
        with pytest.raises(ValueError), active.active_pipeline(module):
            pytest.fail("nested active inference accepted")
        raise RuntimeError("injected")
    assert vars(module) == before
    factory = canonical.capture_sources
    with pytest.raises(RuntimeError), capture.raw_bindings(
        canonical, source_function="capture_sources"
    ):
        with pytest.raises(ValueError), capture.raw_bindings(
            canonical, source_function="capture_sources"
        ):
            pytest.fail("nested active capture accepted")
        raise RuntimeError("injected")
    assert canonical.capture_sources is factory
