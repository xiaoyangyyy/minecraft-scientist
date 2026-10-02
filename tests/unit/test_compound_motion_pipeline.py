# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
import test_controlled_motion_pipeline as fixtures
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline

from minecraft_scientist.exploration.compound_motion import compound_bindings


@pytest.fixture
def compound_acquisition(request, modules, monkeypatch):
    canonical, _, _ = modules
    common = importlib.import_module("compound_motion_common")
    monkeypatch.setattr(
        fixtures, "capture_sources", canonical.capture_sources, raising=False
    )
    with (
        compound_bindings(
            fixtures, source_function="capture_sources", wrapper=common.wrapper_path()
        ),
        compound_bindings(
            canonical, source_function="capture_sources", wrapper=common.wrapper_path()
        ),
    ):
        return request.getfixturevalue("acquisition")


@pytest.fixture
def compound_experiment(compound_acquisition, pipeline, tmp_path):
    common = importlib.import_module("compound_motion_common")
    route, report = compound_acquisition
    _, producer, _ = pipeline
    output = tmp_path / "compound"
    with common.compound_pipeline(producer):
        result = producer.run(route, output)
    assert result["completed"] and len(result["results"]) == 46
    # Synthetic stationary telemetry is an ordering fixture, not accuracy evidence.
    (route / "evaluator-audit.json").write_text(
        json.dumps(
            {
                "positions": [{"xpos": 0, "ypos": 0, "zpos": 0, "yaw": 0, "pitch": 0}]
                * report["frame_count"]
            }
        ),
        encoding="utf-8",
    )
    return route, output


def test_compound_public_validation_without_any_private_file(compound_acquisition):
    common = importlib.import_module("compound_motion_common")
    route, report = compound_acquisition
    config, _, checks = common.validate_acquisition(route)
    assert config.sequence_kind == "paired-controls-v1" and all(checks.values())
    assert report["frame_count"] == 225 and len(report["pairs"]) == 46
    assert not (route / "evaluator-audit.json").exists()


@pytest.mark.parametrize("tamper", ["marker", "pair", "handle", "source"])
def test_compound_public_changes_reject_before_private_reads(
    compound_acquisition, monkeypatch, tamper
):
    common = importlib.import_module("compound_motion_common")
    route, _ = compound_acquisition
    config_path, report_path = route / "config.json", route / "report.json"
    manifest, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    if tamper == "marker":
        manifest["config"].pop("sequence_kind")
    elif tamper == "pair":
        manifest["pairs"].pop(0)
        report["pairs"].pop(0)
    elif tamper == "source":
        (route / "source-snapshot" / "compound_motion.py").write_text("tampered")
    else:
        events_path = route / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        events[26]["executed_action"]["target_id"] = "obj_0"
        events_path.write_text("".join(json.dumps(item) + "\n" for item in events))
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    canonical = importlib.import_module("controlled_motion_common")
    report["config_sha256"] = canonical.digest(config_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    with pytest.raises(ValueError):
        common.validate_acquisition(route)


def test_compound_replays_all_public_prefixes_before_any_private_read(
    compound_experiment, pipeline, monkeypatch
):
    route, output = compound_experiment
    _, _, base = pipeline
    auditor = importlib.import_module("audit_compound_rgb_motion")
    replay, private = [], []
    original_infer, original_read = base.infer_pair, Path.read_bytes

    def inference(*args):
        result = original_infer(*args)
        replay.append(result)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert len(replay) == 46
            private.append(path)
        return original_read(path)

    monkeypatch.setattr(base, "infer_pair", inference)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert result["valid"] and private and len(replay) == 46
    assert len(result["evaluation"]["physical_turns"]) == 8
    assert result["evaluation"]["observed_noncollinear_trials"] == 0
    assert all(
        item["angle_degrees"] is None for item in result["evaluation"]["physical_turns"]
    )


def test_compound_pipeline_restores_aliases_on_failure(modules):
    common = importlib.import_module("compound_motion_common")
    canonical = importlib.import_module("controlled_motion_common")
    keyframes = importlib.import_module("keyframe_motion_common")
    module = SimpleNamespace(
        validate_acquisition=canonical.validate_acquisition,
        configuration=keyframes.configuration,
        runtime_sources=keyframes.runtime_sources,
    )
    before = vars(module).copy()
    with pytest.raises(RuntimeError), common.compound_pipeline(module):
        assert module.configuration is common.configuration
        with pytest.raises(ValueError), common.compound_pipeline(module):
            pytest.fail("nested pipeline accepted")
        raise RuntimeError("injected")
    assert vars(module) == before


def test_stdlib_bytes_cannot_impersonate_project_runtime_source(
    compound_experiment, monkeypatch
):
    common = importlib.import_module("compound_motion_common")
    canonical = importlib.import_module("controlled_motion_common")
    auditor = importlib.import_module("audit_compound_rgb_motion")
    route, output = compound_experiment
    config_path, report_path = output / "config.json", output / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    wrapper = Path(common.compound_bindings.__code__.co_filename)
    config["source_sha256"]["compound_motion.py"] = canonical.digest(wrapper)
    (output / "source-snapshot/compound_motion.py").write_bytes(wrapper.read_bytes())
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report["config_sha256"] = canonical.digest(config_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_compound_private_file_identity_cannot_change_between_scoring_stages(
    compound_experiment, pipeline, monkeypatch
):
    route, output = compound_experiment
    _, _, base = pipeline
    auditor = importlib.import_module("audit_compound_rgb_motion")
    original = base.audit

    def changed(*args):
        result = original(*args)
        (route / "evaluator-audit.json").write_text("changed")
        return result

    monkeypatch.setattr(base, "audit", changed)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["compound_evaluator_identity"] is False


def test_physical_turn_classification_requires_real_two_leg_motion(
    compound_experiment, pipeline
):
    route, output = compound_experiment
    auditor = importlib.import_module("audit_compound_rgb_motion")
    config = json.loads((output / "config.json").read_bytes())
    path = route / "evaluator-audit.json"
    telemetry = json.loads(path.read_bytes())
    positions = telemetry["positions"]
    pair = config["pairs"][0]
    legs = [
        item for item in config["pairs"] if item["id"] in ("leg_one-00", "leg_two-00")
    ]
    positions[legs[0]["second"]]["xpos"] = 1
    positions[legs[1]["second"]]["xpos"] = 1
    positions[legs[1]["second"]]["zpos"] = 1
    assert pair["first"] == legs[0]["first"]
    path.write_text(json.dumps(telemetry), encoding="utf-8")
    result = auditor.audit(route, output)
    assert result["valid"]
    first = result["evaluation"]["physical_turns"][0]
    assert first["leg_horizontal_blocks"] == [1, 1]
    assert first["angle_degrees"] == 90 and first["observed_noncollinear"]
    assert result["evaluation"]["observed_noncollinear_trials"] == 1


def test_compound_missing_experiment_is_an_invalid_audit(modules, tmp_path):
    auditor = importlib.import_module("audit_compound_rgb_motion")
    result = auditor.audit(tmp_path / "missing", tmp_path / "missing")
    assert not result["valid"] and result["evaluation"] is None
    assert result["failure"]["type"] == "FileNotFoundError"
