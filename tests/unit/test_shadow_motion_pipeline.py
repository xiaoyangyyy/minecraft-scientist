# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_compound_motion_pipeline import compound_acquisition
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline


@pytest.fixture
def shadow_experiment(compound_acquisition, pipeline, tmp_path):
    common = importlib.import_module("shadow_motion_common")
    route, report = compound_acquisition
    _, producer, _ = pipeline
    output = tmp_path / "shadow"
    with common.shadow_pipeline(producer):
        result = producer.run(route, output)
    assert result["completed"] and len(result["results"]) == 46
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


def test_all_public_prefix_and_shadow_replays_precede_any_private_read(
    shadow_experiment, pipeline, monkeypatch
):
    route, output = shadow_experiment
    _, _, base = pipeline
    common = importlib.import_module("shadow_motion_common")
    auditor = importlib.import_module("audit_shadow_rgb_motion")
    replay, private = [], []
    original_infer, original_read = common.infer_pair, Path.read_bytes

    def inference(*args):
        result = original_infer(*args)
        replay.append(result)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert len(replay) == 46
            private.append(path)
        return original_read(path)

    monkeypatch.setattr(common, "infer_pair", inference)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert result["valid"] and private and len(replay) == 46
    assert all(result["checks"].values())
    assert all(
        group["shadow_accepted"] == 0
        for group in result["evaluation"]["shadow_groups"].values()
    )
    assert not result["evaluation"]["navigation_accepted"]
    assert (
        base.infer_pair is importlib.import_module("keyframe_motion_common").infer_pair
    )


@pytest.mark.parametrize(
    "tamper", ["drop_shadow", "future_config", "snapshot", "candidate"]
)
def test_tampering_cannot_reach_private_shadow_scoring(
    shadow_experiment, pipeline, monkeypatch, tamper
):
    route, output = shadow_experiment
    auditor = importlib.import_module("audit_shadow_rgb_motion")
    report_path, config_path = output / "report.json", output / "config.json"
    report, config = (
        json.loads(report_path.read_bytes()),
        json.loads(config_path.read_bytes()),
    )
    if tamper == "drop_shadow":
        report["results"][0].pop("shadow")
    elif tamper == "candidate":
        report["results"][0]["shadow"]["candidate"] = {"end_offset": 2}
    elif tamper == "future_config":
        config["shadow"]["future_frames"] = 1
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report["config_sha256"] = auditor.digest(config_path)
    else:
        (output / "source-snapshot" / "shadow_motion.py").write_text("changed")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original_read = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_private_file_change_between_scoring_stages_rejects(
    shadow_experiment, pipeline, monkeypatch
):
    route, output = shadow_experiment
    _, _, base = pipeline
    auditor = importlib.import_module("audit_shadow_rgb_motion")
    original = base.audit

    def changed(*args):
        result = original(*args)
        (route / "evaluator-audit.json").write_text("changed")
        return result

    monkeypatch.setattr(base, "audit", changed)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["shadow_evaluator_identity"] is False


def test_public_file_change_after_base_scoring_is_not_ignored(
    shadow_experiment, pipeline, monkeypatch
):
    route, output = shadow_experiment
    _, _, base = pipeline
    auditor = importlib.import_module("audit_shadow_rgb_motion")
    original = base.audit

    def changed(*args):
        result = original(*args)
        (route / "frames" / "frame-00070.png").write_bytes(b"changed")
        return result

    monkeypatch.setattr(base, "audit", changed)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["shadow_sources_inputs_stable"] is False
