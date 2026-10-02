# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual pipeline")
from test_active_motion_pipeline import active_experiment
from test_compound_motion_pipeline import compound_acquisition, compound_experiment
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline


@pytest.fixture
def diagnosis(active_experiment, tmp_path):
    route, previous, _ = active_experiment
    output = tmp_path / "switched-views"
    result = importlib.import_module("probe_action_switch").run(route, previous, output)
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert len(result["results"]) == 46
    return route, previous, output


def test_public_producer_does_not_consume_private_positions(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("probe_action_switch").run(
        route, previous, tmp_path / "switch"
    )
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert not result["navigation_accepted"]
    assert all(
        item["action_switch"]["comparison"] == "no_candidate"
        for item in result["results"]
    )


def test_all_public_comparisons_replay_before_first_private_read(
    diagnosis, monkeypatch
):
    route, previous, output = diagnosis
    common = importlib.import_module("action_switch_common")
    replayed, private = [], []
    original_replay, original_read = common.replay, Path.read_bytes

    def replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        replayed.extend(result)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert path.parent == route and len(replayed) == 46
            private.append(path)
        return original_read(path)

    monkeypatch.setattr(common, "replay", replay)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_action_switch").audit(
        route, previous, output
    )
    assert result["valid"] and private and all(result["checks"].values())
    assert not result["evaluation"]["navigation_accepted"]
    assert all(
        not item["diagnostic_retained"] for item in result["evaluation"]["samples"]
    )


@pytest.mark.parametrize(
    "tamper", ["policy", "snapshot", "comparison", "target_frames", "drop_pair"]
)
def test_tampered_switched_view_evidence_rejects_before_private_read(
    diagnosis, monkeypatch, tamper
):
    route, previous, output = diagnosis
    if tamper == "snapshot":
        (output / "source-snapshot/action_switch_motion.py").write_text("changed")
    else:
        path = output / ("config.json" if tamper == "policy" else "report.json")
        data = json.loads(path.read_bytes())
        if tamper == "policy":
            data["policy"]["temporal"]["maximum_frames"] = 96
        elif tamper == "comparison":
            data["results"][0]["action_switch"]["comparison"] = (
                "tested_alternatives_rejected"
            )
        elif tamper == "target_frames":
            data["results"][0]["action_switch"]["plan"]["target_frames"] = [80, 90]
        else:
            data["results"].pop()
        path.write_text(json.dumps(data), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_action_switch").audit(
        route, previous, output
    )
    assert not result["valid"] and result["evaluation"] is None


def test_postscore_private_or_public_mutation_invalidates_results(
    diagnosis, monkeypatch
):
    route, previous, output = diagnosis
    auditor = importlib.import_module("audit_action_switch")
    original = auditor.baseline.audit

    def changed(*args):
        result = original(*args)
        assert result["valid"]
        (route / "evaluator-audit.json").write_text("changed after baseline scoring")
        (route / "frames/frame-00000.png").write_bytes(b"changed after replay")
        return result

    monkeypatch.setattr(auditor.baseline, "audit", changed)
    result = auditor.audit(route, previous, output)
    assert not result["valid"] and result["evaluation"] is None


def test_partial_holdout_failure_retains_completed_pairs(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    common = importlib.import_module("action_switch_common")
    original, calls = common.inspect_action_switch, []

    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise RuntimeError("injected switched-view failure")
        return original(*args, **kwargs)

    interrupted.__code__ = interrupted.__code__.replace(
        co_filename=str(Path(__file__).resolve())
    )
    monkeypatch.setattr(common, "inspect_action_switch", interrupted)
    result = importlib.import_module("probe_action_switch").run(
        route, previous, tmp_path / "switch"
    )
    assert not result["completed"] and result["inputs_sources_unchanged"]
    assert len(result["results"]) == 2 and result["failure"]["type"] == "RuntimeError"
    assert not result["navigation_accepted"]


def test_output_refuses_overwrite_before_loading_inputs(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    producer = importlib.import_module("probe_action_switch")
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        producer.common, "inputs", lambda *args: pytest.fail("overwrite read inputs")
    )
    with pytest.raises(FileExistsError):
        producer.run(route, previous, output)
