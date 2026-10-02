# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_compound_motion_pipeline import compound_acquisition
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline
from test_shadow_motion_pipeline import shadow_experiment


def _public_only(monkeypatch):
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)


def test_parallax_diagnosis_replays_every_pair_and_never_reads_private_evidence(
    shadow_experiment, tmp_path, monkeypatch
):
    route, previous = shadow_experiment
    diagnostic = importlib.import_module("inspect_public_parallax")
    original, calls = diagnostic.shadows.infer_pair, []

    def replay(*args):
        result = original(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(diagnostic.shadows, "infer_pair", replay)
    _public_only(monkeypatch)
    result = diagnostic.run(route, previous, tmp_path / "diagnosis")
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert len(calls) == len(result["results"]) == 46
    assert all(
        not item["parallax"]["conditionally_diverse_view"] for item in result["results"]
    )
    assert not result["navigation_accepted"]


@pytest.mark.parametrize("tamper", ["pairs", "snapshot", "config"])
def test_invalid_shadow_declarations_reject_without_private_reads(
    shadow_experiment, tmp_path, monkeypatch, tamper
):
    route, previous = shadow_experiment
    diagnostic = importlib.import_module("inspect_public_parallax")
    if tamper == "snapshot":
        (previous / "source-snapshot/shadow_motion.py").write_text("tampered")
    else:
        path = previous / ("report.json" if tamper == "pairs" else "config.json")
        content = json.loads(path.read_bytes())
        if tamper == "pairs":
            content["results"].pop()
        else:
            content["shadow"]["future_frames"] = 1
        path.write_text(json.dumps(content), encoding="utf-8")
    _public_only(monkeypatch)
    output = tmp_path / "diagnosis"
    with pytest.raises(ValueError, match="invalid public shadow"):
        diagnostic.run(route, previous, output)
    assert not output.exists()


def test_diagnostic_keeps_partial_failures_without_promoting_a_pose(
    shadow_experiment, tmp_path, monkeypatch
):
    route, previous = shadow_experiment
    diagnostic = importlib.import_module("inspect_public_parallax")
    original = diagnostic.shadows.infer_pair

    def interrupted(*args):
        if args[-1] == 2:
            raise RuntimeError("injected replay failure")
        return original(*args)

    monkeypatch.setattr(diagnostic.shadows, "infer_pair", interrupted)
    _public_only(monkeypatch)
    result = diagnostic.run(route, previous, tmp_path / "diagnosis")
    assert not result["completed"] and len(result["results"]) == 2
    assert result["failure"]["type"] == "RuntimeError"
    assert result["inputs_sources_unchanged"] and not result["navigation_accepted"]


def test_public_inputs_remain_frozen_after_final_diagnostic_computation(
    shadow_experiment, tmp_path, monkeypatch
):
    route, previous = shadow_experiment
    diagnostic = importlib.import_module("inspect_public_parallax")
    original = diagnostic.shadows.infer_pair

    def changed(*args):
        result = original(*args)
        if args[-1] == 45:
            (route / "frames/frame-00000.png").write_bytes(b"changed after replay")
        return result

    monkeypatch.setattr(diagnostic.shadows, "infer_pair", changed)
    _public_only(monkeypatch)
    result = diagnostic.run(route, previous, tmp_path / "diagnosis")
    assert result["completed"] and len(result["results"]) == 46
    assert not result["inputs_sources_unchanged"]
    assert not result["navigation_accepted"]
