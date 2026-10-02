# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_compound_motion_pipeline import compound_acquisition, compound_experiment
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline


def test_plan_generation_and_replay_never_open_private_evidence(
    compound_experiment, tmp_path, monkeypatch
):
    route, previous = compound_experiment
    planner = importlib.import_module("plan_active_rgb_motion")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    output = tmp_path / "plan"
    result = planner.build_plan(route, previous, output)
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert not result["choices"]["hypotheses"]
    requested = planner.validate_plan(output)
    assert requested.handle_pairs == ((0, 1), (2, 3), (0, 2), (1, 3))
    assert not any(
        "evaluator" in name
        for name in json.loads((output / "config.json").read_bytes())["input_sha256"]
    )


def test_changed_action_choice_rejects_without_private_scoring(
    compound_experiment, tmp_path, monkeypatch
):
    route, previous = compound_experiment
    planner = importlib.import_module("plan_active_rgb_motion")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    output = tmp_path / "plan"
    assert planner.build_plan(route, previous, output)["completed"]
    path = output / "plan.json"
    plan = json.loads(path.read_bytes())
    plan["choices"]["handle_pairs"][0] = [3, 2]
    path.write_text(json.dumps(plan), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    with pytest.raises(ValueError, match="action selection"):
        planner.validate_plan(output)


def test_existing_plan_refuses_before_any_geometry(
    compound_experiment, tmp_path, monkeypatch
):
    planner = importlib.import_module("plan_active_rgb_motion")
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        planner,
        "replay_prior",
        lambda *args: pytest.fail("existing output reached geometry"),
    )
    with pytest.raises(FileExistsError):
        planner.build_plan(*compound_experiment, output)


def test_changed_prior_bytes_during_replay_rejects_before_capture(
    compound_experiment, tmp_path, monkeypatch
):
    route, previous = compound_experiment
    planner = importlib.import_module("plan_active_rgb_motion")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    output = tmp_path / "plan"
    assert planner.build_plan(route, previous, output)["completed"]
    original, original_read = planner.replay_prior, Path.read_bytes

    def changed(*args):
        result = original(*args)
        (route / "frames/frame-00000.png").write_bytes(b"changed during replay")
        return result

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original_read(path)

    monkeypatch.setattr(planner, "replay_prior", changed)
    monkeypatch.setattr(Path, "read_bytes", reading)
    with pytest.raises(ValueError, match="changed during public replay"):
        planner.validate_plan(output)
