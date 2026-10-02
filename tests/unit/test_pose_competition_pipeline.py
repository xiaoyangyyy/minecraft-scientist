# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_active_motion_pipeline import active_experiment
from test_compound_motion_pipeline import compound_acquisition, compound_experiment
from test_controlled_motion_pipeline import acquisition, modules
from test_keyframe_motion_pipeline import pipeline


@pytest.fixture
def diagnosis(active_experiment, tmp_path):
    route, previous, _ = active_experiment
    output = tmp_path / "competition"
    result = importlib.import_module("inspect_pose_competition").run(
        route, previous, output
    )
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert len(result["results"]) == 46
    return route, previous, output


def test_diagnostic_producer_never_reads_any_private_evidence(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("inspect_pose_competition").run(
        route, previous, tmp_path / "competition"
    )
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert not result["navigation_accepted"]
    assert all(not item["competition"]["trials"] for item in result["results"])


def test_all_public_competition_replays_precede_first_private_read(
    diagnosis, monkeypatch
):
    route, previous, output = diagnosis
    common = importlib.import_module("pose_competition_common")
    original_replay, original_read, completed, private = (
        common.replay,
        Path.read_bytes,
        [],
        [],
    )

    def replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        completed.extend(result)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert path.parent == route and len(completed) == 46
            private.append(path)
        return original_read(path)

    monkeypatch.setattr(common, "replay", replay)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_pose_competition").audit(
        route, previous, output
    )
    assert result["valid"] and private
    assert all(result["checks"].values())
    assert not result["evaluation"]["navigation_accepted"]
    assert all(
        group["selected"] == 0 for group in result["evaluation"]["groups"].values()
    )


@pytest.mark.parametrize("tamper", ["policy", "snapshot", "flag", "drop_pair"])
def test_tampered_diagnosis_rejects_before_private_scoring(
    diagnosis, monkeypatch, tamper
):
    route, previous, output = diagnosis
    if tamper == "snapshot":
        (output / "source-snapshot/pose_competition.py").write_text("tampered")
    else:
        path = output / ("config.json" if tamper == "policy" else "report.json")
        content = json.loads(path.read_bytes())
        if tamper == "policy":
            content["policy"]["direction_separation_degrees"] = 30
        elif tamper == "flag":
            content["results"][0]["competition"]["ambiguity_detected"] = True
        else:
            content["results"].pop()
        path.write_text(json.dumps(content), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_pose_competition").audit(
        route, previous, output
    )
    assert not result["valid"] and result["evaluation"] is None


def test_private_identity_and_public_inputs_stay_frozen_through_scoring(
    diagnosis, monkeypatch
):
    route, previous, output = diagnosis
    auditor = importlib.import_module("audit_pose_competition")
    original = auditor.baseline.audit

    def changed(*args):
        result = original(*args)
        assert result["valid"]
        (route / "evaluator-audit.json").write_text("changed after scoring")
        (route / "frames/frame-00000.png").write_bytes(b"changed after replay")
        return result

    monkeypatch.setattr(auditor.baseline, "audit", changed)
    result = auditor.audit(route, previous, output)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["baseline_evaluator_identity"]
    assert not result["checks"]["public_bytes_stable_through_scoring"]


def test_partial_witness_failure_keeps_previous_results_and_does_not_promote(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    common = importlib.import_module("pose_competition_common")
    original, calls = common.inspect_pose_competition, []

    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise RuntimeError("injected competing-map failure")
        return original(*args, **kwargs)

    # Assertion bytecode caches are shared across Windows and WSL. This mock
    # must declare the executing platform's path, not a cached foreign filename.
    interrupted.__code__ = interrupted.__code__.replace(
        co_filename=str(Path(__file__).resolve())
    )
    monkeypatch.setattr(common, "inspect_pose_competition", interrupted)
    result = importlib.import_module("inspect_pose_competition").run(
        route, previous, tmp_path / "competition"
    )
    assert not result["completed"] and len(result["results"]) == 2
    assert result["failure"]["type"] == "RuntimeError"
    assert result["inputs_sources_unchanged"] and not result["navigation_accepted"]


def test_existing_output_refuses_before_any_prior_replay(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    producer = importlib.import_module("inspect_pose_competition")
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        producer.common,
        "inputs",
        lambda *args: pytest.fail("existing output read prior inputs"),
    )
    with pytest.raises(FileExistsError):
        producer.run(route, previous, output)
