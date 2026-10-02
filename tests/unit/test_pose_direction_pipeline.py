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
    output = tmp_path / "direction-search"
    result = importlib.import_module("probe_pose_direction").run(
        route, previous, output
    )
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert len(result["results"]) == 46
    return route, previous, output


def test_public_direction_producer_never_reads_private_evidence(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("probe_pose_direction").run(
        route, previous, tmp_path / "search"
    )
    assert result["completed"] and result["inputs_sources_unchanged"]
    assert not result["navigation_accepted"]
    assert all(
        item["direction_search"]["reference"] is None for item in result["results"]
    )


def test_complete_direction_replay_precedes_private_scoring(diagnosis, monkeypatch):
    route, previous, output = diagnosis
    common = importlib.import_module("pose_direction_common")
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
    result = importlib.import_module("audit_pose_direction").audit(
        route, previous, output
    )
    assert result["valid"] and private and all(result["checks"].values())
    assert not result["evaluation"]["navigation_accepted"]
    assert all(
        group["selected"] == 0 for group in result["evaluation"]["groups"].values()
    )


@pytest.mark.parametrize(
    "tamper", ["policy", "snapshot", "flag", "reference", "drop_pair"]
)
def test_tampered_direction_evidence_rejects_before_private_read(
    diagnosis, monkeypatch, tamper
):
    route, previous, output = diagnosis
    if tamper == "snapshot":
        (output / "source-snapshot/pose_direction_search.py").write_text("tampered")
    else:
        path = output / ("config.json" if tamper == "policy" else "report.json")
        data = json.loads(path.read_bytes())
        if tamper == "policy":
            data["policy"]["maximum_rotation_adjustment_degrees"] = 2
        elif tamper == "flag":
            data["results"][0]["direction_search"]["ambiguity_detected"] = True
        elif tamper == "reference":
            data["results"][0]["direction_search"][
                "reference_matches_original_checks"
            ] = True
        else:
            data["results"].pop()
        path.write_text(json.dumps(data), encoding="utf-8")
    original = Path.read_bytes

    def reading(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reading)
    result = importlib.import_module("audit_pose_direction").audit(
        route, previous, output
    )
    assert not result["valid"] and result["evaluation"] is None


def test_scoring_cannot_mutate_frozen_public_or_private_identity(
    diagnosis, monkeypatch
):
    route, previous, output = diagnosis
    auditor = importlib.import_module("audit_pose_direction")
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
    assert not result["checks"]["public_bytes_stable_through_scoring"]
    assert not result["checks"]["baseline_evaluator_identity"]


def test_partial_search_failure_preserves_completed_pairs(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    common = importlib.import_module("pose_direction_common")
    original, calls = common.inspect_direction_search, []

    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise RuntimeError("injected direction-search failure")
        return original(*args, **kwargs)

    interrupted.__code__ = interrupted.__code__.replace(
        co_filename=str(Path(__file__).resolve())
    )
    monkeypatch.setattr(common, "inspect_direction_search", interrupted)
    result = importlib.import_module("probe_pose_direction").run(
        route, previous, tmp_path / "search"
    )
    assert not result["completed"] and result["inputs_sources_unchanged"]
    assert len(result["results"]) == 2 and result["failure"]["type"] == "RuntimeError"
    assert not result["navigation_accepted"]


def test_direction_output_refuses_overwrite_before_reading_inputs(
    active_experiment, tmp_path, monkeypatch
):
    route, previous, _ = active_experiment
    producer = importlib.import_module("probe_pose_direction")
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        producer.common,
        "inputs",
        lambda *args: pytest.fail("existing output read inputs"),
    )
    with pytest.raises(FileExistsError):
        producer.run(route, previous, output)
