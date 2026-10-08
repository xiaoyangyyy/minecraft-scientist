"""Private-access guards, tamper rejection, deterministic persisted H1 artifacts."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.cli import app
from minecraft_scientist.evaluation.identification import (
    audit_identification,
    run_identification_benchmark,
)
from minecraft_scientist.world.identification import identification_task


@pytest.fixture
def capture(tmp_path):
    task = identification_task(0, "activation")
    config = {
        "schema": "known-candidate-h1-config-v1",
        "method": "witness_cover",
        "seed": 0,
        "budget": 20,
        "horizon": 5,
        "calibration": list(task.calibration),
        "costs": task.costs,
    }
    candidates = [ir.model_dump(mode="json") for ir in task.candidates]
    result = identify_mechanism(
        task.view, task.candidates, task.calibration, task.costs
    )
    truth = tmp_path / "private-truth.json"
    truth.write_text(task.ground_truth.model_dump_json(), encoding="utf-8")
    return config, candidates, result, truth


def test_complete_public_replay_precedes_independent_private_evaluation(capture):
    audit = audit_identification(*capture)
    assert (
        audit["valid"]
        and audit["public_replay_valid"]
        and audit["private_truth_opened"]
    )
    evaluation = audit["evaluation"]
    assert evaluation["calibration_candidate_traces_agree"]
    assert evaluation["pair_certificates"][0]["minimum_query_cost"] == 4
    assert evaluation["identified_behavior_correctly"]
    assert evaluation["declared_universe_prediction_accuracy"] == 1.0


@pytest.mark.parametrize(
    "case",
    [
        "prediction",
        "budget",
        "order",
        "output",
        "remaining",
        "theory",
        "config",
        "cost",
    ],
)
def test_invalid_public_evidence_cannot_open_private_truth(capture, monkeypatch, case):
    config, candidates, result, truth = copy.deepcopy(capture)
    if case == "prediction":
        result["events"][0]["predictions"]["0"][0] = "invented"
    elif case == "budget":
        result["spent"] -= 1
    elif case == "order":
        result["events"][0], result["events"][1] = (
            result["events"][1],
            result["events"][0],
        )
    elif case == "output":
        event = next(e for e in result["events"] if e["kind"] == "evidence")
        event["outputs"][0] = "invented"
    elif case == "remaining":
        result["remaining_candidates"] = [1]
    elif case == "theory":
        result["theories"][-1]["status"] = "accepted"
    elif case == "config":
        config["private_pose"] = [1, 2, 3]
    else:
        config["costs"][next(iter(config["costs"]))] = 0

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public evidence opened private truth")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_identification(config, candidates, result, truth)
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["evaluation"] is None


def test_private_execution_counterexample_cannot_be_replaced_by_public_replay(capture):
    config, candidates, result, truth = capture
    payload = json.loads(truth.read_text())
    payload["transitions"] = [
        row | {"output": payload["outputs"][0]} for row in payload["transitions"]
    ]
    truth.write_text(json.dumps(payload), encoding="utf-8")
    audit = audit_identification(config, candidates, result, truth)
    assert audit["public_replay_valid"] and not audit["valid"]
    assert not audit["evaluation"]["actual_controls_match_private_reference"]


def test_all_artifacts_replay_exactly_and_existing_output_is_preserved(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    one = run_identification_benchmark(output=first, seeds=1)
    two = run_identification_benchmark(output=second, seeds=1)
    assert one == two and len(one["rows"]) == 30
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest == json.loads((second / "manifest.json").read_text())
    for relative, digest in manifest.items():
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == digest
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert b"\r\n" not in (first / relative).read_bytes()
    original = (first / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_identification_benchmark(output=first, seeds=1)
    assert (first / "report.json").read_bytes() == original


def test_cli_runs_offline_h1_protocol_and_names_its_scope(tmp_path):
    output = tmp_path / "cli"
    result = CliRunner().invoke(
        app, ["benchmark-identification", "--output", str(output), "--seeds", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "not structural generalization" in result.output
    report = json.loads((output / "report.json").read_text())
    assert report["all_independent_audits_valid"]
