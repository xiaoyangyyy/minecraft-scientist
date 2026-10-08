"""Known-family factor attribution, exact aliases, partial resolution and auditing."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minecraft_scientist.agent.diagnosis import diagnose_mechanism
from minecraft_scientist.cli import app
from minecraft_scientist.discovery.diagnosis import (
    NO_DELIVERY,
    JointHypothesis,
    validate_hypotheses,
)
from minecraft_scientist.evaluation.diagnosis import (
    _factor_rollout,
    _matches_factors,
    _public_spec,
    audit_diagnosis,
    run_diagnosis_benchmark,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram, shortest_witness
from minecraft_scientist.world.diagnosis import CONTROLLERS, MECHANISMS, diagnosis_task


def run_task(task, *, method="witness_cover", budget=32):
    return diagnose_mechanism(
        task.view,
        task.hypotheses,
        task.calibration,
        task.costs,
        baseline_mechanism=task.baseline_mechanism,
        baseline_controller=task.baseline_controller,
        method=method,
        seed=20261008,
        budget=budget,
    )


@pytest.mark.parametrize("bypass", [False, True])
def test_composed_ir_matches_independent_factors_and_actual_hidden_world(bypass):
    task = diagnosis_task(2, bypass=bypass)
    for hypothesis in task.hypotheses:
        program = FiniteStateProgram.from_ir(hypothesis.compile())
        spec = _public_spec(hypothesis)
        assert _matches_factors(program.model_dump(mode="json"), spec)
        for length in range(1, 4):
            for word in itertools.product(program.actions, repeat=length):
                assert program.execute(word)[0] == _factor_rollout(spec, word)
    task.view.begin_trial()
    available = {action.target_id: action for action in task.view.available_actions()}
    outputs = [
        task.view.step(available[a]).event.details["signal"] for a in task.calibration
    ]
    assert tuple(outputs) == _factor_rollout(task.private_spec, task.calibration)


@pytest.mark.parametrize("seed", [0, 3, 19])
def test_same_anomaly_different_causes_are_exact_aliases_without_alternate_route(seed):
    task = diagnosis_task(seed, bypass=False)
    fault, changed = task.hypotheses[1], task.hypotheses[6]
    left, right = [FiniteStateProgram.from_ir(h.compile()) for h in (fault, changed)]
    assert left.execute(task.calibration)[0] == right.execute(task.calibration)[0]
    assert shortest_witness(left, right, task.costs) is None
    assert (
        left.execute(task.calibration)[0][-1]
        != (
            FiniteStateProgram.from_ir(task.hypotheses[0].compile()).execute(
                task.calibration
            )[0][-1]
        )
    )
    result = run_task(task)
    assert result["diagnosis"]["mechanism_status"] in ("ambiguous", "unresolved")
    assert not result["diagnosis"]["mechanism_revision_warranted"]
    assert fault.mechanism_id in result["diagnosis"]["possible_mechanisms"]
    assert changed.mechanism_id in result["diagnosis"]["possible_mechanisms"]


@pytest.mark.parametrize("seed", [0, 3, 19])
def test_alternate_actuation_is_a_costly_witness_not_a_fault_flag(seed):
    task = diagnosis_task(seed, bypass=True)
    left, right = [
        FiniteStateProgram.from_ir(task.hypotheses[i].compile()) for i in (1, 6)
    ]
    witness = shortest_witness(left, right, task.costs, maximum_length=5)
    alternate = next(action for action, cost in task.costs.items() if cost == 3)
    assert witness is not None and alternate in witness.actions
    assert (
        witness.action_cost == 6
    )  # load + alternate + wait + take, plus reset separately.
    task.view.begin_trial()
    available = {action.target_id: action for action in task.view.available_actions()}
    for command in task.calibration:
        record = task.view.step(available[command])
        assert record.context["treatment_verification"] == "unknown"
        assert record.intervention_compliance and record.controller_failure is None
        assert set(record.event.details) == {"signal"}
        assert "delivered" not in record.model_dump_json()
        assert NO_DELIVERY not in record.model_dump_json()


@pytest.mark.parametrize(
    "mechanism,controller,bypass,method",
    tuple(
        itertools.product(
            MECHANISMS, CONTROLLERS, (False, True), ("witness_cover", "sequence_ig")
        )
    ),
)
def test_matched_planners_preserve_true_factors_and_never_blame_fault_for_mechanism_change(
    mechanism, controller, bypass, method
):
    task = diagnosis_task(0, mechanism, controller, bypass=bypass)
    result = run_task(task, method=method)
    core, diagnosis = result["identification"], result["diagnosis"]
    truth_index = 3 * MECHANISMS.index(mechanism) + CONTROLLERS.index(controller)
    assert truth_index in core["remaining_candidates"]
    assert core["spent"] <= core["budget"] == 32
    assert core["spent"] == sum(event.get("cost", 0) for event in core["events"])
    if mechanism == "unchanged":
        assert not diagnosis["mechanism_revision_warranted"]
        if bypass:
            assert diagnosis["mechanism_status"] == "unchanged"
    elif bypass:
        assert diagnosis["mechanism_status"] == "changed"
        assert diagnosis["mechanism_revision_warranted"]
    if bypass and mechanism == "activation_edge_removed":
        assert len(diagnosis["possible_controllers"]) == 3
        assert diagnosis["controller_status"] in ("ambiguous", "unresolved")
        assert not diagnosis["unique_factor_pair_identified"]
    assert not diagnosis["mechanism_revision_completed"]
    assert not diagnosis["independent_validation_completed"]
    assert all(
        theory["status"] in ("proposed", "provisional") for theory in core["theories"]
    )


def test_partial_factor_resolution_does_not_require_unique_joint_behavior():
    task = diagnosis_task(0, "delayed", "alternating_drop")
    result = run_task(task)
    assert result["identification"]["status"] == "unresolved"
    assert result["diagnosis"]["mechanism_status"] == "changed"
    assert result["diagnosis"]["controller_status"] == "changed"
    assert not result["diagnosis"]["joint_behavior_identified"]
    assert len(result["diagnosis"]["possible_controllers"]) == 2


@pytest.mark.parametrize("budget", [1, 4, 5])
def test_insufficient_budget_does_not_dispatch_unaffordable_diagnostic_or_claim_revision(
    budget,
):
    task = diagnosis_task(0)
    result = run_task(task, budget=budget)
    core = result["identification"]
    assert core["spent"] == (0 if budget < 5 else 5)
    assert not result["diagnosis"]["mechanism_revision_warranted"]
    assert len([e for e in core["events"] if e["kind"] == "plan"]) == (
        0 if budget < 5 else 1
    )


@pytest.mark.parametrize(
    "case",
    ["missing_baseline", "duplicate", "redefinition", "bad_delivery", "state_cap"],
)
def test_invalid_joint_models_are_rejected_before_dispatch(case):
    task = diagnosis_task(0)
    hypotheses = task.hypotheses
    if case == "missing_baseline":
        hypotheses = hypotheses[1:]
    elif case == "duplicate":
        hypotheses = (*hypotheses, hypotheses[0])
    elif case == "redefinition":
        hypotheses = (
            hypotheses[0],
            hypotheses[1].model_copy(
                update={
                    "mechanism_ir": hypotheses[3].mechanism_ir,
                }
            ),
        )
    elif case == "bad_delivery":
        payload = hypotheses[0].controller_ir.model_dump(mode="json")
        payload["parameters"]["outputs"].append("invalid-delivery")
        hypotheses = (
            JointHypothesis.model_validate(
                hypotheses[0].model_dump()
                | {
                    "controller_ir": payload,
                }
            ),
        )
    else:
        payload = hypotheses[0].mechanism_ir.model_dump(mode="json")
        states = [f"s-{i}" for i in range(9)]
        payload["parameters"].update(
            {
                "states": states,
                "initial_state": states[0],
                "transitions": [
                    {
                        "source": s,
                        "action": a,
                        "target": s,
                        "output": payload["parameters"]["outputs"][0],
                    }
                    for s in states
                    for a in payload["parameters"]["actions"]
                ],
            }
        )
        control = hypotheses[2]
        hypotheses = (
            control.model_copy(
                update={
                    "mechanism_ir": control.mechanism_ir.model_validate(payload),
                    "controller_id": task.baseline_controller,
                }
            ),
        )
    with pytest.raises((ValueError, TypeError)):
        validate_hypotheses(
            hypotheses, task.baseline_mechanism, task.baseline_controller
        )
    assert task.view.observe().step == 0


@pytest.fixture
def capture(tmp_path):
    task = diagnosis_task(0)
    result = run_task(task)
    config = {
        "schema": "joint-diagnosis-config-v1",
        "baseline_mechanism": task.baseline_mechanism,
        "baseline_controller": task.baseline_controller,
        "identification": {
            "schema": "known-candidate-h1-config-v1",
            "method": "witness_cover",
            "seed": 20261008,
            "budget": 32,
            "horizon": 5,
            "calibration": list(task.calibration),
            "costs": task.costs,
        },
    }
    private = tmp_path / "private.json"
    private.write_text(json.dumps(task.private_spec), encoding="utf-8")
    return config, [h.model_dump(mode="json") for h in task.hypotheses], result, private


def test_public_first_audit_independently_replays_factors_and_attribution(capture):
    audit = audit_diagnosis(*capture)
    assert (
        audit["valid"]
        and audit["public_replay_valid"]
        and audit["private_truth_opened"]
    )
    evaluation = audit["evaluation"]
    assert evaluation["actual_controls_match_private_reference"]
    assert evaluation["true_factor_pair_retained"]
    assert evaluation["mechanism_change_correctly_resolved"] is True
    assert not evaluation["false_mechanism_change"]


@pytest.mark.parametrize(
    "case", ["diagnosis", "baseline", "prediction", "budget", "treatment", "status"]
)
def test_invalid_public_claims_never_open_private_truth(capture, monkeypatch, case):
    config, hypotheses, result, private = copy.deepcopy(capture)
    if case == "diagnosis":
        result["diagnosis"]["mechanism_revision_warranted"] = True
    elif case == "baseline":
        config["baseline_mechanism"] = "unknown"
    elif case == "prediction":
        result["identification"]["events"][0]["predictions"]["0"][0] = "invented"
    elif case == "budget":
        result["identification"]["spent"] -= 1
    elif case == "treatment":
        step = next(
            e for e in result["identification"]["events"] if e["kind"] == "step"
        )
        step["transition"]["context"]["treatment_verification"] = "verified"
    else:
        result["identification"]["theories"][-1]["status"] = "accepted"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public diagnosis read private truth")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_diagnosis(config, hypotheses, result, private)
    assert not audit["valid"] and not audit["private_truth_opened"]


def test_independent_private_controller_counterexample_is_not_hidden_by_policy_replay(
    capture,
):
    config, hypotheses, result, private = capture
    truth = json.loads(private.read_text())
    truth["controller"] = _public_spec(JointHypothesis.model_validate(hypotheses[0]))[
        "controller"
    ]
    private.write_text(json.dumps(truth), encoding="utf-8")
    audit = audit_diagnosis(config, hypotheses, result, private)
    assert audit["public_replay_valid"] and audit["private_truth_opened"]
    assert (
        not audit["valid"]
        and not audit["evaluation"]["actual_controls_match_private_reference"]
    )


def test_persisted_development_retains_unresolved_cases_and_replays_exact_bytes(
    tmp_path,
):
    first, second = tmp_path / "first", tmp_path / "second"
    one = run_diagnosis_benchmark(output=first, seeds=1)
    two = run_diagnosis_benchmark(output=second, seeds=1)
    assert one == two and len(one["rows"]) == 36
    assert any(row["status"] == "unresolved" for row in one["rows"])
    assert all(not row["evaluation"]["false_mechanism_change"] for row in one["rows"])
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest == json.loads((second / "manifest.json").read_text())
    for relative, digest in manifest.items():
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == digest
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert b"\r\n" not in (first / relative).read_bytes()
    with pytest.raises(FileExistsError):
        run_diagnosis_benchmark(output=first, seeds=1)


def test_cli_offline_diagnosis_names_its_limits(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "benchmark-diagnosis",
            "--output",
            str(tmp_path / "cli"),
            "--seeds",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not novelty or complete H2" in result.output
