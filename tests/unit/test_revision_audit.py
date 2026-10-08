"""Independent raw edit/behavior checks and private-read ordering attacks."""

import copy
import itertools
import json
from pathlib import Path

import pytest

from minecraft_scientist.agent.revision import revise_online
from minecraft_scientist.evaluation.diagnosis import _factor_rollout
from minecraft_scientist.evaluation.identification import _rollout
from minecraft_scientist.evaluation.revision import (
    _bounded_matches_factors,
    _raw_edit,
    audit_revision,
    run_revision_benchmark,
)
from minecraft_scientist.world.revision import revision_task


def config_for(task, **kwargs):
    return {
        "schema": "online-revision-config-v1",
        "calibration": list(task.calibration),
        "costs": task.costs,
        "protocol": {
            "method": "witness_cover",
            "seed": 20261008,
            "budget": 240,
            "horizon": 5,
            "monitor_trials": 10,
            "search_budget": 1000,
            "shadow_count": 3,
            "candidate_limit": 512,
        }
        | kwargs,
    }


def bundle(case="redirect_start", seed=0, **kwargs):
    task = revision_task(seed, case)
    config = config_for(task, **kwargs)
    result = revise_online(
        task.view, task.bootstrap, task.calibration, task.costs, **config["protocol"]
    )
    return (
        config,
        [h.model_dump(mode="json") for h in task.bootstrap],
        result,
        task.private_spec,
    )


def audit_bundle(data, tmp_path):
    config, hypotheses, result, truth = data
    private = tmp_path / "truth.json"
    private.write_text(json.dumps(truth), encoding="utf-8")
    return audit_revision(config, hypotheses, result, private)


@pytest.fixture(scope="module")
def good_bundle():
    return bundle()


@pytest.mark.parametrize(
    "case",
    [
        "unchanged",
        "controller_fault",
        "redirect_start",
        "redirect_wait",
        "retain_ready",
        "state_insert",
        "outside_grammar",
    ],
)
def test_all_positive_and_negative_runs_pass_independent_replay(case, tmp_path):
    config, _hypotheses, result, _truth = data = bundle(case)
    audit = audit_bundle(data, tmp_path)
    assert audit["valid"], audit
    assert audit["public_replay_valid"] and audit["private_truth_opened"]
    evaluation = audit["evaluation"]
    assert evaluation["bootstrap_behavior_correct_at_declared_horizon"]
    assert not evaluation["false_mechanism_change"]
    assert evaluation["replacement_behavior_correct_at_declared_horizon"] == (
        result["status"] == "conditional_joint_theory_revised"
    )
    assert (
        evaluation["total_experiment_cost"]
        == result["spent"]
        <= config["protocol"]["budget"]
    )
    if case not in ("unchanged", "controller_fault"):
        assert evaluation["mechanism_behavior_changed"]
        assert evaluation["detection_delay_episodes"] >= 0


@pytest.mark.parametrize(
    "attack",
    [
        "budget",
        "prediction",
        "edit",
        "scope",
        "version",
        "evidence",
        "validation_flag",
        "private_hint",
        "receipt",
        "shadow_cost",
    ],
)
def test_public_tampering_is_rejected_before_any_private_read(
    good_bundle, monkeypatch, attack
):
    config, hypotheses, result, truth = copy.deepcopy(good_bundle)
    if attack == "budget":
        result["spent"] += 1
    elif attack == "prediction":
        result["bootstrap"]["events"][0]["predictions"]["0"][0] = "forged"
    elif attack == "edit":
        result["generation"]["proposals"][1]["edits"][0]["from"] = "forged"
    elif attack == "scope":
        result["generation"]["proposals"][1]["ir"]["scope"]["required_context"] = {}
    elif attack == "version":
        result["theory_events"][4]["parent_version"] = 999
    elif attack == "evidence":
        result["generation"]["discovery_evidence_ids"] = ["invented"]
    elif attack == "validation_flag":
        result["replacement_independent_validation_completed"] = False
    elif attack == "private_hint":
        config["protocol"]["boundary_episode"] = truth["boundary_episode"]
    elif attack == "receipt":
        next(e for e in result["monitoring"][0]["events"] if e["kind"] == "step")[
            "transition"
        ]["context"]["treatment_verification"] = "verified"
    elif attack == "shadow_cost":
        result["replacement_validation"]["plan"]["total_cost"] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("private bytes opened before public rejection")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_revision(config, hypotheses, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["evaluation"] is None


def test_wrong_private_boundary_fails_actual_runtime_replay(good_bundle, tmp_path):
    data = copy.deepcopy(good_bundle)
    data[3]["boundary_episode"] = 99
    audit = audit_bundle(data, tmp_path)
    assert audit["public_replay_valid"] and audit["private_truth_opened"]
    assert not audit["valid"]
    assert not audit["evaluation"]["actual_controls_match_private_reference"]


def test_raw_edit_semantics_are_independent_and_check_parent_values(good_bundle):
    generation = good_bundle[2]["generation"]
    original = copy.deepcopy(generation["parent_mechanism_ir"]["parameters"])
    for proposal in generation["proposals"]:
        assert _raw_edit(original, proposal["edits"]) == proposal["ir"]["parameters"]
    assert original == generation["parent_mechanism_ir"]["parameters"]
    edit = copy.deepcopy(generation["proposals"][1]["edits"])
    edit[0]["from"] = "forged"
    with pytest.raises(ValueError, match="parent value"):
        _raw_edit(original, edit)


@pytest.mark.parametrize("horizon", [1, 2, 3, 5])
def test_raw_bounded_product_matches_exhaustive_independent_word_execution(horizon):
    task = revision_task()
    spec = task.private_spec["after"]
    for hypothesis in task.bootstrap:
        program = hypothesis.compile().parameters
        expected = all(
            _rollout(program, word) == _factor_rollout(spec, word)
            for depth in range(1, horizon + 1)
            for word in itertools.product(program["actions"], repeat=depth)
        )
        assert _bounded_matches_factors(program, spec, horizon) == expected


@pytest.mark.parametrize(
    "kwargs", [{"budget": 15}, {"candidate_limit": 1}, {"search_budget": 1}]
)
def test_unresolved_budget_capacity_search_still_have_valid_audits(kwargs, tmp_path):
    data = bundle(**kwargs)
    assert audit_bundle(data, tmp_path)["valid"]
    assert not data[2]["replacement_independent_validation_completed"]


def test_benchmark_preserves_all_budget_failures_and_hashes_artifacts(tmp_path):
    import hashlib

    output = tmp_path / "benchmark"
    report = run_revision_benchmark(output=output, seeds=1, budget=1)
    assert report["all_independent_audits_valid"] and len(report["rows"]) == 14
    assert all(row["status"] == "unresolved" for row in report["rows"])
    assert all(not row["replacement_validated"] for row in report["rows"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_revision_benchmark(output=output, seeds=1, budget=1)
    assert (output / "report.json").read_bytes() == before


def test_cli_registers_new_command_and_validates_budgets():
    from typer.testing import CliRunner

    from minecraft_scientist.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["benchmark-revision", "--help"]).exit_code == 0
    assert (
        runner.invoke(
            app, ["benchmark-revision", "--output", "not-created", "--budget", "0"]
        ).exit_code
        != 0
    )


def test_shared_generator_bug_is_caught_by_independent_edit_interpreter(monkeypatch):
    import minecraft_scientist.agent.revision as agent_module
    from minecraft_scientist.discovery.revision import (
        StructuralProposal,
        StructuralSearch,
    )

    original = agent_module.enumerate_structural_edits

    def flawed(base, **kwargs):
        search = original(base, **kwargs)
        proposals = list(search.proposals)
        candidate = proposals[1]
        ir = candidate.ir.model_copy(deep=True)
        row = ir.parameters["transitions"][0]
        row["target"] = next(s for s in ir.parameters["states"] if s != row["target"])
        proposals[1] = StructuralProposal(candidate.proposal_id, ir, candidate.edits)
        return StructuralSearch(
            tuple(proposals), search.evaluations, search.complete, search.stop_reason
        )

    monkeypatch.setattr(agent_module, "enumerate_structural_edits", flawed)
    config, hypotheses, result, _truth = bundle(budget=50)
    assert result["generation"] is not None

    def forbidden(*args, **kwargs):
        raise AssertionError("private bytes opened after independent edit rejection")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_revision(config, hypotheses, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["failure"] == "independent edit journal mismatch"


def test_shared_compiler_bug_is_caught_by_raw_component_execution(monkeypatch):
    from minecraft_scientist.discovery.diagnosis import JointHypothesis

    original = JointHypothesis.compile

    def flawed(self):
        ir = original(self)
        row = ir.parameters["transitions"][0]
        row["output"] = next(o for o in ir.parameters["outputs"] if o != row["output"])
        return ir

    monkeypatch.setattr(JointHypothesis, "compile", flawed)
    config, hypotheses, result, _truth = bundle(budget=20)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "private bytes opened after independent compiler rejection"
        )

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_revision(config, hypotheses, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["failure"] == "bootstrap compiler disagrees with raw factors"
