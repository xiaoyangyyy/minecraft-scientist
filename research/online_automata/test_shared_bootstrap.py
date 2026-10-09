"""Paid public priors, coverage, lineage and pre-private shared-fault gates."""

import copy

import pytest
import shared_bootstrap
from audit_online import audit_online
from online import online_relearn
from shared_bootstrap import bootstrap_config, learn_shared_bootstrap
from test_online_audit import PrivateProbe

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.world.revision import REVISION_CASES, revision_task


def family_for(task):
    return {
        "hypotheses": [h.model_dump(mode="json") for h in task.bootstrap],
        "calibration": list(task.calibration),
        "discovery_budget": 40,
        "shadow_count": 3,
    }


def matched_bundle(
    seed=0, case="unchanged", method="adaptive", boundary=25, **protocol
):
    task = revision_task(seed, case, boundary=boundary)
    config = {
        "schema": "online-relearning-config-v2",
        "method": method,
        "actions": list(task.costs),
        "outputs": list(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        "costs": task.costs,
        "initial_family": family_for(task),
        "protocol": {
            "budget": 600,
            "old_budget": 250,
            "horizon": 5,
            "seed": 20261008,
            "conformance_count": 8,
            "monitor_rounds": 4,
            "monitor_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        }
        | protocol,
    }
    result = online_relearn(
        task.view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        task.costs,
        method=method,
        initial_family=config["initial_family"],
        **config["protocol"],
    )
    return config, result, task.private_spec


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", REVISION_CASES)
def test_public_family_paid_old_model_covers_sparse_controller_paths(seed, case):
    config, result, truth = matched_bundle(seed, case)
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert audit["evaluation"]["old_behavior_correct_at_H"] is True
    old = result["old"]["run"]
    assert old["status"] == "provisional_validation_passed"
    assert old["spent"] == old["discovery"]["spent"] + old["validation"]["spent"] > 0
    assert any(
        o != config["outputs"][0]
        for q in old["validation"]["plan"]["queries"]
        for o in q["prediction"]
    )
    assert old["validation"]["plan"]["all_public_actions_covered"]
    assert not result["mechanism_change_warranted"]
    assert result["mechanism_attribution_status"] == "unresolved"
    if case == "unchanged":
        assert result["anomaly"] is None


@pytest.mark.parametrize("case", ["unchanged", "controller_fault", "state_insert"])
def test_methods_share_paid_initial_knowledge_evidence_and_monitoring(case):
    left, right = (
        matched_bundle(case=case, method="from_scratch"),
        matched_bundle(case=case),
    )
    for key in ("old", "monitoring", "public_old_theory", "anomaly"):
        assert left[1][key] == right[1][key]
    for config, result, truth in (left, right):
        audit = audit_online(config, result, PrivateProbe(truth))
        assert audit["valid"], audit
        if result["new"] is not None:
            spent = result["old"]["run"]["spent"] + sum(
                q["spent"] for b in result["monitoring"] for q in b["queries"]
            )
            assert (
                result["new"]["config"]["protocol"]["budget"]
                == config["protocol"]["budget"] - spent
            )
        assignment = result["validation_assignments"][0]
        assert len(assignment["validation_ids"]) == 3
        assert not set(assignment["validation_ids"]) & set(assignment["discovery_ids"])


@pytest.mark.parametrize("budget", [1, 5, 10, 15])
def test_unaffordable_discovery_or_full_validation_never_publishes_free_theory(budget):
    config, result, truth = matched_bundle(budget=budget)
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert result["public_old_theory"] is None and not result["theory_events"]
    assert result["spent"] <= budget
    validation = result["old"]["run"]["validation"]
    if validation is not None:
        assert not validation["runs"] and validation["spent"] == 0


@pytest.fixture(scope="module")
def matched_good():
    return matched_bundle()


@pytest.mark.parametrize(
    "attack",
    [
        "prior",
        "budget_bool",
        "cost_bool",
        "coverage",
        "validation_word",
        "validation_prediction",
        "accepted_ir",
        "flatten",
        "evidence_overlap",
        "factor_claim",
        "model_class",
        "failed_as_evidence",
    ],
)
def test_bootstrap_mutations_are_rejected_before_any_private_read(matched_good, attack):
    config, result, _ = copy.deepcopy(matched_good)
    old = result["old"]["run"]
    if attack == "prior":
        config["initial_family"]["hypotheses"] = config["initial_family"]["hypotheses"][
            :-1
        ]
    elif attack == "budget_bool":
        old["discovery"]["budget"] = True
    elif attack == "cost_bool":
        next(e for e in old["discovery"]["events"] if e["kind"] == "reset")["cost"] = (
            True
        )
    elif attack == "coverage":
        old["validation"]["plan"]["all_public_actions_covered"] = False
    elif attack == "validation_word":
        old["validation"]["plan"]["queries"][0]["actions"].reverse()
    elif attack == "validation_prediction":
        q = old["validation"]["plan"]["queries"][0]
        q["prediction"][-1] = next(
            o for o in config["outputs"] if o != q["prediction"][-1]
        )
    elif attack == "accepted_ir":
        old["candidate_ir"]["scope"]["certified_reset_only"] = False
    elif attack == "flatten":
        old["events"].pop()
    elif attack == "evidence_overlap":
        old["validation_query_ids"][0] = old["discovery"]["events"][0]["query_id"]
    elif attack == "factor_claim":
        result["mechanism_change_warranted"] = True
    elif attack == "model_class":
        old["discovery"]["remaining_behavior_classes"] = True
    else:
        e = next(e for e in old["discovery"]["events"] if e["kind"] == "step")
        e["transition"]["event"]["success"] = False
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert not audit["valid"] and probe.calls == 0, (attack, audit)


def test_shared_coverage_executor_fault_is_caught_by_independent_raw_plan(monkeypatch):
    original = shared_bootstrap.validate_shadow

    def faulty(*args, **kwargs):
        result = original(*args, **kwargs)
        result["plan"]["all_public_actions_covered"] = False
        return result

    monkeypatch.setattr(shared_bootstrap, "validate_shadow", faulty)
    config, result, _ = matched_bundle()
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert not audit["valid"] and probe.calls == 0


def test_shared_factor_compiler_bug_is_rejected_even_when_policy_replay_agrees(
    monkeypatch,
):
    from minecraft_scientist.discovery.diagnosis import JointHypothesis

    original = JointHypothesis.compile

    def faulty(self):
        ir = original(self)
        ir.parameters["transitions"][0]["output"] = ir.parameters["outputs"][1]
        return ir

    monkeypatch.setattr(JointHypothesis, "compile", faulty)
    config, result, _ = matched_bundle()
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert not audit["valid"] and probe.calls == 0
    assert "compiler" in audit["failure"]


@pytest.mark.parametrize("boundary", [6, 8, 12, 100])
def test_changes_during_or_after_bootstrap_are_not_given_to_agent(boundary):
    config, result, truth = matched_bundle(case="state_insert", boundary=boundary)
    assert "boundary" not in config["initial_family"]
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert result["spent"] <= config["protocol"]["budget"]


def test_known_bootstrap_control_failure_is_charged_without_eliminating_candidates():
    task = revision_task(0, "unchanged", boundary=100)
    family = family_for(task)
    config = bootstrap_config(
        tuple(task.costs),
        tuple(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        task.costs,
        family,
        budget=250,
        horizon=5,
        seed=20261008,
    )

    def failed(action):
        record = task.view.step(action)
        event = record.event.model_copy(update={"success": False})
        return record.model_copy(
            update={
                "controller_failure": "known-dispatch-failure",
                "event": event,
                "state_after": record.state_after.model_copy(
                    update={"last_event": event}
                ),
            }
        )

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=failed,
        begin_trial=task.view.begin_trial,
    )
    run = learn_shared_bootstrap(view, config)
    assert run["status"] == "unresolved" and run["validation"] is None
    assert run["discovery"]["stop_reason"] == "control_failure"
    assert run["spent"] == 1 + task.costs[family["calibration"][0]]
    assert len(run["discovery"]["remaining_candidates"]) == 9
    assert not any(e["kind"] == "evidence" for e in run["events"])


def test_public_family_is_detached_before_first_world_dispatch():
    task = revision_task(0, "unchanged", boundary=100)
    family = family_for(task)
    config = bootstrap_config(
        tuple(task.costs),
        tuple(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        task.costs,
        family,
        budget=250,
        horizon=5,
        seed=20261008,
    )

    def reset():
        family["hypotheses"].clear()
        family["calibration"].clear()
        return task.view.begin_trial()

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=task.view.step,
        begin_trial=reset,
    )
    run = learn_shared_bootstrap(view, config)
    assert run["status"] == "provisional_validation_passed"
    assert len(config["family"]["hypotheses"]) == 9


@pytest.mark.parametrize(
    "attack",
    ["extra", "bool", "duplicate", "alphabet", "calibration", "large_language"],
)
def test_invalid_public_prior_rejected_without_world_calls(attack):
    task = revision_task(0, "unchanged", boundary=100)
    family, horizon = family_for(task), 5
    if attack == "extra":
        family["private_boundary"] = 25
    elif attack == "bool":
        family["shadow_count"] = True
    elif attack == "duplicate":
        family["hypotheses"][1] = family["hypotheses"][0]
    elif attack == "alphabet":
        family["hypotheses"][0]["controller_ir"]["parameters"]["actions"][0] = (
            "semantic-name"
        )
    elif attack == "calibration":
        family["calibration"] = ["not-public"]
    else:
        horizon = 8
    with pytest.raises((ValueError, TypeError)):
        bootstrap_config(
            tuple(task.costs),
            tuple(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
            task.costs,
            family,
            budget=250,
            horizon=horizon,
            seed=20261008,
        )
