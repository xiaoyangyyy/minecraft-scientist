"""Knowledge/budget/tester matching, full H3 core and pre-private mutations."""

import copy

import pytest
from audit_online import audit_online
from matched_repair import MATCHED_METHODS, POLICY
from online import online_relearn
from test_online_audit import PrivateProbe
from test_shared_bootstrap import family_for

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.world.revision import revision_task


def campaign(method="structural_witness", case="redirect_start", seed=0, **protocol):
    task = revision_task(seed, case, boundary=25)
    config = {
        "schema": "online-relearning-config-v3",
        "method": method,
        "actions": list(task.costs),
        "outputs": list(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        "costs": task.costs,
        "initial_family": family_for(task),
        "repair_policy": copy.deepcopy(POLICY),
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
    run = online_relearn(
        task.view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        task.costs,
        method=method,
        initial_family=config["initial_family"],
        repair_policy=config["repair_policy"],
        **config["protocol"],
    )
    return config, run, task.private_spec


@pytest.fixture(scope="module")
def four():
    return {method: campaign(method) for method in MATCHED_METHODS}


def test_all_four_methods_have_identical_paid_prelude_and_remaining_budget(four):
    reference = four["from_scratch"][1]
    for method, (config, run, truth) in four.items():
        for key in ("old", "public_old_theory", "monitoring", "anomaly"):
            assert run[key] == reference[key], (method, key)
        audit = audit_online(config, run, PrivateProbe(truth))
        assert audit["valid"], (method, audit)
        assert audit["evaluation"]["old_behavior_correct_at_H"]
        assert run["anomaly"] is not None and run["new"] is not None
        paid = run["old"]["run"]["spent"] + sum(
            q["spent"] for b in run["monitoring"] for q in b["queries"]
        )
        assert (
            run["new"]["config"]["protocol"]["budget"]
            == config["protocol"]["budget"] - paid
        )
        if run["status"] == "revision_finitely_validated":
            assignment = run["validation_assignments"][-1]
            assert len(assignment["validation_ids"]) == 11
            assert not set(assignment["validation_ids"]) & set(
                assignment["discovery_ids"]
            )
            assert (
                run["new"]["run"]["random_validation"]["plan"]["seed"]
                == 20261008 + 500_000
            )
            assert run["new"]["run"]["coverage_validation"]["status"] == "passed"
        assert not run["mechanism_change_warranted"]


@pytest.mark.parametrize("method", MATCHED_METHODS)
def test_unaffordable_shared_protocol_never_gives_free_new_theory(method):
    config, run, truth = campaign(method, budget=240)
    audit = audit_online(config, run, PrivateProbe(truth))
    assert audit["valid"], audit
    assert run["spent"] <= 240
    if (
        run["new"] is not None
        and run["new"]["run"]["base_run"] is None
        and run["new"]["run"]["structural"] is None
    ):
        assert run["new"]["run"]["spent"] == 0
        assert run["new"]["run"]["candidate_ir"] is None


@pytest.mark.parametrize(
    "attack",
    [
        "policy",
        "repair_budget",
        "anchor",
        "old_prior",
        "anomaly",
        "spent_bool",
        "new_ir",
        "validation_ids",
        "random_seed",
        "coverage",
        "raw_edit",
        "candidate_admission",
    ],
)
def test_matched_policy_and_structural_mutations_fail_before_private_read(four, attack):
    config, run, _ = copy.deepcopy(four["structural_witness"])
    phase = run["new"]
    assert phase is not None
    new = phase["run"]
    if attack == "policy":
        config["repair_policy"]["shadow_count"] = True
    elif attack == "repair_budget":
        phase["config"]["protocol"]["budget"] += 1
    elif attack == "anchor":
        phase["config"]["anchor_index"] = 1
    elif attack == "old_prior":
        phase["config"]["old_ir"]["scope"]["certified_reset_only"] = False
    elif attack == "anomaly":
        phase["config"]["anomaly"]["observed"][0] = config["outputs"][1]
    elif attack == "spent_bool":
        new["spent"] = True
    elif attack == "new_ir":
        new["candidate_ir"]["scope"]["certified_reset_only"] = False
    elif attack == "validation_ids":
        new["validation_query_ids"][0] = new["structural"]["discovery"]["events"][0][
            "query_id"
        ]
    elif attack == "random_seed":
        new["random_validation"]["plan"]["seed"] += 1
    elif attack == "coverage":
        new["coverage_validation"]["plan"]["all_public_actions_covered"] = not new[
            "coverage_validation"
        ]["plan"]["all_public_actions_covered"]
    elif attack == "raw_edit":
        proposal = next(
            p for p in new["structural"]["generation"]["proposals"] if p["edits"]
        )
        proposal["edits"][0]["from"] = "not-a-parent-row"
    else:
        new["structural"]["generation"]["admitted_joint_indices"].pop()
    probe = PrivateProbe()
    audit = audit_online(config, run, probe)
    assert not audit["valid"] and probe.calls == 0, (attack, audit)


@pytest.mark.parametrize("method", ["structural_witness", "structural_ig"])
def test_full_structural_core_reused_for_state_insertion_and_grammar_mismatch(method):
    config, run, truth = campaign(method, "state_insert")
    audit = audit_online(config, run, PrivateProbe(truth))
    assert audit["valid"], audit
    structural = run["new"]["run"]["structural"]
    assert structural["generation"]["proposals"]
    assert any(len(p["edits"]) == 2 for p in structural["generation"]["proposals"])


def test_reported_failure_during_final_validation_keeps_child_provisional_not_rejected(
    monkeypatch,
):
    import matched_repair

    active = False
    original = matched_repair.random_wp_plan

    def mark_final(*args, **kwargs):
        nonlocal active
        active = True
        return original(*args, **kwargs)

    monkeypatch.setattr(matched_repair, "random_wp_plan", mark_final)
    task = revision_task(0, "redirect_start", boundary=25)
    config, _, truth = campaign()
    active = False

    def step(action):
        record = task.view.step(action)
        if not active:
            return record
        event = record.event.model_copy(update={"success": False})
        return record.model_copy(
            update={
                "event": event,
                "state_after": record.state_after.model_copy(
                    update={"last_event": event}
                ),
            }
        )

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=step,
        begin_trial=task.view.begin_trial,
    )
    run = online_relearn(
        view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        task.costs,
        method=config["method"],
        initial_family=config["initial_family"],
        repair_policy=config["repair_policy"],
        **config["protocol"],
    )
    audit = audit_online(config, run, PrivateProbe(truth))
    assert audit["valid"], audit
    new = run["new"]["run"]
    assert new["proposed_ir"] is not None and new["candidate_ir"] is None
    assert new["validation_counterexample_ids"] == []
    assert [e["status"] for e in run["theory_events"] if e["version"] == 2] == [
        "proposed",
        "provisional",
    ]
    assert audit["evaluation"]["known_failed_dispatches_excluded"] == 1


def test_fabricated_validation_counterexample_cannot_open_private_truth(four):
    config, run, _ = copy.deepcopy(four["structural_witness"])
    new = run["new"]["run"]
    new["validation_counterexample_ids"] = new["validation_query_ids"][:1]
    probe = PrivateProbe()
    audit = audit_online(config, run, probe)
    assert not audit["valid"] and probe.calls == 0
