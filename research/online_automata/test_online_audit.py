"""Public/private gate, global clocks and joint-factor behavioral scoring."""

import copy
import json

import pytest
from audit_online import audit_online
from online import online_relearn
from worlds import online_task


class PrivateProbe:
    def __init__(self, truth=None):
        self.truth = truth
        self.calls = 0

    def read_text(self, **kwargs):
        self.calls += 1
        return json.dumps(self.truth)


def bundle(method="adaptive", case="state_growth", boundary=16, **protocol):
    task = online_task(0, case, boundary=boundary)
    config = {
        "schema": "online-relearning-config-v1",
        "method": method,
        "actions": list(task.actions),
        "outputs": list(task.outputs),
        "costs": task.costs,
        "protocol": {
            "budget": 400,
            "old_budget": 120,
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
        task.actions,
        task.outputs,
        task.costs,
        method=method,
        **config["protocol"],
    )
    return config, result, task.private


@pytest.fixture(scope="module")
def good():
    return bundle()


@pytest.mark.parametrize("method", ["adaptive", "from_scratch"])
@pytest.mark.parametrize(
    "case", ["state_growth", "redirect", "output_change", "unchanged"]
)
def test_actual_online_public_then_private_scoring(method, case):
    config, result, truth = bundle(method, case)
    probe = PrivateProbe(truth)
    audit = audit_online(config, result, probe)
    assert audit["valid"] and probe.calls == 1, audit
    assert audit["evaluation"]["old_behavior_correct_at_H"]
    if case != "unchanged":
        assert audit["evaluation"]["new_behavior_correct_at_H"]
        assert audit["evaluation"]["post_boundary_anomaly_detected"]
    else:
        assert not audit["evaluation"]["behavior_change_within_H"]


@pytest.mark.parametrize("boundary", [0, 2, 8, 1000])
def test_changes_during_bootstrap_or_after_monitor_window_remain_legitimate(boundary):
    config, result, truth = bundle(boundary=boundary)
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    if boundary == 0:
        assert audit["evaluation"]["old_behavior_correct_at_H"] is False
        assert audit["evaluation"]["old_queries_after_private_boundary"] > 0


@pytest.mark.parametrize("budget", [1, 69, 75, 110])
def test_paid_budget_exhaustion_valid_but_never_free_replacement(budget):
    config, result, truth = bundle(budget=budget)
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert result["spent"] <= budget


@pytest.mark.parametrize(
    "attack",
    [
        "global_clock",
        "local_clock",
        "raw_action",
        "receipt_reuse",
        "raw_signal",
        "monitor_cost",
        "monitor_prediction",
        "monitor_plan",
        "monitor_evidence",
        "anomaly",
        "anomaly_deleted",
        "total_cost",
        "repair_budget",
        "prior",
        "parent",
        "old_content",
        "status",
        "validation_overlap",
        "claim",
    ],
)
def test_any_public_mutation_is_rejected_before_private_read(good, attack):
    config, result, _ = copy.deepcopy(good)
    q = result["monitoring"][0]["queries"][0]
    if attack == "global_clock":
        q["raw_controls"][0]["observation"]["episode_id"] = 0
    elif attack == "local_clock":
        q["events"][1]["observation"]["episode_id"] = 50
    elif attack == "raw_action":
        action = q["raw_controls"][1]["transition"]["executed_action"]
        action["target_id"] = next(
            a for a in config["actions"] if a != action["target_id"]
        )
    elif attack == "receipt_reuse":
        q["raw_controls"][1]["transition"]["transition_id"] = result["old"][
            "raw_controls"
        ][1]["transition"]["transition_id"]
    elif attack == "raw_signal":
        q["raw_controls"][1]["transition"]["event"]["details"]["signal"] = "forged"
    elif attack == "monitor_cost":
        q["events"][2]["cost"] = True
    elif attack == "monitor_prediction":
        q["events"][0]["prediction"] = []
    elif attack == "monitor_plan":
        result["monitoring"][0]["plan"]["seed"] += 1
    elif attack == "monitor_evidence":
        q["events"][-1]["outputs"] = []
    elif attack == "anomaly":
        result["anomaly"]["episode_id"] = 0
    elif attack == "anomaly_deleted":
        result["anomaly"] = None
    elif attack == "total_cost":
        result["spent"] -= q["spent"]
    elif attack == "repair_budget":
        result["new"]["config"]["protocol"]["budget"] = config["protocol"]["budget"]
    elif attack == "prior":
        result["new"]["config"]["references"] = []
    elif attack == "parent":
        result["theory_events"][-2]["parent_version"] = None
    elif attack == "old_content":
        result["theory_events"][-1]["ir"]["parameters"]["initial_state"] = "forged"
    elif attack == "status":
        result["theory_events"][3]["status"] = "superseded"
    elif attack == "validation_overlap":
        result["validation_assignments"][-1]["discovery_ids"].append(
            result["validation_assignments"][-1]["validation_ids"][0]
        )
    else:
        result["mechanism_change_warranted"] = True
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert (
        not audit["valid"]
        and probe.calls == 0
        and not audit["private_truth_read_attempted"]
    ), audit


def test_private_read_cannot_mutate_owned_public_snapshot(good):
    config, result, truth = copy.deepcopy(good)

    class MutatingProbe(PrivateProbe):
        def read_text(self, **kwargs):
            result["new"]["run"]["candidate_ir"]["parameters"]["transitions"][0][
                "output"
            ] = "forged"
            return super().read_text(**kwargs)

    audit = audit_online(config, result, MutatingProbe(truth))
    assert audit["valid"] and audit["evaluation"]["new_behavior_correct_at_H"], audit


def test_missing_private_file_is_attempted_not_opened(good, tmp_path):
    config, result, _ = good
    audit = audit_online(config, result, tmp_path / "missing.json")
    assert audit["all_public_valid"] and audit["private_truth_read_attempted"]
    assert not audit["valid"] and not audit["private_truth_opened"]
