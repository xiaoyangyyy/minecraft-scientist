"""Paid acquisition, public prior parity, qualified evidence and no fake crisis."""

import copy

import pytest
from audit_run import audit_adaptive
from paid_transfer import transfer_contexts
from test_matching import counter, view


def run_pair(*, states=3, budget=400, old_budget=120):
    old, new = counter(2), counter(states)
    runs = {
        m: transfer_contexts(
            view(old),
            view(new),
            old.actions,
            old.outputs,
            {a: 1 for a in old.actions},
            method=m,
            budget=budget,
            old_budget=old_budget,
            horizon=8,
            seed=1,
        )
        for m in ("from_scratch", "adaptive")
    }
    return runs


@pytest.mark.parametrize("states", [1, 2, 3, 4])
def test_both_methods_pay_identical_old_phase_and_expose_same_public_history(states):
    runs = run_pair(states=states, budget=3000)
    assert runs["from_scratch"]["old"] == runs["adaptive"]["old"]
    assert (
        runs["from_scratch"]["public_old_theory"]
        == runs["adaptive"]["public_old_theory"]
    )
    for method, result in runs.items():
        assert result["status"] == "two_contexts_finitely_validated", result[
            "stop_reason"
        ]
        assert (
            result["spent"]
            == sum(result[p]["run"]["spent"] for p in ("old", "new"))
            <= result["budget"]
        )
        assert (
            result["new"]["config"]["protocol"]["budget"]
            == result["budget"] - result["old"]["run"]["spent"]
        )
        assert result["new"]["run"]["learning_mode"] == method
        assert result["new"]["config"]["references"] == (
            [result["public_old_theory"]["ir"]] if method == "adaptive" else []
        )
        for p in ("old", "new"):
            phase = result[p]
            audit = audit_adaptive(
                phase["config"], phase["run"], None, public_only=True
            )
            assert audit["valid"] and not audit["private_truth_read_attempted"], audit
            assert phase["run"]["observation_tree"]["epoch"] == p + "-context"
        assert [e["status"] for e in result["theory_events"]] == [
            "proposed",
            "provisional",
            "accepted",
        ] * 2
        assert result["theory_events"][-1]["parent_version"] == 1
        assert result["public_old_theory"]["status"] == "accepted"
        assert all(
            e["status"] not in ("challenged", "superseded")
            for e in result["theory_events"]
        )
        for assignment in result["validation_assignments"]:
            assert not set(assignment["discovery_ids"]) & set(
                assignment["validation_ids"]
            )
            assert len(assignment["validation_ids"]) == 8
            assert all(
                q.startswith(assignment["context"] + "/")
                for q in assignment["discovery_ids"] + assignment["validation_ids"]
            )
        assert result["unannounced_online_mutation_claimed"] is False


@pytest.mark.parametrize("method", ["from_scratch", "adaptive"])
def test_tiny_total_budget_never_gets_free_old_theory_or_new_world_queries(method):
    old = counter(2)
    result = transfer_contexts(
        view(old),
        None,
        old.actions,
        old.outputs,
        {a: 1 for a in old.actions},
        method=method,
        budget=1,
    )
    assert (
        result["status"] == "unresolved"
        and result["stop_reason"] == "old_context_unresolved"
    )
    assert result["new"] is None and result["public_old_theory"] is None
    assert result["theory_events"] == [] and result["spent"] <= 1


def test_old_history_is_not_copied_into_new_epoch_as_evidence():
    result = run_pair()["adaptive"]
    old_ids = {
        e["query_id"] for e in result["old"]["run"]["events"] if e["kind"] == "evidence"
    }
    new_ids = {
        e["query_id"] for e in result["new"]["run"]["events"] if e["kind"] == "evidence"
    }
    # Local IDs may coincide, but all nodes are justified by actual new paid
    # dispatch; qualification keeps scientific provenance globally distinct.
    assert old_ids & new_ids
    assert all(
        set(n["evidence_ids"]) <= new_ids
        for n in result["new"]["run"]["observation_tree"]["nodes"]
    )
    old_assignment, new_assignment = result["validation_assignments"]
    assert not set(
        old_assignment["discovery_ids"] + old_assignment["validation_ids"]
    ) & set(new_assignment["discovery_ids"] + new_assignment["validation_ids"])


def test_full_two_context_event_and_theory_history_is_deterministic_and_owned():
    first, second = run_pair(), run_pair()
    assert first == second
    clone = copy.deepcopy(first)
    clone["adaptive"]["public_old_theory"]["ir"]["parameters"]["transitions"][0][
        "output"
    ] = "forged"
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [{"budget": True}, {"old_budget": 0}, {"horizon": 9}, {"method": "oracle"}],
)
def test_invalid_transfer_protocol_fails_before_world_dispatch(kwargs):
    with pytest.raises(ValueError, match="paid-context-transfer"):
        transfer_contexts(None, None, ("a",), ("0",), {"a": 1}, **kwargs)
