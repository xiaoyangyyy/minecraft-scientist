"""Ordered-rule prototype: complete real traces, wrong references, bounded stops."""

import copy

import pytest
from audit_random_wp import check_random_wp_plan
from learner import learn_adaptive
from test_matching import counter, view

from minecraft_scientist.evaluation.identification import _exactly_equivalent, _rollout


@pytest.mark.parametrize("states", [1, 2, 3, 4])
@pytest.mark.parametrize("shift", [False, True])
def test_real_queries_rebuild_and_match_public_reference_without_private_truth(
    states, shift
):
    reference = counter(states)
    truth = counter(states, initial=int(shift and states > 1))
    result = learn_adaptive(
        view(truth),
        truth.actions,
        truth.outputs,
        {a: 1 for a in truth.actions},
        (reference.to_ir(),),
        budget=3000,
        horizon=8,
        conformance_count=8,
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    for batch in result["conformance"]:
        check_random_wp_plan(batch["plan"], {a: 1 for a in truth.actions})
    assert all(
        b["plan"]["protocol"] == "random-wp-h-conditioned-v1"
        for b in result["conformance"]
    )
    assert sum(e["kind"] == "conformance_plan" for e in result["events"]) == len(
        result["conformance"]
    )
    assert _exactly_equivalent(
        result["candidate_ir"]["parameters"], truth.model_dump(mode="json")
    )
    assert (
        result["reference_models_used"] and not result["all_allowed_queries_certified"]
    )
    assert result["spent"] == sum(
        e["cost"] for e in result["events"] if e["kind"] in ("step", "reset")
    )
    assert sum(e["kind"] == "reset" for e in result["events"]) == result["paid_queries"]
    # Evidence belongs exclusively to charged queries in the fresh epoch.
    query_ids = {e["query_id"] for e in result["events"] if e["kind"] == "evidence"}
    assert all(
        set(n["evidence_ids"]) <= query_ids for n in result["observation_tree"]["nodes"]
    )
    assert sum(r["rule"] == "rebuilding_closed" for r in result["rules"]) == 1
    closed = next(
        i for i, r in enumerate(result["rules"]) if r["rule"] == "rebuilding_closed"
    )
    assert all(r["rule"] not in ("R", "PP") for r in result["rules"][closed + 1 :])
    for event in result["events"]:
        if event["kind"] == "reference_plan":
            assert event["plan"]["epoch"] == "current"
            assert event["plan"]["reference_digest"] == result["reference_digest"]


@pytest.mark.parametrize("reference_states,truth_states", [(1, 2), (2, 3), (3, 2)])
def test_wrong_reference_guidance_never_promotes_unobserved_structure(
    reference_states, truth_states
):
    reference, truth = counter(reference_states), counter(truth_states)
    result = learn_adaptive(
        view(truth),
        truth.actions,
        truth.outputs,
        {a: 1 for a in truth.actions},
        (reference.to_ir(),),
        budget=3000,
        horizon=8,
        conformance_count=8,
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    assert _exactly_equivalent(
        result["candidate_ir"]["parameters"], truth.model_dump(mode="json")
    )
    # Raw public trajectory replay, independent of reference guidance.
    prefix = ()
    for event in result["events"]:
        if event["kind"] == "reset":
            prefix = ()
        elif event["kind"] == "step":
            record = event["transition"]
            prefix += (record["executed_action"]["target_id"],)
            assert (
                record["event"]["details"]["signal"]
                == _rollout(truth.model_dump(mode="json"), prefix)[-1]
            )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"budget": 1}, "reference_query_budget_exhausted"),
        ({"horizon": 1}, "reference_query_horizon_exhausted"),
        ({"operation_budget": 1}, "observation_operation_budget_exhausted"),
        ({"reference_operation_budget": 1}, "reference_operation_budget_exhausted"),
    ],
)
def test_resource_stops_keep_unresolved_no_candidate_and_charged_ledger(kwargs, reason):
    program = counter(3)
    result = learn_adaptive(
        view(program),
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        (program.to_ir(),),
        **kwargs,
    )
    assert result["status"] == "unresolved" and result["candidate_ir"] is None
    assert result["stop_reason"] == reason
    assert result["spent"] <= result["budget"]
    assert result["spent"] == sum(
        e["cost"] for e in result["events"] if e["kind"] in ("step", "reset")
    )


def test_multiple_public_models_and_their_input_mutations_do_not_supply_evidence():
    truth = counter(2)
    references = (counter(1).to_ir(), counter(3).to_ir(), truth.to_ir())
    result = learn_adaptive(
        view(truth),
        truth.actions,
        truth.outputs,
        {a: 1 for a in truth.actions},
        references,
        budget=3000,
        horizon=8,
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    assert _exactly_equivalent(
        result["candidate_ir"]["parameters"], truth.model_dump(mode="json")
    )
    assert result == learn_adaptive(
        view(truth),
        truth.actions,
        truth.outputs,
        {a: 1 for a in truth.actions},
        copy.deepcopy(references),
        budget=3000,
        horizon=8,
    )
