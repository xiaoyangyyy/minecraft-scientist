"""From-scratch and adaptive modes share one learner and paid random-Wp tester."""

import copy

import pytest
from audit_run import audit_adaptive
from learner import learn_adaptive
from reference_matching import ReferencePool, fingerprint
from test_audit_run import PrivateProbe
from test_matching import counter, view

from minecraft_scientist.discovery.active_automata import ObservationTree


def config_for(program, **protocol):
    return {
        "schema": "adaptive-config-v1",
        "actions": list(program.actions),
        "outputs": list(program.outputs),
        "costs": {a: 1 for a in program.actions},
        "references": [],
        "protocol": {
            "budget": 3000,
            "horizon": 8,
            "epoch": "cold",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        }
        | protocol,
    }


def test_empty_reference_pool_never_fabricates_prior_or_matching_evidence():
    pool = ReferencePool((), ("a", "b"))
    tree = ObservationTree(("a", "b"), ("0", "1"), epoch="new")
    before = tree.snapshot()
    assert (
        pool.keys == ()
        and pool.reference_ir == ()
        and pool.reference_digest == fingerprint([])
    )
    for method in (
        pool.rebuilding,
        pool.prioritized_promotion,
        pool.prioritized_separation,
        pool.refinement,
        pool.separation,
    ):
        assert method(tree, [()]) is None
    assert pool.matches(tree, ()) == pool.matches(tree, (), approximate=False) == ()
    assert pool.visits == 0 and tree.snapshot() == before
    with pytest.raises(ValueError, match="unknown observed"):
        pool.matches(tree, ("a",))


@pytest.mark.parametrize("states", [1, 2, 3, 4])
@pytest.mark.parametrize("shift", [False, True])
def test_cold_model_is_executable_paid_and_independently_audited(states, shift):
    program = counter(states, initial=int(shift and states > 1))
    config = config_for(program)
    result = learn_adaptive(
        view(program),
        program.actions,
        program.outputs,
        config["costs"],
        (),
        **config["protocol"],
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    assert (
        result["learning_mode"] == "from_scratch"
        and result["reference_models_used"] is False
    )
    assert result["reference_visits"] == 0 and result[
        "reference_digest"
    ] == fingerprint([])
    assert not any(e["kind"] == "reference_plan" for e in result["events"])
    assert not any(
        r["rule"] in ("R", "PP", "AMR", "APS", "AMS") for r in result["rules"]
    )
    assert all(
        b["plan"]["protocol"] == "random-wp-h-conditioned-v1"
        for b in result["conformance"]
    )
    audit = audit_adaptive(
        config, result, PrivateProbe(program.model_dump(mode="json"))
    )
    assert (
        audit["valid"] and audit["evaluation"]["provisional_behavior_correct_at_H"]
    ), audit
    assert result == learn_adaptive(
        view(program),
        program.actions,
        program.outputs,
        config["costs"],
        (),
        **config["protocol"],
    )


@pytest.mark.parametrize(
    "kwargs", [{"budget": 1}, {"horizon": 1}, {"operation_budget": 1}]
)
def test_cold_budget_and_scope_exhaustion_remain_unresolved(kwargs):
    program = counter(3)
    config = config_for(program, **kwargs)
    result = learn_adaptive(
        view(program),
        program.actions,
        program.outputs,
        config["costs"],
        (),
        **config["protocol"],
    )
    assert result["candidate_ir"] is None and result["status"] == "unresolved"
    assert audit_adaptive(
        config, result, PrivateProbe(program.model_dump(mode="json"))
    )["valid"]


def test_mode_claim_cannot_disguise_missing_references_as_adaptive_learning():
    program = counter(2)
    config = config_for(program)
    result = learn_adaptive(
        view(program),
        program.actions,
        program.outputs,
        config["costs"],
        (),
        **config["protocol"],
    )
    forged = copy.deepcopy(result)
    forged["learning_mode"] = "adaptive"
    audit = audit_adaptive(config, forged, PrivateProbe())
    assert not audit["valid"] and not audit["private_truth_read_attempted"]
