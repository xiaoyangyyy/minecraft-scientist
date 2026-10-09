"""Paid real-prefix replay, candidate lineage and public-only policy auditing."""

from copy import deepcopy

import pytest

from minecraft_scientist.protocols import PrimitiveAction, TheoryStatus
from minecraft_scientist.theory.registry import TheoryRegistry
from research.structural_campaigns.runtime import PROTOCOL, learn, public_config
from research.structural_holdouts.generator import fixture, instantiate

from .audit import raw_score, verify_public
from .paired import prepare, run_branch
from .test_planning import ir, program, view


def simple_snapshot():
    p = program(single=True)
    capability, _ = view(p)
    observation = capability.begin_trial()
    transition = capability.step(PrimitiveAction(kind="interact", target_id="a"))
    registry = TheoryRegistry()
    root = registry.propose(
        "issued-response",
        ir(p),
        created_by="test",
        evidence_ids=(),
        validation_score=0.0,
        confidence=0.0,
    )
    history = [root.model_dump(mode="json")]
    for status in (
        TheoryStatus.PROVISIONAL,
        TheoryStatus.ACCEPTED,
        TheoryStatus.CHALLENGED,
    ):
        history.append(
            registry.transition("issued-response", 1, status).model_dump(mode="json")
        )
    return {
        "config": {
            "actions": ["a", "b"],
            "outputs": ["x", "y"],
            "costs": {"a": 1, "b": 1},
            "protocol": {
                "budget": 30,
                "horizon": 2,
                "seed": 1,
                "operation_budget": 200000,
            },
            "repair_policy": {"shadow_count": 3},
        },
        "prefix_controls": [
            {"kind": "reset", "observation": observation.model_dump(mode="json")},
            {"kind": "step", "transition": transition.model_dump(mode="json")},
        ],
        "prefix_spent": 2,
        "candidate_ir": ir(p).model_dump(mode="json"),
        "eligible": True,
        "ineligible_reason": None,
        "repair_epoch_base": 0,
        "root_theory_events": history,
        "prior_events": [{"kind": "evidence", "query_id": "discovery"}],
        "prior_validation_ids": [],
        "prior_counterexample_ids": [],
        "evidence": [{"query_id": "discovery", "actions": ["a"], "outputs": ["x"]}],
        "used_words": [["a"]],
    }


@pytest.mark.parametrize("policy", ("public_coverage", "public_mutation"))
def test_both_policies_pay_real_prefix_and_keep_inputs_detached(policy):
    snapshot = simple_snapshot()
    untouched = deepcopy(snapshot)
    capability, controls = view(program(single=True))
    result = run_branch(capability, snapshot, policy)
    assert result["status"] == "revision_finitely_validated"
    assert controls[:2] == ["reset", "a"]
    assert (
        result["prefix_spent"] == 2 and result["spent"] == 2 + result["tail"]["spent"]
    )
    assert len(result["tail"]["runs"]) == 3
    assert [e["status"] for e in result["theory_events"]][-2:] == [
        "accepted",
        "superseded",
    ]
    assert verify_public(snapshot, result)["valid"]
    assert snapshot == untouched


def test_prospective_counterexample_rejects_child_not_challenged_parent():
    truth = program(single=True)
    truth["transitions"][1]["output"] = "y"
    capability, controls = view(truth)
    snapshot = simple_snapshot()
    result = run_branch(capability, snapshot, "public_mutation")
    assert controls == ["reset", "a", "reset", "b"]
    assert result["status"] == "candidate_rejected" and result["candidate_ir"] is None
    assert result["theory_events"][-1]["status"] == "rejected"
    assert result["theory_events"][-1]["counterexample_ids"]
    assert [e["status"] for e in result["theory_events"] if e["version"] == 1][
        -1
    ] == "challenged"
    assert verify_public(snapshot, result)["valid"]


@pytest.mark.parametrize("failure", ("control", "measurement"))
def test_tail_failure_is_charged_and_leaves_child_provisional(failure):
    snapshot = simple_snapshot()
    base, _ = view(program(single=True))
    failing, _ = view(program(single=True), fail=failure)
    from minecraft_scientist.protocols import AgentWorldView

    reset_count = 0

    def reset():
        nonlocal reset_count
        reset_count += 1
        # Use one clock/world; transform only subsequent step receipts.
        return base.begin_trial()

    def step(action):
        result = base.step(action)
        if reset_count <= 1:
            return result
        if failure == "control":
            return result.model_copy(
                update={
                    "intervention_compliance": False,
                    "controller_failure": "injected",
                }
            )
        return result.model_copy(
            update={"event": result.event.model_copy(update={"success": False})}
        )

    capability = AgentWorldView(
        observe=base.observe,
        available=failing.available_actions,
        begin_trial=reset,
        step=step,
    )
    result = run_branch(capability, snapshot, "public_mutation")
    assert result["status"] == "unresolved" and result["spent"] == 4
    assert result["theory_events"][-1]["status"] == "provisional"
    assert not result["theory_events"][-1]["counterexample_ids"]
    assert verify_public(snapshot, result)["valid"]


def test_prefix_mismatch_is_not_a_theory_counterexample_or_free_replay():
    truth = program(single=True)
    truth["transitions"][0]["output"] = "y"
    capability, controls = view(truth)
    result = run_branch(capability, simple_snapshot(), "public_mutation")
    assert result["stop_reason"] == "real_prefix_replay_mismatch"
    assert result["spent"] == 2 and controls == ["reset", "a"]
    assert not result["theory_events"] and result["tail"] is None


@pytest.mark.parametrize(
    "mode",
    ("no_candidate", "prior_counterexample", "prior_unresolved", "compute_exhausted"),
)
def test_ineligible_and_compute_exhaustion_are_retained_not_promoted(mode):
    snapshot = simple_snapshot()
    if mode == "compute_exhausted":
        snapshot["config"]["protocol"]["operation_budget"] = 1
    else:
        snapshot["eligible"], snapshot["ineligible_reason"] = False, mode
        if mode == "no_candidate":
            snapshot["candidate_ir"] = None
        elif mode == "prior_counterexample":
            snapshot["prior_counterexample_ids"] = ["prior-q"]
    capability, controls = view(program(single=True))
    result = run_branch(capability, snapshot, "public_mutation")
    assert result["candidate_ir"] is None and controls == ["reset", "a"]
    assert result["spent"] == 2
    assert result["status"] == (
        "candidate_rejected" if mode == "prior_counterexample" else "unresolved"
    )
    assert verify_public(snapshot, result)["valid"]


@pytest.mark.parametrize(
    "tamper", ("fee", "promotion", "prefix", "query", "trailing_receipt")
)
def test_public_verifier_rejects_rehashed_semantic_tampering(tamper):
    snapshot = simple_snapshot()
    capability, _ = view(program(single=True))
    result = run_branch(capability, snapshot, "public_mutation")
    if tamper == "fee":
        result["spent"] -= 1
    elif tamper == "promotion":
        result["theory_events"][-2]["confidence"] = 0.9
    elif tamper == "prefix":
        result["prefix_spent"] -= 1
    elif tamper == "query":
        result["tail"]["plan"]["queries"][0]["prediction"] = ["y"]
    else:
        result["raw_controls"].append(deepcopy(result["raw_controls"][-1]))
        result["spent"] += 1
    with pytest.raises(ValueError):
        verify_public(snapshot, result)


@pytest.fixture(
    scope="module",
    params=("unchanged", "controller_change", "composed", "out_of_family"),
)
def generated(request):
    record = fixture(base_index=0, states=2, case=request.param, boundary=20)
    config = public_config(record["public"], "structural_ig", PROTOCOL)
    historical = learn(instantiate(record["private"]), config)
    return record, config, historical


def test_real_generated_prefix_excludes_coverage_and_reproduces_baseline(generated):
    record, config, historical = generated
    snapshot = prepare(config, historical)
    branch = run_branch(instantiate(record["private"]), snapshot, "public_coverage")
    assert verify_public(snapshot, branch)["valid"]
    assert raw_score(snapshot, branch, record["private"])["controls_match"]
    assert all(e["status"] != "superseded" for e in snapshot["root_theory_events"])
    if snapshot["eligible"]:
        assert branch["tail"] == historical["new"]["run"]["coverage_validation"]
        assert not any(
            e.get("query_id", "").startswith("repair-shadow-")
            for e in snapshot["prior_events"]
        )
        expected = historical["new"]["raw_controls"]
        assert branch["raw_controls"][-len(expected) :] == expected
    else:
        assert branch["tail"] is None


def test_generated_mutation_policy_is_publicly_replayable_and_raw_scored(generated):
    record, config, historical = generated
    snapshot = prepare(config, historical)
    branch = run_branch(instantiate(record["private"]), snapshot, "public_mutation")
    assert verify_public(snapshot, branch)["valid"]
    score = raw_score(snapshot, branch, record["private"])
    assert score["controls_match"] and branch["spent"] <= PROTOCOL["budget"]
    assert branch["prefix_spent"] == snapshot["prefix_spent"]
