"""Independent word enumeration and paid failure semantics for mutation baseline."""

import itertools
import random

import pytest

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import _FiniteWorld

from .execution import validate_challengers
from .planning import Meter, behavior_body, make_plan, shortest_unused, table


def program(seed=0, *, single=False):
    rng = random.Random(seed)
    states = ["q0"] if single else ["q0", "q1"]
    return {
        "language": "total-mealy-v1",
        "states": states,
        "actions": ["a", "b"],
        "outputs": ["x", "y"],
        "initial_state": "q0",
        "transitions": [
            {
                "source": s,
                "action": a,
                "target": states[0] if single else rng.choice(states),
                "output": "x" if single else rng.choice(["x", "y"]),
            }
            for s, a in itertools.product(states, ["a", "b"])
        ],
    }


def reference(p, word):
    state, output = p["initial_state"], []
    for action in word:
        row = next(
            r
            for r in p["transitions"]
            if r["source"] == state and r["action"] == action
        )
        state = row["target"]
        output.append(row["output"])
    return tuple(output)


def ir(p):
    return FiniteStateProgram.model_validate(p).to_ir()


def view(p, *, fail=None):
    world = _FiniteWorld(FiniteStateProgram.model_validate(p))
    controls = []

    def reset():
        controls.append("reset")
        return world.begin_trial()

    def step(action):
        controls.append(action.target_id)
        result = world.step(action)
        if fail == "control":
            result = result.model_copy(
                update={
                    "intervention_compliance": False,
                    "controller_failure": "injected",
                }
            )
        if fail == "measurement":
            result = result.model_copy(
                update={"event": result.event.model_copy(update={"success": False})}
            )
        return result

    return AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=step,
        begin_trial=reset,
    ), controls


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("horizon", (1, 2, 3))
def test_minimum_unused_public_witness_matches_exhaustive_reference(seed, horizon):
    left, right = program(seed), program(seed + 37)
    lr, lt = table(left)
    rr, rt = table(right)
    costs, used = {"a": 9, "b": 1}, {("a",), ("b",), ("b", "b")}
    actual = shortest_unused(lr, lt, rr, rt, costs, used, horizon, Meter(200_000))
    candidates = [
        w
        for n in range(1, horizon + 1)
        for w in itertools.product(["a", "b"], repeat=n)
        if w not in used and reference(left, w) != reference(right, w)
    ]
    expected = (
        None
        if not candidates
        else min(candidates, key=lambda w: (sum(costs[a] for a in w), len(w), w))
    )
    assert actual == expected


def test_behavior_partition_matches_all_two_state_single_action_models():
    models = []
    for choices in itertools.product(range(4), repeat=2):
        models.append(
            {
                "language": "total-mealy-v1",
                "states": ["s0", "s1"],
                "actions": ["a"],
                "outputs": ["x", "y"],
                "initial_state": "s0",
                "transitions": [
                    {
                        "source": f"s{i}",
                        "action": "a",
                        "target": f"s{choice // 2}",
                        "output": ["x", "y"][choice % 2],
                    }
                    for i, choice in enumerate(choices)
                ],
            }
        )
    bodies = [behavior_body(*table(p), {"a": 1}, Meter(10000)) for p in models]
    for i, left in enumerate(models):
        for j, right in enumerate(models):
            expected = all(
                reference(left, ("a",) * n) == reference(right, ("a",) * n)
                for n in range(5)
            )
            assert (bodies[i] == bodies[j]) == expected


def test_exclusion_trie_does_not_prune_alternative_path_to_same_product_state():
    left, right = program(single=True), program(single=True)
    right["transitions"][0]["output"] = "y"
    actual = shortest_unused(
        *table(left),
        *table(right),
        {"a": 1, "b": 1},
        {("a",), ("a", "a"), ("a", "b")},
        2,
        Meter(10000),
    )
    assert actual == ("b", "a")


def test_public_evidence_eliminates_a_class_and_selects_unseen_alternative():
    p = program(single=True)
    plan = make_plan(
        ir(p),
        {"a": 1, "b": 1},
        {("a",)},
        [{"query_id": "q0", "actions": ["a"], "outputs": ["x"]}],
        budget=20,
        count=1,
        horizon=2,
    )
    assert plan["status"] == "ready"
    assert plan["statistics"]["syntactic_mutants"] == 2
    assert plan["statistics"]["behavior_classes"] == 2
    assert plan["statistics"]["evidence_eliminated_classes"] == 1
    assert plan["queries"][0]["actions"] == ["b"]
    assert plan["total_cost"] == 2
    assert not plan["all_H_certified"]


def test_filler_is_explicit_when_all_declared_mutants_have_been_observed():
    p = program(single=True)
    evidence = [{"query_id": a, "actions": [a], "outputs": ["x"]} for a in ("a", "b")]
    plan = make_plan(
        ir(p),
        {"a": 1, "b": 1},
        {("a",), ("b",)},
        evidence,
        budget=20,
        count=3,
        horizon=2,
    )
    assert plan["status"] == "ready"
    assert plan["statistics"]["surviving_classes"] == 0
    assert all(
        not q["challenged_classes"] and "filler" in q["selection"]
        for q in plan["queries"]
    )
    assert not plan["unique_causal_structure_claimed"]


@pytest.mark.parametrize("seed", range(8))
def test_names_and_latent_order_do_not_change_public_alphabet_relative_plan(seed):
    p = program(seed)
    names = {
        "q0": "opaque-8",
        "q1": "opaque-2",
        "a": "id-7",
        "b": "id-3",
        "x": "o-8",
        "y": "o-2",
    }
    renamed = {
        **p,
        "states": [names[s] for s in reversed(p["states"])],
        "actions": [names[a] for a in p["actions"]],
        "outputs": [names[o] for o in p["outputs"]],
        "initial_state": names[p["initial_state"]],
        "transitions": [
            {k: names[v] for k, v in row.items()} for row in reversed(p["transitions"])
        ],
    }
    a = make_plan(ir(p), {"a": 2, "b": 1}, (), (), budget=30, horizon=3)
    b = make_plan(
        ir(renamed), {names["a"]: 2, names["b"]: 1}, (), (), budget=30, horizon=3
    )
    assert a["status"] == b["status"] == "ready"
    assert a["statistics"] == b["statistics"]
    assert [[names[token] for token in q["actions"]] for q in a["queries"]] == [
        q["actions"] for q in b["queries"]
    ]
    assert [c["id"] for c in a["classes"]] == [c["id"] for c in b["classes"]]


@pytest.mark.parametrize(
    "constraint,kwargs",
    (
        ("operation", {"operation_budget": 1}),
        ("mutants", {"mutant_limit": 1}),
        ("money", {"budget": 0}),
    ),
)
def test_exhaustion_cannot_dispatch_or_accidentally_accept(constraint, kwargs):
    p = program(single=True)
    capability, controls = view(p)
    options = {"budget": 20, "horizon": 2} | kwargs
    result = validate_challengers(
        capability, ir(p), {"a": 1, "b": 1}, (), (), **options
    )
    assert result["status"] == "unresolved"
    assert result["spent"] == 0 and not controls


@pytest.mark.parametrize("failure", ("control", "measurement"))
def test_paid_execution_failures_do_not_reject_candidate(failure):
    p = program(single=True)
    capability, controls = view(p, fail=failure)
    result = validate_challengers(
        capability, ir(p), {"a": 1, "b": 1}, (), (), budget=20, count=1, horizon=2
    )
    assert result["status"] == "unresolved"
    assert result["spent"] == 2 and controls == ["reset", "a"]
    assert result["runs"][0]["stop_reason"] == failure + "_failure"


def test_fresh_valid_counterexample_rejects_frozen_candidate():
    p, truth = program(single=True), program(single=True)
    truth["transitions"][0]["output"] = "y"
    capability, controls = view(truth)
    result = validate_challengers(
        capability, ir(p), {"a": 1, "b": 1}, (), (), budget=20, count=1, horizon=2
    )
    assert result["status"] == "rejected"
    assert result["runs"][0]["status"] == "model_mismatch"
    assert controls == ["reset", "a"]
    assert result["plan"]["candidate_ir"] == ir(p).model_dump(mode="json")


def test_success_requires_all_fresh_queries_and_keeps_input_detached():
    p = program(single=True)
    candidate = ir(p)
    capability, controls = view(p)
    result = validate_challengers(
        capability, candidate, {"a": 1, "b": 1}, (), (), budget=20, horizon=2
    )
    assert result["status"] == "passed"
    assert len(result["runs"]) == 3
    assert result["spent"] == result["plan"]["total_cost"]
    assert controls.count("reset") == 3
    result["plan"]["candidate_ir"]["parameters"]["states"].clear()
    assert candidate.parameters["states"] == ["q0"]


@pytest.mark.parametrize("field", ("after", "world", "case", "boundary", "group"))
def test_private_fields_in_evidence_rejected(field):
    item = {"query_id": "q", "actions": ["a"], "outputs": ["x"], field: "secret"}
    with pytest.raises(ValueError):
        make_plan(
            ir(program(single=True)), {"a": 1, "b": 1}, {("a",)}, [item], budget=20
        )


def test_conflicting_or_candidate_inconsistent_evidence_is_not_silently_filtered():
    evidence = [{"query_id": "q", "actions": ["a"], "outputs": ["y"]}]
    plan = make_plan(
        ir(program(single=True)), {"a": 1, "b": 1}, {("a",)}, evidence, budget=20
    )
    assert plan["status"] == "unresolved"
    assert plan["stop_reason"] == "candidate_inconsistent_with_public_evidence"
    with pytest.raises(ValueError):
        make_plan(
            ir(program(single=True)),
            {"a": 1, "b": 1},
            {("a",)},
            evidence + [{"query_id": "q2", "actions": ["a"], "outputs": ["x"]}],
            budget=20,
        )
