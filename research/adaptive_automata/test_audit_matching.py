"""Raw rule checks, recorded-history reconstruction, shared-fault rejection."""

import copy

import pytest
from audit_matching import check_reference_plan
from learner import learn_adaptive
from reference_matching import ReferencePool
from test_matching import counter, view

from minecraft_scientist.discovery.active_automata import ObservationTree, OutputQueries
from minecraft_scientist.theory.finite_state import FiniteStateProgram


def fixture(rule):
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    costs = {"a": 1, "b": 2}
    query = OutputQueries(view(program), tree, costs, budget=100, horizon=8)
    if rule == "R":
        plan = pool.rebuilding(tree, [()])
    elif rule == "AMR":
        plan = pool.refinement(tree, [()])
    else:
        query(("a",), "Ex")
        if rule == "AMS":
            plan = pool.separation(tree, [()])
        else:
            pool.dispatch(tree, query, pool.separation(tree, [()]))
            query(("b",), "Ex")
            query(("a", "b"), "Ex")
            plan = pool.prioritized_separation(tree, [(), ("a",)])
    assert plan is not None and plan.rule == rule
    snapshot = tree.snapshot()
    receipt = pool.dispatch(tree, query, plan)
    return (
        plan.record(),
        snapshot,
        pool.reference_ir,
        program.actions,
        costs,
        receipt["estimated_paid_cost"],
    )


@pytest.mark.parametrize("rule", ["R", "AMR", "APS", "AMS"])
def test_each_rule_is_applicable_under_independent_raw_public_checks(rule):
    check_reference_plan(*fixture(rule))


def replay_plan_checks(result, references, actions, costs):
    """Reconstruct raw prefix/provenance maps; never call ObservationTree."""
    outputs, provenance, plans = {(): None}, {(): set()}, {}
    checked = 0
    for event in result["events"]:
        if event["kind"] == "plan":
            assert event["query_id"] not in plans
            plans[event["query_id"]] = tuple(event["actions"])
        elif event["kind"] == "evidence":
            word, trace = plans[event["query_id"]], event["outputs"]
            assert len(word) == len(trace)
            for i, value in enumerate(trace, 1):
                prefix = word[:i]
                assert prefix not in outputs or outputs[prefix] == value
                outputs[prefix] = value
                provenance.setdefault(prefix, set()).add(event["query_id"])
        elif event["kind"] == "reference_plan":
            snapshot = {
                "epoch": result["observation_tree"]["epoch"],
                "nodes": [
                    {
                        "access": list(p),
                        "incoming_output": outputs[p],
                        "evidence_ids": sorted(provenance[p]),
                    }
                    for p in sorted(outputs, key=lambda p: (len(p), p))
                ],
            }
            check_reference_plan(
                event["plan"],
                snapshot,
                references,
                actions,
                costs,
                event["estimated_paid_cost"],
            )
            checked += 1
    return checked


@pytest.mark.parametrize("reference_states", [1, 2, 3, 4])
@pytest.mark.parametrize("truth_states", [1, 2, 3, 4])
def test_every_recorded_plan_is_valid_against_prior_public_evidence(
    reference_states, truth_states
):
    truth = counter(truth_states)
    references = (counter(reference_states).to_ir(),)
    costs = {a: 1 for a in truth.actions}
    result = learn_adaptive(
        view(truth),
        truth.actions,
        truth.outputs,
        costs,
        references,
        budget=3000,
        horizon=8,
    )
    assert result["status"] == "provisional_conformance_passed"
    checked = replay_plan_checks(
        result, [r.model_dump(mode="json") for r in references], truth.actions, costs
    )
    if reference_states > 1:
        assert checked > 0


@pytest.mark.parametrize(
    "field",
    [
        "epoch",
        "observation_digest",
        "reference_digest",
        "basis",
        "words",
        "rule",
        "guidance",
        "extra",
    ],
)
def test_plan_mutations_are_rejected(field):
    args = list(copy.deepcopy(fixture("R")))
    plan = args[0]
    mutations = {
        "epoch": "old",
        "observation_digest": "forged",
        "reference_digest": "forged",
        "basis": [["b"]],
        "words": [["b"]],
        "rule": "PP",
        "guidance": [],
        "extra": 1,
    }
    plan[field] = mutations[field]
    with pytest.raises(ValueError, match="independent"):
        check_reference_plan(*args)


@pytest.mark.parametrize("value", [True, False, 3.0, "3", 4])
def test_cost_requires_exact_integer_and_exact_virtual_prefix_accounting(value):
    args = list(fixture("R"))
    assert args[-1] == 3
    args[-1] = value
    with pytest.raises(ValueError, match="query-cost mismatch"):
        check_reference_plan(*args)


@pytest.mark.parametrize("rule,index", [("R", 2), ("AMR", 1), ("APS", 4), ("AMS", 3)])
def test_bool_reference_index_cannot_impersonate_zero(rule, index):
    args = list(copy.deepcopy(fixture(rule)))
    args[0]["guidance"][index][0] = False
    with pytest.raises(ValueError, match="guidance schema"):
        check_reference_plan(*args)


def test_shared_bad_matching_can_pass_learner_dispatch_but_not_independent_checker(
    monkeypatch,
):
    raw = counter(3).model_dump(mode="json")
    for row in raw["transitions"]:
        if row["action"] == "b":
            row["output"] = "0" if row["source"] == "s0" else "1"
    program = FiniteStateProgram.model_validate(raw)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    costs = {a: 1 for a in program.actions}
    query = OutputQueries(view(program), tree, costs, budget=100, horizon=8)
    query(("b",), "Ex")
    assert pool.matches(tree, ()) == ((0, "s0"),)
    assert (0, "s2") not in pool.matches(tree, ())
    # A fault shared by learner and policy replay: all states match. The actual
    # generator now chooses a wrong AMR pair with a genuinely unseen separator.
    monkeypatch.setattr(
        ReferencePool, "matches", lambda self, *args, **kwargs: self.keys
    )
    plan = pool.refinement(tree, [()])
    assert plan is not None and plan.rule == "AMR" and plan.words == (("a",),)
    snapshot = tree.snapshot()
    receipt = pool.dispatch(tree, query, plan)
    assert receipt["status"] == "executed" and query.spent > 2
    with pytest.raises(ValueError, match="approximate refinement precondition"):
        check_reference_plan(
            plan.record(),
            snapshot,
            pool.reference_ir,
            program.actions,
            costs,
            receipt["estimated_paid_cost"],
        )
