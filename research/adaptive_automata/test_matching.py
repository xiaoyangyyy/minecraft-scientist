"""Independent edge-level/reference-witness checks, with real charged queries."""

import copy
import itertools
from dataclasses import replace
from fractions import Fraction

import pytest
from reference_matching import ReferencePool, fingerprint

from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
)
from minecraft_scientist.evaluation.identification import _rollout
from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import _FiniteWorld


def counter(states=2, actions=("a", "b"), initial=0):
    names = tuple(f"s{i}" for i in range(states))
    return FiniteStateProgram(
        states=names,
        actions=actions,
        outputs=("0", "1"),
        initial_state=names[initial],
        transitions=tuple(
            {
                "source": s,
                "action": a,
                "target": names[(i + 1) % states] if a == actions[0] else s,
                "output": "1" if a == actions[0] and i == states - 1 else "0",
            }
            for i, s in enumerate(names)
            for a in actions
        ),
    )


def view(program):
    world = _FiniteWorld(program)
    return AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )


def populate(tree, program, depth):
    raw = program.model_dump(mode="json")
    for n in range(1, depth + 1):
        for word in itertools.product(tree.actions, repeat=n):
            tree.add_trace(
                word,
                _rollout(raw, word),
                epoch=tree.epoch,
                evidence_id=f"observed-{word!r}",
            )


def raw_outputs(program, word, state):
    table = {(r["source"], r["action"]): r for r in program["transitions"]}
    trace = []
    for action in word:
        row = table.get((state, action))
        if row is None:
            return tuple(trace), False
        trace.append(row["output"])
        state = row["target"]
    return tuple(trace), True


@pytest.mark.parametrize("states", [1, 2, 3, 4])
def test_reference_pair_witnesses_are_symmetric_minimal_and_actual(states):
    program = counter(states)
    pool = ReferencePool((program.to_ir(),), program.actions)
    raw = program.model_dump(mode="json")
    for left, right in itertools.product(pool.keys, repeat=2):
        witness = pool.separator(left, right)
        assert witness == pool.separator(right, left)
        if left == right:
            assert witness is None
            continue
        assert witness is not None
        assert raw_outputs(raw, witness, left[1]) != raw_outputs(raw, witness, right[1])
        for n in range(len(witness)):
            assert all(
                raw_outputs(raw, w, left[1]) == raw_outputs(raw, w, right[1])
                for w in itertools.product(program.actions, repeat=n)
            )
        assert witness in pool.identifiers(left) and witness in pool.identifiers(right)


def test_matching_degree_counts_edges_after_a_mismatch_not_correct_whole_traces():
    program = counter(states=1, actions=("a",))
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    tree.add_trace(("a",) * 3, ("0", "1", "1"), epoch="new", evidence_id="new-world")
    degree = pool.degree(tree, (), pool.keys[0])
    assert degree.agreeing_edges == 2 and degree.observed_edges == 3
    assert degree.score == Fraction(2, 3)
    assert pool.matches(tree, (), approximate=False) == ()
    assert pool.matches(tree, ()) == pool.keys
    assert pool.observed_separator(tree, (), pool.keys[0]) == ("a",)


@pytest.mark.parametrize("states", [2, 3, 4])
def test_full_edge_degrees_and_exact_matches_equal_independent_public_residuals(states):
    program = counter(states)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    populate(tree, program, states + 1)
    nodes = {tuple(n["access"]): n["incoming_output"] for n in tree.snapshot()["nodes"]}
    raw = program.model_dump(mode="json")
    for prefix in ((), ("a",), ("b",)):
        for key in pool.keys:
            edges = [
                p[len(prefix) :]
                for p in nodes
                if len(p) > len(prefix) and p[: len(prefix)] == prefix
            ]
            total = len(edges)
            matching = sum(
                raw_outputs(raw, w, key[1])[0][-1] == nodes[prefix + w] for w in edges
            )
            degree = pool.degree(tree, prefix, key)
            assert (degree.agreeing_edges, degree.observed_edges) == (matching, total)
            assert (key in pool.matches(tree, prefix, approximate=False)) == (
                matching == total
            )


def test_empty_evidence_retains_all_ties_without_reference_trace_insertion():
    program = counter(3)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="fresh")
    before = tree.snapshot()
    assert pool.matches(tree, ()) == pool.keys
    assert pool.matches(tree, (), approximate=False) == pool.keys
    assert all(pool.degree(tree, (), k).observed_edges == 0 for k in pool.keys)
    assert tree.snapshot() == before


def test_partial_alphabet_matching_is_common_only_but_reference_total_apartness_is_not():
    first, second = counter(1, ("a",)), counter(1, ("b",))
    pool = ReferencePool((first.to_ir(), second.to_ir()), ("a", "b"))
    tree = ObservationTree(("a", "b"), ("0", "1"), epoch="new")
    tree.add_trace(("a",), ("1",), epoch="new", evidence_id="a")
    tree.add_trace(("b",), ("0",), epoch="new", evidence_id="b")
    assert pool.matches(tree, (), approximate=False) == ((0, "s0"),)
    assert pool.separator((0, "s0"), (1, "s0")) == ("a",)
    assert pool.observed_separator(tree, (), (0, "s0")) is None
    assert pool.observed_separator(tree, (), (1, "s0")) == ("b",)


def test_reference_models_are_owned_and_public_snapshots_are_detached():
    ir = counter(2).to_ir()
    pool = ReferencePool((ir,), ("a", "b"))
    before = copy.deepcopy(pool.reference_ir)
    digest = pool.reference_digest
    ir.parameters["transitions"][0]["output"] = "mutated"
    pool.reference_ir[0]["parameters"]["transitions"][0]["output"] = "mutated"
    assert pool.reference_ir == before
    assert pool.reference_digest == digest == fingerprint(before)


def test_state_cover_is_prefix_closed_reachable_and_ordered_independently_of_state_names():
    program = counter(4)
    pool = ReferencePool((program.to_ir(),), program.actions)
    cover = pool.state_cover(0)
    assert cover == tuple(("a",) * i for i in range(4))
    assert {pool.reached(0, w) for w in cover} == set(pool.keys)
    assert all(w[:n] in cover for w in cover for n in range(len(w)))


def test_rebuilding_preflight_accounts_for_prefix_cache_inside_the_query_pair():
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(view(program), tree, {"a": 1, "b": 1}, budget=3, horizon=5)
    plan = pool.rebuilding(tree, [()])
    assert plan.rule == "R" and plan.words == (("a", "a"), ("a",))
    receipt = pool.dispatch(tree, query, plan)
    assert receipt["estimated_paid_cost"] == query.spent == 3
    assert query.queries == 1 and query.cache_hits == 1
    assert pool.prioritized_promotion(tree, [()]) == ("a",)


@pytest.mark.parametrize("change", ["epoch", "evidence", "reference_digest"])
def test_stale_plans_stop_before_any_world_dispatch(change):
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(view(program), tree, {"a": 1, "b": 1}, budget=10, horizon=5)
    plan = pool.rebuilding(tree, [()])
    if change == "epoch":
        plan = replace(plan, epoch="old")
    elif change == "reference_digest":
        plan = replace(plan, reference_digest="forged")
    else:
        tree.add_trace(("b",), ("0",), epoch="new", evidence_id="later")
    with pytest.raises(LearnerStop, match="stale_reference_plan_or_epoch"):
        pool.dispatch(tree, query, plan)
    assert query.spent == 0 and query.events == []


@pytest.mark.parametrize(
    "budget,horizon,reason", [(2, 5, "budget"), (10, 1, "horizon")]
)
def test_unaffordable_or_out_of_scope_reference_pairs_never_partially_dispatch(
    budget, horizon, reason
):
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(
        view(program), tree, {"a": 1, "b": 1}, budget=budget, horizon=horizon
    )
    with pytest.raises(LearnerStop, match=f"reference_query_{reason}_exhausted"):
        pool.dispatch(tree, query, pool.rebuilding(tree, [()]))
    assert (
        query.spent == 0 and query.events == [] and len(tree.snapshot()["nodes"]) == 1
    )


def test_reference_compute_budget_is_exact_inside_search_not_only_between_rules():
    program = counter(3)
    pool = ReferencePool((program.to_ir(),), program.actions, operation_budget=1)
    with pytest.raises(LearnerStop, match="reference_operation_budget_exhausted"):
        pool.separator(pool.keys[0], pool.keys[1])
    assert pool.visits == 1


def test_refinement_is_a_real_query_and_recomputed_as_new_evidence_arrives():
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(view(program), tree, {"a": 1, "b": 1}, budget=10, horizon=5)
    plan = pool.refinement(tree, [()])
    assert plan.rule == "AMR" and plan.words == (("a",),)
    assert pool.matches(tree, ()) == pool.keys
    pool.dispatch(tree, query, plan)
    assert query.spent == 2
    assert pool.matches(tree, ()) == ((0, "s0"),)
    assert pool.refinement(tree, [()]) is None


def test_ams_and_aps_require_actual_public_separation_not_model_identity():
    program = counter(2)
    pool = ReferencePool((program.to_ir(),), program.actions)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(view(program), tree, {"a": 1, "b": 1}, budget=30, horizon=5)
    query(("a",), "Ex")
    assert pool.separation(tree, [()]).rule == "AMS"
    assert pool.prioritized_separation(tree, [()]) is None
    pool.dispatch(tree, query, pool.separation(tree, [()]))
    basis = [(), ("a",)]
    query(("b",), "Ex")
    query(("a", "b"), "Ex")
    plan = pool.prioritized_separation(tree, basis)
    assert plan is not None and plan.rule == "APS"
    child = tuple(plan.guidance[1])
    before = tree.compatible(child, basis)
    pool.dispatch(tree, query, plan)
    assert len(before) == 2 and len(tree.compatible(child, basis)) < 2
