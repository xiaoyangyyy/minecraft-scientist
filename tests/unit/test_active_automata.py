"""Real-query learning checks; a new phase after the frozen H3 full suite."""

import copy
import itertools

import pytest

from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
    conformance_plan,
    consistency_conflict,
    fold_hypothesis,
    learn_apartness,
    process_conflict,
)
from minecraft_scientist.evaluation.identification import _exactly_equivalent, _rollout
from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import _FiniteWorld


def view_for(program):
    world = _FiniteWorld(program)
    return AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )


def counter_program(states=2, actions=("a", "b")):
    names = tuple(f"s{i}" for i in range(states))
    return FiniteStateProgram(
        states=names,
        actions=actions,
        outputs=("0", "1"),
        initial_state=names[0],
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


def populate(tree, program, horizon=4):
    for depth in range(1, horizon + 1):
        for word in itertools.product(program.actions, repeat=depth):
            tree.add_trace(
                word,
                _rollout(program.model_dump(mode="json"), word),
                epoch=tree.epoch,
                evidence_id=f"test-{depth}-{word}",
            )


@pytest.mark.parametrize("states", [1, 2, 3])
def test_known_residual_witnesses_are_exact_symmetric_and_constructive(states):
    program = counter_program(states)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    populate(tree, program, 5)
    prefixes = [()] + [("a",) * depth for depth in range(1, states)]
    for left, right in itertools.product(prefixes, repeat=2):
        witness = tree.witness(left, right)
        assert witness == tree.witness(right, left)
        if left == right:
            assert witness is None
        else:
            assert witness is not None
            assert tree.residual(left, witness) != tree.residual(right, witness)


def test_missing_edges_never_witness_apartness_and_trace_conflict_is_atomic():
    tree = ObservationTree(("a", "b"), ("0", "1"), epoch="new")
    tree.add_trace(("a", "a"), ("0", "1"), epoch="new", evidence_id="q0")
    tree.add_trace(("b",), ("0",), epoch="new", evidence_id="q1")
    assert tree.witness(("a",), ("b",)) is None
    before = tree.snapshot()
    with pytest.raises(LearnerStop, match="nonstationary_or_noisy_trace"):
        tree.add_trace(("a", "b"), ("1", "1"), epoch="new", evidence_id="bad")
    assert tree.snapshot() == before
    with pytest.raises(LearnerStop, match="epoch_mismatch"):
        tree.add_trace(("b", "a"), ("0", "1"), epoch="old", evidence_id="stale")
    assert tree.snapshot() == before


def test_observation_capacity_preflight_never_leaves_a_partial_trace():
    tree = ObservationTree(("a",), ("0",), epoch="new", node_limit=2)
    before = tree.snapshot()
    with pytest.raises(LearnerStop, match="capacity"):
        tree.add_trace(("a", "a"), ("0", "0"), epoch="new", evidence_id="q0")
    assert tree.snapshot() == before


@pytest.mark.parametrize("states", [1, 2, 3])
def test_folding_uses_observed_outputs_and_preserves_basis_access(states):
    program = counter_program(states)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    populate(tree, program, 6)
    basis = tuple(("a",) * depth for depth in range(states))
    hypothesis = fold_hypothesis(tree, basis)
    assert consistency_conflict(tree, hypothesis) is None
    assert _exactly_equivalent(
        program.model_dump(mode="json"), hypothesis.program.model_dump(mode="json")
    )
    for access in basis:
        assert hypothesis.reached_access(access) == access


def test_nonadequate_tree_cannot_invent_a_total_hypothesis():
    program = counter_program()
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    with pytest.raises(ValueError, match="complete"):
        fold_hypothesis(tree, ((),))
    tree.add_trace(("a",), ("0",), epoch="new", evidence_id="q0")
    with pytest.raises(ValueError, match="pairwise"):
        fold_hypothesis(tree, ((), ("a",)))


@pytest.mark.parametrize("seed", range(8))
def test_real_queries_learn_two_state_cycle_without_supplied_candidates(seed):
    program = counter_program()
    result = learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        budget=500,
        horizon=6,
        seed=seed,
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    assert _exactly_equivalent(
        result["candidate_ir"]["parameters"], program.model_dump(mode="json")
    )
    assert (
        not result["all_allowed_queries_certified"]
        and not result["unique_causal_structure_claimed"]
    )
    assert not result["reference_models_used"]
    assert any(rule["rule"] == "conformance_counterexample" for rule in result["rules"])
    assert any(rule["rule"] == "promotion" for rule in result["rules"])
    assert (
        result["spent"]
        == sum(e["cost"] for e in result["events"] if e["kind"] in ("reset", "step"))
        <= 500
    )
    assert all(
        rule["norm_after"] > rule["norm_before"]
        for rule in result["rules"]
        if "norm_before" in rule
    )


@pytest.mark.parametrize("states", [3, 4])
def test_multistate_learning_handles_binary_conflicts_and_separation(states):
    program = counter_program(states)
    result = learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        budget=2000,
        horizon=8,
        seed=0,
    )
    assert result["status"] == "provisional_conformance_passed", result["stop_reason"]
    assert _exactly_equivalent(
        result["candidate_ir"]["parameters"], program.model_dump(mode="json")
    )
    assert any(rule["rule"] == "separation" for rule in result["rules"])


@pytest.mark.parametrize("budget", [1, 2, 12])
def test_budget_failure_never_emits_a_certified_candidate(budget):
    program = counter_program()
    result = learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        budget=budget,
        horizon=5,
    )
    assert result["status"] == "unresolved" and result["candidate_ir"] is None
    assert result["spent"] <= budget


def test_input_cost_mutation_cannot_change_frozen_dispatch_costs():
    program = counter_program(states=1)
    view = view_for(program)
    costs = {a: 1 for a in program.actions}

    def reset():
        costs.update({a: 99 for a in costs})
        return view.begin_trial()

    mutating = AgentWorldView(
        observe=view.observe,
        available=view.available_actions,
        step=view.step,
        begin_trial=reset,
    )
    result = learn_apartness(
        mutating,
        program.actions,
        program.outputs,
        costs,
        budget=100,
        horizon=3,
        conformance_count=3,
    )
    assert result["status"] == "provisional_conformance_passed"
    assert all(e["cost"] == 1 for e in result["events"] if e["kind"] == "step")


@pytest.mark.parametrize("failure", ["compliance", "controller", "event"])
def test_verified_control_failure_is_charged_but_not_inserted_into_tree(failure):
    program = counter_program()
    view = view_for(program)

    def failed(action):
        record = view.step(action)
        if failure == "compliance":
            return record.model_copy(update={"intervention_compliance": False})
        if failure == "controller":
            return record.model_copy(update={"controller_failure": "injected"})
        return record.model_copy(
            update={"event": record.event.model_copy(update={"success": False})}
        )

    bad = AgentWorldView(
        observe=view.observe,
        available=view.available_actions,
        step=failed,
        begin_trial=view.begin_trial,
    )
    result = learn_apartness(
        bad,
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        budget=100,
    )
    assert result["stop_reason"] == "control_failure" and result["spent"] == 2
    assert len(result["observation_tree"]["nodes"]) == 1


def test_operation_budget_is_exact_not_only_checked_between_expensive_rules():
    program = counter_program()
    result = learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        {a: 1 for a in program.actions},
        operation_budget=1,
    )
    assert result["status"] == "unresolved"
    assert result["stop_reason"] == "observation_operation_budget_exhausted"
    assert result["observation_visits"] == 1


def test_scope_stop_does_not_fake_hidden_state_recovery():
    program = counter_program(states=8, actions=("a",))
    result = learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        {"a": 1},
        horizon=5,
        conformance_count=3,
    )
    assert not result["all_allowed_queries_certified"]
    if result["candidate_ir"] is not None:
        assert not _exactly_equivalent(
            result["candidate_ir"]["parameters"], program.model_dump(mode="json")
        )
    assert result["status"] in ("unresolved", "provisional_conformance_passed")


def test_query_cache_never_charges_or_dispatches_again_and_old_epoch_is_rejected():
    program = counter_program()
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    query = OutputQueries(
        view_for(program), tree, {a: 1 for a in program.actions}, budget=20, horizon=4
    )
    first = query(("a", "a"), "extension")
    before = query.spent
    assert query(("a", "a"), "cached") == first
    assert query.spent == before and query.cache_hits == 1
    with pytest.raises(LearnerStop, match="epoch"):
        tree.add_trace(("a", "a"), first, epoch="old", evidence_id="old-data")


def test_conformance_batch_is_frozen_uses_new_words_and_is_deterministic():
    program = counter_program(states=1)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    populate(tree, program, 1)
    hypothesis = fold_hypothesis(tree, ((),))
    plan = conformance_plan(
        tree, hypothesis, {a: 1 for a in program.actions}, horizon=4, count=4, seed=1
    )
    assert plan == conformance_plan(
        tree, hypothesis, {a: 1 for a in program.actions}, horizon=4, count=4, seed=1
    )
    assert all(not tree.contains(tuple(q["actions"])) for q in plan["queries"])
    assert plan["total_cost"] == 20
    frozen = copy.deepcopy(plan)
    populate(tree, program, 4)
    assert plan == frozen


def test_binary_counterexample_processor_reaches_invalid_frontier_merge():
    program = counter_program(states=3)
    tree = ObservationTree(program.actions, program.outputs, epoch="new")
    populate(tree, program, 1)
    hypothesis = fold_hypothesis(tree, ((),))
    # Three real observed steps create a deep conflict with the old one-state fold.
    word = ("a",) * 3
    tree.add_trace(
        word,
        _rollout(program.model_dump(mode="json"), word),
        epoch="new",
        evidence_id="counterexample",
    )
    conflict = consistency_conflict(tree, hypothesis)
    assert conflict is not None
    issued = []

    def query(sequence, reason, current):
        issued.append(sequence)
        tree.add_trace(
            sequence,
            _rollout(program.model_dump(mode="json"), sequence),
            epoch="new",
            evidence_id=f"shorten-{len(issued)}",
        )

    shortened = process_conflict(tree, [()], hypothesis, conflict, query)
    assert shortened in tree.frontier([()])
    assert tree.witness(shortened, ()) is not None
    assert len(issued) <= 2


def test_repeat_determinism_includes_events_tree_and_failed_conformance_history():
    program = counter_program(states=3)
    kwargs = {"budget": 1000, "horizon": 7, "conformance_count": 4, "seed": 9}
    costs = {a: 1 for a in program.actions}
    assert learn_apartness(
        view_for(program), program.actions, program.outputs, costs, **kwargs
    ) == learn_apartness(
        view_for(program), program.actions, program.outputs, costs, **kwargs
    )
