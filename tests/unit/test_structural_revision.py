"""Bounded structural repair, prospective validation and honest failure boundaries."""

import copy
import itertools
import random

import pytest

from minecraft_scientist.agent.revision import control_runs, revise_online
from minecraft_scientist.discovery.discrimination import DiscriminationSpace
from minecraft_scientist.discovery.revision import enumerate_structural_edits
from minecraft_scientist.evaluation.identification import _rollout
from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import (
    FiniteStateProgram,
    bounded_behavior_classes,
    shortest_witness,
)
from minecraft_scientist.theory.ir import TheoryInterpreter
from minecraft_scientist.theory.shadow import validate_shadow
from minecraft_scientist.world.revision import revision_task


def run_task(task, **kwargs):
    return revise_online(
        task.view, task.bootstrap, task.calibration, task.costs, **kwargs
    )


def program_for(task):
    return FiniteStateProgram.from_ir(task.bootstrap[0].compile())


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("horizon", [0, 1, 3])
def test_partition_matches_independent_exhaustive_raw_signatures(seed, horizon):
    rng = random.Random(seed)
    programs = tuple(
        FiniteStateProgram.model_validate(
            {
                "states": ["s0", "s1", "s2"],
                "actions": ["a", "b"],
                "outputs": ["0", "1"],
                "initial_state": "s0",
                "transitions": [
                    {
                        "source": state,
                        "action": action,
                        "target": rng.choice(["s0", "s1", "s2"]),
                        "output": rng.choice(["0", "1"]),
                    }
                    for state, action in itertools.product(
                        ["s0", "s1", "s2"], ["a", "b"]
                    )
                ],
            }
        )
        for _ in range(6)
    )
    programs += (programs[0],)
    words = tuple(
        word
        for depth in range(1, horizon + 1)
        for word in itertools.product(["a", "b"], repeat=depth)
    )
    expected = {}
    for index, program in enumerate(programs):
        signature = tuple(_rollout(program.model_dump(mode="json"), w) for w in words)
        expected.setdefault(signature, []).append(index)
    assert bounded_behavior_classes(programs, horizon) == tuple(
        tuple(indices) for indices in expected.values()
    )


def test_horizon_equivalence_does_not_claim_global_equivalence():
    task = revision_task()
    left = program_for(task)
    right = FiniteStateProgram.from_ir(task.bootstrap[3].compile())
    assert bounded_behavior_classes((left, right), 2) == ((0, 1),)
    assert shortest_witness(left, right, task.costs) is not None
    assert bounded_behavior_classes((left, right), 5) == ((0,), (1,))
    with pytest.raises(ValueError):
        DiscriminationSpace(
            (left.to_ir(), right.to_ir()),
            task.costs,
            horizon=5,
            equivalence_horizon=2,
        )


def test_capacity_is_explicit_not_silent_truncation():
    task = revision_task()
    candidates = (program_for(task).to_ir(),) * 13
    with pytest.raises(ValueError):
        DiscriminationSpace(candidates, task.costs)
    space = DiscriminationSpace(
        candidates, task.costs, horizon=5, candidate_limit=512, equivalence_horizon=5
    )
    assert len(space.programs) == 13
    assert space.classes == ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5", 65])
def test_runtime_rejects_invalid_certified_horizon(bad):
    ir = program_for(revision_task()).to_ir()
    ir.scope["intervention_horizon"] = bad
    with pytest.raises(ValueError):
        TheoryInterpreter.predict(ir, sequence=())


def test_runtime_enforces_reset_only_and_horizon():
    program = program_for(revision_task())
    ir = program.to_ir()
    ir.scope.update(intervention_horizon=2, certified_reset_only=True)
    assert TheoryInterpreter.predict(ir, sequence=(program.actions[0],))["outputs"]
    for inputs in (
        {"sequence": program.actions},
        {"sequence": (), "state": program.initial_state},
    ):
        with pytest.raises(ValueError):
            TheoryInterpreter.predict(ir, **inputs)


def test_edit_grammar_complete_and_owned_with_explicit_exhaustion():
    base = revision_task().bootstrap[0].mechanism_ir.model_copy(deep=True)
    base.scope["custom"] = {"value": 0}
    original = base.model_dump(mode="json")
    complete = enumerate_structural_edits(base)
    assert complete.complete and complete.evaluations == len(complete.proposals) == 129
    assert complete.proposals[0].edits == ()
    assert {edit["operation"] for p in complete.proposals for edit in p.edits} == {
        "redirect_transition",
        "change_output",
        "add_state_from_template",
    }
    base.scope["custom"]["value"] = 7
    for proposal in complete.proposals:
        assert proposal.ir.scope == original["scope"]
        assert proposal.ir.preconditions == base.preconditions
    complete.proposals[0].ir.scope["custom"]["value"] = 8
    assert complete.proposals[1].ir.scope["custom"]["value"] == 0
    partial = enumerate_structural_edits(base, evaluation_budget=1)
    assert not partial.complete and partial.evaluations == 1
    assert partial.stop_reason == "search_budget_exhausted"


@pytest.mark.parametrize(
    "case", ["redirect_start", "redirect_wait", "retain_ready", "state_insert"]
)
def test_new_mechanism_is_generated_not_supplied_as_bootstrap(case):
    task = revision_task(case=case)
    actual = FiniteStateProgram.model_validate(task.private_spec["after"]["mechanism"])
    costs = {action: 1 for action in actual.actions}
    assert all(
        shortest_witness(actual, FiniteStateProgram.from_ir(h.mechanism_ir), costs)
        is not None
        for h in task.bootstrap
    )
    proposals = enumerate_structural_edits(task.bootstrap[0].mechanism_ir).proposals
    matching = [
        p
        for p in proposals
        if shortest_witness(actual, FiniteStateProgram.from_ir(p.ir), costs) is None
    ]
    assert matching and all(p.edits for p in matching)
    if case == "state_insert":
        assert "world-extra-state" not in str([p.as_json() for p in proposals])
        assert any(len(FiniteStateProgram.from_ir(p.ir).states) == 5 for p in matching)


def test_shadow_budget_checked_for_whole_batch_before_dispatch():
    task = revision_task(case="unchanged")

    def forbidden():
        raise AssertionError("must not dispatch any validation control")

    view = AgentWorldView(
        observe=forbidden,
        available=forbidden,
        step=lambda action: forbidden(),
        begin_trial=forbidden,
    )
    result = validate_shadow(
        view, program_for(task).to_ir(), task.costs, {task.calibration}, budget=1
    )
    assert result["status"] == "unresolved"
    assert result["spent"] == 0 and result["runs"] == []
    assert result["stop_reason"] == "validation_budget_exhausted"
    assert task.calibration not in [
        tuple(q["actions"]) for q in result["plan"]["queries"]
    ]


def test_shadow_owns_candidate_and_costs_before_callback_mutation():
    task = revision_task(case="unchanged", boundary=100)
    ir, costs = program_for(task).to_ir(), dict(task.costs)
    original = ir.model_dump(mode="json")

    def reset():
        ir.parameters["transitions"][0]["output"] = "forged"
        costs.update({action: 99 for action in costs})
        return task.view.begin_trial()

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=task.view.step,
        begin_trial=reset,
    )
    result = validate_shadow(view, ir, costs, (), budget=100)
    assert result["status"] == "passed"
    assert result["plan"]["candidate_ir"] == original
    assert result["spent"] == result["plan"]["total_cost"]


def test_shadow_rejection_does_not_refit_to_validation():
    task = revision_task(case="unchanged", boundary=100)
    wrong = FiniteStateProgram.from_ir(task.bootstrap[3].compile()).to_ir()
    result = validate_shadow(task.view, wrong, task.costs, (), budget=100)
    assert result["status"] == "rejected"
    assert result["plan"]["candidate_ir"] == wrong.model_dump(mode="json")
    assert all(
        len(run["theories"]) == 0 or run["theories"][-1]["status"] == "provisional"
        for run in result["runs"]
    )


@pytest.mark.parametrize("used", [[[]], [("unknown",)], [()], [tuple("abcdef")]])
def test_shadow_rejects_invalid_discovery_exclusions(used):
    task = revision_task()
    with pytest.raises(ValueError):
        validate_shadow(
            task.view, program_for(task).to_ir(), task.costs, used, budget=100
        )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "case", ["redirect_start", "redirect_wait", "retain_ready", "state_insert"]
)
def test_online_structural_change_validates_before_replacing_old_version(seed, case):
    task = revision_task(seed, case)
    bootstrap = copy.deepcopy([h.model_dump(mode="json") for h in task.bootstrap])
    result = run_task(task)
    assert result["status"] == "conditional_joint_theory_revised"
    assert result["bootstrap_independent_validation_completed"]
    assert result["replacement_independent_validation_completed"]
    assert result["mechanism_change_warranted"]
    assert not result["unique_causal_structure_claimed"]
    assert result["spent"] == sum(run["spent"] for run in control_runs(result)) <= 240
    events = result["theory_events"]
    assert [(e["version"], e["status"]) for e in events] == [
        (1, "proposed"),
        (1, "provisional"),
        (1, "accepted"),
        (1, "challenged"),
        (2, "proposed"),
        (2, "provisional"),
        (2, "accepted"),
        (1, "superseded"),
    ]
    assert events[4]["parent_version"] == 1
    assert events[0]["ir"] == events[-1]["ir"]
    assert [h.model_dump(mode="json") for h in task.bootstrap] == bootstrap
    used = set()
    for run in control_runs(result):
        words = [tuple(e["actions"]) for e in run["events"] if e["kind"] == "plan"]
        if words and words[0] in [
            tuple(q["actions"])
            for q in result["replacement_validation"]["plan"]["queries"]
        ]:
            assert words[0] not in used
        used.update(words)
    ids = [
        e["query_id"]
        for run in control_runs(result)
        for e in run["events"]
        if e["kind"] == "plan"
    ]
    assert len(ids) == len(set(ids))
    monitoring_resets = [
        e["observation"]["episode_id"]
        for run in result["monitoring"]
        for e in run["events"]
        if e["kind"] == "reset"
    ]
    assert monitoring_resets[-1] >= task.private_spec["boundary_episode"]


@pytest.mark.parametrize("seed", [0, 1])
def test_hidden_controller_failure_never_warrants_mechanism_change(seed):
    result = run_task(revision_task(seed, "controller_fault"))
    assert not result["mechanism_change_warranted"]
    assert result["status"] in ("unresolved", "conditional_joint_theory_revised")
    if seed == 0:
        assert result["stop_reason"] == "structural_discovery_unresolved"


@pytest.mark.parametrize(
    "case,status",
    [("unchanged", "no_anomaly_detected"), ("outside_grammar", "model_mismatch")],
)
def test_negative_cases_do_not_publish_replacement(case, status):
    result = run_task(revision_task(case=case))
    assert result["status"] == status
    assert not result["replacement_independent_validation_completed"]
    assert all(e["version"] == 1 for e in result["theory_events"])


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"search_budget": 1}, "search_budget_exhausted"),
        ({"candidate_limit": 1}, "planning_capacity_exceeded"),
        ({"budget": 15}, "bootstrap_validation_failed"),
    ],
)
def test_incomplete_search_capacity_or_validation_never_promotes(kwargs, reason):
    result = run_task(revision_task(), **kwargs)
    assert result["stop_reason"] == reason
    assert result["status"] == "unresolved"
    assert not result["replacement_independent_validation_completed"]


def test_full_online_run_is_deterministic():
    assert run_task(revision_task()) == run_task(revision_task())


def test_monitoring_budget_exhaustion_is_not_a_no_change_result():
    full = run_task(revision_task(case="unchanged"))
    budget = (
        full["bootstrap"]["spent"]
        + full["bootstrap_validation"]["spent"]
        + full["monitoring"][0]["spent"]
    )
    limited = run_task(revision_task(case="unchanged"), budget=budget)
    assert limited["status"] == "unresolved"
    assert limited["stop_reason"] == "monitoring_budget_exhausted"
    assert limited["spent"] == budget
