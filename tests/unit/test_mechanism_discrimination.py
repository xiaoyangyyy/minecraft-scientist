"""Independent small examples and exhaustive reference checks for H1 machinery."""

import copy
import itertools
import pickle
import random

import pytest
from pydantic import ValidationError

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.discovery.discrimination import DiscriminationSpace
from minecraft_scientist.protocols import AgentWorldView, TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram, shortest_witness
from minecraft_scientist.theory.ir import TheoryInterpreter
from minecraft_scientist.world.identification import identification_task


def space(seed=0, family="activation", **options):
    task = identification_task(seed, family)
    return task, DiscriminationSpace(task.candidates, task.costs, horizon=5, **options)


@pytest.mark.parametrize("family", ["activation", "delay", "toggle", "equivalent"])
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_observational_twins_match_complete_calibration_trace(family, seed):
    task, versions = space(seed, family)
    predictions = [
        program.execute(task.calibration)[0] for program in versions.programs
    ]
    assert predictions[0] == predictions[1]
    initial = task.view.begin_trial()
    assert "state" not in initial.agent_state and "family" not in initial.agent_state
    controls = {action.target_id: action for action in task.view.available_actions()}
    outputs = tuple(
        task.view.step(controls[token]).event.details["signal"]
        for token in task.calibration
    )
    assert outputs == predictions[0]


@pytest.mark.parametrize(
    "family,min_actions", [("activation", 3), ("delay", 4), ("toggle", 5)]
)
def test_product_witness_matches_exhaustive_small_reference(family, min_actions):
    task, versions = space(family=family)
    left, right = versions.programs
    witness = shortest_witness(left, right, task.costs)
    assert len(witness.actions) == witness.action_cost == min_actions
    assert witness.left_outputs != witness.right_outputs
    for length in range(min_actions):
        for word in itertools.product(versions.actions, repeat=length):
            assert left.execute(word)[0] == right.execute(word)[0]
    assert (
        shortest_witness(left, right, task.costs, maximum_length=min_actions - 1)
        is None
    )
    assert (
        shortest_witness(left, right, task.costs, maximum_length=min_actions) == witness
    )


def test_weighted_search_does_not_return_first_expensive_terminal_edge():
    def program(changed):
        rows = []
        for state, action in itertools.product(("s0", "s1"), ("a", "b")):
            output = "one" if changed and (action == "b" or state == "s1") else "zero"
            rows.append(
                {"source": state, "action": action, "target": "s1", "output": output}
            )
        return FiniteStateProgram(
            states=("s0", "s1"),
            actions=("a", "b"),
            outputs=("zero", "one"),
            initial_state="s0",
            transitions=rows,
        )

    left, right = program(False), program(True)
    assert shortest_witness(left, right, {"a": 1, "b": 10}).actions == ("a", "a")
    assert shortest_witness(
        left, right, {"a": 1, "b": 10}, maximum_length=1
    ).actions == ("b",)


@pytest.mark.parametrize(
    "case",
    ["missing", "duplicate", "target", "output", "initial", "private", "language"],
)
def test_program_rejects_malformed_or_private_payload_before_execution(case):
    _task, versions = space()
    data = versions.programs[0].model_dump(mode="json")
    if case == "missing":
        data["transitions"].pop()
    elif case == "duplicate":
        data["transitions"].append(data["transitions"][0])
    elif case in ("target", "output"):
        data["transitions"][0][case] = "outside"
    elif case == "initial":
        data["initial_state"] = "outside"
    elif case == "private":
        data["ground_truth"] = "secret"
    else:
        data["language"] = "python-eval"
    with pytest.raises(ValidationError):
        FiniteStateProgram.model_validate(data)


@pytest.mark.parametrize("cost", [0, -1, True, 1.0, "1", 101])
def test_action_costs_are_exact_positive_and_bounded(cost):
    task = identification_task(0, "activation")
    costs = dict(task.costs)
    costs[next(iter(costs))] = cost
    with pytest.raises(ValueError):
        DiscriminationSpace(task.candidates, costs)


def test_shared_interpreter_executes_ir_and_preserves_context_gate():
    task, versions = space()
    ir = task.candidates[0]
    result = TheoryInterpreter.predict(ir, sequence=task.calibration)
    assert result["outputs"] == versions.programs[0].execute(task.calibration)[0]
    scoped = TheoryIR.model_validate(
        ir.model_dump() | {"scope": {"required_context": {"mode": "x"}}}
    )
    with pytest.raises(ValueError, match="execution_context"):
        TheoryInterpreter.predict(scoped, sequence=task.calibration)
    assert (
        TheoryInterpreter.predict(
            scoped, sequence=task.calibration, execution_context={"mode": "x"}
        )
        == result
    )
    with pytest.raises(TypeError):
        versions.programs[0].table["x"] = "bad"


def test_equivalent_syntax_does_not_manufacture_entropy_or_unique_structure():
    task, versions = space(family="equivalent")
    assert len(versions.classes) == 1 and len(versions.remaining) == 2
    assert shortest_witness(*versions.programs, task.costs) is None
    assert versions.choose("sequence_ig", 20, random.Random(0)) is None
    result = identify_mechanism(
        task.view, task.candidates, task.calibration, task.costs
    )
    assert result["remaining_candidates"] == [0, 1]
    assert not result["unique_causal_structure_claimed"]
    assert result["theories"][-1]["status"] == "provisional"


@pytest.mark.parametrize("family", ["activation", "delay", "toggle"])
@pytest.mark.parametrize(
    "method", ["witness_cover", "sequence_ig", "optimal_reference"]
)
def test_actual_public_execution_identifies_known_class_with_charged_reset(
    family, method
):
    task = identification_task(1, family)
    result = identify_mechanism(
        task.view, task.candidates, task.calibration, task.costs, method=method
    )
    assert result["status"] == "conditional_behavior_identified"
    assert result["remaining_candidates"] == [1]
    assert result["spent"] == sum(event.get("cost", 0) for event in result["events"])
    assert result["spent"] <= result["budget"]
    assert result["theories"][-1]["status"] == "provisional"
    assert not result["independent_validation_completed"]
    assert result["events"][0]["kind"] == "plan"
    assert all(
        event["transition"]["intended_intervention"] is not None
        for event in result["events"]
        if event["kind"] == "step"
    )


def test_one_step_is_an_ablation_not_a_matched_multi_step_victory():
    task = identification_task(0, "activation")
    result = identify_mechanism(
        task.view, task.candidates, task.calibration, task.costs, method="one_step_ig"
    )
    assert result["status"] == "unresolved" and result["spent"] == 6


def test_budget_exhaustion_before_calibration_never_dispatches():
    task = identification_task(0, "activation")

    def forbidden(*args):
        pytest.fail("budget-limited calibration must not dispatch")

    view = AgentWorldView(
        observe=forbidden, available=forbidden, step=forbidden, begin_trial=forbidden
    )
    result = identify_mechanism(
        view, task.candidates, task.calibration, task.costs, budget=5
    )
    assert result["spent"] == 0 and result["events"] == []
    assert result["stop_reason"] == "budget_exhausted"


def test_control_failure_is_charged_preserved_and_not_world_evidence():
    task = identification_task(0, "activation")

    def fail(action):
        return task.view.step(action).model_copy(
            update={"intervention_compliance": False, "controller_failure": "injected"}
        )

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=fail,
        begin_trial=task.view.begin_trial,
    )
    result = identify_mechanism(view, task.candidates, task.calibration, task.costs)
    assert result["remaining_candidates"] == [0, 1]
    assert result["spent"] == 2 and result["completed_queries"] == 0
    assert result["stop_reason"] == "control_failure" and result["theories"] == []
    assert all(event["kind"] != "evidence" for event in result["events"])


def test_out_of_candidate_family_reports_mismatch_not_favorable_selection():
    task = identification_task(0, "model_mismatch")
    result = identify_mechanism(
        task.view, task.candidates, task.calibration, task.costs
    )
    assert result["status"] == "model_mismatch" and result["remaining_candidates"] == []
    assert result["theories"] == []


def test_replay_is_exact_and_public_view_contains_no_truth_payload():
    def run():
        task = identification_task(7, "toggle")
        with pytest.raises(TypeError):
            pickle.dumps(task.view)
        assert not hasattr(task.view, "ground_truth")
        before = copy.deepcopy([ir.model_dump() for ir in task.candidates])
        result = identify_mechanism(
            task.view, task.candidates, task.calibration, task.costs
        )
        assert before == [ir.model_dump() for ir in task.candidates]
        return result

    assert run() == run()
