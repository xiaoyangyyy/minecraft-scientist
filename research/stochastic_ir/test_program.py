"""Independent path-enumeration oracles and real component-world tests."""

import json
import pickle
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from minecraft_scientist.protocols import TheoryRepresentation
from minecraft_scientist.theory.ir import TheoryInterpreter

from . import program as runtime
from .audit import audit_trace
from .composition import compile_joint, lift_deterministic
from .fixture import fixture
from .program import Mass, StochasticProgram


def independent_paths(program, sequence):
    # Enumerate every latent path, without production tables/filter/recurrence.
    paths = [(v.state, (), v.probability.fraction()) for v in program.initial]
    for action in sequence:
        paths = [
            (row.target, word + (row.output,), mass * row.probability.fraction())
            for state, word, mass in paths
            for row in program.transitions
            if row.source == state and row.action == action
        ]
    distribution = defaultdict(Fraction)
    states = defaultdict(lambda: defaultdict(Fraction))
    for state, word, mass in paths:
        distribution[word] += mass
        states[word][state] += mass
    return dict(distribution), states


def independent_component_paths(spec, sequence):
    # Independent factor-path enumeration, never calls the joint compiler. This
    # catches a compiler which gets probabilities wrong but matches its filter.
    def weight(row):
        p = row["probability"]
        return Fraction(p["numerator"], p["denominator"])

    m, c, sensor = (spec[k] for k in ("mechanism", "controller", "sensor"))
    paths = [(m["initial_state"], v["state"], (), weight(v)) for v in c["initial"]]
    for action in sequence:
        new = []
        for ms, cs, word, mass in paths:
            for delivery in c["transitions"]:
                if delivery["source"] != cs or delivery["action"] != action:
                    continue
                target, latent = ms, spec["idle_output"]
                if delivery["output"] != "_no_delivery_":
                    row = next(
                        r
                        for r in m["transitions"]
                        if r["source"] == ms and r["action"] == delivery["output"]
                    )
                    target, latent = row["target"], row["output"]
                for emission in sensor["transitions"]:
                    if emission["action"] == latent:
                        new.append(
                            (
                                target,
                                delivery["target"],
                                word + (emission["output"],),
                                mass * weight(delivery) * weight(emission),
                            )
                        )
        paths = new
    result = defaultdict(Fraction)
    for _, _, word, mass in paths:
        result[word] += mass
    return dict(result)


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_exact_distribution_and_nonreset_filter_match_independent_paths(seed, noise):
    world, ir, actions, _ = fixture(seed, noise=noise)
    program = StochasticProgram.from_ir(ir)
    for sequence in ((), actions, actions + (actions[1], actions[2], actions[3])):
        expected, states = independent_paths(program, sequence)
        actual = program.sequence_distribution(sequence)
        assert actual == expected and sum(actual.values()) == 1
        assert actual == independent_component_paths(world.evaluator_spec(), sequence)
        for word, likelihood in expected.items():
            found, belief = program.filter_history(
                tuple(zip(sequence, word, strict=True))
            )
            assert found == likelihood
            assert belief == {s: p / likelihood for s, p in states[word].items()}


def test_two_part_filter_equals_one_continuous_history_not_repeated_resets():
    _, ir, actions, outputs = fixture(noise=False)
    p = StochasticProgram.from_ir(ir)
    prefix = ((actions[0], outputs[0]), (actions[1], outputs[0]))
    _, belief = p.filter_history(prefix)
    continued = p.sequence_distribution((actions[2], actions[3]), belief=belief)
    fresh = p.sequence_distribution((actions[2], actions[3]))
    assert continued != fresh
    first, belief = p.filter_history(prefix)
    second, final = p.filter_history(
        ((actions[2], outputs[0]), (actions[3], outputs[1])), belief=belief
    )
    whole, whole_final = p.filter_history(
        prefix + ((actions[2], outputs[0]), (actions[3], outputs[1]))
    )
    assert first * second == whole and final == whole_final


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_actual_world_complete_frozen_dispatch_and_raw_independent_replay(seed):
    world, ir, actions, _ = fixture(seed)
    view, model = world.agent_view(), StochasticProgram.from_ir(ir)
    dispatch = tuple(
        entry
        for _ in range(3)
        for entry in (("reset",), *(("step", a) for a in actions * 4))
    )
    belief = None
    for entry in dispatch:
        if entry == ("reset",):
            view.begin_trial()
            belief = model.initial_belief()
        else:
            prediction = model.predict_step(entry[1], belief=belief)
            action = next(
                a for a in view.available_actions() if a.target_id == entry[1]
            )
            record = view.step(action)
            measured = record.event.details["signal"]
            assert measured in prediction
            likelihood, belief = model.filter(entry[1], measured, belief=belief)
            assert likelihood > 0
            assert record.context["treatment_verification"] == "unknown"
            assert record.intervention_compliance is True
    audit = audit_trace(
        world.evaluator_spec(), world.evaluator_trace(), expected_dispatch=dispatch
    )
    assert (
        audit["valid"] is True
        and audit["resets"] == 3
        and audit["atomic_actions"] == 48
    )
    assert not hasattr(view, "evaluator_spec") and not hasattr(view, "_ms")
    with pytest.raises(TypeError):
        pickle.dumps(view)


@pytest.mark.parametrize(
    "corruption",
    ["drop", "extra", "clock", "signal", "compliance_integer", "treatment", "action"],
)
def test_complete_audit_rejects_trace_forgery(corruption):
    world, _, actions, outputs = fixture()
    view = world.agent_view()
    view.begin_trial()
    view.step(next(a for a in view.available_actions() if a.target_id == actions[0]))
    trace = world.evaluator_trace()
    dispatch = (("reset",), ("step", actions[0]))
    if corruption == "drop":
        trace.pop()
    elif corruption == "extra":
        trace.append(deepcopy(trace[-1]))
    elif corruption == "clock":
        trace[-1]["transition"]["state_after"]["step"] += 1
    elif corruption == "signal":
        old = trace[-1]["transition"]["event"]["details"]["signal"]
        trace[-1]["transition"]["event"]["details"]["signal"] = next(
            o for o in outputs if o != old
        )
    elif corruption == "compliance_integer":
        trace[-1]["transition"]["intervention_compliance"] = 1
    elif corruption == "treatment":
        trace[-1]["transition"]["context"]["treatment_verification"] = "verified"
    else:
        trace[-1]["transition"]["executed_action"]["target_id"] = actions[1]
    with pytest.raises(ValueError):
        audit_trace(world.evaluator_spec(), trace, expected_dispatch=dispatch)


def test_seeded_replay_is_identical_and_resets_do_not_rewind_randomness():
    worlds = [fixture(21)[0] for _ in range(2)]
    actions = fixture(21)[2]
    for world in worlds:
        view = world.agent_view()
        for _ in range(4):
            view.begin_trial()
            for a in actions * 3:
                view.step(next(v for v in view.available_actions() if v.target_id == a))
    assert worlds[0].evaluator_trace() == worlds[1].evaluator_trace()
    trials = [worlds[0].evaluator_trace()[i * 13 + 1 : (i + 1) * 13] for i in range(4)]
    signals = [
        tuple(r["transition"]["event"]["details"]["signal"] for r in trial)
        for trial in trials
    ]
    assert len(set(signals)) > 1


@pytest.mark.parametrize(
    "numerator,denominator",
    [(0, 1), (2, 1), (2, 4), (True, 1), (1, True), (1.0, 2), (1, 1_000_001)],
)
def test_mass_schema_rejects_noncanonical_or_coerced_values(numerator, denominator):
    with pytest.raises(ValidationError):
        Mass(numerator=numerator, denominator=denominator)


@pytest.mark.parametrize(
    "fault",
    [
        "initial_mass",
        "duplicate_initial",
        "unknown_target",
        "duplicate_branch",
        "missing_pair",
        "nonunit_mass",
        "unknown_output",
    ],
)
def test_total_program_schema_rejects_malformed_models(fault):
    _, ir, _, _ = fixture()
    data = deepcopy(ir.parameters)
    if fault == "initial_mass":
        data["initial"].pop()
    elif fault == "duplicate_initial":
        data["initial"].append(deepcopy(data["initial"][0]))
    elif fault == "unknown_target":
        data["transitions"][0]["target"] = "not-a-state"
    elif fault == "unknown_output":
        data["transitions"][0]["output"] = "not-an-output"
    elif fault == "duplicate_branch":
        data["transitions"].append(deepcopy(data["transitions"][0]))
    elif fault == "missing_pair":
        pair = data["transitions"][0]["source"], data["transitions"][0]["action"]
        data["transitions"] = [
            r for r in data["transitions"] if (r["source"], r["action"]) != pair
        ]
    else:
        data["transitions"][0]["probability"] = {"numerator": 1, "denominator": 127}
    with pytest.raises(ValidationError):
        StochasticProgram.model_validate(data)


@pytest.mark.parametrize(
    "belief",
    [
        {},
        {"unknown": Fraction(1)},
        {"joint-0-0": 1.0},
        {"joint-0-0": Fraction(1, 2)},
        {"joint-0-0": Fraction(-1)},
    ],
)
def test_invalid_belief_rejected(belief):
    _, ir, actions, _ = fixture()
    with pytest.raises(ValueError):
        StochasticProgram.from_ir(ir).predict_step(actions[0], belief=belief)


def test_zero_probability_observation_is_mismatch_not_automatic_reset():
    _, ir, actions, outputs = fixture(noise=False)
    p = StochasticProgram.from_ir(ir)
    with pytest.raises(ValueError, match="model mismatch"):
        p.filter(actions[2], outputs[1])


@pytest.mark.parametrize("guard", ["MAX_BRANCH_OPERATIONS", "MAX_OUTPUT_PREFIXES"])
def test_probability_search_exhaustion_never_truncates_distribution(monkeypatch, guard):
    _, ir, actions, _ = fixture()
    monkeypatch.setattr(runtime, guard, 1)
    with pytest.raises(ValueError, match="bound exhausted"):
        StochasticProgram.from_ir(ir).sequence_distribution(actions)


def test_shared_ir_roundtrip_and_no_silent_default_interpreter_integration():
    _, ir, actions, _ = fixture()
    assert ir.representation is TheoryRepresentation.STATE_MACHINE
    parsed = StochasticProgram.from_ir(ir)
    assert StochasticProgram.from_ir(parsed.to_ir()).model_dump() == parsed.model_dump()
    with pytest.raises(ValueError):
        TheoryInterpreter.predict(ir, sequence=actions)
    with pytest.raises(ValueError):
        StochasticProgram.from_ir(
            ir.model_copy(update={"scope": {**ir.scope, "certified_reset_only": True}})
        )
    with pytest.raises(ValueError):
        StochasticProgram.from_ir(
            ir.model_copy(update={"preconditions": ("unimplemented-condition",)})
        )


def test_lift_deterministic_predictions_match_existing_ir():
    world, _, actions, _ = fixture()
    from minecraft_scientist.theory.finite_state import FiniteStateProgram

    p = FiniteStateProgram.model_validate(world.evaluator_spec()["mechanism"])
    lifted = StochasticProgram.from_ir(lift_deterministic(p.to_ir()))
    for word in ((), actions, actions * 3):
        assert lifted.sequence_distribution(word) == {p.execute(word)[0]: Fraction(1)}


def test_public_serialization_has_no_actual_state_delivery_or_seed():
    world, _, _, _ = fixture()
    view = world.agent_view()
    view.begin_trial()
    receipts = [view.step(a).model_dump(mode="json") for a in view.available_actions()]
    serialized = json.dumps(receipts)
    assert all(
        key not in serialized
        for key in (
            '"seed"',
            '"delivered_action"',
            '"controller_state"',
            '"mechanism_state"',
        )
    )
    copied = world.evaluator_spec()
    copied["seed"] = 999
    assert world.evaluator_spec()["seed"] == 0


def test_completely_uninformative_sensor_hides_distinct_mechanisms():
    from minecraft_scientist.theory.finite_state import FiniteStateProgram
    from minecraft_scientist.world.identification import _program

    world, _, actions, outputs = fixture()
    spec = world.evaluator_spec()
    original = FiniteStateProgram.model_validate(spec["mechanism"])
    changed = _program(actions, outputs, activation=False)
    controller = StochasticProgram.model_validate(spec["controller"])
    sensor = deepcopy(spec["sensor"])
    for row in sensor["transitions"]:
        row["probability"] = {"numerator": 1, "denominator": 2}
    sensor_ir = StochasticProgram.model_validate(sensor).to_ir()
    models = [
        StochasticProgram.from_ir(
            compile_joint(
                p.to_ir(), controller.to_ir(), sensor_ir, idle_output=outputs[0]
            )
        )
        for p in (original, changed)
    ]
    assert (
        original.execute((actions[0], actions[2], actions[3]))[0]
        != changed.execute((actions[0], actions[2], actions[3]))[0]
    )
    for word in (
        (),
        actions,
        (actions[0], actions[2], actions[3]),
        actions + (actions[1],),
    ):
        left, right = [p.sequence_distribution(word) for p in models]
        assert left == right
        assert all(
            probability == Fraction(1, 2 ** len(word)) for probability in left.values()
        )


def test_measurement_noise_is_not_a_deterministic_contradiction():
    _, ir, actions, outputs = fixture(noise=True)
    probability, belief = StochasticProgram.from_ir(ir).filter(actions[0], outputs[1])
    assert probability == Fraction(1, 8)
    assert sum(belief.values()) == 1


def test_parsed_model_is_detached_from_mutable_ir_parameters():
    _, ir, actions, _ = fixture()
    model = StochasticProgram.from_ir(ir)
    predicted = model.sequence_distribution(actions)
    ir.parameters["transitions"][0]["target"] = "tampered-input"
    assert model.sequence_distribution(actions) == predicted
    with pytest.raises(TypeError):
        model.table[model.states[0], model.actions[0]] = ()
