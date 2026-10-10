"""Exact independent likelihood products and finite path/statistical identities."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from research.stochastic_ir.test_program import independent_component_paths

from . import belief
from .belief import ModelBank, ModelMismatch, WorkBound
from .fixture import make_world, public_family
from .planning import choose, value


def test_model_and_state_posterior_matches_independent_factor_path_likelihoods():
    candidates, actions, outputs = public_family()
    bank = ModelBank(candidates)
    word, observed = actions, (outputs[0], outputs[0], outputs[0], outputs[1])
    likelihoods = []
    for candidate in candidates:
        spec = {
            "mechanism": candidate.mechanism_ir.parameters,
            "controller": candidate.controller_ir.parameters,
            "sensor": candidate.sensor_ir.parameters,
            "idle_output": candidate.idle_output,
        }
        likelihoods.append(
            independent_component_paths(spec, word).get(observed, Fraction(0))
        )
    normalizer = sum(likelihoods)
    for a, o in zip(word, observed, strict=True):
        bank.advance(a, o)
    assert bank._weights == tuple(p / normalizer for p in likelihoods)
    assert sum(bank.project().values()) == 1
    before = bank._weights
    bank.reset_state()
    assert bank._weights == before
    assert bank._beliefs == tuple(p.initial_belief() for p in bank.programs)


def test_explicit_reset_not_silent_reset_between_steps():
    candidates, actions, outputs = public_family()
    bank = ModelBank(candidates)
    bank.advance(actions[0], outputs[0])
    bank.advance(actions[1], outputs[0])
    continuing = bank.forecast((actions[2], actions[3]), reset=False)[2]
    reset = bank.forecast((actions[2], actions[3]), reset=True)[2]
    assert continuing != reset


def test_single_measurement_noise_does_not_eliminate_all_models():
    candidates, actions, outputs = public_family()
    bank = ModelBank(candidates)
    assert bank.advance(actions[0], outputs[1]) == Fraction(1, 16)
    assert sum(bank._weights) == 1
    assert bank.project(factor="sensor")[candidates[1].sensor_id] == 1


def test_identical_behavior_does_not_force_factor_attribution():
    candidates, _, _ = public_family()
    original = candidates[0]
    renamed_factor = replace(original, key="second", mechanism_id="other-mechanism")
    bank = ModelBank((original, renamed_factor))
    prior = bank.project()
    for action in bank.actions:
        # Use an actually possible public output; the final retrieve in a
        # deterministic ready machine does not emit the idle output.
        bank.advance(action, next(iter(bank.predict_step(action)[1])))
    assert bank.project() == prior
    assert len(bank.project()) == 2
    reduction, information, distinguishing = value(
        bank, bank.forecast((bank.actions[0],), reset=True)
    )
    assert reduction == 0 and information == "0" and not distinguishing


def test_zero_likelihood_mismatch_is_transactional():
    candidates, actions, outputs = public_family()
    bank = ModelBank((candidates[0],))
    before = bank.snapshot()
    with pytest.raises(ModelMismatch):
        bank.advance(actions[0], outputs[1])
    assert bank.snapshot() == before


def test_exact_work_bound_never_partially_updates_evidence(monkeypatch):
    candidates, actions, outputs = public_family()
    bank = ModelBank(candidates)
    before = bank.snapshot()
    monkeypatch.setattr(belief, "MAX_RATIONAL_BITS", 1)
    with pytest.raises(WorkBound):
        bank.advance(actions[0], outputs[1])
    assert bank.snapshot() == before


@pytest.mark.parametrize("bad", [True, 1.0, "1", Fraction(0), Fraction(-1)])
def test_prior_types_and_domain_reject(bad):
    candidate = public_family()[0][0]
    with pytest.raises(ValueError):
        ModelBank((replace(candidate, prior=bad),))


def test_redefined_factor_or_duplicate_candidate_is_rejected():
    candidates, _, _ = public_family()
    with pytest.raises(ValueError, match="redefine"):
        ModelBank(
            (
                candidates[0],
                replace(candidates[6], mechanism_id=candidates[0].mechanism_id),
            )
        )
    with pytest.raises(ValueError):
        ModelBank((candidates[0], candidates[0]))


def test_caller_ir_and_forecast_mutation_do_not_change_internal_models():
    candidates, actions, _ = public_family()
    bank = ModelBank(candidates)
    expected = bank.forecast(actions, reset=True)[2]
    candidates[0].mechanism_ir.parameters["transitions"][0]["target"] = "corrupt"
    forecast = bank.forecast(actions, reset=True)
    forecast[1][0].clear()
    definitions = bank.definitions()
    definitions[0]["mechanism_id"] = "corrupt"
    assert bank.forecast(actions, reset=True)[2] == expected
    assert bank.definitions()[0]["mechanism_id"] != "corrupt"


def test_joint_information_and_factor_risk_are_not_the_same_objective():
    candidates, actions, _ = public_family()
    bank = ModelBank(candidates)
    calibration = value(bank, bank.forecast((actions[0],), reset=True))
    assert calibration[0] == 0
    assert float(calibration[1]) > 0
    mechanism_probe = value(
        bank, bank.forecast((actions[0], actions[2], actions[3]), reset=True)
    )
    assert mechanism_probe[0] > 0


def test_complete_word_space_bound_fails_not_shortens_horizon(monkeypatch):
    from . import planning

    bank = ModelBank(public_family()[0])
    monkeypatch.setattr(planning, "MAX_WORDS", 1)
    with pytest.raises(WorkBound, match="never shorten"):
        choose(
            bank,
            costs={a: 1 for a in bank.actions},
            remaining=20,
            horizon=3,
            policy="mechanism_risk",
        )


@pytest.mark.parametrize("policy", ["joint_information_gain", "mechanism_risk"])
def test_plan_cost_and_gain_within_prepaid_budget(policy):
    bank = ModelBank(public_family()[0])
    costs = {a: i + 1 for i, a in enumerate(bank.actions)}
    selected = choose(bank, costs=costs, remaining=8, horizon=3, policy=policy)
    assert selected.cost == sum(costs[a] for a in selected.word) + int(selected.reset)
    assert selected.cost <= 8
    assert selected.mechanism_risk_reduction >= 0
    assert selected.distinguishing_in_declared_word_space


def test_input_family_identical_across_all_private_conditions():
    families = [
        ModelBank(make_world(13, condition)[1]).definitions()
        for condition in (
            "no_change",
            "control_only",
            "mechanism_only",
            "both",
            "unmodeled_sensor",
        )
    ]
    assert all(f == families[0] for f in families)


def test_copied_candidate_state_alphabet_renaming_preserves_predictions():
    candidates, actions, _ = public_family()
    copied = []
    for candidate in candidates:
        ir = deepcopy(candidate.mechanism_ir)
        names = {s: f"renamed-{i}" for i, s in enumerate(ir.parameters["states"])}
        ir.parameters["states"] = [names[s] for s in ir.parameters["states"]]
        ir.parameters["initial_state"] = names[ir.parameters["initial_state"]]
        for row in ir.parameters["transitions"]:
            row["source"], row["target"] = names[row["source"]], names[row["target"]]
        copied.append(replace(candidate, mechanism_ir=ir))
    left, right = ModelBank(candidates), ModelBank(tuple(copied))
    assert (
        left.forecast(actions, reset=True)[2] == right.forecast(actions, reset=True)[2]
    )
