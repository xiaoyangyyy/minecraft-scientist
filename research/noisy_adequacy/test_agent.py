"""Real online monitoring, unchanged planner/evidence and independent audits."""

from copy import deepcopy
from fractions import Fraction

import pytest

from minecraft_scientist.protocols import AgentWorldView
from research.noisy_competition.agent import run_agent
from research.noisy_competition.audit import verify_public
from research.noisy_competition.belief import ModelBank
from research.noisy_competition.fixture import make_world
from research.stochastic_ir.audit import audit_trace
from research.stochastic_ir.program import StochasticProgram

from .agent import run_monitored_agent
from .audit import verify_monitor
from .monitor import FamilyAdequacyMonitor


@pytest.mark.parametrize(
    "policy", ["joint_information_gain", "mechanism_risk", "uniform_word"]
)
def test_same_actual_trajectory_with_and_without_online_shadow_monitor(policy):
    world, candidates, costs, _ = make_world(4, "unmodeled_sensor")
    config = {"costs": costs, "budget": 16, "horizon": 2, "policy": policy, "seed": 4}
    monitored = run_monitored_agent(world.agent_view(), candidates, **config)
    plain_world, _, _, _ = make_world(4, "unmodeled_sensor")
    plain = run_agent(plain_world.agent_view(), candidates, **config)
    assert monitored["agent"] == plain
    assert world.evaluator_trace() == plain_world.evaluator_trace()
    public = verify_public(candidates, config, monitored["agent"])
    raw = audit_trace(
        world.evaluator_spec(),
        world.evaluator_trace(),
        expected_dispatch=public["dispatch"],
    )
    audit = verify_monitor(ModelBank(candidates).definitions(), config, monitored)
    assert audit["actual_cost"] == public["actual_cost"]
    assert audit["atomic_actions"] == raw["atomic_actions"]


def test_predictions_are_frozen_before_actual_dispatch():
    world, candidates, costs, _ = make_world(0, "control_only")
    view, shadow = world.agent_view(), []
    called = []

    def step(action):
        assert (
            shadow[-1]["prediction_before_dispatch"]["issued_command"]
            == action.target_id
        )
        assert "transition" not in shadow[-1]
        called.append(True)
        return view.step(action)

    spy = AgentWorldView(
        observe=view.observe,
        available=view.available_actions,
        step=step,
        begin_trial=view.begin_trial,
    )
    result = run_monitored_agent(
        spy,
        candidates,
        costs=costs,
        budget=8,
        horizon=2,
        policy="uniform_word",
        monitor_ledger=shadow,
    )
    assert called and len(called) == result["monitor"]["steps"]


def fixture():
    world, candidates, costs, _ = make_world(0, "control_only")
    config = {
        "costs": costs,
        "budget": 10,
        "horizon": 2,
        "policy": "uniform_word",
        "seed": 0,
    }
    result = run_monitored_agent(world.agent_view(), candidates, **config)
    return ModelBank(candidates).definitions(), config, result


def test_independent_monitor_audit_never_calls_update_or_prediction(monkeypatch):
    definitions, config, result = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("not an independent raw likelihood check")

    for name in ("prepare", "finish", "reset_state"):
        monkeypatch.setattr(FamilyAdequacyMonitor, name, forbidden)
    for name in ("predict_step", "filter", "filter_history", "sequence_distribution"):
        monkeypatch.setattr(StochasticProgram, name, forbidden)
    assert verify_monitor(definitions, config, result)["valid"]


@pytest.mark.parametrize(
    "kind",
    [
        "likelihood",
        "e",
        "alpha",
        "prediction",
        "reset",
        "first_cost",
        "omit",
        "bool",
        "posterior_claim",
    ],
)
def test_monitor_forgery_rejected(kind):
    definitions, config, result = fixture()
    result = deepcopy(result)
    step = next(e for e in result["monitor_evidence"] if e["kind"] == "step")
    if kind == "likelihood":
        step["after"]["model_likelihoods"][0]["numerator"] += 1
    elif kind == "e":
        result["monitor"]["e_value"] = {"kind": "infinite"}
    elif kind == "alpha":
        result["monitor"]["alpha"] = {"numerator": 1, "denominator": 2}
    elif kind == "prediction":
        step["prediction_before_dispatch"]["component_predictions"][0][
            "output_distribution"
        ] = {}
    elif kind == "reset":
        result["monitor_evidence"][0]["after"]["steps"] = 10
    elif kind == "first_cost":
        result["first_rejection_actual_cost"] = 1
    elif kind == "omit":
        result["monitor_evidence"].pop()
    elif kind == "bool":
        step["after"]["model_likelihoods"][0]["numerator"] = True
    else:
        result["accepted_theory"] = "true"
    with pytest.raises(ValueError):
        verify_monitor(definitions, config, result)


def test_changed_caller_alpha_cannot_reuse_certificate():
    definitions, config, result = fixture()
    with pytest.raises(ValueError):
        verify_monitor(definitions, config, result, alpha=Fraction(1, 4))
