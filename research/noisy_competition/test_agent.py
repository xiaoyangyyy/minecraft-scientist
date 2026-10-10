"""Actual capability queries, public replay, forecasts and paid reset invariants."""

import random
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from research.stochastic_ir.audit import audit_trace

from .agent import run_agent
from .audit import verify_public
from .belief import ModelBank
from .fixture import make_world, public_family
from .planning import choose, value


@pytest.mark.parametrize(
    "policy", ["joint_information_gain", "mechanism_risk", "uniform_word"]
)
def test_real_capability_loop_public_and_raw_audits(policy):
    world, candidates, costs, _ = make_world(0, "control_only")
    config = {"costs": costs, "budget": 16, "horizon": 2, "policy": policy, "seed": 0}
    result = run_agent(world.agent_view(), candidates, **config)
    public = verify_public(candidates, config, result)
    raw = audit_trace(
        world.evaluator_spec(),
        world.evaluator_trace(),
        expected_dispatch=public["dispatch"],
    )
    assert public["actual_cost"] == result["spent"] <= 16
    assert raw["atomic_actions"] == public["atomic_actions"]
    assert result["accepted_theory"] is None
    assert all(
        e["prediction_before_dispatch"]
        for e in result["evidence"]
        if e["kind"] == "step"
    )
    assert all(
        e["transition"]["context"]["treatment_verification"] == "unknown"
        for e in result["evidence"]
        if e["kind"] == "step"
    )


@pytest.mark.parametrize(
    "policy", ["joint_information_gain", "mechanism_risk", "uniform_word"]
)
def test_same_seed_replays_events_and_posterior(policy):
    results = []
    for _ in range(2):
        world, candidates, costs, _ = make_world(3, "mechanism_only")
        results.append(
            run_agent(
                world.agent_view(),
                candidates,
                costs=costs,
                budget=10,
                horizon=2,
                policy=policy,
                seed=3,
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "corruption", ["prediction", "posterior", "cost", "receipt", "hidden_config"]
)
def test_public_replay_rejects_forged_policy_outputs(corruption):
    world, candidates, costs, _ = make_world(0, "control_only")
    config = {
        "costs": costs,
        "budget": 10,
        "horizon": 2,
        "policy": "uniform_word",
        "seed": 0,
    }
    result = run_agent(world.agent_view(), candidates, **config)
    if corruption == "prediction":
        entry = next(e for e in result["evidence"] if e["kind"] == "step")
        entry["prediction_before_dispatch"] = {}
    elif corruption == "posterior":
        result["posterior"]["weights"][0]["numerator"] += 1
    elif corruption == "cost":
        result["spent"] -= 1
    elif corruption == "receipt":
        next(e for e in result["evidence"] if e["kind"] == "step")["transition"][
            "intervention_compliance"
        ] = 1
    else:
        config["costs"] = {k: True for k in costs}
    with pytest.raises(ValueError):
        verify_public(candidates, config, result)


def test_query_gain_analytic_one_bit_and_half_risk_reduction():
    candidates, actions, _ = public_family()
    bank = ModelBank((candidates[0], candidates[12]))
    reduction, information, distinguishing = value(
        bank, bank.forecast((actions[0], actions[2], actions[3]), reset=True)
    )
    assert reduction == Fraction(1, 2)
    assert Decimal(information) == 1
    assert distinguishing


def test_uniform_query_baseline_includes_all_words_and_modes_not_just_informative():
    bank = ModelBank(public_family()[0])
    plans = [
        choose(
            bank,
            costs={a: 1 for a in bank.actions},
            remaining=2,
            horizon=1,
            policy="uniform_word",
            rng=random.Random(i),
        )
        for i in range(100)
    ]
    assert {(p.word, p.reset) for p in plans} == {
        ((a,), mode) for a in bank.actions for mode in (False, True)
    }


def test_action_and_output_renaming_preserves_plan_ranking():
    candidates, actions, outputs = public_family()
    mapping = {a: f"action-reverse-{3 - i}" for i, a in enumerate(actions)} | {
        o: f"output-reverse-{1 - i}" for i, o in enumerate(outputs)
    }
    copied = []
    for c in candidates:
        factors = []
        for old in (c.mechanism_ir, c.controller_ir, c.sensor_ir):
            ir = deepcopy(old)
            ir.parameters["actions"] = [
                mapping.get(a, a) for a in ir.parameters["actions"]
            ]
            ir.parameters["outputs"] = [
                mapping.get(o, o) for o in ir.parameters["outputs"]
            ]
            for row in ir.parameters["transitions"]:
                row["action"], row["output"] = (
                    mapping.get(row["action"], row["action"]),
                    mapping.get(row["output"], row["output"]),
                )
            factors.append(ir)
        copied.append(
            replace(
                c,
                mechanism_ir=factors[0],
                controller_ir=factors[1],
                sensor_ir=factors[2],
                idle_output=mapping[c.idle_output],
            )
        )
    left, right = ModelBank(candidates), ModelBank(tuple(copied))
    for policy in ("joint_information_gain", "mechanism_risk"):
        lp = choose(
            left, costs={a: 1 for a in actions}, remaining=8, horizon=3, policy=policy
        )
        rp = choose(
            right,
            costs={mapping[a]: 1 for a in actions},
            remaining=8,
            horizon=3,
            policy=policy,
        )
        assert tuple(mapping[a] for a in lp.word) == rp.word
        assert (
            lp.reset,
            lp.cost,
            lp.mechanism_risk_reduction,
            lp.information_gain_bits,
        ) == (rp.reset, rp.cost, rp.mechanism_risk_reduction, rp.information_gain_bits)


@pytest.mark.parametrize(
    "budget,horizon", [(True, 2), (0, 2), (1001, 2), (5, True), (5, 0), (5, 5)]
)
def test_invalid_protocol_never_dispatches(budget, horizon):
    world, candidates, costs, _ = make_world(0, "no_change")
    with pytest.raises(ValueError):
        run_agent(
            world.agent_view(), candidates, costs=costs, budget=budget, horizon=horizon
        )
    assert world.evaluator_trace() == []
