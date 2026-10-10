"""Actual finite-family query loop, with declared priors and no private inputs."""

import random
from dataclasses import asdict

from .belief import ModelBank, ModelMismatch, WorkBound, rational
from .planning import POLICIES, choose


def run_agent(
    view,
    candidates,
    *,
    costs,
    budget=96,
    horizon=4,
    policy="mechanism_risk",
    seed=0,
    ledger=None,
):
    if (
        type(budget) is not int
        or not 1 <= budget <= 1000
        or type(seed) is not int
        or not 0 <= seed < 2**64
        or policy not in POLICIES
    ):
        raise ValueError("invalid noisy Agent budget/seed/policy")
    # Validate public costs/horizon before actual actions, without using a world.
    bank = ModelBank(candidates)
    if (
        type(costs) is not dict
        or set(costs) != set(bank.actions)
        or any(type(v) is not int or not 1 <= v <= 100 for v in costs.values())
        or type(horizon) is not int
        or not 1 <= horizon <= 4
    ):
        raise ValueError("invalid action costs/horizon")
    events = [] if ledger is None else ledger
    if type(events) is not list or events:
        raise ValueError("fresh caller-owned evidence list required")
    rng, remaining, query = random.Random(seed), budget, 0
    events.append(
        {
            "kind": "reset",
            "purpose": "initialization",
            "cost": 1,
            "before": bank.snapshot(),
        }
    )
    events[-1]["observation"] = view.begin_trial().model_dump(mode="json")
    bank.reset_state()
    remaining -= 1
    status = "budget_exhausted"
    while remaining:
        try:
            plan = choose(
                bank,
                costs=costs,
                remaining=remaining,
                horizon=horizon,
                policy=policy,
                rng=rng,
            )
        except WorkBound as error:
            status = "planning_bound_exhausted"
            events.append(
                {
                    "kind": "planning_failure",
                    "reason": str(error),
                    "remaining": remaining,
                }
            )
            break
        if plan is None:
            status = (
                "bounded_no_discriminating_query"
                if policy != "uniform_word" and remaining >= min(costs.values())
                else "budget_exhausted"
            )
            break
        events.append(
            {
                "kind": "plan",
                "query": query,
                "plan": {
                    **asdict(plan),
                    "mechanism_risk_reduction": rational(plan.mechanism_risk_reduction),
                },
                "before": bank.snapshot(),
            }
        )
        if plan.reset:
            events.append(
                {
                    "kind": "reset",
                    "purpose": "chosen_experiment",
                    "query": query,
                    "cost": 1,
                }
            )
            events[-1]["observation"] = view.begin_trial().model_dump(mode="json")
            bank.reset_state()
            remaining -= 1
        for command in plan.word:
            _, prediction = bank.predict_step(command)
            item = {
                "kind": "step",
                "query": query,
                "cost": costs[command],
                "issued_command": command,
                "prediction_before_dispatch": {
                    s: rational(p) for s, p in sorted(prediction.items())
                },
            }
            events.append(item)
            action = next(a for a in view.available_actions() if a.target_id == command)
            record = view.step(action)
            item["transition"] = record.model_dump(mode="json")
            remaining -= costs[command]
            try:
                item["mixture_likelihood"] = rational(
                    bank.advance(command, record.event.details["signal"])
                )
            except (ModelMismatch, WorkBound) as error:
                status = (
                    "zero_likelihood_model_mismatch"
                    if isinstance(error, ModelMismatch)
                    else "inference_bound_exhausted"
                )
                item["inference_failure"] = str(error)
                break
            item["after"] = bank.snapshot()
        if status in ("zero_likelihood_model_mismatch", "inference_bound_exhausted"):
            break
        query += 1
    if remaining < 0:
        raise ArithmeticError("experiment exceeded prepaid budget")
    return {
        "schema": "finite-family-noisy-agent-v1",
        "policy": policy,
        "status": status,
        "spent": budget - remaining,
        "budget": budget,
        "queries": query,
        "posterior": bank.snapshot(),
        "factors": {
            factor: {key: rational(p) for key, p in bank.project(factor=factor).items()}
            for factor in ("mechanism", "controller", "sensor")
        },
        "candidate_definitions": bank.definitions(),
        "evidence": events,
        "accepted_theory": None,
        "scope": "finite supplied factor family and prior; no unknown-structure discovery, posterior calibration or independent theory promotion",
    }
