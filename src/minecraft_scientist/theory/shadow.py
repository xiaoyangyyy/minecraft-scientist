"""Prospective shadow experiments, separate from structural candidate generation.

Queries are selected by a declared public coverage protocol, not a private oracle.
Validation outcomes cannot fit this candidate; a failed validation is terminal
for the batch. Costs include every reset/action through the shared H1 executor.
"""

from __future__ import annotations

import hashlib
import itertools

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram, validate_costs


def validate_shadow(
    view,
    candidate,
    costs,
    used_words,
    *,
    budget,
    count=3,
    horizon=5,
    seed=20261009,
    prefix="shadow-",
):
    if (
        type(budget) is not int
        or budget < 0
        or type(count) is not int
        or not 1 <= count <= 16
        or type(seed) is not int
        or seed < 0
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(prefix) is not str
        or not 1 <= len(prefix) <= 32
    ):
        raise ValueError("invalid shadow experiment configuration")
    ir = TheoryIR.model_validate(candidate.model_dump())
    certified = ir.scope.get("intervention_horizon")
    if certified is not None and (
        type(certified) is not int or not 1 <= horizon <= certified <= 64
    ):
        raise ValueError("validation queries exceed the candidate's certified horizon")
    if ir.scope.get("required_context") is not None:
        raise ValueError(
            "this shadow protocol does not implement conditional execution contexts"
        )
    program = FiniteStateProgram.from_ir(ir)
    validate_costs(program.actions, costs)
    costs = dict(costs)
    if sum(len(program.actions) ** depth for depth in range(1, horizon + 1)) > 50_000:
        raise ValueError("shadow query universe exceeds the computation cap")
    if type(used_words) not in (tuple, list, set, frozenset) or any(
        type(word) is not tuple
        or not 1 <= len(word) <= horizon
        or any(
            type(action) is not str or action not in program.actions for action in word
        )
        for word in used_words
    ):
        raise ValueError("used validation exclusions must be declared query tuples")
    used = set(used_words)
    words = [
        word
        for length in range(1, horizon + 1)
        for word in itertools.product(sorted(program.actions), repeat=length)
        if word not in used
    ]
    predictions = {word: program.execute(word)[0] for word in words}
    positive = [
        word
        for word in words
        if any(output != program.outputs[0] for output in predictions[word])
    ]
    selected, covered = [], set()

    def cost(word):
        return 1 + sum(costs[action] for action in word)

    def tie(word):
        return hashlib.sha256(f"{seed}:{word!r}".encode()).hexdigest()

    if positive:
        selected.append(min(positive, key=lambda word: (cost(word), tie(word), word)))
        covered.update(selected[0])
    while len(selected) < count and len(selected) < len(words):
        word = min(
            (word for word in words if word not in selected),
            key=lambda word: (
                -len(set(word) - covered) / cost(word),
                cost(word),
                tie(word),
                word,
            ),
        )
        selected.append(word)
        covered.update(word)
    plan = {
        "schema": "prospective-shadow-plan-v1",
        "seed": seed,
        "queries": [
            {
                "actions": list(word),
                "prediction": list(predictions[word]),
                "cost": cost(word),
            }
            for word in selected
        ],
        "total_cost": sum(cost(word) for word in selected),
        "prior_used_word_count": len(used),
        "prediction_coverage": "positive-if-available then public action coverage",
        "all_public_actions_covered": covered == set(program.actions),
        "candidate_ir": ir.model_dump(mode="json"),
    }
    result = {
        "schema": "prospective-shadow-run-v1",
        "plan": plan,
        "runs": [],
        "spent": 0,
        "status": "unresolved",
    }
    if len(selected) != count:
        result["stop_reason"] = "insufficient_unused_queries"
        return result
    if plan["total_cost"] > budget:
        result["stop_reason"] = "validation_budget_exhausted"
        return result  # No partial dispatch then accidental acceptance.
    for index, word in enumerate(selected):
        trial = identify_mechanism(
            view,
            (ir,),
            word,
            costs,
            budget=budget - result["spent"],
            horizon=horizon,
            query_prefix=f"{prefix}{index}-",
        )
        result["runs"].append(trial)
        result["spent"] += trial["spent"]
        if trial["status"] != "conditional_behavior_identified":
            result["status"], result["stop_reason"] = (
                "rejected",
                trial["stop_reason"] or trial["status"],
            )
            return result
    result["status"], result["stop_reason"] = "passed", None
    return result
