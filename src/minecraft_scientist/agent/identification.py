"""Known-candidate H1 agent using the shared IR, capability view and registry."""

from __future__ import annotations

import random

from minecraft_scientist.discovery.discrimination import METHODS, DiscriminationSpace
from minecraft_scientist.protocols import (
    AgentWorldView,
    InterventionSpec,
    TheoryIR,
    TheoryStatus,
)
from minecraft_scientist.theory.registry import TheoryRegistry


def identify_mechanism(
    view: AgentWorldView,
    candidates: tuple[TheoryIR, ...],
    calibration: tuple[str, ...],
    costs: dict[str, int],
    *,
    method="witness_cover",
    seed=0,
    budget=20,
    horizon=5,
):
    """No ground-truth/world-family argument. All resets and actions are charged.

    A resolved behavior class is only PROVISIONAL: independent shadow validation
    and open-family revision are later gates, not silently skipped here.
    """
    if (
        method not in METHODS
        or type(seed) is not int
        or seed < 0
        or (type(budget) is not int or budget < 1)
    ):
        raise ValueError("invalid method/seed/budget")
    space = DiscriminationSpace(candidates, costs, horizon=horizon)
    rng, events, spent, queries = random.Random(seed), [], 0, 0
    stopped = None

    def execute(word, phase):
        nonlocal spent, queries
        cost = space.cost(word)
        if cost > budget - spent:
            return "budget_exhausted"
        predictions = {
            str(index): list(space.prediction(index, word)) for index in space.remaining
        }
        query_id = f"query-{queries}"
        intervention = InterventionSpec(
            intervention_id=query_id,
            mechanism="finite_transducer",
            controlled_variables={"reset": True},
            target_variable="output_trace",
            fixed_context={"language": "total-mealy-v1"},
            predicted_outcomes=predictions,
        )
        # Prediction record precedes actual reset/control dispatch.
        events.append(
            {
                "kind": "plan",
                "query_id": query_id,
                "phase": phase,
                "actions": list(word),
                "predictions": predictions,
                "remaining": list(space.remaining),
                "total_cost": cost,
                "intervention": intervention.model_dump(mode="json"),
            }
        )
        observation = view.begin_trial()
        spent += space.reset_cost
        events.append(
            {
                "kind": "reset",
                "query_id": query_id,
                "cost": space.reset_cost,
                "observation": observation.model_dump(mode="json"),
            }
        )
        available = {action.target_id: action for action in view.available_actions()}
        outputs = []
        for token in word:
            if token not in available:
                raise ValueError("planned action is unavailable after reset")
            transition = view.step(available[token]).model_copy(
                update={"intended_intervention": intervention}
            )
            spent += costs[token]
            events.append(
                {
                    "kind": "step",
                    "query_id": query_id,
                    "cost": costs[token],
                    "transition": transition.model_dump(mode="json"),
                }
            )
            if not transition.intervention_compliance or transition.controller_failure:
                events.append({"kind": "control_failure", "query_id": query_id})
                return "control_failure"  # Do not eliminate world theories.
            outputs.append(transition.event.details["signal"])
        queries += 1
        space.observe(word, tuple(outputs))
        events.append(
            {
                "kind": "evidence",
                "query_id": query_id,
                "outputs": outputs,
                "remaining": list(space.remaining),
            }
        )
        return "completed"

    calibration_status = execute(calibration, "calibration")
    if calibration_status != "completed":
        stopped = calibration_status
    while stopped is None and len(space.representatives()) > 1:
        try:
            word = space.choose(method, budget - spent, rng)
        except RuntimeError:
            stopped = "planning_budget_exhausted"
            break
        if word is None:
            stopped = "no_affordable_selected_query"
            break
        execution_status = execute(word, "discrimination")
        if execution_status != "completed":
            stopped = execution_status
            break
    classes = space.representatives()
    status = (
        "model_mismatch"
        if not space.remaining
        else (
            "conditional_behavior_identified"
            if len(classes) == 1 and stopped is None
            else "unresolved"
        )
    )
    registry, theory_events = TheoryRegistry(), []
    if status == "conditional_behavior_identified":
        representative = min(
            space.remaining,
            key=lambda index: (
                len(str(candidates[index].model_dump(mode="json"))),
                index,
            ),
        )
        ir = TheoryIR.model_validate(candidates[representative].model_dump())
        version = registry.propose(
            "identified-mechanism",
            ir,
            evidence_ids=tuple(
                event["query_id"] for event in events if event["kind"] == "evidence"
            ),
            validation_score=0.0,
            confidence=0.0,
            created_by="known-candidate-h1",
        )
        theory_events.append(version.model_dump(mode="json"))
        version = registry.transition(
            version.theory_id, version.version, TheoryStatus.PROVISIONAL
        )
        theory_events.append(version.model_dump(mode="json"))
    return {
        "schema": "known-candidate-h1-run-v1",
        "method": method,
        "seed": seed,
        "budget": budget,
        "spent": spent,
        "completed_queries": queries,
        "remaining_candidates": list(space.remaining),
        "remaining_behavior_classes": len(classes),
        "initial_behavior_classes": len(space.classes),
        "status": status,
        "stop_reason": stopped,
        "unique_causal_structure_claimed": False,
        "independent_validation_completed": False,
        "prediction_cache_misses": space.prediction_calls,
        "events": events,
        "theories": theory_events,
    }
