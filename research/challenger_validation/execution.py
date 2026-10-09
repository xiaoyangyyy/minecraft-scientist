"""Actual paid fresh validation through the existing H1 executor."""

from minecraft_scientist.agent.identification import identify_mechanism

from .planning import make_plan


def validate_challengers(
    view,
    candidate,
    costs,
    used_words,
    evidence,
    *,
    budget,
    count=3,
    horizon=5,
    mutant_limit=512,
    operation_budget=200_000,
    prefix="challenge-",
):
    if type(prefix) is not str or not 1 <= len(prefix) <= 32:
        raise ValueError("invalid mutation query prefix")
    plan = make_plan(
        candidate,
        costs,
        used_words,
        evidence,
        budget=budget,
        count=count,
        horizon=horizon,
        mutant_limit=mutant_limit,
        operation_budget=operation_budget,
    )
    result = {
        "schema": "public-mutation-validation-run-v1",
        "plan": plan,
        "runs": [],
        "spent": 0,
        "status": "unresolved",
        "stop_reason": plan["stop_reason"],
    }
    if plan["status"] != "ready":
        return result  # No partial dispatch followed by accidental acceptance.
    for index, query in enumerate(plan["queries"]):
        trial = identify_mechanism(
            view,
            (candidate,),
            tuple(query["actions"]),
            costs,
            budget=budget - result["spent"],
            horizon=horizon,
            query_prefix=f"{prefix}{index}-",
        )
        result["runs"].append(trial)
        result["spent"] += trial["spent"]
        if trial["status"] == "model_mismatch":
            result["status"], result["stop_reason"] = (
                "rejected",
                "prospective_candidate_counterexample",
            )
            return result
        if trial["status"] != "conditional_behavior_identified":
            result["stop_reason"] = trial["stop_reason"] or trial["status"]
            return result  # Control/measurement failure is not a theory rejection.
    result["status"], result["stop_reason"] = "passed", None
    return result
