"""Different discovery policies, one paid prospective promotion protocol."""

import copy

from paid_transfer import METHODS, learn_phase, phase_config
from random_wp import random_wp_plan

from minecraft_scientist.agent.revision import discover_structural_repair
from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
    TreeHypothesis,
)
from minecraft_scientist.discovery.diagnosis import JointHypothesis
from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.theory.shadow import validate_shadow

MATCHED_METHODS = (*METHODS, "structural_witness", "structural_ig")
POLICY = {
    "validation": "random-wp-plus-coverage-v1",
    "shadow_count": 3,
    "search_budget": 1000,
    "candidate_limit": 512,
}


def check_policy(policy):
    if (
        type(policy) is not dict
        or set(policy) != set(POLICY)
        or policy["validation"] != POLICY["validation"]
    ):
        raise ValueError("unregistered matched repair policy")
    for key, cap in (
        ("shadow_count", 16),
        ("search_budget", 10_000),
        ("candidate_limit", 512),
    ):
        if type(policy[key]) is not int or not 1 <= policy[key] <= cap:
            raise ValueError("invalid exact matched repair policy bound")
    return copy.deepcopy(policy)


def repair_config(
    actions,
    outputs,
    costs,
    *,
    method,
    family,
    anchor_index,
    old_ir,
    anomaly,
    policy,
    protocol,
):
    policy = check_policy(policy)
    if (
        method not in MATCHED_METHODS
        or type(anchor_index) is not int
        or not 0 <= anchor_index < len(family["hypotheses"])
    ):
        raise ValueError("invalid matched method/public anchor")
    return copy.deepcopy(
        {
            "schema": "matched-repair-config-v1",
            "method": method,
            "actions": list(actions),
            "outputs": list(outputs),
            "costs": costs,
            "family": family,
            "anchor_index": anchor_index,
            "old_ir": old_ir,
            "anomaly": anomaly,
            "policy": policy,
            "protocol": protocol,
        }
    )


def learn_matched_repair(view, config):
    config = copy.deepcopy(config)
    p, policy = config["protocol"], check_policy(config["policy"])
    result = {
        "schema": "matched-repair-run-v1",
        "method": config["method"],
        "base_config": None,
        "base_run": None,
        "structural": None,
        "random_validation": None,
        "coverage_validation": None,
        "events": [],
        "spent": 0,
        "candidate_ir": None,
        "proposed_ir": None,
        "validation_query_ids": [],
        "validation_counterexample_ids": [],
        "status": "unresolved",
        "stop_reason": None,
        "scope": "shared paid finite prospective protocol; no all-H or causal-uniqueness certification",
    }
    # A declared conservative upper bound reserves the same final protocol for
    # every discovery method. The actual final bill, not this bound, is charged.
    reserve = (p["conformance_count"] + policy["shadow_count"]) * (
        1 + p["horizon"] * max(config["costs"].values())
    )
    if p["budget"] <= reserve:
        result["stop_reason"] = "no_discovery_budget_after_validation_reserve"
        return result
    candidate = None
    if config["method"] in METHODS:
        references = (
            (TheoryIR.model_validate(config["old_ir"]),)
            if config["method"] == "adaptive"
            else ()
        )
        base_config = phase_config(
            tuple(config["actions"]),
            tuple(config["outputs"]),
            config["costs"],
            references,
            **(p | {"budget": p["budget"] - reserve}),
        )
        run = learn_phase(view, base_config)
        result["base_config"], result["base_run"] = base_config, run
        result["events"].extend(run["events"])
        result["spent"] += run["spent"]
        if run["status"] == "provisional_conformance_passed":
            candidate = TheoryIR.model_validate(run["candidate_ir"])
        else:
            result["stop_reason"] = "base_learning_" + (
                run["stop_reason"] or run["status"]
            )
            return result
    else:
        joint = tuple(
            JointHypothesis.model_validate(h) for h in config["family"]["hypotheses"]
        )
        anomaly = config["anomaly"]
        run = discover_structural_repair(
            view,
            joint,
            joint[config["anchor_index"]],
            tuple(anomaly["actions"]),
            tuple(anomaly["observed"]),
            (anomaly["query_id"],),
            config["costs"],
            method="witness_cover"
            if config["method"] == "structural_witness"
            else "sequence_ig",
            seed=p["seed"],
            budget=p["budget"],
            horizon=p["horizon"],
            search_budget=policy["search_budget"],
            candidate_limit=policy["candidate_limit"],
            validation_reserve=reserve,
        )
        result["structural"] = run
        if run["discovery"] is not None:
            result["events"].extend(run["discovery"]["events"])
        result["spent"] += run["spent"]
        if run["status"] == "provisional_structure_identified":
            candidate = TheoryIR.model_validate(run["candidate_ir"])
        else:
            result["stop_reason"] = "structural_" + (
                run["stop_reason"] or run["status"]
            )
            return result
    result["proposed_ir"] = candidate.model_dump(mode="json")
    hypothesis = TreeHypothesis(FiniteStateProgram.from_ir(candidate), ())
    try:
        plan = random_wp_plan(
            hypothesis,
            config["costs"],
            horizon=p["horizon"],
            count=p["conformance_count"],
            seed=p["seed"] + 500_000,
        )
    except LearnerStop as error:
        result["stop_reason"] = str(error)
        return result
    batch = {"plan": plan, "queries": [], "status": "unresolved"}
    result["random_validation"] = batch
    if len(plan["queries"]) < p["conformance_count"]:
        batch["status"], result["stop_reason"] = (
            "insufficient_samples",
            "validation_insufficient_samples",
        )
        return result
    if plan["total_cost"] > p["budget"] - result["spent"]:
        batch["status"], result["stop_reason"] = (
            "budget_exhausted",
            "validation_budget_exhausted",
        )
        return result
    for i, item in enumerate(plan["queries"]):
        tree = ObservationTree(
            tuple(config["actions"]),
            tuple(config["outputs"]),
            epoch="repair-epoch",
            operation_budget=p["operation_budget"],
        )
        query = OutputQueries(
            view,
            tree,
            config["costs"],
            budget=p["budget"] - result["spent"],
            horizon=p["horizon"],
            prefix=f"repair-check-{i}-",
        )
        status, reason = "matched", None
        try:
            outputs = query(
                tuple(item["actions"]), "matched-validation", hypothesis, fresh=True
            )
            if outputs != tuple(item["prediction"]):
                status = "counterexample"
        except LearnerStop as error:
            status, reason = "stopped", str(error)
        result["spent"] += query.spent
        result["events"].extend(query.events)
        batch["queries"].append(
            {
                "epoch": "repair-epoch",
                "budget": query.budget,
                "spent": query.spent,
                "events": query.events,
                "status": status,
                "stop_reason": reason,
            }
        )
        evidence = [e["query_id"] for e in query.events if e["kind"] == "evidence"]
        result["validation_query_ids"].extend(evidence)
        if status == "counterexample":
            result["validation_counterexample_ids"].extend(evidence)
        if status != "matched":
            batch["status"], result["stop_reason"] = (
                status,
                "final_validation_" + (reason or status),
            )
            return result  # Prospective evidence never refits this candidate.
    batch["status"] = "passed"
    used = {tuple(e["actions"]) for e in result["events"] if e["kind"] == "plan"}
    coverage = validate_shadow(
        view,
        candidate,
        config["costs"],
        used,
        budget=p["budget"] - result["spent"],
        count=policy["shadow_count"],
        horizon=p["horizon"],
        seed=p["seed"] + 500_001,
        prefix="repair-shadow-",
    )
    result["coverage_validation"] = coverage
    result["spent"] += coverage["spent"]
    for run in coverage["runs"]:
        result["events"].extend(run["events"])
        evidence = [e["query_id"] for e in run["events"] if e["kind"] == "evidence"]
        result["validation_query_ids"].extend(evidence)
        if run["status"] == "model_mismatch":
            result["validation_counterexample_ids"].extend(evidence)
    if coverage["status"] != "passed":
        result["stop_reason"] = "final_coverage_" + coverage["stop_reason"]
        return result
    result["validation_query_ids"] = [
        e["query_id"]
        for q in batch["queries"]
        for e in q["events"]
        if e["kind"] == "evidence"
    ] + [
        e["query_id"]
        for run in coverage["runs"]
        for e in run["events"]
        if e["kind"] == "evidence"
    ]
    result["candidate_ir"] = candidate.model_dump(mode="json")
    result["status"] = "provisional_matched_validation_passed"
    return result
