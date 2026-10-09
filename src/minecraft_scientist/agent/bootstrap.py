"""A paid public-family bootstrap shared by online baseline conditions.

This is known-family identification, not unconstrained structure invention.
It reuses the production identification and prospective shadow executors. No
world, private schedule, replacement model or truth callback reaches this API.
"""

import json

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.discovery.diagnosis import JointHypothesis, validate_hypotheses
from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram, validate_costs
from minecraft_scientist.theory.shadow import validate_shadow


def bootstrap_config(actions, outputs, costs, family, *, budget, horizon, seed):
    if (
        type(budget) is not int
        or budget < 1
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(seed) is not int
        or seed < 0
        or type(family) is not dict
        or set(family)
        != {"hypotheses", "calibration", "discovery_budget", "shadow_count"}
        or type(family["hypotheses"]) is not list
        or not 1 <= len(family["hypotheses"]) <= 12
        or type(family["discovery_budget"]) is not int
        or family["discovery_budget"] < 1
        or type(family["shadow_count"]) is not int
        or not 1 <= family["shadow_count"] <= 16
        or type(family["calibration"]) is not list
        or not 1 <= len(family["calibration"]) <= horizon
        or any(type(a) is not str or a not in actions for a in family["calibration"])
        or type(actions) is not tuple
        or type(outputs) is not tuple
        or any(type(a) is not str for a in (*actions, *outputs))
        or len(set(actions)) != len(actions)
        or len(set(outputs)) != len(outputs)
        or not actions
        or not outputs
        or sum(len(actions) ** n for n in range(1, horizon + 1)) > 50_000
    ):
        raise ValueError("invalid bounded public-family bootstrap")
    validate_costs(actions, costs)
    family = json.loads(json.dumps(family, allow_nan=False))
    joint = tuple(JointHypothesis.model_validate(h) for h in family["hypotheses"])
    if [h.model_dump(mode="json") for h in joint] != family["hypotheses"]:
        raise ValueError("public bootstrap hypotheses must have canonical exact types")
    candidates = validate_hypotheses(
        joint, joint[0].mechanism_id, joint[0].controller_id
    )
    for ir in candidates:
        program = FiniteStateProgram.from_ir(ir)
        if program.actions != actions or program.outputs != outputs:
            raise ValueError("public bootstrap candidate alphabets differ")
    return {
        "schema": "public-family-bootstrap-config-v1",
        "actions": list(actions),
        "outputs": list(outputs),
        "costs": dict(costs),
        "family": family,
        "protocol": {
            "budget": budget,
            "horizon": horizon,
            "seed": seed,
            "epoch": "old-epoch",
        },
    }


def learn_shared_bootstrap(view, config):
    config = bootstrap_config(
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        config["family"],
        **{k: config["protocol"][k] for k in ("budget", "horizon", "seed")},
    )
    family, protocol = config["family"], config["protocol"]
    joint = tuple(JointHypothesis.model_validate(h) for h in family["hypotheses"])
    candidates = tuple(h.compile() for h in joint)
    discovery = identify_mechanism(
        view,
        candidates,
        tuple(family["calibration"]),
        config["costs"],
        method="witness_cover",
        seed=protocol["seed"],
        budget=min(family["discovery_budget"], protocol["budget"]),
        horizon=protocol["horizon"],
        equivalence_horizon=protocol["horizon"],
        query_prefix="bootstrap-query-",
    )
    result = {
        "schema": "public-family-bootstrap-run-v1",
        "discovery": discovery,
        "validation": None,
        "events": list(discovery["events"]),
        "spent": discovery["spent"],
        "candidate_ir": None,
        "validation_query_ids": [],
        "status": "unresolved",
        "stop_reason": discovery["stop_reason"] or discovery["status"],
        "scope": "paid supplied-family behavior identification with fresh coverage validation; not unique causal factors or all-H certification",
    }
    if discovery["status"] != "conditional_behavior_identified":
        return result
    ir = TheoryIR.model_validate(
        candidates[min(discovery["remaining_candidates"])].model_dump()
    )
    ir.scope.update(
        {
            "intervention_horizon": protocol["horizon"],
            "certified_reset_only": True,
            "claim": "conditional behavior only, not unique factors",
        }
    )
    used = {tuple(e["actions"]) for e in discovery["events"] if e["kind"] == "plan"}
    validation = validate_shadow(
        view,
        ir,
        config["costs"],
        used,
        budget=protocol["budget"] - result["spent"],
        count=family["shadow_count"],
        horizon=protocol["horizon"],
        seed=protocol["seed"] + 1,
        prefix="bootstrap-shadow-",
    )
    result["validation"] = validation
    result["spent"] += validation["spent"]
    for run in validation["runs"]:
        result["events"].extend(run["events"])
    result["stop_reason"] = validation["stop_reason"]
    if validation["status"] == "passed":
        result["status"] = "provisional_validation_passed"
        result["candidate_ir"] = ir.model_dump(mode="json")
        result["validation_query_ids"] = [
            e["query_id"]
            for run in validation["runs"]
            for e in run["events"]
            if e["kind"] == "evidence"
        ]
    return result


def final_validation_ids(run):
    return (
        run["validation_query_ids"]
        if run["schema"] in ("public-family-bootstrap-run-v1", "matched-repair-run-v1")
        else run["conformance"][-1]["query_ids"]
    )


def validated(run):
    return run["status"] in (
        "provisional_validation_passed",
        "provisional_conformance_passed",
        "provisional_matched_validation_passed",
    )
