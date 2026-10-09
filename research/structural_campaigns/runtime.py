"""Shared lifecycle adapter; no new learner or private-data planner."""

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT / "scripts",
    ROOT / "research" / "adaptive_automata",
    ROOT / "research" / "online_automata",
):
    sys.path.insert(0, str(directory))

from audit_online import audit_online, control_groups, validate_online_public
from matched_repair import MATCHED_METHODS, POLICY
from online import online_relearn

from research.structural_holdouts.audit_corpus import validate_public
from research.structural_holdouts.raw_tables import compose, equivalent, validate

PROTOCOL = {
    "budget": 600,
    "old_budget": 250,
    "horizon": 5,
    "seed": 20261008,
    "conformance_count": 8,
    "monitor_rounds": 4,
    "monitor_count": 8,
    "operation_budget": 200_000,
    "reference_operation_budget": 200_000,
}


def check_public(config, result):
    return validate_online_public(config, result)


def public_config(public, method, protocol):
    validate_public(public)
    if (
        method not in MATCHED_METHODS
        or type(protocol) is not dict
        or set(protocol) != set(PROTOCOL)
    ):
        raise ValueError("invalid shared development method/protocol")
    if any(type(v) is not int or v < 1 for v in protocol.values()) or not (
        1 <= protocol["horizon"] <= 5
        and protocol["conformance_count"] <= 64
        and protocol["monitor_count"] <= 64
        and protocol["monitor_rounds"] <= 16
    ):
        raise ValueError("invalid shared protocol bounds")
    return deepcopy(
        {
            "schema": "online-relearning-config-v3",
            "method": method,
            **public,
            "repair_policy": POLICY,
            "protocol": protocol,
        }
    )


def learn(view, config):
    # This boundary intentionally takes only capability + public config.
    return online_relearn(
        view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        method=config["method"],
        initial_family=config["initial_family"],
        repair_policy=config["repair_policy"],
        **config["protocol"],
    )


def prelude(result):
    return {k: result[k] for k in ("old", "public_old_theory", "monitoring", "anomaly")}


def raw_score(config, result, truth):
    """Evaluator-only raw products; independent of IR compiler/execute."""
    before, after = (compose(**truth[epoch]) for epoch in ("before", "after"))
    horizon = config["protocol"]["horizon"]
    old_ir = result["old"]["run"]["candidate_ir"]
    new = result["new"]
    new_ir = None if new is None else new["run"]["candidate_ir"]
    proposed_ir = None if new is None else new["run"].get("proposed_ir")

    def correct(ir, target):
        return (
            None
            if ir is None
            else equivalent(ir["parameters"], target, horizon=horizon)
        )

    matching, skipped, resets, actions = True, 0, 0, 0
    state, table = None, None
    for group in control_groups(result):
        for control in group["raw_controls"]:
            if control["kind"] == "reset":
                observation = control["observation"]
                program = (
                    after
                    if observation["episode_id"] >= truth["boundary_episode"]
                    else before
                )
                table, state = validate(program), program["initial_state"]
                matching &= observation["agent_state"] == {
                    "signal": config["outputs"][0]
                }
                resets += 1
            else:
                actions += 1
                record = control["transition"]
                if (
                    not record["intervention_compliance"]
                    or record["controller_failure"]
                    or not record["event"]["success"]
                ):
                    skipped += 1
                    continue
                state, output = table[state, record["executed_action"]["target_id"]]
                matching &= record["event"]["details"]["signal"] == output
    return {
        "controls_match": bool(matching),
        "reported_failed_dispatches": skipped,
        "reset_count": resets,
        "atomic_action_count": actions,
        "old_correct_at_H": correct(old_ir, before),
        "promoted_correct_at_H": correct(new_ir, after),
        "proposed_correct_at_H": correct(proposed_ir, after),
        "behavior_change_at_H": not equivalent(before, after, horizon=horizon),
        "scope": "independent raw deterministic reset issued behavior; not physical factor identification or noisy confidence",
    }


def audit(config, result, private_path):
    verified = audit_online(config, result, private_path)
    if not verified["valid"]:
        return verified
    score = raw_score(
        config, result, json.loads(private_path.read_text(encoding="utf-8"))
    )
    evaluation = verified["evaluation"]
    if (
        not score["controls_match"]
        or score["old_correct_at_H"] != evaluation["old_behavior_correct_at_H"]
        or score["promoted_correct_at_H"] != evaluation["new_behavior_correct_at_H"]
        or score["behavior_change_at_H"] != evaluation["behavior_change_within_H"]
        or score["reported_failed_dispatches"]
        != evaluation["known_failed_dispatches_excluded"]
    ):
        return verified | {"valid": False, "failure": "independent raw score mismatch"}
    return verified | {"raw_score": score}
