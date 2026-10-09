"""All-context public gates precede either private read; raw lineage + replay."""

import json

from audit_run import audit_adaptive, canonical
from paid_transfer import METHODS, phase_config, transfer_contexts

from minecraft_scientist.protocols import (
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)


def replay_capability(run):
    controls = [e for e in run["events"] if e["kind"] in ("reset", "step")]
    cursor, current = 0, None

    def reset():
        nonlocal cursor, current
        if cursor >= len(controls) or controls[cursor]["kind"] != "reset":
            raise ValueError("transfer replay reset chronology mismatch")
        current = Observation.model_validate(controls[cursor]["observation"])
        cursor += 1
        return current

    def available():
        return tuple(
            PrimitiveAction(kind=o.affordances[0], target_id=o.object_id)
            for o in current.objects
        )

    def step(action):
        nonlocal cursor, current
        if cursor >= len(controls) or controls[cursor]["kind"] != "step":
            raise ValueError("transfer replay action chronology mismatch")
        record = TransitionRecord.model_validate(controls[cursor]["transition"])
        if record.executed_action != action or record.state_before != current:
            raise ValueError("transfer replay public action/state mismatch")
        cursor += 1
        current = record.state_after
        return record

    return AgentWorldView(
        observe=lambda: current, available=available, step=step, begin_trial=reset
    ), lambda: cursor == len(controls)


def validate_transfer_public(config, result):
    if (
        type(config) is not dict
        or set(config)
        != {"schema", "method", "actions", "outputs", "costs", "protocol"}
        or config["schema"] != "paid-transfer-config-v1"
        or config["method"] not in METHODS
    ):
        raise ValueError("unregistered transfer campaign")
    protocol = config["protocol"]
    if set(protocol) != {
        "budget",
        "old_budget",
        "horizon",
        "seed",
        "conformance_count",
        "operation_budget",
        "reference_operation_budget",
    }:
        raise ValueError("unregistered transfer resource protocol")
    for name, low, high in (
        ("budget", 1, None),
        ("old_budget", 1, None),
        ("horizon", 1, 8),
        ("seed", 0, None),
        ("conformance_count", 1, 64),
        ("operation_budget", 1, None),
        ("reference_operation_budget", 1, None),
    ):
        value = protocol[name]
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError("independent transfer protocol type/bounds mismatch")
    if (
        result["schema"] != "paid-context-transfer-v1"
        or result["method"] != config["method"]
        or type(result["budget"]) is not int
        or result["budget"] != protocol["budget"]
    ):
        raise ValueError("independent transfer identity/budget mismatch")
    for claim in (
        "unique_causal_structure_claimed",
        "all_allowed_queries_certified",
        "unannounced_online_mutation_claimed",
    ):
        if result[claim] is not False:
            raise ValueError("independent transfer falsely claims broader scope")
    phases = [result["old"]] + ([] if result["new"] is None else [result["new"]])
    for phase in phases:
        checked = audit_adaptive(phase["config"], phase["run"], None, public_only=True)
        if not checked["valid"] or checked["private_truth_read_attempted"]:
            raise ValueError(
                "all-context public gate: "
                + checked.get("failure", "unexpected private read")
            )
    spent = sum(phase["run"]["spent"] for phase in phases)
    if (
        type(result["spent"]) is not int
        or result["spent"] != spent
        or spent > protocol["budget"]
    ):
        raise ValueError("independent combined paid ledger mismatch")
    if (
        result["old_budget_cap"] != min(protocol["old_budget"], protocol["budget"])
        or type(result["old_budget_cap"]) is not int
    ):
        raise ValueError("independent old budget cap mismatch")
    common = {
        k: protocol[k]
        for k in (
            "horizon",
            "seed",
            "conformance_count",
            "operation_budget",
            "reference_operation_budget",
        )
    }
    expected_old = phase_config(
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        (),
        budget=result["old_budget_cap"],
        epoch="old-context",
        **common,
    )
    if canonical(expected_old) != canonical(result["old"]["config"]):
        raise ValueError("independent old-phase public prior/protocol mismatch")
    events, assignments, old_theory = [], [], None
    for index, phase in enumerate(phases, 1):
        run = phase["run"]
        if run["status"] != "provisional_conformance_passed":
            continue
        context = "old-context" if index == 1 else "new-context"
        final = run["conformance"][-1]["query_ids"]
        discovery = [
            f"{context}/{e['query_id']}"
            for e in run["events"]
            if e["kind"] == "evidence" and e["query_id"] not in final
        ]
        validation = [f"{context}/{qid}" for qid in final]
        if len(set(validation)) != protocol["conformance_count"] or set(
            discovery
        ) & set(validation):
            raise ValueError("independent qualified discovery/validation overlap")
        ir = json.loads(canonical(run["candidate_ir"]))
        ir["scope"]["context_id"] = context
        base = {
            "theory_id": "issued-response",
            "version": index,
            "ir": ir,
            "parent_version": None if index == 1 else 1,
            "evidence_ids": discovery,
            "counterexample_ids": [],
            "validation_score": 0.0,
            "confidence": 0.0,
            "created_by": "ordered-lsharp-paid-context-transfer",
        }
        for status in ("proposed", "provisional", "accepted"):
            events.append(base | {"status": status})
        assignments.append(
            {
                "context": context,
                "version": index,
                "discovery_ids": discovery,
                "validation_ids": validation,
                "confidence_interpretation": "not estimated; stored zero is not a posterior",
            }
        )
        if index == 1:
            old_theory = events[-1]
    if (
        canonical(events) != canonical(result["theory_events"])
        or canonical(assignments) != canonical(result["validation_assignments"])
        or canonical(old_theory) != canonical(result["public_old_theory"])
    ):
        raise ValueError("independent theory/context/validation lineage mismatch")
    if result["new"] is not None:
        if old_theory is None:
            raise ValueError("new context acquired a free/unvalidated old theory")
        new_config = result["new"]["config"]
        references = [old_theory["ir"]] if config["method"] == "adaptive" else []
        if (
            new_config["references"] != references
            or new_config["protocol"]["budget"]
            != protocol["budget"] - result["old"]["run"]["spent"]
        ):
            raise ValueError(
                "independent new-context reference/remaining budget mismatch"
            )
        expected_protocol = common | {
            "budget": protocol["budget"] - result["old"]["run"]["spent"],
            "epoch": "new-context",
        }
        if canonical(expected_protocol) != canonical(new_config["protocol"]) or any(
            new_config[k] != config[k] for k in ("actions", "outputs", "costs")
        ):
            raise ValueError(
                "independent new-context alphabet/tester protocol mismatch"
            )
    old_view, old_consumed = replay_capability(result["old"]["run"])
    if result["new"] is None:
        new_view, new_consumed = None, lambda: True
    else:
        new_view, new_consumed = replay_capability(result["new"]["run"])
    replay = transfer_contexts(
        old_view,
        new_view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        method=config["method"],
        **protocol,
    )
    if (
        not old_consumed()
        or not new_consumed()
        or canonical(replay) != canonical(result)
    ):
        raise ValueError("public full transfer campaign replay mismatch")


def audit_transfer(config, result, old_private, new_private):
    public = {"all_contexts_public_valid": False, "private_truth_read_attempted": False}
    try:
        # Detach a stable audit snapshot before any callbacks/private artifact
        # reads; callers cannot alter the second context during the first read.
        config, result = json.loads(canonical(config)), json.loads(canonical(result))
        validate_transfer_public(config, result)
        public["all_contexts_public_valid"] = True
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
    ) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}
    public["private_truth_read_attempted"] = True
    old = audit_adaptive(result["old"]["config"], result["old"]["run"], old_private)
    new = (
        None
        if result["new"] is None
        else audit_adaptive(result["new"]["config"], result["new"]["run"], new_private)
    )
    valid = old["valid"] and (new is None or new["valid"])
    old_correct = (
        None
        if old["evaluation"] is None
        else old["evaluation"]["provisional_behavior_correct_at_H"]
    )
    new_correct = (
        None
        if new is None or new["evaluation"] is None
        else new["evaluation"]["provisional_behavior_correct_at_H"]
    )
    chain = (
        None
        if old_correct is None or new_correct is None
        else old_correct and new_correct
    )
    return public | {
        "valid": valid,
        "old_audit": old,
        "new_audit": new,
        "evaluation": {
            "old_behavior_correct_at_H": old_correct,
            "new_behavior_correct_at_H": new_correct,
            "both_contexts_correct_at_H": chain,
            "scope": "paid explicit-context behavior transfer only; no private scoring feedback to learner, hidden mutation or causal factor claim",
        },
    }
