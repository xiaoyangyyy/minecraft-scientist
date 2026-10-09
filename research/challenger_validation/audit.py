"""Public receipt-policy replay, followed by independent private raw scoring.

The policy replay verifies chronology/prediction/promotion, not algorithm
originality. Raw-table scoring does not call the TheoryIR interpreter.
"""

from copy import deepcopy

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)
from research.structural_holdouts.raw_tables import compose, equivalent, validate

from .paired import run_branch


def receipt_capability(controls, actions):
    """Replay only public receipts; reject extra, reordered or changed dispatches."""
    controls = deepcopy(controls)
    position, current = 0, None
    available = tuple(
        PrimitiveAction(kind=ActionKind.INTERACT, target_id=a) for a in sorted(actions)
    )

    def next_control(kind):
        nonlocal position
        if position >= len(controls) or controls[position]["kind"] != kind:
            raise ValueError("public receipt dispatch order/coverage mismatch")
        control = controls[position]
        position += 1
        return control

    def reset():
        nonlocal current
        current = Observation.model_validate(next_control("reset")["observation"])
        return current.model_copy(deep=True)

    def step(action):
        nonlocal current
        transition = TransitionRecord.model_validate(next_control("step")["transition"])
        if transition.executed_action != action or transition.state_before != current:
            raise ValueError("public receipt action/state continuity mismatch")
        current = transition.state_after
        return transition.model_copy(deep=True)

    def observe():
        if current is None:
            raise ValueError("no observation before paid reset")
        return current.model_copy(deep=True)

    return (
        AgentWorldView(
            observe=observe,
            available=lambda: available,
            step=step,
            begin_trial=reset,
        ),
        lambda: position == len(controls),
    )


def verify_public(snapshot, result):
    if result["schema"] != "paid-validation-ablation-run-v1":
        raise ValueError("invalid paid validation result schema")
    config = snapshot["config"]
    controls = result["raw_controls"]
    paid = sum(
        1
        if e["kind"] == "reset"
        else config["costs"][e["transition"]["executed_action"]["target_id"]]
        for e in controls
    )
    if (
        type(result["spent"]) is not int
        or paid != result["spent"]
        or paid > result["budget"]
    ):
        raise ValueError("paid validation cost/budget mismatch")
    view, exhausted = receipt_capability(controls, config["actions"])
    replayed = run_branch(view, snapshot, result["policy"])
    if replayed != result or not exhausted():
        raise ValueError("public validation policy/lineage replay mismatch")
    for assignment in result["validation_assignments"]:
        if set(assignment["discovery_ids"]) & set(assignment["validation_ids"]):
            raise ValueError("discovery evidence reused for independent validation")
    return {"valid": True, "paid_cost": paid, "controls": len(controls)}


def raw_score(snapshot, result, truth):
    """Evaluator only; hidden rules never feed selection or revision."""
    config = snapshot["config"]
    before, after = (compose(**truth[epoch]) for epoch in ("before", "after"))
    state, rows, current = None, None, None
    previous_episode, steps, resets, failures = -1, 0, 0, 0
    for control in result["raw_controls"]:
        if control["kind"] == "reset":
            observation = control["observation"]
            if (
                observation["episode_id"] != previous_episode + 1
                or observation["step"] != 0
            ):
                raise ValueError("raw reset chronology mismatch")
            previous_episode = observation["episode_id"]
            program = after if previous_episode >= truth["boundary_episode"] else before
            rows, state = validate(program), program["initial_state"]
            if observation["agent_state"] != {"signal": config["outputs"][0]}:
                raise ValueError("raw reset observation mismatch")
            current = observation
            resets += 1
            continue
        record = control["transition"]
        following = record["state_after"]
        if (
            current is None
            or record["state_before"] != current
            or following["episode_id"] != previous_episode
            or following["step"] != current["step"] + 1
        ):
            raise ValueError("raw step continuity mismatch")
        steps += 1
        if (
            not record["intervention_compliance"]
            or record["controller_failure"]
            or not record["event"]["success"]
        ):
            failures += 1  # Declared failure is not a causal counterexample.
        else:
            state, output = rows[state, record["executed_action"]["target_id"]]
            if (
                record["event"]["details"]["signal"] != output
                or following["agent_state"] != {"signal": output}
                or following["last_event"] != record["event"]
                or record["observed_delta"] != {"signal": output}
            ):
                raise ValueError("independent raw output mismatch")
        current = following
    horizon = config["protocol"]["horizon"]

    def correct(ir):
        return (
            None if ir is None else equivalent(ir["parameters"], after, horizon=horizon)
        )

    return {
        "controls_match": True,
        "reset_count": resets,
        "atomic_action_count": steps,
        "reported_failed_dispatches": failures,
        "proposed_correct_at_H": correct(snapshot["candidate_ir"]),
        "promoted_correct_at_H": correct(result["candidate_ir"]),
        "behavior_change_at_H": not equivalent(before, after, horizon=horizon),
        "scope": "raw deterministic reset issued behavior; failed dispatches unscored; not physical factor identity or arbitrary structural completeness",
    }


def summarize(rows):
    return {
        "runs": len(rows),
        "eligible_prefixes": sum(r["eligible"] for r in rows),
        "finite_replacements": sum(
            r["status"] == "revision_finitely_validated" for r in rows
        ),
        "promoted_correct_at_H": sum(
            r["raw_score"]["promoted_correct_at_H"] is True for r in rows
        ),
        "promoted_wrong_at_H": sum(
            r["raw_score"]["promoted_correct_at_H"] is False for r in rows
        ),
        "candidate_rejected": sum(r["status"] == "candidate_rejected" for r in rows),
        "unresolved": sum(r["status"] == "unresolved" for r in rows),
        "rejected_wrong_at_H": sum(
            r["status"] == "candidate_rejected"
            and r["raw_score"]["proposed_correct_at_H"] is False
            for r in rows
        ),
        "rejected_correct_at_H": sum(
            r["status"] == "candidate_rejected"
            and r["raw_score"]["proposed_correct_at_H"] is True
            for r in rows
        ),
        "actual_total_cost": sum(r["spent"] for r in rows),
        "actual_prefix_cost": sum(r["prefix_spent"] for r in rows),
        "mutation_operations": sum(r["mutation_operations"] for r in rows),
    }
