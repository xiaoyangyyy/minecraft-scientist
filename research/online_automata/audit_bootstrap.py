"""Independent raw-factor, evidence and coverage checks before private I/O."""

import hashlib
import itertools
from collections import deque

from audit_run import canonical
from audit_transfer import replay_capability
from shared_bootstrap import bootstrap_config, learn_shared_bootstrap

from minecraft_scientist.discovery.diagnosis import JointHypothesis
from minecraft_scientist.evaluation.diagnosis import (
    _factor_rollout,
    _matches_factors,
    _public_spec,
)
from minecraft_scientist.evaluation.identification import _rollout
from minecraft_scientist.protocols import ActionKind, PrimitiveAction, TransitionRecord


def classes(programs, horizon, active):
    def equal(left, right):
        lt = {(r["source"], r["action"]): r for r in left["transitions"]}
        rt = {(r["source"], r["action"]): r for r in right["transitions"]}
        pending = deque([(left["initial_state"], right["initial_state"], 0)])
        seen = set()
        while pending:
            ls, rs, d = pending.popleft()
            if (ls, rs, d) in seen or d == horizon:
                continue
            seen.add((ls, rs, d))
            for a in left["actions"]:
                l, r = lt[ls, a], rt[rs, a]
                if l["output"] != r["output"]:
                    return False
                pending.append((l["target"], r["target"], d + 1))
        return True

    representatives = []
    for i in active:
        if not any(equal(programs[i], programs[j]) for j in representatives):
            representatives.append(i)
    return len(representatives)


def check_identification(
    run, programs, costs, *, specs=None, horizon, equivalence=True
):
    remaining, spent, ids, completed = list(range(len(programs))), 0, set(), 0
    active, cursor, outputs, failed = None, 0, [], False
    reset = False

    def predict(i, word):
        return list(
            _factor_rollout(specs[i], word)
            if specs is not None
            else _rollout(programs[i], word)
        )

    for event in run["events"]:
        kind = event["kind"]
        if kind == "plan":
            if active is not None or event["query_id"] in ids:
                raise ValueError("bootstrap duplicated or unfinished frozen query")
            word = event["actions"]
            if (
                type(word) is not list
                or not 1 <= len(word) <= horizon
                or any(type(a) is not str or a not in costs for a in word)
            ):
                raise ValueError("bootstrap query outside public language")
            if (
                type(event["total_cost"]) is not int
                or event["total_cost"] != 1 + sum(costs[a] for a in word)
                or event["total_cost"] > run["budget"] - spent
                or canonical(event["remaining"]) != canonical(remaining)
                or canonical(event["predictions"])
                != canonical({str(i): predict(i, word) for i in remaining})
                or canonical(event["intervention"]["predicted_outcomes"])
                != canonical(event["predictions"])
            ):
                raise ValueError(
                    "independent bootstrap frozen predictions/cost mismatch"
                )
            active, cursor, outputs, failed, reset = event, 0, [], False, False
            ids.add(event["query_id"])
        else:
            if active is None or event["query_id"] != active["query_id"]:
                raise ValueError("bootstrap unplanned or unordered receipt")
            if kind == "reset":
                if reset or type(event["cost"]) is not int or event["cost"] != 1:
                    raise ValueError("bootstrap reset cost/order mismatch")
                spent, reset = spent + 1, True
            elif kind == "step":
                if not reset or failed or cursor >= len(active["actions"]):
                    raise ValueError("bootstrap step after failure/outside frozen word")
                record = TransitionRecord.model_validate(event["transition"])
                token = active["actions"][cursor]
                if (
                    canonical(record.model_dump(mode="json"))
                    != canonical(event["transition"])
                    or record.executed_action
                    != PrimitiveAction(kind=ActionKind.INTERACT, target_id=token)
                    or canonical(event["transition"]["intended_intervention"])
                    != canonical(active["intervention"])
                    or type(event["cost"]) is not int
                    or event["cost"] != costs[token]
                ):
                    raise ValueError(
                        "bootstrap actual action/intent/type/cost mismatch"
                    )
                spent += event["cost"]
                cursor += 1
                failed = (
                    not record.intervention_compliance
                    or bool(record.controller_failure)
                    or not record.event.success
                )
                outputs.append(record.event.details["signal"])
            elif kind == "evidence":
                if (
                    not reset
                    or failed
                    or cursor != len(active["actions"])
                    or event["outputs"] != outputs
                ):
                    raise ValueError(
                        "bootstrap incomplete/failed query retained as evidence"
                    )
                remaining = [
                    i for i in remaining if predict(i, active["actions"]) == outputs
                ]
                if canonical(event["remaining"]) != canonical(remaining):
                    raise ValueError(
                        "bootstrap independent candidate filtering mismatch"
                    )
                active, completed = None, completed + 1
            elif kind == "control_failure":
                if not failed or run["stop_reason"] != "control_failure":
                    raise ValueError("bootstrap failure silently discarded or invented")
                active = None
            else:
                raise ValueError("unregistered bootstrap event")
    if (
        active is not None
        or type(run["spent"]) is not int
        or run["spent"] != spent
        or type(run["budget"]) is not int
        or spent > run["budget"]
        or type(run["completed_queries"]) is not int
        or run["completed_queries"] != completed
        or canonical(run["remaining_candidates"]) != canonical(remaining)
        or run["unique_causal_structure_claimed"] is not False
        or run["independent_validation_completed"] is not False
    ):
        raise ValueError("bootstrap independent paid evidence ledger mismatch")
    count = classes(programs, horizon, remaining) if equivalence else len(remaining)
    initial_count = (
        classes(programs, horizon, range(len(programs)))
        if equivalence
        else len(programs)
    )
    if (
        type(run["initial_behavior_classes"]) is not int
        or run["initial_behavior_classes"] != initial_count
    ):
        raise ValueError("bootstrap independent initial behavior-class count mismatch")
    if (
        type(run["remaining_behavior_classes"]) is not int
        or run["remaining_behavior_classes"] != count
    ):
        raise ValueError("bootstrap independent behavior-class count mismatch")
    expected = (
        "model_mismatch"
        if not remaining
        else "conditional_behavior_identified"
        if count == 1 and run["stop_reason"] is None
        else "unresolved"
    )
    if run["status"] != expected:
        raise ValueError("bootstrap identification claim not supported")


def coverage_plan(ir, costs, used, *, count, horizon, seed):
    """Reimplement declared coverage selection using raw tables, not execute()."""
    program = ir["parameters"]
    words = [
        w
        for n in range(1, horizon + 1)
        for w in itertools.product(sorted(program["actions"]), repeat=n)
        if w not in used
    ]
    predictions = {w: list(_rollout(program, w)) for w in words}
    positive = [
        w for w in words if any(o != program["outputs"][0] for o in predictions[w])
    ]
    cost = lambda w: 1 + sum(costs[a] for a in w)
    tie = lambda w: hashlib.sha256(f"{seed}:{w!r}".encode()).hexdigest()
    selected, covered = [], set()
    if positive:
        selected.append(min(positive, key=lambda w: (cost(w), tie(w), w)))
        covered.update(selected[0])
    while len(selected) < count and len(selected) < len(words):
        w = min(
            (w for w in words if w not in selected),
            key=lambda w: (-len(set(w) - covered) / cost(w), cost(w), tie(w), w),
        )
        selected.append(w)
        covered.update(w)
    return {
        "schema": "prospective-shadow-plan-v1",
        "seed": seed,
        "queries": [
            {"actions": list(w), "prediction": predictions[w], "cost": cost(w)}
            for w in selected
        ],
        "total_cost": sum(cost(w) for w in selected),
        "prior_used_word_count": len(used),
        "prediction_coverage": "positive-if-available then public action coverage",
        "all_public_actions_covered": covered == set(program["actions"]),
        "candidate_ir": ir,
    }


def check_bootstrap_public(phase):
    config, result = phase["config"], phase["run"]
    expected = bootstrap_config(
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        config["family"],
        **{k: config["protocol"][k] for k in ("budget", "horizon", "seed")},
    )
    if (
        canonical(expected) != canonical(config)
        or result["schema"] != "public-family-bootstrap-run-v1"
    ):
        raise ValueError("unregistered public-family bootstrap config/run")
    family, protocol = config["family"], config["protocol"]
    joint = tuple(JointHypothesis.model_validate(h) for h in family["hypotheses"])
    specs, candidates = (
        [_public_spec(h) for h in joint],
        [h.compile().model_dump(mode="json") for h in joint],
    )
    programs = [h["parameters"] for h in candidates]
    if not all(_matches_factors(p, s) for p, s in zip(programs, specs)):
        raise ValueError("independent bootstrap compiler disagrees with public factors")
    discovery = result["discovery"]
    check_identification(
        discovery, programs, config["costs"], specs=specs, horizon=protocol["horizon"]
    )
    events, spent, validation_ids = list(discovery["events"]), discovery["spent"], []
    validation = result["validation"]
    if validation is not None:
        if discovery["status"] != "conditional_behavior_identified":
            raise ValueError("bootstrap validation without identified class")
        ir = candidates[min(discovery["remaining_candidates"])]
        ir["scope"].update(
            {
                "intervention_horizon": protocol["horizon"],
                "certified_reset_only": True,
                "claim": "conditional behavior only, not unique factors",
            }
        )
        used = {tuple(e["actions"]) for e in events if e["kind"] == "plan"}
        plan = coverage_plan(
            ir,
            config["costs"],
            used,
            count=family["shadow_count"],
            horizon=protocol["horizon"],
            seed=protocol["seed"] + 1,
        )
        if canonical(plan) != canonical(validation["plan"]):
            raise ValueError("independent prospective coverage plan mismatch")
        full = len(plan["queries"]) == family["shadow_count"]
        affordable = plan["total_cost"] <= protocol["budget"] - spent
        if (not full or not affordable) and validation["runs"]:
            raise ValueError(
                "bootstrap unaffordable/insufficient batch partially dispatched"
            )
        if len(validation["runs"]) > len(plan["queries"]):
            raise ValueError("bootstrap validation exceeds frozen batch")
        paid, rejected = 0, False
        for i, run in enumerate(validation["runs"]):
            if rejected or run["budget"] != protocol["budget"] - spent - paid:
                raise ValueError(
                    "bootstrap validation continued after failure/extra budget"
                )
            check_identification(
                run,
                [ir["parameters"]],
                config["costs"],
                horizon=protocol["horizon"],
                equivalence=False,
            )
            plans = [e for e in run["events"] if e["kind"] == "plan"]
            if len(plans) != 1 or plans[0]["actions"] != plan["queries"][i]["actions"]:
                raise ValueError("bootstrap validation word differs from frozen batch")
            paid += run["spent"]
            events.extend(run["events"])
            validation_ids.extend(
                e["query_id"] for e in run["events"] if e["kind"] == "evidence"
            )
            rejected = run["status"] != "conditional_behavior_identified"
        status = (
            "unresolved"
            if not full or not affordable
            else "rejected"
            if rejected
            else "passed"
            if len(validation["runs"]) == family["shadow_count"]
            else "incomplete"
        )
        if (
            type(validation["spent"]) is not int
            or validation["spent"] != paid
            or validation["status"] != status
        ):
            raise ValueError("bootstrap independent shadow status/cost mismatch")
        spent += paid
    accepted = validation is not None and validation["status"] == "passed"
    if (
        canonical(events) != canonical(result["events"])
        or type(result["spent"]) is not int
        or spent != result["spent"]
        or spent > protocol["budget"]
        or canonical(result["candidate_ir"]) != canonical(ir if accepted else None)
        or canonical(result["validation_query_ids"])
        != canonical(validation_ids if accepted else [])
        or result["status"]
        != ("provisional_validation_passed" if accepted else "unresolved")
    ):
        raise ValueError("bootstrap flattening/publication/paid total mismatch")
    discovery_ids = {
        e["query_id"] for e in discovery["events"] if e["kind"] == "evidence"
    }
    if discovery_ids.intersection(validation_ids) or len(set(validation_ids)) != len(
        validation_ids
    ):
        raise ValueError("bootstrap discovery/validation evidence reused")
    view, consumed = replay_capability(result)
    if (
        canonical(learn_shared_bootstrap(view, config)) != canonical(result)
        or not consumed()
    ):
        raise ValueError("full public-family bootstrap replay mismatch")
