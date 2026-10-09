"""Global chronology, raw monitoring/lineage and replay BEFORE private scoring."""

import json
from collections import deque

from audit_random_wp import check_random_wp_plan
from audit_run import PublicLedger, canonical
from audit_transfer import replay_capability
from online import online_relearn
from paid_transfer import METHODS, learn_phase, phase_config

from minecraft_scientist.evaluation.diagnosis import _factor_step, _rows
from minecraft_scientist.protocols import (
    ActionKind,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram

PROTOCOL_FIELDS = {
    "budget",
    "old_budget",
    "horizon",
    "seed",
    "conformance_count",
    "monitor_rounds",
    "monitor_count",
    "operation_budget",
    "reference_operation_budget",
}


def issued_scope(record):
    """A declared controller warning does not verify a physical do-treatment."""
    return record.possible_confounders == (
        "unverified-actuator-delivery",
    ) and record.context == {
        "intervention_compliance_scope": "issued-command",
        "treatment_verification": "unknown",
    }


class IssuedBehaviorLedger(PublicLedger):
    def control(self, event):
        original = event
        if event["kind"] == "step":
            record = TransitionRecord.model_validate(event["transition"])
            if canonical(record.model_dump(mode="json")) != canonical(
                event["transition"]
            ):
                raise ValueError("independent original issued receipt type mismatch")
            if record.possible_confounders and issued_scope(record):
                # Project this explicit warning only for the inherited issued-
                # behavior checker. Keep original warnings in receipts and
                # replay; neither treatment delivery nor causal factors become
                # verified through this projection.
                event = json.loads(canonical(event))
                event["transition"]["possible_confounders"] = []
        super().control(event)
        self.controls[-1] = original


def check_phase_public(phase):
    ledger = IssuedBehaviorLedger(phase["config"], phase["run"])
    ledger.validate()
    view, consumed = replay_capability(phase["run"])
    if (
        canonical(learn_phase(view, phase["config"])) != canonical(phase["run"])
        or not consumed()
    ):
        raise ValueError("full public issued-behavior phase replay mismatch")


def control_groups(result):
    yield result["old"]
    for batch in result["monitoring"]:
        yield from batch["queries"]
    if result["new"] is not None:
        yield result["new"]


def local_events(group):
    return group["run"]["events"] if "run" in group else group["events"]


def check_global_controls(config, result):
    """Original world clocks remain continuous; local rebasing is reversible.

    This does not call RecordedEpoch.local. Issued intent is frozen in the query
    ledger and attached above the underlying world, so only that field may differ
    between a raw receipt and its corresponding learner receipt.
    """
    episode, tick, current = -1, 0, None
    receipts, controls = set(), []

    def observation(raw):
        parsed = Observation.model_validate(raw)
        if (
            canonical(parsed.model_dump(mode="json")) != canonical(raw)
            or type(raw["episode_id"]) is not int
            or type(raw["step"]) is not int
            or parsed.episode_id != episode
            or parsed.step != tick
            or set(parsed.agent_state) != {"signal"}
            or parsed.agent_state["signal"] not in config["outputs"]
            or len(parsed.objects) != len(config["actions"])
            or {o.object_id for o in parsed.objects} != set(config["actions"])
            or any(o.affordances != (ActionKind.INTERACT,) for o in parsed.objects)
        ):
            raise ValueError("independent global clock/alphabet/type mismatch")
        return parsed

    for group in control_groups(result):
        raw = group["raw_controls"]
        local = [e for e in local_events(group) if e["kind"] in ("reset", "step")]
        if type(raw) is not list or len(raw) != len(local):
            raise ValueError("independent local/global receipt count mismatch")
        base = None
        for original, rebased in zip(raw, local):
            if original["kind"] != rebased["kind"]:
                raise ValueError("independent local/global receipt order mismatch")
            if original["kind"] == "reset":
                if set(original) != {"kind", "observation"}:
                    raise ValueError("unregistered global reset fields")
                episode, tick = episode + 1, 0
                current = observation(original["observation"])
                if current.last_event is not None or current.inventory:
                    raise ValueError("independent global reset retained state")
                if base is None:
                    base = episode
                expected = json.loads(canonical(original["observation"]))
                expected["episode_id"] -= base
                if canonical(expected) != canonical(rebased["observation"]):
                    raise ValueError("independent reversible reset mapping mismatch")
            elif original["kind"] == "step":
                if set(original) != {"kind", "transition"} or current is None:
                    raise ValueError("unregistered/pre-reset global step")
                value = original["transition"]
                record = TransitionRecord.model_validate(value)
                if canonical(record.model_dump(mode="json")) != canonical(value):
                    raise ValueError("independent global receipt exact type mismatch")
                if (
                    record.transition_id in receipts
                    or record.state_before != current
                    or record.executed_action.kind is not ActionKind.INTERACT
                    or record.executed_action.target_id not in config["actions"]
                ):
                    raise ValueError(
                        "independent global action/receipt continuity mismatch"
                    )
                tick += 1
                after = observation(value["state_after"])
                if (
                    record.event.event_type != "public_signal"
                    or set(record.event.details) != {"signal"}
                    or after.last_event != record.event
                    or after.agent_state != record.event.details
                    or record.observed_delta != record.event.details
                    or record.state_before.objects != after.objects
                    or record.state_before.inventory != after.inventory
                    or (record.possible_confounders and not issued_scope(record))
                ):
                    raise ValueError(
                        "independent global public signal consistency mismatch"
                    )
                expected = json.loads(canonical(value))
                for key in ("state_before", "state_after"):
                    expected[key]["episode_id"] -= base
                expected["intended_intervention"] = rebased["transition"][
                    "intended_intervention"
                ]
                if canonical(expected) != canonical(rebased["transition"]):
                    raise ValueError("independent reversible step mapping mismatch")
                receipts.add(record.transition_id)
                current = after
            else:
                raise ValueError("unregistered global control kind")
            controls.append(original)
    return controls


def check_monitor_query(config, q, item, *, index, sample_index, remaining):
    epoch = f"monitor-{index}-{sample_index}"
    qid = epoch + "-0"
    if q["epoch"] != epoch or type(q["budget"]) is not int or q["budget"] != remaining:
        raise ValueError("independent monitor epoch/remaining budget mismatch")
    intervention = {
        "intervention_id": qid,
        "mechanism": "finite_transducer",
        "controlled_variables": {"reset": True},
        "target_variable": "output_trace",
        "fixed_context": {"language": "total-mealy-v1", "evidence_epoch": epoch},
        "predicted_outcomes": {"hypothesis": item["prediction"]},
        "stopping_condition": "terminal_event",
    }
    expected_plan = {
        "kind": "plan",
        "query_id": qid,
        "epoch": epoch,
        "reason": "monitor",
        "actions": item["actions"],
        "prediction": item["prediction"],
        "total_cost": item["cost"],
        "intervention": intervention,
    }
    events = q["events"]
    if type(events) is not list or not 3 <= len(events) <= len(item["actions"]) + 3:
        raise ValueError("independent monitor event bound mismatch")
    if canonical(events[0]) != canonical(expected_plan):
        raise ValueError("independent frozen monitor intent/prediction mismatch")
    reset = events[1]
    if (
        set(reset) != {"kind", "query_id", "cost", "observation"}
        or reset["kind"] != "reset"
        or reset["query_id"] != qid
        or type(reset["cost"]) is not int
        or reset["cost"] != 1
        or reset["observation"]["episode_id"] != 0
        or reset["observation"]["step"] != 0
    ):
        raise ValueError("independent paid monitor reset mismatch")
    spent, outputs, bad = 1, [], False
    for event in events[2:-1]:
        cursor = len(outputs)
        if (
            set(event) != {"kind", "query_id", "cost", "transition"}
            or event["kind"] != "step"
            or event["query_id"] != qid
            or cursor >= len(item["actions"])
            or bad
        ):
            raise ValueError("independent unplanned monitor step")
        record = TransitionRecord.model_validate(event["transition"])
        action = item["actions"][cursor]
        if (
            record.executed_action
            != PrimitiveAction(kind=ActionKind.INTERACT, target_id=action)
            or canonical(event["transition"]["intended_intervention"])
            != canonical(intervention)
            or type(event["cost"]) is not int
            or event["cost"] != config["costs"][action]
        ):
            raise ValueError("independent monitor action/intent/cost mismatch")
        outputs.append(record.event.details["signal"])
        spent += event["cost"]
        bad = (
            not record.intervention_compliance
            or bool(record.controller_failure)
            or not record.event.success
        )
    terminal = events[-1]
    if bad:
        if (
            canonical(terminal)
            != canonical({"kind": "control_failure", "query_id": qid})
            or q["status"] != "stopped"
            or q["stop_reason"] != "control_failure"
        ):
            raise ValueError("independent failed monitor retained as evidence")
    else:
        if len(outputs) != len(item["actions"]) or canonical(terminal) != canonical(
            {"kind": "evidence", "query_id": qid, "outputs": outputs}
        ):
            raise ValueError("independent incomplete monitor evidence")
        expected = "matched" if outputs == item["prediction"] else "counterexample"
        if q["status"] != expected or q["stop_reason"] is not None:
            raise ValueError("independent monitor result/prediction mismatch")
    if type(q["spent"]) is not int or q["spent"] != spent or spent > remaining:
        raise ValueError("independent monitor paid ledger mismatch")
    return outputs, bad


def check_lineage(result):
    events, assignments = [], []

    def publish(phase, epoch, version, parent):
        run = phase["run"]
        final = run["conformance"][-1]["query_ids"]
        discovery = [
            f"{epoch}/{e['query_id']}"
            for e in run["events"]
            if e["kind"] == "evidence" and e["query_id"] not in final
        ]
        validation = [f"{epoch}/{q}" for q in final]
        if (
            set(discovery) & set(validation)
            or len(set(validation)) != phase["config"]["protocol"]["conformance_count"]
        ):
            raise ValueError("independent online validation evidence overlap/count")
        ir = json.loads(canonical(run["candidate_ir"]))
        ir["scope"]["evidence_epoch"] = epoch
        base = {
            "theory_id": "issued-response",
            "version": version,
            "ir": ir,
            "parent_version": parent,
            "evidence_ids": discovery,
            "counterexample_ids": [],
            "validation_score": 0.0,
            "confidence": 0.0,
            "created_by": "ordered-lsharp-paid-online-relearning",
        }
        for status in ("proposed", "provisional", "accepted"):
            events.append(base | {"status": status})
        assignments.append(
            {
                "epoch": epoch,
                "version": version,
                "discovery_ids": discovery,
                "validation_ids": validation,
                "confidence_interpretation": "not estimated; stored zero is not a posterior",
            }
        )
        return events[-1]

    old = None
    if result["old"]["run"]["status"] == "provisional_conformance_passed":
        old = publish(result["old"], "old-epoch", 1, None)
    if canonical(old) != canonical(result["public_old_theory"]):
        raise ValueError("independent online paid old theory mismatch")
    if result["anomaly"] is not None:
        if old is None:
            raise ValueError("independent crisis without validated old theory")
        challenged = old | {
            "status": "challenged",
            "counterexample_ids": [result["anomaly"]["query_id"]],
        }
        events.append(challenged)
        if (
            result["new"] is not None
            and result["new"]["run"]["status"] == "provisional_conformance_passed"
        ):
            publish(result["new"], "repair-epoch", 2, 1)
            events.append(challenged | {"status": "superseded"})
    if canonical(events) != canonical(result["theory_events"]) or canonical(
        assignments
    ) != canonical(result["validation_assignments"]):
        raise ValueError("independent online theory/validation lineage mismatch")


def validate_online_public(config, result):
    if (
        type(config) is not dict
        or set(config)
        != {"schema", "method", "actions", "outputs", "costs", "protocol"}
        or config["schema"] != "online-relearning-config-v1"
        or type(config["method"]) is not str
        or config["method"] not in METHODS
    ):
        raise ValueError("unregistered online campaign")
    protocol = config["protocol"]
    if type(protocol) is not dict or set(protocol) != PROTOCOL_FIELDS:
        raise ValueError("unregistered online resource protocol")
    for name, low, high in (
        ("budget", 1, None),
        ("old_budget", 1, None),
        ("horizon", 1, 8),
        ("seed", 0, None),
        ("conformance_count", 1, 64),
        ("monitor_count", 1, 64),
        ("monitor_rounds", 1, 16),
        ("operation_budget", 1, None),
        ("reference_operation_budget", 1, None),
    ):
        value = protocol[name]
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError("independent exact online protocol bound mismatch")
    if (
        result["schema"] != "paid-online-relearning-v1"
        or result["method"] != config["method"]
        or type(result["budget"]) is not int
        or result["budget"] != protocol["budget"]
    ):
        raise ValueError("independent online identity/budget mismatch")
    if (
        any(
            result[k] is not False
            for k in (
                "mechanism_change_warranted",
                "unique_causal_structure_claimed",
                "all_allowed_queries_certified",
            )
        )
        or result["mechanism_attribution_status"] != "unresolved"
    ):
        raise ValueError("independent unsupported online causal claim")
    phases = [result["old"]] + ([] if result["new"] is None else [result["new"]])
    for phase in phases:
        check_phase_public(phase)
    controls = check_global_controls(config, result)
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
        budget=min(protocol["old_budget"], protocol["budget"]),
        epoch="old-epoch",
        **common,
    )
    if canonical(expected_old) != canonical(result["old"]["config"]):
        raise ValueError("independent online bootstrap prior/protocol mismatch")
    spent, anomaly, terminal = result["old"]["run"]["spent"], None, False
    if (
        type(result["monitoring"]) is not list
        or len(result["monitoring"]) > protocol["monitor_rounds"]
    ):
        raise ValueError("independent bounded monitor count mismatch")
    for index, batch in enumerate(result["monitoring"]):
        if terminal or result["public_old_theory"] is None:
            raise ValueError("independent monitor after terminal/without old theory")
        plan = batch["plan"]
        check_random_wp_plan(plan, config["costs"])
        if (
            canonical(plan["hypothesis_program"])
            != canonical(result["old"]["run"]["candidate_ir"]["parameters"])
            or plan["seed"] != protocol["seed"] + 100_000 + index
            or plan["horizon"] != protocol["horizon"]
            or plan["requested_count"] != protocol["monitor_count"]
        ):
            raise ValueError("independent monitor tester/frozen theory mismatch")
        if len(plan["queries"]) < protocol["monitor_count"]:
            if batch["queries"] or batch["status"] != "insufficient_samples":
                raise ValueError("independent insufficient monitor samples dispatched")
            terminal = True
            continue
        if plan["total_cost"] > protocol["budget"] - spent:
            if batch["queries"] or batch["status"] != "budget_exhausted":
                raise ValueError("independent unaffordable monitor batch dispatched")
            terminal = True
            continue
        if (
            type(batch["queries"]) is not list
            or not 1 <= len(batch["queries"]) <= protocol["monitor_count"]
        ):
            raise ValueError("independent monitor query count mismatch")
        expected_status = "passed"
        for sample_index, (q, item) in enumerate(
            zip(batch["queries"], plan["queries"])
        ):
            if terminal:
                raise ValueError("independent monitor continued after failure/anomaly")
            observed, bad = check_monitor_query(
                config,
                q,
                item,
                index=index,
                sample_index=sample_index,
                remaining=protocol["budget"] - spent,
            )
            spent += q["spent"]
            if bad:
                terminal, expected_status = True, "stopped"
            elif observed != item["prediction"]:
                terminal, expected_status = True, "counterexample"
                anomaly = {
                    "query_id": f"{q['epoch']}/{q['epoch']}-0",
                    "episode_id": q["raw_controls"][0]["observation"]["episode_id"],
                    "actions": item["actions"],
                    "prediction": item["prediction"],
                    "observed": observed,
                    "claim": "issued behavior counterexample, not physical mechanism attribution",
                }
        if batch["status"] != expected_status or (
            not terminal and len(batch["queries"]) != protocol["monitor_count"]
        ):
            raise ValueError("independent monitor completion status mismatch")
    if canonical(anomaly) != canonical(result["anomaly"]):
        raise ValueError("independent online anomaly/execution mismatch")
    if result["new"] is not None:
        if anomaly is None or result["public_old_theory"] is None:
            raise ValueError("independent repair without observed crisis")
        references = (
            [result["public_old_theory"]["ir"]]
            if config["method"] == "adaptive"
            else []
        )
        expected = {
            "schema": "adaptive-config-v1",
            "actions": config["actions"],
            "outputs": config["outputs"],
            "costs": config["costs"],
            "references": references,
            "protocol": common
            | {"budget": protocol["budget"] - spent, "epoch": "repair-epoch"},
        }
        if canonical(expected) != canonical(result["new"]["config"]):
            raise ValueError("independent repair prior/remaining budget mismatch")
        spent += result["new"]["run"]["spent"]
    if (
        type(result["spent"]) is not int
        or result["spent"] != spent
        or spent > protocol["budget"]
    ):
        raise ValueError("independent total online paid ledger mismatch")
    check_lineage(result)
    view, consumed = replay_capability({"events": controls})
    replay = online_relearn(
        view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        method=config["method"],
        **protocol,
    )
    if not consumed() or canonical(replay) != canonical(result):
        raise ValueError("complete public online policy replay mismatch")
    return controls


def raw_spec(spec, config):
    if type(spec) is not dict or set(spec) not in (
        {"program"},
        {"mechanism", "controller"},
    ):
        raise ValueError("unregistered private behavioral components")
    for name, value in spec.items():
        program = FiniteStateProgram.model_validate(value).model_dump(mode="json")
        if (
            canonical(program) != canonical(value)
            or (name != "mechanism" and program["actions"] != config["actions"])
            or (
                name == "mechanism"
                and not set(program["actions"]).issubset(config["actions"])
            )
            or (name != "controller" and program["outputs"] != config["outputs"])
        ):
            raise ValueError("private/public component/alphabet mismatch")
    return spec


def initial(spec):
    return (
        (spec["program"]["initial_state"],)
        if "program" in spec
        else (spec["mechanism"]["initial_state"], spec["controller"]["initial_state"])
    )


def raw_step(spec, state, action):
    if "program" in spec:
        row = _rows(spec["program"])[state[0], action]
        return (row["target"],), row["output"]
    mechanism, controller = spec["mechanism"], spec["controller"]
    left, right, output = _factor_step(
        mechanism, _rows(mechanism), _rows(controller), state[0], state[1], action
    )
    return (left, right), output


def same_behavior(left, right, actions, horizon):
    pending, seen = deque([(initial(left), initial(right), 0)]), set()
    while pending:
        ls, rs, depth = pending.popleft()
        if (ls, rs, depth) in seen or depth == horizon:
            continue
        seen.add((ls, rs, depth))
        for action in actions:
            ln, lo = raw_step(left, ls, action)
            rn, ro = raw_step(right, rs, action)
            if lo != ro:
                return False
            pending.append((ln, rn, depth + 1))
    return True


def audit_online(config, result, private_path):
    public = {
        "all_public_valid": False,
        "private_truth_read_attempted": False,
        "private_truth_opened": False,
    }
    try:
        config, result = json.loads(canonical(config)), json.loads(canonical(result))
        controls = validate_online_public(config, result)
        public["all_public_valid"] = True
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
    try:
        truth = json.loads(private_path.read_text(encoding="utf-8"))
        public["private_truth_opened"] = True
        if (
            type(truth) is not dict
            or set(truth) != {"schema", "boundary_episode", "before", "after"}
            or truth["schema"]
            not in ("online-automata-private-v1", "online-revision-private-v1")
            or type(truth["boundary_episode"]) is not int
            or truth["boundary_episode"] < 0
        ):
            raise ValueError("unregistered private online schedule")
        before, after = (
            raw_spec(truth["before"], config),
            raw_spec(truth["after"], config),
        )
        valid, skipped, episode = True, 0, None
        for control in controls:
            if control["kind"] == "reset":
                observation = control["observation"]
                episode = observation["episode_id"]
                spec = after if episode >= truth["boundary_episode"] else before
                state = initial(spec)
                valid &= observation["agent_state"] == {"signal": config["outputs"][0]}
            else:
                record = control["transition"]
                if (
                    not record["intervention_compliance"]
                    or record["controller_failure"]
                    or not record["event"]["success"]
                ):
                    skipped += 1
                    continue
                state, output = raw_step(
                    spec, state, record["executed_action"]["target_id"]
                )
                valid &= record["event"]["details"]["signal"] == output
        old_ir, new_ir = (
            result["old"]["run"]["candidate_ir"],
            None if result["new"] is None else result["new"]["run"]["candidate_ir"],
        )
        actions, horizon = config["actions"], config["protocol"]["horizon"]
        changed = not same_behavior(before, after, actions, horizon)
        anomaly = result["anomaly"]
        detected = (
            anomaly is not None
            and changed
            and anomaly["episode_id"] >= truth["boundary_episode"]
        )
        old_correct = (
            None
            if old_ir is None
            else same_behavior(
                {"program": old_ir["parameters"]}, before, actions, horizon
            )
        )
        restored = (
            None
            if new_ir is None
            else same_behavior(
                {"program": new_ir["parameters"]}, after, actions, horizon
            )
        )
        return public | {
            "valid": bool(valid),
            "evaluation": {
                "successful_controls_match_private_reference": bool(valid),
                "known_failed_dispatches_excluded": skipped,
                "old_behavior_correct_at_H": old_correct,
                "new_behavior_correct_at_H": restored,
                "behavior_change_within_H": changed,
                "post_boundary_anomaly_detected": detected,
                "detection_delay_episodes": anomaly["episode_id"]
                - truth["boundary_episode"]
                if detected
                else None,
                "anomaly_without_in_H_change_or_before_boundary": anomaly is not None
                and not detected,
                "old_queries_after_private_boundary": sum(
                    c["kind"] == "reset"
                    and c["observation"]["episode_id"] >= truth["boundary_episode"]
                    for c in result["old"]["raw_controls"]
                ),
                "scope": "raw bounded issued behavior only; no mechanism/controller attribution, noisy change detection, statistical confidence or full H3/H4 claim",
            },
        }
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OSError,
        AttributeError,
    ) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}
