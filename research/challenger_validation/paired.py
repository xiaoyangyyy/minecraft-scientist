"""Prospective validation ablation on a paid, frozen PUBLIC learning prefix.

This re-executes every prefix action in a fresh capability world. Discovery
plans are inherited, not recomputed: this is a validation-policy ablation, not
a claim of complete end-to-end scientific compute comparison.
"""

from copy import deepcopy

from recorded_epoch import RecordedEpoch

from minecraft_scientist.protocols import PrimitiveAction, TheoryIR, TheoryStatus
from minecraft_scientist.theory.registry import TheoryRegistry
from minecraft_scientist.theory.shadow import validate_shadow
from research.structural_campaigns.runtime import check_public

from .execution import validate_challengers

POLICIES = ("public_coverage", "public_mutation")


def event_controls(events):
    return [e for e in events if e["kind"] in ("reset", "step")]


def prepare(config, historical_result):
    # All chronology/candidate/policy/evidence checked WITHOUT private truth.
    check_public(config, historical_result)
    raw = list(historical_result["old"]["raw_controls"])
    for batch in historical_result["monitoring"]:
        for query in batch["queries"]:
            raw.extend(query["raw_controls"])
    candidate, prior, eligible, epoch_base = None, [], False, None
    source_new = historical_result["new"]
    reason = "no_frozen_proposal"
    if source_new is not None:
        repaired = source_new["run"]
        candidate = repaired["proposed_ir"]
        random = repaired["random_validation"]
        eligible = (
            candidate is not None
            and random is not None
            and random["status"] == "passed"
        )
        if eligible:
            if repaired["base_run"] is not None:
                prior.extend(repaired["base_run"]["events"])
            elif repaired["structural"]["discovery"] is not None:
                prior.extend(repaired["structural"]["discovery"]["events"])
            for query in random["queries"]:
                prior.extend(query["events"])
            controls = source_new["raw_controls"][: len(event_controls(prior))]
            reason = None
        else:
            # No coverage tail can exist unless its preceding eight tests pass.
            prior = [e for e in repaired["events"]]
            controls = source_new["raw_controls"]
            reason = repaired["stop_reason"] or "prior_random_validation_not_passed"
        if controls:
            if controls[0]["kind"] != "reset":
                raise ValueError("repair epoch does not begin at a reset")
            epoch_base = controls[0]["observation"]["episode_id"]
        raw.extend(controls)
    plans = {e["query_id"]: e for e in prior if e["kind"] == "plan"}
    evidence = [
        {
            "query_id": e["query_id"],
            "actions": plans[e["query_id"]]["actions"],
            "outputs": e["outputs"],
        }
        for e in prior
        if e["kind"] == "evidence"
    ]
    used = sorted({tuple(e["actions"]) for e in plans.values()})
    spent = sum(
        1
        if e["kind"] == "reset"
        else config["costs"][e["transition"]["executed_action"]["target_id"]]
        for e in raw
    )
    if spent > config["protocol"]["budget"]:
        raise ValueError("inherited prefix already exceeds paid total budget")
    roots = []
    for event in historical_result["theory_events"]:
        if event["version"] != 1 or event["status"] == "superseded":
            break
        roots.append(event)
    last = (
        None
        if not raw
        else (
            raw[-1]["observation"]
            if raw[-1]["kind"] == "reset"
            else raw[-1]["transition"]["state_after"]
        )
    )
    return deepcopy(
        {
            "schema": "paid-public-validation-prefix-v1",
            "config": config,
            "prefix_controls": raw,
            "prefix_spent": spent,
            "prefix_last_observation": last,
            "candidate_ir": candidate,
            "eligible": eligible,
            "ineligible_reason": reason,
            "repair_epoch_base": epoch_base,
            "prior_events": prior,
            "evidence": evidence,
            "used_words": [list(w) for w in used],
            "root_theory_events": roots,
            "prior_validation_ids": []
            if not eligible
            else [
                e["query_id"]
                for q in source_new["run"]["random_validation"]["queries"]
                for e in q["events"]
                if e["kind"] == "evidence"
            ],
            "prior_counterexample_ids": []
            if source_new is None
            else source_new["run"]["validation_counterexample_ids"]
            if not eligible
            else [],
            "scope": "paid frozen public discovery/Wp prefix; original coverage outcome and private evaluator data excluded; inherited planning compute not reexecuted",
        }
    )


def root_registry(snapshot):
    registry, events = TheoryRegistry(), []
    for event in snapshot["root_theory_events"]:
        if event["status"] == "proposed":
            version = registry.propose(
                event["theory_id"],
                TheoryIR.model_validate(event["ir"]),
                evidence_ids=tuple(event["evidence_ids"]),
                validation_score=event["validation_score"],
                confidence=event["confidence"],
                created_by=event["created_by"],
                parent_version=event["parent_version"],
            )
        else:
            version = registry.transition(
                event["theory_id"],
                1,
                TheoryStatus(event["status"]),
                counterexample_ids=tuple(event["counterexample_ids"]),
            )
        if version.model_dump(mode="json") != event:
            raise ValueError("inherited root theory history mismatch")
        events.append(version.model_dump(mode="json"))
    return registry, events


def run_branch(view, snapshot, policy):
    if policy not in POLICIES:
        raise ValueError("unknown validation ablation policy")
    snapshot = deepcopy(snapshot)
    config = snapshot["config"]
    result = {
        "schema": "paid-validation-ablation-run-v1",
        "policy": policy,
        "budget": config["protocol"]["budget"],
        "prefix_spent": 0,
        "tail": None,
        "spent": 0,
        "raw_controls": [],
        "candidate_ir": None,
        "theory_events": [],
        "validation_assignments": [],
        "status": "unresolved",
        "stop_reason": snapshot["ineligible_reason"],
        "scope": "finite validation-policy ablation, not regenerated discovery/unique causal factors/complete scientific compute",
    }
    for control in snapshot["prefix_controls"]:
        if control["kind"] == "reset":
            observation = view.begin_trial()
            actual = {
                "kind": "reset",
                "observation": observation.model_dump(mode="json"),
            }
            result["spent"] += 1
        else:
            action = PrimitiveAction.model_validate(
                control["transition"]["executed_action"]
            )
            transition = view.step(action)
            actual = {"kind": "step", "transition": transition.model_dump(mode="json")}
            result["spent"] += config["costs"][action.target_id]
        result["raw_controls"].append(actual)
        if actual != control:
            result["prefix_spent"] = result["spent"]
            result["stop_reason"] = "real_prefix_replay_mismatch"
            return result
    result["prefix_spent"] = result["spent"]
    if result["prefix_spent"] != snapshot["prefix_spent"]:
        raise ValueError("paid prefix accounting mismatch")
    registry, result["theory_events"] = root_registry(snapshot)
    candidate = snapshot["candidate_ir"]
    if candidate is None:
        return result
    frozen = TheoryIR.model_validate(candidate)
    frozen.scope["evidence_epoch"] = "repair-epoch"
    validation_ids = set(snapshot["prior_validation_ids"])
    acquired = [
        e["query_id"] for e in snapshot["prior_events"] if e["kind"] == "evidence"
    ]
    discovery = ["repair-epoch/" + qid for qid in acquired if qid not in validation_ids]
    child = registry.propose(
        "issued-response",
        frozen,
        evidence_ids=tuple(discovery),
        parent_version=1,
        validation_score=0.0,
        confidence=0.0,
        created_by="paid-validation-policy-ablation",
    )
    result["theory_events"].append(child.model_dump(mode="json"))
    child = registry.transition(
        "issued-response", child.version, TheoryStatus.PROVISIONAL
    )
    result["theory_events"].append(child.model_dump(mode="json"))
    counterexamples = list(snapshot["prior_counterexample_ids"])
    if snapshot["eligible"]:
        recorder = RecordedEpoch(view)
        recorder.base = snapshot["repair_epoch_base"]
        words = {tuple(w) for w in snapshot["used_words"]}
        kwargs = {
            "budget": result["budget"] - result["spent"],
            "count": config["repair_policy"]["shadow_count"],
            "horizon": config["protocol"]["horizon"],
        }
        if policy == "public_coverage":
            tail = validate_shadow(
                recorder.view,
                TheoryIR.model_validate(candidate),
                config["costs"],
                words,
                **kwargs,
                seed=config["protocol"]["seed"] + 500_001,
                prefix="repair-shadow-",
            )
        else:
            tail = validate_challengers(
                recorder.view,
                TheoryIR.model_validate(candidate),
                config["costs"],
                words,
                snapshot["evidence"],
                **kwargs,
                mutant_limit=512,
                operation_budget=config["protocol"]["operation_budget"],
            )
        result["tail"] = tail
        result["spent"] += tail["spent"]
        result["raw_controls"].extend(recorder.controls)
        for trial in tail["runs"]:
            ids = [e["query_id"] for e in trial["events"] if e["kind"] == "evidence"]
            if trial["status"] == "model_mismatch":
                counterexamples.extend(ids)
            validation_ids.update(ids)
        result["stop_reason"] = tail["stop_reason"]
        if tail["status"] == "passed":
            accepted = registry.transition(
                "issued-response", child.version, TheoryStatus.ACCEPTED
            )
            superseded = registry.transition(
                "issued-response", 1, TheoryStatus.SUPERSEDED
            )
            result["theory_events"].extend(
                [accepted.model_dump(mode="json"), superseded.model_dump(mode="json")]
            )
            result["candidate_ir"] = deepcopy(candidate)
            result["status"] = "revision_finitely_validated"
    if counterexamples:
        rejected = registry.transition(
            "issued-response",
            child.version,
            TheoryStatus.REJECTED,
            counterexample_ids=tuple("repair-epoch/" + qid for qid in counterexamples),
        )
        result["theory_events"].append(rejected.model_dump(mode="json"))
        result["status"] = "candidate_rejected"
    result["validation_assignments"].append(
        {
            "version": child.version,
            "discovery_ids": discovery,
            "validation_ids": ["repair-epoch/" + qid for qid in sorted(validation_ids)],
            "confidence_interpretation": "zero is not a posterior; finite validation only",
        }
    )
    return result
