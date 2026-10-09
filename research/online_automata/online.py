"""Paid bootstrap -> bounded monitoring -> one behavior crisis -> real relearning.

No boundary or post-change model is supplied. Both methods see the same paid
old history, monitor identically, then either ignore or reuse its public IR.
An anomaly challenges issued-command behavior, NOT a unique causal factor.
"""

from matched_repair import (
    MATCHED_METHODS,
    check_policy,
    learn_matched_repair,
    repair_config,
)
from paid_transfer import METHODS, learn_phase, phase_config
from random_wp import random_wp_plan
from recorded_epoch import RecordedEpoch
from shared_bootstrap import (
    bootstrap_config,
    final_validation_ids,
    learn_shared_bootstrap,
    validated,
)

from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
    TreeHypothesis,
)
from minecraft_scientist.protocols import TheoryIR, TheoryStatus
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.theory.registry import TheoryRegistry


def online_relearn(
    world,
    actions,
    outputs,
    costs,
    *,
    method="adaptive",
    budget=400,
    old_budget=120,
    horizon=5,
    seed=20261008,
    conformance_count=8,
    monitor_rounds=4,
    monitor_count=8,
    operation_budget=200_000,
    reference_operation_budget=200_000,
    initial_family=None,
    repair_policy=None,
):
    if (
        type(method) is not str
        or method not in (MATCHED_METHODS if repair_policy is not None else METHODS)
        or any(
            type(v) is not int or v < 1
            for v in (budget, old_budget, operation_budget, reference_operation_budget)
        )
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(seed) is not int
        or seed < 0
        or any(
            type(v) is not int or not 1 <= v <= 64
            for v in (conformance_count, monitor_count)
        )
        or type(monitor_rounds) is not int
        or not 1 <= monitor_rounds <= 16
    ):
        raise ValueError("invalid bounded online relearning protocol")
    if repair_policy is not None:
        repair_policy = check_policy(repair_policy)
        if initial_family is None or horizon > 5:
            raise ValueError("matched repair requires a public family and H <= 5")
    initial_config = (
        None
        if initial_family is None
        else bootstrap_config(
            actions,
            outputs,
            costs,
            initial_family,
            budget=min(old_budget, budget),
            horizon=horizon,
            seed=seed,
        )
    )
    registry = TheoryRegistry()
    result = {
        "schema": "paid-online-relearning-v3"
        if repair_policy is not None
        else "paid-online-relearning-v2"
        if initial_config is not None
        else "paid-online-relearning-v1",
        "method": method,
        "budget": budget,
        "spent": 0,
        "old": None,
        "monitoring": [],
        "new": None,
        "public_old_theory": None,
        "anomaly": None,
        "theory_events": [],
        "validation_assignments": [],
        "status": "unresolved",
        "stop_reason": None,
        "mechanism_change_warranted": False,
        "mechanism_attribution_status": "unresolved",
        "unique_causal_structure_claimed": False,
        "all_allowed_queries_certified": False,
        "scope": "one bounded issued-behavior crisis with unannounced episode changes; trusted resets, deterministic queries, finite validation; no factor attribution, noisy continuous-world or full H3/H4 claim",
    }

    def publish(run, epoch, parent, *, accepted=True):
        final = final_validation_ids(run)
        acquired = [e["query_id"] for e in run["events"] if e["kind"] == "evidence"]
        discovery = tuple(f"{epoch}/{q}" for q in acquired if q not in final)
        validation = tuple(f"{epoch}/{q}" for q in final)
        if set(discovery) & set(validation):
            raise ValueError("online generation/validation evidence overlap")
        ir = TheoryIR.model_validate(
            run["candidate_ir"] if accepted else run["proposed_ir"]
        )
        ir.scope["evidence_epoch"] = epoch
        version = registry.propose(
            "issued-response",
            ir,
            evidence_ids=discovery,
            parent_version=parent,
            validation_score=0.0,
            confidence=0.0,
            created_by="shared-paid-scientific-lifecycle"
            if repair_policy is not None
            else "ordered-lsharp-paid-online-relearning",
        )
        result["theory_events"].append(version.model_dump(mode="json"))
        for status in (
            (TheoryStatus.PROVISIONAL, TheoryStatus.ACCEPTED)
            if accepted
            else (TheoryStatus.PROVISIONAL,)
        ):
            version = registry.transition("issued-response", version.version, status)
            result["theory_events"].append(version.model_dump(mode="json"))
        result["validation_assignments"].append(
            {
                "epoch": epoch,
                "version": version.version,
                "discovery_ids": list(discovery),
                "validation_ids": list(validation),
                "confidence_interpretation": "not estimated; stored zero is not a posterior",
            }
        )
        return version

    common = {
        "horizon": horizon,
        "seed": seed,
        "conformance_count": conformance_count,
        "operation_budget": operation_budget,
        "reference_operation_budget": reference_operation_budget,
    }

    def phase(epoch, references, limit):
        config = phase_config(
            actions, outputs, costs, references, budget=limit, epoch=epoch, **common
        )
        recorder = RecordedEpoch(world)
        run = learn_phase(recorder.view, config)
        result["spent"] += run["spent"]
        return {"config": config, "run": run, "raw_controls": recorder.controls}

    if initial_config is None:
        result["old"] = phase("old-epoch", (), min(old_budget, budget))
    else:
        recorder = RecordedEpoch(world)
        run = learn_shared_bootstrap(recorder.view, initial_config)
        result["spent"] += run["spent"]
        result["old"] = {
            "config": initial_config,
            "run": run,
            "raw_controls": recorder.controls,
        }
    old = result["old"]["run"]
    if not validated(old):
        result["stop_reason"] = "old_learning_unresolved"
        return result
    old_version = publish(old, "old-epoch", None)
    result["public_old_theory"] = old_version.model_dump(mode="json")
    hypothesis = TreeHypothesis(
        FiniteStateProgram.from_ir(old_version.ir),
        tuple(tuple(p) for p in old.get("basis", [])),
    )
    for index in range(monitor_rounds):
        try:
            plan = random_wp_plan(
                hypothesis,
                costs,
                horizon=horizon,
                count=monitor_count,
                seed=seed + 100_000 + index,
            )
        except LearnerStop as error:
            result["stop_reason"] = str(error)
            return result
        batch = {"plan": plan, "queries": [], "status": "unresolved"}
        result["monitoring"].append(batch)
        if len(plan["queries"]) < monitor_count:
            batch["status"] = "insufficient_samples"
            result["stop_reason"] = "monitor_insufficient_samples"
            return result
        if plan["total_cost"] > budget - result["spent"]:
            batch["status"] = "budget_exhausted"
            result["stop_reason"] = "monitor_budget_exhausted"
            return result
        for sample_index, item in enumerate(plan["queries"]):
            epoch = f"monitor-{index}-{sample_index}"
            recorder = RecordedEpoch(world)
            tree = ObservationTree(
                actions, outputs, epoch=epoch, operation_budget=operation_budget
            )
            query = OutputQueries(
                recorder.view,
                tree,
                costs,
                budget=budget - result["spent"],
                horizon=horizon,
                prefix=epoch + "-",
            )
            status, stopped = "matched", None
            try:
                observed = query(
                    tuple(item["actions"]), "monitor", hypothesis, fresh=True
                )
                if observed != tuple(item["prediction"]):
                    status = "counterexample"
            except LearnerStop as error:
                status, stopped = "stopped", str(error)
            result["spent"] += query.spent
            receipt = {
                "epoch": epoch,
                "budget": query.budget,
                "spent": query.spent,
                "status": status,
                "stop_reason": stopped,
                "events": query.events,
                "raw_controls": recorder.controls,
            }
            batch["queries"].append(receipt)
            if stopped is not None:
                batch["status"] = "stopped"
                result["stop_reason"] = "monitor_" + stopped
                return result
            if status == "counterexample":
                batch["status"] = "counterexample"
                reset = recorder.controls[0]["observation"]
                qid = f"{epoch}/{query.prefix}0"
                result["anomaly"] = {
                    "query_id": qid,
                    "episode_id": reset["episode_id"],
                    "actions": item["actions"],
                    "prediction": item["prediction"],
                    "observed": list(observed),
                    "claim": "issued behavior counterexample, not physical mechanism attribution",
                }
                challenged = registry.transition(
                    "issued-response",
                    1,
                    TheoryStatus.CHALLENGED,
                    counterexample_ids=(qid,),
                )
                result["theory_events"].append(challenged.model_dump(mode="json"))
                break
        if result["anomaly"] is not None:
            break
        batch["status"] = "passed"
    if result["anomaly"] is None:
        result["status"] = "bounded_monitoring_completed_no_anomaly"
        return result
    remaining = budget - result["spent"]
    if remaining < 1:
        result["stop_reason"] = "no_relearning_budget"
        return result
    references = (old_version.ir,) if method == "adaptive" else ()
    if repair_policy is None:
        result["new"] = phase("repair-epoch", references, remaining)
    else:
        config = repair_config(
            actions,
            outputs,
            costs,
            method=method,
            family=initial_config["family"],
            anchor_index=min(old["discovery"]["remaining_candidates"]),
            old_ir=old_version.ir.model_dump(mode="json"),
            anomaly=result["anomaly"],
            policy=repair_policy,
            protocol=common | {"budget": remaining, "epoch": "repair-epoch"},
        )
        recorder = RecordedEpoch(world)
        run = learn_matched_repair(recorder.view, config)
        result["spent"] += run["spent"]
        result["new"] = {
            "config": config,
            "run": run,
            "raw_controls": recorder.controls,
        }
    new = result["new"]["run"]
    if not validated(new):
        if repair_policy is not None and new["proposed_ir"] is not None:
            version = publish(new, "repair-epoch", 1, accepted=False)
            if new["validation_counterexample_ids"]:
                rejected = registry.transition(
                    "issued-response",
                    version.version,
                    TheoryStatus.REJECTED,
                    counterexample_ids=tuple(
                        f"repair-epoch/{q}"
                        for q in new["validation_counterexample_ids"]
                    ),
                )
                result["theory_events"].append(rejected.model_dump(mode="json"))
        result["stop_reason"] = "repair_learning_unresolved"
        return result
    publish(new, "repair-epoch", 1)
    superseded = registry.transition("issued-response", 1, TheoryStatus.SUPERSEDED)
    result["theory_events"].append(superseded.model_dump(mode="json"))
    result["status"] = "revision_finitely_validated"
    return result
