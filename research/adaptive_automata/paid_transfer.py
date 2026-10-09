"""Two explicit contexts, paid old discovery, fresh caches and matched tester.

Not an unannounced online mutation/causal attribution experiment. Both methods
learn the same old context and receive its public theory/history; the scratch
baseline deliberately ignores that prior when building the new model.
"""

from learner import learn_adaptive

from minecraft_scientist.protocols import TheoryIR, TheoryStatus
from minecraft_scientist.theory.registry import TheoryRegistry

METHODS = ("from_scratch", "adaptive")


def phase_config(
    actions,
    outputs,
    costs,
    references,
    *,
    budget,
    horizon,
    seed,
    epoch,
    conformance_count,
    operation_budget,
    reference_operation_budget,
):
    return {
        "schema": "adaptive-config-v1",
        "actions": list(actions),
        "outputs": list(outputs),
        "costs": dict(costs),
        "references": [r.model_dump(mode="json") for r in references],
        "protocol": {
            "budget": budget,
            "horizon": horizon,
            "epoch": epoch,
            "seed": seed,
            "conformance_count": conformance_count,
            "operation_budget": operation_budget,
            "reference_operation_budget": reference_operation_budget,
        },
    }


def learn_phase(view, config):
    return learn_adaptive(
        view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        tuple(TheoryIR.model_validate(r) for r in config["references"]),
        **config["protocol"],
    )


def transfer_contexts(
    old_view,
    new_view,
    actions,
    outputs,
    costs,
    *,
    method="adaptive",
    budget=360,
    old_budget=120,
    horizon=5,
    seed=20261008,
    conformance_count=8,
    operation_budget=200_000,
    reference_operation_budget=200_000,
):
    if (
        method not in METHODS
        or type(method) is not str
        or any(
            type(v) is not int or v < 1
            for v in (budget, old_budget, operation_budget, reference_operation_budget)
        )
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(seed) is not int
        or seed < 0
        or type(conformance_count) is not int
        or not 1 <= conformance_count <= 64
    ):
        raise ValueError("invalid paid-context-transfer protocol")
    registry = TheoryRegistry()
    result = {
        "schema": "paid-context-transfer-v1",
        "method": method,
        "budget": budget,
        "old_budget_cap": min(old_budget, budget),
        "spent": 0,
        "status": "unresolved",
        "stop_reason": None,
        "old": None,
        "new": None,
        "public_old_theory": None,
        "theory_events": [],
        "validation_assignments": [],
        "unique_causal_structure_claimed": False,
        "all_allowed_queries_certified": False,
        "unannounced_online_mutation_claimed": False,
        "scope": "two explicit contexts with stable anonymous alphabets, stationary trusted-reset behavior; finite validation, not causal structure, hidden online change or symbol-library transfer",
    }

    def publish(run, context, parent):
        final_ids = run["conformance"][-1]["query_ids"]
        acquired = [e["query_id"] for e in run["events"] if e["kind"] == "evidence"]
        discovery = tuple(f"{context}/{q}" for q in acquired if q not in final_ids)
        validation = tuple(f"{context}/{q}" for q in final_ids)
        if set(discovery) & set(validation):
            raise ValueError("generation and validation evidence overlap")
        ir = TheoryIR.model_validate(run["candidate_ir"])
        ir.scope["context_id"] = context
        # No calibrated posterior is estimated from a passing finite batch.
        # confidence=0 is explicitly uncalibrated, not a success probability.
        version = registry.propose(
            "issued-response",
            ir,
            evidence_ids=discovery,
            parent_version=parent,
            validation_score=0.0,
            confidence=0.0,
            created_by="ordered-lsharp-paid-context-transfer",
        )
        result["theory_events"].append(version.model_dump(mode="json"))
        for status in (TheoryStatus.PROVISIONAL, TheoryStatus.ACCEPTED):
            version = registry.transition("issued-response", version.version, status)
            result["theory_events"].append(version.model_dump(mode="json"))
        result["validation_assignments"].append(
            {
                "context": context,
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
    old_config = phase_config(
        actions,
        outputs,
        costs,
        (),
        budget=min(old_budget, budget),
        epoch="old-context",
        **common,
    )
    old_run = learn_phase(old_view, old_config)
    result["old"] = {"config": old_config, "run": old_run}
    result["spent"] += old_run["spent"]
    if old_run["status"] != "provisional_conformance_passed":
        result["stop_reason"] = "old_context_unresolved"
        return result
    old_theory = publish(old_run, "old-context", None)
    result["public_old_theory"] = old_theory.model_dump(mode="json")
    remaining = budget - result["spent"]
    if remaining < 1:
        result["stop_reason"] = "no_new_context_budget"
        return result
    # Both outer methods see the same old theory/history. The scratch method's
    # declared prior-ignore ablation passes no references to its new learner.
    references = (old_theory.ir,) if method == "adaptive" else ()
    new_config = phase_config(
        actions,
        outputs,
        costs,
        references,
        budget=remaining,
        epoch="new-context",
        **common,
    )
    new_run = learn_phase(new_view, new_config)
    result["new"] = {"config": new_config, "run": new_run}
    result["spent"] += new_run["spent"]
    if new_run["status"] != "provisional_conformance_passed":
        result["stop_reason"] = "new_context_unresolved"
        return result
    publish(new_run, "new-context", old_theory.version)
    result["status"] = "two_contexts_finitely_validated"
    # Old context remains valid, not challenged/superseded: the context switch
    # is explicit and is not evidence of an in-place theory crisis.
    return result
