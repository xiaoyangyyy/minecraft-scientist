"""Isolated ordered adaptive rules with declared H-conditioned random-Wp tests.

Reuses the main observation tree, real output queries, fold, consistency and
binary processor. Does not claim the paper's exact MAT guarantees or complete
Adaptive L# baseline validation. Main source/test files stay frozen meanwhile.
"""

from __future__ import annotations

from random_wp import random_wp_plan
from reference_matching import ReferencePool

from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
    consistency_conflict,
    fold_hypothesis,
    process_conflict,
)


def learn_adaptive(
    view,
    actions,
    outputs,
    costs,
    references,
    *,
    budget=240,
    horizon=5,
    epoch="current",
    seed=20261008,
    conformance_count=8,
    operation_budget=200_000,
    reference_operation_budget=200_000,
):
    if (
        type(seed) is not int
        or seed < 0
        or type(conformance_count) is not int
        or not 1 <= conformance_count <= 64
    ):
        raise ValueError("invalid strategy seed/conformance count")
    tree = ObservationTree(
        actions, outputs, epoch=epoch, operation_budget=operation_budget
    )
    query = OutputQueries(
        view, tree, costs, budget=budget, horizon=horizon, prefix="adaptive-query-"
    )
    pool = ReferencePool(
        references, actions, operation_budget=reference_operation_budget
    )
    basis, rules, batches, candidate = [()], [], [], None
    status, reason = "unresolved", None

    def promote(prefix, rule):
        if len(basis) == 16:
            raise LearnerStop("theory_state_capacity_exhausted")
        basis.append(prefix)
        rules.append({"rule": rule, "access": list(prefix)})

    def apply(plan):
        before = len(tree.snapshot()["nodes"])
        receipt = pool.dispatch(tree, query, plan)
        after = len(tree.snapshot()["nodes"])
        if after <= before:
            raise ValueError("reference rule did not acquire new public evidence")
        rules.append(receipt | {"nodes_before": before, "nodes_after": after})

    try:
        # The rebuilding/PP phase runs once, never restarts after an EQ failure.
        while True:
            plan = pool.rebuilding(tree, basis)
            if plan is not None:
                apply(plan)
                continue
            promotion = pool.prioritized_promotion(tree, basis)
            if promotion is not None:
                promote(promotion, "PP")
                continue
            rules.append({"rule": "rebuilding_closed"})
            break
        while True:
            # Explicit FM 2024 section-6 priority, unlike the basic L# loop.
            missing = [
                q + (a,) for q in basis for a in actions if not tree.contains(q + (a,))
            ]
            if missing:
                word = min(missing, key=lambda w: (len(w), w))
                before = tree.norm(basis)
                query(word, "Ex")
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError("extension lacks constructive progress")
                rules.append(
                    {
                        "rule": "Ex",
                        "actions": list(word),
                        "norm_before": before,
                        "norm_after": after,
                    }
                )
                continue
            plan = pool.prioritized_separation(tree, basis)
            if plan is not None:
                before = tree.norm(basis)
                apply(plan)
                if tree.norm(basis) <= before:
                    raise ValueError("APS lacks actual observed apartness progress")
                continue
            frontier = tree.frontier(basis)
            compatible = {q: tree.compatible(q, basis) for q in frontier}
            ambiguous = next((q for q in frontier if len(compatible[q]) > 1), None)
            if ambiguous is not None:
                left, right = compatible[ambiguous][:2]
                witness = tree.witness(left, right)
                before = tree.norm(basis)
                query(ambiguous + witness, "S")
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError("fallback separation lacks constructive progress")
                rules.append(
                    {
                        "rule": "S",
                        "access": list(ambiguous),
                        "witness": list(witness),
                        "norm_before": before,
                        "norm_after": after,
                    }
                )
                continue
            isolated = next((q for q in frontier if not compatible[q]), None)
            if isolated is not None:
                promote(isolated, "P")
                continue
            plan = pool.refinement(tree, basis)
            if plan is not None:
                apply(plan)
                continue
            plan = pool.separation(tree, basis)
            if plan is not None:
                apply(plan)
                continue
            candidate = fold_hypothesis(tree, basis)
            conflict = consistency_conflict(tree, candidate)
            if conflict is not None:
                before = tree.norm(basis)
                shortened = process_conflict(tree, basis, candidate, conflict, query)
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError("consistency conflict lacks constructive progress")
                rules.append(
                    {
                        "rule": "consistency_counterexample",
                        "conflict": list(conflict),
                        "frontier": list(shortened),
                        "norm_before": before,
                        "norm_after": after,
                    }
                )
                continue
            plan = random_wp_plan(
                candidate,
                query.costs,
                horizon=horizon,
                count=conformance_count,
                seed=seed + len(batches),
            )
            batch = {"plan": plan, "query_ids": [], "status": "unresolved"}
            batches.append(batch)
            query.events.append({"kind": "conformance_plan", "plan": plan})
            if len(plan["queries"]) < conformance_count:
                raise LearnerStop("insufficient_admissible_random_wp_samples")
            if plan["total_cost"] > budget - query.spent:
                raise LearnerStop("conformance_budget_exhausted")
            conflict = None
            before = tree.norm(basis)
            for item in plan["queries"]:
                word = tuple(item["actions"])
                observed = query(word, "conformance", candidate, fresh=True)
                batch["query_ids"].append(f"{query.prefix}{query.queries - 1}")
                if observed != tuple(item["prediction"]):
                    batch["status"] = "counterexample"
                    conflict = consistency_conflict(tree, candidate)
                    if conflict is None:
                        raise ValueError("nonconforming trace lacks a conflict")
                    shortened = process_conflict(
                        tree, basis, candidate, conflict, query
                    )
                    after = tree.norm(basis)
                    if after <= before:
                        raise ValueError("counterexample lacks constructive progress")
                    rules.append(
                        {
                            "rule": "conformance_counterexample",
                            "conflict": list(conflict),
                            "frontier": list(shortened),
                            "norm_before": before,
                            "norm_after": after,
                        }
                    )
                    break
            if conflict is not None:
                continue
            batch["status"] = "passed"
            status = "provisional_conformance_passed"
            break
    except LearnerStop as error:
        reason = str(error)
    return {
        "schema": "isolated-adaptive-rules-v2",
        "status": status,
        "stop_reason": reason,
        "budget": budget,
        "spent": query.spent,
        "paid_queries": query.queries,
        "cache_hits": query.cache_hits,
        "observation_visits": tree.visits,
        "reference_visits": pool.visits,
        "conformance_sampler_visits": sum(b["plan"]["sampler_visits"] for b in batches),
        "operation_counters_are_complete_cpu_cost": False,
        "basis": [list(p) for p in basis],
        "observation_tree": tree.snapshot(),
        "reference_digest": pool.reference_digest,
        "events": query.events,
        "rules": rules,
        "conformance": batches,
        "candidate_ir": None
        if candidate is None or status != "provisional_conformance_passed"
        else candidate.to_ir(horizon).model_dump(mode="json"),
        "unique_causal_structure_claimed": False,
        "all_allowed_queries_certified": False,
        "reference_models_used": True,
        "limitations": "isolated ordered adaptive-rule adaptation; H-conditioned random-Wp with recorded rejected samples, canonical pair-BFS suffix sets and Python RNG; not bit-for-bit LearnLib, full baseline acceptance or exact MAT convergence",
    }
