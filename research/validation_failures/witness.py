"""Bounded minimum-cost raw-table counterexample, for post-hoc audit only."""

import heapq

from research.structural_holdouts.raw_tables import trace, validate


def minimum_witness(candidate, reference, costs, *, horizon):
    left, right = validate(candidate), validate(reference)
    actions = sorted(candidate["actions"])
    if (
        set(actions) != set(reference["actions"])
        or set(costs) != set(actions)
        or any(type(v) is not int or v < 1 for v in costs.values())
        or type(horizon) is not int
        or not 0 <= horizon <= 8
    ):
        raise ValueError("invalid bounded raw witness protocol")
    pending = [
        (0, 0, (), False, candidate["initial_state"], reference["initial_state"])
    ]
    best = {}
    while pending:
        cost, depth, word, mismatch, ls, rs = heapq.heappop(pending)
        if mismatch:
            return {
                "actions": list(word),
                "atomic_cost": cost,
                "reset_inclusive_cost": cost + 1,
                "predicted": list(trace(candidate, word)),
                "reference": list(trace(reference, word)),
                "horizon": horizon,
                "scope": "post-hoc evaluator oracle; bounded declared raw deterministic reset behavior, NOT an Agent experiment planner",
            }
        key = ls, rs, depth
        if key in best and best[key] <= (cost, word):
            continue
        best[key] = cost, word
        if depth == horizon:
            continue
        # Mismatches also enter the cost-ordered queue: returning the first
        # outgoing differing edge would be wrong with nonuniform action costs.
        for action in actions:
            ln, lo = left[ls, action]
            rn, ro = right[rs, action]
            extended = (*word, action)
            price = cost + costs[action]
            if lo != ro:
                heapq.heappush(pending, (price, depth + 1, extended, True, ln, rn))
            else:
                heapq.heappush(pending, (price, depth + 1, extended, False, ln, rn))
    return None
