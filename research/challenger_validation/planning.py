"""Bounded public fault-model coverage, with explicit search accounting.

No world, evaluator, private fixture or ground truth argument. This is a
mutation-testing baseline, not a new mechanism-discovery algorithm.
"""

import hashlib
import heapq
import itertools
import json
from copy import deepcopy
from fractions import Fraction

from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram, validate_costs


class ComputeStop(RuntimeError):
    pass


class Meter:
    def __init__(self, budget):
        if type(budget) is not int or not 1 <= budget <= 10_000_000:
            raise ValueError("invalid mutation compute budget")
        self.budget, self.spent = budget, 0

    def touch(self, count=1):
        if self.spent + count > self.budget:
            raise ComputeStop("mutation_operation_budget_exhausted")
        self.spent += count


def table(program):
    # Only compile a caller-supplied PUBLIC model through the shared language.
    parsed = FiniteStateProgram.model_validate(program)
    raw = parsed.model_dump(mode="json")
    return raw, {
        (row["source"], row["action"]): (row["target"], row["output"])
        for row in raw["transitions"]
    }


def execute(raw, rows, word, meter):
    state, outputs = raw["initial_state"], []
    for action in word:
        meter.touch()
        state, output = rows[state, action]
        outputs.append(output)
    return tuple(outputs)


def behavior_body(raw, rows, costs, meter):
    """Full deterministic behavior identity under this fixed public alphabet.

    Partition refinement + rooted quotient BFS, not a finite trace hash. Numeric
    action/output indices preserve the caller's declared alphabet order while
    removing latent state names. No private physical-factor inference is made.
    """
    states, actions = raw["states"], raw["actions"]
    output_index = {o: i for i, o in enumerate(raw["outputs"])}
    colors = dict.fromkeys(states, 0)
    while True:
        labels, refined = {}, {}
        for state in states:
            signature = []
            for action in actions:
                meter.touch()
                target, output = rows[state, action]
                signature.append((output_index[output], colors[target]))
            signature = tuple(signature)
            refined[state] = labels.setdefault(signature, len(labels))
        if refined == colors:
            break
        colors = refined
    representatives = {}
    for state in states:
        representatives.setdefault(colors[state], state)
    root = colors[raw["initial_state"]]
    ids, pending, graph = {root: 0}, [root], []
    for color in pending:
        row = []
        for action in actions:
            meter.touch()
            target, output = rows[representatives[color], action]
            target_color = colors[target]
            if target_color not in ids:
                ids[target_color] = len(ids)
                pending.append(target_color)
            row.append([output_index[output], ids[target_color]])
        graph.append(row)
    return json.dumps(
        {
            "costs": [costs[a] for a in actions],
            "outputs": len(raw["outputs"]),
            "graph": graph,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def shortest_unused(left, lrows, right, rrows, costs, used, horizon, meter):
    """Cost-ordered product search with an exclusion trie in the search state.

    Dropping the trie would wrongly prune a second path to the same state pair
    when the cheapest differing word had already been used.
    """
    actions = left["actions"]
    prefixes = {word[:n] for word in used for n in range(len(word) + 1)}
    queue = [(0, 0, (), left["initial_state"], right["initial_state"], False)]
    best = {}
    while queue:
        meter.touch()
        cost, depth, ranks, ls, rs, differs = heapq.heappop(queue)
        word = tuple(actions[i] for i in ranks)
        exclusion = word if word in prefixes else None
        key = ls, rs, depth, differs, exclusion
        if key in best and best[key] <= (cost, ranks):
            continue
        best[key] = cost, ranks
        if differs and word not in used:
            return word
        if depth == horizon:
            continue
        for rank, action in enumerate(actions):
            meter.touch()
            ln, lo = lrows[ls, action]
            rn, ro = rrows[rs, action]
            heapq.heappush(
                queue,
                (
                    cost + costs[action],
                    depth + 1,
                    (*ranks, rank),
                    ln,
                    rn,
                    differs or lo != ro,
                ),
            )
    return None


def make_plan(
    candidate,
    costs,
    used_words,
    evidence,
    *,
    budget,
    count=3,
    horizon=5,
    mutant_limit=512,
    operation_budget=200_000,
):
    if (
        type(budget) is not int
        or budget < 0
        or type(count) is not int
        or not 1 <= count <= 16
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(mutant_limit) is not int
        or not 1 <= mutant_limit <= 4096
    ):
        raise ValueError("invalid public mutation validation protocol")
    ir = TheoryIR.model_validate(candidate.model_dump())
    certified = ir.scope.get("intervention_horizon")
    if ir.scope.get("required_context") is not None or (
        certified is not None
        and (type(certified) is not int or not horizon <= certified <= 64)
    ):
        raise ValueError("unsupported candidate context/horizon")
    raw, rows = table(FiniteStateProgram.from_ir(ir).model_dump(mode="json"))
    validate_costs(tuple(raw["actions"]), costs)
    costs = dict(costs)
    if type(used_words) not in (tuple, list, set, frozenset) or any(
        type(w) is not tuple
        or not 1 <= len(w) <= horizon
        or any(a not in raw["actions"] for a in w)
        for w in used_words
    ):
        raise ValueError("invalid previous public words")
    used = set(used_words)
    if type(evidence) not in (tuple, list) or len(evidence) > 512:
        raise ValueError("invalid public evidence")
    detached, by_word, ids = [], {}, set()
    for item in evidence:
        if (
            type(item) is not dict
            or set(item) != {"query_id", "actions", "outputs"}
            or type(item["query_id"]) is not str
            or not 1 <= len(item["query_id"]) <= 100
            or item["query_id"] in ids
            or type(item["actions"]) is not list
            or type(item["outputs"]) is not list
        ):
            raise ValueError("invalid public evidence record")
        word, output = tuple(item["actions"]), tuple(item["outputs"])
        if (
            word not in used
            or len(word) != len(output)
            or any(o not in raw["outputs"] for o in output)
            or (word in by_word and by_word[word] != output)
        ):
            raise ValueError("conflicting/unregistered public evidence")
        by_word[word] = output
        ids.add(item["query_id"])
        detached.append(deepcopy(item))
    meter = Meter(operation_budget)
    result = {
        "schema": "public-mutation-validation-plan-v1",
        "candidate_ir": ir.model_dump(mode="json"),
        "costs": costs,
        "budget": budget,
        "count": count,
        "horizon": horizon,
        "mutant_limit": mutant_limit,
        "operation_budget": operation_budget,
        "used_words": [list(w) for w in sorted(used)],
        "evidence": detached,
        "fault_domain": "one existing-row target or output replacement in public issued-response model; no state insertion, unknown-controller guarantee or arbitrary DSL completeness",
        "classes": [],
        "queries": [],
        "total_cost": 0,
        "statistics": {
            "syntactic_mutants": 0,
            "candidate_equivalent_mutants": 0,
            "behavior_classes": 0,
            "evidence_eliminated_classes": 0,
            "surviving_classes": 0,
            "horizon_or_exclusion_unresolved_classes": 0,
        },
        "status": "unresolved",
        "stop_reason": None,
        "compute_operations": 0,
        "all_H_certified": False,
        "unique_causal_structure_claimed": False,
    }
    stats = result["statistics"]
    try:
        for word, expected in by_word.items():
            if execute(raw, rows, word, meter) != expected:
                result["stop_reason"] = "candidate_inconsistent_with_public_evidence"
                return result
        anticipated = len(raw["transitions"]) * (
            len(raw["states"]) - 1 + len(raw["outputs"]) - 1
        )
        if anticipated > mutant_limit:
            result["stop_reason"] = "complete_fault_domain_exceeds_mutant_limit"
            return result  # Do not turn a convenient prefix into a complete domain.
        original = behavior_body(raw, rows, costs, meter)
        classes = {}
        for index, original_row in enumerate(raw["transitions"]):
            for field, domain in (
                ("target", raw["states"]),
                ("output", raw["outputs"]),
            ):
                for value in domain:
                    if value == original_row[field]:
                        continue
                    meter.touch()
                    stats["syntactic_mutants"] += 1
                    mutant = deepcopy(raw)
                    mutant["transitions"][index][field] = value
                    mutated_rows = dict(rows)
                    changed = mutant["transitions"][index]
                    mutated_rows[changed["source"], changed["action"]] = (
                        changed["target"],
                        changed["output"],
                    )
                    body = behavior_body(mutant, mutated_rows, costs, meter)
                    if body == original:
                        stats["candidate_equivalent_mutants"] += 1
                        continue
                    if body in classes:
                        continue
                    classes[body] = mutant, mutated_rows
        stats["behavior_classes"] = len(classes)
        live, pool = {}, set()
        for body, (mutant, mutated_rows) in sorted(classes.items()):
            if any(
                execute(mutant, mutated_rows, word, meter) != expected
                for word, expected in by_word.items()
            ):
                stats["evidence_eliminated_classes"] += 1
                continue
            identity = hashlib.sha256(body.encode()).hexdigest()
            witness = shortest_unused(
                raw, rows, mutant, mutated_rows, costs, used, horizon, meter
            )
            result["classes"].append(
                {
                    "id": identity,
                    "canonical_behavior": body,
                    "program": mutant,
                    "minimum_unused_witness": None
                    if witness is None
                    else list(witness),
                }
            )
            live[identity] = mutant, mutated_rows
            if witness is None:
                stats["horizon_or_exclusion_unresolved_classes"] += 1
            else:
                pool.add(witness)
        stats["surviving_classes"] = len(live)
        remaining = set(live)
        rank = {a: i for i, a in enumerate(raw["actions"])}

        def price(word):
            return 1 + sum(costs[a] for a in word)

        coverage = {}
        for word in pool:
            expected = execute(raw, rows, word, meter)
            coverage[word] = {
                identity
                for identity, (mutant, mutated_rows) in live.items()
                if execute(mutant, mutated_rows, word, meter) != expected
            }
        while len(result["queries"]) < count:
            affordable = [
                w
                for w in pool
                if price(w) <= budget - result["total_cost"] and coverage[w] & remaining
            ]
            if not affordable:
                break
            word = min(
                affordable,
                key=lambda w: (
                    -Fraction(len(coverage[w] & remaining), price(w)),
                    price(w),
                    len(w),
                    tuple(rank[a] for a in w),
                ),
            )
            killed = coverage[word] & remaining
            result["queries"].append(
                {
                    "actions": list(word),
                    "prediction": list(execute(raw, rows, word, meter)),
                    "cost": price(word),
                    "challenged_classes": sorted(killed),
                    "selection": "behavior-class coverage per paid reset/action cost",
                }
            )
            result["total_cost"] += price(word)
            remaining -= killed
            pool.remove(word)
        if len(result["queries"]) < count:
            universe = sum(len(raw["actions"]) ** n for n in range(1, horizon + 1))
            if universe > 50_000:
                raise ComputeStop("mutation_filler_universe_exceeds_cap")
            selected = {tuple(q["actions"]) for q in result["queries"]}
            fillers = []
            for n in range(1, horizon + 1):
                for word in itertools.product(raw["actions"], repeat=n):
                    meter.touch()
                    if word not in used and word not in selected:
                        fillers.append(word)
            for word in sorted(
                fillers, key=lambda w: (price(w), len(w), tuple(rank[a] for a in w))
            ):
                if len(result["queries"]) == count:
                    break
                if result["total_cost"] + price(word) > budget:
                    continue
                result["queries"].append(
                    {
                        "actions": list(word),
                        "prediction": list(execute(raw, rows, word, meter)),
                        "cost": price(word),
                        "challenged_classes": [],
                        "selection": "declared inexpensive fresh filler; not fault-domain proof",
                    }
                )
                result["total_cost"] += price(word)
        result["remaining_unchallenged_classes"] = sorted(remaining)
        if len(result["queries"]) != count:
            result["stop_reason"] = "insufficient_affordable_unused_queries"
        else:
            result["status"] = "ready"
    except ComputeStop as error:
        result["stop_reason"] = str(error)
    finally:
        result["compute_operations"] = meter.spent
    return result
