"""Raw public sampler replay; no learner, sampler class, reference pool or I/O."""

import hashlib
import itertools
import json
import random
from collections import deque

from minecraft_scientist.theory.finite_state import FiniteStateProgram


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def check_random_wp_plan(plan, costs):
    fields = {
        "protocol",
        "seed",
        "horizon",
        "minimal_size",
        "random_length",
        "max_samples",
        "requested_count",
        "queries",
        "samples",
        "total_cost",
        "sampler_visits",
        "hypothesis_digest",
        "hypothesis_program",
        "status",
        "limitations",
    }
    if (
        type(plan) is not dict
        or set(plan) != fields
        or plan["protocol"] != "random-wp-h-conditioned-v1"
    ):
        raise ValueError("independent random-Wp schema mismatch")
    for name, low, high in (
        ("seed", 0, None),
        ("horizon", 1, 8),
        ("minimal_size", 1, 32),
        ("random_length", 0, 32),
        ("max_samples", 1, 8192),
        ("requested_count", 1, 64),
    ):
        value = plan[name]
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError("independent random-Wp configuration mismatch")
    if (
        plan["max_samples"] < plan["requested_count"]
        or type(plan["samples"]) is not list
        or not 1 <= len(plan["samples"]) <= plan["max_samples"]
        or type(plan["queries"]) is not list
        or len(plan["queries"]) > plan["requested_count"]
        or type(plan["sampler_visits"]) is not int
        or not 1 <= plan["sampler_visits"] <= 200_000
    ):
        raise ValueError("independent random-Wp bounds mismatch")
    program = FiniteStateProgram.model_validate(plan["hypothesis_program"]).model_dump(
        mode="json"
    )
    digest = hashlib.sha256(canonical(program).encode()).hexdigest()
    if (
        canonical(program) != canonical(plan["hypothesis_program"])
        or digest != plan["hypothesis_digest"]
    ):
        raise ValueError("independent random-Wp hypothesis mismatch")
    actions = tuple(sorted(program["actions"]))
    if (
        type(costs) is not dict
        or set(costs) != set(actions)
        or any(type(c) is not int or c < 1 for c in costs.values())
    ):
        raise ValueError("independent random-Wp cost configuration mismatch")
    rows = {(r["source"], r["action"]): r for r in program["transitions"]}
    visits = 0

    def touch():
        nonlocal visits
        visits += 1
        if visits > 200_000:
            raise ValueError("independent random-Wp compute cap exceeded")

    def outputs(state, word):
        trace = []
        for action in word:
            row = rows[state, action]
            trace.append(row["output"])
            state = row["target"]
        return trace, state

    pending, seen, cover = deque([(program["initial_state"], [])]), set(), []
    while pending:
        state, word = pending.popleft()
        touch()
        if state in seen:
            continue
        seen.add(state)
        cover.append(word)
        pending.extend((rows[state, a]["target"], word + [a]) for a in actions)
    suffixes, local = set(), {s: set() for s in program["states"]}
    for first, second in itertools.combinations(program["states"], 2):
        pending, seen, witness = deque([(first, second, ())]), set(), None
        while pending and witness is None:
            left, right, word = pending.popleft()
            touch()
            if (left, right) in seen:
                continue
            seen.add((left, right))
            for a in actions:
                lrow, rrow = rows[left, a], rows[right, a]
                if lrow["output"] != rrow["output"]:
                    witness = word + (a,)
                    break
                pending.append((lrow["target"], rrow["target"], word + (a,)))
        if witness is not None:
            suffixes.add(witness)
            local[first].add(witness)
            local[second].add(witness)
    suffixes = sorted(suffixes, key=lambda w: (len(w), w))
    local = {s: sorted(words, key=lambda w: (len(w), w)) for s, words in local.items()}
    rng, expected_samples, expected_queries = random.Random(plan["seed"]), [], []
    for index in range(len(plan["samples"])):
        if len(expected_queries) == plan["requested_count"]:
            raise ValueError("independent random-Wp extra samples after frozen bound")
        touch()
        prefix, middle = rng.choice(cover), []
        while len(middle) < plan["minimal_size"] or rng.random() > 1 / (
            plan["random_length"] + 1
        ):
            touch()
            middle.append(rng.choice(actions))
        _, state = outputs(program["initial_state"], prefix + middle)
        mode = "global" if rng.getrandbits(1) else "local"
        options = suffixes if mode == "global" else local[state]
        suffix = list(rng.choice(options)) if options else []
        word = prefix + middle + suffix
        prediction, _ = outputs(program["initial_state"], word)
        item = {
            "prefix": prefix,
            "middle": middle,
            "suffix": suffix,
            "suffix_mode": mode,
            "suffix_state": state,
            "actions": word,
            "prediction": prediction,
        }
        admitted = len(word) <= plan["horizon"]
        expected_samples.append(
            item
            | {
                "sample_index": index,
                "admitted": admitted,
                "rejection_reason": None if admitted else "query_horizon",
            }
        )
        if admitted:
            expected_queries.append(
                item | {"sample_index": index, "cost": 1 + sum(costs[a] for a in word)}
            )
    ready = len(expected_queries) == plan["requested_count"]
    if not ready and len(expected_samples) != plan["max_samples"]:
        raise ValueError("independent random-Wp unexplained early sample stop")
    if (
        canonical(plan["samples"]) != canonical(expected_samples)
        or canonical(plan["queries"]) != canonical(expected_queries)
        or type(plan["total_cost"]) is not int
        or plan["total_cost"] != sum(q["cost"] for q in expected_queries)
        or plan["sampler_visits"] != visits
        or plan["status"] != ("planned" if ready else "insufficient_admissible_samples")
    ):
        raise ValueError(
            "independent random-Wp raw replay/prediction/admission/cost mismatch"
        )
