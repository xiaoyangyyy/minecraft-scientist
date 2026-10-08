"""H-bounded Random-Wp adaptation with explicit, untruncated rejection records.

Sampling stages follow LearnLib 0.18 RandomWpMethodEQOracle: uniform state
cover, minimum plus geometric middle, fair coin for global/local suffix sets.
Canonical pair-BFS suffix sets and Python MT RNG differ from Java/AutomataLib.
H rejection conditions the published unbounded distribution; no MAT guarantee.
"""

import itertools
import random
from collections import deque

from reference_matching import fingerprint

from minecraft_scientist.discovery.active_automata import LearnerStop
from minecraft_scientist.theory.finite_state import FiniteStateProgram


class RandomWpSampler:
    def __init__(
        self,
        program,
        *,
        seed,
        minimal_size=3,
        random_length=3,
        operation_budget=200_000,
    ):
        if (
            type(seed) is not int
            or seed < 0
            or type(minimal_size) is not int
            or not 1 <= minimal_size <= 32
            or type(random_length) is not int
            or not 0 <= random_length <= 32
            or type(operation_budget) is not int
            or not 1 <= operation_budget <= 200_000
        ):
            raise ValueError("invalid bounded random-Wp configuration")
        # Revalidate declarative data only: warmed executors contain a cached
        # MappingProxyType table that cannot be deep-copied/pickled.
        self.program = FiniteStateProgram.model_validate(
            program.model_dump(mode="json")
        )
        self.actions = tuple(sorted(program.actions))
        self.table = self.program.table
        self.rng = random.Random(seed)
        self.seed, self.minimal_size, self.random_length = (
            seed,
            minimal_size,
            random_length,
        )
        self.operation_budget, self.visits = operation_budget, 0
        pending, seen, cover = deque([(program.initial_state, ())]), set(), []
        while pending:
            state, word = pending.popleft()
            self.touch()
            if state in seen:
                continue
            seen.add(state)
            cover.append(word)
            for action in self.actions:
                pending.append((self.table[state, action].target, word + (action,)))
        self.cover = tuple(cover)
        local, global_words = {s: set() for s in program.states}, set()
        for left, right in itertools.combinations(program.states, 2):
            word = self.separator(left, right)
            if word is not None:
                local[left].add(word)
                local[right].add(word)
                global_words.add(word)
        ordered = lambda words: tuple(sorted(words, key=lambda w: (len(w), w)))
        self.global_suffixes = ordered(global_words)
        self.local_suffixes = {s: ordered(words) for s, words in local.items()}

    def touch(self):
        if self.visits == self.operation_budget:
            raise LearnerStop("random_wp_operation_budget_exhausted")
        self.visits += 1

    def separator(self, left, right):
        pending, seen = deque([(left, right, ())]), set()
        while pending:
            left, right, word = pending.popleft()
            self.touch()
            if (left, right) in seen:
                continue
            seen.add((left, right))
            for action in self.actions:
                lrow, rrow = self.table[left, action], self.table[right, action]
                if lrow.output != rrow.output:
                    return word + (action,)
                pending.append((lrow.target, rrow.target, word + (action,)))
        return None

    def trace(self, word, state=None):
        state, outputs = self.program.initial_state if state is None else state, []
        for action in word:
            row = self.table[state, action]
            outputs.append(row.output)
            state = row.target
        return tuple(outputs), state

    def sample(self):
        self.touch()
        prefix = self.rng.choice(self.cover)
        middle = []
        while len(middle) < self.minimal_size or self.rng.random() > 1 / (
            self.random_length + 1
        ):
            self.touch()
            middle.append(self.rng.choice(self.actions))
        _, reached = self.trace(prefix + tuple(middle))
        mode = "global" if self.rng.getrandbits(1) else "local"
        choices = (
            self.global_suffixes if mode == "global" else self.local_suffixes[reached]
        )
        suffix = self.rng.choice(choices) if choices else ()
        word = prefix + tuple(middle) + suffix
        prediction, _ = self.trace(word)
        return {
            "prefix": list(prefix),
            "middle": middle,
            "suffix": list(suffix),
            "suffix_mode": mode,
            "suffix_state": reached,
            "actions": list(word),
            "prediction": list(prediction),
        }


def random_wp_plan(
    hypothesis,
    costs,
    *,
    horizon,
    count,
    seed,
    max_samples=512,
    minimal_size=3,
    random_length=3,
    operation_budget=200_000,
):
    """Freeze all accepted tests and rejected full samples before paid dispatch.

    Previously observed or duplicate inputs remain eligible: each accepted test is
    physically reset/re-executed. Rejection does NOT truncate words or hide samples.
    Fewer than count admissible tests yields unresolved, never a passing certificate.
    """
    if (
        type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(count) is not int
        or not 1 <= count <= 64
        or type(max_samples) is not int
        or not count <= max_samples <= 8192
        or type(costs) is not dict
        or set(costs) != set(hypothesis.program.actions)
        or any(type(v) is not int or v < 1 for v in costs.values())
    ):
        raise ValueError("invalid bounded random-Wp plan configuration")
    sampler = RandomWpSampler(
        hypothesis.program,
        seed=seed,
        minimal_size=minimal_size,
        random_length=random_length,
        operation_budget=operation_budget,
    )
    samples, queries = [], []
    for i in range(max_samples):
        item = sampler.sample()
        admitted = len(item["actions"]) <= horizon
        samples.append(
            item
            | {
                "sample_index": i,
                "admitted": admitted,
                "rejection_reason": None if admitted else "query_horizon",
            }
        )
        if admitted:
            queries.append(
                item
                | {
                    "sample_index": i,
                    "cost": 1 + sum(costs[a] for a in item["actions"]),
                }
            )
            if len(queries) == count:
                break
    return {
        "protocol": "random-wp-h-conditioned-v1",
        "seed": seed,
        "horizon": horizon,
        "minimal_size": minimal_size,
        "random_length": random_length,
        "max_samples": max_samples,
        "requested_count": count,
        "queries": queries,
        "samples": samples,
        "total_cost": sum(q["cost"] for q in queries),
        "sampler_visits": sampler.visits,
        "hypothesis_digest": fingerprint(hypothesis.program.model_dump(mode="json")),
        "hypothesis_program": hypothesis.program.model_dump(mode="json"),
        "status": "planned"
        if len(queries) == count
        else "insufficient_admissible_samples",
        "limitations": "H-conditioned sampling, canonical pair-BFS characterizing sets and Python MT RNG; not bit-for-bit LearnLib, exact MAT equivalence or an all-H certificate",
    }
