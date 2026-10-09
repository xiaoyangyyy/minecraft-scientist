"""Isolated Adaptive L# rule adaptation over the existing public observation tree.

Kruger/Junges/Rot, FM 2024, arXiv:2406.19714v1: sections 4--7, Tables 1/3.
No old-model trace is inserted as new-world evidence. Matching is guidance, not
an oracle or causal certificate. This is not a complete validated baseline.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from minecraft_scientist.discovery.active_automata import LearnerStop
from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MatchDegree:
    agreeing_edges: int
    observed_edges: int

    @property
    def score(self):
        # Explicit no-evidence convention: every state ties; never confidence.
        return (
            Fraction(self.agreeing_edges, self.observed_edges)
            if self.observed_edges
            else Fraction(1)
        )


@dataclass(frozen=True)
class ReferencePlan:
    rule: str
    words: tuple[tuple[str, ...], ...]
    epoch: str
    observation_digest: str
    reference_digest: str
    guidance: tuple  # Reference state keys / observed prefixes, not private labels.
    basis: tuple[tuple[str, ...], ...]

    def record(self):
        return {
            "rule": self.rule,
            "words": [list(w) for w in self.words],
            "epoch": self.epoch,
            "observation_digest": self.observation_digest,
            "reference_digest": self.reference_digest,
            "guidance": json.loads(json.dumps(self.guidance)),
            "basis": [list(p) for p in self.basis],
        }


class ReferencePool:
    """Owned public models, exact edge-degree ties and finite product separators."""

    def __init__(self, references, actions, *, operation_budget=200_000):
        if (
            type(references) is not tuple
            or not 0 <= len(references) <= 4
            or any(not isinstance(ir, TheoryIR) for ir in references)
            or type(actions) is not tuple
            or not 1 <= len(actions) <= 8
            or len(set(actions)) != len(actions)
            or any(type(a) is not str or not 1 <= len(a) <= 80 for a in actions)
            or type(operation_budget) is not int
            or operation_budget < 1
        ):
            raise ValueError(
                "bounded public reference/alphabet/compute contract required"
            )
        owned = tuple(ir.model_copy(deep=True) for ir in references)
        self.programs = tuple(FiniteStateProgram.from_ir(ir) for ir in owned)
        self._reference_json = json.dumps(
            tuple(ir.model_dump(mode="json") for ir in owned),
            sort_keys=True,
            allow_nan=False,
        )
        self.reference_digest = hashlib.sha256(
            self._reference_json.encode()
        ).hexdigest()
        self.actions = actions
        self.keys = tuple((i, s) for i, p in enumerate(self.programs) for s in p.states)
        self.operation_budget, self.visits = operation_budget, 0
        self._separators, self._covers, self._identifiers = {}, {}, {}

    @property
    def reference_ir(self):
        return tuple(json.loads(self._reference_json))

    def _touch(self):
        if self.visits == self.operation_budget:
            raise LearnerStop("reference_operation_budget_exhausted")
        self.visits += 1

    def successor(self, key, action):
        program = self.programs[key[0]]
        if key[1] not in program.states:
            raise ValueError("unknown public reference state")
        if action not in program.actions:
            return None
        row = program.table[key[1], action]
        return (key[0], row.target), row.output

    def separator(self, left, right):
        """Total apartness across refs: defined-vs-undefined is distinguishable.

        Missing observation-tree edges are NOT total-apartness evidence; this
        helper compares two supplied reference programs only.
        """
        if left not in self.keys or right not in self.keys:
            raise ValueError("unknown reference state key")
        if (left, right) in self._separators:
            return self._separators[left, right]
        pending, seen = deque([(left, right, ())]), set()
        answer = None
        while pending:
            lkey, rkey, word = pending.popleft()
            self._touch()
            if (lkey, rkey) in seen:
                continue
            seen.add((lkey, rkey))
            for action in sorted(self.actions):
                lnext, rnext = (
                    self.successor(lkey, action),
                    self.successor(rkey, action),
                )
                if (lnext is None) != (rnext is None) or (
                    lnext is not None and rnext is not None and lnext[1] != rnext[1]
                ):
                    answer = word + (action,)
                    pending.clear()
                    break
                if lnext is not None:
                    pending.append((lnext[0], rnext[0], word + (action,)))
        self._separators[left, right] = self._separators[right, left] = answer
        return answer

    def state_cover(self, index):
        if index not in range(len(self.programs)):
            raise ValueError("unknown reference index")
        if index not in self._covers:
            pending = deque([((index, self.programs[index].initial_state), ())])
            reached, words = set(), []
            while pending:
                key, word = pending.popleft()
                self._touch()
                if key in reached:
                    continue
                reached.add(key)
                words.append(word)
                for action in sorted(self.actions):
                    next_step = self.successor(key, action)
                    if next_step is not None:
                        pending.append((next_step[0], word + (action,)))
            self._covers[index] = tuple(words)
        return self._covers[index]

    def reached(self, index, word):
        key = (index, self.programs[index].initial_state)
        for action in word:
            step = self.successor(key, action)
            if step is None:
                raise ValueError("reference access is undefined")
            key = step[0]
        return key

    def identifiers(self, key):
        if key not in self._identifiers:
            self._identifiers[key] = tuple(
                sorted(
                    {
                        w
                        for other in self.keys
                        if (w := self.separator(key, other)) is not None
                    },
                    key=lambda w: (len(w), w),
                )
            )
        return self._identifiers[key]

    def degree(self, tree, prefix, key):
        if not tree.contains(prefix) or key not in self.keys:
            raise ValueError("unknown observed/reference state")
        pending, agreeing, total = deque([(prefix, key)]), 0, 0
        while pending:
            access, reference = pending.popleft()
            self._touch()
            for action in sorted(self.actions):
                child = access + (action,)
                next_step = self.successor(reference, action)
                if next_step is None or not tree.contains(child):
                    continue
                total += 1
                agreeing += next_step[1] == tree.trace(child)[-1]
                # Still compare later edges after an earlier mismatch: mdeg is
                # an edge agreement rate, not the proportion of correct traces.
                pending.append((child, next_step[0]))
        return MatchDegree(agreeing, total)

    def matches(self, tree, prefix, *, approximate=True):
        if not tree.contains(prefix):
            raise ValueError("unknown observed state")
        degrees = {key: self.degree(tree, prefix, key) for key in self.keys}
        if not degrees:
            return ()
        if not approximate:
            return tuple(
                k for k, d in degrees.items() if d.agreeing_edges == d.observed_edges
            )
        best = max(d.score for d in degrees.values())
        return tuple(k for k, d in degrees.items() if d.score == best)

    def observed_separator(self, tree, prefix, key):
        """Known common-word inequality between public evidence and a reference."""
        pending = deque([(prefix, key, ())])
        while pending:
            access, reference, word = pending.popleft()
            self._touch()
            for action in sorted(self.actions):
                child = access + (action,)
                step = self.successor(reference, action)
                if step is None or not tree.contains(child):
                    continue
                if tree.trace(child)[-1] != step[1]:
                    return word + (action,)
                pending.append((child, step[0], word + (action,)))
        return None

    def _plan(self, tree, basis, rule, words, guidance):
        return ReferencePlan(
            rule,
            tuple(dict.fromkeys(words)),
            tree.epoch,
            fingerprint(tree.snapshot()),
            self.reference_digest,
            tuple(guidance),
            tuple(basis),
        )

    def rebuilding(self, tree, basis):
        for index in range(len(self.programs)):
            cover = self.state_cover(index)
            for q, other in itertools.product(basis, repeat=2):
                if q not in cover or other not in cover:
                    continue
                for action in sorted(self.actions):
                    child = q + (action,)
                    if child not in cover or child in basis:
                        continue
                    if tree.contains(child) and tree.witness(child, other) is not None:
                        continue
                    pair = self.reached(index, child), self.reached(index, other)
                    witness = self.separator(*pair)
                    if witness is None:
                        continue
                    words = child + witness, other + witness
                    if any(not tree.contains(w) for w in words):
                        return self._plan(
                            tree, basis, "R", words, (child, other, *pair, witness)
                        )
        return None

    def prioritized_promotion(self, tree, basis):
        cover = set().union(*(self.state_cover(i) for i in range(len(self.programs))))
        return next(
            (
                p
                for p in tree.frontier(basis)
                if p in cover and not tree.compatible(p, basis)
            ),
            None,
        )

    def refinement(self, tree, basis):
        for q in basis:
            matches = self.matches(tree, q)
            for left, right in itertools.combinations(matches, 2):
                witness = self.separator(left, right)
                if witness is not None and not tree.contains(q + witness):
                    return self._plan(
                        tree, basis, "AMR", (q + witness,), (q, left, right, witness)
                    )
        return None

    def prioritized_separation(self, tree, basis):
        for child in tree.frontier(basis):
            compatible = tree.compatible(child, basis)
            if len(compatible) < 2:
                continue
            q, action = child[:-1], child[-1]
            for key in self.matches(tree, q):
                successor = self.successor(key, action)
                if successor is None:
                    continue
                for witness in self.identifiers(successor[0]):
                    for left, right in itertools.combinations(compatible, 2):
                        ltrace, rtrace = (
                            tree.residual(left, witness),
                            tree.residual(right, witness),
                        )
                        if (
                            ltrace is not None
                            and rtrace is not None
                            and ltrace != rtrace
                        ):
                            return self._plan(
                                tree,
                                basis,
                                "APS",
                                (child + witness,),
                                (q, child, left, right, key, successor[0], witness),
                            )
        return None

    def separation(self, tree, basis):
        matches = {q: self.matches(tree, q) for q in basis}
        represented = set().union(*map(set, matches.values()))
        for child in tree.frontier(basis):
            q, action = child[:-1], child[-1]
            for key in matches[q]:
                successor = self.successor(key, action)
                if successor is None or successor[0] in represented:
                    continue
                next_key = successor[0]
                if self.observed_separator(tree, child, next_key) is not None:
                    continue
                for other in tree.compatible(child, basis):
                    witness = self.observed_separator(tree, other, next_key)
                    if witness is None:
                        raise ValueError(
                            "unmatched reference successor lacks an observed witness"
                        )
                    return self._plan(
                        tree,
                        basis,
                        "AMS",
                        (child + witness,),
                        (q, child, other, key, next_key, witness),
                    )
        return None

    def dispatch(self, tree, query, plan, hypothesis=None):
        if (
            plan.epoch != tree.epoch
            or query.tree is not tree
            or plan.observation_digest != fingerprint(tree.snapshot())
            or plan.reference_digest != self.reference_digest
        ):
            raise LearnerStop("stale_reference_plan_or_epoch")
        if any(not 1 <= len(w) <= query.horizon for w in plan.words):
            raise LearnerStop("reference_query_horizon_exhausted")
        virtual_known = {tuple(n["access"]) for n in tree.snapshot()["nodes"]}
        cost = 0
        for word in plan.words:
            if word not in virtual_known:
                cost += 1 + sum(query.costs[a] for a in word)
                virtual_known.update(word[:i] for i in range(1, len(word) + 1))
        if cost > query.budget - query.spent:
            raise LearnerStop("reference_query_budget_exhausted")
        # All reference/real hypotheses frozen before the first dispatch. A
        # failed second query retains first-query cost/evidence but no promotion.
        receipt = plan.record() | {
            "estimated_paid_cost": cost,
            "outputs": [],
            "status": "pending",
        }
        query.events.append(
            {
                "kind": "reference_plan",
                "plan": plan.record(),
                "estimated_paid_cost": cost,
            }
        )
        for word in plan.words:
            receipt["outputs"].append(list(query(word, plan.rule, hypothesis)))
        receipt["status"] = "executed"
        return receipt
