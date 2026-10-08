"""Basic L# adaptation; not a published Adaptive-L# implementation.

Observation trees, constructive apartness, strategic rules and binary conflict
processing follow Vaandrager et al. (TACAS 2022), arXiv:2107.05419v4 §3.
Finite real conformance batches replace exact EQ. No global convergence claim.
Models compile into the shared TheoryIR, with epoch-local public evidence only.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import deque
from dataclasses import dataclass

from minecraft_scientist.protocols import AgentWorldView, InterventionSpec
from minecraft_scientist.theory.finite_state import FiniteStateProgram, validate_costs


class LearnerStop(RuntimeError):
    """Explicit budget, scope, control or nonstationarity stop, not success."""


class ObservationTree:
    """Prefix identities and atomic same-epoch traces, never private state IDs."""

    def __init__(
        self, actions, outputs, *, epoch, node_limit=20_000, operation_budget=200_000
    ):
        for alphabet in (actions, outputs):
            if (
                type(alphabet) is not tuple
                or not 1 <= len(alphabet) <= 8
                or len(set(alphabet)) != len(alphabet)
                or any(type(x) is not str or not 1 <= len(x) <= 80 for x in alphabet)
            ):
                raise ValueError("bounded exact public alphabets required")
        if type(epoch) is not str or not 1 <= len(epoch) <= 80:
            raise ValueError("explicit evidence epoch required")
        if type(node_limit) is not int or not 1 <= node_limit <= 50_000:
            raise ValueError("invalid observation node limit")
        if type(operation_budget) is not int or operation_budget < 1:
            raise ValueError("invalid observation operation budget")
        self.actions, self.outputs, self.epoch = actions, outputs, epoch
        self.node_limit = node_limit
        self.operation_budget = operation_budget
        self._nodes = {(): None}
        self._evidence = {}
        self.visits = 0

    def _touch(self):
        if self.visits == self.operation_budget:
            raise LearnerStop("observation_operation_budget_exhausted")
        self.visits += 1

    def _word(self, word):
        if (
            type(word) is not tuple
            or len(word) > 64
            or any(type(a) is not str or a not in self.actions for a in word)
        ):
            raise ValueError("invalid public input sequence")

    def contains(self, word):
        self._word(word)
        return word in self._nodes

    def trace(self, word):
        self._word(word)
        if word not in self._nodes:
            return None
        return tuple(self._nodes[word[:i]] for i in range(1, len(word) + 1))

    def add_trace(self, word, outputs, *, epoch, evidence_id):
        self._word(word)
        if epoch != self.epoch:
            raise LearnerStop("evidence_epoch_mismatch")
        if (
            type(outputs) is not tuple
            or len(outputs) != len(word)
            or any(type(o) is not str or o not in self.outputs for o in outputs)
            or type(evidence_id) is not str
            or not 1 <= len(evidence_id) <= 100
        ):
            raise ValueError("invalid output trace or evidence identity")
        proposed = {word[:i]: outputs[i - 1] for i in range(1, len(word) + 1)}
        if any(
            prefix in self._nodes and self._nodes[prefix] != output
            for prefix, output in proposed.items()
        ):
            raise LearnerStop("nonstationary_or_noisy_trace")
        if len(set(self._nodes) | set(proposed)) > self.node_limit:
            raise LearnerStop("observation_capacity_exhausted")
        # Preflight the entire trace before changing any node or provenance.
        self._nodes.update(proposed)
        for prefix in proposed:
            self._evidence.setdefault(prefix, set()).add(evidence_id)

    def residual(self, prefix, suffix):
        self._word(prefix)
        self._word(suffix)
        self._touch()
        if prefix not in self._nodes:
            raise ValueError("unknown observation state")
        trace = self.trace(prefix + suffix)
        return None if trace is None else trace[len(prefix) :]

    def witness(self, left, right):
        """Shortest known suffix witnessing inequality, not a missing-edge guess."""
        if not self.contains(left) or not self.contains(right):
            raise ValueError("unknown observation states")
        pending = deque([(left, right, ())])
        while pending:
            lp, rp, word = pending.popleft()
            self._touch()
            for action in sorted(self.actions):
                lc, rc = lp + (action,), rp + (action,)
                if lc not in self._nodes or rc not in self._nodes:
                    continue
                if self._nodes[lc] != self._nodes[rc]:
                    return word + (action,)
                pending.append((lc, rc, word + (action,)))
        return None

    def frontier(self, basis):
        return tuple(
            sorted(
                {
                    prefix + (action,)
                    for prefix in basis
                    for action in self.actions
                    if prefix + (action,) in self._nodes
                    and prefix + (action,) not in basis
                },
                key=lambda p: (len(p), p),
            )
        )

    def compatible(self, prefix, basis):
        return tuple(state for state in basis if self.witness(prefix, state) is None)

    def norm(self, basis):
        frontier = self.frontier(basis)
        return (
            len(basis) * (len(basis) + 1) // 2
            + sum(
                prefix + (a,) in self._nodes for prefix in basis for a in self.actions
            )
            + sum(self.witness(q, s) is not None for q in frontier for s in basis)
        )

    def snapshot(self):
        return {
            "epoch": self.epoch,
            "nodes": [
                {
                    "access": list(p),
                    "incoming_output": self._nodes[p],
                    "evidence_ids": sorted(self._evidence.get(p, ())),
                }
                for p in sorted(self._nodes, key=lambda p: (len(p), p))
            ],
        }


@dataclass(frozen=True)
class TreeHypothesis:
    program: FiniteStateProgram
    basis: tuple[tuple[str, ...], ...]

    def access(self, state):
        return self.basis[self.program.states.index(state)]

    def reached_access(self, word):
        return self.access(self.program.execute(word)[1])

    def to_ir(self, horizon):
        ir = self.program.to_ir()
        ir.scope.update(
            intervention_horizon=horizon,
            certified_reset_only=True,
            claim="provisional finite conformance evidence; not all-query or unique causal identification",
        )
        return ir


def fold_hypothesis(tree, basis):
    """Adequate tree -> total program in the existing IR; no guessed outputs."""
    if not basis or basis[0] != () or len(basis) > 16 or len(set(basis)) != len(basis):
        raise ValueError("bounded unique prefix basis required")
    for prefix in basis:
        if not tree.contains(prefix) or any(
            prefix[:i] not in basis for i in range(len(prefix))
        ):
            raise ValueError("basis must be an observed subtree")
    if any(tree.witness(a, b) is None for a, b in itertools.combinations(basis, 2)):
        raise ValueError("basis states must be pairwise constructively apart")
    states = tuple(f"basis-{i}" for i in range(len(basis)))
    rows = []
    for prefix, state in zip(basis, states):
        for action in tree.actions:
            child = prefix + (action,)
            if not tree.contains(child):
                raise ValueError("basis is not complete")
            compatible = (child,) if child in basis else tree.compatible(child, basis)
            if len(compatible) != 1:
                raise ValueError("frontier is not identified")
            rows.append(
                {
                    "source": state,
                    "action": action,
                    "target": states[basis.index(compatible[0])],
                    "output": tree.trace(child)[-1],
                }
            )
    return TreeHypothesis(
        FiniteStateProgram(
            states=states,
            actions=tree.actions,
            outputs=tree.outputs,
            initial_state=states[0],
            transitions=tuple(rows),
        ),
        tuple(basis),
    )


def consistency_conflict(tree, hypothesis):
    """BFS constructive conflict in T x H, using only already acquired evidence."""
    pending = deque([((), hypothesis.program.initial_state)])
    while pending:
        prefix, state = pending.popleft()
        if tree.witness(prefix, hypothesis.access(state)) is not None:
            return prefix
        for action in sorted(tree.actions):
            child = prefix + (action,)
            if tree.contains(child):
                pending.append((child, hypothesis.program.table[state, action].target))
    return None


def process_conflict(tree, basis, hypothesis, conflict, query):
    """Binary shortening (L# Algorithm 3); queries are real and can exhaust H/B."""
    while True:
        reached = hypothesis.reached_access(conflict)
        witness = tree.witness(reached, conflict)
        if witness is None:
            raise ValueError("counterexample processing requires constructive conflict")
        frontier = tree.frontier(basis)
        if conflict in basis or conflict in frontier:
            return conflict
        access = next(
            (
                conflict[:i]
                for i in range(1, len(conflict) + 1)
                if conflict[:i] in frontier
            ),
            None,
        )
        if access is None:
            raise ValueError("conflict lacks an observed frontier prefix")
        middle = (len(access) + len(conflict)) // 2
        first, suffix = conflict[:middle], conflict[middle:]
        reached_first = hypothesis.reached_access(first)
        query(reached_first + suffix + witness, "counterexample", hypothesis)
        conflict = (
            first
            if tree.witness(reached_first, first) is not None
            else reached_first + suffix
        )


class OutputQueries:
    """Epoch-local cache plus charged primitive dispatch; no equivalence oracle."""

    def __init__(self, view, tree, costs, *, budget, horizon, prefix="aal-query-"):
        validate_costs(tree.actions, costs)
        if (
            type(budget) is not int
            or budget < 1
            or type(horizon) is not int
            or not 1 <= horizon <= 8
        ):
            raise ValueError("invalid real query budget/horizon")
        if type(prefix) is not str or not 1 <= len(prefix) <= 32:
            raise ValueError("invalid query namespace")
        self.view, self.tree, self.costs = view, tree, dict(costs)
        self.budget, self.horizon, self.prefix = budget, horizon, prefix
        self.spent, self.queries, self.cache_hits = 0, 0, 0
        self.events = []

    def __call__(self, word, reason, hypothesis=None, *, fresh=False):
        self.tree._word(word)
        if not 1 <= len(word) <= self.horizon:
            raise LearnerStop("query_horizon_exhausted")
        trace = self.tree.trace(word)
        if trace is not None and not fresh:
            self.cache_hits += 1
            self.events.append(
                {
                    "kind": "cache",
                    "epoch": self.tree.epoch,
                    "actions": list(word),
                    "reason": reason,
                }
            )
            return trace
        cost = 1 + sum(self.costs[a] for a in word)
        if cost > self.budget - self.spent:
            raise LearnerStop("experiment_budget_exhausted")
        qid = f"{self.prefix}{self.queries}"
        prediction = (
            [] if hypothesis is None else list(hypothesis.program.execute(word)[0])
        )
        intervention = InterventionSpec(
            intervention_id=qid,
            mechanism="finite_transducer",
            controlled_variables={"reset": True},
            target_variable="output_trace",
            fixed_context={
                "language": "total-mealy-v1",
                "evidence_epoch": self.tree.epoch,
            },
            predicted_outcomes={"hypothesis": prediction}
            if hypothesis is not None
            else {},
        )
        self.events.append(
            {
                "kind": "plan",
                "query_id": qid,
                "epoch": self.tree.epoch,
                "reason": reason,
                "actions": list(word),
                "prediction": prediction,
                "total_cost": cost,
                "intervention": intervention.model_dump(mode="json"),
            }
        )
        observed = self.view.begin_trial()
        self.spent += 1
        self.queries += 1
        self.events.append(
            {
                "kind": "reset",
                "query_id": qid,
                "cost": 1,
                "observation": observed.model_dump(mode="json"),
            }
        )
        available = {a.target_id: a for a in self.view.available_actions()}
        outputs = []
        for action in word:
            if action not in available:
                raise LearnerStop("planned_action_unavailable")
            record = self.view.step(available[action]).model_copy(
                update={"intended_intervention": intervention}
            )
            self.spent += self.costs[action]
            self.events.append(
                {
                    "kind": "step",
                    "query_id": qid,
                    "cost": self.costs[action],
                    "transition": record.model_dump(mode="json"),
                }
            )
            if (
                not record.intervention_compliance
                or record.controller_failure
                or not record.event.success
            ):
                self.events.append({"kind": "control_failure", "query_id": qid})
                raise LearnerStop("control_failure")
            outputs.append(record.event.details["signal"])
        self.tree.add_trace(
            word, tuple(outputs), epoch=self.tree.epoch, evidence_id=qid
        )
        self.events.append({"kind": "evidence", "query_id": qid, "outputs": outputs})
        return tuple(outputs)


def conformance_plan(tree, hypothesis, costs, *, horizon, count, seed):
    """Declared H-bounded long-trace coverage, not the paper's Hybrid-ADS tester.

    All words/predictions frozen before executing the batch. Previously known
    words are excluded. These tests cannot certify the entire finite language.
    """
    words = [
        word
        for depth in range(1, horizon + 1)
        for word in itertools.product(sorted(tree.actions), repeat=depth)
        if not tree.contains(word)
    ]
    chosen, covered = [], set()
    while len(chosen) < count and len(chosen) < len(words):
        word = min(
            (w for w in words if w not in chosen),
            key=lambda w: (
                -len(w),
                -len(set(w) - covered),
                hashlib.sha256(f"{seed}:{w!r}".encode()).hexdigest(),
                w,
            ),
        )
        chosen.append(word)
        covered.update(word)
    return {
        "candidate_ir": hypothesis.to_ir(horizon).model_dump(mode="json"),
        "queries": [
            {
                "actions": list(w),
                "prediction": list(hypothesis.program.execute(w)[0]),
                "cost": 1 + sum(costs[a] for a in w),
            }
            for w in chosen
        ],
        "total_cost": sum(1 + sum(costs[a] for a in w) for w in chosen),
        "scope": "finite fresh conformance batch, not global/H-complete equivalence",
    }


def learn_apartness(
    view: AgentWorldView,
    actions,
    outputs,
    costs,
    *,
    budget=240,
    horizon=5,
    epoch="current",
    seed=20261008,
    conformance_count=8,
    operation_budget=200_000,
):
    """Strategic basic L# adaptation, bounded and provisional, no reference model."""
    if (
        type(seed) is not int
        or seed < 0
        or type(conformance_count) is not int
        or not 1 <= conformance_count <= 64
        or type(operation_budget) is not int
        or operation_budget < 1
    ):
        raise ValueError("invalid learning/validation/operation protocol")
    tree = ObservationTree(
        actions, outputs, epoch=epoch, operation_budget=operation_budget
    )
    query = OutputQueries(view, tree, costs, budget=budget, horizon=horizon)
    if sum(len(actions) ** d for d in range(1, horizon + 1)) > 50_000:
        raise ValueError("conformance query universe exceeds computation cap")
    basis, rules, batches, candidate = [()], [], [], None
    status, reason = "unresolved", None
    try:
        while True:
            if tree.visits > operation_budget:
                raise LearnerStop("observation_operation_budget_exhausted")
            frontier = tree.frontier(basis)
            compatible = {q: tree.compatible(q, basis) for q in frontier}
            isolated = next((q for q in frontier if not compatible[q]), None)
            if isolated is not None:
                if len(basis) == 16:
                    raise LearnerStop("theory_state_capacity_exhausted")
                basis.append(isolated)
                rules.append({"rule": "promotion", "access": list(isolated)})
                continue
            missing = [
                q + (a,) for q in basis for a in actions if not tree.contains(q + (a,))
            ]
            if missing:
                word = min(missing, key=lambda w: (len(w), w))
                before = tree.norm(basis)
                query(word, "extension")
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError("extension failed constructive progress")
                rules.append(
                    {
                        "rule": "extension",
                        "actions": list(word),
                        "norm_before": before,
                        "norm_after": after,
                    }
                )
                continue
            ambiguous = next((q for q in frontier if len(compatible[q]) > 1), None)
            if ambiguous is not None:
                left, right = compatible[ambiguous][:2]
                witness = tree.witness(left, right)
                if witness is None:
                    raise ValueError("basis lost pairwise apartness")
                before = tree.norm(basis)
                query(ambiguous + witness, "separation")
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError("separation failed constructive progress")
                rules.append(
                    {
                        "rule": "separation",
                        "access": list(ambiguous),
                        "witness": list(witness),
                        "norm_before": before,
                        "norm_after": after,
                    }
                )
                continue
            candidate = fold_hypothesis(tree, basis)
            conflict = consistency_conflict(tree, candidate)
            if conflict is not None:
                before = tree.norm(basis)
                shortened = process_conflict(tree, basis, candidate, conflict, query)
                after = tree.norm(basis)
                if after <= before:
                    raise ValueError(
                        "consistency counterexample failed constructive progress"
                    )
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
            plan = conformance_plan(
                tree,
                candidate,
                query.costs,
                horizon=horizon,
                count=conformance_count,
                seed=seed + len(batches),
            )
            batch = {"plan": plan, "query_ids": [], "status": "unresolved"}
            batches.append(batch)
            if len(plan["queries"]) < conformance_count:
                raise LearnerStop("insufficient_fresh_conformance_queries")
            if plan["total_cost"] > budget - query.spent:
                raise LearnerStop("conformance_budget_exhausted")
            conflict = None
            before = tree.norm(basis)
            for item in plan["queries"]:
                word = tuple(item["actions"])
                outputs_observed = query(word, "conformance", candidate, fresh=True)
                batch["query_ids"].append(f"{query.prefix}{query.queries - 1}")
                if outputs_observed != tuple(item["prediction"]):
                    batch["status"] = "counterexample"
                    conflict = consistency_conflict(tree, candidate)
                    if conflict is None:
                        raise ValueError(
                            "nonconforming trace lacks constructive conflict"
                        )
                    shortened = process_conflict(
                        tree, basis, candidate, conflict, query
                    )
                    after = tree.norm(basis)
                    if after <= before:
                        raise ValueError(
                            "conformance counterexample failed constructive progress"
                        )
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
    except LearnerStop as stop:
        reason = str(stop)
    return {
        "schema": "apartness-learning-v1",
        "status": status,
        "stop_reason": reason,
        "budget": budget,
        "spent": query.spent,
        "paid_queries": query.queries,
        "cache_hits": query.cache_hits,
        "observation_visits": tree.visits,
        "observation_operations_are_complete_cpu_cost": False,
        "basis": [list(p) for p in basis],
        "observation_tree": tree.snapshot(),
        "events": query.events,
        "rules": rules,
        "conformance": batches,
        "candidate_ir": None
        if candidate is None or status != "provisional_conformance_passed"
        else candidate.to_ir(horizon).model_dump(mode="json"),
        "unique_causal_structure_claimed": False,
        "all_allowed_queries_certified": False,
        "reference_models_used": False,
        "limitations": "basic L# adaptation with finite real testing; not Adaptive L#, noise inference or complete H3 baseline matrix",
    }
