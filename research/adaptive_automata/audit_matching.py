"""Independent raw public plan checks; not yet a whole adaptive-run auditor.

No ReferencePool, model executor, learner matching or private file is used here.
This checks rule applicability, not the learner's completeness or sample gains.
"""

import hashlib
import itertools
import json
from collections import deque
from fractions import Fraction

from minecraft_scientist.evaluation.active_automata import _known_witness
from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def check_reference_plan(
    plan, snapshot, references, actions, costs, estimated_paid_cost
):
    """Raise before private I/O if shared replay could agree on wrong guidance."""
    fields = {
        "rule",
        "words",
        "epoch",
        "observation_digest",
        "reference_digest",
        "guidance",
        "basis",
    }
    arities = {"R": 5, "AMR": 4, "APS": 7, "AMS": 6}
    if (
        type(plan) is not dict
        or set(plan) != fields
        or type(snapshot) is not dict
        or set(snapshot) != {"epoch", "nodes"}
        or type(actions) is not tuple
        or not 1 <= len(actions) <= 8
        or any(type(a) is not str or not a for a in actions)
        or len(set(actions)) != len(actions)
        or type(costs) is not dict
        or set(costs) != set(actions)
        or any(type(c) is not int or c < 1 for c in costs.values())
        or type(references) not in (tuple, list)
        or not 1 <= len(references) <= 4
        or type(plan["rule"]) is not str
        or plan["rule"] not in arities
        or type(plan["epoch"]) is not str
        or type(snapshot["epoch"]) is not str
        or type(plan["guidance"]) is not list
        or len(plan["guidance"]) != arities[plan["rule"]]
        or type(plan["words"]) is not list
        or not 1 <= len(plan["words"]) <= 2
        or type(plan["basis"]) is not list
        or not 1 <= len(plan["basis"]) <= 16
        or type(snapshot["nodes"]) is not list
        or not 1 <= len(snapshot["nodes"]) <= 20_000
    ):
        raise ValueError("independent reference plan schema mismatch")

    def public_word(word, *, empty=True):
        return (
            type(word) is list
            and (empty or bool(word))
            and len(word) <= 8
            and all(type(a) is str and a in actions for a in word)
        )

    if (
        any(not public_word(w, empty=False) for w in plan["words"])
        or any(not public_word(w) for w in plan["basis"])
        or any(
            type(n) is not dict
            or set(n) != {"access", "incoming_output", "evidence_ids"}
            or not public_word(n["access"])
            or (
                n["incoming_output"] is not None
                and type(n["incoming_output"]) is not str
            )
            or type(n["evidence_ids"]) is not list
            or any(type(e) is not str for e in n["evidence_ids"])
            for n in snapshot["nodes"]
        )
    ):
        raise ValueError("independent public observation/word schema mismatch")
    if (
        plan["epoch"] != snapshot["epoch"]
        or plan["observation_digest"] != _digest(snapshot)
        or plan["reference_digest"] != _digest(references)
    ):
        raise ValueError("independent reference/observation scope mismatch")
    programs = [
        FiniteStateProgram.from_ir(TheoryIR.model_validate(r)).model_dump(mode="json")
        for r in references
    ]
    rows = [{(r["source"], r["action"]): r for r in p["transitions"]} for p in programs]
    keys = [(i, s) for i, p in enumerate(programs) for s in p["states"]]
    nodes = {tuple(n["access"]): n["incoming_output"] for n in snapshot["nodes"]}
    if (
        len(nodes) != len(snapshot["nodes"])
        or () not in nodes
        or nodes[()] is not None
        or any(p and (p[:-1] not in nodes or nodes[p] is None) for p in nodes)
    ):
        raise ValueError("independent observation tree schema mismatch")

    def valid_key(key):
        return (
            type(key) is list
            and len(key) == 2
            and type(key[0]) is int
            and type(key[1]) is str
            and tuple(key) in keys
        )

    key_positions = {"R": (2, 3), "AMR": (1, 2), "APS": (4, 5), "AMS": (3, 4)}
    for i, entry in enumerate(plan["guidance"]):
        if not (
            valid_key(entry) if i in key_positions[plan["rule"]] else public_word(entry)
        ):
            raise ValueError("independent reference guidance schema mismatch")
    if not plan["guidance"][-1]:
        raise ValueError("independent reference witness must be nonempty")
    basis = [tuple(p) for p in plan["basis"]]
    if (
        not basis
        or basis[0] != ()
        or len(set(basis)) != len(basis)
        or any(
            p not in nodes or any(p[:i] not in basis for i in range(len(p)))
            for p in basis
        )
    ):
        raise ValueError("independent reference basis is not an observed subtree")
    if any(
        _known_witness(nodes, l, r, actions) is None
        for l, r in itertools.combinations(basis, 2)
    ):
        raise ValueError("independent reference basis lacks actual apartness")
    frontier = {
        p + (a,)
        for p in basis
        for a in actions
        if p + (a,) in nodes and p + (a,) not in basis
    }

    def next_row(key, action):
        key = tuple(key)
        if len(key) != 2 or type(key[0]) is not int or key not in keys:
            raise ValueError("independent unknown reference state")
        row = rows[key[0]].get((key[1], action))
        return None if row is None else ((key[0], row["target"]), row["output"])

    def degree(prefix, key):
        # Separate derivation: enumerate public descendant edges, then follow
        # their parent words in the supplied raw reference table.
        agree, total = 0, 0
        for child, observed in nodes.items():
            if len(child) <= len(prefix) or child[: len(prefix)] != prefix:
                continue
            suffix = child[len(prefix) :]
            state, expected = tuple(key), None
            for action in suffix:
                step = next_row(state, action)
                if step is None:
                    break
                state, expected = step
            else:
                agree += expected == observed
                total += 1
        return Fraction(agree, total) if total else Fraction(1)

    def matches(prefix):
        values = {k: degree(prefix, k) for k in keys}
        best = max(values.values())
        return {k for k, v in values.items() if v == best}

    def known_reference_witness(prefix, key):
        pending = deque([(prefix, tuple(key), ())])
        while pending:
            path, state, word = pending.popleft()
            for action in sorted(actions):
                child = path + (action,)
                step = next_row(state, action)
                if child not in nodes or step is None:
                    continue
                if step[1] != nodes[child]:
                    return word + (action,)
                pending.append((child, step[0], word + (action,)))
        return None

    def separator(left, right):
        pending, seen = deque([(tuple(left), tuple(right), ())]), set()
        while pending:
            left, right, word = pending.popleft()
            if (left, right) in seen:
                continue
            seen.add((left, right))
            for action in sorted(actions):
                lstep, rstep = next_row(left, action), next_row(right, action)
                if (lstep is None) != (rstep is None) or (
                    lstep is not None and rstep is not None and lstep[1] != rstep[1]
                ):
                    return word + (action,)
                if lstep is not None:
                    pending.append((lstep[0], rstep[0], word + (action,)))
        return None

    def cover(index):
        pending, seen, reached = (
            deque([((index, programs[index]["initial_state"]), ())]),
            set(),
            {},
        )
        while pending:
            state, word = pending.popleft()
            if state in seen:
                continue
            seen.add(state)
            reached[word] = state
            for action in sorted(actions):
                step = next_row(state, action)
                if step is not None:
                    pending.append((step[0], word + (action,)))
        return reached

    def residual(prefix, suffix):
        if prefix + suffix not in nodes:
            return None
        return tuple(nodes[prefix + suffix[:i]] for i in range(1, len(suffix) + 1))

    guide, rule = plan["guidance"], plan["rule"]
    if rule == "R":
        child, other, left, right, word = map(tuple, guide)
        index = left[0]
        reached = cover(index)
        if (
            child[:-1] not in basis
            or other not in basis
            or child in basis
            or child not in reached
            or other not in reached
            or left != reached[child]
            or right != reached[other]
            or (
                child in nodes
                and _known_witness(nodes, child, other, actions) is not None
            )
            or separator(left, right) != word
            or (child + word in nodes and other + word in nodes)
        ):
            raise ValueError("independent rebuilding precondition mismatch")
        expected_words = [child + word, other + word]
    elif rule == "AMR":
        q, left, right, word = map(tuple, guide)
        if (
            q not in basis
            or left == right
            or left not in matches(q)
            or right not in matches(q)
            or separator(left, right) != word
            or q + word in nodes
        ):
            raise ValueError("independent approximate refinement precondition mismatch")
        expected_words = [q + word]
    elif rule == "APS":
        q, child, left, right, key, successor, word = map(tuple, guide)
        step = next_row(key, child[-1]) if child else None
        identifiers = {separator(successor, other) for other in keys} - {None}
        if (
            q not in basis
            or child not in frontier
            or child[:-1] != q
            or left == right
            or left not in basis
            or right not in basis
            or _known_witness(nodes, child, left, actions) is not None
            or _known_witness(nodes, child, right, actions) is not None
            or key not in matches(q)
            or step is None
            or step[0] != successor
            or word not in identifiers
            or residual(left, word) is None
            or residual(right, word) is None
            or residual(left, word) == residual(right, word)
        ):
            raise ValueError(
                "independent approximate prioritized-separation precondition mismatch"
            )
        expected_words = [child + word]
    elif rule == "AMS":
        q, child, other, key, successor, word = map(tuple, guide)
        step = next_row(key, child[-1]) if child else None
        if (
            q not in basis
            or child not in frontier
            or child[:-1] != q
            or other not in basis
            or key not in matches(q)
            or step is None
            or step[0] != successor
            or any(successor in matches(p) for p in basis)
            or _known_witness(nodes, child, other, actions) is not None
            or known_reference_witness(child, successor) is not None
            or known_reference_witness(other, successor) != word
        ):
            raise ValueError("independent approximate separation precondition mismatch")
        expected_words = [child + word]
    else:
        raise ValueError("independent unknown reference rule")
    expected_words = list(dict.fromkeys(expected_words))
    if plan["words"] != [list(w) for w in expected_words]:
        raise ValueError("independent reference query-word mismatch")
    virtual, amount = set(nodes), 0
    for word in expected_words:
        if word not in virtual:
            amount += 1 + sum(costs[a] for a in word)
            virtual.update(word[:i] for i in range(1, len(word) + 1))
    if type(estimated_paid_cost) is not int or amount != estimated_paid_cost:
        raise ValueError("independent reference query-cost mismatch")
