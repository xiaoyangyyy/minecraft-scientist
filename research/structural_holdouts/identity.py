"""Cost-colored rooted Mealy identities and leakage-connected data splits.

This independent raw-table utility does not use the learner, IR compiler or
private evaluator score. It belongs to benchmark construction, never the Agent.
Exact rooted reachable-graph identity and full deterministic reset-behavior
identity are different keys; neither proves unique physical causal structure.
"""

import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CanonicalizationBudgetExceeded(ValueError):
    """No exact identity was produced; do not replace it with a partial digest."""


@dataclass(frozen=True)
class Identity:
    mode: str
    sha256: str
    canonical: str
    enumerations: int


def _program(raw):
    if (
        type(raw) is not dict
        or set(raw)
        != {"language", "states", "actions", "outputs", "initial_state", "transitions"}
        or raw["language"] != "total-mealy-v1"
    ):
        raise ValueError("expected exact raw total-mealy-v1 contract")
    for key, maximum in (("states", 16), ("actions", 6), ("outputs", 3)):
        values = raw[key]
        if type(values) is not list or not 1 <= len(values) <= maximum:
            raise ValueError("identity alphabet outside declared bounds")
        if any(type(v) is not str or not 1 <= len(v) <= 80 for v in values):
            raise ValueError("identity requires bounded exact string labels")
        if len(set(values)) != len(values):
            raise ValueError("duplicate identity alphabet label")
    if (
        type(raw["initial_state"]) is not str
        or raw["initial_state"] not in raw["states"]
    ):
        raise ValueError("invalid identity initial state")
    if type(raw["transitions"]) is not list or len(raw["transitions"]) > 128:
        raise ValueError("invalid identity transition table")
    table = {}
    for row in raw["transitions"]:
        if type(row) is not dict or set(row) != {
            "source",
            "action",
            "target",
            "output",
        }:
            raise ValueError("invalid raw identity row")
        if any(type(v) is not str for v in row.values()):
            raise ValueError("non-string identity row")
        if (
            row["source"] not in raw["states"]
            or row["target"] not in raw["states"]
            or row["action"] not in raw["actions"]
            or row["output"] not in raw["outputs"]
        ):
            raise ValueError("unknown identity row label")
        key = row["source"], row["action"]
        if key in table:
            raise ValueError("duplicate identity transition")
        table[key] = row["target"], row["output"]
    if set(table) != set(itertools.product(raw["states"], raw["actions"])):
        raise ValueError("identity transition table is not total")
    return raw, table


def _reachable(raw, table):
    states, seen = [], {raw["initial_state"]}
    pending = deque([raw["initial_state"]])
    while pending:
        state = pending.popleft()
        states.append(state)
        for action in raw["actions"]:
            target, _ = table[state, action]
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return states


def _quotient(raw, table, mode):
    reachable = _reachable(raw, table)
    if mode == "structure":
        return raw["initial_state"], table
    colors = dict.fromkeys(reachable, 0)
    while True:
        signatures, refined = {}, {}
        for state in reachable:
            signature = (
                colors[state],
                tuple(
                    (table[state, action][1], colors[table[state, action][0]])
                    for action in raw["actions"]
                ),
            )
            refined[state] = signatures.setdefault(signature, len(signatures))
        if refined == colors:
            break
        colors = refined
    representatives = {}
    for state in reachable:
        representatives.setdefault(colors[state], state)
    quotient = {
        (color, action): (colors[table[state, action][0]], table[state, action][1])
        for color, state in representatives.items()
        for action in raw["actions"]
    }
    return colors[raw["initial_state"]], quotient


def _rooted(initial, table, actions, output_index):
    numbers, pending, rows = {initial: 0}, deque([initial]), []
    while pending:
        state = pending.popleft()
        row = []
        for action in actions:
            target, output = table[state, action]
            if target not in numbers:
                numbers[target] = len(numbers)
                pending.append(target)
            row.append([numbers[target], output_index[output]])
        rows.append(row)
    return rows


def fingerprint(
    programs, costs, *, reset_output, mode="structure", permutation_budget=4096
):
    """Exact identity of one machine or an ordered shared-alphabet machine tuple.

    State/action/output names and table ordering do not affect the key. Actions
    may permute only within equal-cost classes; the visible reset output remains
    anchored. Unreachable states are deliberately ignored. Behavior mode removes
    bisimilar states before graph canonicalization (all finite reset words, not H).
    Additional action affordances are not supported by this contract.
    """
    if (
        type(programs) is not tuple
        or not 1 <= len(programs) <= 4
        or mode not in ("structure", "behavior")
        or type(permutation_budget) is not int
        or not 1 <= permutation_budget <= 100_000
    ):
        raise ValueError("invalid exact identity protocol")
    programs = tuple(_program(json.loads(canonical_json(p))) for p in programs)
    actions, outputs = programs[0][0]["actions"], programs[0][0]["outputs"]
    if any(
        set(raw["actions"]) != set(actions) or set(raw["outputs"]) != set(outputs)
        for raw, _ in programs
    ):
        raise ValueError("machine tuple needs shared public alphabets")
    if (
        type(costs) is not dict
        or set(costs) != set(actions)
        or any(type(v) is not int or not 1 <= v <= 100 for v in costs.values())
    ):
        raise ValueError("invalid identity costs")
    if type(reset_output) is not str or reset_output not in outputs:
        raise ValueError("reset output must be a declared visible output")
    classes = [
        tuple(a for a in actions if costs[a] == cost)
        for cost in sorted(set(costs.values()))
    ]
    other_outputs = tuple(o for o in outputs if o != reset_output)
    enumerations = math.prod(
        math.factorial(len(group)) for group in classes
    ) * math.factorial(len(other_outputs))
    if enumerations > permutation_budget:
        raise CanonicalizationBudgetExceeded(
            "exact identity permutation budget exhausted"
        )
    machines = [_quotient(raw, table, mode) for raw, table in programs]
    best = None
    for groups in itertools.product(
        *(itertools.permutations(group) for group in classes)
    ):
        order = tuple(itertools.chain.from_iterable(groups))
        for tail in itertools.permutations(other_outputs):
            output_index = {o: i for i, o in enumerate((reset_output,) + tail)}
            encoded = canonical_json(
                {
                    "schema": "cost-colored-rooted-mealy-identity-v1",
                    "mode": mode,
                    "costs": [costs[a] for a in order],
                    "output_count": len(outputs),
                    "machines": [
                        _rooted(initial, table, order, output_index)
                        for initial, table in machines
                    ],
                }
            )
            if best is None or encoded < best:
                best = encoded
    return Identity(mode, hashlib.sha256(best.encode()).hexdigest(), best, enumerations)


def partition_records(records, *, salt, weights=(60, 20, 20)):
    """Group every connected leakage-key collision before hashing a split.

    Caller supplies evaluator-only keys for the declared leakage policy, e.g.
    target structure, target behavior and before/after pair identities. Connected
    groups may be imbalanced; never claim independent worlds from repeated IDs.
    Freeze the record set: adding a bridge can merge groups and change assignment.
    """
    if (
        type(records) is not tuple
        or not 1 <= len(records) <= 100_000
        or type(salt) is not str
        or not 1 <= len(salt) <= 256
        or type(weights) is not tuple
        or len(weights) != 3
        or any(type(w) is not int or w <= 0 for w in weights)
        or sum(weights) != 100
    ):
        raise ValueError("invalid group split protocol")
    identifiers, keys = [], []
    for record in records:
        if (
            type(record) is not dict
            or set(record) != {"id", "leakage_keys"}
            or type(record["id"]) is not str
            or not record["id"]
            or type(record["leakage_keys"]) is not tuple
            or not 1 <= len(record["leakage_keys"]) <= 64
            or any(
                type(k) is not str or not 1 <= len(k) <= 256
                for k in record["leakage_keys"]
            )
        ):
            raise ValueError("invalid split record")
        identifiers.append(record["id"])
        keys.append(record["leakage_keys"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate split record ID")
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, values in enumerate(keys):
        for key in values:
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    components = {}
    for index, values in enumerate(keys):
        components.setdefault(root(index), set()).update(values)
    assignments, group_ids = {}, {}
    for index, identifier in enumerate(identifiers):
        group = hashlib.sha256(
            canonical_json(sorted(components[root(index)])).encode()
        ).hexdigest()
        number = (
            int(hashlib.sha256(canonical_json([salt, group]).encode()).hexdigest(), 16)
            % 100
        )
        assignments[identifier] = (
            "development"
            if number < weights[0]
            else "tuning"
            if number < sum(weights[:2])
            else "evaluation"
        )
        group_ids[identifier] = group
    return {
        "schema": "leakage-connected-split-v1",
        "salt": salt,
        "weights": list(weights),
        "assignments": dict(sorted(assignments.items())),
        "groups": dict(sorted(group_ids.items())),
        "independent_group_count": len(components),
    }
