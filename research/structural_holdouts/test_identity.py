"""Raw-table identity and partition invariants, separate from learning results."""

import itertools
import json
import random
from copy import deepcopy

import pytest

from .identity import (
    CanonicalizationBudgetExceeded,
    fingerprint,
    partition_records,
)


def machine(states=("empty", "loaded"), actions=("push", "read")):
    return {
        "language": "total-mealy-v1",
        "states": list(states),
        "actions": list(actions),
        "outputs": ["zero", "one"],
        "initial_state": states[0],
        "transitions": [
            {
                "source": state,
                "action": action,
                "target": states[-1] if action == actions[0] else state,
                "output": "one"
                if state == states[-1] and action == actions[-1]
                else "zero",
            }
            for state, action in itertools.product(states, actions)
        ],
    }


def identity(raw, costs=None, **kwargs):
    return fingerprint(
        (raw,), costs or dict.fromkeys(raw["actions"], 1), reset_output="zero", **kwargs
    )


def rename(raw, costs, seed):
    rng = random.Random(seed)
    maps = {
        key: {
            value: f"opaque-{key}-{seed}-{i}"
            for i, value in enumerate(rng.sample(raw[key], len(raw[key])))
        }
        for key in ("states", "actions", "outputs")
    }
    result = deepcopy(raw)
    for key, mapping in maps.items():
        result[key] = [mapping[v] for v in rng.sample(raw[key], len(raw[key]))]
    result["initial_state"] = maps["states"][raw["initial_state"]]
    for row in result["transitions"]:
        for key, alphabet in (
            ("source", "states"),
            ("target", "states"),
            ("action", "actions"),
            ("output", "outputs"),
        ):
            row[key] = maps[alphabet][row[key]]
    rng.shuffle(result["transitions"])
    return (
        result,
        {maps["actions"][a]: c for a, c in costs.items()},
        maps["outputs"]["zero"],
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("mode", ("structure", "behavior"))
def test_opaque_names_and_all_table_orders_do_not_change_identity(seed, mode):
    raw = machine()
    costs = {"push": 1, "read": 3}
    changed, changed_costs, reset = rename(raw, costs, seed)
    left = fingerprint((raw,), costs, reset_output="zero", mode=mode)
    right = fingerprint((changed,), changed_costs, reset_output=reset, mode=mode)
    assert left.sha256 == right.sha256
    assert left.canonical == right.canonical
    assert left.enumerations == right.enumerations == 1
    assert "opaque" not in right.canonical and "loaded" not in left.canonical


@pytest.mark.parametrize("mode", ("structure", "behavior"))
def test_action_cost_colors_are_not_discarded(mode):
    raw = machine(states=("empty",))
    assert (
        identity(raw, {"push": 1, "read": 3}, mode=mode).canonical
        != identity(raw, {"push": 3, "read": 1}, mode=mode).canonical
    )


def test_duplicate_hidden_states_are_structural_not_behavioral_difference():
    one = machine(states=("empty",))
    two = machine()
    for raw in (one, two):
        for row in raw["transitions"]:
            row["output"] = "zero"
    assert (
        identity(one, mode="structure").canonical
        != identity(two, mode="structure").canonical
    )
    assert (
        identity(one, mode="behavior").canonical
        == identity(two, mode="behavior").canonical
    )


@pytest.mark.parametrize("mode", ("structure", "behavior"))
def test_unreachable_junk_cannot_create_a_new_holdout_identity(mode):
    raw = machine()
    changed = deepcopy(raw)
    changed["states"].append("unreachable")
    for action in changed["actions"]:
        changed["transitions"].append(
            {
                "source": "unreachable",
                "action": action,
                "target": "unreachable",
                "output": "one",
            }
        )
    assert identity(raw, mode=mode).canonical == identity(changed, mode=mode).canonical


@pytest.mark.parametrize("mode", ("structure", "behavior"))
def test_revision_tuple_preserves_shared_action_roles(mode):
    before = machine(states=("empty",))
    after = deepcopy(before)
    for row in after["transitions"]:
        row["output"] = "one" if row["output"] == "zero" else "zero"
    assert identity(before, mode=mode).canonical == identity(after, mode=mode).canonical
    costs = dict.fromkeys(before["actions"], 1)
    assert (
        fingerprint((before, before), costs, reset_output="zero", mode=mode).canonical
        != fingerprint((before, after), costs, reset_output="zero", mode=mode).canonical
    )


def test_canonicalization_does_not_mutate_or_expose_input_aliases():
    raw = machine()
    snapshot = deepcopy(raw)
    result = identity(raw)
    assert raw == snapshot
    raw["transitions"][0]["target"] = "empty"
    assert result == identity(snapshot)
    with pytest.raises(AttributeError):
        result.sha256 = "fake"


def test_exact_permutation_cap_refuses_a_partial_identity():
    raw = machine(states=("empty",), actions=tuple(f"a-{i}" for i in range(6)))
    with pytest.raises(CanonicalizationBudgetExceeded):
        identity(raw, permutation_budget=100)
    assert identity(raw, permutation_budget=720).enumerations == 720


@pytest.mark.parametrize(
    "mutation",
    (
        lambda p: p.update(language="other"),
        lambda p: p.update(extra="hidden"),
        lambda p: p["transitions"].pop(),
        lambda p: p["transitions"].append(p["transitions"][0]),
        lambda p: p["transitions"][0].update(target="unknown"),
        lambda p: p["transitions"][0].update(output=True),
        lambda p: p["states"].append(p["states"][0]),
        lambda p: p.update(initial_state="unknown"),
    ),
)
def test_invalid_raw_tables_are_rejected(mutation):
    raw = machine()
    mutation(raw)
    with pytest.raises(ValueError):
        identity(raw)


@pytest.mark.parametrize(
    "options",
    ({"mode": "approximate"}, {"permutation_budget": True}, {"permutation_budget": 0}),
)
def test_invalid_identity_protocols_rejected(options):
    with pytest.raises(ValueError):
        identity(machine(), **options)


def test_cost_boolean_and_mismatched_alphabets_rejected():
    raw = machine()
    with pytest.raises(ValueError):
        identity(raw, {"push": True, "read": 1})
    changed, _, _ = rename(raw, {"push": 1, "read": 1}, 0)
    with pytest.raises(ValueError):
        fingerprint((raw, changed), {"push": 1, "read": 1}, reset_output="zero")


def records():
    return (
        {"id": "renaming-a", "leakage_keys": ("structure:A", "behavior:Q")},
        {"id": "renaming-b", "leakage_keys": ("structure:B", "behavior:Q")},
        {"id": "bridge", "leakage_keys": ("structure:B", "behavior:R")},
        {"id": "different", "leakage_keys": ("structure:C", "behavior:S")},
    )


def test_transitive_structure_behavior_collisions_never_cross_splits():
    result = partition_records(records(), salt="frozen-protocol-v1")
    assert result["independent_group_count"] == 2
    assert (
        len({result["groups"][key] for key in ("renaming-a", "renaming-b", "bridge")})
        == 1
    )
    assert (
        len(
            {
                result["assignments"][key]
                for key in ("renaming-a", "renaming-b", "bridge")
            }
        )
        == 1
    )
    assert result["groups"]["different"] != result["groups"]["bridge"]


def test_record_and_key_order_does_not_change_assignments():
    original = records()
    changed = tuple(
        {"id": r["id"], "leakage_keys": tuple(reversed(r["leakage_keys"]))}
        for r in reversed(original)
    )
    assert partition_records(original, salt="frozen") == partition_records(
        changed, salt="frozen"
    )


def test_record_names_do_not_change_group_assignment_or_group_identity():
    original = records()
    changed = tuple(
        {"id": f"opaque-{i}", "leakage_keys": r["leakage_keys"]}
        for i, r in enumerate(original)
    )
    left, right = (
        partition_records(original, salt="frozen"),
        partition_records(changed, salt="frozen"),
    )
    for i, r in enumerate(original):
        assert left["groups"][r["id"]] == right["groups"][f"opaque-{i}"]
        assert left["assignments"][r["id"]] == right["assignments"][f"opaque-{i}"]


@pytest.mark.parametrize(
    "bad",
    (
        (),
        ({"id": "a", "leakage_keys": ()},),
        ({"id": "a", "leakage_keys": (True,)},),
        ({"id": "a", "leakage_keys": ("x",)}, {"id": "a", "leakage_keys": ("y",)}),
        ({"id": "a", "leakage_keys": ("x",), "split": "evaluation"},),
    ),
)
def test_invalid_split_records_rejected(bad):
    with pytest.raises(ValueError):
        partition_records(bad, salt="frozen")


@pytest.mark.parametrize(
    "weights", ((True, 20, 79), (0, 50, 50), (60, 20, 19), [60, 20, 20])
)
def test_invalid_split_weights_rejected(weights):
    with pytest.raises(ValueError):
        partition_records(records(), salt="frozen", weights=weights)


def test_private_identity_metadata_remains_plain_reproducible_data():
    result = identity(machine(), mode="behavior")
    assert json.loads(result.canonical)["mode"] == "behavior"
    assert len(result.sha256) == 64
    assert partition_records(records(), salt="frozen") == partition_records(
        records(), salt="frozen"
    )


def test_behavior_partition_matches_independent_exhaustive_small_trace_reference():
    # Two machines have <=4 reachable paired states. Four equal one-action
    # outputs suffice for this tiny reference; no learner/minimizer is reused.
    reference, actual = {}, {}
    for index, cells in enumerate(
        itertools.product(itertools.product(range(2), range(2)), repeat=2)
    ):
        raw = machine(actions=("push",))
        for row, (target, output) in zip(raw["transitions"], cells, strict=True):
            row.update(target=raw["states"][target], output=raw["outputs"][output])
        state, trace = 0, []
        for _ in range(4):
            state, output = cells[state]
            trace.append(output)
        reference.setdefault(tuple(trace), set()).add(index)
        actual.setdefault(identity(raw, mode="behavior").canonical, set()).add(index)
    assert {frozenset(v) for v in reference.values()} == {
        frozenset(v) for v in actual.values()
    }


def test_structure_partition_matches_independent_brute_action_isomorphisms():
    # All 128 rooted, reachable two-state tables with 0/push->1 fixed. Initial
    # state fixes the state permutation; brute action swaps are the only choice.
    reference, actual = {}, {}
    for index, choices in enumerate(itertools.product(range(2), repeat=7)):
        targets, outputs = (1,) + choices[:3], choices[3:]
        raw = machine()
        for row, target, output in zip(
            raw["transitions"], targets, outputs, strict=True
        ):
            row.update(target=raw["states"][target], output=raw["outputs"][output])
        keys = []
        for order in ((0, 1), (1, 0)):
            keys.append(
                tuple(
                    (targets[2 * state + action], outputs[2 * state + action])
                    for state in range(2)
                    for action in order
                )
            )
        reference.setdefault(min(keys), set()).add(index)
        actual.setdefault(identity(raw, mode="structure").canonical, set()).add(index)
    assert {frozenset(v) for v in reference.values()} == {
        frozenset(v) for v in actual.values()
    }
