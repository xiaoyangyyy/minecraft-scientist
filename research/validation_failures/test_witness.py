"""Independent exhaustive action-word reference for diagnostic witness cost."""

import itertools
import random

import pytest

from research.structural_holdouts.raw_tables import trace
from research.validation_failures.witness import minimum_witness


def program(rng):
    return {
        "language": "total-mealy-v1",
        "states": ["s0", "s1"],
        "actions": ["a", "b"],
        "outputs": ["x", "y"],
        "initial_state": "s0",
        "transitions": [
            {
                "source": s,
                "action": a,
                "target": rng.choice(["s0", "s1"]),
                "output": rng.choice(["x", "y"]),
            }
            for s, a in itertools.product(["s0", "s1"], ["a", "b"])
        ],
    }


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("horizon", (0, 1, 3))
def test_dijkstra_matches_exhaustive_cost_order(seed, horizon):
    rng = random.Random(seed)
    left, right = program(rng), program(rng)
    costs = {"a": 9, "b": 1}
    words = [
        word
        for n in range(horizon + 1)
        for word in itertools.product(["a", "b"], repeat=n)
        if trace(left, word) != trace(right, word)
    ]
    actual = minimum_witness(left, right, costs, horizon=horizon)
    if not words:
        assert actual is None
    else:
        expected = min(words, key=lambda w: (sum(costs[a] for a in w), len(w), w))
        assert actual["actions"] == list(expected)
        assert actual["reset_inclusive_cost"] == 1 + sum(costs[a] for a in expected)
        assert actual["predicted"] != actual["reference"]


@pytest.mark.parametrize("horizon", (-1, 9, True, 1.2))
def test_invalid_horizon_rejected(horizon):
    p = program(random.Random(0))
    with pytest.raises(ValueError):
        minimum_witness(p, p, {"a": 1, "b": 2}, horizon=horizon)


def test_two_cheap_steps_beat_one_expensive_discriminating_edge():
    left = program(random.Random(0))
    for row in left["transitions"]:
        row["output"] = "x"
        row["target"] = "s1" if row["action"] == "b" else "s0"
    right = {**left, "transitions": [dict(row) for row in left["transitions"]]}
    for row in right["transitions"]:
        if row["action"] == "a" or row["source"] == "s1":
            row["output"] = "y"
    actual = minimum_witness(left, right, {"a": 9, "b": 1}, horizon=2)
    assert actual["actions"] == ["b", "b"]
    assert actual["reset_inclusive_cost"] == 3


@pytest.mark.parametrize("seed", range(10))
def test_opaque_latent_names_do_not_change_minimum_cost(seed):
    rng = random.Random(seed)
    left, right = program(rng), program(rng)

    def rename(p):
        return {
            **p,
            "states": ["opaque-9", "opaque-1"],
            "initial_state": "opaque-9",
            "transitions": [
                {
                    **row,
                    "source": "opaque-9" if row["source"] == "s0" else "opaque-1",
                    "target": "opaque-9" if row["target"] == "s0" else "opaque-1",
                }
                for row in p["transitions"]
            ],
        }

    original = minimum_witness(left, right, {"a": 9, "b": 1}, horizon=3)
    changed = minimum_witness(rename(left), rename(right), {"a": 9, "b": 1}, horizon=3)
    assert original == changed
