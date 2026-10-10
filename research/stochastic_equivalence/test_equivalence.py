"""Independent raw enumeration, exact closure and malicious-proof regressions."""

import itertools
import random
from copy import deepcopy
from fractions import Fraction

import pytest

from research.noisy_competition.belief import ModelBank
from research.noisy_competition.fixture import public_family, sensor_ir
from research.stochastic_ir.composition import compile_joint
from research.stochastic_ir.program import Mass, StochasticProgram

from .audit import verify_certificate
from .search import EquivalenceBound, compare


def pair():
    bank = ModelBank(public_family()[0])
    return bank.programs[0].to_ir(), bank.programs[12].to_ir()


def raw_probability(ir, actions, outputs, belief=None):
    spec = ir.parameters
    states = (
        {
            e["state"]: Fraction(
                e["probability"]["numerator"], e["probability"]["denominator"]
            )
            for e in spec["initial"]
        }
        if belief is None
        else dict(belief)
    )
    for action, output in zip(actions, outputs, strict=True):
        following = {}
        for row in spec["transitions"]:
            if row["action"] == action and row["output"] == output:
                mass = states.get(row["source"], Fraction(0)) * Fraction(
                    row["probability"]["numerator"], row["probability"]["denominator"]
                )
                following[row["target"]] = (
                    following.get(row["target"], Fraction(0)) + mass
                )
        states = following
    return sum(states.values(), Fraction(0))


def generated(seed):
    rng = random.Random(seed)
    rows = []
    for source in ("s0", "s1"):
        for action in ("a0", "a1"):
            outcomes = {}
            for _ in range(4):
                target, output = rng.choice(("s0", "s1")), rng.choice(("o0", "o1"))
                outcomes[target, output] = outcomes.get(
                    (target, output), Fraction(0)
                ) + Fraction(1, 4)
            rows.extend(
                {
                    "source": source,
                    "action": action,
                    "target": target,
                    "output": output,
                    "probability": Mass.of(probability),
                }
                for (target, output), probability in sorted(outcomes.items())
            )
    return StochasticProgram(
        states=("s0", "s1"),
        actions=("a0", "a1"),
        outputs=("o0", "o1"),
        initial=({"state": "s0", "probability": Mass.of(Fraction(1))},),
        transitions=tuple(rows),
    ).to_ir()


@pytest.mark.parametrize("seed", range(12))
def test_matches_full_independent_dimension_bounded_word_enumeration(seed):
    left, right = generated(seed), generated(seed + 20)
    brute_equal = all(
        raw_probability(left, a, o) == raw_probability(right, a, o)
        for length in range(1, 4)
        for a in itertools.product(left.parameters["actions"], repeat=length)
        for o in itertools.product(left.parameters["outputs"], repeat=length)
    )
    certificate = compare(left, right)
    assert (certificate["status"] == "equivalent") == brute_equal
    assert verify_certificate(left, right, certificate)["valid"]
    if certificate["witness"]:
        witness = certificate["witness"]
        assert raw_probability(
            left, witness["actions"], witness["outputs"]
        ) != raw_probability(right, witness["actions"], witness["outputs"])


def delayed(emits):
    states = tuple(f"delayed-{i}" for i in range(7))
    return StochasticProgram(
        states=states,
        actions=("advance",),
        outputs=("idle", "event"),
        initial=({"state": states[0], "probability": Mass.of(Fraction(1))},),
        transitions=tuple(
            {
                "source": s,
                "action": "advance",
                "target": states[min(i + 1, 6)],
                "output": "event" if emits and i == 6 else "idle",
                "probability": Mass.of(Fraction(1)),
            }
            for i, s in enumerate(states)
        ),
    ).to_ir()


def test_finds_difference_beyond_planners_horizon_four():
    left, right = delayed(True), delayed(False)
    assert all(
        raw_probability(left, ("advance",) * n, ("idle",) * n)
        == raw_probability(right, ("advance",) * n, ("idle",) * n)
        for n in range(1, 5)
    )
    certificate = compare(left, right)
    assert certificate["status"] == "distinguishable"
    assert len(certificate["witness"]["actions"]) == 7
    assert verify_certificate(left, right, certificate)["witness_length"] == 7


def test_uniform_sensor_hides_distinct_mechanisms_for_all_words():
    candidates, _, outputs = public_family()
    left = candidates[0]
    right = candidates[12]
    sensor = sensor_ir(outputs, Fraction(1, 2))
    lir = compile_joint(
        left.mechanism_ir, left.controller_ir, sensor, idle_output=outputs[0]
    )
    rir = compile_joint(
        right.mechanism_ir, right.controller_ir, sensor, idle_output=outputs[0]
    )
    certificate = compare(lir, rir)
    assert certificate["status"] == "equivalent"
    assert verify_certificate(lir, rir, certificate)["valid"]


def test_state_alias_permutation_preserves_behavior():
    left = generated(4)
    right = deepcopy(left)
    mapping = {"s0": "renamed-z", "s1": "renamed-a"}
    right.parameters["states"] = [
        mapping[s] for s in reversed(left.parameters["states"])
    ]
    for entry in right.parameters["initial"]:
        entry["state"] = mapping[entry["state"]]
    right.parameters["transitions"].reverse()
    for entry in right.parameters["transitions"]:
        entry["source"], entry["target"] = (
            mapping[entry["source"]],
            mapping[entry["target"]],
        )
    certificate = compare(left, right)
    assert certificate["status"] == "equivalent"
    assert verify_certificate(left, right, certificate)["valid"]


def test_current_belief_is_not_implicitly_reset():
    ir = delayed(True)
    states = ir.parameters["states"]
    lb, rb = {states[0]: Fraction(1)}, {states[-1]: Fraction(1)}
    certificate = compare(ir, ir, left_belief=lb, right_belief=rb)
    assert certificate["status"] == "distinguishable"
    assert verify_certificate(ir, ir, certificate, left_belief=lb, right_belief=rb)[
        "valid"
    ]
    with pytest.raises(ValueError, match="caller-pinned state beliefs"):
        verify_certificate(ir, ir, certificate)


def test_restricted_action_scope_does_not_prove_full_scope():
    left, right = pair()
    certificate = compare(left, right, allowed_actions=())
    assert certificate["status"] == "equivalent"
    assert verify_certificate(left, right, certificate, allowed_actions=())["valid"]
    with pytest.raises(ValueError, match="caller-pinned allowed action scope"):
        verify_certificate(left, right, certificate)


def test_audit_does_not_invoke_search_or_prediction_runtime(monkeypatch):
    from . import search

    left, _ = pair()
    certificate = compare(left, left)

    def forbidden(*args, **kwargs):
        raise AssertionError("not an independent raw certificate audit")

    monkeypatch.setattr(search, "compare", forbidden)
    monkeypatch.setattr(search.Span, "add", forbidden)
    for name in ("predict_step", "sequence_distribution", "filter", "filter_history"):
        monkeypatch.setattr(StochasticProgram, name, forbidden)
    assert verify_certificate(left, left, certificate)["valid"]


@pytest.mark.parametrize(
    "kind",
    [
        "vector",
        "trace",
        "omit",
        "duplicate",
        "scope",
        "hash",
        "dimension",
        "noncanonical",
        "bool",
    ],
)
def test_equivalence_proof_corruption_rejected(kind):
    left, _ = pair()
    certificate = compare(left, left)
    if kind == "vector":
        certificate["basis"][0]["vector"][0]["numerator"] += 1
    elif kind == "trace":
        certificate["basis"][0]["trace"] = [
            {
                "action": left.parameters["actions"][0],
                "output": left.parameters["outputs"][0],
            }
        ]
    elif kind == "omit":
        certificate["basis"] = certificate["basis"][:1]
    elif kind == "duplicate":
        certificate["basis"].append(deepcopy(certificate["basis"][0]))
    elif kind == "scope":
        certificate["allowed_actions"] = []
    elif kind == "hash":
        certificate["right_ir_sha256"] = "f" * 64
    elif kind == "dimension":
        certificate["dimension"] = 1
    elif kind == "noncanonical":
        certificate["basis"][0]["vector"][0] = {"numerator": 2, "denominator": 2}
    else:
        certificate["basis"][0]["vector"][0]["numerator"] = True
    with pytest.raises(ValueError):
        verify_certificate(left, left, certificate)


@pytest.mark.parametrize("kind", ["probability", "output", "length", "status", "bool"])
def test_inequivalence_witness_corruption_rejected(kind):
    left, right = pair()
    certificate = compare(left, right)
    witness = certificate["witness"]
    if kind == "probability":
        witness["left_probability"]["numerator"] += 1
    elif kind == "output":
        witness["outputs"][0] = "unknown"
    elif kind == "length":
        witness["actions"].pop()
    elif kind == "status":
        certificate["status"] = "equivalent"
    else:
        witness["left_probability"]["numerator"] = True
    with pytest.raises(ValueError):
        verify_certificate(left, right, certificate)


@pytest.mark.parametrize(
    "limits",
    [
        {"operation_limit": 1},
        {"bit_limit": 1},
        {"operation_limit": True},
        {"bit_limit": True},
    ],
)
def test_bounds_fail_not_approximate_or_partial(limits):
    # Fractional branches require >1 bit; deterministic unit coefficients do not.
    left, right = generated(0), generated(1)
    with pytest.raises((EquivalenceBound, ValueError)):
        compare(left, right, **limits)


def test_tiny_probability_difference_not_rounded_into_equivalence():
    left = StochasticProgram(
        states=("s",),
        actions=("a",),
        outputs=("o0", "o1"),
        initial=({"state": "s", "probability": Mass.of(Fraction(1))},),
        transitions=(
            {
                "source": "s",
                "action": "a",
                "target": "s",
                "output": "o0",
                "probability": Mass.of(Fraction(1)),
            },
        ),
    ).to_ir()
    right = deepcopy(left)
    right.parameters["transitions"] = [
        {
            "source": "s",
            "action": "a",
            "target": "s",
            "output": output,
            "probability": {"numerator": numerator, "denominator": 524288},
        }
        for output, numerator in (("o0", 524287), ("o1", 1))
    ]
    certificate = compare(left, right)
    assert certificate["status"] == "distinguishable"
    assert verify_certificate(left, right, certificate)["valid"]


@pytest.mark.parametrize(
    "belief", [{}, {"unknown": Fraction(1)}, {"s0": 1}, {"s0": Fraction(1, 2)}]
)
def test_invalid_beliefs_rejected(belief):
    ir = generated(0)
    with pytest.raises(ValueError):
        compare(ir, ir, left_belief=belief)
