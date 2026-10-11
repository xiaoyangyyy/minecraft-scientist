"""Exact adaptive finite-horizon false-rejection mass and inference invariants."""

import itertools
from fractions import Fraction

import pytest

from research.stochastic_ir.program import Mass, StochasticProgram

from .monitor import AdequacyBound, FamilyAdequacyMonitor


def bernoulli(pa, pb=None):
    pb = pa if pb is None else pb
    rows = []
    for action, p in (("a", pa), ("b", pb)):
        for output, mass in (("0", 1 - p), ("1", p)):
            if mass:
                rows.append(
                    {
                        "source": "s",
                        "action": action,
                        "target": "s",
                        "output": output,
                        "probability": Mass.of(mass),
                    }
                )
    return StochasticProgram(
        states=("s",),
        actions=("a", "b"),
        outputs=("0", "1"),
        initial=({"state": "s", "probability": Mass.of(Fraction(1))},),
        transitions=tuple(rows),
    )


def test_reset_retains_cumulative_likelihood_and_rejection():
    monitor = FamilyAdequacyMonitor((bernoulli(Fraction(1, 8)),), ("null",))
    for _ in range(2):
        monitor.prepare("a")
        monitor.finish("1")
    before = monitor.snapshot()
    assert before["model_likelihoods"] == [{"numerator": 1, "denominator": 64}]
    assert before["e_value"] == {
        "kind": "finite",
        "value": {"numerator": 16, "denominator": 1},
    }
    assert not before["family_rejected"]
    monitor.reset_state()
    assert monitor.snapshot()["model_likelihoods"] == before["model_likelihoods"]
    monitor.prepare("a")
    after = monitor.finish("1")
    assert after["family_rejected"] and after["first_rejection_step"] == 3
    monitor.reset_state()
    assert monitor.snapshot()["first_rejection_step"] == 3


def test_maximum_null_likelihood_not_prior_mixture_denominator():
    monitor = FamilyAdequacyMonitor(
        (bernoulli(Fraction(1, 2)), bernoulli(Fraction(1, 128))), ("true", "other")
    )
    monitor.prepare("a")
    after = monitor.finish("1")
    assert after["e_value"]["value"] == {"numerator": 1, "denominator": 1}
    assert not after["family_rejected"]
    # A tiny prior on the true null would falsely reject at step one on an event
    # of probability 1/2, much above alpha=1/20, if a prior mixture were used.
    incorrect = Fraction(1, 2) / (
        Fraction(1, 100) * Fraction(1, 2) + Fraction(99, 100) * Fraction(1, 128)
    )
    assert incorrect > 20


def test_switching_maximum_model_and_nonmonotonic_e_process():
    monitor = FamilyAdequacyMonitor(
        (bernoulli(Fraction(1, 4)), bernoulli(Fraction(3, 4))),
        ("one", "two"),
        alpha=Fraction(1, 4),
    )
    values = []
    for output in ("0", "1", "0"):
        monitor.prepare("a")
        values.append(monitor.finish(output))
    assert values[0]["e_value"]["value"] == {"numerator": 2, "denominator": 3}
    assert values[1]["e_value"]["value"] == {"numerator": 4, "denominator": 3}
    assert values[2]["e_value"]["value"] == {"numerator": 8, "denominator": 9}
    assert values[2]["running_max_e_value"] == values[1]["e_value"]


def test_all_zero_null_likelihood_is_explicit_infinity_not_json_float():
    monitor = FamilyAdequacyMonitor((bernoulli(Fraction(0)),), ("null",))
    monitor.prepare("a")
    result = monitor.finish("1")
    assert result["e_value"] == {"kind": "infinite"}
    assert result["family_rejected"] and result["first_rejection_step"] == 1
    monitor.reset_state()
    assert monitor.prepare("a")["component_predictions"][0] == {
        "key": "null",
        "null_prefix_possible": False,
        "output_distribution": {},
    }
    assert monitor.finish("0")["model_likelihoods"] == [
        {"numerator": 0, "denominator": 1}
    ]


def test_exact_adaptive_path_false_rejection_mass_bounded_under_every_null():
    programs = (
        bernoulli(Fraction(1, 4), Fraction(3, 4)),
        bernoulli(Fraction(3, 4), Fraction(1, 4)),
    )
    risks = [Fraction(0), Fraction(0)]
    alpha = Fraction(1, 4)
    for word in itertools.product(("0", "1"), repeat=10):
        monitor = FamilyAdequacyMonitor(programs, ("one", "two"), alpha=alpha)
        probabilities = [Fraction(1), Fraction(1)]
        previous = "0"
        for index, output in enumerate(word):
            if index % 2 == 0:
                monitor.reset_state()
            action = "a" if previous == "0" else "b"
            for i, program in enumerate(programs):
                # Independent direct branch calculation under the SAME adaptive
                # policy; state is scalar, no runtime filter/likelihood helper.
                probabilities[i] *= sum(
                    row.probability.fraction()
                    for row in program.transitions
                    if row.action == action and row.output == output
                )
            monitor.prepare(action)
            monitor.finish(output)
            previous = output
        if monitor.snapshot()["family_rejected"]:
            for i, probability in enumerate(probabilities):
                risks[i] += probability
    assert all(0 < risk <= alpha for risk in risks)


def test_prepare_return_is_detached_and_phase_errors_are_transactional():
    monitor = FamilyAdequacyMonitor((bernoulli(Fraction(1, 8)),), ("null",))
    with pytest.raises(ValueError):
        monitor.finish("1")
    proposal = monitor.prepare("a")
    proposal["component_predictions"][0]["output_distribution"] = {}
    with pytest.raises(ValueError):
        monitor.prepare("a")
    with pytest.raises(ValueError):
        monitor.reset_state()
    assert monitor.finish("1")["model_likelihoods"] == [
        {"numerator": 1, "denominator": 8}
    ]


def test_finish_bit_exhaustion_never_commits_partial_evidence():
    monitor = FamilyAdequacyMonitor((bernoulli(Fraction(0)),), ("null",), bit_limit=2)
    monitor.prepare("a")
    monitor.finish("0")
    monitor.prepare("a")
    before = monitor.snapshot()
    with pytest.raises(AdequacyBound, match="coefficient"):
        monitor.finish("0")
    assert monitor.snapshot() == before


def test_observation_bound_cannot_be_restarted_by_reset():
    monitor = FamilyAdequacyMonitor(
        (bernoulli(Fraction(1, 2)),), ("null",), max_steps=1
    )
    monitor.prepare("a")
    monitor.finish("0")
    monitor.reset_state()
    with pytest.raises(AdequacyBound, match="observation"):
        monitor.prepare("a")


@pytest.mark.parametrize(
    "alpha",
    [True, 0.05, Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 1_000_001)],
)
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        FamilyAdequacyMonitor((bernoulli(Fraction(1, 2)),), ("null",), alpha=alpha)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_steps": True},
        {"max_steps": 257},
        {"bit_limit": True},
        {"bit_limit": 12001},
    ],
)
def test_invalid_resource_contract_rejected(limits):
    with pytest.raises(ValueError):
        FamilyAdequacyMonitor((bernoulli(Fraction(1, 2)),), ("null",), **limits)
