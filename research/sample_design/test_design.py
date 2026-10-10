"""Independent arithmetic oracles, not new Agent/world experimental evidence."""

import itertools
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from research.sample_design import binomial as design
from research.sample_design.report import sensitivity_report


def enumeration_tail(n, k, p):
    # Independent sample-space enumeration: no binomial coefficient, recurrence,
    # power-planner internals or external statistical package.
    return sum(
        (
            p ** sum(bits) * (1 - p) ** (n - sum(bits))
            for bits in itertools.product((0, 1), repeat=n)
            if sum(bits) <= k
        ),
        Fraction(0),
    )


def polynomial_tail(n, k, p):
    # Separate closed-form oracle for larger n: no production recurrence.
    return sum(
        (math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(min(n, k) + 1)),
        Fraction(0),
    )


@pytest.mark.parametrize(
    "p", [Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(4, 5), Fraction(1)]
)
def test_exact_tail_matches_sample_space(p):
    for n in range(1, 8):
        for k in range(min(n + 1, 4)):
            assert design.tail_probability(n, k, p) == enumeration_tail(n, k, p)


@pytest.mark.parametrize("family,n", [(1, 59), (8, 99)])
def test_known_zero_failure_design(family, n):
    result = design.fixed_allowance_plan(
        risk=Fraction(1, 20), alpha=Fraction(1, 20), family_size=family
    )
    assert result["status"] == "feasible"
    assert result["design"]["groups"] == n
    assert design.tail_probability(n, 0, Fraction(1, 20)) <= Fraction(1, 20 * family)
    assert design.tail_probability(n - 1, 0, Fraction(1, 20)) > Fraction(1, 20 * family)


@pytest.mark.parametrize("risk", [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)])
@pytest.mark.parametrize("family", [1, 2])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_fixed_allowance_minimality_independent_brute_force(risk, family, k):
    alpha = Fraction(1, 4)
    expected = next(
        (n for n in range(1, 61) if polynomial_tail(n, k, risk) <= alpha / family),
        None,
    )
    result = design.fixed_allowance_plan(
        risk=risk, alpha=alpha, family_size=family, allowed_failures=k, max_groups=60
    )
    assert result["design"]["groups"] == expected


@pytest.mark.parametrize(
    "risk,good,power",
    [
        (Fraction(1, 2), Fraction(0), Fraction(4, 5)),
        (Fraction(1, 3), Fraction(1, 12), Fraction(4, 5)),
        (Fraction(1, 2), Fraction(1, 4), Fraction(9, 10)),
        (Fraction(1, 2), Fraction(49, 100), Fraction(9, 10)),
    ],
)
@pytest.mark.parametrize("family", [1, 3])
def test_powered_minimum_matches_exhaustive_n_and_k(risk, good, power, family):
    alpha = Fraction(1, 4)
    feasible = [
        (n, k)
        for n in range(1, 61)
        for k in range(5)
        if polynomial_tail(n, k, risk) <= alpha / family
        and polynomial_tail(n, k, good) >= power
    ]
    expected = min(feasible) if feasible else None
    result = design.power_constrained_plan(
        risk=risk,
        good=good,
        power=power,
        alpha=alpha,
        family_size=family,
        max_groups=60,
        max_allowed_failures=4,
    )
    if expected is None:
        assert result["status"] == "no_design_within_declared_limits"
        assert result["design"] is None
    else:
        assert result["status"] == "feasible"
        assert (
            result["design"]["groups"],
            result["design"]["allowed_failed_groups"],
        ) == expected
        assert all(not row["power_met"] for row in result["examined_allowances"][:-1])


def test_nonzero_good_rate_can_make_zero_failure_design_underpowered():
    risk, good, alpha = Fraction(1, 20), Fraction(1, 80), Fraction(1, 20)
    zero = design.fixed_allowance_plan(risk=risk, alpha=alpha)["design"]
    assert design.tail_probability(zero["groups"], 0, good) < Fraction(4, 5)
    powered = design.power_constrained_plan(
        risk=risk, good=good, power=Fraction(4, 5), alpha=alpha
    )["design"]
    assert powered["allowed_failed_groups"] > 0
    assert powered["groups"] > zero["groups"]


@pytest.mark.parametrize(
    "groups,family", [(1, 1), (18, 1), (27, 1), (18, 8), (27, 8), (59, 1)]
)
def test_zero_bound_is_exact_enclosing_bracket(groups, family):
    result = design.zero_failure_upper_bracket(
        groups=groups, alpha=Fraction(1, 20), family_size=family
    )
    bracket = result["upper_bound_bracket"]
    lower, upper = Fraction(bracket["lower"]), Fraction(bracket["upper"])
    adjusted = Fraction(1, 20 * family)
    assert (1 - lower) ** groups >= adjusted >= (1 - upper) ** groups
    assert upper - lower == Fraction(1, 10**12)


@pytest.mark.parametrize(
    "value",
    [
        Fraction(0),
        Fraction(1),
        Fraction(1, 3),
        Fraction(1, 10**40),
        Fraction(10**40 - 1, 10**40),
    ],
)
def test_decimal_display_encloses_exact_probability(value):
    bracket = design.decimal_bracket(value)
    lower, upper = Fraction(bracket["lower"]), Fraction(bracket["upper"])
    assert lower <= value <= upper
    assert upper - lower <= Fraction(1, 10**design.DECIMAL_PLACES)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        0.05,
        "1/20",
        0,
        Fraction(0),
        Fraction(1),
        Fraction(-1, 2),
        Fraction(3, 2),
        Fraction(1, 1_000_001),
    ],
)
def test_invalid_risk_rejected(bad):
    with pytest.raises(ValueError):
        design.fixed_allowance_plan(risk=bad, alpha=Fraction(1, 20))


@pytest.mark.parametrize(
    "name,bad",
    [
        ("family_size", True),
        ("family_size", 0),
        ("family_size", 129),
        ("max_groups", 0),
        ("max_groups", 5001),
        ("allowed_failures", -1),
        ("allowed_failures", 65),
        ("allowed_failures", 1.0),
    ],
)
def test_integer_bounds_reject_not_clip(name, bad):
    with pytest.raises(ValueError):
        design.fixed_allowance_plan(
            risk=Fraction(1, 20), alpha=Fraction(1, 20), **{name: bad}
        )


@pytest.mark.parametrize(
    "good,power",
    [
        (Fraction(1, 20), Fraction(4, 5)),
        (Fraction(1, 10), Fraction(4, 5)),
        (Fraction(-1, 20), Fraction(4, 5)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
        (0.0, Fraction(4, 5)),
    ],
)
def test_invalid_power_design_rejected(good, power):
    with pytest.raises(ValueError):
        design.power_constrained_plan(
            risk=Fraction(1, 20), good=good, power=power, alpha=Fraction(1, 20)
        )


def test_exhaustion_reports_declared_limit_not_impossibility():
    result = design.fixed_allowance_plan(
        risk=Fraction(1, 20), alpha=Fraction(1, 20), max_groups=58
    )
    assert result["status"] == "no_design_within_declared_limits"
    assert result["design"] is None
    assert result["declared_limits"]["max_groups"] == 58


def test_work_exhaustion_raises_without_partial_design(monkeypatch):
    monkeypatch.setattr(design, "MAX_TAIL_EVALUATIONS", 1)
    with pytest.raises(ValueError, match="never truncate"):
        design.fixed_allowance_plan(risk=Fraction(1, 20), alpha=Fraction(1, 20))


def test_exact_tail_equality_is_a_pass_not_float_boundary_failure():
    result = design.fixed_allowance_plan(risk=Fraction(1, 2), alpha=Fraction(1, 4))
    assert result["design"]["groups"] == 2
    assert result["design"]["tail_at_risk_ceiling"]["lower"] == "0.250000000000000000"


def test_group_limit_smaller_than_failure_allowance_returns_no_design():
    result = design.fixed_allowance_plan(
        risk=Fraction(1, 2), alpha=Fraction(1, 4), max_groups=2, allowed_failures=3
    )
    assert result["design"] is None


@pytest.mark.parametrize(
    "alpha", [True, 0.05, Fraction(0), Fraction(1), Fraction(1, 1_000_001)]
)
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        design.fixed_allowance_plan(risk=Fraction(1, 20), alpha=alpha)


@pytest.mark.parametrize("groups", [True, 0, 5001, 18.0])
def test_invalid_bound_group_count_rejected(groups):
    with pytest.raises(ValueError):
        design.zero_failure_upper_bracket(groups=groups, alpha=Fraction(1, 20))


def test_report_has_all_sensitivity_conditions_and_no_completion_claim():
    report = sensitivity_report()
    assert len(report["rows"]) == 9
    assert len(report["hypothetical_zero_failure_bounds"]) == 6
    assert (
        report[
            "risk_and_power_values_are_sensitivity_choices_not_acceptance_thresholds"
        ]
        is True
    )
    assert "not formal preregistration" in report["scope"]
    assert len(report["remaining_preregistration_requirements"]) == 5
    assert all(row["powered_design"]["status"] == "feasible" for row in report["rows"])
    assert not any("runs/" in key or "sealed" in key for key in report["source_sha256"])


def test_cli_deterministic_no_corpus_or_world_inputs():
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "scripts" / "plan_sample_precision.py")]
    runs = [
        subprocess.run(command, cwd=root, capture_output=True, check=True, timeout=60)
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout
    assert (
        json.loads(runs[0].stdout)["schema"] == "bernoulli-sample-design-sensitivity-v1"
    )
