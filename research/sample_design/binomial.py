"""Exact, bounded fixed-sample Bernoulli designs; no evidence loader or peeking.

The sampling unit and its IID Bernoulli interpretation are assumptions, not
properties proved by leakage-group closure. Decisions use integer arithmetic.
Displayed probabilities are enclosing decimal brackets, never decision inputs.
"""

from fractions import Fraction

MAX_GROUPS = 5000
MAX_ALLOWANCE = 64
MAX_FAMILY = 128
MAX_DENOMINATOR = 1_000_000
MAX_TAIL_EVALUATIONS = 2000
DECIMAL_PLACES = 18


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def probability(value, name, *, open_interval=False):
    if type(value) is not Fraction:
        raise ValueError(f"{name} must be an exact Fraction, not float/bool/string")
    if not 0 <= value <= 1 or (open_interval and value in (0, 1)):
        raise ValueError(f"{name} outside its probability domain")
    if value.denominator > MAX_DENOMINATOR:
        raise ValueError(f"{name} denominator exceeds resource bound")
    return value


def _tail(n, k, p):
    """P(Binomial(n,p) <= k), as unreduced integers; exact recurrence."""
    if k >= n or p == 0:
        return 1, 1
    if p == 1:
        return 0, 1
    a, b = p.numerator, p.denominator
    q = b - a
    term = q**n
    numerator = term
    for j in range(k):
        term, remainder = divmod(term * (n - j) * a, (j + 1) * q)
        if remainder:
            raise ArithmeticError("binomial recurrence lost integer exactness")
        numerator += term
    return numerator, b**n


def tail_probability(n, k, p):
    integer(n, "groups", 1, MAX_GROUPS)
    integer(k, "allowed_failures", 0, MAX_ALLOWANCE)
    probability(p, "failure_probability")
    a, b = _tail(n, k, p)
    return Fraction(a, b)


def _decimal_units(units, places):
    scale = 10**places
    return f"{units // scale}.{units % scale:0{places}d}"


def decimal_bracket(value):
    """Outward-rounded exact bracket; only the small display integers stringify."""
    if type(value) is not Fraction or not 0 <= value <= 1:
        raise ValueError("display value must be a probability Fraction")
    scale = 10**DECIMAL_PLACES
    lower, remainder = divmod(value.numerator * scale, value.denominator)
    return {
        "lower": _decimal_units(lower, DECIMAL_PLACES),
        "upper": _decimal_units(lower + bool(remainder), DECIMAL_PLACES),
        "decimal_places": DECIMAL_PLACES,
    }


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


class _Counter:
    def __init__(self):
        self.evaluations = 0

    def tail(self, n, k, p):
        self.evaluations += 1
        if self.evaluations > MAX_TAIL_EVALUATIONS:
            raise ValueError("exact tail work bound exceeded; never truncate a design")
        return _tail(n, k, p)


def _passes(n, k, risk, alpha, counter):
    a, b = counter.tail(n, k, risk)
    return a * alpha.denominator <= b * alpha.numerator


def _minimum_n(k, risk, alpha, limit, counter):
    # For fixed k the lower tail decreases with n. Exponential bracketing then
    # binary search proves minimality without rounded logarithms at boundaries.
    if k >= limit:
        return None
    low, high = k, k + 1
    while not _passes(high, k, risk, alpha, counter):
        low = high
        if high == limit:
            return None
        high = min(limit, high * 2)
    while high - low > 1:
        middle = (low + high) // 2
        if _passes(middle, k, risk, alpha, counter):
            high = middle
        else:
            low = middle
    return high


def _inputs(risk, alpha, family_size, max_groups):
    probability(risk, "risk_ceiling", open_interval=True)
    probability(alpha, "family_alpha", open_interval=True)
    integer(family_size, "family_size", 1, MAX_FAMILY)
    integer(max_groups, "max_groups", 1, MAX_GROUPS)
    return alpha / family_size


def _result(n, k, risk, alpha, family_size, counter, good=None):
    adjusted = alpha / family_size
    tail = Fraction(*counter.tail(n, k, risk))
    previous = Fraction(*counter.tail(n - 1, k, risk)) if n > 1 else Fraction(1)
    if not tail <= adjusted < previous:
        raise ArithmeticError("fixed-allowance minimality certificate failed")
    result = {
        "groups": n,
        "allowed_failed_groups": k,
        "risk_ceiling": rational(risk),
        "family_alpha": rational(alpha),
        "family_size": family_size,
        "per_endpoint_alpha": rational(adjusted),
        "tail_at_risk_ceiling": decimal_bracket(tail),
        "tail_at_one_fewer_group": decimal_bracket(previous),
        "exact_minimality_checks": {"n_passes": True, "n_minus_one_fails": True},
    }
    if good is not None:
        result["assumed_good_failure_probability"] = rational(good)
        result["pass_probability_at_good_rate"] = decimal_bracket(
            Fraction(*counter.tail(n, k, good))
        )
    return result


def fixed_allowance_plan(
    *, risk, alpha, family_size=1, allowed_failures=0, max_groups=MAX_GROUPS
):
    adjusted = _inputs(risk, alpha, family_size, max_groups)
    integer(allowed_failures, "allowed_failures", 0, MAX_ALLOWANCE)
    counter = _Counter()
    n = _minimum_n(allowed_failures, risk, adjusted, max_groups, counter)
    return {
        "status": "feasible" if n is not None else "no_design_within_declared_limits",
        "design": None
        if n is None
        else _result(n, allowed_failures, risk, alpha, family_size, counter),
        "declared_limits": {
            "max_groups": max_groups,
            "allowed_failures": allowed_failures,
        },
        "exact_tail_evaluations": counter.evaluations,
    }


def power_constrained_plan(
    *,
    risk,
    good,
    power,
    alpha,
    family_size=1,
    max_groups=MAX_GROUPS,
    max_allowed_failures=MAX_ALLOWANCE,
):
    """Smallest n among fixed-n lower-tail designs within declared k/n bounds.

    For each k, n(k) is the least n passing the risk test; power decreases with
    n at fixed k. n(k) increases with k, so the first feasible k has minimal n.
    This is marginal per-endpoint power, NOT joint power of the full family.
    """
    adjusted = _inputs(risk, alpha, family_size, max_groups)
    probability(good, "good_failure_probability")
    probability(power, "required_power", open_interval=True)
    if good >= risk:
        raise ValueError("assumed good rate must be below the risk ceiling")
    integer(max_allowed_failures, "max_allowed_failures", 0, MAX_ALLOWANCE)
    counter = _Counter()
    examined = []
    design = None
    for k in range(max_allowed_failures + 1):
        n = _minimum_n(k, risk, adjusted, max_groups, counter)
        if n is None:
            # n(k) is increasing, so no later allowance fits the group limit.
            break
        a, b = counter.tail(n, k, good)
        enough = a * power.denominator >= b * power.numerator
        examined.append(
            {"allowed_failed_groups": k, "minimum_groups": n, "power_met": enough}
        )
        if enough:
            design = _result(n, k, risk, alpha, family_size, counter, good)
            design["required_marginal_power"] = rational(power)
            break
    return {
        "status": "feasible" if design else "no_design_within_declared_limits",
        "design": design,
        "examined_allowances": examined,
        "declared_limits": {
            "max_groups": max_groups,
            "max_allowed_failures": max_allowed_failures,
        },
        "exact_tail_evaluations": counter.evaluations,
    }


def zero_failure_upper_bracket(*, groups, alpha, family_size=1, places=12):
    """Bracket 1-(alpha/m)^(1/n) by exact rational tests, not float roots."""
    integer(groups, "groups", 1, MAX_GROUPS)
    integer(places, "places", 1, 12)
    adjusted = _inputs(Fraction(1, 2), alpha, family_size, groups)
    scale = 10**places
    low, high = 0, scale
    # The root lies where (1-u)^n = alpha/m. The low point has greater tail;
    # the high point has smaller/equal tail. Endpoints need no special floats.
    while high - low > 1:
        middle = (low + high) // 2
        if (
            scale - middle
        ) ** groups * adjusted.denominator <= scale**groups * adjusted.numerator:
            high = middle
        else:
            low = middle
    return {
        "groups": groups,
        "assumed_observed_failures": 0,
        "family_size": family_size,
        "per_endpoint_alpha": rational(adjusted),
        "upper_bound_bracket": {
            "lower": _decimal_units(low, places),
            "upper": _decimal_units(high, places),
            "decimal_places": places,
        },
        "scope": "hypothetical fixed-n IID Bernoulli endpoint; not observed evidence or proof that these groups are independent",
    }
