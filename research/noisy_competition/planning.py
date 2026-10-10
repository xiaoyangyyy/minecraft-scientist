"""Strong joint MI baseline and ordinary Bayes mechanism-risk design criterion.

Neither criterion is claimed novel. All words, including uninformative words,
belong to the same declared query space. Exact factors/likelihoods drive updates.
"""

import itertools
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

from .belief import WorkBound

MAX_WORDS = 512
PRECISION = 50
POLICIES = ("joint_information_gain", "mechanism_risk", "uniform_word")


def decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


@dataclass(frozen=True)
class Plan:
    word: tuple[str, ...]
    reset: bool
    cost: int
    mechanism_risk_reduction: Fraction
    information_gain_bits: str
    distinguishing_in_declared_word_space: bool


def value(bank, forecast):
    weights, predictions, mixture = forecast
    prior_risk = 1 - max(bank.project(weights).values())
    expected_risk, information, terms = Fraction(0), Decimal(0), []
    positive = [p for p, w in zip(predictions, weights, strict=True) if w]
    distinguishing = any(p != positive[0] for p in positive[1:])
    if not distinguishing:
        return Fraction(0), "0", False
    with localcontext() as context:
        context.prec = PRECISION
        for observed, probability in sorted(mixture.items()):
            posterior = tuple(
                w * dist.get(observed, Fraction(0)) / probability
                for w, dist in zip(weights, predictions, strict=True)
            )
            expected_risk += probability * (1 - max(bank.project(posterior).values()))
            for weight, dist in zip(weights, predictions, strict=True):
                conditional = dist.get(observed, Fraction(0))
                if weight and conditional:
                    terms.append(
                        decimal(weight * conditional)
                        * decimal(conditional / probability).ln()
                        / Decimal(2).ln()
                    )
        # Numeric ordering, not opaque-output spelling, fixes summation roundoff.
        information = sum(sorted(terms), Decimal(0))
        if information < 0:
            if information < Decimal("-1e-40"):
                raise ArithmeticError(
                    "negative mutual information beyond numeric tolerance"
                )
            information = Decimal(0)
        information = +information
    reduction = prior_risk - expected_risk
    if reduction < 0:
        raise ArithmeticError("Bayes risk increased under exact expected conditioning")
    return reduction, str(information), True


def choose(bank, *, costs, remaining, horizon, policy, rng=None):
    if (
        type(costs) is not dict
        or set(costs) != set(bank.actions)
        or any(type(v) is not int or not 1 <= v <= 100 for v in costs.values())
        or type(remaining) is not int
        or remaining < 0
        or type(horizon) is not int
        or not 1 <= horizon <= 4
        or policy not in POLICIES
    ):
        raise ValueError("invalid public planning contract")
    count = sum(len(bank.actions) ** length for length in range(1, horizon + 1))
    if count > MAX_WORDS:
        raise WorkBound("complete word space exceeds bound; never shorten the horizon")
    options = [
        (word, reset, sum(costs[a] for a in word) + int(reset))
        for length in range(1, horizon + 1)
        for word in itertools.product(bank.actions, repeat=length)
        for reset in (False, True)
        if sum(costs[a] for a in word) + int(reset) <= remaining
    ]
    if not options:
        return None
    if policy == "uniform_word":
        if rng is None:
            raise ValueError("random baseline requires an explicitly seeded RNG")
        word, reset, cost = options[rng.randrange(len(options))]
        return Plan(word, reset, cost, Fraction(0), "0", False)
    best, best_rank = None, None
    score_cache = {}
    with localcontext() as context:
        context.prec = PRECISION
        for word, reset, cost in options:
            forecast = bank.forecast(word, reset=reset)
            signature = tuple(tuple(sorted(p.items())) for p in forecast[1])
            if signature not in score_cache:
                score_cache[signature] = value(bank, forecast)
            reduction, information, distinguishing = score_cache[signature]
            if not distinguishing:
                continue
            risk_rate, info_rate = reduction / cost, Decimal(information) / cost
            rank = (
                (risk_rate, info_rate, -cost)
                if policy == "mechanism_risk"
                else (info_rate, risk_rate, -cost)
            )
            # Declared action/query order breaks ties; no label spelling ranking.
            if best is None or rank > best_rank:
                best_rank = rank
                best = Plan(word, reset, cost, reduction, information, True)
    return best
