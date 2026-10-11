"""Standard universal-inference e-process over fixed public stochastic models.

E_t = uniform output likelihood / max_i fixed-model likelihood_i.
Under any correctly specified fixed null model, this is dominated by that
model's likelihood-ratio test supermartingale. Crossing 1/alpha is an anytime
rejection, conditional on the declared output/control/reset law. The minimum
of these ratios need not itself be a supermartingale. Nonrejection is NOT fit.
"""

from copy import deepcopy
from fractions import Fraction

from research.stochastic_ir.program import StochasticProgram

MAX_STEPS = 256
MAX_BITS = 12000
MAX_PREDICTION_BRANCHES = 2_000_000
SCOPE = "Fixed finite stationary public-program family; predictable uniform public-output alternative, cumulative likelihood across resets and per-trajectory alpha. Rejection is not fault attribution or a replacement theory; nonrejection is not adequacy/acceptance. Validity requires the declared conditional output/reset law, not just deterministic PRNG seeds."


class AdequacyBound(ValueError):
    """No partial likelihood update or successful certificate after exhaustion."""


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def e_record(value):
    return (
        {"kind": "infinite"}
        if value is None
        else {"kind": "finite", "value": rational(value)}
    )


def maximum_e(left, right):
    return None if left is None or right is None else max(left, right)


class FamilyAdequacyMonitor:
    def __init__(
        self,
        programs,
        keys,
        *,
        alpha=Fraction(1, 20),
        max_steps=MAX_STEPS,
        bit_limit=MAX_BITS,
    ):
        if (
            type(programs) is not tuple
            or not 1 <= len(programs) <= 32
            or any(type(p) is not StochasticProgram for p in programs)
        ):
            raise ValueError("one to 32 fixed shared-IR programs required")
        if (
            type(keys) is not tuple
            or len(keys) != len(programs)
            or len(set(keys)) != len(keys)
            or any(type(k) is not str or not 1 <= len(k) <= 80 for k in keys)
        ):
            raise ValueError("unique bounded public model keys required")
        if (
            type(alpha) is not Fraction
            or not 0 < alpha < 1
            or alpha.denominator > 1_000_000
        ):
            raise ValueError("strict bounded rational alpha in (0,1) required")
        if (
            type(max_steps) is not int
            or not 1 <= max_steps <= MAX_STEPS
            or type(bit_limit) is not int
            or not 1 <= bit_limit <= MAX_BITS
        ):
            raise ValueError("strict bounded monitor limits required")
        self.programs, self.keys, self.alpha = programs, keys, alpha
        self.actions, self.outputs = programs[0].actions, programs[0].outputs
        if any(
            p.actions != self.actions or p.outputs != self.outputs for p in programs
        ):
            raise ValueError("shared declared public action/output order required")
        self.max_steps, self.bit_limit = max_steps, bit_limit
        self._beliefs = tuple(p.initial_belief() for p in programs)
        self._likelihoods = (Fraction(1),) * len(programs)
        self._alternative, self._e, self._max_e = Fraction(1), Fraction(1), Fraction(1)
        self.steps, self.resets, self.first_rejection_step = 0, 0, None
        self.prediction_branches = 0
        self._pending = None
        for p in programs:
            for mass in p.initial_belief().values():
                self._bounded(mass)

    def _bounded(self, value):
        if (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            > self.bit_limit
        ):
            raise AdequacyBound("exact likelihood coefficient bound; no rounding")
        return value

    def snapshot(self):
        return {
            "model_keys": list(self.keys),
            "alpha": rational(self.alpha),
            "alternative": "uniform-public-output-v1",
            "steps": self.steps,
            "resets": self.resets,
            "prediction_branches": self.prediction_branches,
            "model_likelihoods": [rational(v) for v in self._likelihoods],
            "alternative_likelihood": rational(self._alternative),
            "maximum_null_likelihood": rational(max(self._likelihoods)),
            "e_value": e_record(self._e),
            "running_max_e_value": e_record(self._max_e),
            "family_rejected": self.first_rejection_step is not None,
            "first_rejection_step": self.first_rejection_step,
            "scope": SCOPE,
        }

    def reset_state(self):
        if self._pending is not None:
            raise ValueError("cannot reset while acquired output is pending")
        # The action resets latent state, NOT the statistical evidence/alpha.
        self._beliefs = tuple(p.initial_belief() for p in self.programs)
        self.resets += 1

    def prepare(self, action):
        if (
            self._pending is not None
            or type(action) is not str
            or action not in self.actions
        ):
            raise ValueError("one known issued command pending at a time")
        if self.steps >= self.max_steps:
            raise AdequacyBound("observation bound; no implicit new testing epoch")
        branch_count = sum(
            len(p.table[s, action])
            for p, b, likelihood in zip(
                self.programs, self._beliefs, self._likelihoods, strict=True
            )
            if likelihood
            for s, mass in b.items()
            if mass
        )
        if self.prediction_branches + branch_count > MAX_PREDICTION_BRANCHES:
            raise AdequacyBound(
                "monitor prediction branch bound; no null-model pruning"
            )
        distributions = tuple(
            p.predict_step(action, belief=b) if likelihood else {}
            for p, b, likelihood in zip(
                self.programs, self._beliefs, self._likelihoods, strict=True
            )
        )
        proposal = {
            "atomic_index": self.steps + 1,
            "issued_command": action,
            "alternative_probability": rational(
                self._bounded(Fraction(1, len(self.outputs)))
            ),
            "component_predictions": [
                {
                    "key": key,
                    "null_prefix_possible": bool(likelihood),
                    "output_distribution": {
                        o: rational(self._bounded(sum(states.values(), Fraction(0))))
                        for o, states in sorted(distribution.items())
                    },
                }
                for key, likelihood, distribution in zip(
                    self.keys, self._likelihoods, distributions, strict=True
                )
            ],
        }
        self.prediction_branches += branch_count
        self._pending = distributions, proposal
        return deepcopy(proposal)

    def finish(self, observed):
        if (
            self._pending is None
            or type(observed) is not str
            or observed not in self.outputs
        ):
            raise ValueError("prepared command and declared observed output required")
        following, likelihoods = [], []
        distributions, _ = self._pending
        for prior, distribution in zip(self._likelihoods, distributions, strict=True):
            states = distribution.get(observed, {})
            conditional = self._bounded(sum(states.values(), Fraction(0)))
            likelihoods.append(self._bounded(prior * conditional))
            following.append(
                {s: self._bounded(mass / conditional) for s, mass in states.items()}
                if conditional
                else {}
            )
        alternative = self._bounded(self._alternative / len(self.outputs))
        max_null = max(likelihoods)
        current_e = self._bounded(alternative / max_null) if max_null else None
        running_max = maximum_e(self._max_e, current_e)
        first = self.first_rejection_step
        if first is None and (current_e is None or current_e >= 1 / self.alpha):
            first = self.steps + 1
        # Commit only after every exact inference/bound check passes.
        self._beliefs, self._likelihoods = tuple(following), tuple(likelihoods)
        self._alternative, self._e, self._max_e = alternative, current_e, running_max
        self.steps += 1
        self.first_rejection_step, self._pending = first, None
        return self.snapshot()
