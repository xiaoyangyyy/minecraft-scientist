"""An explicit sensitivity grid, not an automatically chosen preregistration."""

import hashlib
from fractions import Fraction
from pathlib import Path

from .binomial import (
    fixed_allowance_plan,
    power_constrained_plan,
    rational,
    zero_failure_upper_bracket,
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "Pre-data illustrative design arithmetic only; not formal preregistration, observed failure bounds, comparative power, independent-world evidence, novelty, H1/H2/H3/H4 or Minecraft/RGB acceptance."


def hashes():
    paths = sorted((ROOT / "research" / "sample_design").glob("*.py"))
    paths += [
        ROOT / "research" / "sample_design" / "README.md",
        ROOT / "docs" / "sample-design-contract.md",
        ROOT / "scripts" / "plan_sample_precision.py",
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def sensitivity_report():
    alpha = Fraction(1, 20)
    rows = []
    for risk in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10)):
        for family_size in (1, 4, 8):
            rows.append(
                {
                    "risk_ceiling": rational(risk),
                    "family_size": family_size,
                    "zero_failure_design": fixed_allowance_plan(
                        risk=risk, alpha=alpha, family_size=family_size
                    ),
                    "one_failure_design": fixed_allowance_plan(
                        risk=risk,
                        alpha=alpha,
                        family_size=family_size,
                        allowed_failures=1,
                    ),
                    "powered_design": power_constrained_plan(
                        risk=risk,
                        good=risk / 4,
                        power=Fraction(4, 5),
                        alpha=alpha,
                        family_size=family_size,
                    ),
                }
            )
    return {
        "schema": "bernoulli-sample-design-sensitivity-v1",
        "source_sha256": hashes(),
        "scope": SCOPE,
        "assumptions": [
            "Sampling units are IID draws from a defined target distribution with a common Bernoulli failure probability per endpoint.",
            "An endpoint is binary per independently sampled group; any member failure can define group failure only with a prespecified, identical within-group assessment protocol.",
            "Renamings/replays/fixtures within a leakage group are not new independent Bernoulli units; group closure itself does not establish IID sampling.",
            "Family members and risk ceilings must be fixed before assessment; Bonferroni coverage does not require independence between endpoints.",
            "Fixed sample size and no optional stopping, success filtering, post-hoc endpoints or replacement of failed units.",
            "Illustrative power is marginal at a good failure rate equal to one-quarter the risk ceiling; it is not joint-family or comparative-method power.",
        ],
        "family_alpha": rational(alpha),
        "risk_and_power_values_are_sensitivity_choices_not_acceptance_thresholds": True,
        "rows": rows,
        "hypothetical_zero_failure_bounds": [
            zero_failure_upper_bracket(groups=n, alpha=alpha, family_size=m)
            for n in (1, 18, 27)
            for m in (1, 8)
        ],
        "remaining_preregistration_requirements": [
            "Define the target world distribution, genuinely independent sampling design, structural holdouts and fixed within-unit assessment.",
            "Choose primary safety/performance endpoints, scientific risk ceilings, marginal/joint power and complete multiplicity family before sealed data are opened.",
            "Specify comparative paired effect sizes/precision and associated sample planning; this binomial tool does not do that.",
            "Prespecify treatment of unresolved, no-anomaly, no-candidate and failed-control outcomes, coverage safeguards and all denominators.",
            "Freeze generator, hypotheses, method/baseline fidelity, budgets, analysis code and stopping rules; retain failures and all compute/API costs.",
        ],
        "references": [
            "https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm",
            "https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm",
        ],
    }
