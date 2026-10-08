"""Budgeted joint diagnosis reuses H1 planning, execution and evidence events."""

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.discovery.diagnosis import (
    project_diagnosis,
    validate_hypotheses,
)


def diagnose_mechanism(
    view,
    hypotheses,
    calibration,
    costs,
    *,
    baseline_mechanism,
    baseline_controller,
    method="witness_cover",
    seed=0,
    budget=32,
    horizon=5,
):
    """No private truth, world condition or world seed is accepted here.

    A baseline is supplied public reference knowledge, not a theory promoted by
    this run. Diagnostic planning is conditional on the declared joint family.
    """
    candidates = validate_hypotheses(
        hypotheses, baseline_mechanism, baseline_controller
    )
    identification = identify_mechanism(
        view,
        candidates,
        calibration,
        costs,
        method=method,
        seed=seed,
        budget=budget,
        horizon=horizon,
    )
    return {
        "schema": "joint-diagnosis-run-v1",
        "identification": identification,
        "diagnosis": project_diagnosis(
            hypotheses, identification, baseline_mechanism, baseline_controller
        ),
    }
