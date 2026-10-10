"""Independent arithmetic and denominator negatives, not scientific samples."""

import itertools
import math
from copy import deepcopy

import pytest

from research.challenger_validation.paired import POLICIES
from research.structural_campaigns.runtime import MATCHED_METHODS

from .metrics import analyze, paired_range, quantile


def selection(groups=None):
    groups = groups or {"a": "large", "b": "large", "c": "small"}
    return {"split": "development", "fixture_ids": sorted(groups), "groups": groups}


def synthetic(kind="structural", selected=None):
    selected = selected or selection()
    rows = []
    for fid, method in itertools.product(selected["fixture_ids"], MATCHED_METHODS):
        policies = (None,) if kind == "structural" else POLICIES
        for policy in policies:
            good = method == "structural_witness" and fid != "c"
            row = {
                "fixture_id": fid,
                "method": method,
                "independent_group": selected["groups"][fid],
                "case": "unchanged",
                "renaming_index": 0,
                "status": "revision_finitely_validated" if good else "unresolved",
                "spent": 10,
                "raw_score": {
                    "promoted_correct_at_H": True if good else None,
                    "proposed_correct_at_H": True if good else None,
                    "controls_match": True,
                    "behavior_change_at_H": False,
                    "old_correct_at_H": True,
                    "reset_count": 2,
                    "atomic_action_count": 8,
                    "reported_failed_dispatches": 0,
                },
            }
            if kind == "structural":
                row["evaluation"] = {
                    "post_boundary_anomaly_detected": False,
                    "behavior_change_within_H": False,
                    "anomaly_without_in_H_change_or_before_boundary": False,
                }
                row["construction_diagnostics"] = {
                    "behavior_change_at_H": False,
                    "target_exact_in_public_family": True,
                    "within_declared_mechanism_edit_family": True,
                }
            else:
                row.update(
                    {
                        "policy": policy,
                        "eligible": True,
                        "prefix_outcome": "eligible_tail",
                        "prefix_spent": 8,
                        "mutation_operations": 0,
                    }
                )
            rows.append(row)
    return rows


def test_uneven_groups_have_distinct_fixture_and_equal_group_estimands():
    result = analyze(synthetic(), selection(), kind="structural", resamples=100)
    cell = result["strata"]["all"]["cells"]["structural_witness"]
    assert cell["fixture_totals"]["promotion_correct_at_h"] == 2
    assert cell["fixture_weighted_mean"]["promotion_correct_at_h"] == 2 / 3
    assert cell["equal_group_mean"]["promotion_correct_at_h"] == 0.5
    assert cell["fixtures"] == 3 and cell["leakage_groups"] == 2
    assert result["rows"] == 12 and result["leakage_groups"] == 2
    contrast = next(
        c
        for c in result["strata"]["all"]["contrasts"]
        if (c["left"], c["right"]) == ("from_scratch", "structural_witness")
    )
    assert contrast["paired_group_differences"]["large"]["promotion_correct_at_h"] == -1
    assert contrast["paired_group_differences"]["small"]["promotion_correct_at_h"] == 0
    assert contrast["metrics"]["promotion_correct_at_h"]["mean"] == -0.5


def test_duplicate_renamings_within_group_do_not_create_new_uncertainty_units():
    original = analyze(synthetic(), selection(), kind="structural", resamples=100)
    enlarged = selection(
        {"a": "large", "b": "large", "c": "small", "d": "large", "e": "large"}
    )
    result = analyze(
        synthetic(selected=enlarged), enlarged, kind="structural", resamples=100
    )
    assert result["fixtures"] == 5 and result["leakage_groups"] == 2
    assert (
        result["strata"]["all"]["cells"]["structural_witness"]["fixture_weighted_mean"][
            "promotion_correct_at_h"
        ]
        == 4 / 5
    )
    for left, right in zip(
        original["strata"]["all"]["contrasts"],
        result["strata"]["all"]["contrasts"],
        strict=True,
    ):
        assert left["metrics"] == right["metrics"]


def test_one_group_and_empty_stratum_produce_no_uncertainty_claim():
    chosen = selection({"a": "one", "b": "one"})
    result = analyze(
        synthetic(selected=chosen), chosen, kind="structural", resamples=100
    )
    for contrast in result["strata"]["all"]["contrasts"]:
        assert all(
            metric["exploratory_percentile_range"] is None
            for metric in contrast["metrics"].values()
        )
    empty = result["strata"]["case/insert"]
    assert empty["fixture_ids"] == [] and empty["group_fixture_counts"] == {}
    assert empty["cells"]["adaptive"]["equal_group_mean"]["actual_cost"] is None
    assert (
        result["strata"]["all"]["cells"]["adaptive"]["equal_group_mean"]["unresolved"]
        == 1
    )


def test_row_permutation_and_repeated_analysis_are_deterministic():
    rows = synthetic()
    expected = analyze(rows, selection(), kind="structural", resamples=100)
    assert expected == analyze(
        list(reversed(rows)), selection(), kind="structural", resamples=100
    )


def test_known_paired_sampling_support_and_quantile_arithmetic():
    result = paired_range([0, 2], resamples=1000, seed=0)
    assert result["mean"] == 1
    assert result["exploratory_percentile_range"] == [0, 2]
    # Enumerated possible means of two draws from two known group effects.
    possible = sorted(sum(pair) / 2 for pair in itertools.product((0, 2), repeat=2))
    assert possible == [0, 1, 1, 2]
    assert quantile(possible, 0.025) == pytest.approx(0.075)
    assert quantile(possible, 0.975) == pytest.approx(1.925)


def test_zero_observed_variation_is_flagged_not_generalization_certified():
    result = paired_range([2, 2, 2], resamples=100, seed=0)
    assert result["exploratory_percentile_range"] == [2, 2]
    assert result["zero_observed_between_group_variation"] is True
    assert "development-only" in result["scope"]


def test_missing_or_wrong_old_models_are_distinct_and_never_dropped():
    rows = synthetic()
    rows[0]["raw_score"]["old_correct_at_H"] = None
    rows[4]["raw_score"]["old_correct_at_H"] = False
    result = analyze(rows, selection(), kind="structural", resamples=100)
    cell = result["strata"]["all"]["cells"]["from_scratch"]
    assert cell["fixtures"] == 3
    assert cell["fixture_totals"]["old_model_absent"] == 1
    assert cell["fixture_totals"]["old_wrong_at_h"] == 1
    assert cell["fixture_totals"]["old_correct_at_h"] == 1


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "duplicate",
        "wrong_group",
        "method",
        "case",
        "metadata",
        "status",
        "contradictory_promotion",
        "bool_cost",
        "negative_cost",
        "nonbool_change",
        "controls",
        "missing_field",
        "missing_score",
    ),
)
def test_row_tampering_never_filters_into_a_complete_analysis(change):
    rows = synthetic()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change in ("wrong_group", "method", "case", "status"):
        rows[0][{"wrong_group": "independent_group"}.get(change, change)] = "forged"
    elif change == "metadata":
        rows[0]["renaming_index"] = 1
    elif change == "contradictory_promotion":
        rows[0]["raw_score"]["promoted_correct_at_H"] = False
    elif change in ("bool_cost", "negative_cost"):
        rows[0]["spent"] = True if change == "bool_cost" else -1
    elif change == "nonbool_change":
        rows[0]["raw_score"]["behavior_change_at_H"] = 0
    elif change == "controls":
        rows[0]["raw_score"]["controls_match"] = False
    elif change == "missing_field":
        del rows[0]["fixture_id"]
    else:
        del rows[0]["raw_score"]["promoted_correct_at_H"]
    with pytest.raises(ValueError):
        analyze(rows, selection(), kind="structural", resamples=100)


@pytest.mark.parametrize(
    "change", ("split", "duplicate_fid", "missing_group", "empty", "group_type")
)
def test_invalid_or_sealed_selection_rejected(change):
    chosen = selection()
    if change == "split":
        chosen["split"] = "evaluation"
    elif change == "duplicate_fid":
        chosen["fixture_ids"].append("c")
    elif change == "missing_group":
        chosen["groups"].pop("c")
    elif change == "empty":
        chosen["fixture_ids"], chosen["groups"] = [], {}
    else:
        chosen["groups"]["a"] = 1
    with pytest.raises(ValueError):
        analyze(synthetic(), chosen, kind="structural", resamples=100)


@pytest.mark.parametrize(
    "resamples,seed", ((True, 0), (99, 0), (20001, 0), (100, True), (100, -1))
)
def test_invalid_resampling_contract(resamples, seed):
    with pytest.raises(ValueError):
        paired_range([0, 1], resamples=resamples, seed=seed)


@pytest.mark.parametrize("values", ([], [True], [math.nan], [math.inf]))
def test_nonfinite_or_empty_effects_rejected(values):
    with pytest.raises(ValueError):
        paired_range(values, resamples=100, seed=0)


@pytest.mark.parametrize(
    "change",
    (
        "prefix_cost",
        "eligibility",
        "no_anomaly",
        "missing_policy",
        "negative_tail",
        "float_operations",
    ),
)
def test_validation_prefix_denominator_and_cost_negatives(change):
    rows = synthetic("validation")
    if change == "prefix_cost":
        rows[0]["prefix_spent"] = 7
    elif change == "eligibility":
        rows[0]["eligible"] = False
        rows[0]["prefix_outcome"] = "no_repair_candidate"
    elif change == "no_anomaly":
        rows[0]["eligible"] = False
        rows[0]["prefix_outcome"] = "bounded_no_anomaly"
    elif change == "missing_policy":
        rows.pop()
    elif change == "negative_tail":
        rows[0]["prefix_spent"] = 11
    else:
        rows[0]["mutation_operations"] = 0.0
    with pytest.raises(ValueError):
        analyze(
            rows, selection(), kind="validation", source_rows=synthetic(), resamples=100
        )


def test_paired_validation_reports_added_cost_and_lost_correct_candidates():
    rows = synthetic("validation")
    for row in rows:
        if row["policy"] == "public_mutation":
            row["spent"] += 2
            row["mutation_operations"] = 7
            if row["raw_score"]["promoted_correct_at_H"] is True:
                row["status"] = "candidate_rejected"
                row["raw_score"]["promoted_correct_at_H"] = None
    result = analyze(
        rows, selection(), kind="validation", source_rows=synthetic(), resamples=100
    )
    witness = next(
        c
        for c in result["strata"]["all"]["contrasts"]
        if c["left"] == "structural_witness/public_mutation"
    )
    assert witness["metrics"]["actual_cost"]["mean"] == 2
    assert witness["metrics"]["promotion_correct_at_h"]["mean"] == -0.5
    assert witness["metrics"]["rejected_correct_at_h"]["mean"] == 0.5
    assert result["rows"] == 24


def test_resampling_budget_failure_does_not_silently_reduce_groups(monkeypatch):
    from . import metrics

    monkeypatch.setattr(metrics, "MAX_OPERATIONS", 1)
    with pytest.raises(ValueError, match="budget"):
        analyze(synthetic(), selection(), kind="structural", resamples=100)


def test_prior_behavior_and_construction_family_context_cannot_be_hidden():
    rows = synthetic()
    for row in rows:
        if row["fixture_id"] == "c":
            row["construction_diagnostics"]["target_exact_in_public_family"] = False
            row["construction_diagnostics"]["within_declared_mechanism_edit_family"] = (
                False
            )
    result = analyze(rows, selection(), kind="structural", resamples=100)
    assert result["strata"]["target_behavior_in_public_family/0"]["fixture_ids"] == [
        "c"
    ]
    assert result["strata"]["construction_one_edit_family/0"]["fixture_ids"] == ["c"]
    assert result["strata"]["target_behavior_in_public_family/1"]["fixture_ids"] == [
        "a",
        "b",
    ]


def test_all_registered_renaming_indices_are_retained():
    rows = synthetic()
    for row in rows:
        if row["fixture_id"] == "c":
            row["renaming_index"] = 2
    result = analyze(rows, selection(), kind="structural", resamples=100)
    assert result["strata"]["renaming/2"]["fixture_ids"] == ["c"]


@pytest.mark.parametrize(
    "failure",
    (
        "missing_source",
        "wrong_case",
        "changed_context",
        "false_context_type",
        "different_method_context",
    ),
)
def test_validation_requires_complete_consistent_audited_source_context(failure):
    source_rows = synthetic()
    rows = synthetic("validation")
    if failure == "missing_source":
        source_rows = None
    elif failure == "wrong_case":
        for row in source_rows:
            row["case"] = "output"
    elif failure == "changed_context":
        for row in source_rows:
            row["raw_score"]["behavior_change_at_H"] = True
            row["evaluation"]["behavior_change_within_H"] = True
            row["construction_diagnostics"]["behavior_change_at_H"] = True
    elif failure == "false_context_type":
        source_rows[0]["construction_diagnostics"]["target_exact_in_public_family"] = 1
    else:
        source_rows[0]["construction_diagnostics"][
            "within_declared_mechanism_edit_family"
        ] = False
    with pytest.raises(ValueError):
        analyze(
            rows, selection(), kind="validation", source_rows=source_rows, resamples=100
        )
