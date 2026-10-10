"""All-denominator paired diagnostics at the registered leakage-group unit.

Resampling ranges are exploratory and assume exchangeable independent groups.
Leakage separation itself does not establish that assumption or formal inference.
"""

import itertools
import math
import random

from research.challenger_validation.paired import POLICIES
from research.structural_campaigns.runtime import MATCHED_METHODS
from research.structural_holdouts.generator import CASES

DEFAULT_RESAMPLES = 5000
DEFAULT_SEED = 20261010
MAX_RESAMPLES = 20000
MAX_OPERATIONS = 300_000_000
PREFIX_OUTCOMES = (
    "eligible_tail",
    "bootstrap_incomplete",
    "monitoring_incomplete",
    "bounded_no_anomaly",
    "prior_validation_rejected",
    "no_repair_candidate",
    "prior_validation_unresolved",
)
STRUCTURAL_STATUSES = (
    "revision_finitely_validated",
    "unresolved",
    "bounded_monitoring_completed_no_anomaly",
)
VALIDATION_STATUSES = (
    "revision_finitely_validated",
    "unresolved",
    "candidate_rejected",
)


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid exact nonnegative integer: {label}")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"invalid exact Boolean: {label}")
    return value


def parameters(resamples, seed):
    integer(seed, "seed")
    if type(resamples) is not int or not 100 <= resamples <= MAX_RESAMPLES:
        raise ValueError("resamples must be an integer in [100, 20000]")


def selection_check(selection):
    if type(selection) is not dict or selection.get("split") != "development":
        raise ValueError("analysis is development-only, not sealed evaluation")
    fids, groups = selection.get("fixture_ids"), selection.get("groups")
    if (
        type(fids) is not list
        or not fids
        or any(type(fid) is not str or not fid for fid in fids)
        or fids != sorted(set(fids))
        or type(groups) is not dict
        or set(groups) != set(fids)
        or any(type(group) is not str or not group for group in groups.values())
    ):
        raise ValueError("invalid complete registered fixture/group selection")
    return fids, groups


def normalized(row, kind):
    score = row["raw_score"]
    required = {
        "promoted_correct_at_H",
        "controls_match",
        "behavior_change_at_H",
        "reset_count",
        "atomic_action_count",
        "reported_failed_dispatches",
    }
    required.add(
        "old_correct_at_H" if kind == "structural" else "proposed_correct_at_H"
    )
    if type(score) is not dict or not required <= set(score):
        raise ValueError("invalid raw score")
    promoted = score["promoted_correct_at_H"]
    if promoted is not None and type(promoted) is not bool:
        raise ValueError("promotion must be Boolean or unknown")
    if (promoted is not None) != (row["status"] == "revision_finitely_validated"):
        raise ValueError("promotion/status contradiction")
    boolean(score["controls_match"], "controls_match")
    if score["controls_match"] is not True:
        raise ValueError("unverified control chronology")
    changed = boolean(score["behavior_change_at_H"], "behavior_change_at_H")
    result = {
        "promotion_correct_at_h": int(promoted is True),
        "promotion_wrong_at_h": int(promoted is False),
        "not_promoted": int(promoted is None),
        "unresolved": int(row["status"] == "unresolved"),
        "changed_at_h": int(changed),
        "actual_cost": integer(row["spent"], "spent"),
        "reset_count": integer(score["reset_count"], "reset_count"),
        "atomic_action_count": integer(
            score["atomic_action_count"], "atomic_action_count"
        ),
        "reported_failed_dispatches": integer(
            score["reported_failed_dispatches"], "reported_failed_dispatches"
        ),
    }
    if kind == "structural":
        evaluation = row["evaluation"]
        if type(evaluation) is not dict or not {
            "post_boundary_anomaly_detected",
            "behavior_change_within_H",
            "anomaly_without_in_H_change_or_before_boundary",
        } <= set(evaluation):
            raise ValueError("invalid evaluation fields")
        detected = boolean(evaluation["post_boundary_anomaly_detected"], "detected")
        old_correct = score["old_correct_at_H"]
        if old_correct is not None and type(old_correct) is not bool:
            raise ValueError("old correctness must be Boolean or unknown")
        if (
            boolean(evaluation["behavior_change_within_H"], "evaluation change")
            != changed
        ):
            raise ValueError("evaluation/raw change contradiction")
        result.update(
            {
                "old_correct_at_h": int(old_correct is True),
                "old_wrong_at_h": int(old_correct is False),
                "old_model_absent": int(old_correct is None),
                "bounded_no_anomaly": int(
                    row["status"] == "bounded_monitoring_completed_no_anomaly"
                ),
                "detected_post_boundary": int(detected),
                "undetected_changed_at_h": int(changed and not detected),
                "false_or_early_anomaly": int(
                    boolean(
                        evaluation["anomaly_without_in_H_change_or_before_boundary"],
                        "false_or_early",
                    )
                ),
            }
        )
    else:
        eligible = boolean(row["eligible"], "eligible")
        tag = row["prefix_outcome"]
        if tag not in PREFIX_OUTCOMES or eligible != (tag == "eligible_tail"):
            raise ValueError("eligible/prefix-outcome contradiction")
        proposed = score["proposed_correct_at_H"]
        if proposed is not None and type(proposed) is not bool:
            raise ValueError("proposed correctness must be Boolean or unknown")
        prefix = integer(row["prefix_spent"], "prefix_spent")
        if prefix > result["actual_cost"]:
            raise ValueError("prefix cost exceeds actual total")
        rejected = row["status"] == "candidate_rejected"
        result.update(
            {
                "eligible_tail": int(eligible),
                "candidate_rejected": int(rejected),
                "rejected_correct_at_h": int(rejected and proposed is True),
                "rejected_wrong_at_h": int(rejected and proposed is False),
                "rejected_unscored": int(rejected and proposed is None),
                "bounded_no_anomaly_prefix": int(tag == "bounded_no_anomaly"),
                "actual_prefix_cost": prefix,
                "actual_tail_cost": result["actual_cost"] - prefix,
                "mutation_operations": integer(
                    row["mutation_operations"], "mutation_operations"
                ),
            }
        )
    return result


def index_rows(rows, selection, kind):
    if kind not in ("structural", "validation") or type(rows) is not list:
        raise ValueError("invalid analysis kind/rows")
    fids, groups = selection_check(selection)
    policies = (None,) if kind == "structural" else POLICIES
    expected = set(itertools.product(fids, MATCHED_METHODS, policies))
    index, metadata = {}, {}
    for row in rows:
        required = {
            "fixture_id",
            "method",
            "independent_group",
            "case",
            "renaming_index",
            "status",
            "spent",
            "raw_score",
        }
        required.update(
            {"evaluation", "construction_diagnostics"}
            if kind == "structural"
            else {
                "policy",
                "eligible",
                "prefix_outcome",
                "prefix_spent",
                "mutation_operations",
            }
        )
        if type(row) is not dict or not required <= set(row):
            raise ValueError("invalid result row")
        fid = row["fixture_id"]
        if (
            type(fid) is not str
            or type(row["method"]) is not str
            or (row.get("policy") is not None and type(row["policy"]) is not str)
        ):
            raise ValueError("invalid fixture/method/policy labels")
        key = (fid, row["method"], row.get("policy"))
        if (
            key not in expected
            or key in index
            or row["independent_group"] != groups[fid]
        ):
            raise ValueError(
                "missing/duplicate/unregistered row or wrong leakage group"
            )
        if (
            row["case"] not in CASES
            or type(row["renaming_index"]) is not int
            or row["renaming_index"] < 0
        ):
            raise ValueError("invalid case/renaming metadata")
        statuses = STRUCTURAL_STATUSES if kind == "structural" else VALIDATION_STATUSES
        if row["status"] not in statuses:
            raise ValueError("unknown outcome status")
        values = normalized(row, kind)
        facts = (row["case"], row["renaming_index"], values["changed_at_h"])
        if kind == "structural":
            diagnostic = row["construction_diagnostics"]
            if type(diagnostic) is not dict or not {
                "target_exact_in_public_family",
                "within_declared_mechanism_edit_family",
                "behavior_change_at_H",
            } <= set(diagnostic):
                raise ValueError("missing bounded construction context")
            if boolean(
                diagnostic["behavior_change_at_H"], "construction change"
            ) != bool(values["changed_at_h"]):
                raise ValueError("construction/raw behavior change mismatch")
            facts += (
                boolean(
                    diagnostic["target_exact_in_public_family"],
                    "prior behavior coverage",
                ),
                boolean(
                    diagnostic["within_declared_mechanism_edit_family"],
                    "construction one-edit family",
                ),
            )
        if metadata.setdefault(fid, facts) != facts:
            raise ValueError("paired fixture metadata differs between methods/policies")
        index[key] = values
    if set(index) != expected:
        raise ValueError("incomplete denominator; retain every method/policy/fixture")
    if kind == "validation":
        for fid, method in itertools.product(fids, MATCHED_METHODS):
            pair = [index[fid, method, policy] for policy in POLICIES]
            if any(
                pair[0][key] != pair[1][key]
                for key in (
                    "eligible_tail",
                    "actual_prefix_cost",
                    "bounded_no_anomaly_prefix",
                )
            ):
                raise ValueError("paid policy prefix knowledge/cost differs")
    return index, metadata


def quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[math.ceil(position)] * fraction
    )


def resampling_weights(n, resamples, seed):
    # One shared draw of entire paired group vectors per stratum. Count vectors
    # avoid repeating RNG/search work independently for every endpoint/contrast.
    rng = random.Random(f"group-analysis-v1:{seed}")
    weights = []
    for _ in range(resamples):
        counts = [0] * n
        for _ in range(n):
            counts[rng.randrange(n)] += 1
        weights.append(tuple(counts))
    return weights


def effect_range(values, weights):
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        raise ValueError("nonempty finite paired group values required")
    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return {
            "mean": mean,
            "groups": n,
            "exploratory_percentile_range": None,
            "reason": "one leakage group cannot estimate between-group uncertainty",
        }
    draws = sorted(
        math.fsum(value * count for value, count in zip(values, counts, strict=True))
        / n
        for counts in weights
    )
    return {
        "mean": mean,
        "groups": n,
        "exploratory_percentile_range": [
            quantile(draws, 0.025),
            quantile(draws, 0.975),
        ],
        "zero_observed_between_group_variation": all(v == values[0] for v in values),
        "scope": "paired group bootstrap percentiles; independence/exchangeability assumed, development-only, no multiplicity-adjusted significance or generalization certificate",
    }


def paired_range(values, *, resamples, seed):
    """Equal-group mean, NOT a fixture-iid or confirmatory confidence claim."""
    parameters(resamples, seed)
    if len(values) * resamples > MAX_OPERATIONS:
        raise ValueError("analysis resampling budget exceeded")
    return effect_range(
        values,
        resampling_weights(len(values), resamples, seed) if len(values) > 1 else None,
    )


def analyze(
    rows,
    selection,
    *,
    kind,
    source_rows=None,
    resamples=DEFAULT_RESAMPLES,
    seed=DEFAULT_SEED,
):
    parameters(resamples, seed)
    index, metadata = index_rows(rows, selection, kind)
    if kind == "validation":
        if source_rows is None:
            raise ValueError(
                "validation context requires complete audited structural rows"
            )
        _, source_metadata = index_rows(source_rows, selection, "structural")
        if any(metadata[fid] != source_metadata[fid][:3] for fid in metadata):
            raise ValueError("validation/source construction context mismatch")
        metadata = source_metadata
    elif source_rows is not None:
        raise ValueError("structural context is already in its audited rows")
    fids, assignments = selection_check(selection)
    labels = (
        list(MATCHED_METHODS)
        if kind == "structural"
        else [
            f"{method}/{policy}"
            for method, policy in itertools.product(MATCHED_METHODS, POLICIES)
        ]
    )
    label_keys = {
        label: (label, None) if kind == "structural" else tuple(label.split("/"))
        for label in labels
    }
    comparisons = (
        list(itertools.combinations(labels, 2))
        if kind == "structural"
        else [(f"{m}/{POLICIES[1]}", f"{m}/{POLICIES[0]}") for m in MATCHED_METHODS]
    )
    metrics = tuple(next(iter(index.values())))
    slices = {"all": fids}
    slices.update(
        {
            f"case/{case}": [fid for fid in fids if metadata[fid][0] == case]
            for case in CASES
        }
    )
    slices.update(
        {
            f"change_at_h/{flag}": [fid for fid in fids if metadata[fid][2] == flag]
            for flag in (0, 1)
        }
    )
    slices.update(
        {
            f"renaming/{flag}": [fid for fid in fids if metadata[fid][1] == flag]
            for flag in sorted({0, 1} | {metadata[fid][1] for fid in fids})
        }
    )
    slices.update(
        {
            f"target_behavior_in_public_family/{flag}": [
                fid for fid in fids if metadata[fid][3] == bool(flag)
            ]
            for flag in (0, 1)
        }
    )
    slices.update(
        {
            f"construction_one_edit_family/{flag}": [
                fid for fid in fids if metadata[fid][4] == bool(flag)
            ]
            for flag in (0, 1)
        }
    )
    operations = (
        sum(len({assignments[f] for f in subset}) for subset in slices.values())
        * resamples
        * len(comparisons)
        * len(metrics)
    )
    if operations > MAX_OPERATIONS:
        raise ValueError(
            "analysis resampling budget exceeded; never trim cases or groups"
        )
    output = {}
    for slice_name, subset in slices.items():
        groups = {
            group: [fid for fid in subset if assignments[fid] == group]
            for group in sorted({assignments[f] for f in subset})
        }
        weights = (
            resampling_weights(len(groups), resamples, seed)
            if len(groups) > 1
            else None
        )
        group_values, cells = {}, {}
        for label in labels:
            method, policy = label_keys[label]
            group_values[label] = {
                group: {
                    metric: math.fsum(
                        index[fid, method, policy][metric] for fid in members
                    )
                    / len(members)
                    for metric in metrics
                }
                for group, members in groups.items()
            }
            cells[label] = {
                "fixtures": len(subset),
                "leakage_groups": len(groups),
                "fixture_totals": {
                    metric: sum(index[fid, method, policy][metric] for fid in subset)
                    for metric in metrics
                },
                "fixture_weighted_mean": {
                    metric: math.fsum(
                        index[fid, method, policy][metric] for fid in subset
                    )
                    / len(subset)
                    if subset
                    else None
                    for metric in metrics
                },
                "equal_group_mean": {
                    metric: math.fsum(group_values[label][g][metric] for g in groups)
                    / len(groups)
                    if groups
                    else None
                    for metric in metrics
                },
            }
        contrasts = []
        for left, right in comparisons:
            differences = {
                group: {
                    metric: group_values[left][group][metric]
                    - group_values[right][group][metric]
                    for metric in metrics
                }
                for group in groups
            }
            contrasts.append(
                {
                    "left": left,
                    "right": right,
                    "sign": "left minus right; not an automatic improvement score",
                    "paired_group_differences": differences,
                    "metrics": {
                        metric: effect_range(
                            [differences[group][metric] for group in groups], weights
                        )
                        if groups
                        else {
                            "mean": None,
                            "groups": 0,
                            "exploratory_percentile_range": None,
                            "reason": "no registered instances in stratum",
                        }
                        for metric in metrics
                    },
                }
            )
        output[slice_name] = {
            "fixture_ids": list(subset),
            "group_fixture_counts": {
                group: len(members) for group, members in groups.items()
            },
            "cells": cells,
            "contrasts": contrasts,
        }
    return {
        "schema": "paired-leakage-group-development-analysis-v1",
        "kind": kind,
        "fixtures": len(fids),
        "leakage_groups": len(set(assignments.values())),
        "rows": len(rows),
        "resamples": resamples,
        "seed": seed,
        "resampling_operation_bound": operations,
        "strata": output,
        "interpretation": {
            "unit": "registered leakage group, equal weight after averaging paired fixtures within each group",
            "unresolved_and_ineligible": "retained in full denominators; absence of promotion is not correct theory recovery",
            "changed_at_h": "private post-freeze bounded issued behavior, not uniquely identifiable physical factors",
            "undetected_changed_at_h": "all changed targets without detected post-boundary anomaly; not necessarily a change inside completed monitoring window",
            "costs": "actual paid reset/actions/validation, not inferred full computation or API cost",
            "resampling": "exploratory development diagnostics, assumes group independence/exchangeability; leakage grouping is not proof of iid sampling; no formal p-values or novelty claim",
            "validation": "validation-policy ablation, inherited discovery search not reexecuted; candidate rejections may arise in common Wp prefix",
            "prior_behavior_coverage": "unbounded deterministic reset behavioral equivalence to a public old joint candidate, not syntax/physical structure identity",
            "construction_one_edit_family": "generator's independent one-edit-family diagnostic, not proof of complete learner DSL coverage",
        },
    }
