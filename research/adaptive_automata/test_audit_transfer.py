"""Whole-campaign private gate, cost and version/history mutation negatives."""

import copy

import pytest
from audit_transfer import audit_transfer
from paid_transfer import transfer_contexts
from test_audit_run import PrivateProbe
from test_matching import counter, view


def bundle(method="adaptive", **protocol):
    old, new = counter(2), counter(3)
    config = {
        "schema": "paid-transfer-config-v1",
        "method": method,
        "actions": list(old.actions),
        "outputs": list(old.outputs),
        "costs": {a: 1 for a in old.actions},
        "protocol": {
            "budget": 400,
            "old_budget": 120,
            "horizon": 8,
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        }
        | protocol,
    }
    result = transfer_contexts(
        view(old),
        view(new),
        old.actions,
        old.outputs,
        config["costs"],
        method=method,
        **config["protocol"],
    )
    return config, result, old.model_dump(mode="json"), new.model_dump(mode="json")


@pytest.fixture(scope="module")
def good_bundle():
    return bundle()


@pytest.mark.parametrize("method", ["from_scratch", "adaptive"])
def test_paid_campaign_public_replay_then_both_private_scores(method):
    config, result, old, new = bundle(method)
    first, second = PrivateProbe(old), PrivateProbe(new)
    audit = audit_transfer(config, result, first, second)
    assert audit["valid"] and audit["all_contexts_public_valid"], audit
    assert first.calls == second.calls == 1
    assert audit["evaluation"]["both_contexts_correct_at_H"]


@pytest.mark.parametrize(
    "attack",
    [
        "new_receipt",
        "old_receipt",
        "total_cost",
        "new_budget",
        "old_budget",
        "source_theory",
        "version",
        "parent",
        "scope",
        "old_evidence",
        "validation_overlap",
        "validation_free",
        "mode",
        "old_phase_prior",
        "online_claim",
        "status",
    ],
)
def test_any_context_or_lineage_mutation_rejects_before_either_private_read(
    good_bundle, attack
):
    config, result, _, _ = copy.deepcopy(good_bundle)
    if attack in ("new_receipt", "old_receipt"):
        phase = "new" if attack == "new_receipt" else "old"
        next(e for e in result[phase]["run"]["events"] if e["kind"] == "step")[
            "cost"
        ] = True
    elif attack == "total_cost":
        result["spent"] -= result["old"]["run"]["spent"]
    elif attack == "new_budget":
        result["new"]["config"]["protocol"]["budget"] = config["protocol"]["budget"]
    elif attack == "old_budget":
        result["old_budget_cap"] = True
    elif attack == "source_theory":
        result["public_old_theory"]["ir"]["parameters"]["transitions"][0]["output"] = (
            "forged"
        )
    elif attack == "version":
        result["theory_events"][-1]["version"] = 1
    elif attack == "parent":
        result["theory_events"][-1]["parent_version"] = None
    elif attack == "scope":
        result["theory_events"][-1]["ir"]["scope"]["context_id"] = "old-context"
    elif attack == "old_evidence":
        result["theory_events"][-1]["evidence_ids"].append(
            result["theory_events"][0]["evidence_ids"][0]
        )
    elif attack == "validation_overlap":
        result["validation_assignments"][-1]["discovery_ids"].append(
            result["validation_assignments"][-1]["validation_ids"][0]
        )
    elif attack == "validation_free":
        result["validation_assignments"][-1]["validation_ids"] = []
    elif attack == "mode":
        config["method"] = "from_scratch"
    elif attack == "old_phase_prior":
        result["old"]["config"]["references"] = result["new"]["config"]["references"]
    elif attack == "online_claim":
        result["unannounced_online_mutation_claimed"] = True
    else:
        result["theory_events"][2]["status"] = "superseded"
    first, second = PrivateProbe(), PrivateProbe()
    audit = audit_transfer(config, result, first, second)
    assert (
        not audit["valid"]
        and not audit["private_truth_read_attempted"]
        and first.calls == second.calls == 0
    ), audit


@pytest.mark.parametrize("budget", [1, 50])
def test_unresolved_acquisition_is_a_valid_paid_campaign_not_free_transfer(budget):
    config, result, old, new = bundle(budget=budget)
    assert result["status"] == "unresolved"
    first, second = PrivateProbe(old), PrivateProbe(new)
    audit = audit_transfer(config, result, first, second)
    assert (
        audit["valid"] and audit["evaluation"]["both_contexts_correct_at_H"] is None
    ), audit
    assert second.calls == 0 if result["new"] is None else second.calls == 1


def test_owned_audit_snapshot_prevents_first_reader_mutating_second_public_context(
    good_bundle,
):
    config, result, old, new = copy.deepcopy(good_bundle)

    class FirstReader(PrivateProbe):
        def read_text(self, **kwargs):
            result["new"]["run"]["spent"] = -999
            return super().read_text(**kwargs)

    audit = audit_transfer(config, result, FirstReader(old), PrivateProbe(new))
    assert audit["valid"], audit
    assert result["new"]["run"]["spent"] == -999
