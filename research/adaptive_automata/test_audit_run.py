"""Private-I/O gate, raw public chronology, policy replay and control negatives."""

import copy
import itertools
import json

import learner
import pytest
from audit_run import audit_adaptive
from test_matching import counter, view

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.active_automata import automata_task
from minecraft_scientist.world.identification import _FiniteWorld


class PrivateProbe:
    def __init__(self, truth=None):
        self.truth, self.calls = truth, 0

    def read_text(self, *, encoding):
        assert encoding == "utf-8"
        self.calls += 1
        assert self.truth is not None, "private read before public rejection"
        return json.dumps(self.truth)


def bundle(states=2, reference_states=None, shift=False, capability=None, **protocol):
    truth = counter(states, initial=int(shift and states > 1))
    reference = counter(states if reference_states is None else reference_states)
    config = {
        "schema": "adaptive-config-v1",
        "actions": list(truth.actions),
        "outputs": list(truth.outputs),
        "costs": {a: 1 for a in truth.actions},
        "references": [reference.to_ir().model_dump(mode="json")],
        "protocol": {
            "budget": 3000,
            "horizon": 8,
            "epoch": "new",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        }
        | protocol,
    }
    result = learner.learn_adaptive(
        capability or view(truth),
        truth.actions,
        truth.outputs,
        config["costs"],
        (reference.to_ir(),),
        **config["protocol"],
    )
    return config, result, truth.model_dump(mode="json")


@pytest.fixture(scope="module")
def good_bundle():
    return bundle()


@pytest.mark.parametrize("states", [1, 2, 3, 4])
@pytest.mark.parametrize("shift", [False, True])
def test_complete_public_policy_audit_then_raw_private_behavior(states, shift):
    config, result, truth = bundle(states=states, shift=shift)
    private = PrivateProbe(truth)
    audit = audit_adaptive(config, result, private)
    assert audit["valid"], audit
    assert (
        audit["raw_public_ledger_valid"]
        and audit["public_replay_valid"]
        and private.calls == 1
    )
    assert audit["evaluation"]["provisional_behavior_correct_at_H"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 1},
        {"horizon": 1},
        {"operation_budget": 1},
        {"reference_operation_budget": 1},
        {"budget": 18},
    ],
)
def test_resource_exhaustion_is_valid_unresolved_and_not_promoted(kwargs):
    config, result, truth = bundle(**kwargs)
    assert result["status"] == "unresolved" and result["candidate_ir"] is None
    audit = audit_adaptive(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert audit["evaluation"]["provisional_behavior_correct_at_H"] is None


@pytest.mark.parametrize(
    "attack",
    [
        "cost",
        "bool_cost",
        "query_order",
        "evidence",
        "epoch",
        "ref_digest",
        "guidance",
        "promotion",
        "hypothesis",
        "sampler_prediction",
        "fake_pass",
        "scope",
        "basis",
        "reset_tick",
        "step_signal",
        "delta",
        "last_event",
        "transition_id",
        "intervention",
        "extra_event",
        "drop_event",
        "ledger",
        "reference_prior",
        "protocol",
        "rebuilding",
    ],
)
def test_forged_records_never_open_private_truth(good_bundle, attack):
    config, result, _ = copy.deepcopy(good_bundle)
    events = result["events"]
    step = next(e for e in events if e["kind"] == "step")
    if attack == "cost":
        step["cost"] += 1
    elif attack == "bool_cost":
        step["cost"] = True
    elif attack == "query_order":
        next(e for e in events if e["kind"] == "plan")["actions"].reverse()
        next(e for e in events if e["kind"] == "plan")["query_id"] = "adaptive-query-9"
    elif attack == "evidence":
        next(e for e in events if e["kind"] == "evidence")["outputs"].pop()
    elif attack in ("epoch", "ref_digest", "guidance"):
        plan = next(e for e in events if e["kind"] == "reference_plan")["plan"]
        if attack == "guidance":
            plan["guidance"][2][0] = False
        else:
            plan["epoch" if attack == "epoch" else "reference_digest"] = "forged"
    elif attack == "promotion":
        next(e for e in events if e["kind"] == "basis_promotion")["access"] = ["b"]
    elif attack == "hypothesis":
        next(e for e in events if e["kind"] == "hypothesis_plan")["candidate_ir"][
            "parameters"
        ]["transitions"][0]["output"] = "1"
    elif attack == "sampler_prediction":
        next(e for e in events if e["kind"] == "conformance_plan")["plan"]["queries"][
            0
        ]["prediction"][0] = "forged"
    elif attack == "fake_pass":
        result["conformance"][-1]["status"] = "unresolved"
    elif attack == "scope":
        result["all_allowed_queries_certified"] = True
    elif attack == "basis":
        result["basis"][-1] = ["b"]
    elif attack == "reset_tick":
        next(e for e in events if e["kind"] == "reset")["observation"]["step"] = True
    elif attack == "step_signal":
        step["transition"]["state_after"]["agent_state"]["signal"] = "forged"
    elif attack == "delta":
        step["transition"]["observed_delta"] = {}
    elif attack == "last_event":
        step["transition"]["state_after"]["last_event"] = None
    elif attack == "transition_id":
        steps = [e for e in events if e["kind"] == "step"]
        steps[1]["transition"]["transition_id"] = steps[0]["transition"][
            "transition_id"
        ]
    elif attack == "intervention":
        step["transition"]["intended_intervention"]["controlled_variables"]["reset"] = 1
    elif attack == "extra_event":
        events.append({"kind": "private_oracle"})
    elif attack == "drop_event":
        events.remove(next(e for e in events if e["kind"] == "hypothesis_plan"))
    elif attack == "ledger":
        result["spent"] += 1
    elif attack == "reference_prior":
        config["references"][0]["parameters"]["transitions"][0]["output"] = "1"
    elif attack == "protocol":
        config["protocol"]["seed"] += 1
    else:
        events.append(
            copy.deepcopy(next(e for e in events if e["kind"] == "rebuilding_closed"))
        )
    private = PrivateProbe()
    audit = audit_adaptive(config, result, private)
    assert (
        not audit["valid"] and not audit["private_truth_opened"] and private.calls == 0
    ), audit


@pytest.mark.parametrize("fault", ["output", "target", "scope", "bool_scope"])
def test_shared_ir_serializer_defects_do_not_survive_raw_fold_gate(monkeypatch, fault):
    # to_ir is shared by run construction and policy replay; raw checks must not
    # merely agree with that serializer's corrupted output.
    from minecraft_scientist.discovery.active_automata import TreeHypothesis

    original_ir = TreeHypothesis.to_ir

    def bad_ir(self, horizon):
        ir = original_ir(self, horizon)
        if fault == "output":
            row = ir.parameters["transitions"][0]
            row["output"] = next(
                o for o in ir.parameters["outputs"] if o != row["output"]
            )
        elif fault == "target":
            row = ir.parameters["transitions"][0]
            row["target"] = next(
                s for s in ir.parameters["states"] if s != row["target"]
            )
        elif fault == "scope":
            ir.scope["intervention_horizon"] = 99
        else:
            ir.scope["certified_reset_only"] = 1
        return ir

    monkeypatch.setattr(TreeHypothesis, "to_ir", bad_ir)
    config, result, _ = bundle()
    private = PrivateProbe()
    audit = audit_adaptive(config, result, private)
    assert not audit["valid"] and private.calls == 0, audit


def fault_view(program, *, at_step, failure):
    world, calls = _FiniteWorld(program), 0

    def step(action):
        nonlocal calls
        record = world.step(action)
        calls += 1
        if calls != at_step:
            return record
        if failure == "compliance":
            return record.model_copy(update={"intervention_compliance": False})
        if failure == "controller":
            return record.model_copy(
                update={"controller_failure": "declared_terminal_failure"}
            )
        if failure == "ambiguous":
            return record.model_copy(
                update={"possible_confounders": ("unknown_delivery",)}
            )
        event = record.event.model_copy(update={"success": False})
        return record.model_copy(
            update={
                "event": event,
                "state_after": record.state_after.model_copy(
                    update={"last_event": event}
                ),
            }
        )

    return AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=step,
        begin_trial=world.begin_trial,
    )


@pytest.mark.parametrize("failure", ["compliance", "controller", "event"])
@pytest.mark.parametrize("at_step", [1, 2, 3])
def test_known_failed_dispatch_is_charged_not_retained_and_valid_unresolved(
    failure, at_step
):
    config, result, truth = bundle(
        capability=fault_view(counter(2), at_step=at_step, failure=failure)
    )
    assert result["stop_reason"] == "control_failure" and result["candidate_ir"] is None
    failed_qid = result["events"][-1]["query_id"]
    assert not any(
        e["kind"] == "evidence" and e["query_id"] == failed_qid
        for e in result["events"]
    )
    assert result["spent"] == sum(
        e["cost"] for e in result["events"] if e["kind"] in ("reset", "step")
    )
    audit = audit_adaptive(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert audit["evaluation"]["known_failed_dispatches_excluded"] == 1


def test_ambiguous_delivery_cannot_be_claimed_as_clean_stationary_protocol():
    config, result, _ = bundle(
        capability=fault_view(counter(2), at_step=1, failure="ambiguous")
    )
    audit = audit_adaptive(config, result, PrivateProbe())
    assert not audit["valid"] and not audit["private_truth_opened"]


@pytest.mark.parametrize("family", ["dense", "sparse", "cycle", "branch"])
@pytest.mark.parametrize("states", [3, 5])
def test_non_counter_anonymous_programs_are_auditable_even_if_unresolved_or_wrong(
    family, states
):
    task, reference = automata_task(0, family, states), automata_task(0, "cycle", 2)
    config = {
        "schema": "adaptive-config-v1",
        "actions": list(task.actions),
        "outputs": list(task.outputs),
        "costs": task.costs,
        "references": [reference.ground_truth.to_ir().model_dump(mode="json")],
        "protocol": {
            "budget": 1000,
            "horizon": 8,
            "epoch": "new",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        },
    }
    result = learner.learn_adaptive(
        task.view,
        task.actions,
        task.outputs,
        task.costs,
        (reference.ground_truth.to_ir(),),
        **config["protocol"],
    )
    audit = audit_adaptive(
        config, result, PrivateProbe(task.ground_truth.model_dump(mode="json"))
    )
    assert audit["valid"], audit
    if result["status"] == "unresolved":
        assert audit["evaluation"]["provisional_behavior_correct_at_H"] is None
    if family == "branch" and states == 5:
        # This declared fixture actually executes both consistency and paid
        # binary conformance-counterexample processing, not just a good path.
        assert any(r["rule"] == "consistency_counterexample" for r in result["rules"])
        assert any(
            e["kind"] == "plan" and e["reason"] == "counterexample"
            for e in result["events"]
        )


@pytest.mark.parametrize("position", [1, 2])
def test_conformance_control_failure_keeps_partial_cost_but_no_passing_batch(
    good_bundle, position
):
    _config, good, _truth = good_bundle
    plans = {e["query_id"]: e for e in good["events"] if e["kind"] == "plan"}
    before = 0
    for e in good["events"]:
        if e["kind"] == "step":
            if plans[e["query_id"]]["reason"] == "conformance":
                break
            before += 1
    config, result, truth = bundle(
        capability=fault_view(
            counter(2), at_step=before + position, failure="controller"
        )
    )
    assert result["stop_reason"] == "control_failure" and result["candidate_ir"] is None
    assert (
        result["conformance"][-1]["status"] == "unresolved"
        and result["conformance"][-1]["query_ids"] == []
    )
    audit = audit_adaptive(config, result, PrivateProbe(truth))
    assert (
        audit["valid"] and audit["evaluation"]["known_failed_dispatches_excluded"] == 1
    ), audit


def test_valid_protocol_can_make_a_wrong_provisional_proposal_and_is_scored_not_hidden():
    base = FiniteStateProgram(
        states=("root",),
        actions=("a", "b"),
        outputs=("0", "1"),
        initial_state="root",
        transitions=tuple(
            {"source": "root", "action": a, "target": "root", "output": "0"}
            for a in ("a", "b")
        ),
    )
    config = {
        "schema": "adaptive-config-v1",
        "actions": list(base.actions),
        "outputs": list(base.outputs),
        "costs": {a: 1 for a in base.actions},
        "references": [base.to_ir().model_dump(mode="json")],
        "protocol": {
            "budget": 240,
            "horizon": 5,
            "epoch": "new",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        },
    }
    known = learner.learn_adaptive(
        view(base),
        base.actions,
        base.outputs,
        config["costs"],
        (base.to_ir(),),
        **config["protocol"],
    )
    paid = [tuple(e["actions"]) for e in known["events"] if e["kind"] == "plan"]
    secret = next(
        w
        for w in itertools.product(base.actions, repeat=5)
        if all(p[:5] != w for p in paid)
    )
    names = tuple(f"q{i}" for i in range(6))
    truth = FiniteStateProgram(
        states=names,
        actions=base.actions,
        outputs=base.outputs,
        initial_state=names[0],
        transitions=tuple(
            {
                "source": s,
                "action": a,
                "target": names[i + 1] if i < 5 and a == secret[i] else names[5],
                "output": "1" if i == 4 and a == secret[i] else "0",
            }
            for i, s in enumerate(names)
            for a in base.actions
        ),
    )
    result = learner.learn_adaptive(
        view(truth),
        base.actions,
        base.outputs,
        config["costs"],
        (base.to_ir(),),
        **config["protocol"],
    )
    assert result == known and result["status"] == "provisional_conformance_passed"
    private = PrivateProbe(truth.model_dump(mode="json"))
    audit = audit_adaptive(config, result, private)
    assert audit["valid"] and audit["public_replay_valid"] and private.calls == 1, audit
    assert audit["evaluation"]["provisional_behavior_correct_at_H"] is False


def test_audit_does_not_mutate_caller_result_or_configuration(good_bundle):
    config, result, truth = copy.deepcopy(good_bundle)
    before = copy.deepcopy((config, result))
    assert audit_adaptive(config, result, PrivateProbe(truth))["valid"]
    assert (config, result) == before


def test_shared_query_cache_defect_is_rejected_before_private(monkeypatch):
    original = learner.OutputQueries.__call__

    def bad_query(self, word, reason, hypothesis=None, *, fresh=False):
        return original(self, word, reason, hypothesis, fresh=fresh or reason == "R")

    monkeypatch.setattr(learner.OutputQueries, "__call__", bad_query)
    config, result, _ = bundle()
    # The shared learner and replay both overpay and call this a real R query.
    # Raw ledger independently knows its shorter word was already acquired.
    audit = audit_adaptive(config, result, PrivateProbe())
    assert not audit["valid"] and not audit["private_truth_read_attempted"]
    assert "cached membership" in audit["failure"]


def test_wrong_private_reference_is_rejected_only_after_public_checks(good_bundle):
    config, result, truth = copy.deepcopy(good_bundle)
    truth["transitions"][0]["output"] = "1"
    private = PrivateProbe(truth)
    audit = audit_adaptive(config, result, private)
    assert (
        audit["raw_public_ledger_valid"]
        and audit["public_replay_valid"]
        and private.calls == 1
    )
    assert (
        not audit["valid"]
        and not audit["evaluation"]["successful_controls_match_private_reference"]
    )


def test_failed_private_io_distinguishes_attempt_from_successful_read(good_bundle):
    config, result, _ = copy.deepcopy(good_bundle)

    class MissingPrivate:
        def read_text(self, **kwargs):
            raise FileNotFoundError("declared missing evaluator artifact")

    audit = audit_adaptive(config, result, MissingPrivate())
    assert audit["public_replay_valid"] and audit["private_truth_read_attempted"]
    assert not audit["private_truth_opened"] and not audit["valid"]
