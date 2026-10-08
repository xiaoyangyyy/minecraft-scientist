"""Private-read ordering and shared-serialization fault counterexamples."""

import copy
import itertools
import json
from pathlib import Path

import pytest
from test_active_automata import counter_program, view_for

import minecraft_scientist.discovery.active_automata as learner
from minecraft_scientist.evaluation.active_automata import audit_apartness
from minecraft_scientist.theory.finite_state import FiniteStateProgram


def bundle(states=2, **kwargs):
    program = counter_program(states=states)
    config = {
        "schema": "apartness-config-v1",
        "actions": list(program.actions),
        "outputs": list(program.outputs),
        "costs": {a: 1 for a in program.actions},
        "protocol": {
            "budget": 1000,
            "horizon": 7,
            "epoch": "new",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
        }
        | kwargs,
    }
    result = learner.learn_apartness(
        view_for(program),
        program.actions,
        program.outputs,
        config["costs"],
        **config["protocol"],
    )
    return config, result, program.model_dump(mode="json")


@pytest.fixture(scope="module")
def good_bundle():
    return bundle()


@pytest.mark.parametrize("states", [1, 2, 3, 4])
def test_public_fold_audit_and_private_bounded_behavior_check(states, tmp_path):
    config, result, truth = bundle(states=states, horizon=8, budget=3000)
    private = tmp_path / "truth.json"
    private.write_text(json.dumps(truth), encoding="utf-8")
    audit = audit_apartness(config, result, private)
    assert audit["valid"], audit
    assert audit["public_replay_valid"] and audit["private_truth_opened"]
    if result["status"] == "provisional_conformance_passed":
        assert audit["evaluation"]["provisional_behavior_correct_at_H"]


@pytest.mark.parametrize(
    "attack",
    [
        "budget",
        "epoch",
        "tree",
        "basis",
        "scope_claim",
        "candidate",
        "plan",
        "receipt",
        "reset_episode",
        "reset_tick",
        "step_episode",
        "state_signal",
        "last_event",
        "delta",
        "duplicate_receipt",
        "public_alphabet",
        "numeric_type",
        "cost_type",
    ],
)
def test_forged_public_outputs_never_open_private_truth(
    good_bundle, monkeypatch, attack
):
    config, result, _truth = copy.deepcopy(good_bundle)
    if attack == "budget":
        result["spent"] += 1
    elif attack == "epoch":
        result["observation_tree"]["epoch"] = "old"
    elif attack == "tree":
        result["observation_tree"]["nodes"][-1]["evidence_ids"] = ["forged"]
    elif attack == "basis":
        result["basis"][-1].append("b")
    elif attack == "scope_claim":
        result["all_allowed_queries_certified"] = True
    elif attack == "candidate":
        result["candidate_ir"]["parameters"]["transitions"][0]["target"] = "forged"
    elif attack == "plan":
        result["conformance"][0]["plan"]["total_cost"] = 0
    elif attack == "receipt":
        next(e for e in result["events"] if e["kind"] == "step")["transition"][
            "state_after"
        ]["step"] = 999
    elif attack == "numeric_type":
        next(e for e in result["events"] if e["kind"] == "step")["transition"][
            "state_after"
        ]["step"] = True
    elif attack == "cost_type":
        next(e for e in result["events"] if e["kind"] == "reset")["cost"] = True
    elif attack in ("reset_episode", "reset_tick", "public_alphabet"):
        observation = next(e for e in result["events"] if e["kind"] == "reset")[
            "observation"
        ]
        if attack == "reset_episode":
            observation["episode_id"] = 99
        elif attack == "reset_tick":
            observation["step"] = 1
        else:
            observation["objects"][0]["object_id"] = "forged"
    else:
        steps = [e["transition"] for e in result["events"] if e["kind"] == "step"]
        if attack == "step_episode":
            steps[0]["state_after"]["episode_id"] = 99
        elif attack == "state_signal":
            steps[0]["state_after"]["agent_state"]["signal"] = "1"
        elif attack == "last_event":
            steps[0]["state_after"]["last_event"] = None
        elif attack == "delta":
            steps[0]["observed_delta"] = {"signal": "1"}
        elif attack == "duplicate_receipt":
            steps[1]["transition_id"] = steps[0]["transition_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("private read before public rejection")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_apartness(config, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["evaluation"] is None


def test_shared_ir_serializer_bug_is_caught_by_raw_frozen_prediction_check(monkeypatch):
    original = learner.TreeHypothesis.to_ir

    def broken(self, horizon):
        ir = original(self, horizon)
        row = ir.parameters["transitions"][0]
        row["output"] = next(o for o in ir.parameters["outputs"] if o != row["output"])
        return ir

    monkeypatch.setattr(learner.TreeHypothesis, "to_ir", broken)
    config, result, _truth = bundle(states=1)
    assert result["status"] == "provisional_conformance_passed"

    def forbidden(*args, **kwargs):
        raise AssertionError("private read despite shared serializer error")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_apartness(config, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]
    assert audit["failure"] == "independent conformance frozen prediction mismatch"


@pytest.mark.parametrize(
    "fault", ["horizon", "reset", "effects", "preconditions", "alphabet"]
)
def test_shared_serializer_scope_or_contract_fault_is_rejected_before_private(
    monkeypatch, fault
):
    original = learner.TreeHypothesis.to_ir

    def broken(self, horizon):
        ir = original(self, horizon)
        if fault == "horizon":
            ir.scope["intervention_horizon"] = 999
        elif fault == "reset":
            ir.scope["certified_reset_only"] = False
        elif fault == "effects":
            ir.effects["output"] = "invented"
        elif fault == "preconditions":
            ir = ir.model_copy(update={"preconditions": ("unverified",)})
        else:
            ir.parameters["outputs"].append("unregistered")
        return ir

    monkeypatch.setattr(learner.TreeHypothesis, "to_ir", broken)
    config, result, _truth = bundle(states=1)
    assert result["status"] == "provisional_conformance_passed"

    def forbidden(*args, **kwargs):
        raise AssertionError("private read despite shared scope/contract error")

    monkeypatch.setattr(Path, "read_text", forbidden)
    audit = audit_apartness(config, result, Path("never-open-private.json"))
    assert not audit["valid"] and not audit["private_truth_opened"]


def test_false_private_world_rejected_only_after_valid_public_replay(
    good_bundle, tmp_path
):
    config, result, truth = copy.deepcopy(good_bundle)
    truth["transitions"][0]["output"] = "1"
    private = tmp_path / "wrong-truth.json"
    private.write_text(json.dumps(truth), encoding="utf-8")
    audit = audit_apartness(config, result, private)
    assert audit["public_replay_valid"] and audit["private_truth_opened"]
    assert not audit["valid"]
    assert not audit["evaluation"]["actual_controls_match_private_reference"]


@pytest.mark.parametrize(
    "kwargs", [{"budget": 1}, {"budget": 12}, {"operation_budget": 1}]
)
def test_resource_exhaustion_is_auditable_not_promoted(tmp_path, kwargs):
    config, result, truth = bundle(**kwargs)
    private = tmp_path / "truth.json"
    private.write_text(json.dumps(truth), encoding="utf-8")
    audit = audit_apartness(config, result, private)
    assert audit["valid"], audit
    assert result["status"] == "unresolved" and result["candidate_ir"] is None


def test_a_genuine_wrong_provisional_model_is_scored_not_hidden_as_audit_failure(
    tmp_path,
):
    # Pick a real untested H=5 word from the frozen one-state test protocol.
    base = counter_program(states=1)
    tree = learner.ObservationTree(base.actions, base.outputs, epoch="new")
    for action in base.actions:
        tree.add_trace((action,), ("0",), epoch="new", evidence_id=action)
    hypothesis = learner.fold_hypothesis(tree, ((),))
    plan = learner.conformance_plan(
        tree, hypothesis, {"a": 1, "b": 1}, horizon=5, count=8, seed=1
    )
    tested = {tuple(item["actions"]) for item in plan["queries"]}
    secret = next(
        w for w in itertools.product(base.actions, repeat=5) if w not in tested
    )
    names = tuple(f"q{i}" for i in range(6))
    truth = FiniteStateProgram(
        states=names,
        actions=base.actions,
        outputs=base.outputs,
        initial_state="q0",
        transitions=tuple(
            {
                "source": state,
                "action": action,
                "target": names[i + 1] if i < 5 and action == secret[i] else "q5",
                "output": "1" if i == 4 and action == secret[i] else "0",
            }
            for i, state in enumerate(names)
            for action in base.actions
        ),
    )
    config = {
        "schema": "apartness-config-v1",
        "actions": list(base.actions),
        "outputs": list(base.outputs),
        "costs": {"a": 1, "b": 1},
        "protocol": {
            "budget": 240,
            "horizon": 5,
            "epoch": "new",
            "seed": 1,
            "conformance_count": 8,
            "operation_budget": 200_000,
        },
    }
    result = learner.learn_apartness(
        view_for(truth),
        truth.actions,
        truth.outputs,
        config["costs"],
        **config["protocol"],
    )
    assert result["status"] == "provisional_conformance_passed"
    assert not result["all_allowed_queries_certified"]
    private = tmp_path / "truth.json"
    private.write_text(truth.model_dump_json(), encoding="utf-8")
    audit = audit_apartness(config, result, private)
    assert audit["valid"], audit
    assert audit["evaluation"]["actual_controls_match_private_reference"]
    assert audit["evaluation"]["provisional_behavior_correct_at_H"] is False
