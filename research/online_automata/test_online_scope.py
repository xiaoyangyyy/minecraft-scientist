"""Known issued/delivered distinction, failed control and shared-fault checks."""

import copy

import online
import pytest
from audit_online import audit_online
from online import online_relearn
from test_online_audit import PrivateProbe, bundle
from worlds import online_task

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.world.revision import revision_task


@pytest.mark.parametrize(
    "case", ["unchanged", "controller_fault", "redirect_start", "state_insert"]
)
@pytest.mark.parametrize("method", ["from_scratch", "adaptive"])
def test_existing_h3_worlds_are_scored_as_issued_behavior_not_factor_attribution(
    case, method
):
    task = revision_task(0, case, boundary=25)
    config = {
        "schema": "online-relearning-config-v1",
        "method": method,
        "actions": list(task.costs),
        "outputs": list(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        "costs": task.costs,
        "protocol": {
            "budget": 600,
            "old_budget": 250,
            "horizon": 5,
            "seed": 20261008,
            "conformance_count": 8,
            "monitor_rounds": 4,
            "monitor_count": 8,
            "operation_budget": 200_000,
            "reference_operation_budget": 200_000,
        },
    }
    result = online_relearn(
        task.view,
        tuple(config["actions"]),
        tuple(config["outputs"]),
        config["costs"],
        method=method,
        **config["protocol"],
    )
    audit = audit_online(config, result, PrivateProbe(task.private_spec))
    assert audit["valid"], audit
    assert not result["mechanism_change_warranted"]
    assert result["mechanism_attribution_status"] == "unresolved"
    for e in result["old"]["run"]["events"]:
        if e["kind"] == "step":
            assert e["transition"]["possible_confounders"] == [
                "unverified-actuator-delivery"
            ]
            assert e["transition"]["context"]["treatment_verification"] == "unknown"


def test_known_monitor_dispatch_failure_is_paid_and_never_challenges_world_theory():
    task = online_task(0, "unchanged")
    config, _, truth = bundle(case="unchanged")
    episode = -1

    def reset():
        nonlocal episode
        observation = task.view.begin_trial()
        episode = observation.episode_id
        return observation

    def step(action):
        record = task.view.step(action)
        if episode < 15:
            return record
        event = record.event.model_copy(update={"success": False})
        return record.model_copy(
            update={
                "controller_failure": "known-dispatch-failure",
                "event": event,
                "state_after": record.state_after.model_copy(
                    update={"last_event": event}
                ),
            }
        )

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=step,
        begin_trial=reset,
    )
    result = online_relearn(
        view, task.actions, task.outputs, task.costs, **config["protocol"]
    )
    audit = audit_online(config, result, PrivateProbe(truth))
    assert audit["valid"], audit
    assert result["stop_reason"] == "monitor_control_failure"
    assert result["anomaly"] is None and result["new"] is None
    assert [v["status"] for v in result["theory_events"]] == [
        "proposed",
        "provisional",
        "accepted",
    ]
    assert audit["evaluation"]["known_failed_dispatches_excluded"] == 1
    assert result["monitoring"][0]["queries"][0]["spent"] == 2


def test_shared_monitor_sampler_cost_bug_is_rejected_before_private_read(monkeypatch):
    original = online.random_wp_plan

    def faulty(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan["queries"][0]["cost"] = True
        return plan

    monkeypatch.setattr(online, "random_wp_plan", faulty)
    config, result, _ = bundle()
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert (
        not audit["valid"]
        and probe.calls == 0
        and not audit["private_truth_read_attempted"]
    )


@pytest.mark.parametrize("warning", ["unmodeled-noise", "unverified-actuator-delivery"])
def test_unknown_warning_or_forged_delivery_scope_is_not_silently_cleared(warning):
    config, result, _ = copy.deepcopy(bundle())
    source = next(e for e in result["old"]["run"]["events"] if e["kind"] == "step")
    source["transition"]["possible_confounders"] = [warning]
    source["transition"]["context"] = {
        "intervention_compliance_scope": "physical-treatment",
        "treatment_verification": "verified",
    }
    probe = PrivateProbe()
    audit = audit_online(config, result, probe)
    assert not audit["valid"] and probe.calls == 0


def test_private_scoring_does_not_call_production_program_executor(monkeypatch):
    from audit_online import same_behavior

    from minecraft_scientist.theory.finite_state import FiniteStateProgram

    _, _, truth = bundle()

    def forbidden(*args, **kwargs):
        raise AssertionError("private behavior score reused production executor")

    monkeypatch.setattr(FiniteStateProgram, "execute", forbidden)
    assert not same_behavior(
        truth["before"], truth["after"], list(truth["before"]["program"]["actions"]), 5
    )
