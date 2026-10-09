"""The shared scientific entry rejects unsuccessful measurements, not theories."""

from minecraft_scientist.agent.bootstrap import bootstrap_config, learn_shared_bootstrap
from minecraft_scientist.evaluation.bootstrap import check_bootstrap_public
from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.world.revision import revision_task


def test_reported_measurement_failure_is_paid_without_filtering_or_publication():
    task = revision_task(0, "unchanged", boundary=100)
    config = bootstrap_config(
        tuple(task.costs),
        tuple(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
        task.costs,
        {
            "hypotheses": [h.model_dump(mode="json") for h in task.bootstrap],
            "calibration": list(task.calibration),
            "discovery_budget": 40,
            "shadow_count": 3,
        },
        budget=250,
        horizon=5,
        seed=20261008,
    )

    def failed(action):
        record = task.view.step(action)
        event = record.event.model_copy(update={"success": False})
        return record.model_copy(
            update={
                "event": event,
                "state_after": record.state_after.model_copy(
                    update={"last_event": event}
                ),
            }
        )

    view = AgentWorldView(
        observe=task.view.observe,
        available=task.view.available_actions,
        step=failed,
        begin_trial=task.view.begin_trial,
    )
    run = learn_shared_bootstrap(view, config)
    assert run["discovery"]["stop_reason"] == "measurement_failure"
    assert len(run["discovery"]["remaining_candidates"]) == 9
    assert run["candidate_ir"] is None and run["validation"] is None
    assert run["spent"] == 1 + task.costs[task.calibration[0]]
    assert not any(e["kind"] == "evidence" for e in run["events"])
    check_bootstrap_public({"config": config, "run": run})
