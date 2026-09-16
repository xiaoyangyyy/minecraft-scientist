from pathlib import Path

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.protocols import AgentWorldView, RunConfig, TheoryStatus
from minecraft_scientist.world import WorldBundle, WorldFactory


def test_full_run_discovers_revises_and_evaluates() -> None:
    result = ScientistAgent(RunConfig(seed=0)).run()
    report = result.report
    assert report.hidden_intervention_accuracy == 1.0
    assert report.task_success_rate == 1.0
    assert report.mechanism_accuracy == {
        "machine_protocol": 1.0,
        "tool_capability": 1.0,
        "transformation_chain": 1.0,
    }
    assert report.theory_versions["tool-capability"] == 2
    tool_history = [t for t in result.theories if t.theory_id == "tool-capability"]
    assert [theory.status for theory in tool_history] == [
        TheoryStatus.SUPERSEDED,
        TheoryStatus.ACCEPTED,
    ]
    assert tool_history[1].parent_version == 1
    assert tool_history[0].counterexample_ids
    assert result.symbols
    assert all(
        experiment.intervention.predicted_outcomes
        for experiment in result.experiments
    )


def test_run_artifacts_are_reproducible(tmp_path: Path) -> None:
    config = RunConfig(seed=9)
    first = ScientistAgent(config).run()
    second = ScientistAgent(config).run()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first.persist(first_dir)
    second.persist(second_dir)
    for filename in (
        "config.json",
        "events.jsonl",
        "theories.jsonl",
        "symbols.json",
        "report.json",
    ):
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()


def test_randomized_ids_do_not_change_recovery_quality() -> None:
    reports = [ScientistAgent(RunConfig(seed=seed)).run().report for seed in range(5)]
    assert all(report.hidden_intervention_accuracy == 1.0 for report in reports)
    assert all(report.theory_versions["tool-capability"] == 2 for report in reports)


def test_controller_failure_is_recorded_but_never_used_as_world_evidence() -> None:
    base = WorldFactory.create(23, change_episode=1)
    failed_once = False

    def flaky_step(action):
        nonlocal failed_once
        transition = base.agent_view.step(action)
        if not failed_once:
            failed_once = True
            return transition.model_copy(
                update={
                    "intervention_compliance": False,
                    "controller_failure": "injected_transient_failure",
                }
            )
        return transition

    view = AgentWorldView(
        observe=base.agent_view.observe,
        available=base.agent_view.available_actions,
        step=flaky_step,
        begin_trial=base.agent_view.begin_trial,
    )
    bundle = WorldBundle(
        agent_view=view,
        controller=base.controller,
        evaluator=base.evaluator,
    )
    result = ScientistAgent(RunConfig(seed=23), bundle=bundle).run()
    assert result.report.hidden_intervention_accuracy == 1.0
    assert any(not experiment.compliant for experiment in result.experiments)
