"""Actual hidden switches, charged monitoring and immutable version lineage."""

import inspect

import pytest
from online import online_relearn
from worlds import online_task


def learn(method="adaptive", case="state_growth", *, boundary=16, **protocol):
    task = online_task(0, case, boundary=boundary)
    return online_relearn(
        task.view, task.actions, task.outputs, task.costs, method=method, **protocol
    )


@pytest.mark.parametrize("method", ["from_scratch", "adaptive"])
@pytest.mark.parametrize("case", ["output_change", "redirect", "state_growth"])
def test_unannounced_change_is_detected_before_paid_relearning_and_replacement(
    method, case
):
    result = learn(method, case)
    assert result["status"] == "revision_finitely_validated", result["stop_reason"]
    assert result["anomaly"]["episode_id"] >= 16
    assert result["spent"] <= result["budget"]
    assert result["spent"] == (
        result["old"]["run"]["spent"]
        + sum(q["spent"] for b in result["monitoring"] for q in b["queries"])
        + result["new"]["run"]["spent"]
    )
    history = result["theory_events"]
    assert [v["status"] for v in history] == [
        "proposed",
        "provisional",
        "accepted",
        "challenged",
        "proposed",
        "provisional",
        "accepted",
        "superseded",
    ]
    assert history[0]["ir"] == history[3]["ir"] == history[-1]["ir"]
    assert history[-2]["parent_version"] == 1
    assert result["new"]["config"]["references"] == (
        [result["public_old_theory"]["ir"]] if method == "adaptive" else []
    )
    for assignment in result["validation_assignments"]:
        assert len(assignment["validation_ids"]) == 8
        assert not set(assignment["validation_ids"]) & set(assignment["discovery_ids"])
    assert not result["mechanism_change_warranted"]
    assert result["mechanism_attribution_status"] == "unresolved"


def test_both_methods_have_identical_paid_bootstrap_and_monitor_history():
    scratch, adaptive = learn("from_scratch"), learn("adaptive")
    for key in ("old", "public_old_theory", "monitoring", "anomaly"):
        assert scratch[key] == adaptive[key]
    assert scratch["new"]["config"]["protocol"] == adaptive["new"]["config"]["protocol"]


def test_unchanged_world_has_bounded_monitoring_not_false_replacement():
    result = learn(case="unchanged")
    assert result["status"] == "bounded_monitoring_completed_no_anomaly"
    assert result["anomaly"] is None and result["new"] is None
    assert len(result["monitoring"]) == 4
    assert all(b["status"] == "passed" for b in result["monitoring"])
    assert len(result["theory_events"]) == 3


@pytest.mark.parametrize("budget", [1, 69, 75])
def test_exhausted_budget_never_becomes_completed_or_free_validation(budget):
    result = learn(budget=budget)
    assert result["status"] == "unresolved" and result["spent"] <= budget
    assert result["new"] is None
    assert not any(v["status"] == "superseded" for v in result["theory_events"])


def test_boundary_is_not_an_agent_argument_and_global_episode_clock_is_retained():
    assert "boundary" not in inspect.signature(online_relearn).parameters
    result = learn()
    assert result == learn()
    raw = (
        result["old"]["raw_controls"]
        + [
            c
            for b in result["monitoring"]
            for q in b["queries"]
            for c in q["raw_controls"]
        ]
        + result["new"]["raw_controls"]
    )
    episodes = [c["observation"]["episode_id"] for c in raw if c["kind"] == "reset"]
    assert episodes == list(range(len(episodes)))
    assert result["new"]["run"]["events"][2]["kind"] != "cache"


@pytest.mark.parametrize("boundary", [0, 2, 8, 1000])
def test_early_late_and_absent_in_window_changes_remain_explicit_outcomes(boundary):
    result = learn(boundary=boundary)
    assert result["spent"] <= result["budget"]
    if result["old"]["run"]["status"] != "provisional_conformance_passed":
        assert result["stop_reason"] == "old_learning_unresolved"
        assert not result["theory_events"]
    if result["anomaly"] is None:
        assert not any(v["status"] == "superseded" for v in result["theory_events"])


@pytest.mark.parametrize(
    "protocol",
    [{"budget": True}, {"monitor_rounds": 17}, {"monitor_count": 0}, {"seed": -1}],
)
def test_invalid_protocol_fails_before_dispatch(protocol):
    with pytest.raises(ValueError, match="online relearning protocol"):
        learn(**protocol)
