from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minecraft_scientist.cli import app
from minecraft_scientist.evaluation import run_persistent_exploration_benchmark
from minecraft_scientist.exploration import ExplorationMemory, FrontierStatus


def test_cross_session_frontier_planning_uses_visual_memory_and_known_routes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exploration.sqlite3"
    with ExplorationMemory(path, default_match_radius=2) as first_session:
        origin = first_session.observe_place((0, 0), episode_id=0, step=0)
        distant = first_session.observe_place((20, 0), episode_id=0, step=1)
        route = first_session.register_frontier(
            origin.place_id,
            ("advance",),
            episode_id=0,
            step=2,
            expected_cost=2,
            uncertainty=0.9,
        )
        resolved = first_session.record_frontier_outcome(
            route.frontier_id,
            episode_id=0,
            step=3,
            success=True,
            cost=2,
            discovered_place_id=distant.place_id,
        )
        assert resolved.status is FrontierStatus.RESOLVED
        local = first_session.register_frontier(
            origin.place_id,
            ("risky_probe",),
            episode_id=0,
            step=4,
            uncertainty=0.1,
            risk=1.0,
        )
        remote = first_session.register_frontier(
            distant.place_id,
            ("unknown_corridor",),
            episode_id=0,
            step=5,
            uncertainty=1.0,
            risk=0.0,
        )

    with ExplorationMemory.open(path) as second_session:
        recognized = second_session.observe_place((1, 0), episode_id=1, step=0)
        assert recognized.place_id == origin.place_id
        plan = second_session.plan_next_frontier(origin.place_id)
        assert plan is not None
        assert plan.frontier_id == remote.frontier_id
        assert plan.frontier_id != local.frontier_id
        assert plan.place_path == (origin.place_id, distant.place_id)
        assert plan.travel_action_roles == (("advance",),)
        assert plan.action_roles == ("unknown_corridor",)
        known_route = second_session.plan_route(origin.place_id, distant.place_id)
        assert known_route is not None
        assert known_route.place_path == (origin.place_id, distant.place_id)
        assert known_route.travel_action_roles == (("advance",),)
        assert second_session.plan_route(distant.place_id, origin.place_id) is None
        assert second_session.summary() == {
            "format_version": 1,
            "place_count": 2,
            "multi_view_place_count": 1,
            "calibrated_place_count": 0,
            "perceptual_place_count": 0,
            "multi_view_perceptual_place_count": 0,
            "modality_place_count": 0,
            "calibrated_modality_place_count": 0,
            "frontier_count": 3,
            "active_frontier_count": 2,
            "resolved_frontier_count": 1,
            "blocked_frontier_count": 0,
            "transition_count": 1,
            "episode_count": 2,
            "event_count": 7,
            "integrity_valid": True,
        }

    result = CliRunner().invoke(
        app,
        ["inspect-exploration-memory", str(path), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["integrity_valid"] is True
    assert payload["place_count"] == 2
    assert payload["transition_count"] == 1


def test_persistent_exploration_benchmark_is_deterministic(tmp_path: Path) -> None:
    first = run_persistent_exploration_benchmark(
        seed=17,
        output=tmp_path / "first",
    )
    second = run_persistent_exploration_benchmark(
        seed=17,
        output=tmp_path / "second",
    )

    assert first == second
    assert first["metrics"] == {
        "coverage": 1.0,
        "place_recall_accuracy": 1.0,
        "duplicate_place_count": 0,
        "frontier_attempts": 24,
        "known_route_travel_actions": 13,
        "session_restarts": 4,
        "restart_recognitions": 4,
        "unresolved_frontiers": 0,
        "integrity_valid": True,
    }
    with ExplorationMemory.open(tmp_path / "first" / "exploration.sqlite3") as memory:
        assert memory.verify_integrity()
