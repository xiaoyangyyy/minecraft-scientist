"""Deterministic multi-session gate for persistent frontier exploration."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from minecraft_scientist.exploration import ExplorationMemory

_GRAPH: dict[int, tuple[int | None, ...]] = {
    0: (1, 2, None),
    1: (0, 3, None),
    2: (0, 4, None),
    3: (1, 4, None),
    4: (2, 3, 5, None),
    5: (4, None),
}


def _role(seed: int, source: int, slot: int) -> str:
    digest = hashlib.blake2s(
        f"{seed}:frontier:{source}:{slot}".encode(), digest_size=8
    ).hexdigest()
    return f"act_{digest}"


def _signature(seed: int, node: int, episode: int) -> tuple[int, ...]:
    """Return an anonymous place view with deterministic cross-session nuisance."""

    rng = random.Random((seed + 1) * 1_000_003 + node * 10_007 + episode * 101)
    center = (node * 24, (node % 3) * 20, (node // 3) * 28)
    return tuple(value + rng.randint(-1, 1) for value in center)


def _actions(seed: int, node: int) -> tuple[str, ...]:
    return tuple(_role(seed, node, slot) for slot in range(len(_GRAPH[node])))


def _target(seed: int, node: int, role: str) -> int | None:
    roles = _actions(seed, node)
    try:
        slot = roles.index(role)
    except ValueError as error:
        raise ValueError("unknown opaque exploration action") from error
    return _GRAPH[node][slot]


def run_persistent_exploration_benchmark(
    *,
    seed: int,
    output: Path,
    budget: int = 60,
    session_length: int = 5,
    overwrite: bool = False,
) -> dict[str, object]:
    """Explore a hidden graph across forced restarts using only opaque public views.

    This is an algorithmic release gate, not evidence of Minecraft open-world
    competence. Hidden node identities and adjacency remain evaluator-side; the
    memory receives only perturbed integer signatures and opaque action roles.
    """

    if budget < 1 or session_length < 1:
        raise ValueError("budget and session length must be positive")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    database_path = output / "exploration.sqlite3"
    if (report_path.exists() or database_path.exists()) and not overwrite:
        raise FileExistsError(f"exploration output already exists: {output}")
    if overwrite:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{database_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
        if report_path.exists():
            report_path.unlink()

    memory = ExplorationMemory(
        database_path,
        default_match_radius=4,
        ambiguity_margin=0.5,
        max_exemplars=6,
        block_after_failures=2,
    )
    episode = 0
    hidden_node = 0
    attempts = 0
    travel_actions = 0
    restart_recognitions = 0
    recognition_trials = 0
    recognition_successes = 0
    expected_place_by_hidden: dict[int, str] = {}

    def observe(node: int, step: int) -> str:
        nonlocal recognition_successes, recognition_trials
        place = memory.observe_place(
            _signature(seed, node, episode),
            episode_id=episode,
            step=step,
            uncertainty=max(0.05, 1 - len(expected_place_by_hidden) / len(_GRAPH)),
        )
        previous = expected_place_by_hidden.get(node)
        if previous is None:
            expected_place_by_hidden[node] = place.place_id
        else:
            recognition_trials += 1
            if place.place_id != previous:
                raise RuntimeError("multi-session place recognition split a known node")
            recognition_successes += 1
        return place.place_id

    current_place = observe(hidden_node, 0)
    step = 0
    try:
        while attempts < budget:
            for role in _actions(seed, hidden_node):
                memory.register_frontier(
                    current_place,
                    (role,),
                    episode_id=episode,
                    step=step,
                    expected_cost=1,
                    uncertainty=1,
                    risk=0,
                )
            plan = memory.plan_next_frontier(current_place)
            if plan is None:
                break
            hidden_by_place = {
                place_id: node for node, place_id in expected_place_by_hidden.items()
            }
            try:
                source_hidden = hidden_by_place[plan.place_path[-1]]
            except KeyError as error:
                raise RuntimeError(
                    "planner routed through an unrecognized place"
                ) from error
            travel_actions += len(plan.place_path) - 1
            hidden_node = source_hidden
            current_place = plan.place_path[-1]
            role = plan.action_roles[0]
            target = _target(seed, hidden_node, role)
            attempts += 1
            step += 1
            if target is None:
                memory.record_frontier_outcome(
                    plan.frontier_id,
                    episode_id=episode,
                    step=step,
                    success=False,
                    cost=1,
                )
            else:
                hidden_node = target
                discovered_place = observe(hidden_node, step)
                memory.record_frontier_outcome(
                    plan.frontier_id,
                    episode_id=episode,
                    step=step,
                    success=True,
                    cost=1,
                    discovered_place_id=discovered_place,
                )
                current_place = discovered_place

            if attempts % session_length == 0:
                memory.close()
                episode += 1
                hidden_node = 0
                memory = ExplorationMemory.open(database_path)
                before = expected_place_by_hidden[0]
                current_place = observe(hidden_node, 0)
                if current_place == before:
                    restart_recognitions += 1
                step = 0

        summary = memory.summary()
        unresolved = summary["active_frontier_count"]
        coverage = len(expected_place_by_hidden) / len(_GRAPH)
        place_recall = (
            recognition_successes / recognition_trials if recognition_trials else 1.0
        )
        duplicate_places = summary["place_count"] - len(expected_place_by_hidden)
        report: dict[str, object] = {
            "suite": "persistent_exploration",
            "seed": seed,
            "configuration": {
                "budget": budget,
                "session_length": session_length,
                "hidden_place_count": len(_GRAPH),
                "public_semantics": "opaque signatures and action roles only",
            },
            "metrics": {
                "coverage": coverage,
                "place_recall_accuracy": place_recall,
                "duplicate_place_count": duplicate_places,
                "frontier_attempts": attempts,
                "known_route_travel_actions": travel_actions,
                "session_restarts": episode,
                "restart_recognitions": restart_recognitions,
                "unresolved_frontiers": unresolved,
                "integrity_valid": summary["integrity_valid"],
            },
            "memory": summary,
            "scope": (
                "Deterministic hidden-graph release gate for multi-session memory and "
                "frontier planning; it is not a real-Minecraft open-world result."
            ),
        }
        if not (
            coverage == 1.0
            and duplicate_places == 0
            and unresolved == 0
            and summary["integrity_valid"] is True
            and episode >= 1
            and restart_recognitions == episode
        ):
            raise RuntimeError("persistent exploration benchmark did not meet its gate")
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        memory.close()
