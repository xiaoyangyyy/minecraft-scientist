"""Real vanilla-server smoke gate for an opaque open-world movement frontier."""

from __future__ import annotations

import json
import math
from pathlib import Path

from minecraft_scientist.exploration import ExplorationMemory
from minecraft_scientist.protocols import ActionKind, Observation

from .adapter import MinecraftConnectionConfig, MinecraftWorldAdapter
from .server import MinecraftServerProcess


def _place_signature(observation: Observation, stride: int) -> tuple[int, ...]:
    return tuple(
        math.floor(float(observation.agent_state[axis]) / stride)
        for axis in ("x", "y", "z")
    )


def run_open_world_movement_probe(
    *,
    server_directory: Path,
    output: Path,
    seed: int,
    stride: int = 4,
    username: str = "ExplorerBot",
    overwrite: bool = False,
) -> dict[str, object]:
    """Try bounded opaque frontiers until one real pathfinder move succeeds."""

    if stride < 2 or stride > 32:
        raise ValueError("movement stride must be between 2 and 32")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    events_path = output / "events.jsonl"
    memory_path = output / "exploration.sqlite3"
    known_outputs = (report_path, events_path, memory_path)
    if any(path.exists() for path in known_outputs) and not overwrite:
        raise FileExistsError(f"movement probe output already exists: {output}")
    if overwrite:
        for path in known_outputs:
            if path.exists():
                path.unlink()
        for suffix in ("-wal", "-shm"):
            candidate = Path(f"{memory_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    config = MinecraftConnectionConfig(
        username=username,
        radius=max(8, stride),
        timeout=45,
        pathfinder_timeout=30,
        exploration_stride=stride,
    )
    server = MinecraftServerProcess(
        server_directory,
        min_memory="1G",
        max_memory="2G",
        flavor="vanilla",
    )
    attempts: list[dict[str, object]] = []
    with server, MinecraftWorldAdapter(config) as world:
        initial = world.reset(seed)
        current = initial
        attempted_roles: set[tuple[int, int, int]] = set()
        for _ in range(4):
            actions = tuple(
                action
                for action in world.available_actions()
                if action.kind is ActionKind.MOVE
            )
            if len(actions) != 4:
                raise RuntimeError(
                    "real adapter did not expose four movement frontiers"
                )
            candidates: list[tuple[str, tuple[int, int, int], object, object]] = []
            current_position = tuple(
                float(current.agent_state[axis]) for axis in ("x", "y", "z")
            )
            for action in actions:
                target = next(
                    item
                    for item in current.objects
                    if item.object_id == action.target_id
                )
                target_position = tuple(
                    float(target.observable_features[axis]) for axis in ("x", "y", "z")
                )
                role = tuple(
                    round((target_value - source_value) / stride)
                    for source_value, target_value in zip(
                        current_position, target_position, strict=True
                    )
                )
                if role not in attempted_roles:
                    candidates.append((action.target_id or "", role, action, target))
            if not candidates:
                break
            _, role, selected, target = min(candidates, key=lambda item: item[0])
            attempted_roles.add(role)
            target_position = tuple(
                float(target.observable_features[axis]) for axis in ("x", "y", "z")
            )
            transition = world.step(selected)
            after = transition.state_after
            after_position = tuple(
                float(after.agent_state[axis]) for axis in ("x", "y", "z")
            )
            displacement = math.dist(current_position, after_position)
            target_error = math.dist(after_position, target_position)
            movement_valid = (
                transition.intervention_compliance
                and transition.event.success
                and displacement >= stride / 2
                and target_error <= 2
                and _place_signature(current, stride) != _place_signature(after, stride)
            )
            attempts.append(
                {
                    "before": current,
                    "target_id": selected.target_id or "",
                    "target_position": target_position,
                    "role": role,
                    "transition": transition,
                    "after": after,
                    "displacement": displacement,
                    "target_error": target_error,
                    "movement_valid": movement_valid,
                }
            )
            current = after
            if movement_valid:
                break

    if not attempts:
        raise RuntimeError("real adapter produced no movement attempt")
    final_attempt = attempts[-1]
    final_transition = final_attempt["transition"]
    assert hasattr(final_transition, "intervention_compliance")
    movement_valid = bool(final_attempt["movement_valid"])

    with ExplorationMemory(
        memory_path,
        default_match_radius=0,
        max_exemplars=4,
        block_after_failures=2,
    ) as memory:
        for attempt in attempts:
            before = attempt["before"]
            after = attempt["after"]
            assert isinstance(before, Observation)
            assert isinstance(after, Observation)
            source = memory.observe_place(
                _place_signature(before, stride),
                episode_id=before.episode_id,
                step=before.step,
                uncertainty=1,
            )
            destination = memory.observe_place(
                _place_signature(after, stride),
                episode_id=after.episode_id,
                step=after.step,
                uncertainty=0.5,
            )
            frontier = memory.register_frontier(
                source.place_id,
                (str(attempt["target_id"]),),
                episode_id=before.episode_id,
                step=before.step,
                expected_cost=1,
                uncertainty=1,
            )
            outcome = memory.record_frontier_outcome(
                frontier.frontier_id,
                episode_id=after.episode_id,
                step=after.step,
                success=bool(attempt["movement_valid"]),
                cost=1,
                discovered_place_id=(
                    destination.place_id if bool(attempt["movement_valid"]) else None
                ),
            )
            attempt["frontier_id"] = frontier.frontier_id
            attempt["frontier_status"] = outcome.status.value
        memory_summary = memory.summary()

    events_path.write_text(
        "".join(f"{attempt['transition'].model_dump_json()}\n" for attempt in attempts),
        encoding="utf-8",
    )
    public_attempts = [
        {
            "frontier_id": attempt["frontier_id"],
            "frontier_status": attempt["frontier_status"],
            "target_id": attempt["target_id"],
            "target_position": attempt["target_position"],
            "displacement": attempt["displacement"],
            "target_error": attempt["target_error"],
            "intervention_compliance": attempt["transition"].intervention_compliance,
            "controller_success": attempt["transition"].event.success,
            "movement_valid": attempt["movement_valid"],
            "controller_failure": attempt["transition"].controller_failure,
        }
        for attempt in attempts
    ]
    initial_position = tuple(
        float(initial.agent_state[axis]) for axis in ("x", "y", "z")
    )
    final_observation = final_attempt["after"]
    assert isinstance(final_observation, Observation)
    final_position = tuple(
        float(final_observation.agent_state[axis]) for axis in ("x", "y", "z")
    )
    report: dict[str, object] = {
        "suite": "real_open_world_movement",
        "seed": seed,
        "server_flavor": "vanilla",
        "minecraft_version": config.version,
        "movement_stride": stride,
        "attempt_count": len(attempts),
        "attempts": public_attempts,
        "frontier_id": final_attempt["frontier_id"],
        "frontier_status": final_attempt["frontier_status"],
        "before_position": initial_position,
        "target_position": final_attempt["target_position"],
        "after_position": final_position,
        "displacement": final_attempt["displacement"],
        "total_displacement": math.dist(initial_position, final_position),
        "target_error": final_attempt["target_error"],
        "intervention_compliance": final_transition.intervention_compliance,
        "controller_success": final_transition.event.success,
        "movement_valid": movement_valid,
        "memory": memory_summary,
        "public_semantics": "opaque movement target and public physical coordinates",
        "scope": (
            "A bounded real vanilla-server search over four opaque movement "
            "frontiers; not long-horizon autonomous survival or open-world coverage."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not movement_valid or memory_summary["integrity_valid"] is not True:
        raise RuntimeError("real open-world movement probe failed its gate")
    return report
