"""Persistent multi-session exploration on a real vanilla Minecraft server."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from minecraft_scientist.exploration import (
    AnonymousSceneEncoder,
    ExplorationMemory,
    FrontierMemory,
    FrontierStatus,
)
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)

from .adapter import MinecraftConnectionConfig, MinecraftWorldAdapter
from .server import MinecraftServerProcess


def _position(observation: Observation) -> tuple[float, float, float]:
    return tuple(float(observation.agent_state[axis]) for axis in ("x", "y", "z"))


def _cell_signature(
    position: tuple[float, float, float], stride: int
) -> tuple[int, int, int]:
    return (
        math.floor(position[0] / stride),
        math.floor(position[1] / stride),
        math.floor(position[2] / stride),
    )


def _observation_signature(
    observation: Observation, stride: int
) -> tuple[int, int, int]:
    return _cell_signature(_position(observation), stride)


def _target_position(target: ObjectObservation) -> tuple[float, float, float]:
    return tuple(float(target.observable_features[axis]) for axis in ("x", "y", "z"))


def _movement_valid(
    transition: TransitionRecord,
    target_position: tuple[float, float, float],
    stride: int,
) -> tuple[bool, float, float]:
    before = _position(transition.state_before)
    after = _position(transition.state_after)
    displacement = math.dist(before, after)
    target_error = math.dist(after, target_position)
    valid = (
        transition.intervention_compliance
        and transition.event.success
        and displacement >= stride / 2
        and target_error <= 2
        and _cell_signature(before, stride) != _cell_signature(after, stride)
    )
    return valid, displacement, target_error


@dataclass(frozen=True)
class _Candidate:
    action: PrimitiveAction
    target: ObjectObservation
    frontier: FrontierMemory
    predicted_signature: tuple[int, int, int]
    novel: bool


@dataclass(frozen=True)
class _RouteReplay:
    action: PrimitiveAction
    target: ObjectObservation
    expected_place_id: str


def _movement_objects(
    observation: Observation,
) -> dict[str, ObjectObservation]:
    return {
        item.object_id: item
        for item in observation.objects
        if item.affordances == (ActionKind.MOVE,)
    }


def _register_candidates(
    memory: ExplorationMemory,
    observation: Observation,
    current_place_id: str,
    actions: tuple[PrimitiveAction, ...],
    *,
    stride: int,
    episode_id: int,
    step: int,
) -> tuple[_Candidate, ...]:
    targets = _movement_objects(observation)
    candidates: list[_Candidate] = []
    for action in actions:
        if action.kind is not ActionKind.MOVE or action.target_id not in targets:
            continue
        assert action.target_id is not None
        target = targets[action.target_id]
        predicted = _cell_signature(_target_position(target), stride)
        frontier = memory.register_frontier(
            current_place_id,
            (action.target_id,),
            episode_id=episode_id,
            step=step,
            expected_cost=1,
            uncertainty=1,
        )
        if frontier.status is FrontierStatus.ACTIVE:
            candidates.append(
                _Candidate(
                    action=action,
                    target=target,
                    frontier=frontier,
                    predicted_signature=predicted,
                    novel=memory.match_place(predicted) is None,
                )
            )
    return tuple(candidates)


def _select_candidate(candidates: tuple[_Candidate, ...]) -> _Candidate | None:
    """Prefer unvisited cells, then least-tested and least-failed frontiers."""

    return min(
        candidates,
        key=lambda item: (
            not item.novel,
            item.frontier.attempts,
            item.frontier.failure_streak,
            item.frontier.frontier_id,
        ),
        default=None,
    )


def _route_replay(
    memory: ExplorationMemory,
    observation: Observation,
    current_place_id: str,
    actions: tuple[PrimitiveAction, ...],
) -> _RouteReplay | None:
    """Bind the first edge of a known route to a current opaque target handle."""

    plan = memory.plan_next_frontier(current_place_id)
    if plan is None or len(plan.place_path) < 2:
        return None
    next_place_id = plan.place_path[1]
    targets = _movement_objects(observation)
    action_by_target = {
        action.target_id: action
        for action in actions
        if action.kind is ActionKind.MOVE and action.target_id is not None
    }
    options = [
        transition
        for transition in memory.transitions()
        if transition.source_place_id == current_place_id
        and transition.target_place_id == next_place_id
        and len(transition.action_roles) == 1
        and transition.action_roles[0] in action_by_target
        and transition.action_roles[0] in targets
    ]
    if not options:
        return None
    transition = max(
        options,
        key=lambda item: (item.confidence, -item.mean_cost, item.action_roles),
    )
    target_id = transition.action_roles[0]
    return _RouteReplay(
        action=action_by_target[target_id],
        target=targets[target_id],
        expected_place_id=next_place_id,
    )


def _known_outputs(output: Path) -> tuple[Path, ...]:
    return (
        output / "config.json",
        output / "events.jsonl",
        output / "report.json",
        output / "exploration.sqlite3",
    )


def run_open_world_exploration(
    *,
    server_directory: Path,
    output: Path,
    seed: int,
    move_budget: int = 12,
    restart_interval: int = 3,
    stride: int = 4,
    minimum_successful_moves: int = 8,
    minimum_unique_places: int = 6,
    perceptual_tolerance: float = 0,
    perceptual_ambiguity_margin: float = 0,
    username: str = "ExplorerBot",
    overwrite: bool = False,
) -> dict[str, object]:
    """Explore real terrain across controller restarts with durable frontier memory."""

    if move_budget < 1:
        raise ValueError("move budget must be positive")
    if restart_interval < 1:
        raise ValueError("restart interval must be positive")
    if stride < 2 or stride > 32:
        raise ValueError("movement stride must be between 2 and 32")
    if not 1 <= minimum_successful_moves <= move_budget:
        raise ValueError("minimum successful moves must be in [1, move budget]")
    if minimum_unique_places < 2:
        raise ValueError("minimum unique places must be at least two")
    if not math.isfinite(perceptual_tolerance) or perceptual_tolerance < 0:
        raise ValueError("perceptual tolerance must be finite and nonnegative")
    if (
        not math.isfinite(perceptual_ambiguity_margin)
        or perceptual_ambiguity_margin < 0
    ):
        raise ValueError("perceptual ambiguity margin must be finite and nonnegative")

    output.mkdir(parents=True, exist_ok=True)
    known_outputs = _known_outputs(output)
    if any(path.exists() for path in known_outputs) and not overwrite:
        raise FileExistsError(f"open-world exploration output already exists: {output}")
    if overwrite:
        for path in known_outputs:
            if path.exists():
                path.unlink()
        memory_path = output / "exploration.sqlite3"
        for suffix in ("-wal", "-shm"):
            candidate = Path(f"{memory_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    config_payload = {
        "seed": seed,
        "move_budget": move_budget,
        "restart_interval": restart_interval,
        "movement_stride": stride,
        "minimum_successful_moves": minimum_successful_moves,
        "minimum_unique_places": minimum_unique_places,
        "perceptual_tolerance": perceptual_tolerance,
        "perceptual_ambiguity_margin": perceptual_ambiguity_margin,
        "username": username,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    memory_path = output / "exploration.sqlite3"
    event_payloads: list[dict[str, object]] = []
    attempts = 0
    successful_moves = 0
    controller_failures = 0
    route_replays = 0
    route_replay_successes = 0
    restart_trials = 0
    restart_recognitions = 0
    perceptual_restart_recognitions = 0
    perceptual_restart_rejections = 0
    perceptual_restart_false_matches = 0
    perceptual_known_place_trials = 0
    perceptual_known_place_recognitions = 0
    perceptual_known_place_rejections = 0
    perceptual_known_place_false_matches = 0
    perceptual_novelty_collisions = 0
    sessions = 0
    total_path_length = 0.0
    initial_position: tuple[float, float, float] | None = None
    final_position: tuple[float, float, float] | None = None
    maximum_distance = 0.0
    stalled = False
    scene_encoder = AnonymousSceneEncoder()

    connection = MinecraftConnectionConfig(
        username=username,
        radius=max(10, stride),
        max_blocks=2048,
        timeout=45,
        pathfinder_timeout=12,
        exploration_stride=stride,
    )
    server = MinecraftServerProcess(
        server_directory,
        min_memory="1G",
        max_memory="2G",
        flavor="vanilla",
    )
    with server:
        while attempts < move_budget and not stalled:
            session_id = sessions
            sessions += 1
            with MinecraftWorldAdapter(connection) as world:
                current = world.reset(seed)
                if not any(
                    "diggable" in item.observable_features for item in current.objects
                ):
                    current = world.step(
                        PrimitiveAction(kind=ActionKind.WAIT, ticks=10)
                    ).state_after
                current_position = _position(current)
                if initial_position is None:
                    initial_position = current_position
                final_position = current_position
                if memory_path.exists():
                    memory = ExplorationMemory.open(memory_path)
                else:
                    memory = ExplorationMemory(
                        memory_path,
                        default_match_radius=0,
                        max_exemplars=4,
                        block_after_failures=2,
                    )
                with memory:
                    matched_before = memory.match_place(
                        _observation_signature(current, stride)
                    )
                    scene_before = scene_encoder.encode(current)
                    perceptual_before = memory.match_perceptual_place(
                        scene_before,
                        tolerance=perceptual_tolerance,
                        ambiguity_margin=perceptual_ambiguity_margin,
                    )
                    if session_id > 0:
                        restart_trials += 1
                        restart_recognitions += int(matched_before is not None)
                    current_place = memory.observe_place(
                        _observation_signature(current, stride),
                        episode_id=session_id,
                        step=attempts,
                        uncertainty=0.5 if matched_before is not None else 1,
                    )
                    if session_id > 0:
                        if perceptual_before == current_place.place_id:
                            perceptual_restart_recognitions += 1
                        elif perceptual_before is None:
                            perceptual_restart_rejections += 1
                        else:
                            perceptual_restart_false_matches += 1
                    memory.record_perceptual_view(
                        current_place.place_id,
                        scene_before,
                        episode_id=session_id,
                        step=attempts,
                    )
                    session_attempts = 0
                    while (
                        attempts < move_budget and session_attempts < restart_interval
                    ):
                        actions = world.available_actions()
                        candidates = _register_candidates(
                            memory,
                            current,
                            current_place.place_id,
                            actions,
                            stride=stride,
                            episode_id=session_id,
                            step=attempts,
                        )
                        selected = _select_candidate(candidates)
                        replay = None
                        mode = "frontier"
                        if selected is None:
                            replay = _route_replay(
                                memory,
                                current,
                                current_place.place_id,
                                actions,
                            )
                            if replay is None:
                                stalled = True
                                break
                            mode = "route_replay"
                            action = replay.action
                            target = replay.target
                            route_replays += 1
                        else:
                            action = selected.action
                            target = selected.target

                        transition = world.step(action)
                        valid, displacement, target_error = _movement_valid(
                            transition,
                            _target_position(target),
                            stride,
                        )
                        destination_signature = _observation_signature(
                            transition.state_after, stride
                        )
                        matched_destination = memory.match_place(destination_signature)
                        scene_destination = scene_encoder.encode(transition.state_after)
                        perceptual_destination = memory.match_perceptual_place(
                            scene_destination,
                            tolerance=perceptual_tolerance,
                            ambiguity_margin=perceptual_ambiguity_margin,
                        )
                        destination = memory.observe_place(
                            destination_signature,
                            episode_id=session_id,
                            step=attempts + 1,
                            uncertainty=(0.5 if matched_destination is not None else 1),
                        )
                        if matched_destination is not None:
                            perceptual_known_place_trials += 1
                            if perceptual_destination == destination.place_id:
                                perceptual_known_place_recognitions += 1
                            elif perceptual_destination is None:
                                perceptual_known_place_rejections += 1
                            else:
                                perceptual_known_place_false_matches += 1
                        elif perceptual_destination is not None:
                            perceptual_novelty_collisions += 1
                        memory.record_perceptual_view(
                            destination.place_id,
                            scene_destination,
                            episode_id=session_id,
                            step=attempts + 1,
                        )
                        route_expected = None
                        if selected is not None:
                            outcome = memory.record_frontier_outcome(
                                selected.frontier.frontier_id,
                                episode_id=session_id,
                                step=attempts + 1,
                                success=valid,
                                cost=max(displacement, 0.001),
                                discovered_place_id=(
                                    destination.place_id if valid else None
                                ),
                            )
                            frontier_id = outcome.frontier_id
                            frontier_status = outcome.status.value
                        else:
                            assert replay is not None
                            route_expected = replay.expected_place_id
                            valid = valid and destination.place_id == route_expected
                            route_replay_successes += int(valid)
                            frontier_id = None
                            frontier_status = None

                        attempts += 1
                        session_attempts += 1
                        successful_moves += int(valid)
                        controller_failures += int(
                            not transition.intervention_compliance
                            or not transition.event.success
                        )
                        total_path_length += displacement
                        final_position = _position(transition.state_after)
                        assert initial_position is not None
                        maximum_distance = max(
                            maximum_distance,
                            math.dist(initial_position, final_position),
                        )
                        event_payloads.append(
                            {
                                "attempt": attempts,
                                "session": session_id,
                                "mode": mode,
                                "source_place_id": current_place.place_id,
                                "destination_place_id": destination.place_id,
                                "destination_was_known": (
                                    matched_destination is not None
                                ),
                                "perceptual_prediction": perceptual_destination,
                                "perceptual_signature_sha256": hashlib.sha256(
                                    json.dumps(scene_destination).encode("utf-8")
                                ).hexdigest(),
                                "route_expected_place_id": route_expected,
                                "frontier_id": frontier_id,
                                "frontier_status": frontier_status,
                                "movement_valid": valid,
                                "displacement": displacement,
                                "target_error": target_error,
                                "transition": transition.model_dump(mode="json"),
                            }
                        )
                        current = transition.state_after
                        current_place = destination

    assert initial_position is not None and final_position is not None
    with ExplorationMemory.open(memory_path) as memory:
        memory_summary = memory.summary()
    restarts = max(0, sessions - 1)
    gate_passed = (
        successful_moves >= minimum_successful_moves
        and int(memory_summary["place_count"]) >= minimum_unique_places
        and restart_trials >= 1
        and restart_recognitions == restart_trials
        and perceptual_restart_recognitions == restart_trials
        and perceptual_restart_false_matches == 0
        and memory_summary["integrity_valid"] is True
    )
    report: dict[str, object] = {
        "suite": "real_open_world_exploration",
        "seed": seed,
        "server_flavor": "vanilla",
        "minecraft_version": connection.version,
        "movement_stride": stride,
        "move_budget": move_budget,
        "attempts": attempts,
        "successful_moves": successful_moves,
        "controller_failures": controller_failures,
        "sessions": sessions,
        "restarts": restarts,
        "restart_trials": restart_trials,
        "restart_recognitions": restart_recognitions,
        "perceptual_restart_recognitions": perceptual_restart_recognitions,
        "perceptual_restart_rejections": perceptual_restart_rejections,
        "perceptual_restart_false_matches": perceptual_restart_false_matches,
        "perceptual_known_place_trials": perceptual_known_place_trials,
        "perceptual_known_place_recognitions": perceptual_known_place_recognitions,
        "perceptual_known_place_rejections": perceptual_known_place_rejections,
        "perceptual_known_place_false_matches": (perceptual_known_place_false_matches),
        "perceptual_novelty_collisions": perceptual_novelty_collisions,
        "perceptual_descriptor_dimensions": scene_encoder.DIMENSIONS,
        "perceptual_tolerance": perceptual_tolerance,
        "perceptual_ambiguity_margin": perceptual_ambiguity_margin,
        "route_replays": route_replays,
        "route_replay_successes": route_replay_successes,
        "stalled": stalled,
        "initial_position": initial_position,
        "final_position": final_position,
        "net_displacement": math.dist(initial_position, final_position),
        "total_path_length": total_path_length,
        "maximum_distance_from_start": maximum_distance,
        "memory": memory_summary,
        "gate": {
            "minimum_successful_moves": minimum_successful_moves,
            "minimum_unique_places": minimum_unique_places,
            "required_restart_recognitions": restart_trials,
            "required_perceptual_restart_recognitions": restart_trials,
            "maximum_perceptual_restart_false_matches": 0,
            "requires_memory_integrity": True,
            "passed": gate_passed,
        },
        "scope": (
            "A bounded multi-session vanilla-server movement exploration with "
            "coordinate-cell supervision and anonymous structural scene retrieval; "
            "not pixel-visual localization, survival, resource gathering, or "
            "unbounded open-world competence."
        ),
    }
    (output / "events.jsonl").write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in event_payloads
        ),
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate_passed:
        raise RuntimeError("real open-world exploration failed its gate")
    return report
