"""Deterministic opaque structured world used by the MVP benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


@dataclass(frozen=True, slots=True)
class _GroundTruthSpec:
    tools: tuple[str, ...]
    blocks: tuple[str, ...]
    base_requirements: dict[str, int]
    shifted_requirements: dict[str, int]
    machine_id: str
    machine_input: str
    machine_output: str
    machine_protocol: tuple[str, ...]
    processors: tuple[str, ...]
    resources: tuple[str, ...]
    transformations: dict[str, tuple[str, str]]
    change_episode: int | None

    def requirements(self, episode_id: int) -> dict[str, int]:
        if self.change_episode is not None and episode_id >= self.change_episode:
            return self.shifted_requirements
        return self.base_requirements

    def tool_outcome(self, tool_id: str, block_id: str, episode_id: int) -> bool:
        rank = self.tools.index(tool_id)
        return rank >= self.requirements(episode_id)[block_id]


class WorldEvaluator:
    """Privileged benchmark handle. Never pass this object to the agent."""

    __slots__ = ("__truth",)

    def __init__(self, truth: _GroundTruthSpec) -> None:
        self.__truth = truth

    @property
    def tools(self) -> tuple[str, ...]:
        return self.__truth.tools

    @property
    def blocks(self) -> tuple[str, ...]:
        return self.__truth.blocks

    @property
    def processors(self) -> tuple[str, ...]:
        return self.__truth.processors

    @property
    def resources(self) -> tuple[str, ...]:
        return self.__truth.resources

    @property
    def machine_id(self) -> str:
        return self.__truth.machine_id

    def tool_outcome(self, tool_id: str, block_id: str, episode_id: int) -> bool:
        return self.__truth.tool_outcome(tool_id, block_id, episode_id)

    def active_requirements(self, episode_id: int) -> dict[str, int]:
        return dict(self.__truth.requirements(episode_id))

    def machine_rule(self) -> tuple[str, tuple[str, ...], str]:
        return (
            self.__truth.machine_input,
            self.__truth.machine_protocol,
            self.__truth.machine_output,
        )

    def transformations(self) -> dict[str, tuple[str, str]]:
        return dict(self.__truth.transformations)


class WorldController:
    """Benchmark-only lifecycle control, kept outside AgentWorldView."""

    __slots__ = ("__world",)

    def __init__(self, world: _StructuredTestWorld) -> None:
        self.__world = world

    def begin_episode(self, episode_id: int) -> Observation:
        return self.__world.begin_episode(episode_id)


@dataclass(frozen=True, slots=True)
class WorldBundle:
    agent_view: AgentWorldView
    controller: WorldController
    evaluator: WorldEvaluator


class _StructuredTestWorld:
    def __init__(self, truth: _GroundTruthSpec) -> None:
        self.__truth = truth
        self.__episode_id = 0
        self.__step = 0
        self.__selected: str | None = None
        self.__last_event: WorldEvent | None = None
        self.__sequences: dict[str, list[str]] = {}
        self.__pending_inputs: dict[str, str] = {}

    def reset(self, seed: int) -> Observation:
        del seed
        self.__episode_id = 0
        self.__step = 0
        self.__selected = None
        self.__last_event = None
        self.__sequences.clear()
        self.__pending_inputs.clear()
        return self.observe()

    def begin_episode(self, episode_id: int) -> Observation:
        self.__episode_id = episode_id
        self.__step = 0
        self.__selected = None
        return self.begin_trial()

    def begin_trial(self) -> Observation:
        self.__sequences.clear()
        self.__pending_inputs.clear()
        self.__last_event = None
        return self.observe()

    def observe(self) -> Observation:
        selectable = tuple(
            ObjectObservation(
                object_id=object_id,
                affordances=(ActionKind.SELECT,),
                observable_features={"shape_token": f"f{index % 2}"},
            )
            for index, object_id in enumerate(self.__truth.tools)
        )
        targets = tuple(
            ObjectObservation(
                object_id=object_id,
                affordances=(ActionKind.INTERACT,),
                observable_features={"surface_token": f"s{index % 2}"},
            )
            for index, object_id in enumerate(self.__truth.blocks)
        )
        stations = tuple(
            ObjectObservation(
                object_id=object_id,
                affordances=(
                    ActionKind.PLACE_INPUT,
                    ActionKind.ACTIVATE,
                    ActionKind.INTERACT,
                ),
                observable_features={"fixture_token": f"m{index % 2}"},
            )
            for index, object_id in enumerate(
                (self.__truth.machine_id, *self.__truth.processors)
            )
        )
        inventory = {
            object_id: 99
            for object_id in (*self.__truth.tools, *self.__truth.resources)
        }
        return Observation(
            episode_id=self.__episode_id,
            step=self.__step,
            inventory=inventory,
            selected_id=self.__selected,
            objects=selectable + targets + stations,
            last_event=self.__last_event,
        )

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        actions: list[PrimitiveAction] = []
        actions.extend(
            PrimitiveAction(kind=ActionKind.SELECT, target_id=tool)
            for tool in self.__truth.tools
        )
        actions.extend(
            PrimitiveAction(kind=ActionKind.INTERACT, target_id=block)
            for block in self.__truth.blocks
        )
        for station in (self.__truth.machine_id, *self.__truth.processors):
            actions.extend(
                PrimitiveAction(
                    kind=ActionKind.PLACE_INPUT,
                    target_id=station,
                    item_id=item,
                )
                for item in self.__truth.resources
            )
            actions.append(PrimitiveAction(kind=ActionKind.ACTIVATE, target_id=station))
            actions.append(PrimitiveAction(kind=ActionKind.INTERACT, target_id=station))
        actions.extend(
            (
                PrimitiveAction(kind=ActionKind.WAIT, ticks=1),
                PrimitiveAction(kind=ActionKind.WAIT, ticks=2),
            )
        )
        return tuple(actions)

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        before = self.observe()
        self.__step += 1
        event, compliant, failure = self.__apply(action)
        self.__last_event = event
        after = self.observe()
        return TransitionRecord(
            transition_id=f"ep{self.__episode_id:03d}-step{self.__step:05d}",
            context={"episode_id": self.__episode_id},
            state_before=before,
            executed_action=action,
            intervention_compliance=compliant,
            state_after=after,
            observed_delta={
                "event_type": event.event_type,
                "success": event.success,
                "output_id": event.output_id,
                "duration": event.duration,
            },
            event=event,
            controller_failure=failure,
            provenance={"simulator": "structured-mvp", "step": self.__step},
        )

    def __apply(self, action: PrimitiveAction) -> tuple[WorldEvent, bool, str | None]:
        if action.kind is ActionKind.SELECT:
            if action.target_id not in self.__truth.tools:
                return self.__invalid(action, "not_selectable")
            self.__selected = action.target_id
            return (
                WorldEvent(event_type="selection", target_id=action.target_id),
                True,
                None,
            )

        if (
            action.kind is ActionKind.INTERACT
            and action.target_id in self.__truth.blocks
        ):
            if self.__selected is None:
                return self.__invalid(action, "nothing_selected")
            success = self.__truth.tool_outcome(
                self.__selected, action.target_id, self.__episode_id
            )
            rank = self.__truth.tools.index(self.__selected)
            requirement = self.__truth.requirements(self.__episode_id)[action.target_id]
            duration = 1 + max(0, requirement - rank)
            return (
                WorldEvent(
                    event_type="interaction_result",
                    target_id=action.target_id,
                    success=success,
                    duration=duration,
                    details={"selected_id": self.__selected},
                ),
                True,
                None,
            )

        stations = (self.__truth.machine_id, *self.__truth.processors)
        if action.kind is ActionKind.PLACE_INPUT:
            if (
                action.target_id not in stations
                or action.item_id not in self.__truth.resources
            ):
                return self.__invalid(action, "invalid_station_input")
            self.__sequences.setdefault(action.target_id, []).append("place")
            self.__pending_inputs[action.target_id] = action.item_id
            return (
                WorldEvent(event_type="input_placed", target_id=action.target_id),
                True,
                None,
            )

        if action.kind is ActionKind.ACTIVATE:
            if action.target_id not in stations:
                return self.__invalid(action, "invalid_station")
            self.__sequences.setdefault(action.target_id, []).append("activate")
            return (
                WorldEvent(event_type="activation", target_id=action.target_id),
                True,
                None,
            )

        if action.kind is ActionKind.WAIT:
            for station in tuple(self.__sequences):
                self.__sequences[station].append(f"wait:{action.ticks}")
            return (
                WorldEvent(event_type="time_passed", duration=action.ticks),
                True,
                None,
            )

        if action.kind is ActionKind.INTERACT and action.target_id in stations:
            station = action.target_id
            sequence = (*self.__sequences.get(station, []), "interact")
            input_id = self.__pending_inputs.get(station)
            if station == self.__truth.machine_id:
                expected_input = self.__truth.machine_input
                expected_protocol = self.__truth.machine_protocol
                output = self.__truth.machine_output
            else:
                expected_input, output = self.__truth.transformations[station]
                expected_protocol = ("place", "activate", "wait:1", "interact")
            success = input_id == expected_input and self.__protocol_matches(
                expected_protocol, sequence
            )
            self.__sequences.pop(station, None)
            self.__pending_inputs.pop(station, None)
            return (
                WorldEvent(
                    event_type="station_result",
                    target_id=station,
                    success=success,
                    output_id=output if success else None,
                    details={"protocol_length": len(sequence)},
                ),
                True,
                None,
            )

        return self.__invalid(action, "unsupported_action")

    @staticmethod
    def __protocol_matches(
        expected: tuple[str, ...], observed: tuple[str, ...]
    ) -> bool:
        if expected == observed:
            return True
        expected_waits = [token for token in expected if token.startswith("wait:")]
        observed_waits = [token for token in observed if token.startswith("wait:")]
        if len(expected_waits) != 1 or len(observed_waits) != 1:
            return False
        expected_shape = tuple(
            "wait" if token.startswith("wait:") else token for token in expected
        )
        observed_shape = tuple(
            "wait" if token.startswith("wait:") else token for token in observed
        )
        return (
            expected_shape == observed_shape
            and int(observed_waits[0].split(":", 1)[1])
            >= int(expected_waits[0].split(":", 1)[1])
        )

    @staticmethod
    def __invalid(action: PrimitiveAction, reason: str) -> tuple[WorldEvent, bool, str]:
        return (
            WorldEvent(
                event_type="control_failure",
                target_id=action.target_id,
                success=False,
                details={"reason": reason},
            ),
            False,
            reason,
        )


class WorldFactory:
    """Builds paired public/private capabilities from one deterministic seed."""

    @staticmethod
    def create(seed: int, change_episode: int | None = 1) -> WorldBundle:
        rng = random.Random(seed)
        numbers = rng.sample(range(100, 999), 14)
        ids = tuple(f"obj_{number:03d}" for number in numbers)
        tools = ids[0:3]
        blocks = ids[3:6]
        machine_id = ids[6]
        processors = ids[7:10]
        resources = ids[10:14]

        block_order = list(blocks)
        rng.shuffle(block_order)
        base_requirements = {
            block_order[0]: 0,
            block_order[1]: 1,
            block_order[2]: 2,
        }
        shifted_requirements = {
            block_order[0]: 2,
            block_order[1]: 0,
            block_order[2]: 1,
        }
        protocols = (
            ("place", "activate", "wait:1", "interact"),
            ("place", "activate", "wait:2", "interact"),
            ("activate", "place", "wait:1", "interact"),
            ("place", "wait:1", "activate", "interact"),
            ("place", "activate", "interact"),
        )
        machine_protocol = protocols[rng.randrange(len(protocols))]
        truth = _GroundTruthSpec(
            tools=tools,
            blocks=blocks,
            base_requirements=base_requirements,
            shifted_requirements=shifted_requirements,
            machine_id=machine_id,
            machine_input=resources[rng.randrange(len(resources))],
            machine_output=resources[-1],
            machine_protocol=machine_protocol,
            processors=processors,
            resources=resources,
            transformations={
                processors[index]: (resources[index], resources[index + 1])
                for index in range(3)
            },
            change_episode=change_episode,
        )
        world = _StructuredTestWorld(truth)
        world.reset(seed)
        view = AgentWorldView(
            observe=world.observe,
            available=world.available_actions,
            step=world.step,
            begin_trial=world.begin_trial,
        )
        return WorldBundle(
            agent_view=view,
            controller=WorldController(world),
            evaluator=WorldEvaluator(truth),
        )
