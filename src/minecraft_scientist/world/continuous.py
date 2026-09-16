"""Opaque intervention world for discovering a piecewise fall-damage equation."""

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
class _FallTruth:
    armor_slopes: dict[str, float]
    platform_heights: dict[str, int]
    safe_height: int

    def damage(self, armor_id: str, height: int) -> float:
        return self.armor_slopes[armor_id] * max(0, height - self.safe_height)


class FallWorldEvaluator:
    __slots__ = ("__truth",)

    def __init__(self, truth: _FallTruth) -> None:
        self.__truth = truth

    @property
    def armor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.__truth.armor_slopes))

    @property
    def heights(self) -> tuple[int, ...]:
        return tuple(sorted(self.__truth.platform_heights.values()))

    def damage(self, armor_id: str, height: int) -> float:
        return self.__truth.damage(armor_id, height)


class _FallWorld:
    def __init__(self, truth: _FallTruth) -> None:
        self.__truth = truth
        self.__selected: str | None = None
        self.__step = 0
        self.__last_event: WorldEvent | None = None

    def reset(self, seed: int) -> Observation:
        del seed
        self.__selected = None
        self.__step = 0
        self.__last_event = None
        return self.observe()

    def begin_trial(self) -> Observation:
        self.__selected = None
        self.__last_event = None
        return self.observe()

    def observe(self) -> Observation:
        armor = tuple(
            ObjectObservation(
                object_id=armor_id,
                affordances=(ActionKind.SELECT,),
                observable_features={"form_token": f"f{index % 2}"},
            )
            for index, armor_id in enumerate(sorted(self.__truth.armor_slopes))
        )
        platforms = tuple(
            ObjectObservation(
                object_id=platform_id,
                affordances=(ActionKind.INTERACT,),
                observable_features={"height": height},
            )
            for platform_id, height in sorted(self.__truth.platform_heights.items())
        )
        return Observation(
            episode_id=0,
            step=self.__step,
            inventory={armor_id: 1 for armor_id in self.__truth.armor_slopes},
            selected_id=self.__selected,
            objects=armor + platforms,
            last_event=self.__last_event,
        )

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        return tuple(
            [
                PrimitiveAction(kind=ActionKind.SELECT, target_id=armor_id)
                for armor_id in sorted(self.__truth.armor_slopes)
            ]
            + [
                PrimitiveAction(kind=ActionKind.INTERACT, target_id=platform_id)
                for platform_id in sorted(self.__truth.platform_heights)
            ]
        )

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        before = self.observe()
        self.__step += 1
        compliant = True
        failure: str | None = None
        if (
            action.kind is ActionKind.SELECT
            and action.target_id in self.__truth.armor_slopes
        ):
            self.__selected = action.target_id
            event = WorldEvent(event_type="armor_selected", target_id=action.target_id)
        elif (
            action.kind is ActionKind.INTERACT
            and action.target_id in self.__truth.platform_heights
            and self.__selected is not None
        ):
            height = self.__truth.platform_heights[action.target_id]
            event = WorldEvent(
                event_type="fall_result",
                target_id=action.target_id,
                details={
                    "height": height,
                    "damage": self.__truth.damage(self.__selected, height),
                },
            )
        else:
            compliant = False
            failure = "invalid_fall_intervention"
            event = WorldEvent(
                event_type="control_failure",
                target_id=action.target_id,
                success=False,
                details={"reason": failure},
            )
        self.__last_event = event
        after = self.observe()
        return TransitionRecord(
            transition_id=f"fall-step-{self.__step:05d}",
            context={"selected_armor": before.selected_id or "none"},
            executed_action=action,
            intervention_compliance=compliant,
            state_after=after,
            state_before=before,
            observed_delta=event.details,
            event=event,
            controller_failure=failure,
            provenance={"simulator": "continuous-fall", "step": self.__step},
        )


@dataclass(frozen=True, slots=True)
class FallWorldBundle:
    agent_view: AgentWorldView
    evaluator: FallWorldEvaluator


class FallWorldFactory:
    @staticmethod
    def create(
        seed: int,
        *,
        law_seed: int | None = None,
        rule_shift: bool = False,
    ) -> FallWorldBundle:
        rng = random.Random(seed)
        law_rng = random.Random(seed if law_seed is None else law_seed)
        numbers = rng.sample(range(1000, 9999), 13)
        armor_ids = tuple(f"obj_{number}" for number in numbers[:3])
        platform_ids = tuple(f"obj_{number}" for number in numbers[3:])
        slopes = [0.4, 0.7, 1.0]
        law_rng.shuffle(slopes)
        safe_height = law_rng.choice((2, 3, 4))
        if rule_shift:
            slopes = slopes[1:] + slopes[:1]
            safe_height = {2: 3, 3: 4, 4: 2}[safe_height]
        heights = list(range(1, 11))
        rng.shuffle(heights)
        truth = _FallTruth(
            armor_slopes=dict(zip(armor_ids, slopes, strict=True)),
            platform_heights=dict(zip(platform_ids, heights, strict=True)),
            safe_height=safe_height,
        )
        world = _FallWorld(truth)
        world.reset(seed)
        return FallWorldBundle(
            agent_view=AgentWorldView(
                observe=world.observe,
                available=world.available_actions,
                step=world.step,
                begin_trial=world.begin_trial,
            ),
            evaluator=FallWorldEvaluator(truth),
        )
