"""World protocols and the capability-limited agent facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from .models import Observation, PrimitiveAction, TransitionRecord


@runtime_checkable
class StructuredWorld(Protocol):
    def reset(self, seed: int) -> Observation: ...

    def observe(self) -> Observation: ...

    def available_actions(self) -> tuple[PrimitiveAction, ...]: ...

    def step(self, action: PrimitiveAction) -> TransitionRecord: ...


@dataclass(frozen=True, slots=True)
class _WorldCapabilities:
    observe: Callable[[], Observation]
    available: Callable[[], tuple[PrimitiveAction, ...]]
    step: Callable[[PrimitiveAction], TransitionRecord]
    begin_trial: Callable[[], Observation]


_CAPABILITIES: WeakKeyDictionary[AgentWorldView, _WorldCapabilities]


class AgentWorldView:
    """A narrow capability object: hidden rules never enter the public state."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        *,
        observe: Callable[[], Observation],
        available: Callable[[], tuple[PrimitiveAction, ...]],
        step: Callable[[PrimitiveAction], TransitionRecord],
        begin_trial: Callable[[], Observation],
    ) -> None:
        _CAPABILITIES[self] = _WorldCapabilities(
            observe=observe,
            available=available,
            step=step,
            begin_trial=begin_trial,
        )

    def observe(self) -> Observation:
        return _CAPABILITIES[self].observe()

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        return _CAPABILITIES[self].available()

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        return _CAPABILITIES[self].step(action)

    def begin_trial(self) -> Observation:
        return _CAPABILITIES[self].begin_trial()

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("AgentWorldView capabilities cannot be serialized")


_CAPABILITIES = WeakKeyDictionary()
