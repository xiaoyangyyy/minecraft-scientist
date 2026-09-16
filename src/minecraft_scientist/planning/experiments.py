"""Translate epistemic interventions into primitive action traces."""

from __future__ import annotations

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    ExperimentResult,
    InterventionSpec,
    PrimitiveAction,
    TransitionRecord,
)


class ExperimentExecutor:
    def __init__(self, world: AgentWorldView) -> None:
        self.world = world
        self._counter = 0

    def run_tool_test(
        self,
        tool_id: str,
        target_id: str,
        *,
        predicted_outcomes: dict[str, object] | None = None,
    ) -> ExperimentResult:
        intervention = self._intervention(
            "tool_capability",
            {"tool_id": tool_id, "target_id": target_id},
            "success",
            predicted_outcomes=predicted_outcomes,
        )
        self.world.begin_trial()
        transitions = self._execute(
            intervention,
            (
                PrimitiveAction(kind=ActionKind.SELECT, target_id=tool_id),
                PrimitiveAction(kind=ActionKind.INTERACT, target_id=target_id),
            ),
        )
        event = transitions[-1].event
        return ExperimentResult(
            experiment_id=intervention.intervention_id,
            mechanism=intervention.mechanism,
            intervention=intervention,
            transitions=transitions,
            outcome={"success": event.success, "duration": event.duration},
            compliant=all(t.intervention_compliance for t in transitions),
            cost=float(len(transitions)),
        )

    def run_protocol_test(
        self,
        machine_id: str,
        input_id: str,
        protocol: tuple[str, ...],
        *,
        mechanism: str = "machine_protocol",
        predicted_outcomes: dict[str, object] | None = None,
    ) -> ExperimentResult:
        intervention = self._intervention(
            mechanism,
            {
                "machine_id": machine_id,
                "input_id": input_id,
                "protocol": "|".join(protocol),
            },
            "output_id",
            predicted_outcomes=predicted_outcomes,
        )
        self.world.begin_trial()
        actions = tuple(
            self._protocol_action(token, machine_id, input_id) for token in protocol
        )
        transitions = self._execute(intervention, actions)
        event = transitions[-1].event
        return ExperimentResult(
            experiment_id=intervention.intervention_id,
            mechanism=mechanism,
            intervention=intervention,
            transitions=transitions,
            outcome={"success": event.success, "output_id": event.output_id},
            compliant=all(t.intervention_compliance for t in transitions),
            cost=float(len(transitions)),
        )

    def _intervention(
        self,
        mechanism: str,
        controls: dict[str, str | int | float | bool],
        target: str,
        *,
        predicted_outcomes: dict[str, object] | None,
    ) -> InterventionSpec:
        self._counter += 1
        return InterventionSpec(
            intervention_id=f"experiment-{self._counter:04d}",
            mechanism=mechanism,
            controlled_variables=controls,
            target_variable=target,
            predicted_outcomes=predicted_outcomes or {},
        )

    def _execute(
        self,
        intervention: InterventionSpec,
        actions: tuple[PrimitiveAction, ...],
    ) -> tuple[TransitionRecord, ...]:
        transitions = []
        for action in actions:
            transition = self.world.step(action)
            transitions.append(
                transition.model_copy(update={"intended_intervention": intervention})
            )
        return tuple(transitions)

    @staticmethod
    def _protocol_action(token: str, target_id: str, input_id: str) -> PrimitiveAction:
        if token == "place":
            return PrimitiveAction(
                kind=ActionKind.PLACE_INPUT,
                target_id=target_id,
                item_id=input_id,
            )
        if token == "activate":
            return PrimitiveAction(kind=ActionKind.ACTIVATE, target_id=target_id)
        if token == "interact":
            return PrimitiveAction(kind=ActionKind.INTERACT, target_id=target_id)
        if token.startswith("wait:"):
            return PrimitiveAction(
                kind=ActionKind.WAIT, ticks=int(token.split(":", 1)[1])
            )
        raise ValueError(f"unknown protocol token: {token}")
