"""Evaluator-owned hidden episode switch on anonymous finite development tables."""

from dataclasses import dataclass

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.active_automata import automata_task
from minecraft_scientist.world.identification import _FiniteWorld

CASES = ("unchanged", "output_change", "redirect", "state_growth")


@dataclass(frozen=True)
class OnlineTask:
    actions: tuple[str, ...]
    outputs: tuple[str, ...]
    costs: dict[str, int]
    view: AgentWorldView
    private: dict  # Runner/evaluator only; never passed to the learner.


class _ChangingFiniteWorld(_FiniteWorld):
    def __init__(self, before, after, boundary):
        super().__init__(before)
        self._before, self._after, self._boundary = before, after, boundary

    def begin_trial(self):
        program = self._after if self._episode + 1 >= self._boundary else self._before
        self._initial = program.initial_state
        self._rows = {
            (r.source, r.action): (r.target, r.output) for r in program.transitions
        }
        return super().begin_trial()


def online_task(seed=0, case="state_growth", *, boundary=16):
    if (
        type(seed) is not int
        or seed < 0
        or type(case) is not str
        or case not in CASES
        or type(boundary) is not int
        or not 0 <= boundary <= 1000
    ):
        raise ValueError("invalid hidden online development task")
    old = automata_task(seed, "cycle", 2)
    before = old.ground_truth
    raw = before.model_dump(mode="json")
    if case == "state_growth":
        raw = automata_task(seed, "cycle", 3).ground_truth.model_dump(mode="json")
    elif case == "output_change":
        row = next(r for r in raw["transitions"] if r["output"] == old.outputs[1])
        row["output"] = old.outputs[0]
    elif case == "redirect":
        row = next(
            r
            for r in raw["transitions"]
            if r["source"] == before.initial_state and r["action"] == old.actions[0]
        )
        row["target"] = before.initial_state
    after = FiniteStateProgram.model_validate(raw)
    world = _ChangingFiniteWorld(before, after, boundary)
    return OnlineTask(
        old.actions,
        old.outputs,
        old.costs,
        AgentWorldView(
            observe=world.observe,
            available=world.available_actions,
            step=world.step,
            begin_trial=world.begin_trial,
        ),
        {
            "schema": "online-automata-private-v1",
            "boundary_episode": boundary,
            "before": {"program": before.model_dump(mode="json")},
            "after": {"program": after.model_dump(mode="json")},
        },
    )
