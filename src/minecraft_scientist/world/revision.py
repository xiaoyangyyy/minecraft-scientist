"""One hidden online change, executed as raw controller/mechanism components.

The Agent gets bootstrap alternatives, never the post-change rule or boundary.
These fixed development edits are not structural train/test holdouts.
"""

from dataclasses import dataclass

from minecraft_scientist.discovery.diagnosis import JointHypothesis
from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.diagnosis import _ControlledWorld, diagnosis_task
from minecraft_scientist.world.identification import _program

REVISION_CASES = (
    "unchanged",
    "controller_fault",
    "redirect_start",
    "redirect_wait",
    "retain_ready",
    "state_insert",
    "outside_grammar",
)


@dataclass(frozen=True)
class RevisionTask:
    bootstrap: tuple[JointHypothesis, ...]
    calibration: tuple[str, ...]
    costs: dict[str, int]
    view: AgentWorldView
    private_spec: dict


class _SwitchingWorld(_ControlledWorld):
    def __init__(self, before, after, boundary):
        super().__init__(*before)
        self._before, self._after, self._boundary = before, after, boundary

    def begin_trial(self):
        mechanism, controller = (
            self._after if self._episode + 1 >= self._boundary else self._before
        )
        self._initial = mechanism.initial_state
        self._rows = {
            (row.source, row.action): (row.target, row.output)
            for row in mechanism.transitions
        }
        self._controller_initial = controller.initial_state
        self._control_rows = {
            (row.source, row.action): (row.target, row.output)
            for row in controller.transitions
        }
        return super().begin_trial()


def revision_task(seed=0, case="redirect_start", *, boundary=8):
    if (
        type(seed) is not int
        or seed < 0
        or case not in REVISION_CASES
        or type(boundary) is not int
        or not 6 <= boundary <= 100
    ):
        raise ValueError("invalid online revision development world")
    base_task = diagnosis_task(seed, "unchanged", "healthy", bypass=True)
    base_program = FiniteStateProgram.from_ir(base_task.hypotheses[0].mechanism_ir)
    controllers = tuple(base_task.hypotheses[i].controller_ir for i in range(3))
    mechanisms = (
        base_program,
        _program(base_program.actions, base_program.outputs, activation=False),
        _program(base_program.actions, base_program.outputs, toggle=True),
    )
    bootstrap = tuple(
        JointHypothesis(
            mechanism_id=f"bootstrap-mechanism-{mi}",
            controller_id=f"bootstrap-controller-{ci}",
            mechanism_ir=program.to_ir(),
            controller_ir=control,
        )
        for mi, program in enumerate(mechanisms)
        for ci, control in enumerate(controllers)
    )
    old = mechanisms[2 if seed % 2 else 0]
    changed = old.model_dump(mode="json")
    load, start, wait, take = old.actions
    loaded = old.table[old.initial_state, load].target
    active = old.table[loaded, start].target
    ready = old.table[active, wait].target
    for row in changed["transitions"]:
        if case == "redirect_start" and (row["source"], row["action"]) == (
            loaded,
            start,
        ):
            row["target"] = loaded
        elif case == "redirect_wait" and (row["source"], row["action"]) == (
            active,
            wait,
        ):
            row["target"] = old.initial_state
        elif case == "retain_ready" and (row["source"], row["action"]) == (ready, take):
            row["target"] = ready
        elif case == "outside_grammar" and (row["source"], row["action"]) in (
            (old.initial_state, load),
            (active, wait),
        ):
            row["output"] = old.outputs[1]
    if case == "state_insert":
        name = "world-extra-state"
        changed["states"].append(name)
        for action in old.actions:
            original = old.table[active, action]
            changed["transitions"].append(
                {
                    "source": name,
                    "action": action,
                    "target": name if original.target == active else original.target,
                    "output": original.output,
                }
            )
        for row in changed["transitions"]:
            if (row["source"], row["action"]) == (active, wait):
                row["target"] = name
    after_mechanism = FiniteStateProgram.model_validate(changed)
    healthy = FiniteStateProgram.from_ir(controllers[0])
    after_controller = (
        FiniteStateProgram.from_ir(controllers[1])
        if case == "controller_fault"
        else healthy
    )
    world = _SwitchingWorld(
        (old, healthy), (after_mechanism, after_controller), boundary
    )
    private = {
        "schema": "online-revision-private-v1",
        "boundary_episode": boundary,
        "before": {
            "mechanism": old.model_dump(mode="json"),
            "controller": healthy.model_dump(mode="json"),
        },
        "after": {
            "mechanism": after_mechanism.model_dump(mode="json"),
            "controller": after_controller.model_dump(mode="json"),
        },
    }
    view = AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )
    return RevisionTask(
        bootstrap, base_task.calibration, base_task.costs, view, private
    )
