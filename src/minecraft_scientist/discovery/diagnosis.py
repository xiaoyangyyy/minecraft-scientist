"""Joint controller/mechanism hypotheses, compiled into the shared finite IR.

Attribution is conditional on the supplied model family. Behavior identification
does NOT imply that controller and mechanism factors are separately identifiable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram

NO_DELIVERY = "_no_delivery_"


class JointHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mechanism_id: StrictStr = Field(min_length=1, max_length=80)
    controller_id: StrictStr = Field(min_length=1, max_length=80)
    mechanism_ir: TheoryIR
    controller_ir: TheoryIR

    def compile(self) -> TheoryIR:
        mechanism = FiniteStateProgram.from_ir(self.mechanism_ir)
        controller = FiniteStateProgram.from_ir(self.controller_ir)
        if NO_DELIVERY in mechanism.actions or any(
            output not in (*mechanism.actions, NO_DELIVERY)
            for output in controller.outputs
        ):
            raise ValueError(
                "controller outputs must be mechanism actions or nondelivery"
            )
        if len(mechanism.states) * len(controller.states) > 16:
            raise ValueError("joint state product exceeds the shared IR cap")
        names = {
            (mstate, cstate): f"joint-{mi}-{ci}"
            for mi, mstate in enumerate(mechanism.states)
            for ci, cstate in enumerate(controller.states)
        }
        rows = []
        for (mstate, cstate), source in names.items():
            for command in controller.actions:
                control = controller.table[cstate, command]
                target, output = mstate, mechanism.outputs[0]
                if control.output != NO_DELIVERY:
                    row = mechanism.table[mstate, control.output]
                    target, output = row.target, row.output
                rows.append(
                    {
                        "source": source,
                        "action": command,
                        "target": names[target, control.target],
                        "output": output,
                    }
                )
        program = FiniteStateProgram(
            states=tuple(names.values()),
            actions=controller.actions,
            outputs=mechanism.outputs,
            initial_state=names[mechanism.initial_state, controller.initial_state],
            transitions=tuple(rows),
        )
        return program.to_ir().model_copy(
            update={
                "composition": (self.mechanism_id, self.controller_id),
                "latent_variables": ("mechanism_state", "controller_state"),
                "scope": {
                    "language": program.language,
                    "reset": "both-initial-states",
                    "treatment": "issued-command; delivery-latent",
                },
            }
        )


def validate_hypotheses(hypotheses, baseline_mechanism, baseline_controller):
    if type(hypotheses) is not tuple or not 1 <= len(hypotheses) <= 12:
        raise ValueError("one to twelve joint hypotheses are required")
    pairs, definitions = set(), {}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, JointHypothesis):
            raise TypeError("validated JointHypothesis objects are required")
        pair = hypothesis.mechanism_id, hypothesis.controller_id
        if pair in pairs:
            raise ValueError("duplicate joint identifier pair")
        pairs.add(pair)
        for kind, identifier, ir in (
            ("mechanism", hypothesis.mechanism_id, hypothesis.mechanism_ir),
            ("controller", hypothesis.controller_id, hypothesis.controller_ir),
        ):
            key, value = (kind, identifier), ir.model_dump(mode="json")
            if key in definitions and definitions[key] != value:
                raise ValueError("a factor identifier cannot redefine its program")
            definitions[key] = value
    if (baseline_mechanism, baseline_controller) not in pairs:
        raise ValueError("the declared baseline pair must be in the candidate family")
    return tuple(hypothesis.compile() for hypothesis in hypotheses)


def project_diagnosis(
    hypotheses, identification, baseline_mechanism, baseline_controller
):
    """Conservative projections over ALL surviving factors, not a representative."""
    remaining = identification["remaining_candidates"]
    mechanisms = sorted({hypotheses[i].mechanism_id for i in remaining})
    controllers = sorted({hypotheses[i].controller_id for i in remaining})
    resolved = identification["status"] == "conditional_behavior_identified"

    def change_status(values, baseline):
        if not remaining:
            return "model_mismatch"
        if values == [baseline]:
            return "unchanged"
        if baseline not in values:
            return "changed"
        return "ambiguous" if resolved else "unresolved"

    mechanism_status = change_status(mechanisms, baseline_mechanism)
    controller_status = change_status(controllers, baseline_controller)
    return {
        "schema": "joint-factor-diagnosis-v1",
        "mechanism_status": mechanism_status,
        "controller_status": controller_status,
        "possible_mechanisms": mechanisms,
        "possible_controllers": controllers,
        "mechanism_revision_warranted": mechanism_status == "changed",
        "mechanism_revision_completed": False,
        "unique_factor_pair_identified": resolved and len(remaining) == 1,
        "joint_behavior_identified": resolved,
        "factor_attribution_basis": "all surviving models given acquired evidence",
        "independent_validation_completed": False,
        "scope": "deterministic supplied joint family, not arbitrary causal attribution",
    }
