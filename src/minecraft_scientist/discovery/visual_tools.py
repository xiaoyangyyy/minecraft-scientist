"""Active tool hypothesis elimination from controlled, label-free pixel evidence."""
from dataclasses import dataclass
from typing import Callable

from minecraft_scientist.discovery.hypotheses import ToolVersionSpace
from minecraft_scientist.minecraft.visual_evidence import PixelChangeEvidence
from minecraft_scientist.protocols import TheoryIR


@dataclass(frozen=True)
class VisualExperiment:
    tool_id: str
    target_id: str
    evidence_id: str
    evidence: PixelChangeEvidence


@dataclass(frozen=True)
class VisualDiscoveryResult:
    theory: TheoryIR | None
    experiments: tuple[VisualExperiment, ...]
    resolved: bool
    candidate_count: int


def discover_visual_tools(
    tools: tuple[str, ...],
    targets: tuple[str, ...],
    probe: Callable[[str, str, int], tuple[str, PixelChangeEvidence]],
    *,
    budget: int,
    execution_context: dict,
) -> VisualDiscoveryResult:
    """Choose live probes by information gain; unstable controls never fit labels.

    The fixture owns physical identity bindings. The learner sees only opaque IDs
    and cropped-image evidence. This is not a visual object identity learner.
    """
    if budget < 1 or len(tools) < 2 or not targets:
        raise ValueError("positive budget, at least two tools and nonempty targets required")
    if any(not value.startswith("obj_") for value in (*tools, *targets)):
        raise ValueError("visual discovery requires opaque IDs")
    if len(set(tools)) != len(tools) or len(set(targets)) != len(targets):
        raise ValueError("duplicate IDs")
    if not execution_context:
        raise ValueError("visual theories require an explicit execution context")
    space = ToolVersionSpace(tools, targets, include_unreachable=True)
    experiments = []
    for index in range(budget):
        if space.resolved():
            break
        pair = space.next_experiment()
        if pair is None:
            break
        tool_id, target_id = pair
        evidence_id, evidence = probe(tool_id, target_id, index)
        experiments.append(VisualExperiment(tool_id, target_id, evidence_id, evidence))
        if not evidence.control_stable or evidence.changed is None:
            space.record_control_failure(tool_id, target_id)
            continue
        space.observe(tool_id, target_id, evidence.changed, evidence_id)
    theory = None
    if space.resolved():
        theory = space.to_ir()
        theory = theory.model_copy(update={"scope": {
            **theory.scope, "required_context": dict(execution_context)
        }})
    return VisualDiscoveryResult(theory, tuple(experiments), space.resolved(),
                                 len(space.candidates))
