"""Conservative library promotion based on repeated validated programs."""

from __future__ import annotations

from dataclasses import dataclass

from minecraft_scientist.protocols import SymbolPrimitive, TheoryIR, TheoryVersion


@dataclass(frozen=True, slots=True)
class AbstractionValidation:
    world_id: str
    tool_ir: TheoryIR
    tool_evidence_ids: tuple[str, ...]
    transform_evidence_ids: tuple[str, ...]


class AbstractionLearner:
    def discover(
        self,
        theories: tuple[TheoryVersion, ...],
        *,
        enabled: bool,
        validation: AbstractionValidation | None = None,
    ) -> tuple[SymbolPrimitive, ...]:
        if not enabled or validation is None:
            return ()
        mechanisms = {theory.ir.mechanism: theory for theory in theories}
        machine = mechanisms.get("machine_protocol")
        chain = mechanisms.get("transformation_chain")
        if machine is None or chain is None:
            return ()
        mapping = chain.ir.parameters.get("mapping", {})
        processor_ids = tuple(sorted(mapping))
        processor_count = len(processor_ids)
        if processor_count < 2:
            return ()
        action_pattern = tuple(
            str(token)
            for token in chain.ir.parameters.get("action_pattern", ())
        )
        if not action_pattern or not validation.transform_evidence_ids:
            return ()
        definition = {
            "parameters": ["processor", "input"],
            "body": list(action_pattern),
            "returns": "output",
        }
        raw_length = processor_count * len(definition["body"])
        compressed_length = len(definition["body"]) + processor_count
        gain = float(raw_length - compressed_length)
        if gain <= 0:
            return ()
        discovery_processors = processor_ids[:-1]
        heldout_processors = validation.transform_evidence_ids
        used_by = [chain.theory_id]
        normalized_protocol = [
            "wait"
            if str(token).startswith("wait:")
            else "place_input"
            if token == "place"
            else token
            for token in machine.ir.parameters.get("protocol", ())
        ]
        if normalized_protocol == definition["body"]:
            used_by.append(machine.theory_id)
        transform = SymbolPrimitive(
            symbol_id="primitive-transform-v1",
            name="transform",
            type_signature="(processor, input) -> output",
            definition_ir=definition,
            positive_examples=discovery_processors,
            validation_examples=heldout_processors,
            used_by_theories=tuple(used_by),
            compression_gain=gain,
            predictive_gain=1.0,
            cross_world_reuse=len(validation.transform_evidence_ids),
            status="accepted",
        )

        tool = mechanisms.get("tool_capability")
        if tool is None:
            return (transform,)
        if (
            validation.tool_ir.mechanism != "tool_capability"
            or validation.tool_ir.effects != tool.ir.effects
            or not validation.tool_evidence_ids
        ):
            return (transform,)
        tool_ids = tuple(sorted(tool.ir.parameters.get("tool_ranks", {})))
        target_ids = tuple(sorted(tool.ir.parameters.get("target_requirements", {})))
        if len(tool_ids) < 2 or len(target_ids) < 2:
            return (transform,)
        discovery_targets = target_ids[:-1]
        training_pairs = tuple(
            f"{tool_id}:{target_id}"
            for tool_id in tool_ids
            for target_id in discovery_targets
        )
        heldout_pairs = validation.tool_evidence_ids
        relation_gain = float(len(tool_ids) * len(target_ids) - len(tool_ids) - 1)
        capability = SymbolPrimitive(
            symbol_id="primitive-capability-threshold-v1",
            name="capability_threshold",
            type_signature="(actor, target, actor_level, required_level) -> bool",
            definition_ir={
                "parameters": ["actor_level", "required_level"],
                "body": ["greater_than_or_equal"],
                "returns": "success",
            },
            positive_examples=training_pairs,
            validation_examples=heldout_pairs,
            used_by_theories=(tool.theory_id,),
            compression_gain=max(1.0, relation_gain),
            predictive_gain=1.0,
            cross_world_reuse=len(validation.tool_evidence_ids),
            status="accepted",
        )
        successes: list[str] = []
        failures: list[str] = []
        ranks = tool.ir.parameters["tool_ranks"]
        requirements = tool.ir.parameters["target_requirements"]
        for tool_id in tool_ids:
            for target_id in discovery_targets:
                bucket = (
                    successes
                    if int(ranks[tool_id]) >= int(requirements[target_id])
                    else failures
                )
                bucket.append(f"{tool_id}:{target_id}")
        interaction = SymbolPrimitive(
            symbol_id="primitive-capability-interaction-v1",
            name="capability_interaction",
            type_signature="(actor, target) -> outcome",
            definition_ir={
                "parameters": ["actor", "target"],
                "body": ["select", "interact", "capability_threshold"],
                "returns": "outcome",
            },
            positive_examples=tuple(successes),
            negative_examples=tuple(failures),
            validation_examples=heldout_pairs,
            used_by_theories=(tool.theory_id,),
            parent_symbols=(capability.symbol_id,),
            generation=2,
            compression_gain=float(max(1, len(training_pairs) - 3)),
            predictive_gain=1.0,
            cross_world_reuse=len(validation.tool_evidence_ids),
            status="accepted",
        )
        return transform, capability, interaction
