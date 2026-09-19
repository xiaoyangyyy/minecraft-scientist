"""Execution and canonicalization of the unified theory intermediate representation."""

from __future__ import annotations

import json
from typing import Any

from minecraft_scientist.protocols import TheoryIR


class TheoryInterpreter:
    @staticmethod
    def predict(ir: TheoryIR, **inputs: Any) -> dict[str, Any]:
        required_context = ir.scope.get("required_context")
        if required_context is not None:
            actual_context = inputs.get("execution_context")
            if not isinstance(required_context, dict) or not isinstance(actual_context, dict):
                raise ValueError("scoped theory requires an execution_context mapping")
            if any(key not in actual_context or actual_context[key] != value
                   for key, value in required_context.items()):
                raise ValueError("execution context is outside the theory scope")
        mechanism = ir.mechanism
        if mechanism == "tool_capability":
            tool_id = str(inputs["tool_id"])
            target_id = str(inputs["target_id"])
            ranks = ir.parameters["tool_ranks"]
            requirements = ir.parameters["target_requirements"]
            return {"success": int(ranks[tool_id]) >= int(requirements[target_id])}
        if mechanism == "machine_protocol":
            expected_protocol = tuple(ir.parameters["protocol"])
            observed_protocol = tuple(inputs["protocol"])
            exact_or_sufficient_wait = expected_protocol == observed_protocol
            expected_waits = [
                token for token in expected_protocol if token.startswith("wait:")
            ]
            observed_waits = [
                token for token in observed_protocol if token.startswith("wait:")
            ]
            if len(expected_waits) == len(observed_waits) == 1:
                expected_shape = tuple(
                    "wait" if token.startswith("wait:") else token
                    for token in expected_protocol
                )
                observed_shape = tuple(
                    "wait" if token.startswith("wait:") else token
                    for token in observed_protocol
                )
                exact_or_sufficient_wait = (
                    expected_shape == observed_shape
                    and int(observed_waits[0].split(":", 1)[1])
                    >= int(expected_waits[0].split(":", 1)[1])
                )
            return {
                "success": (
                    inputs["input_id"] == ir.parameters["input_id"]
                    and exact_or_sufficient_wait
                ),
                "output_id": ir.effects.get("output_id"),
            }
        if mechanism == "transformation_chain":
            mapping = ir.parameters["mapping"]
            pair = mapping.get(str(inputs["processor_id"]))
            success = pair is not None and pair[0] == inputs["input_id"]
            return {"success": success, "output_id": pair[1] if success else None}
        if mechanism == "fall_damage":
            height = float(inputs["height"])
            armor_id = str(inputs["armor_id"])
            safe_height = float(ir.parameters["safe_height"])
            slopes = ir.parameters["armor_slopes"]
            damage = float(slopes[armor_id]) * max(0.0, height - safe_height)
            return {"damage": damage}
        raise ValueError(f"Unknown mechanism: {mechanism}")

    @staticmethod
    def canonical_payload(ir: TheoryIR) -> str:
        return json.dumps(
            ir.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def description_length(cls, ir: TheoryIR) -> int:
        return len(cls.canonical_payload(ir))
