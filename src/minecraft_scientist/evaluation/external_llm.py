"""Cost-gated, reproducible external-LLM comparison baselines.

The model receives only public opaque observations and intervention outcomes.  The
privileged evaluator is used after the response solely to score predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from minecraft_scientist.protocols import ActionKind, Observation, PrimitiveAction
from minecraft_scientist.world import WorldFactory


class ExternalLLMAuthorizationError(RuntimeError):
    """Raised before network access when credentials or a cost gate are missing."""


class _Responses(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    responses: _Responses


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str
    target_id: str
    success: bool


class PredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictions: tuple[Prediction, ...]
    summary: str


class SkillMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    principles: tuple[str, ...] = Field(min_length=1, max_length=12)


@dataclass(frozen=True, slots=True)
class ExternalLLMConfig:
    model: str = "gpt-5.6-luna"
    first_seed: int = 0
    seeds: int = 3
    evidence_budget: int = 6
    max_api_calls: int = 9
    max_output_tokens: int = 1_200
    max_total_tokens: int = 50_000
    request_timeout_seconds: float = 120.0

    def validate(self) -> None:
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.evidence_budget < 0:
            raise ValueError("evidence_budget cannot be negative")
        if self.max_output_tokens < 1 or self.max_total_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        required = self.seeds * 3
        if self.max_api_calls < required:
            raise ValueError(
                f"max_api_calls={self.max_api_calls} is below the required {required}"
            )


class _BudgetedResponses:
    def __init__(
        self, client: _Client, config: ExternalLLMConfig, ledger_path: Path
    ) -> None:
        self._client = client
        self._config = config
        self.calls = 0
        self.total_tokens = 0
        self.records: list[dict[str, Any]] = []
        self._ledger_path = ledger_path

    def call(
        self,
        *,
        purpose: str,
        instructions: str,
        payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.calls >= self._config.max_api_calls:
            raise ExternalLLMAuthorizationError("maximum API call count reached")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        # UTF-8 bytes are a conservative upper bound for tokenizer input units.
        reserved = len((instructions + serialized).encode("utf-8"))
        reserved += self._config.max_output_tokens
        if self.total_tokens + reserved > self._config.max_total_tokens:
            raise ExternalLLMAuthorizationError(
                "the next request could exceed max_total_tokens"
            )
        response = self._client.responses.create(
            model=self._config.model,
            store=False,
            instructions=instructions,
            input=serialized,
            max_output_tokens=self._config.max_output_tokens,
            timeout=self._config.request_timeout_seconds,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        self.calls += 1
        usage = _usage(response)
        self.total_tokens += usage["total_tokens"]
        if self.total_tokens > self._config.max_total_tokens:
            raise ExternalLLMAuthorizationError(
                "provider-reported usage exceeded max_total_tokens; no more calls allowed"
            )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise RuntimeError("Responses API result did not contain output_text")
        record = {
            "purpose": purpose,
            "response_id": str(getattr(response, "id", "unknown")),
            "model": str(getattr(response, "model", self._config.model)),
            "status": str(getattr(response, "status", "completed")),
            "usage": usage,
            "prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        }
        self.records.append(record)
        with self._ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if record["status"] != "completed":
            raise RuntimeError(
                f"Responses API request ended with status {record['status']}"
            )
        value = json.loads(output_text)
        if not isinstance(value, dict):
            raise RuntimeError("structured model output must be a JSON object")
        return value


def _usage(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage", None)
    if raw is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def read(name: str) -> int:
        if isinstance(raw, dict):
            return int(raw.get(name, 0) or 0)
        return int(getattr(raw, name, 0) or 0)

    input_tokens = read("input_tokens")
    output_tokens = read("output_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": read("total_tokens") or input_tokens + output_tokens,
    }


def _public_context(observation: Observation) -> dict[str, Any]:
    return observation.model_dump(mode="json", exclude={"last_event"})


def _tool_target_ids(observation: Observation) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tools = tuple(
        item.object_id
        for item in observation.objects
        if item.affordances == (ActionKind.SELECT,)
    )
    targets = tuple(
        item.object_id
        for item in observation.objects
        if item.affordances == (ActionKind.INTERACT,)
    )
    return tools, targets


def _collect_public_evidence(seed: int, budget: int) -> dict[str, Any]:
    bundle = WorldFactory.create(seed, change_episode=None)
    view = bundle.agent_view
    observation = view.observe()
    tools, targets = _tool_target_ids(observation)
    evidence: list[dict[str, Any]] = []
    for tool in tools:
        for target in targets:
            if len(evidence) >= budget:
                break
            view.begin_trial()
            select = view.step(PrimitiveAction(kind=ActionKind.SELECT, target_id=tool))
            interaction = view.step(
                PrimitiveAction(kind=ActionKind.INTERACT, target_id=target)
            )
            evidence.append(
                {
                    "tool_id": tool,
                    "target_id": target,
                    "selection_compliant": select.intervention_compliance,
                    "interaction_compliant": interaction.intervention_compliance,
                    "success": interaction.event.success,
                    "duration": interaction.event.duration,
                }
            )
        if len(evidence) >= budget:
            break
    return {
        "seed": seed,
        "observation": _public_context(observation),
        "tools": tools,
        "targets": targets,
        "evidence": evidence,
    }


def _prediction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_id": {"type": "string"},
                        "target_id": {"type": "string"},
                        "success": {"type": "boolean"},
                    },
                    "required": ["tool_id", "target_id", "success"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["predictions", "summary"],
        "additionalProperties": False,
    }


_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "principles": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["principles"],
    "additionalProperties": False,
}


def _score(seed: int, raw: dict[str, Any]) -> dict[str, Any]:
    batch = PredictionBatch.model_validate(raw)
    bundle = WorldFactory.create(seed, change_episode=None)
    valid_pairs = {
        (tool, target)
        for tool in bundle.evaluator.tools
        for target in bundle.evaluator.blocks
    }
    predictions: dict[tuple[str, str], bool] = {}
    invalid = 0
    for item in batch.predictions:
        pair = (item.tool_id, item.target_id)
        if pair not in valid_pairs or pair in predictions:
            invalid += 1
            continue
        predictions[pair] = item.success
    correct = sum(
        predictions.get(pair) is bundle.evaluator.tool_outcome(*pair, 0)
        for pair in valid_pairs
    )
    return {
        "accuracy": correct / len(valid_pairs),
        "coverage": len(predictions) / len(valid_pairs),
        "invalid_or_duplicate_predictions": invalid,
        "summary": batch.summary,
    }


def run_external_llm_baselines(
    *,
    config: ExternalLLMConfig,
    output: Path,
    confirm_api_cost: bool,
    overwrite: bool = False,
    api_key: str | None = None,
    client: _Client | None = None,
) -> dict[str, Any]:
    """Run context-only and Voyager-style baselines behind explicit cost gates."""

    config.validate()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not confirm_api_cost:
        raise ExternalLLMAuthorizationError(
            "API cost is not authorized; pass --confirm-api-cost explicitly"
        )
    if not key:
        raise ExternalLLMAuthorizationError("OPENAI_API_KEY is not configured")
    report_path = output / "external-llm-baselines.json"
    ledger_path = output / "api-usage.jsonl"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"external LLM report already exists: {report_path}")
    if ledger_path.exists() and not overwrite:
        raise FileExistsError(f"external LLM usage ledger already exists: {ledger_path}")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI extra is missing; run: uv sync --extra openai"
            ) from exc
        client = OpenAI(api_key=key)

    output.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")
    budget = _BudgetedResponses(client, config, ledger_path)
    context_results: list[dict[str, Any]] = []
    voyager_results: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    prediction_instructions = (
        "Infer the latent tool-capability relation from opaque public evidence. "
        "Return exactly one prediction for every listed tool-target pair. Never "
        "invent identifiers and do not use names as semantic clues."
    )
    for offset in range(config.seeds):
        source_seed = config.first_seed + offset * 2
        target_seed = source_seed + 1
        source = _collect_public_evidence(source_seed, config.evidence_budget)
        target = _collect_public_evidence(target_seed, config.evidence_budget)
        context_raw = budget.call(
            purpose=f"context_only:{target_seed}",
            instructions=prediction_instructions,
            payload={"current_world": target},
            schema_name="tool_predictions",
            schema=_prediction_schema(),
        )
        context_results.append({"seed": target_seed, **_score(target_seed, context_raw)})

        memory_raw = budget.call(
            purpose=f"voyager_memory:{source_seed}",
            instructions=(
                "Compress the opaque intervention history into reusable structural "
                "principles. Do not preserve or interpret object names."
            ),
            payload={"source_world": source},
            schema_name="skill_memory",
            schema=_MEMORY_SCHEMA,
        )
        memory = SkillMemory.model_validate(memory_raw)
        memories.append({"source_seed": source_seed, **memory.model_dump(mode="json")})
        voyager_raw = budget.call(
            purpose=f"voyager_transfer:{source_seed}->{target_seed}",
            instructions=prediction_instructions,
            payload={
                "reusable_skill_memory": memory.model_dump(mode="json"),
                "current_world": target,
            },
            schema_name="tool_predictions",
            schema=_prediction_schema(),
        )
        voyager_results.append({"seed": target_seed, **_score(target_seed, voyager_raw)})

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "mean_accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
            "mean_coverage": sum(float(row["coverage"]) for row in rows) / len(rows),
            "samples": rows,
        }

    payload = {
        "suite": "external-llm-baselines",
        "configuration": {
            **asdict(config),
            "api_key_present": True,
            "api_key_persisted": False,
            "store": False,
        },
        "context_only_llm": aggregate(context_results),
        "voyager_style_llm": aggregate(voyager_results),
        "voyager_memories": memories,
        "api_usage": {
            "calls": budget.calls,
            "total_tokens": budget.total_tokens,
            "records": budget.records,
        },
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
