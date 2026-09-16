"""Shared-evidence, multi-mechanism external LLM benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from minecraft_scientist.discovery import (
    HeuristicHypothesisGenerator,
    MachineVersionSpace,
    ToolVersionSpace,
    TransformationLearner,
)
from minecraft_scientist.discovery.hypotheses import PROTOCOL_CANDIDATES
from minecraft_scientist.protocols import ActionKind, PrimitiveAction
from minecraft_scientist.world import WorldFactory

from .external_llm import (
    ExternalLLMAuthorizationError,
    ExternalLLMConfig,
    SkillMemory,
    _BudgetedResponses,
    _Client,
    _MEMORY_SCHEMA,
    _public_context,
    create_responses_client,
    resolve_external_api_key,
)


MECHANISMS = (
    "tool_capability",
    "machine_protocol",
    "transformation_chain",
)
TRANSFORM_PROTOCOL = ("place", "activate", "wait:1", "interact")


class InterventionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intervention_id: str
    success: bool
    output_id: str | None


class InterventionPredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictions: tuple[InterventionPrediction, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class _EvidenceWorld:
    public: dict[str, Any]
    truth: dict[str, dict[str, Any]]
    symbolic: dict[str, Any]
    evidence_counts: dict[str, int]


def _intervention_id(mechanism: str, action: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"mechanism": mechanism, "action": action},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"iv_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _candidate(mechanism: str, action: dict[str, Any]) -> dict[str, Any]:
    return {
        "intervention_id": _intervention_id(mechanism, action),
        "mechanism": mechanism,
        "action": action,
    }


def _run_tool(view: Any, tool_id: str, target_id: str) -> tuple[bool, str | None]:
    view.begin_trial()
    selected = view.step(PrimitiveAction(kind=ActionKind.SELECT, target_id=tool_id))
    if not selected.intervention_compliance:
        raise RuntimeError("tool selection was not compliant")
    result = view.step(PrimitiveAction(kind=ActionKind.INTERACT, target_id=target_id))
    if not result.intervention_compliance:
        raise RuntimeError("tool interaction was not compliant")
    return bool(result.event.success), result.event.output_id


def _run_station(
    view: Any,
    station_id: str,
    input_id: str,
    protocol: tuple[str, ...],
) -> tuple[bool, str | None]:
    view.begin_trial()
    result = None
    for token in protocol:
        if token == "place":
            action = PrimitiveAction(
                kind=ActionKind.PLACE_INPUT,
                target_id=station_id,
                item_id=input_id,
            )
        elif token == "activate":
            action = PrimitiveAction(kind=ActionKind.ACTIVATE, target_id=station_id)
        elif token == "interact":
            action = PrimitiveAction(kind=ActionKind.INTERACT, target_id=station_id)
        elif token.startswith("wait:"):
            action = PrimitiveAction(
                kind=ActionKind.WAIT,
                ticks=int(token.split(":", 1)[1]),
            )
        else:
            raise ValueError(f"unknown protocol token: {token}")
        result = view.step(action)
        if not result.intervention_compliance:
            raise RuntimeError("station intervention was not compliant")
    if result is None or result.event.event_type != "station_result":
        raise RuntimeError("station protocol did not terminate with an interaction")
    return bool(result.event.success), result.event.output_id


def _prediction_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "intervention_id": {"type": "string"},
                        "success": {"type": "boolean"},
                        "output_id": {"type": ["string", "null"]},
                    },
                    "required": ["intervention_id", "success", "output_id"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["predictions", "summary"],
        "additionalProperties": False,
    }


def _collect_world(seed: int, evidence_budget: int) -> _EvidenceWorld:
    bundle = WorldFactory.create(seed, change_episode=None)
    observation = bundle.controller.begin_episode(0)
    groups = HeuristicHypothesisGenerator().classify(observation)
    tools = tuple(groups["tools"])
    targets = tuple(groups["targets"])
    machine_id = str(groups["machine"])
    processors = tuple(groups["processors"])
    resources = tuple(groups["resources"])
    view = bundle.agent_view

    candidates: list[dict[str, Any]] = []
    for tool_id in tools:
        for target_id in targets:
            candidates.append(
                _candidate(
                    "tool_capability",
                    {"tool_id": tool_id, "target_id": target_id},
                )
            )
    for input_id in resources:
        for protocol in PROTOCOL_CANDIDATES:
            candidates.append(
                _candidate(
                    "machine_protocol",
                    {
                        "machine_id": machine_id,
                        "input_id": input_id,
                        "protocol": list(protocol),
                    },
                )
            )
    for processor_id in processors:
        for input_id in resources:
            candidates.append(
                _candidate(
                    "transformation_chain",
                    {
                        "processor_id": processor_id,
                        "input_id": input_id,
                        "protocol": list(TRANSFORM_PROTOCOL),
                    },
                )
            )
    candidate_by_key = {
        (
            candidate["mechanism"],
            json.dumps(candidate["action"], sort_keys=True),
        ): candidate
        for candidate in candidates
    }

    evidence: list[dict[str, Any]] = []
    tool_space = ToolVersionSpace(tools, targets)
    for index in range(evidence_budget):
        pair = tool_space.next_experiment()
        if pair is None:
            break
        success, output_id = _run_tool(view, *pair)
        action = {"tool_id": pair[0], "target_id": pair[1]}
        candidate = candidate_by_key[
            ("tool_capability", json.dumps(action, sort_keys=True))
        ]
        evidence_id = f"tool-evidence-{index:03d}"
        tool_space.observe(*pair, success, evidence_id)
        evidence.append(
            {
                **candidate,
                "outcome": {"success": success, "output_id": output_id},
            }
        )

    machine_space = MachineVersionSpace(resources)
    for index in range(evidence_budget):
        pair = machine_space.next_experiment()
        if pair is None:
            break
        input_id, protocol = pair
        success, output_id = _run_station(view, machine_id, input_id, protocol)
        action = {
            "machine_id": machine_id,
            "input_id": input_id,
            "protocol": list(protocol),
        }
        candidate = candidate_by_key[
            ("machine_protocol", json.dumps(action, sort_keys=True))
        ]
        evidence_id = f"machine-evidence-{index:03d}"
        machine_space.observe(input_id, protocol, success, evidence_id, output_id)
        evidence.append(
            {
                **candidate,
                "outcome": {"success": success, "output_id": output_id},
            }
        )

    transform_space = TransformationLearner(processors, resources)
    for index in range(evidence_budget):
        pair = transform_space.next_experiment()
        if pair is None:
            break
        processor_id, input_id = pair
        success, output_id = _run_station(
            view,
            processor_id,
            input_id,
            TRANSFORM_PROTOCOL,
        )
        action = {
            "processor_id": processor_id,
            "input_id": input_id,
            "protocol": list(TRANSFORM_PROTOCOL),
        }
        candidate = candidate_by_key[
            ("transformation_chain", json.dumps(action, sort_keys=True))
        ]
        evidence_id = f"transform-evidence-{index:03d}"
        transform_space.observe(
            processor_id,
            input_id,
            success,
            output_id,
            evidence_id,
        )
        evidence.append(
            {
                **candidate,
                "outcome": {"success": success, "output_id": output_id},
            }
        )

    true_input, true_protocol, true_machine_output = bundle.evaluator.machine_rule()
    true_transforms = bundle.evaluator.transformations()
    truth: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        mechanism = candidate["mechanism"]
        action = candidate["action"]
        if mechanism == "tool_capability":
            success = bundle.evaluator.tool_outcome(
                action["tool_id"], action["target_id"], 0
            )
            output_id = None
        elif mechanism == "machine_protocol":
            success = action["input_id"] == true_input and _protocol_success(
                true_protocol,
                tuple(action["protocol"]),
            )
            output_id = true_machine_output if success else None
        else:
            source, target = true_transforms[action["processor_id"]]
            success = action["input_id"] == source
            output_id = target if success else None
        truth[candidate["intervention_id"]] = {
            "mechanism": mechanism,
            "success": success,
            "output_id": output_id,
        }

    observed_machine_output = next(
        (row[4] for row in machine_space.observations if row[2] and row[4]),
        None,
    )
    symbolic_predictions: list[dict[str, Any]] = []
    for candidate in candidates:
        mechanism = candidate["mechanism"]
        action = candidate["action"]
        if mechanism == "tool_capability":
            probability = tool_space.predictive_outcomes(
                action["tool_id"], action["target_id"]
            )["success"]
            success = probability >= 0.5
            output_id = None
        elif mechanism == "machine_protocol":
            probability = machine_space.predictive_outcomes(
                action["input_id"], tuple(action["protocol"])
            )["success"]
            success = probability >= 0.5
            output_id = observed_machine_output if success else None
        else:
            outcomes = transform_space.predictive_outcomes(
                action["processor_id"], action["input_id"]
            )
            selected = min(outcomes, key=lambda key: (-outcomes[key], key))
            output_id = None if selected == "<no-output>" else selected
            success = output_id is not None
        symbolic_predictions.append(
            {
                "intervention_id": candidate["intervention_id"],
                "success": success,
                "output_id": output_id,
            }
        )

    counts = {
        mechanism: sum(row["mechanism"] == mechanism for row in evidence)
        for mechanism in MECHANISMS
    }
    public = {
        "seed_label": f"world-{seed}",
        "observation": _public_context(observation),
        "candidate_interventions": candidates,
        "evidence": evidence,
        "evidence_counts": counts,
    }
    return _EvidenceWorld(
        public=public,
        truth=truth,
        symbolic={
            "predictions": symbolic_predictions,
            "summary": "finite symbolic posterior over the shared public evidence",
        },
        evidence_counts=counts,
    )


def _protocol_success(
    expected: tuple[str, ...], observed: tuple[str, ...]
) -> bool:
    if expected == observed:
        return True
    expected_waits = [token for token in expected if token.startswith("wait:")]
    observed_waits = [token for token in observed if token.startswith("wait:")]
    if len(expected_waits) != 1 or len(observed_waits) != 1:
        return False
    expected_shape = tuple(
        "wait" if token.startswith("wait:") else token for token in expected
    )
    observed_shape = tuple(
        "wait" if token.startswith("wait:") else token for token in observed
    )
    return expected_shape == observed_shape and int(
        observed_waits[0].split(":", 1)[1]
    ) >= int(expected_waits[0].split(":", 1)[1])


def _score(world: _EvidenceWorld, raw: dict[str, Any]) -> dict[str, Any]:
    batch = InterventionPredictionBatch.model_validate(raw)
    predictions: dict[str, InterventionPrediction] = {}
    invalid = 0
    for item in batch.predictions:
        if item.intervention_id not in world.truth or item.intervention_id in predictions:
            invalid += 1
            continue
        predictions[item.intervention_id] = item

    mechanism_accuracy: dict[str, float] = {}
    mechanism_coverage: dict[str, float] = {}
    for mechanism in MECHANISMS:
        identifiers = [
            identifier
            for identifier, expected in world.truth.items()
            if expected["mechanism"] == mechanism
        ]
        correct = sum(
            identifier in predictions
            and predictions[identifier].success == world.truth[identifier]["success"]
            and predictions[identifier].output_id
            == world.truth[identifier]["output_id"]
            for identifier in identifiers
        )
        mechanism_accuracy[mechanism] = correct / len(identifiers)
        mechanism_coverage[mechanism] = sum(
            identifier in predictions for identifier in identifiers
        ) / len(identifiers)
    total = len(world.truth)
    overall_correct = sum(
        identifier in predictions
        and predictions[identifier].success == expected["success"]
        and predictions[identifier].output_id == expected["output_id"]
        for identifier, expected in world.truth.items()
    )
    return {
        "accuracy": overall_correct / total,
        "coverage": len(predictions) / total,
        "mechanism_accuracy": mechanism_accuracy,
        "mechanism_coverage": mechanism_coverage,
        "invalid_or_duplicate_predictions": invalid,
        "summary": batch.summary,
        "evidence_interventions": sum(world.evidence_counts.values()),
    }


def _summary(values: list[float], *, bounded: bool = False) -> dict[str, Any]:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * standard_deviation / math.sqrt(len(values))
    lower = mean - margin
    upper = mean + margin
    if bounded:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence_interval_95": [lower, upper],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accuracy": _summary([float(row["accuracy"]) for row in rows], bounded=True),
        "coverage": _summary([float(row["coverage"]) for row in rows], bounded=True),
        "mechanism_accuracy": {
            mechanism: _summary(
                [float(row["mechanism_accuracy"][mechanism]) for row in rows],
                bounded=True,
            )
            for mechanism in MECHANISMS
        },
        "experiment_efficiency": _summary(
            [
                float(row["accuracy"]) / max(int(row["evidence_interventions"]), 1)
                for row in rows
            ]
        ),
        "samples": rows,
    }


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(int(row["usage"]["input_tokens"]) for row in records)
    output_tokens = sum(int(row["usage"]["output_tokens"]) for row in records)
    return {
        "calls": len(records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_peak_cache_miss_cost_cny": (
            input_tokens * 2.0 + output_tokens * 8.0
        )
        / 1_000_000,
    }


def run_expanded_external_llm_baselines(
    *,
    config: ExternalLLMConfig,
    output: Path,
    confirm_api_cost: bool,
    overwrite: bool = False,
    resume: bool = False,
    max_transient_retries: int = 0,
    api_key: str | None = None,
    client: _Client | None = None,
) -> dict[str, Any]:
    """Compare LLM and symbolic induction on identical multi-mechanism evidence."""

    config.validate()
    if not confirm_api_cost:
        raise ExternalLLMAuthorizationError(
            "API cost is not authorized; pass --confirm-api-cost explicitly"
        )
    key, credential_source = resolve_external_api_key(config, api_key)
    key_name = (
        "DEEPSEEK_API_KEY" if config.provider == "deepseek" else "OPENAI_API_KEY"
    )
    if not key:
        raise ExternalLLMAuthorizationError(f"{key_name} is not configured")
    report_path = output / "expanded-external-llm-baselines.json"
    ledger_path = output / "expanded-api-usage.jsonl"
    progress_path = output / "expanded-progress.jsonl"
    if report_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"expanded LLM report already exists: {report_path}")
    if ledger_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"expanded LLM ledger already exists: {ledger_path}")
    if max_transient_retries < 0:
        raise ValueError("max_transient_retries cannot be negative")
    if client is None:
        client = create_responses_client(config, key)

    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        ledger_path.write_text("", encoding="utf-8")
        progress_path.write_text("", encoding="utf-8")
    elif not ledger_path.exists():
        ledger_path.write_text("", encoding="utf-8")
    if not progress_path.exists():
        progress_path.write_text("", encoding="utf-8")
    budget = _BudgetedResponses(client, config, ledger_path)
    context_rows: list[dict[str, Any]] = []
    voyager_rows: list[dict[str, Any]] = []
    symbolic_rows: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    if resume:
        existing_records = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        budget.records.extend(existing_records)
        budget.calls = len(existing_records)
        budget.total_tokens = sum(
            int(record["usage"]["total_tokens"]) for record in existing_records
        )
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            checkpoint = json.loads(line)
            context_rows.append(checkpoint["context"])
            voyager_rows.append(checkpoint["voyager"])
            symbolic_rows.append(checkpoint["symbolic"])
            memories.append(checkpoint["memory"])
    completed_target_seeds = {int(row["seed"]) for row in context_rows}
    retries_used = 0

    def call_with_retry(**kwargs: Any) -> dict[str, Any]:
        nonlocal retries_used
        while True:
            try:
                return budget.call(**kwargs)
            except Exception as exc:
                transient = isinstance(exc, json.JSONDecodeError) or type(exc).__name__ in {
                    "APIConnectionError",
                    "APITimeoutError",
                    "RateLimitError",
                    "InternalServerError",
                }
                if not transient:
                    raise
                if retries_used >= max_transient_retries:
                    raise
                retries_used += 1
                time.sleep(min(2 ** (retries_used - 1), 8))
    instructions = (
        "Infer three latent mechanisms from opaque public interventions: ordered "
        "tool capability, a hidden machine protocol, and a transformation chain. "
        "Return exactly one prediction for every candidate intervention_id. Set "
        "output_id to null on failure. Never infer meaning from identifier spelling."
    )
    for offset in range(config.seeds):
        source_seed = config.first_seed + offset * 2
        target_seed = source_seed + 1
        if target_seed in completed_target_seeds:
            continue
        source = _collect_world(source_seed, config.evidence_budget)
        target = _collect_world(target_seed, config.evidence_budget)
        context_raw = call_with_retry(
            purpose=f"expanded_context_only:{target_seed}",
            instructions=instructions,
            payload={"current_world": target.public},
            schema_name="multi_mechanism_predictions",
            schema=_prediction_schema(),
        )
        context_row = {"seed": target_seed, **_score(target, context_raw)}

        memory_raw = call_with_retry(
            purpose=f"expanded_voyager_memory:{source_seed}",
            instructions=(
                "Compress public opaque intervention evidence across tool, machine, "
                "and transformation mechanisms into reusable structural principles."
            ),
            payload={"source_world": source.public},
            schema_name="skill_memory",
            schema=_MEMORY_SCHEMA,
        )
        memory = SkillMemory.model_validate(memory_raw)
        memory_row = {"source_seed": source_seed, **memory.model_dump(mode="json")}
        voyager_raw = call_with_retry(
            purpose=f"expanded_voyager_transfer:{source_seed}->{target_seed}",
            instructions=instructions,
            payload={
                "reusable_skill_memory": memory.model_dump(mode="json"),
                "current_world": target.public,
            },
            schema_name="multi_mechanism_predictions",
            schema=_prediction_schema(),
        )
        voyager_row = {"seed": target_seed, **_score(target, voyager_raw)}
        symbolic_row = {"seed": target_seed, **_score(target, target.symbolic)}
        checkpoint = {
            "source_seed": source_seed,
            "target_seed": target_seed,
            "context": context_row,
            "voyager": voyager_row,
            "symbolic": symbolic_row,
            "memory": memory_row,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
        context_rows.append(context_row)
        voyager_rows.append(voyager_row)
        symbolic_rows.append(symbolic_row)
        memories.append(memory_row)

    context_records = [
        row for row in budget.records if row["purpose"].startswith("expanded_context")
    ]
    voyager_records = [
        row for row in budget.records if row["purpose"].startswith("expanded_voyager")
    ]
    payload = {
        "suite": "expanded-external-llm-baselines",
        "configuration": {
            **asdict(config),
            "mechanisms": list(MECHANISMS),
            "credential_source": credential_source,
            "api_key_persisted_in_artifacts": False,
            "shared_evidence": True,
            "pricing_snapshot": {
                "currency": "CNY",
                "input_cache_miss_per_million_peak": 2.0,
                "output_per_million_peak": 8.0,
                "assumption": "conservative peak rate, all input treated as cache miss",
            },
            "resume_enabled": resume,
            "max_transient_retries": max_transient_retries,
            "transient_retries_used": retries_used,
        },
        "context_only_llm": _aggregate(context_rows),
        "voyager_style_llm": _aggregate(voyager_rows),
        "symbolic_shared_evidence": _aggregate(symbolic_rows),
        "voyager_memories": memories,
        "api_usage": {
            "all": _usage_summary(budget.records),
            "context_only": _usage_summary(context_records),
            "voyager_style": _usage_summary(voyager_records),
            "records": budget.records,
        },
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
