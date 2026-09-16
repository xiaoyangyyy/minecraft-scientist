from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from minecraft_scientist.evaluation.expanded_external_llm import (
    MECHANISMS,
    _collect_world,
    run_expanded_external_llm_baselines,
)
from minecraft_scientist.evaluation.external_llm import ExternalLLMConfig


class _FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        request = json.loads(kwargs["input"])
        if kwargs["text"]["format"]["name"] == "skill_memory":
            output = {"principles": ["reuse opaque structural relations"]}
        else:
            world = request["current_world"]
            output = {
                "predictions": [
                    {
                        "intervention_id": candidate["intervention_id"],
                        "success": False,
                        "output_id": None,
                    }
                    for candidate in world["candidate_interventions"]
                ],
                "summary": "deterministic fake predictions",
            }
        return SimpleNamespace(
            id=f"expanded-response-{len(self.requests)}",
            model=kwargs["model"],
            status="completed",
            output_text=json.dumps(output),
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _TransientResponses(_FakeResponses):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def create(self, **kwargs: Any) -> Any:
        if not self.failed_once:
            self.failed_once = True
            self.requests.append(kwargs)
            error_type = type("APIConnectionError", (RuntimeError,), {})
            raise error_type("temporary connection failure")
        return super().create(**kwargs)


class _TransientClient:
    def __init__(self) -> None:
        self.responses = _TransientResponses()


def test_collect_world_keeps_truth_out_of_public_payload() -> None:
    world = _collect_world(seed=7, evidence_budget=2)

    assert len(world.public["candidate_interventions"]) == 41
    assert set(world.evidence_counts) == set(MECHANISMS)
    assert all(value <= 2 for value in world.evidence_counts.values())
    assert len(world.truth) == 41
    assert "truth" not in json.dumps(world.public).lower()


def test_expanded_baseline_compares_three_methods_on_shared_evidence(tmp_path) -> None:
    client = _FakeClient()
    payload = run_expanded_external_llm_baselines(
        config=ExternalLLMConfig(
            provider="deepseek",
            model="deepseek-flash",
            seeds=1,
            evidence_budget=2,
            max_api_calls=3,
            max_output_tokens=4_000,
            max_total_tokens=100_000,
            reasoning_effort="none",
        ),
        output=tmp_path,
        confirm_api_cost=True,
        api_key="fake-key",
        client=client,
    )

    assert payload["configuration"]["shared_evidence"] is True
    assert payload["configuration"]["mechanisms"] == list(MECHANISMS)
    assert payload["context_only_llm"]["accuracy"]["n"] == 1
    assert payload["voyager_style_llm"]["accuracy"]["n"] == 1
    assert payload["symbolic_shared_evidence"]["accuracy"]["n"] == 1
    assert set(payload["context_only_llm"]["mechanism_accuracy"]) == set(MECHANISMS)
    assert payload["api_usage"]["all"]["calls"] == 3
    assert payload["api_usage"]["context_only"]["calls"] == 1
    assert payload["api_usage"]["voyager_style"]["calls"] == 2
    report = (tmp_path / "expanded-external-llm-baselines.json").read_text()
    assert "fake-key" not in report
    assert len((tmp_path / "expanded-progress.jsonl").read_text().splitlines()) == 1


def test_expanded_baseline_resumes_completed_pairs_without_rebilling(tmp_path) -> None:
    client = _FakeClient()
    first_config = ExternalLLMConfig(
        provider="deepseek",
        model="deepseek-flash",
        seeds=1,
        evidence_budget=1,
        max_api_calls=3,
        max_output_tokens=4_000,
        max_total_tokens=100_000,
        reasoning_effort="none",
    )
    run_expanded_external_llm_baselines(
        config=first_config,
        output=tmp_path,
        confirm_api_cost=True,
        api_key="fake-key",
        client=client,
    )
    resumed = run_expanded_external_llm_baselines(
        config=ExternalLLMConfig(
            provider="deepseek",
            model="deepseek-flash",
            seeds=2,
            evidence_budget=1,
            max_api_calls=7,
            max_output_tokens=4_000,
            max_total_tokens=200_000,
            reasoning_effort="none",
        ),
        output=tmp_path,
        confirm_api_cost=True,
        resume=True,
        max_transient_retries=1,
        api_key="fake-key",
        client=client,
    )

    assert len(client.responses.requests) == 6
    assert resumed["context_only_llm"]["accuracy"]["n"] == 2
    assert resumed["api_usage"]["all"]["calls"] == 6
    assert len((tmp_path / "expanded-progress.jsonl").read_text().splitlines()) == 2


def test_expanded_baseline_retries_transient_connection_failure(tmp_path) -> None:
    client = _TransientClient()
    payload = run_expanded_external_llm_baselines(
        config=ExternalLLMConfig(
            provider="deepseek",
            model="deepseek-flash",
            seeds=1,
            evidence_budget=1,
            max_api_calls=4,
            max_output_tokens=4_000,
            max_total_tokens=100_000,
            reasoning_effort="none",
        ),
        output=tmp_path,
        confirm_api_cost=True,
        max_transient_retries=1,
        api_key="fake-key",
        client=client,
    )

    assert len(client.responses.requests) == 4
    assert payload["configuration"]["transient_retries_used"] == 1
    assert payload["api_usage"]["all"]["calls"] == 3
