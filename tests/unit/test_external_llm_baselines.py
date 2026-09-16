from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_scientist.evaluation.external_llm import (
    ExternalLLMAuthorizationError,
    ExternalLLMConfig,
    _decode_structured_object,
    run_external_llm_baselines,
)


class FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        payload = json.loads(kwargs["input"])
        if kwargs["text"]["format"]["name"] == "skill_memory":
            output = {"principles": ["infer an ordered latent capability relation"]}
        else:
            world = payload["current_world"]
            output = {
                "predictions": [
                    {"tool_id": tool, "target_id": target, "success": True}
                    for tool in world["tools"]
                    for target in world["targets"]
                ],
                "summary": "fake deterministic prediction",
            }
        return SimpleNamespace(
            id=f"response-{len(self.requests)}",
            model=kwargs["model"],
            status="completed",
            output_text=json.dumps(output),
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_external_baseline_fails_closed_before_network(tmp_path) -> None:
    client = FakeClient()
    config = ExternalLLMConfig(seeds=1, max_api_calls=3)
    with pytest.raises(ExternalLLMAuthorizationError, match="not authorized"):
        run_external_llm_baselines(
            config=config,
            output=tmp_path,
            confirm_api_cost=False,
            api_key="test-key",
            client=client,
        )
    assert client.responses.requests == []

    with pytest.raises(ExternalLLMAuthorizationError, match="OPENAI_API_KEY"):
        run_external_llm_baselines(
            config=config,
            output=tmp_path,
            confirm_api_cost=True,
            api_key=None,
            client=client,
        )
    assert client.responses.requests == []


def test_external_baseline_uses_structured_nonstored_responses(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeClient()
    payload = run_external_llm_baselines(
        config=ExternalLLMConfig(
            seeds=1,
            max_api_calls=3,
            max_total_tokens=20_000,
        ),
        output=tmp_path,
        confirm_api_cost=True,
        api_key="test-key",
        client=client,
    )

    assert payload["api_usage"]["calls"] == 3
    assert payload["api_usage"]["total_tokens"] == 360
    assert payload["context_only_llm"]["mean_coverage"] == 1.0
    assert payload["voyager_style_llm"]["mean_coverage"] == 1.0
    assert all(request["store"] is False for request in client.responses.requests)
    assert all(
        request["text"]["format"]["strict"] is True
        for request in client.responses.requests
    )
    report = json.loads((tmp_path / "external-llm-baselines.json").read_text())
    ledger = (tmp_path / "api-usage.jsonl").read_text().splitlines()
    assert len(ledger) == 3
    assert report["configuration"]["api_key_persisted"] is False
    assert "test-key" not in json.dumps(report)


def test_external_baseline_requires_capacity_for_three_calls_per_seed() -> None:
    with pytest.raises(ValueError, match="required 6"):
        ExternalLLMConfig(seeds=2, max_api_calls=5).validate()


def test_deepseek_baseline_uses_provider_stateless_responses(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = FakeClient()
    payload = run_external_llm_baselines(
        config=ExternalLLMConfig(
            provider="deepseek",
            model="deepseek-flash",
            seeds=1,
            max_api_calls=3,
            max_total_tokens=20_000,
            reasoning_effort="none",
        ),
        output=tmp_path,
        confirm_api_cost=True,
        api_key="test-deepseek-key",
        client=client,
    )

    assert payload["configuration"]["provider"] == "deepseek"
    assert payload["configuration"]["storage_semantics"] == "provider-stateless"
    assert all("store" not in request for request in client.responses.requests)
    assert all(
        "strict" not in request["text"]["format"]
        for request in client.responses.requests
    )
    assert all(
        request["reasoning"] == {"effort": "none"}
        for request in client.responses.requests
    )
    assert all(
        record["provider"] == "deepseek"
        for record in payload["api_usage"]["records"]
    )


def test_external_baseline_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        ExternalLLMConfig(reasoning_effort="ultra").validate()


def test_structured_decoder_audits_trailing_provider_output() -> None:
    value, normalization = _decode_structured_object(
        '{"predictions": [], "summary": "ok"}\nprovider footer'
    )

    assert value["summary"] == "ok"
    assert normalization["mode"] == "first-json-object"
    assert normalization["trailing_bytes"] == len("provider footer")
    assert len(normalization["trailing_sha256"]) == 64
