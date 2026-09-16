from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_scientist.minecraft.benchmark as benchmark_module
from minecraft_scientist.protocols import EvaluationReport


def test_live_benchmark_composes_all_research_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structured = EvaluationReport(
        seed=4,
        hidden_intervention_accuracy=1.0,
        experiment_count=31,
        experiment_efficiency=3 / 31,
        change_detection_delay=1,
        revision_experiments=8,
        description_length=1200,
        symbol_count=3,
        symbol_reuse=7,
        task_success_rate=1.0,
        mechanism_accuracy={"tool_capability": 1.0},
        theory_versions={"tool-capability": 2},
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_structured(**kwargs: object) -> SimpleNamespace:
        calls.append(("structured", kwargs))
        return SimpleNamespace(report=structured)

    def fake_fall(**kwargs: object) -> dict[str, object]:
        calls.append(("fall", kwargs))
        return {
            "standard_law_recovery": 1.0,
            "standard_experiments": 3,
            "description_length": 140,
            "world_transfer": {"law_recovery": 1.0},
            "change_detection_delay": 2,
            "revision_experiments": 4,
            "all_passed": True,
        }

    monkeypatch.setattr(benchmark_module, "run_live_discovery", fake_structured)
    monkeypatch.setattr(benchmark_module, "run_live_fall_discovery", fake_fall)
    output = tmp_path / "live"

    payload = benchmark_module.run_live_benchmark(
        server_directory=tmp_path / "server",
        output=output,
        seed=4,
        planner="learned",
        planner_checkpoint=tmp_path / "policy.pt",
    )

    assert [name for name, _ in calls] == ["structured", "fall"]
    assert all(kwargs["planner"] == "learned" for _, kwargs in calls)
    assert all(
        kwargs["planner_checkpoint"] == tmp_path / "policy.pt"
        for _, kwargs in calls
    )
    assert payload["planner"] == "learned"
    assert payload["planner_checkpoint"] == str(tmp_path / "policy.pt")
    assert payload["all_passed"] is True
    assert payload["metrics"]["law_recovery"] == 1.0
    assert payload["metrics"]["description_length"] == 1340
    assert payload["metrics"]["world_transfer"] == 1.0
    assert payload["metrics"]["experiment_count"] == 38
    persisted = json.loads((output / "live-benchmark.json").read_text())
    assert persisted == payload

    with pytest.raises(FileExistsError):
        benchmark_module.run_live_benchmark(
            server_directory=tmp_path / "server",
            output=output,
            seed=4,
        )
