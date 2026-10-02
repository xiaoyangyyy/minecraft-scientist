from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_scientist.evaluation.full import run_full_benchmark


def test_full_benchmark_recovers_multiform_laws_transfer_and_revision(
    tmp_path: Path,
) -> None:
    payload = run_full_benchmark(
        seed=0,
        intervention_budget=80,
        output=tmp_path / "full",
        overwrite=False,
    )
    assert payload["all_passed"] is True
    assert payload["continuous_law"]["representation"] == "equation"
    assert payload["continuous_law"]["law_recovery"] == 1.0
    assert payload["world_transfer"]["experiment_savings"] > 0
    assert payload["world_transfer"]["law_recovery"] == 1.0
    assert payload["theory_revision"]["shadow_validation_passed"] is True
    history = payload["theory_revision"]["history"]
    assert [item["status"] for item in history] == ["superseded", "accepted"]
    assert history[0]["counterexamples"]
    assert history[1]["parent_version"] == 1
    assert len(payload["causal_discovery"]["graph"]["edges"]) == 2
    assert payload["metrics"]["law_recovery"] == 1.0
    assert payload["metrics"]["causal_intervention_accuracy"] == 1.0
    assert payload["metrics"]["world_transfer"] == 1.0
    assert payload["metrics"]["task_success"] == 1.0
    assert (tmp_path / "full" / "experience.sqlite3").exists()


def test_fall_experiment_design_never_reads_privileged_evaluator(tmp_path, monkeypatch):
    from minecraft_scientist.evaluation import full

    original_factory = full.FallWorldFactory.create

    class ForbiddenEvaluator:
        def __getattribute__(self, name):
            pytest.fail(f"experiment design accessed evaluator field {name}")

    def public_bundle(*args, **kwargs):
        bundle = original_factory(*args, **kwargs)
        return SimpleNamespace(
            agent_view=bundle.agent_view, evaluator=ForbiddenEvaluator()
        )

    monkeypatch.setattr(full.FallWorldFactory, "create", public_bundle)
    # This test isolates experiment design from intentionally privileged scoring;
    # the preceding real integration test still checks true behavioral accuracy.
    monkeypatch.setattr(full, "_behavioral_accuracy", lambda *args: 1.0)
    result = full.run_full_benchmark(
        seed=7, intervention_budget=80, output=tmp_path / "public-only", overwrite=False
    )
    assert (
        result["all_passed"] and result["theory_revision"]["shadow_validation_passed"]
    )
