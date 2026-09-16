from __future__ import annotations

from pathlib import Path

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
