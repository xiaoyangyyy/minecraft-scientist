from __future__ import annotations

from pathlib import Path

from minecraft_scientist.evaluation.baselines import run_baseline_matrix


def test_baseline_matrix_is_deterministic_and_keeps_external_runs_explicit(
    tmp_path: Path,
) -> None:
    first = run_baseline_matrix(
        seed=3,
        seeds=3,
        intervention_budget=80,
        output=tmp_path / "first",
        overwrite=False,
    )
    second = run_baseline_matrix(
        seed=3,
        seeds=3,
        intervention_budget=80,
        output=tmp_path / "second",
        overwrite=False,
    )

    assert first == second
    assert first["all_offline_baselines_passed"] is True
    assert (
        first["active_vs_random"]["active_mean_experiments"]
        < first["active_vs_random"]["random_mean_experiments"]
    )
    assert first["external_model_baselines"]["context_only_llm"]["status"] == "not_run"
