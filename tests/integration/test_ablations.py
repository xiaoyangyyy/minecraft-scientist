import json
import pytest
from pathlib import Path

from minecraft_scientist.evaluation.ablations import run_ablation_suite


def test_strict_ablation_suite_is_paired_and_measures_each_component(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    checkpoint = Path("checkpoints/four-form-experiment-policy.pt")
    payload = run_ablation_suite(
        seed=0,
        seeds=2,
        intervention_budget=80,
        planner_checkpoint=checkpoint,
        output=tmp_path / "ablations",
        overwrite=False,
    )

    assert payload["seeds"] == [0, 1]
    assert payload["design"]["same_seed"]
    assert payload["design"]["same_intervention_budget"]
    assert payload["integrity"]["all_conditions_completed"]
    assert payload["integrity"]["all_full_runs_succeeded"]
    assert payload["integrity"]["all_learned_runs_succeeded"]
    assert payload["integrity"]["all_rule_shifts_detected_without_repair"]
    effects = payload["effects"]
    assert effects["active_information_gain"]["experiment_savings"]["mean"] > 0
    assert effects["theory_crisis"]["post_change_accuracy_recovery"]["mean"] > 0
    assert effects["theory_crisis"]["completed_crises"] == 2
    assert effects["theory_crisis"]["ablation_completed_crises"] == 0
    assert effects["library_learning"]["description_length_reduction"]["mean"] > 0
    assert effects["library_learning"]["symbol_gain"]["mean"] > 0
    assert effects["cross_world_memory"]["transfer_experiment_savings"]["mean"] > 0
    assert effects["learned_planner"]["accuracy_gain"]["mean"] == 0

    persisted = json.loads(
        (tmp_path / "ablations" / "ablation-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == payload
