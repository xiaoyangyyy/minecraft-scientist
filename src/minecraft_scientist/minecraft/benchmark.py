"""One-command composition of every live Minecraft acceptance curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fall_live import run_live_fall_discovery
from .live import run_live_discovery


def run_live_benchmark(
    *,
    server_directory: Path,
    output: Path,
    seed: int,
    intervention_budget: int = 80,
    fall_intervention_budget: int = 40,
    username: str = "ScientistBot",
    overwrite: bool = False,
    server_mode: str = "fabric",
    planner: str = "information_gain",
    planner_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run live rule/state/program and equation suites, then combine metrics."""

    report_path = output / "live-benchmark.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"live benchmark already exists: {report_path}")
    structured_result = run_live_discovery(
        server_directory=server_directory,
        output=output / "rule-state-program",
        seed=seed,
        intervention_budget=intervention_budget,
        change_episode=1,
        username=username,
        overwrite=overwrite,
        server_mode=server_mode,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    fall = run_live_fall_discovery(
        server_directory=server_directory,
        output=output / "equation",
        seed=seed,
        intervention_budget=fall_intervention_budget,
        username=username,
        overwrite=overwrite,
        server_mode=server_mode,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    structured = structured_result.report.model_dump(mode="json")
    law_recovery = (
        3.0 * float(structured["hidden_intervention_accuracy"])
        + float(fall["standard_law_recovery"])
    ) / 4.0
    experiment_count = (
        int(structured["experiment_count"])
        + int(fall["standard_experiments"])
        + int(fall["revision_experiments"])
    )
    payload: dict[str, Any] = {
        "suite": "minecraft-live",
        "seed": seed,
        "planner": planner,
        "planner_checkpoint": (
            str(planner_checkpoint) if planner_checkpoint is not None else None
        ),
        "rule_state_program": structured,
        "equation": fall,
        "metrics": {
            "law_recovery": law_recovery,
            "causal_intervention_accuracy": law_recovery,
            "description_length": int(structured["description_length"])
            + int(fall["description_length"]),
            "world_transfer": float(fall["world_transfer"]["law_recovery"]),
            "theory_revision_detection_experiments": max(
                int(structured["change_detection_delay"] or 0),
                int(fall["change_detection_delay"]),
            ),
            "experiment_count": experiment_count,
            "experiment_efficiency": law_recovery / max(experiment_count, 1),
            "task_success": (
                float(structured["task_success_rate"])
                + float(fall["standard_law_recovery"])
            )
            / 2.0,
        },
        "all_passed": (
            float(structured["hidden_intervention_accuracy"]) == 1.0
            and float(structured["task_success_rate"]) == 1.0
            and bool(fall["all_passed"])
        ),
        "notes": ["world transfer is measured on a second physical equation arena"],
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
