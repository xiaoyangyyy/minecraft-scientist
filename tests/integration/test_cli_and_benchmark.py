import json
from pathlib import Path

from typer.testing import CliRunner

from minecraft_scientist.cli import app
from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.evaluation.benchmark import run_mvp_benchmark


def test_benchmark_reports_baseline_randomization_and_library_transfer(
    tmp_path: Path,
) -> None:
    payload = run_mvp_benchmark(
        seed=0,
        intervention_budget=80,
        output=tmp_path / "benchmark",
        overwrite=False,
    )
    assert payload["randomized_id_suite"]["all_passed"]
    assert payload["ablation"]["description_length_delta"] < 0
    assert payload["ablation"]["downstream_experiment_savings"] > 0
    assert payload["baselines"]["active_tool_discovery_experiments"] <= 9
    assert (tmp_path / "benchmark" / "benchmark.json").exists()


def test_cli_loads_world_config_and_writes_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 3,
                "intervention_budget": 80,
                "change_episode": 1,
                "library_learning": True,
                "hypothesis_generator": "heuristic",
                "world_name": "mvp",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        ["run", "--world-config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["seed"] == 3
    assert report["hidden_intervention_accuracy"] == 1.0

    inspected = CliRunner().invoke(
        app,
        ["inspect-theory", str(output), "tool-capability"],
    )
    assert inspected.exit_code == 0, inspected.output
    assert "experiment-" in inspected.output
    assert "superseded" in inspected.output
    assert "accepted" in inspected.output


def test_cli_inspects_visual_route_lifecycle(tmp_path: Path) -> None:
    topology = VisualTopologicalMap()
    old = topology.record_route((1,), (2,), ("direct",), success=True)
    replacement = topology.record_route(
        (1,), (2,), ("detour", "advance"), success=True
    )
    topology.challenge_route(old, reason="blocked after layout mutation")
    topology.supersede_route(
        old,
        replacement,
        reason="shadow validation passed",
    )
    path = tmp_path / "visual-map.json"
    topology.save(path)

    result = CliRunner().invoke(app, ["inspect-visual-map", str(path)])
    machine = CliRunner().invoke(
        app, ["inspect-visual-map", str(path), "--json"]
    )

    assert result.exit_code == 0, result.output
    assert "superseded" in result.output
    assert "shadow validation passed" in result.output
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["format_version"] == VisualTopologicalMap.FORMAT_VERSION
    assert payload["route_count"] == 2
