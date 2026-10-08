"""Command-line interface for experiments, benchmarks, and theory inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.evaluation.benchmark import run_mvp_benchmark
from minecraft_scientist.evaluation.full import run_full_benchmark
from minecraft_scientist.protocols import RunConfig

app = typer.Typer(
    no_args_is_help=True, help="Active symbolic world-model discovery MVP"
)
console = Console()


def _load_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    if not path.exists():
        raise typer.BadParameter(f"world config does not exist: {path}")
    return RunConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _report_table(title: str, report: dict[str, object]) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in (
        "hidden_intervention_accuracy",
        "experiment_count",
        "change_detection_delay",
        "revision_experiments",
        "description_length",
        "symbol_count",
        "symbol_reuse",
        "task_success_rate",
    ):
        table.add_row(key, str(report[key]))
    return table


@app.command()
def run(
    seed: Annotated[
        int | None, typer.Option(help="Override deterministic seed")
    ] = None,
    intervention_budget: Annotated[
        int | None, typer.Option(min=1, help="Override maximum experiments")
    ] = None,
    change_episode: Annotated[
        int | None, typer.Option(help="Override hidden rule-change episode")
    ] = None,
    no_rule_change: Annotated[
        bool, typer.Option("--no-rule-change", help="Disable the hidden rule change")
    ] = False,
    world_config: Annotated[
        Path | None, typer.Option(help="JSON RunConfig file")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Run artifact directory")] = None,
    no_library_learning: Annotated[
        bool,
        typer.Option(
            "--no-library-learning", help="Disable abstraction learning for ablation"
        ),
    ] = False,
    no_theory_crisis: Annotated[
        bool,
        typer.Option(
            "--no-theory-crisis", help="Detect rule changes but disable theory repair"
        ),
    ] = False,
    hypothesis_generator: Annotated[
        str | None, typer.Option(help="Override with heuristic or openai")
    ] = None,
    planner: Annotated[
        str | None,
        typer.Option(help="Override with information_gain, learned, or random"),
    ] = None,
    planner_checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint required by the learned planner")
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite", help="Replace known artifact files in the output directory"
        ),
    ] = False,
) -> None:
    base = _load_config(world_config)
    updates: dict[str, object] = {}
    if seed is not None:
        updates["seed"] = seed
    if intervention_budget is not None:
        updates["intervention_budget"] = intervention_budget
    if change_episode is not None:
        updates["change_episode"] = change_episode
    if no_rule_change:
        updates["change_episode"] = None
    if no_library_learning:
        updates["library_learning"] = False
    if no_theory_crisis:
        updates["theory_crisis"] = False
    if hypothesis_generator is not None:
        updates["hypothesis_generator"] = hypothesis_generator
    if planner is not None:
        updates["planner"] = planner
    if planner_checkpoint is not None:
        updates["planner_checkpoint"] = str(planner_checkpoint)
    config = base.model_copy(update=updates)
    destination = output or Path("runs") / f"run-seed-{config.seed}"
    result = ScientistAgent(config).run()
    result.persist(destination, overwrite=overwrite)
    console.print(_report_table("Minecraft Scientist run", result.report.model_dump()))
    console.print(f"Artifacts: {destination.resolve()}")


@app.command()
def benchmark(
    suite: Annotated[str, typer.Option(help="Benchmark suite: mvp or full")] = "mvp",
    seed: Annotated[int | None, typer.Option(help="Override benchmark seed")] = None,
    intervention_budget: Annotated[
        int | None,
        typer.Option(min=1, help="Override maximum experiments per condition"),
    ] = None,
    world_config: Annotated[
        Path | None, typer.Option(help="JSON RunConfig file")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Benchmark artifact directory")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace known benchmark artifact files")
    ] = False,
    planner: Annotated[
        str, typer.Option(help="information_gain or learned")
    ] = "information_gain",
    planner_checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint required by the learned planner")
    ] = None,
) -> None:
    if suite not in {"mvp", "full"}:
        raise typer.BadParameter("suite must be mvp or full")
    base = _load_config(world_config)
    selected_seed = base.seed if seed is None else seed
    selected_budget = (
        base.intervention_budget if intervention_budget is None else intervention_budget
    )
    destination = output or Path("runs") / f"benchmark-{suite}-seed-{selected_seed}"
    runner = run_mvp_benchmark if suite == "mvp" else run_full_benchmark
    payload = runner(
        seed=selected_seed,
        intervention_budget=selected_budget,
        output=destination,
        overwrite=overwrite,
        planner=planner,
        planner_checkpoint=(
            str(planner_checkpoint) if planner_checkpoint is not None else None
        ),
    )
    report = payload["full"] if suite == "mvp" else payload["structured"]
    console.print(_report_table(f"{suite.upper()} benchmark", report))
    console.print(f"Artifacts: {destination.resolve()}")


@app.command("benchmark-baselines")
def benchmark_baselines(
    seed: Annotated[int, typer.Option(help="First deterministic baseline seed")] = 0,
    seeds: Annotated[
        int, typer.Option(min=1, help="Number of randomized-ID worlds")
    ] = 10,
    intervention_budget: Annotated[
        int, typer.Option(min=1, help="Budget for fixed-DSL and full agents")
    ] = 80,
    output: Annotated[Path, typer.Option(help="Baseline artifact directory")] = Path(
        "runs/baseline-matrix"
    ),
    overwrite: Annotated[
        bool, typer.Option(help="Replace the known baseline report")
    ] = False,
) -> None:
    from minecraft_scientist.evaluation.baselines import run_baseline_matrix

    payload = run_baseline_matrix(
        seed=seed,
        seeds=seeds,
        intervention_budget=intervention_budget,
        output=output,
        overwrite=overwrite,
    )
    console.print_json(json.dumps(payload, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-ablations")
def benchmark_ablations(
    seed: Annotated[int, typer.Option(help="First deterministic paired seed")] = 0,
    seeds: Annotated[
        int, typer.Option(min=1, help="Number of paired randomized-ID worlds")
    ] = 10,
    intervention_budget: Annotated[
        int, typer.Option(min=1, help="Same experiment budget for every condition")
    ] = 80,
    planner_checkpoint: Annotated[
        Path, typer.Option(help="Trained planner checkpoint used by the full pair")
    ] = Path("checkpoints/four-form-experiment-policy.pt"),
    output: Annotated[
        Path, typer.Option(help="Strict ablation artifact directory")
    ] = Path("runs/strict-ablations"),
    overwrite: Annotated[
        bool, typer.Option(help="Replace the known ablation report")
    ] = False,
) -> None:
    from minecraft_scientist.evaluation.ablations import run_ablation_suite

    payload = run_ablation_suite(
        seed=seed,
        seeds=seeds,
        intervention_budget=intervention_budget,
        planner_checkpoint=planner_checkpoint,
        output=output,
        overwrite=overwrite,
    )
    console.print_json(json.dumps(payload["effects"], sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-external-llm")
def benchmark_external_llm(
    provider: Annotated[str, typer.Option(help="openai or deepseek")] = "openai",
    model: Annotated[
        str, typer.Option(help="Responses API model identifier")
    ] = "gpt-5.6-luna",
    seed: Annotated[
        int, typer.Option(help="First deterministic source-world seed")
    ] = 0,
    seeds: Annotated[
        int, typer.Option(min=1, help="Number of source/target pairs")
    ] = 3,
    evidence_budget: Annotated[
        int, typer.Option(min=0, help="Public interventions supplied per world")
    ] = 6,
    max_api_calls: Annotated[
        int, typer.Option(min=1, help="Hard cap on paid API requests")
    ] = 9,
    max_output_tokens: Annotated[
        int, typer.Option(min=1, help="Per-request output-token cap")
    ] = 1_200,
    max_total_tokens: Annotated[
        int, typer.Option(min=1, help="Conservative total input/output-token cap")
    ] = 50_000,
    request_timeout_seconds: Annotated[
        float, typer.Option(min=1, help="Per-request network timeout")
    ] = 120.0,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(help="Optional Responses API reasoning effort: none/low/high/max"),
    ] = None,
    confirm_api_cost: Annotated[
        bool,
        typer.Option(
            "--confirm-api-cost",
            help="Explicitly authorize API usage within the supplied hard caps",
        ),
    ] = False,
    output: Annotated[
        Path, typer.Option(help="External baseline artifact directory")
    ] = Path("runs/external-llm-baselines"),
    overwrite: Annotated[
        bool, typer.Option(help="Replace the known external baseline report")
    ] = False,
) -> None:
    from minecraft_scientist.evaluation.external_llm import (
        ExternalLLMAuthorizationError,
        ExternalLLMConfig,
        run_external_llm_baselines,
    )

    try:
        payload = run_external_llm_baselines(
            config=ExternalLLMConfig(
                provider=provider,
                model=model,
                first_seed=seed,
                seeds=seeds,
                evidence_budget=evidence_budget,
                max_api_calls=max_api_calls,
                max_output_tokens=max_output_tokens,
                max_total_tokens=max_total_tokens,
                request_timeout_seconds=request_timeout_seconds,
                reasoning_effort=reasoning_effort,
            ),
            output=output,
            confirm_api_cost=confirm_api_cost,
            overwrite=overwrite,
        )
    except (ExternalLLMAuthorizationError, ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(payload, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-external-llm-expanded")
def benchmark_external_llm_expanded(
    provider: Annotated[str, typer.Option(help="openai or deepseek")] = "deepseek",
    model: Annotated[
        str, typer.Option(help="Responses API model identifier")
    ] = "deepseek-flash",
    seed: Annotated[
        int, typer.Option(help="First deterministic source-world seed")
    ] = 0,
    pairs: Annotated[
        int, typer.Option(min=1, help="Number of source/target world pairs")
    ] = 20,
    evidence_budget: Annotated[
        int, typer.Option(min=0, help="Shared interventions per mechanism and world")
    ] = 6,
    max_api_calls: Annotated[
        int | None,
        typer.Option(min=1, help="Hard cap; defaults to exactly three calls per pair"),
    ] = None,
    max_transient_retries: Annotated[
        int,
        typer.Option(min=0, help="Audited retries for completed but invalid JSON"),
    ] = 6,
    max_output_tokens: Annotated[
        int, typer.Option(min=1, help="Per-request output-token cap")
    ] = 4_000,
    max_total_tokens: Annotated[
        int, typer.Option(min=1, help="Conservative total input/output-token cap")
    ] = 1_500_000,
    request_timeout_seconds: Annotated[
        float, typer.Option(min=1, help="Per-request network timeout")
    ] = 120.0,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(help="Optional Responses API reasoning effort: none/low/high/max"),
    ] = "none",
    confirm_api_cost: Annotated[
        bool,
        typer.Option(
            "--confirm-api-cost",
            help="Explicitly authorize API usage within the supplied hard caps",
        ),
    ] = False,
    output: Annotated[
        Path, typer.Option(help="Expanded external baseline artifact directory")
    ] = Path("runs/deepseek-expanded-baselines"),
    overwrite: Annotated[
        bool, typer.Option(help="Replace the known expanded baseline artifacts")
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Resume from the per-pair progress checkpoint")
    ] = False,
) -> None:
    from minecraft_scientist.evaluation.expanded_external_llm import (
        run_expanded_external_llm_baselines,
    )
    from minecraft_scientist.evaluation.external_llm import (
        ExternalLLMAuthorizationError,
        ExternalLLMConfig,
    )

    required_calls = pairs * 3
    try:
        payload = run_expanded_external_llm_baselines(
            config=ExternalLLMConfig(
                provider=provider,
                model=model,
                first_seed=seed,
                seeds=pairs,
                evidence_budget=evidence_budget,
                max_api_calls=max_api_calls or required_calls + max_transient_retries,
                max_output_tokens=max_output_tokens,
                max_total_tokens=max_total_tokens,
                request_timeout_seconds=request_timeout_seconds,
                reasoning_effort=reasoning_effort,
            ),
            output=output,
            confirm_api_cost=confirm_api_cost,
            overwrite=overwrite,
            resume=resume,
            max_transient_retries=max_transient_retries,
        )
    except (ExternalLLMAuthorizationError, ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(payload, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("inspect-theory")
def inspect_theory(
    run_directory: Annotated[Path, typer.Argument(help="Completed run directory")],
    theory_id: Annotated[str, typer.Argument(help="Theory ID to inspect")],
) -> None:
    path = run_directory / "theories.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"missing theory log: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [item for item in rows if item["theory_id"] == theory_id]
    if not matches:
        raise typer.BadParameter(f"theory not found: {theory_id}")
    table = Table(title=theory_id)
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Parent")
    table.add_column("Representation")
    table.add_column("Score")
    table.add_column("Confidence")
    table.add_column("Evidence")
    table.add_column("Counterexamples")
    for item in matches:
        table.add_row(
            str(item["version"]),
            item["status"],
            str(item["parent_version"] or "-"),
            item["ir"]["representation"],
            f"{item['validation_score']:.4f}",
            f"{item['confidence']:.3f}",
            str(len(item["evidence_ids"])),
            str(len(item["counterexample_ids"])),
        )
    console.print(table)
    for item in matches:
        console.print(f"\n[bold]v{item['version']} status[/bold]: {item['status']}")
        console.print(
            f"[bold]v{item['version']} evidence IDs[/bold]: "
            + (", ".join(item["evidence_ids"]) or "-")
        )
        console.print(
            f"[bold]v{item['version']} counterexample IDs[/bold]: "
            + (", ".join(item["counterexample_ids"]) or "-")
        )
        console.print(f"[bold]v{item['version']} TheoryIR[/bold]")
        console.print_json(json.dumps(item["ir"], sort_keys=True))


@app.command("inspect-visual-map")
def inspect_visual_map(
    map_path: Annotated[Path, typer.Argument(help="Visual topological map JSON")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON")
    ] = False,
) -> None:
    """Inspect route evidence and lifecycle state without exposing motor handles."""

    from minecraft_scientist.cognitive_map import VisualTopologicalMap

    if not map_path.exists():
        raise typer.BadParameter(f"visual map does not exist: {map_path}")
    try:
        topology = VisualTopologicalMap.load(map_path)
        saved_format_version = json.loads(map_path.read_text(encoding="utf-8"))[
            "format_version"
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"invalid visual map: {exc}") from exc
    routes = topology.routes()
    landmarks = topology.landmarks()
    payload = {
        "format_version": saved_format_version,
        "landmark_count": len(landmarks),
        "calibrated_landmark_count": sum(
            landmark.calibration is not None for landmark in landmarks
        ),
        "route_count": len(routes),
        "landmarks": [landmark.model_dump(mode="json") for landmark in landmarks],
        "routes": [
            {
                **route.model_dump(mode="json"),
                "confidence": route.confidence,
            }
            for route in routes
        ],
    }
    if json_output:
        console.print_json(json.dumps(payload, sort_keys=True))
        return
    console.print(
        f"Visual map v{saved_format_version}: "
        f"{payload['landmark_count']} landmarks "
        f"({payload['calibrated_landmark_count']} calibrated), "
        f"{payload['route_count']} routes"
    )
    calibrated = [
        landmark for landmark in landmarks if landmark.calibration is not None
    ]
    if calibrated:
        landmark_table = Table(title="Calibrated landmarks")
        landmark_table.add_column("Landmark")
        landmark_table.add_column("Radius", justify="right")
        landmark_table.add_column("Positive", justify="right")
        landmark_table.add_column("Negative", justify="right")
        landmark_table.add_column("Nearest negative", justify="right")
        for landmark in calibrated:
            calibration = landmark.calibration
            assert calibration is not None
            landmark_table.add_row(
                landmark.node_id,
                f"{calibration.radius:.3f}",
                str(calibration.positive_samples),
                str(calibration.negative_samples),
                f"{calibration.nearest_negative_distance:.3f}",
            )
        console.print(landmark_table)
    table = Table(title=map_path.name)
    table.add_column("Route")
    table.add_column("Status")
    table.add_column("Trials", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Mean cost", justify="right")
    table.add_column("Roles")
    for route in routes:
        table.add_row(
            route.route_id,
            route.status.value,
            str(route.trials),
            str(route.successes),
            f"{route.confidence:.3f}",
            str(route.failure_streak),
            f"{route.mean_cost:.3f}",
            " -> ".join(route.action_roles),
        )
    console.print(table)
    for route in routes:
        if route.status_reason or route.parent_route_id or route.superseded_by:
            console.print(
                f"{route.route_id}: reason={route.status_reason or '-'}, "
                f"parent={route.parent_route_id or '-'}, "
                f"superseded_by={route.superseded_by or '-'}"
            )


@app.command("inspect-exploration-memory")
def inspect_exploration_memory(
    database: Annotated[Path, typer.Argument(help="Exploration memory SQLite file")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON")
    ] = False,
) -> None:
    """Inspect durable places and frontiers without exposing semantic labels."""

    from minecraft_scientist.exploration import ExplorationMemory

    try:
        with ExplorationMemory.open(database) as memory:
            summary = memory.summary()
            places = memory.places()
            frontiers = memory.frontiers()
            payload = {
                **summary,
                "places": [item.model_dump(mode="json") for item in places],
                "frontiers": [item.model_dump(mode="json") for item in frontiers],
                "transitions": [
                    item.model_dump(mode="json") for item in memory.transitions()
                ],
            }
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise typer.BadParameter(f"invalid exploration memory: {exc}") from exc
    if json_output:
        console.print_json(json.dumps(payload, sort_keys=True))
        return
    console.print(
        f"Exploration memory v{summary['format_version']}: "
        f"{summary['place_count']} places, "
        f"{summary['active_frontier_count']} active frontiers, "
        f"{summary['transition_count']} known transitions, "
        f"integrity={'valid' if summary['integrity_valid'] else 'INVALID'}"
    )
    table = Table(title=database.name)
    table.add_column("Frontier")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Attempts", justify="right")
    table.add_column("Uncertainty", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Roles")
    for frontier in frontiers:
        table.add_row(
            frontier.frontier_id,
            frontier.status.value,
            frontier.source_place_id,
            str(frontier.attempts),
            f"{frontier.uncertainty:.3f}",
            f"{frontier.risk:.3f}",
            " -> ".join(frontier.action_roles),
        )
    console.print(table)


@app.command("benchmark-exploration")
def benchmark_exploration(
    output: Annotated[
        Path, typer.Option(help="Persistent exploration artifact directory")
    ] = Path("runs/persistent-exploration"),
    seed: Annotated[int, typer.Option(help="Deterministic world seed")] = 0,
    budget: Annotated[
        int, typer.Option(min=1, help="Maximum unknown-frontier attempts")
    ] = 60,
    session_length: Annotated[
        int, typer.Option(min=1, help="Attempts between forced restarts")
    ] = 5,
    overwrite: Annotated[
        bool, typer.Option(help="Replace known benchmark artifacts")
    ] = False,
) -> None:
    """Run the offline cross-session frontier-memory release gate."""

    from minecraft_scientist.evaluation import (
        run_persistent_exploration_benchmark,
    )

    try:
        report = run_persistent_exploration_benchmark(
            seed=seed,
            output=output,
            budget=budget,
            session_length=session_length,
            overwrite=overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    table = Table(title="Persistent exploration benchmark")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(str(key), str(value))
    console.print(table)
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-rgb-generalization")
def benchmark_rgb_generalization(
    evidence_root: Annotated[
        Path, typer.Option(help="Disjoint real MineStudio RGB evidence roles")
    ] = Path("runs/adaptive-landmark-calibration/evidence"),
    output: Annotated[
        Path, typer.Option(help="RGB generalization artifact directory")
    ] = Path("runs/rgb-place-generalization"),
    margin: Annotated[
        float, typer.Option(min=0, help="Positive-coverage calibration margin")
    ] = 5,
    overwrite: Annotated[
        bool, typer.Option(help="Replace known RGB benchmark artifacts")
    ] = False,
) -> None:
    """Calibrate and audit persistent RGB place recognition."""

    from minecraft_scientist.evaluation import run_rgb_generalization_benchmark

    try:
        report = run_rgb_generalization_benchmark(
            evidence_root=evidence_root,
            output=output,
            margin=margin,
            overwrite=overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(report, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("minecraft-exploration-smoke")
def minecraft_exploration_smoke(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared vanilla Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Movement probe artifact directory")
    ] = Path("runs/minecraft-open-world-movement"),
    seed: Annotated[int, typer.Option(help="Opaque-ID seed")] = 0,
    stride: Annotated[
        int, typer.Option(min=2, max=32, help="Movement frontier stride")
    ] = 4,
    username: Annotated[
        str, typer.Option(help="Offline Mineflayer username")
    ] = "ExplorerBot",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known movement-probe artifacts")
    ] = False,
) -> None:
    """Run a bounded real vanilla-server opaque movement-frontier search."""

    from minecraft_scientist.minecraft import run_open_world_movement_probe

    try:
        report = run_open_world_movement_probe(
            server_directory=server_directory,
            output=output,
            seed=seed,
            stride=stride,
            username=username,
            overwrite=overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(report, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("minecraft-explore")
def minecraft_explore(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared vanilla Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Persistent open-world exploration artifacts")
    ] = Path("runs/minecraft-open-world-exploration"),
    seed: Annotated[int, typer.Option(help="Opaque-ID seed")] = 0,
    move_budget: Annotated[
        int, typer.Option(min=1, help="Maximum real movement attempts")
    ] = 12,
    restart_interval: Annotated[
        int, typer.Option(min=1, help="Attempts between controller restarts")
    ] = 3,
    stride: Annotated[
        int, typer.Option(min=2, max=32, help="Movement frontier stride")
    ] = 4,
    minimum_successful_moves: Annotated[
        int, typer.Option(min=1, help="Successful-movement release threshold")
    ] = 8,
    minimum_unique_places: Annotated[
        int, typer.Option(min=2, help="Unique-place release threshold")
    ] = 6,
    perceptual_tolerance: Annotated[
        float, typer.Option(min=0, help="Anonymous-scene retrieval radius")
    ] = 0,
    perceptual_ambiguity_margin: Annotated[
        float, typer.Option(min=0, help="Fail-closed scene ambiguity margin")
    ] = 0,
    username: Annotated[
        str, typer.Option(help="Offline Mineflayer username")
    ] = "ExplorerBot",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known exploration artifacts")
    ] = False,
) -> None:
    """Run a bounded multi-session real Minecraft exploration episode."""

    from minecraft_scientist.minecraft import run_open_world_exploration

    try:
        report = run_open_world_exploration(
            server_directory=server_directory,
            output=output,
            seed=seed,
            move_budget=move_budget,
            restart_interval=restart_interval,
            stride=stride,
            minimum_successful_moves=minimum_successful_moves,
            minimum_unique_places=minimum_unique_places,
            perceptual_tolerance=perceptual_tolerance,
            perceptual_ambiguity_margin=perceptual_ambiguity_margin,
            username=username,
            overwrite=overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(report, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("train-planner")
def train_planner(
    output: Annotated[Path, typer.Option(help="Planner checkpoint path")] = Path(
        "checkpoints/four-form-experiment-policy.pt"
    ),
    seed: Annotated[int, typer.Option(help="Training seed")] = 0,
    worlds: Annotated[
        int, typer.Option(min=1, help="Procedural training worlds")
    ] = 5_000,
    validation_worlds: Annotated[
        int, typer.Option(min=1, help="Held-out evaluation worlds")
    ] = 500,
    epochs: Annotated[int, typer.Option(min=1, help="Training epochs")] = 30,
    batch_size: Annotated[int, typer.Option(min=1, help="Batch size")] = 256,
    hidden_size: Annotated[
        int, typer.Option(min=8, help="Permutation-equivariant policy width")
    ] = 64,
    learning_rate: Annotated[
        float, typer.Option(min=1e-8, help="AdamW learning rate")
    ] = 3e-4,
    exploration_rate: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Fraction of off-oracle partial states in the curriculum",
        ),
    ] = 0.2,
    fall_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Fraction of continuous-law worlds"),
    ] = 0.25,
    machine_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Fraction of state-machine worlds"),
    ] = 0.20,
    transformation_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Fraction of program-chain worlds"),
    ] = 0.20,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda")] = "auto",
) -> None:
    try:
        from minecraft_scientist.training import TrainingConfig, train_policy
    except ImportError as exc:
        raise typer.BadParameter(
            "training dependencies are missing; run: uv sync --extra train"
        ) from exc
    config = TrainingConfig(
        seed=seed,
        worlds=worlds,
        validation_worlds=validation_worlds,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        exploration_rate=exploration_rate,
        fall_fraction=fall_fraction,
        machine_fraction=machine_fraction,
        transformation_fraction=transformation_fraction,
        device=device,
    )
    metrics = train_policy(config, output)
    console.print_json(json.dumps(metrics, sort_keys=True))
    console.print(f"Checkpoint: {output.resolve()}")


@app.command("evaluate-planner")
def evaluate_planner(
    checkpoint: Annotated[Path, typer.Argument(help="Trained planner checkpoint")],
    worlds: Annotated[int, typer.Option(min=1, help="Held-out worlds")] = 1_000,
    seed: Annotated[int, typer.Option(help="Evaluation seed")] = 2_000_000,
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda")] = "auto",
    output: Annotated[
        Path | None, typer.Option(help="Optional JSON evaluation report")
    ] = None,
) -> None:
    try:
        from minecraft_scientist.training import evaluate_checkpoint
    except ImportError as exc:
        raise typer.BadParameter(
            "training dependencies are missing; run: uv sync --extra train"
        ) from exc
    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint does not exist: {checkpoint}")
    metrics = evaluate_checkpoint(
        checkpoint,
        worlds=worlds,
        seed=seed,
        device=device,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    console.print_json(json.dumps(metrics, sort_keys=True))


@app.command("check-minecraft-backend")
def check_minecraft_backend() -> None:
    from minecraft_scientist.minecraft import check_backend

    report = check_backend()
    console.print_json(json.dumps(report, sort_keys=True))
    if not report["ready_for_connection"]:
        raise typer.Exit(code=1)


@app.command("check-minestudio-backend")
def check_minestudio() -> None:
    from minecraft_scientist.minecraft import check_minestudio_backend

    report = check_minestudio_backend()
    console.print_json(json.dumps(report, sort_keys=True))
    if not report["ready_for_visual_simulation"]:
        raise typer.Exit(code=1)


@app.command("minecraft-server-status")
def minecraft_server_status(
    directory: Annotated[Path, typer.Option(help="Minecraft server directory")] = Path(
        "minecraft-server"
    ),
) -> None:
    from minecraft_scientist.minecraft import server_status

    console.print_json(json.dumps(server_status(directory), sort_keys=True))


@app.command("minecraft-server-prepare")
def minecraft_server_prepare(
    directory: Annotated[Path, typer.Option(help="Vanilla server directory")] = Path(
        "minecraft-server"
    ),
    seed: Annotated[int, typer.Option(help="Deterministic Minecraft world seed")] = 0,
    accept_eula: Annotated[
        bool,
        typer.Option(
            "--accept-eula",
            help="Confirm that you reviewed and accept the Minecraft EULA",
        ),
    ] = False,
    overwrite_config: Annotated[
        bool,
        typer.Option(help="Replace server.properties, but never an unknown jar"),
    ] = False,
    fabric: Annotated[
        bool,
        typer.Option(
            "--fabric/--vanilla",
            help="Install the compiled research rule mod and Fabric runtime",
        ),
    ] = True,
    mod_bundle: Annotated[
        Path,
        typer.Option(help="Directory produced by the Fabric stageServerMods task"),
    ] = Path("mod/scientist-world-rules/build/server-mods"),
    overwrite_mods: Annotated[
        bool,
        typer.Option(help="Replace known research mods when their build changes"),
    ] = False,
) -> None:
    from minecraft_scientist.minecraft import (
        MinecraftEulaRequired,
        prepare_fabric_server,
        prepare_server,
    )

    try:
        if fabric:
            report = prepare_fabric_server(
                directory,
                accept_eula=accept_eula,
                mod_bundle=mod_bundle,
                seed=seed,
                overwrite_config=overwrite_config,
                overwrite_mods=overwrite_mods,
            )
        else:
            report = prepare_server(
                directory,
                accept_eula=accept_eula,
                seed=seed,
                overwrite_config=overwrite_config,
            )
    except MinecraftEulaRequired as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(report, sort_keys=True))


@app.command("minecraft-server-start")
def minecraft_server_start(
    directory: Annotated[
        Path, typer.Option(help="Prepared vanilla server directory")
    ] = Path("minecraft-server"),
    min_memory: Annotated[str, typer.Option(help="Initial Java heap, e.g. 1G")] = "1G",
    max_memory: Annotated[str, typer.Option(help="Maximum Java heap, e.g. 3G")] = "3G",
    flavor: Annotated[str, typer.Option(help="auto, fabric, or vanilla")] = "auto",
) -> None:
    from minecraft_scientist.minecraft import run_server

    raise typer.Exit(
        code=run_server(
            directory,
            min_memory=min_memory,
            max_memory=max_memory,
            flavor=flavor,
        )
    )


@app.command("minecraft-observe")
def minecraft_observe(
    host: Annotated[str, typer.Option(help="Minecraft server host")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Minecraft server port")
    ] = 25565,
    username: Annotated[str, typer.Option(help="Bot username")] = "ScientistBot",
    auth: Annotated[str, typer.Option(help="offline or microsoft")] = "offline",
    version: Annotated[
        str, typer.Option(help="Minecraft protocol version")
    ] = "1.21.11",
    radius: Annotated[int, typer.Option(min=1, max=32, help="Observation radius")] = 8,
    timeout: Annotated[
        float, typer.Option(min=1, help="Connection timeout seconds")
    ] = 30.0,
    output: Annotated[
        Path | None, typer.Option(help="Optional JSON observation path")
    ] = None,
) -> None:
    from minecraft_scientist.minecraft import MinecraftBridgeClient

    with MinecraftBridgeClient(timeout=timeout) as client:
        connection = client.request(
            "connect",
            {
                "host": host,
                "port": port,
                "username": username,
                "auth": auth,
                "version": version,
                "timeout_ms": int(timeout * 1000),
            },
            timeout=timeout + 5,
        )
        observation = client.request("observe", {"radius": radius})
    payload = {"connection": connection, "observation": observation}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    console.print_json(json.dumps(payload, sort_keys=True))


@app.command("minecraft-live-smoke")
def minecraft_live_smoke(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Opaque live-smoke artifact directory")
    ] = Path("runs/minecraft-live-smoke"),
    seed: Annotated[int, typer.Option(help="Opaque-ID and scenario seed")] = 0,
    username: Annotated[
        str, typer.Option(help="Offline bot username")
    ] = "ScientistBot",
    variant: Annotated[
        str, typer.Option(help="standard or shifted server-side rule variant")
    ] = "standard",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known smoke-test artifacts")
    ] = False,
    server_mode: Annotated[str, typer.Option(help="fabric or gradle")] = "fabric",
) -> None:
    from minecraft_scientist.minecraft import (
        MinecraftConnectionConfig,
        MinecraftWorldAdapter,
        ScientistCurriculumScenario,
        create_server_process,
    )
    from minecraft_scientist.protocols import ActionKind, PrimitiveAction

    report_path = output / "report.json"
    map_path = output / "cognitive-map.json"
    if report_path.exists() and not overwrite:
        raise typer.BadParameter(f"smoke report already exists: {report_path}")
    output.mkdir(parents=True, exist_ok=True)
    with (
        create_server_process(server_directory, mode=server_mode) as server,
        MinecraftWorldAdapter(MinecraftConnectionConfig(username=username)) as world,
    ):
        initial = world.reset(seed)
        ScientistCurriculumScenario(
            username=username,
            seed=seed,
            variant=variant,
        ).apply(server)
        synchronization = world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=20))
        observation = world.observe()
        world.cognitive_map.save(map_path)
    payload = {
        "seed": seed,
        "minecraft_version": "1.21.11",
        "variant": variant,
        "initial_visible_objects": len(initial.objects),
        "arena_visible_objects": len(observation.objects),
        "inventory_types": len(observation.inventory),
        "controller_compliant": synchronization.intervention_compliance,
        "opaque_ids_only": all(
            item.object_id.startswith("obj_") for item in observation.objects
        ),
        "cognitive_map_nodes": world.cognitive_map.graph.number_of_nodes(),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.print_json(json.dumps(payload, sort_keys=True))


@app.command("minecraft-live-run")
def minecraft_live_run(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared Fabric Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Live discovery artifact directory")
    ] = Path("runs/minecraft-live-run"),
    seed: Annotated[int, typer.Option(help="World, mechanism, and opaque-ID seed")] = 0,
    intervention_budget: Annotated[
        int, typer.Option(min=1, help="Maximum live causal experiments")
    ] = 80,
    change_episode: Annotated[
        int | None, typer.Option(help="Episode that activates shifted rules")
    ] = 1,
    username: Annotated[
        str, typer.Option(help="Offline bot username")
    ] = "ScientistBot",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known live-run artifacts")
    ] = False,
    server_mode: Annotated[str, typer.Option(help="fabric or gradle")] = "fabric",
    planner: Annotated[
        str, typer.Option(help="information_gain or learned")
    ] = "information_gain",
    planner_checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint required by the learned planner")
    ] = None,
) -> None:
    from minecraft_scientist.minecraft import run_live_discovery

    result = run_live_discovery(
        server_directory=server_directory,
        output=output,
        seed=seed,
        intervention_budget=intervention_budget,
        change_episode=change_episode,
        username=username,
        overwrite=overwrite,
        server_mode=server_mode,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    console.print(_report_table("Minecraft live discovery", result.report.model_dump()))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("minecraft-live-fall")
def minecraft_live_fall(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared Fabric Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Live equation artifact directory")
    ] = Path("runs/minecraft-live-fall"),
    seed: Annotated[int, typer.Option(help="Hidden live equation seed")] = 0,
    intervention_budget: Annotated[
        int, typer.Option(min=1, help="Maximum experiments per equation version")
    ] = 40,
    username: Annotated[
        str, typer.Option(help="Offline bot username")
    ] = "ScientistBot",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known live equation artifacts")
    ] = False,
    server_mode: Annotated[str, typer.Option(help="fabric or gradle")] = "fabric",
    planner: Annotated[
        str, typer.Option(help="information_gain or learned")
    ] = "information_gain",
    planner_checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint required by the learned planner")
    ] = None,
) -> None:
    from minecraft_scientist.minecraft import run_live_fall_discovery

    payload = run_live_fall_discovery(
        server_directory=server_directory,
        output=output,
        seed=seed,
        intervention_budget=intervention_budget,
        username=username,
        overwrite=overwrite,
        server_mode=server_mode,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    console.print_json(json.dumps(payload, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("minecraft-live-benchmark")
def minecraft_live_benchmark(
    server_directory: Annotated[
        Path, typer.Option(help="Prepared Fabric Minecraft server directory")
    ] = Path("minecraft-server"),
    output: Annotated[
        Path, typer.Option(help="Combined live benchmark artifact directory")
    ] = Path("runs/minecraft-live-benchmark"),
    seed: Annotated[int, typer.Option(help="Deterministic live benchmark seed")] = 0,
    intervention_budget: Annotated[
        int, typer.Option(min=1, help="Rule/state/program experiment budget")
    ] = 80,
    fall_intervention_budget: Annotated[
        int, typer.Option(min=1, help="Equation experiments per theory version")
    ] = 40,
    username: Annotated[
        str, typer.Option(help="Offline bot username")
    ] = "ScientistBot",
    overwrite: Annotated[
        bool, typer.Option(help="Replace known combined live artifacts")
    ] = False,
    server_mode: Annotated[str, typer.Option(help="fabric or gradle")] = "fabric",
    planner: Annotated[
        str, typer.Option(help="information_gain or learned")
    ] = "information_gain",
    planner_checkpoint: Annotated[
        Path | None, typer.Option(help="Checkpoint required by the learned planner")
    ] = None,
) -> None:
    from minecraft_scientist.minecraft import run_live_benchmark

    payload = run_live_benchmark(
        server_directory=server_directory,
        output=output,
        seed=seed,
        intervention_budget=intervention_budget,
        fall_intervention_budget=fall_intervention_budget,
        username=username,
        overwrite=overwrite,
        server_mode=server_mode,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    console.print_json(json.dumps(payload, sort_keys=True))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-identification")
def benchmark_identification(
    output: Annotated[Path, typer.Option(help="New artifact directory")],
    seed: Annotated[int, typer.Option(min=0)] = 0,
    seeds: Annotated[int, typer.Option(min=1)] = 4,
    budget: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Known-candidate H1 development protocol; not an innovation claim."""
    from minecraft_scientist.evaluation.identification import (
        run_identification_benchmark,
    )

    result = run_identification_benchmark(
        output=output, seed=seed, seeds=seeds, budget=budget
    )
    console.print_json(
        json.dumps({
            "runs": len(result["rows"]),
            "audits_valid": result["all_independent_audits_valid"],
            "scope": result["scope"],
        })
    )
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-diagnosis")
def benchmark_diagnosis(
    output: Annotated[Path, typer.Option(help="New artifact directory")],
    seed: Annotated[int, typer.Option(min=0)] = 0,
    seeds: Annotated[int, typer.Option(min=1)] = 2,
    budget: Annotated[int, typer.Option(min=1)] = 32,
) -> None:
    """Known-family hidden-control H2 development, not complete fault diagnosis."""
    from minecraft_scientist.evaluation.diagnosis import run_diagnosis_benchmark

    result = run_diagnosis_benchmark(output=output, seed=seed, seeds=seeds, budget=budget)
    console.print_json(json.dumps({
        "runs": len(result["rows"]),
        "audits_valid": result["all_independent_audits_valid"],
        "scope": result["scope"],
    }))
    console.print(f"Artifacts: {output.resolve()}")


@app.command("benchmark-revision")
def benchmark_revision(
    output: Annotated[Path, typer.Option(help="New artifact directory")],
    seed: Annotated[int, typer.Option(min=0)] = 0,
    seeds: Annotated[int, typer.Option(min=1)] = 2,
    budget: Annotated[int, typer.Option(min=1)] = 240,
) -> None:
    """Bounded online H3 development; not open-family discovery or novelty proof."""
    from minecraft_scientist.evaluation.revision import run_revision_benchmark

    result = run_revision_benchmark(output=output, seed=seed, seeds=seeds, budget=budget)
    console.print_json(json.dumps({
        "runs": len(result["rows"]),
        "audits_valid": result["all_independent_audits_valid"],
        "summaries": result["summaries"],
        "scope": result["scope"],
    }))
    console.print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    app()
