"""Offline multi-seed discovery/revision release check with exact repeat artifacts.

This validates the structured scientific core, not live navigation, broad RGB
generalization, long-operation recovery or completion of the full project goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

from minecraft_scientist.discovery import (
    PROTOCOL_CANDIDATES,
    HeuristicHypothesisGenerator,
)
from minecraft_scientist.evaluation.benchmark import _primitive_protocol
from minecraft_scientist.evaluation.full import run_full_benchmark
from minecraft_scientist.planning import ExperimentExecutor
from minecraft_scientist.protocols import (
    SymbolPrimitive,
    TheoryVersion,
    TransitionRecord,
)
from minecraft_scientist.world import WorldFactory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    root = Path(__file__).resolve().parents[1]
    package = Path(run_full_benchmark.__code__.co_filename).resolve().parents[1]
    return {
        f"package/{path.relative_to(package).as_posix()}": path
        for path in sorted(package.rglob("*.py"))
    } | {
        "verify_core_release.py": Path(__file__).resolve(),
        "pyproject.toml": root / "pyproject.toml",
        "uv.lock": root / "uv.lock",
    }


def artifact_ledger(directory):
    expected = {
        "full-benchmark.json",
        "fall-theories.json",
        "causal-graph.json",
        "experience.sqlite3",
        "structured/benchmark.json",
        "structured/transfer-evidence.json",
        "structured/library-transfer-events.jsonl",
        "structured/no-library-transfer-events.jsonl",
    }
    expected.update(
        f"structured/{condition}/{name}"
        for condition in ("full", "no-library")
        for name in (
            "config.json",
            "events.jsonl",
            "theories.jsonl",
            "symbols.json",
            "report.json",
        )
    )
    files = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file()
    }
    if set(files) != expected or any(path.is_symlink() for path in files.values()):
        raise ValueError("full benchmark artifact set is incomplete or unexpected")
    return {name: digest(path) for name, path in sorted(files.items())}


def verify_transfer(directory):
    evidence = json.loads((directory / "transfer-evidence.json").read_bytes())
    symbols = json.loads((directory / "full/symbols.json").read_bytes())
    mvp = json.loads((directory / "benchmark.json").read_bytes())
    if evidence["seed"] != mvp["seed"] + 10_000:
        raise ValueError("transfer must use the declared independent world")
    for name, metric in (
        ("library", "full_transfer_experiments"),
        ("no-library", "no_library_transfer_experiments"),
    ):
        metadata = evidence[name]
        if name == "library":
            if metadata["symbol"] not in symbols:
                raise ValueError("transfer primitive was not actually learned")
            primitive = SymbolPrimitive.model_validate(metadata["symbol"])
            protocols = (_primitive_protocol(primitive),)
        else:
            if metadata["symbol"] is not None:
                raise ValueError("scratch transfer cannot receive a learned symbol")
            protocols = tuple(sorted(PROTOCOL_CANDIDATES))
        if metadata["protocol_candidates"] != [
            list(protocol) for protocol in protocols
        ]:
            raise ValueError(
                "transfer candidate programs disagree with learned definition"
            )
        records = [
            TransitionRecord.model_validate_json(line)
            for line in (directory / f"{name}-transfer-events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        bundle = WorldFactory.create(evidence["seed"], change_episode=None)
        bundle.controller.begin_episode(0)
        groups = HeuristicHypothesisGenerator().classify(bundle.agent_view.observe())
        processor, resources = (
            next(iter(groups["processors"])),
            tuple(groups["resources"]),
        )
        executor, replay, experiment_ids = ExperimentExecutor(bundle.agent_view), [], []
        terminal = None
        for protocol in protocols:
            for item in resources:
                terminal = executor.run_protocol_test(
                    processor, item, protocol, mechanism="transformation_chain"
                )
                experiment_ids.append(terminal.experiment_id)
                replay.extend(terminal.transitions)
                if (
                    terminal.compliant
                    and terminal.outcome["success"]
                    and terminal.outcome["output_id"] is not None
                ):
                    break
            else:
                continue
            break
        if (
            terminal is None
            or not terminal.compliant
            or not terminal.outcome["success"]
            or terminal.outcome["output_id"] is None
            or [record.model_dump(mode="json") for record in replay]
            != [record.model_dump(mode="json") for record in records]
            or metadata["experiment_ids"] != experiment_ids
            or metadata["experiment_count"] != len(experiment_ids)
            or mvp["ablation"][metric] != len(experiment_ids)
            or metadata["output_id"] != terminal.outcome["output_id"]
            or metadata["successful_input_id"]
            != terminal.intervention.controlled_variables["input_id"]
        ):
            raise ValueError("persisted transfer actions did not independently replay")
    return True


def verify_persisted(directory, payload):
    # Only closed, checkpointed databases qualify for immutable read-only replay.
    # This rejects pending WAL state before opening and avoids creating sidecars.
    artifact_ledger(directory)
    stored = json.loads((directory / "full-benchmark.json").read_bytes())
    if stored != payload:
        raise ValueError("persisted report disagrees with executed benchmark")
    mvp = json.loads((directory / "structured/benchmark.json").read_bytes())
    checks = {
        "benchmark_gate": payload["all_passed"] is True,
        "all_four_laws": payload["continuous_law"]["representation"] == "equation"
        and payload["continuous_law"]["law_recovery"] == 1.0
        and payload["structured"]["hidden_intervention_accuracy"] == 1.0
        and set(payload["structured"]["mechanism_accuracy"])
        == {"tool_capability", "machine_protocol", "transformation_chain"}
        and all(
            value == 1.0
            for value in payload["structured"]["mechanism_accuracy"].values()
        ),
        "equation_shadow_validation": payload["continuous_law"]["validation_passed"]
        is True,
        "unchanged_law_transfer": payload["world_transfer"]["law_recovery"] == 1.0
        and payload["world_transfer"]["experiment_savings"] > 0,
        "equation_revision": payload["theory_revision"]["shadow_validation_passed"]
        is True
        and payload["theory_revision"]["new_law_recovery"] == 1.0
        and payload["theory_revision"]["change_detection_delay"] > 0,
        "symbol_ablation": mvp["ablation"]["description_length_delta"] < 0
        and mvp["ablation"]["downstream_experiment_savings"] > 0
        and mvp["no_library"]["symbol_count"] == 0
        and mvp["no_library"]["hidden_intervention_accuracy"] == 1.0,
        "task_utility": payload["metrics"]["task_success"] == 1.0,
        "executable_symbol_transfer_replayed": verify_transfer(
            directory / "structured"
        ),
    }
    for condition, report_key in (("full", "full"), ("no-library", "no_library")):
        path = directory / "structured" / condition
        report = json.loads((path / "report.json").read_bytes())
        records = [
            TransitionRecord.model_validate_json(line)
            for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        theories = [
            TheoryVersion.model_validate_json(line)
            for line in (path / "theories.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        symbols = [
            SymbolPrimitive.model_validate(item)
            for item in json.loads((path / "symbols.json").read_bytes())
        ]
        tool_history = sorted(
            (item for item in theories if item.theory_id == "tool-capability"),
            key=lambda item: item.version,
        )
        checks[f"{condition}_persisted_evidence"] = (
            report == mvp[report_key]
            and bool(records)
            and bool(theories)
            and len({record.transition_id for record in records}) == len(records)
            and len(symbols) == report["symbol_count"]
            and [item.status.value for item in tool_history]
            == ["superseded", "accepted"]
            and bool(tool_history[0].counterexample_ids)
            and tool_history[1].parent_version == tool_history[0].version
        )
    fall_theories = [
        TheoryVersion.model_validate(item)
        for item in json.loads((directory / "fall-theories.json").read_bytes())
    ]
    checks["persisted_equation_history"] = (
        [item.status.value for item in fall_theories] == ["superseded", "accepted"]
        and bool(fall_theories[0].counterexample_ids)
        and fall_theories[1].parent_version == fall_theories[0].version
    )
    connection = sqlite3.connect(
        (directory / "experience.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    try:
        rows = connection.execute(
            "SELECT sequence, transition_id, mechanism, compliant, controller_failure, payload_json FROM transitions ORDER BY sequence"
        ).fetchall()
        records = [TransitionRecord.model_validate_json(row[5]) for row in rows]
        checks["read_only_experience_replay"] = len(rows) == payload["continuous_law"][
            "experience_transitions"
        ] > 0 and all(
            row[0] == index + 1
            and row[1] == record.transition_id
            and row[2] == "fall_damage"
            and bool(row[3]) == record.intervention_compliance
            and row[4] == record.controller_failure
            for index, (row, record) in enumerate(zip(rows, records, strict=True))
        )
    finally:
        connection.close()
    if not all(checks.values()):
        raise ValueError(f"structured release checks failed: {checks}")
    return checks


def verify_release(*, output, seed=0, seeds=30, intervention_budget=80):
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 1000
    ):
        raise ValueError("release seeds must be positive and bounded")
    if type(intervention_budget) is not int or not 1 <= intervention_budget <= 1000:
        raise ValueError("release intervention budget must be positive and bounded")
    if output.exists():
        raise FileExistsError("refusing to overwrite any existing release directory")
    sources = source_paths()
    source_hashes = {name: digest(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    manifest = {
        "seed": seed,
        "seeds": list(range(seed, seed + seeds)),
        "intervention_budget": intervention_budget,
        "repetitions": 2,
        "planner": "information_gain",
        "hypothesis_generator": "heuristic",
        "source_sha256": source_hashes,
        "scope": "offline four-form structured discovery/revision only; not live, visual or full-project completion",
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = digest(config_path)
    results, failure = [], None
    try:
        for current in manifest["seeds"]:
            runs, reports, validations = [], [], []
            for repetition in range(2):
                directory = output / f"seed-{current:04d}" / f"repeat-{repetition}"
                report = run_full_benchmark(
                    seed=current,
                    intervention_budget=intervention_budget,
                    output=directory,
                    overwrite=False,
                )
                validations.append(verify_persisted(directory, report))
                runs.append(artifact_ledger(directory))
                reports.append(report)
            identical = runs[0] == runs[1] and reports[0] == reports[1]
            results.append(
                {
                    "seed": current,
                    "passed": identical,
                    "checks": validations[0],
                    "repeat_checks": validations[1],
                    "identical_artifacts": identical,
                    "artifact_sha256": runs,
                    "metrics": reports[0]["metrics"],
                    "symbol_count": reports[0]["structured"]["symbol_count"],
                    "symbol_reuse": reports[0]["structured"]["symbol_reuse"],
                    "transfer_experiment_savings": reports[0]["world_transfer"][
                        "experiment_savings"
                    ],
                }
            )
            print(
                json.dumps(
                    {
                        "seed": current,
                        "passed": identical,
                        "identical_artifact_count": len(runs[0]) if identical else 0,
                    }
                ),
                flush=True,
            )
            if not identical:
                raise ValueError(f"seed {current} changed deterministic artifacts")
    except Exception as error:  # noqa: BLE001 -- preserve partial release evidence and fail the gate
        failure = {"type": type(error).__name__, "message": str(error)}
    checks = {
        "all_requested_seeds_complete": [item["seed"] for item in results]
        == manifest["seeds"],
        "all_discovery_replay_checks": bool(results)
        and all(item["passed"] for item in results),
        "sources_unchanged": all(
            digest(path) == source_hashes[name] for name, path in sources.items()
        ),
        "source_snapshots_verified": all(
            digest(output / "source-snapshot" / name) == value
            for name, value in source_hashes.items()
        ),
        "preregistration_unchanged": digest(config_path) == config_hash,
        "no_runtime_failure": failure is None,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "config_sha256": config_hash,
        "failure": failure,
        "runs": results,
        "scope": manifest["scope"],
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
        "summary": {
            "completed_seeds": len(results),
            "mean_experiments": statistics.mean(
                item["metrics"]["experiment_count"] for item in results
            )
            if results
            else None,
            "minimum_equation_transfer_savings": min(
                (item["transfer_experiment_savings"] for item in results), default=None
            ),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--intervention-budget", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_release(
        output=args.output,
        seed=args.seed,
        seeds=args.seeds,
        intervention_budget=args.intervention_budget,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
