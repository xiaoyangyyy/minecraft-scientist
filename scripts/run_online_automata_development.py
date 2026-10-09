"""Paid online baseline development, not a full fair H3/structural holdout suite."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "adaptive_automata"))
sys.path.insert(0, str(ROOT / "research" / "online_automata"))

from audit_online import audit_online, control_groups, local_events
from online import online_relearn
from paid_transfer import METHODS
from worlds import CASES, online_task

from minecraft_scientist.world.revision import REVISION_CASES, revision_task


def dump(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def hashes():
    paths = list((ROOT / "src" / "minecraft_scientist").rglob("*.py"))
    for directory in ("adaptive_automata", "online_automata"):
        paths += list((ROOT / "research" / directory).glob("*.py"))
        paths += [ROOT / "research" / directory / "README.md"]
    paths += [
        ROOT / "scripts" / "run_paired_transfer_development.py",
        Path(__file__).resolve(),
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def run(*, output, seed=0, seeds=2, suite="all", budget=600, old_budget=250, horizon=5):
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 16
        or type(suite) is not str
        or suite not in ("all", "automata", "controlled")
        or any(type(v) is not int or v < 1 for v in (budget, old_budget))
        or type(horizon) is not int
        or not 1 <= horizon <= 8
    ):
        raise ValueError("invalid online development runner protocol")
    output.mkdir(parents=True, exist_ok=False)
    sources = hashes()
    protocol = {
        "budget": budget,
        "old_budget": old_budget,
        "horizon": horizon,
        "seed": 20261008,
        "conformance_count": 8,
        "monitor_rounds": 4,
        "monitor_count": 8,
        "operation_budget": 200_000,
        "reference_operation_budget": 200_000,
    }
    scope = "paired one-crisis issued-behavior baseline development with hidden episode switches; not full knowledge-matched H3, factor attribution, structural holdout, unknown noise, learned symbol transfer or novelty"
    grid = []
    for world_seed in range(seed, seed + seeds):
        if suite in ("all", "automata"):
            grid += [
                (world_seed, "automata", case, boundary)
                for case, boundary in itertools.product(CASES, (0, 8, 16, 24, 1000))
            ]
        if suite in ("all", "controlled"):
            grid += [(world_seed, "controlled", case, 25) for case in REVISION_CASES]
    registration = {
        "schema": "online-automata-development-registration-v1",
        "source_sha256": sources,
        "protocol": protocol,
        "methods": list(METHODS),
        "private_grid": [list(c) for c in grid],
        "scope": scope,
        "knowledge_boundary": "only capability/anonymous alphabet/costs reach both agents; controller/bootstrap candidates stay evaluator-owned. Existing H3 full method has additional bootstrap/DSL/factor knowledge, so not compared as matched here",
    }
    dump(output / "registration.json", registration)
    rows, pairs, timings = [], [], []
    for world_seed, family, case, boundary in grid:
        previous, paired = None, {}
        for method in METHODS:
            directory = (
                output
                / f"seed-{world_seed}"
                / family
                / case
                / f"boundary-{boundary}"
                / method
            )
            directory.mkdir(parents=True)
            if family == "automata":
                task = online_task(world_seed, case, boundary=boundary)
                actions, outputs, costs, private, view = (
                    task.actions,
                    task.outputs,
                    task.costs,
                    task.private,
                    task.view,
                )
            else:
                task = revision_task(world_seed, case, boundary=boundary)
                actions = tuple(task.costs)
                outputs = tuple(task.bootstrap[0].mechanism_ir.parameters["outputs"])
                costs, private, view = task.costs, task.private_spec, task.view
            config = {
                "schema": "online-relearning-config-v1",
                "method": method,
                "actions": list(actions),
                "outputs": list(outputs),
                "costs": costs,
                "protocol": protocol,
            }
            dump(directory / "config.json", config)
            wall, cpu = perf_counter(), process_time()
            result = online_relearn(
                view, actions, outputs, costs, method=method, **protocol
            )
            learning_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "run.json", result)
            pre = {
                k: result[k]
                for k in ("old", "public_old_theory", "monitoring", "anomaly")
            }
            if previous is not None and previous != pre:
                raise RuntimeError(
                    "Different paid bootstrap/monitoring history; retain partial evidence"
                )
            previous = pre
            events = (
                {
                    "epoch": group.get(
                        "epoch",
                        group.get("config", {}).get("protocol", {}).get("epoch"),
                    ),
                    "event": event,
                }
                for group in control_groups(result)
                for event in local_events(group)
            )
            (directory / "events.jsonl").write_text(
                "".join(
                    json.dumps(e, sort_keys=True, allow_nan=False) + "\n"
                    for e in events
                ),
                encoding="utf-8",
                newline="\n",
            )
            (directory / "theories.jsonl").write_text(
                "".join(
                    json.dumps(e, sort_keys=True, allow_nan=False) + "\n"
                    for e in result["theory_events"]
                ),
                encoding="utf-8",
                newline="\n",
            )
            private_path = directory / "evaluator" / "world.json"
            private_path.parent.mkdir()
            dump(private_path, private)
            wall, cpu = perf_counter(), process_time()
            audit = audit_online(config, result, private_path)
            audit_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "audit.json", audit)
            if not audit["valid"]:
                raise RuntimeError(
                    f"Online audit failed: {directory}; retain partial evidence"
                )
            monitor_spent = sum(
                q["spent"] for b in result["monitoring"] for q in b["queries"]
            )
            row = {
                "seed": world_seed,
                "family": family,
                "case": case,
                "private_boundary": boundary,
                "method": method,
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "spent": result["spent"],
                "old_spent": result["old"]["run"]["spent"],
                "monitor_spent": monitor_spent,
                "repair_spent": 0
                if result["new"] is None
                else result["new"]["run"]["spent"],
                "evaluation": audit["evaluation"],
            }
            rows.append(row)
            paired[method] = row
            timings.append(
                {
                    k: row[k]
                    for k in ("seed", "family", "case", "private_boundary", "method")
                }
                | {
                    "complete_online_learning_monitoring_and_validation": learning_time,
                    "complete_audit_and_private_score": audit_time,
                }
            )
        left, right = paired["from_scratch"], paired["adaptive"]
        both_correct = all(
            r["evaluation"]["new_behavior_correct_at_H"] is True for r in (left, right)
        )
        pairs.append(
            {
                "seed": world_seed,
                "family": family,
                "case": case,
                "private_boundary": boundary,
                "same_paid_bootstrap_and_monitoring": True,
                "both_methods_repaired_behavior_correct": both_correct,
                "adaptive_minus_scratch_total_cost": right["spent"] - left["spent"],
                "cost_difference_is_repair_success_conditioned": both_correct,
            }
        )
    if sources != hashes():
        raise RuntimeError("Source drift; retain partial data without final manifest")
    summary = {"pairs": len(pairs), "campaigns": len(rows), "methods": {}}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "runs": len(selected),
            "unresolved": sum(r["status"] == "unresolved" for r in selected),
            "bounded_monitoring_no_anomaly": sum(
                r["status"] == "bounded_monitoring_completed_no_anomaly"
                for r in selected
            ),
            "finite_replacements": sum(
                r["status"] == "revision_finitely_validated" for r in selected
            ),
            "correct_replacements_at_H": sum(
                r["evaluation"]["new_behavior_correct_at_H"] is True for r in selected
            ),
            "wrong_replacements_at_H": sum(
                r["evaluation"]["new_behavior_correct_at_H"] is False for r in selected
            ),
            "wrong_old_models_at_H": sum(
                r["evaluation"]["old_behavior_correct_at_H"] is False for r in selected
            ),
            "anomaly_without_in_H_change_or_before_boundary": sum(
                r["evaluation"]["anomaly_without_in_H_change_or_before_boundary"]
                for r in selected
            ),
            "post_boundary_anomalies_detected": sum(
                r["evaluation"]["post_boundary_anomaly_detected"] for r in selected
            ),
        }
    success_pairs = [p for p in pairs if p["both_methods_repaired_behavior_correct"]]
    summary |= {
        "both_repaired_correct_pairs": len(success_pairs),
        "adaptive_cheaper_among_both_repaired_correct": sum(
            p["adaptive_minus_scratch_total_cost"] < 0 for p in success_pairs
        ),
        "scratch_cheaper_among_both_repaired_correct": sum(
            p["adaptive_minus_scratch_total_cost"] > 0 for p in success_pairs
        ),
        "tied_cost_among_both_repaired_correct": sum(
            p["adaptive_minus_scratch_total_cost"] == 0 for p in success_pairs
        ),
    }
    report = {
        "schema": "online-automata-development-report-v1",
        "registration": registration,
        "scope": scope,
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "all_campaign_audits_valid": True,
        "limitations": "all unresolved/wrong/early/late/no-change cases retained; repair-success-conditioned costs are not detection/causal success or statistical superiority",
    }
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {
            "schema": "online-automata-timing-v1",
            "rows": timings,
            "scope": "complete online computation and complete audit separately; excludes setup/serialization; nondeterministic",
        },
    )
    manifest = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    dump(
        output / "manifest.json",
        {
            "schema": "online-automata-manifest-v1",
            "sha256": manifest,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    print(
        json.dumps(summary | {"deterministic_artifacts": len(manifest)}, sort_keys=True)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--suite", choices=("all", "automata", "controlled"), default="all"
    )
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--old-budget", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=5)
    run(**vars(parser.parse_args()))
