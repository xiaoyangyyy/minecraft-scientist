"""Shared paid lifecycle, four discovery methods, early/normal/late hidden changes.

Development templates, NOT structural holdouts or a statistical novelty claim.
"""

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
sys.path.insert(0, str(ROOT / "scripts"))

from audit_online import audit_online, control_groups, local_events
from matched_repair import MATCHED_METHODS, POLICY
from online import online_relearn
from run_online_automata_development import dump
from run_online_automata_development import hashes as upstream_hashes

from minecraft_scientist.world.revision import REVISION_CASES, revision_task


def hashes():
    return upstream_hashes() | {
        Path(__file__).relative_to(ROOT).as_posix(): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    }


def summarize(rows):
    return {
        "runs": len(rows),
        "old_correct_at_H": sum(
            r["evaluation"]["old_behavior_correct_at_H"] is True for r in rows
        ),
        "new_correct_at_H": sum(
            r["evaluation"]["new_behavior_correct_at_H"] is True for r in rows
        ),
        "new_wrong_at_H": sum(
            r["evaluation"]["new_behavior_correct_at_H"] is False for r in rows
        ),
        "finite_replacements": sum(
            r["status"] == "revision_finitely_validated" for r in rows
        ),
        "unresolved": sum(r["status"] == "unresolved" for r in rows),
        "bounded_no_anomaly": sum(
            r["status"] == "bounded_monitoring_completed_no_anomaly" for r in rows
        ),
        "detected_post_boundary": sum(
            r["evaluation"]["post_boundary_anomaly_detected"] for r in rows
        ),
        "false_or_early_anomaly": sum(
            r["evaluation"]["anomaly_without_in_H_change_or_before_boundary"]
            for r in rows
        ),
        "actual_total_cost": sum(r["spent"] for r in rows),
    }


def run(
    *,
    output,
    seed=0,
    seeds=2,
    budget=600,
    old_budget=250,
    horizon=5,
    boundaries=(6, 25, 100),
):
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 16
        or any(type(v) is not int or v < 1 for v in (budget, old_budget))
        or type(horizon) is not int
        or not 4 <= horizon <= 5
        or type(boundaries) is not tuple
        or not boundaries
        or len(set(boundaries)) != len(boundaries)
        or any(type(v) is not int or not 6 <= v <= 100 for v in boundaries)
    ):
        raise ValueError("invalid matched H3 development protocol")
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
    grid = list(
        itertools.product(range(seed, seed + seeds), REVISION_CASES, boundaries)
    )
    scope = "shared paid known-family prelude/monitoring/final validation; four discovery methods, declared edit priors versus general finite-state learning; fixed deterministic development templates, not causal uniqueness, structure holdout, noisy-control solution or innovation"
    registration = {
        "schema": "matched-h3-development-registration-v1",
        "source_sha256": sources,
        "methods": list(MATCHED_METHODS),
        "protocol": protocol,
        "repair_policy": POLICY,
        "private_grid": [list(x) for x in grid],
        "scope": scope,
        "knowledge_boundary": "all receive identical public bootstrap factor family, calibration, old paid IR and anomaly; structured methods use declared edit grammar/controllers, baselines use their declared general finite-state hypothesis language. Representation/search bias disclosed, not a pure planning-only or unique-factor comparison",
    }
    dump(output / "registration.json", registration)
    rows, comparisons, timings = [], [], []
    for world_seed, case, boundary in grid:
        prelude, group = None, {}
        for method in MATCHED_METHODS:
            task = revision_task(world_seed, case, boundary=boundary)
            family = {
                "hypotheses": [h.model_dump(mode="json") for h in task.bootstrap],
                "calibration": list(task.calibration),
                "discovery_budget": 40,
                "shadow_count": 3,
            }
            config = {
                "schema": "online-relearning-config-v3",
                "method": method,
                "actions": list(task.costs),
                "outputs": list(task.bootstrap[0].mechanism_ir.parameters["outputs"]),
                "costs": task.costs,
                "initial_family": family,
                "repair_policy": POLICY,
                "protocol": protocol,
            }
            directory = (
                output / f"seed-{world_seed}" / case / f"boundary-{boundary}" / method
            )
            directory.mkdir(parents=True)
            dump(directory / "config.json", config)
            wall, cpu = perf_counter(), process_time()
            result = online_relearn(
                task.view,
                tuple(config["actions"]),
                tuple(config["outputs"]),
                task.costs,
                method=method,
                initial_family=family,
                repair_policy=POLICY,
                **protocol,
            )
            learning_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "run.json", result)
            current = {
                k: result[k]
                for k in ("old", "public_old_theory", "monitoring", "anomaly")
            }
            if prelude is not None and prelude != current:
                raise RuntimeError(
                    "Paid initial history/monitoring mismatch; retain partial artifacts"
                )
            prelude = current
            for filename, events in (
                (
                    "events.jsonl",
                    (
                        {
                            "epoch": g.get(
                                "epoch",
                                g.get("config", {}).get("protocol", {}).get("epoch"),
                            ),
                            "event": e,
                        }
                        for g in control_groups(result)
                        for e in local_events(g)
                    ),
                ),
                ("theories.jsonl", iter(result["theory_events"])),
            ):
                (directory / filename).write_text(
                    "".join(
                        json.dumps(e, sort_keys=True, allow_nan=False) + "\n"
                        for e in events
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
            private_path = directory / "evaluator" / "world.json"
            private_path.parent.mkdir()
            dump(private_path, task.private_spec)
            wall, cpu = perf_counter(), process_time()
            audit = audit_online(config, result, private_path)
            audit_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "audit.json", audit)
            if not audit["valid"]:
                raise RuntimeError(
                    f"Matched public-first audit failed: {directory}: {audit}; preserve partial evidence"
                )
            monitor_spent = sum(
                q["spent"] for b in result["monitoring"] for q in b["queries"]
            )
            row = {
                "seed": world_seed,
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
            controls = [e for g in control_groups(result) for e in g["raw_controls"]]
            row["reset_count"] = sum(e["kind"] == "reset" for e in controls)
            row["atomic_action_count"] = sum(e["kind"] == "step" for e in controls)
            row["known_failed_dispatch_count"] = sum(
                e["kind"] == "step"
                and (
                    not e["transition"]["intervention_compliance"]
                    or bool(e["transition"]["controller_failure"])
                    or not e["transition"]["event"]["success"]
                )
                for e in controls
            )
            rows.append(row)
            group[method] = row
            timings.append(
                {k: row[k] for k in ("seed", "case", "private_boundary", "method")}
                | {
                    "complete_learning_monitoring_and_validation": learning_time,
                    "complete_audit_and_private_score": audit_time,
                }
            )
        reference = group["from_scratch"]
        for method in MATCHED_METHODS[1:]:
            row = group[method]
            correct = all(
                r["evaluation"]["new_behavior_correct_at_H"] is True
                for r in (reference, row)
            )
            comparisons.append(
                {
                    "seed": world_seed,
                    "case": case,
                    "private_boundary": boundary,
                    "method": method,
                    "reference": "from_scratch",
                    "same_paid_prelude_and_monitoring": True,
                    "same_final_protocol": True,
                    "both_repaired_correct": correct,
                    "method_minus_scratch_total_cost": row["spent"]
                    - reference["spent"],
                    "cost_difference_is_success_conditioned": correct,
                }
            )
    if sources != hashes():
        raise RuntimeError(
            "Source drift; preserve partial data without certified manifest"
        )
    summary = {
        "worlds": len(grid),
        "campaigns": len(rows),
        "methods": {
            m: summarize([r for r in rows if r["method"] == m]) for m in MATCHED_METHODS
        },
        "by_boundary": {
            str(b): {
                m: summarize(
                    [r for r in rows if r["method"] == m and r["private_boundary"] == b]
                )
                for m in MATCHED_METHODS
            }
            for b in boundaries
        },
    }
    report = {
        "schema": "matched-h3-development-report-v1",
        "registration": registration,
        "scope": scope,
        "rows": rows,
        "comparisons": comparisons,
        "summary": summary,
        "all_public_first_audits_valid": True,
        "limitations": "all failed/wrong/unresolved/missed/late cases retained; finite validation not H certification; same-seed reruns not independent samples; no statistical superiority or innovation claim",
    }
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {
            "schema": "matched-h3-timing-v1",
            "rows": timings,
            "scope": "whole scientific computation and whole audit; excludes setup/serialization, nondeterministic",
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
            "schema": "matched-h3-manifest-v1",
            "sha256": manifest,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    print(
        json.dumps(
            {"summary": summary, "deterministic_artifacts": len(manifest)},
            sort_keys=True,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--old-budget", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--boundaries", type=int, nargs="+", default=[6, 25, 100])
    args = vars(parser.parse_args())
    args["boundaries"] = tuple(args["boundaries"])
    run(**args)
