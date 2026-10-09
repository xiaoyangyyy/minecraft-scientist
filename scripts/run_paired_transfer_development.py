"""Paired explicit-context transfer: same paid old evidence, budget and tester.

This is synthetic development, not structural holdouts, unannounced online
mutation, causal-factor comparison or symbol abstraction evidence.
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

from audit_transfer import audit_transfer
from paid_transfer import METHODS, transfer_contexts

from minecraft_scientist.world.active_automata import (
    AUTOMATA_FAMILIES,
    automata_task,
)


def dump(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def hashes():
    paths = list((ROOT / "src" / "minecraft_scientist").rglob("*.py"))
    paths += list((ROOT / "research" / "adaptive_automata").glob("*.py"))
    paths += [
        ROOT / "research" / "adaptive_automata" / "README.md",
        Path(__file__).resolve(),
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def run(
    *, output, seed=0, seeds=2, maximum_states=5, budget=360, old_budget=120, horizon=5
):
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 16
        or type(maximum_states) is not int
        or not 2 <= maximum_states <= 8
        or any(type(v) is not int or v < 1 for v in (budget, old_budget))
        or type(horizon) is not int
        or not 1 <= horizon <= 8
    ):
        raise ValueError("invalid paired-transfer development protocol")
    output.mkdir(parents=True, exist_ok=False)
    sources = hashes()
    protocol = {
        "budget": budget,
        "old_budget": old_budget,
        "horizon": horizon,
        "seed": 20261008,
        "conformance_count": 8,
        "operation_budget": 200_000,
        "reference_operation_budget": 200_000,
    }
    scope = "paired paid explicit-context development, stable anonymous alphabets; not structural holdout, hidden online mutation, calibrated confidence, causal-factor attribution or symbol-library transfer"
    registration = {
        "schema": "paired-paid-transfer-development-registration-v1",
        "source_sha256": sources,
        "seeds": list(range(seed, seed + seeds)),
        "families": list(AUTOMATA_FAMILIES),
        "private_new_state_counts": list(range(2, maximum_states + 1)),
        "methods": list(METHODS),
        "old_context": "two-state cycle, never supplied as ground-truth model to the learner",
        "protocol": protocol,
        "scope": scope,
    }
    dump(output / "registration.json", registration)
    rows, timings, pairs = [], [], []
    for world_seed, family, states in itertools.product(
        range(seed, seed + seeds), AUTOMATA_FAMILIES, range(2, maximum_states + 1)
    ):
        previous_old, previous_theory, paired = None, None, {}
        for method in METHODS:
            directory = (
                output / f"seed-{world_seed}" / family / f"states-{states}" / method
            )
            directory.mkdir(parents=True)
            old, new = (
                automata_task(world_seed, "cycle", 2),
                automata_task(world_seed, family, states),
            )
            config = {
                "schema": "paid-transfer-config-v1",
                "method": method,
                "actions": list(old.actions),
                "outputs": list(old.outputs),
                "costs": old.costs,
                "protocol": protocol,
            }
            dump(directory / "config.json", config)
            wall, cpu = perf_counter(), process_time()
            result = transfer_contexts(
                old.view,
                new.view,
                old.actions,
                old.outputs,
                old.costs,
                method=method,
                **protocol,
            )
            learning_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "run.json", result)
            if previous_old is not None and (
                previous_old != result["old"]
                or previous_theory != result["public_old_theory"]
            ):
                raise RuntimeError(
                    "Methods received different paid old history; retain partial run"
                )
            previous_old, previous_theory = result["old"], result["public_old_theory"]
            events = (
                {"context": p + "-context", "event": e}
                for p in ("old", "new")
                if result[p] is not None
                for e in result[p]["run"]["events"]
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
            private = directory / "evaluator"
            private.mkdir()
            old_path, new_path = private / "old.json", private / "new.json"
            dump(old_path, old.ground_truth.model_dump(mode="json"))
            dump(new_path, new.ground_truth.model_dump(mode="json"))
            wall, cpu = perf_counter(), process_time()
            audit = audit_transfer(config, result, old_path, new_path)
            audit_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "audit.json", audit)
            if not audit["valid"]:
                raise RuntimeError(
                    f"Whole-campaign audit failed: {directory}; retain partial evidence"
                )
            row = {
                "seed": world_seed,
                "family": family,
                "private_new_states": states,
                "method": method,
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "spent": result["spent"],
                "old_spent": result["old"]["run"]["spent"],
                "new_spent": 0
                if result["new"] is None
                else result["new"]["run"]["spent"],
                "evaluation": audit["evaluation"],
            }
            rows.append(row)
            paired[method] = row
            timings.append(
                {
                    "seed": world_seed,
                    "family": family,
                    "private_new_states": states,
                    "method": method,
                    "both_context_learning_and_validation": learning_time,
                    "complete_audit_and_scoring": audit_time,
                }
            )
        left, right = paired["from_scratch"], paired["adaptive"]
        both = (
            left["evaluation"]["both_contexts_correct_at_H"] is True
            and right["evaluation"]["both_contexts_correct_at_H"] is True
        )
        pairs.append(
            {
                "seed": world_seed,
                "family": family,
                "private_new_states": states,
                "same_paid_old_history": True,
                "both_methods_both_contexts_correct": both,
                "adaptive_minus_scratch_total_cost": right["spent"] - left["spent"],
                "cost_difference_is_success_conditioned": both,
                "from_scratch_outcome": left["evaluation"][
                    "both_contexts_correct_at_H"
                ],
                "adaptive_outcome": right["evaluation"]["both_contexts_correct_at_H"],
            }
        )
    if sources != hashes():
        raise RuntimeError(
            "Source drift; retain partial artifacts without certified manifest"
        )
    per_method = {
        m: {
            "runs": sum(r["method"] == m for r in rows),
            "both_contexts_correct": sum(
                r["method"] == m
                and r["evaluation"]["both_contexts_correct_at_H"] is True
                for r in rows
            ),
            "incorrect_provisional_chain": sum(
                r["method"] == m
                and r["evaluation"]["both_contexts_correct_at_H"] is False
                for r in rows
            ),
            "unresolved": sum(
                r["method"] == m
                and r["evaluation"]["both_contexts_correct_at_H"] is None
                for r in rows
            ),
        }
        for m in METHODS
    }
    summary = {
        "pairs": len(pairs),
        "methods": per_method,
        "both_methods_correct_pairs": sum(
            p["both_methods_both_contexts_correct"] for p in pairs
        ),
        "adaptive_cheaper_among_both_correct": sum(
            p["both_methods_both_contexts_correct"]
            and p["adaptive_minus_scratch_total_cost"] < 0
            for p in pairs
        ),
        "scratch_cheaper_among_both_correct": sum(
            p["both_methods_both_contexts_correct"]
            and p["adaptive_minus_scratch_total_cost"] > 0
            for p in pairs
        ),
        "tied_cost_among_both_correct": sum(
            p["both_methods_both_contexts_correct"]
            and p["adaptive_minus_scratch_total_cost"] == 0
            for p in pairs
        ),
    }
    report = {
        "schema": "paired-paid-transfer-development-report-v1",
        "registration": registration,
        "scope": scope,
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "all_campaign_audits_valid": True,
        "limitations": "Cost comparisons success-conditioned with failures/unresolved retained separately; 2 development seeds, no significance or innovation claim",
    }
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {
            "schema": "paired-transfer-timing-v1",
            "rows": timings,
            "scope": "both contexts learning/actual validation/registry and complete audit separately; excludes setup/serialization; not total inference bounded by visit counters",
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
            "schema": "paired-transfer-manifest-v1",
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
    parser.add_argument("--maximum-states", type=int, default=5)
    parser.add_argument("--budget", type=int, default=360)
    parser.add_argument("--old-budget", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=5)
    run(**vars(parser.parse_args()))
