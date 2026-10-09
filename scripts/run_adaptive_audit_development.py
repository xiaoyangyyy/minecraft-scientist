"""Stationary audited development grid, not a fair method or structural holdout.

The two-state cycle reference is a provided public prior; its learning cost is
not included. This must not be presented as paid online transfer or superiority.
All new-world discovery, resets, failed dispatch and validation are charged.
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

from audit_run import audit_adaptive
from learner import learn_adaptive

from minecraft_scientist.world.active_automata import (
    AUTOMATA_FAMILIES,
    automata_task,
)


def dump(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_hashes():
    files = list((ROOT / "src" / "minecraft_scientist").rglob("*.py"))
    files += list((ROOT / "research" / "adaptive_automata").glob("*.py"))
    files += [
        ROOT / "research" / "adaptive_automata" / "README.md",
        Path(__file__).resolve(),
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def run(*, output, seed=0, seeds=2, maximum_states=5, budget=240, horizon=5):
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 16
        or type(maximum_states) is not int
        or not 2 <= maximum_states <= 8
        or type(budget) is not int
        or budget < 1
        or type(horizon) is not int
        or not 1 <= horizon <= 8
    ):
        raise ValueError("invalid bounded development grid")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    protocol = {
        "budget": budget,
        "horizon": horizon,
        "epoch": "development-current",
        "seed": 20261008,
        "conformance_count": 8,
        "operation_budget": 200_000,
        "reference_operation_budget": 200_000,
    }
    scope = "stationary deterministic provided-prior development; no paid old-model learning, online revision, causal factor attribution, same-tester comparison, structural holdout or superiority"
    registration = {
        "schema": "adaptive-audit-development-registration-v1",
        "source_sha256": sources,
        "seeds": list(range(seed, seed + seeds)),
        "families": list(AUTOMATA_FAMILIES),
        "private_state_counts": list(range(2, maximum_states + 1)),
        "protocol": protocol,
        "public_prior": "two-state cycle model with matching opaque alphabets; not privately selected using the target family/state count",
        "scope": scope,
    }
    dump(output / "registration.json", registration)
    rows, timing = [], []
    for current_seed, family, states in itertools.product(
        range(seed, seed + seeds), AUTOMATA_FAMILIES, range(2, maximum_states + 1)
    ):
        directory = output / f"seed-{current_seed}" / family / f"states-{states}"
        directory.mkdir(parents=True)
        task, old = (
            automata_task(current_seed, family, states),
            automata_task(current_seed, "cycle", 2),
        )
        references = (old.ground_truth.to_ir(),)
        config = {
            "schema": "adaptive-config-v1",
            "actions": list(task.actions),
            "outputs": list(task.outputs),
            "costs": task.costs,
            "references": [r.model_dump(mode="json") for r in references],
            "protocol": protocol,
        }
        dump(directory / "config.json", config)
        wall, cpu = perf_counter(), process_time()
        result = learn_adaptive(
            task.view, task.actions, task.outputs, task.costs, references, **protocol
        )
        learning = {
            "wall_seconds": perf_counter() - wall,
            "cpu_seconds": process_time() - cpu,
        }
        dump(directory / "run.json", result)
        (directory / "events.jsonl").write_text(
            "".join(
                json.dumps(e, sort_keys=True, allow_nan=False) + "\n"
                for e in result["events"]
            ),
            encoding="utf-8",
            newline="\n",
        )
        private = directory / "evaluator" / "ground-truth.json"
        private.parent.mkdir()
        dump(private, task.ground_truth.model_dump(mode="json"))
        wall, cpu = perf_counter(), process_time()
        audit = audit_adaptive(config, result, private)
        audit_time = {
            "wall_seconds": perf_counter() - wall,
            "cpu_seconds": process_time() - cpu,
        }
        dump(directory / "audit.json", audit)
        if not audit["valid"]:
            raise RuntimeError(
                f"Independent adaptive audit failed: {directory}; retain partial artifacts"
            )
        paid = [e for e in result["events"] if e["kind"] in ("reset", "step")]
        conformance_ids = {
            e["query_id"]
            for e in result["events"]
            if e["kind"] == "plan" and e["reason"] == "conformance"
        }
        row = {
            "seed": current_seed,
            "family": family,
            "private_states": states,
            "status": result["status"],
            "stop_reason": result["stop_reason"],
            "spent": result["spent"],
            "paid_queries": result["paid_queries"],
            "paid_actions": sum(e["kind"] == "step" for e in paid),
            "conformance_cost": sum(
                e["cost"] for e in paid if e["query_id"] in conformance_ids
            ),
            "conformance_rejections": sum(
                not s["admitted"]
                for b in result["conformance"]
                for s in b["plan"]["samples"]
            ),
            "counterexample_revisions": sum(
                "counterexample" in r["rule"] for r in result["rules"]
            ),
            "observation_visits": result["observation_visits"],
            "reference_visits": result["reference_visits"],
            "sampler_visits": result["conformance_sampler_visits"],
            "evaluation": audit["evaluation"],
        }
        rows.append(row)
        timing.append(
            {
                "seed": current_seed,
                "family": family,
                "private_states": states,
                "learning_and_real_validation": learning,
                "independent_audit_and_private_scoring": audit_time,
            }
        )
    if sources != source_hashes():
        raise RuntimeError(
            "Source drift during development run; do not certify, retain artifacts"
        )
    report = {
        "schema": "adaptive-audit-development-report-v1",
        "registration": registration,
        "scope": scope,
        "rows": rows,
        "all_independent_audits_valid": True,
        "summary": {
            "runs": len(rows),
            "correct_provisional_at_H": sum(
                r["evaluation"]["provisional_behavior_correct_at_H"] is True
                for r in rows
            ),
            "incorrect_provisional_at_H": sum(
                r["evaluation"]["provisional_behavior_correct_at_H"] is False
                for r in rows
            ),
            "unresolved": sum(r["status"] == "unresolved" for r in rows),
        },
    }
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {
            "schema": "adaptive-audit-nondeterministic-timing-v1",
            "rows": timing,
            "scope": "complete learner discovery/real validation and audit wall/CPU; not all environment setup/serialization or prior learning costs",
        },
    )
    hashes = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    dump(
        output / "manifest.json",
        {
            "schema": "adaptive-audit-development-manifest-v1",
            "sha256": hashes,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    print(
        json.dumps(
            report["summary"]
            | {"deterministic_artifacts": len(hashes), "scope": scope},
            sort_keys=True,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--maximum-states", type=int, default=5)
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=5)
    run(**vars(parser.parse_args()))
