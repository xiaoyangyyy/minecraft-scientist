"""Export declared unit-fixture traces, not a scientific benchmark or fair baseline.

Public reference programs are fixture priors, not paid old-world discoveries.
The export checks raw reference/sampler plans, not full policy/control/private-I/O
auditing. Keeping that boundary explicit is essential for interpreting results.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "research" / "adaptive_automata"
sys.path.insert(0, str(PROTOTYPE))

from audit_random_wp import check_random_wp_plan
from learner import learn_adaptive
from test_audit_matching import replay_plan_checks
from test_matching import counter, view

from minecraft_scientist.evaluation.identification import (
    _exactly_equivalent,
)


def dump(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_hashes():
    files = list(PROTOTYPE.glob("*.py")) + [
        PROTOTYPE / "README.md",
        Path(__file__).resolve(),
    ]
    files += list((ROOT / "src" / "minecraft_scientist").rglob("*.py"))
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    registration = {
        "schema": "adaptive-unit-fixture-export-v1",
        "source_sha256": sources,
        "scope": "8 declared counter/initial-shift unit fixtures; public reference priors supplied without learning cost; not structural holdouts, fair method comparison or full-run audit",
        "protocol": {
            "budget": 3000,
            "horizon": 8,
            "seed": 20261008,
            "conformance_count": 8,
        },
    }
    dump(output / "registration.json", registration)
    rows = []
    for states in range(1, 5):
        for shift in (False, True):
            directory = output / f"counter-{states}-shift-{int(shift)}"
            directory.mkdir()
            truth, reference = (
                counter(states, initial=int(shift and states > 1)),
                counter(states),
            )
            references, costs = (reference.to_ir(),), {a: 1 for a in truth.actions}
            public_references = [r.model_dump(mode="json") for r in references]
            result = learn_adaptive(
                view(truth),
                truth.actions,
                truth.outputs,
                costs,
                references,
                **registration["protocol"],
            )
            dump(
                directory / "config.json",
                {
                    "actions": list(truth.actions),
                    "outputs": list(truth.outputs),
                    "costs": costs,
                    "references": public_references,
                    "protocol": registration["protocol"],
                },
            )
            dump(directory / "run.json", result)
            count = replay_plan_checks(result, public_references, truth.actions, costs)
            for batch in result["conformance"]:
                check_random_wp_plan(batch["plan"], costs)
            correct = (
                None
                if result["candidate_ir"] is None
                else _exactly_equivalent(
                    result["candidate_ir"]["parameters"], truth.model_dump(mode="json")
                )
            )
            checks = {
                "raw_reference_plans_checked": count,
                "raw_sampler_plans_checked": len(result["conformance"]),
                "scope": "public plan checks only, no complete policy, control chronology or private-I/O audit",
                "fixture_evaluator_behavior_correct": correct,
            }
            dump(directory / "plan-checks.json", checks)
            rows.append(
                {
                    "fixture": directory.name,
                    "status": result["status"],
                    "stop_reason": result["stop_reason"],
                    "spent": result["spent"],
                    "paid_queries": result["paid_queries"],
                    **checks,
                }
            )
    if sources != source_hashes():
        raise RuntimeError("source drift; keep partial artifacts, do not certify")
    report = {
        "schema": "adaptive-unit-fixture-export-report-v1",
        "scope": registration["scope"],
        "rows": rows,
    }
    dump(output / "report.json", report)
    hashes = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    dump(
        output / "manifest.json",
        {"schema": "adaptive-unit-fixture-manifest-v1", "sha256": hashes},
    )
    print(
        json.dumps(
            {
                "fixtures": len(rows),
                "artifacts": len(hashes),
                "correct": sum(
                    r["fixture_evaluator_behavior_correct"] is True for r in rows
                ),
                "scope": registration["scope"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
