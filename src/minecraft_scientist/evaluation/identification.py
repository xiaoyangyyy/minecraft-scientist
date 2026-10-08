"""Independent control replay and matched deterministic H1 development runs.

The evaluator reads private truth only AFTER a complete public agent replay.
Small fixed templates / randomized names are not structural heldouts, noisy
causal discovery, open-family synthesis or a method-novelty demonstration.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections import deque
from pathlib import Path

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.discovery.discrimination import METHODS
from minecraft_scientist.protocols import (
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TheoryIR,
    TransitionRecord,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import FAMILIES, identification_task


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _write_json(path: Path, value):
    # Fixed LF is part of the artifact contract, even on Windows.
    path.write_text(_json(value), encoding="utf-8", newline="\n")


def _rollout(program, word):
    """Independent raw-table reference: no production interpreter execute call."""
    state = program["initial_state"]
    rows = {(row["source"], row["action"]): row for row in program["transitions"]}
    outputs = []
    for action in word:
        row = rows[state, action]
        state = row["target"]
        outputs.append(row["output"])
    return tuple(outputs)


def _exactly_equivalent(left, right):
    lt = {(r["source"], r["action"]): r for r in left["transitions"]}
    rt = {(r["source"], r["action"]): r for r in right["transitions"]}
    pending, visited = deque([(left["initial_state"], right["initial_state"])]), set()
    while pending:
        pair = pending.popleft()
        if pair in visited:
            continue
        visited.add(pair)
        for action in sorted(left["actions"]):
            lr, rr = lt[pair[0], action], rt[pair[1], action]
            if lr["output"] != rr["output"]:
                return False
            pending.append((lr["target"], rr["target"]))
    return True


def replay_identification(config, candidates, result):
    """Public-only replay, shared by H1 and joint H2 audits; no private path."""
    public = {"public_replay_valid": False, "private_truth_opened": False}
    try:
        if set(config) != {
            "schema",
            "method",
            "seed",
            "budget",
            "horizon",
            "calibration",
            "costs",
        }:
            raise ValueError("unregistered public configuration fields")
        if config["schema"] != "known-candidate-h1-config-v1":
            raise ValueError("unregistered public configuration schema")
        irs = tuple(TheoryIR.model_validate(ir) for ir in candidates)
        controls = [
            event for event in result["events"] if event["kind"] in ("reset", "step")
        ]
        cursor, current = 0, None

        def reset():
            nonlocal cursor, current
            if cursor >= len(controls) or controls[cursor]["kind"] != "reset":
                raise ValueError("public reset chronology mismatch")
            current = Observation.model_validate(controls[cursor]["observation"])
            cursor += 1
            return current

        def available():
            return tuple(
                PrimitiveAction(kind=obj.affordances[0], target_id=obj.object_id)
                for obj in current.objects
            )

        def step(action):
            nonlocal cursor, current
            if cursor >= len(controls) or controls[cursor]["kind"] != "step":
                raise ValueError("public action chronology mismatch")
            record = TransitionRecord.model_validate(controls[cursor]["transition"])
            if record.executed_action != action or record.state_before != current:
                raise ValueError("actual public control/state mismatch")
            cursor += 1
            current = record.state_after
            return record

        view = AgentWorldView(
            observe=lambda: current, available=available, step=step, begin_trial=reset
        )
        replay = identify_mechanism(
            view,
            irs,
            tuple(config["calibration"]),
            config["costs"],
            method=config["method"],
            seed=config["seed"],
            budget=config["budget"],
            horizon=config["horizon"],
        )
        if replay != result or cursor != len(controls):
            raise ValueError("public plan/prediction/budget/theory replay mismatch")
        public["public_replay_valid"] = True
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}

    return public | {"valid": True}


def audit_identification(config, candidates, result, private_truth_path: Path):
    """Replay the complete policy first; private path is opened only afterwards."""
    public = replay_identification(config, candidates, result)
    if not public["valid"]:
        return public
    irs = tuple(TheoryIR.model_validate(ir) for ir in candidates)

    truth = FiniteStateProgram.model_validate_json(
        private_truth_path.read_text(encoding="utf-8")
    )
    public["private_truth_opened"] = True
    truth_data = truth.model_dump(mode="json")
    independent_controls = True
    episode, prefix, tick = -1, [], 0
    for event in result["events"]:
        if event["kind"] == "reset":
            prefix, tick = [], 0
            episode += 1
            observed = event["observation"]
            independent_controls &= (
                observed["episode_id"] == episode
                and observed["step"] == 0
                and observed["agent_state"] == {"signal": truth_data["outputs"][0]}
            )
        elif event["kind"] == "step":
            record = event["transition"]
            prefix.append(record["executed_action"]["target_id"])
            tick += 1
            expected = _rollout(truth_data, tuple(prefix))[-1]
            independent_controls &= (
                record["event"]["details"]["signal"] == expected
                and record["state_after"]["agent_state"] == {"signal": expected}
                and record["state_after"]["episode_id"] == episode
                and record["state_after"]["step"] == tick
            )
    programs = [FiniteStateProgram.from_ir(ir).model_dump(mode="json") for ir in irs]
    calibration_outputs = [
        _rollout(program, tuple(config["calibration"])) for program in programs
    ]
    words = [
        word
        for depth in range(1, config["horizon"] + 1)
        for word in itertools.product(sorted(truth.actions), repeat=depth)
    ]
    witnesses = []
    for left, right in itertools.combinations(range(len(programs)), 2):
        separating = [
            word
            for word in words
            if _rollout(programs[left], word) != _rollout(programs[right], word)
        ]
        minimum = min(
            (1 + sum(config["costs"][a] for a in word) for word in separating),
            default=None,
        )
        witnesses.append(
            {
                "pair": [left, right],
                "minimum_query_cost": minimum,
                "exactly_equivalent": _exactly_equivalent(
                    programs[left], programs[right]
                ),
            }
        )
    remaining = result["remaining_candidates"]
    # Conditional coverage, never score absence of a submitted theory as accuracy=1.
    prediction_accuracy = (
        None
        if not remaining or result["status"] != "conditional_behavior_identified"
        else sum(
            _rollout(programs[remaining[0]], word) == _rollout(truth_data, word)
            for word in words
        )
        / len(words)
    )
    identified_correctly = (
        result["status"] == "conditional_behavior_identified"
        and bool(remaining)
        and _exactly_equivalent(programs[remaining[0]], truth_data)
    )
    return public | {
        "valid": bool(independent_controls),
        "evaluation": {
            "actual_controls_match_private_reference": bool(independent_controls),
            "calibration_candidate_traces_agree": all(
                trace == calibration_outputs[0] for trace in calibration_outputs
            ),
            "pair_certificates": witnesses,
            "declared_query_count": len(words),
            "identified_behavior_correctly": bool(identified_correctly),
            "declared_universe_prediction_accuracy": prediction_accuracy,
            "unique_causal_structure_claimed": False,
            "scope": "known deterministic candidates, trusted capability boundary, fixed templates",
        },
    }


def run_identification_benchmark(*, output: Path, seed=0, seeds=4, budget=20):
    if any(type(value) is not int or value < 1 for value in (seeds, budget)) or (
        type(seed) is not int or seed < 0
    ):
        raise ValueError("invalid benchmark seeds/budget")
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    registration = {
        "schema": "h1-development-registration-v1",
        "seed": seed,
        "seeds": seeds,
        "budget": budget,
        "horizon": 5,
        "families": list(FAMILIES),
        "methods": list(METHODS),
        "agent_rng_seed": 20261008,
        "world_seed_passed_to_agent": False,
        "source_sha256": sources,
        "prior": "uniform over exact behavior classes",
        "split": "development: fixed templates with randomized opaque labels",
        "claims": {
            "structure_synthesis": False,
            "structural_holdout": False,
            "noise_or_fault_diagnosis": False,
            "method_novelty": False,
            "shadow_validation": False,
            "minecraft_or_rgb": False,
        },
    }
    _write_json(output / "registration.json", registration)
    rows = []
    for current_seed in range(seed, seed + seeds):
        for family in FAMILIES:
            for method in METHODS:
                directory = output / f"seed-{current_seed}" / family / method
                directory.mkdir(parents=True)
                task = identification_task(current_seed, family)
                config = {
                    "schema": "known-candidate-h1-config-v1",
                    "method": method,
                    "seed": 20261008,
                    "budget": budget,
                    "horizon": 5,
                    "calibration": list(task.calibration),
                    "costs": task.costs,
                }
                candidates = [ir.model_dump(mode="json") for ir in task.candidates]
                for name, payload in (
                    ("config.json", config),
                    ("candidates.json", candidates),
                ):
                    _write_json(directory / name, payload)
                result = identify_mechanism(
                    task.view,
                    task.candidates,
                    task.calibration,
                    task.costs,
                    method=method,
                    seed=config["seed"],
                    budget=budget,
                )
                events = result["events"]
                (directory / "events.jsonl").write_text(
                    "".join(
                        json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                        for event in events
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                _write_json(directory / "run.json", result)
                truth_path = directory / "evaluator" / "ground-truth.json"
                truth_path.parent.mkdir()
                _write_json(truth_path, task.ground_truth.model_dump(mode="json"))
                audit = audit_identification(config, candidates, result, truth_path)
                _write_json(directory / "audit.json", audit)
                if not audit["valid"]:
                    raise RuntimeError(f"independent H1 audit failed: {directory}")
                rows.append(
                    {
                        "seed": current_seed,
                        "family": family,
                        "method": method,
                        "status": result["status"],
                        "spent": result["spent"],
                        "completed_queries": result["completed_queries"],
                        "prediction_cache_misses": result["prediction_cache_misses"],
                        "evaluation": audit["evaluation"],
                    }
                )
    current = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    if current != sources:
        raise RuntimeError("source changed during benchmark; retain partial run")
    summaries = []
    for family, method in itertools.product(FAMILIES, METHODS):
        samples = [
            row for row in rows if row["family"] == family and row["method"] == method
        ]
        summaries.append(
            {
                "family": family,
                "method": method,
                "n": len(samples),
                "mean_total_cost": statistics.mean(row["spent"] for row in samples),
                "identified_correctly": sum(
                    row["evaluation"]["identified_behavior_correctly"]
                    for row in samples
                ),
                "status_counts": {
                    status: sum(row["status"] == status for row in samples)
                    for status in sorted({row["status"] for row in samples})
                },
            }
        )
    report = {
        "schema": "h1-development-benchmark-v1",
        "registration": registration,
        "rows": rows,
        "summaries": summaries,
        "all_independent_audits_valid": True,
        "scope": "development protocol check; not structural generalization or novel-method evidence",
    }
    _write_json(output / "report.json", report)
    files = {
        path.relative_to(output).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write_json(output / "manifest.json", files)
    return report
