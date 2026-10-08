"""Public-first joint diagnosis audit and independent raw-factor execution.

Development fixture only: known deterministic families, resettable hidden control,
optional alternate actuation. No stochastic diagnosis or structure synthesis claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections import deque
from pathlib import Path

from minecraft_scientist.agent.diagnosis import diagnose_mechanism
from minecraft_scientist.discovery.diagnosis import (
    NO_DELIVERY,
    JointHypothesis,
    project_diagnosis,
    validate_hypotheses,
)
from minecraft_scientist.evaluation.identification import (
    _write_json,
    replay_identification,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.diagnosis import CONTROLLERS, MECHANISMS, diagnosis_task

DIAGNOSIS_METHODS = ("witness_cover", "sequence_ig")


def _rows(program):
    return {(row["source"], row["action"]): row for row in program["transitions"]}


def _factor_step(mechanism, mrows, crows, mstate, cstate, command):
    control = crows[cstate, command]
    if control["output"] == NO_DELIVERY:
        return mstate, control["target"], mechanism["outputs"][0]
    row = mrows[mstate, control["output"]]
    return row["target"], control["target"], row["output"]


def _factor_rollout(spec, word):
    mechanism, controller = spec["mechanism"], spec["controller"]
    mrows, crows = _rows(mechanism), _rows(controller)
    mstate, cstate = mechanism["initial_state"], controller["initial_state"]
    outputs = []
    for command in word:
        mstate, cstate, output = _factor_step(
            mechanism, mrows, crows, mstate, cstate, command
        )
        outputs.append(output)
    return tuple(outputs)


def _matches_factors(program, spec):
    """Exact reachable-state comparison; does not compile or execute either IR."""
    mechanism, controller = spec["mechanism"], spec["controller"]
    if set(program["actions"]) != set(controller["actions"]):
        return False
    mrows, crows, prows = _rows(mechanism), _rows(controller), _rows(program)
    pending = deque(
        [
            (
                program["initial_state"],
                mechanism["initial_state"],
                controller["initial_state"],
            )
        ]
    )
    visited = set()
    while pending:
        pstate, mstate, cstate = pending.popleft()
        if (pstate, mstate, cstate) in visited:
            continue
        visited.add((pstate, mstate, cstate))
        for command in sorted(controller["actions"]):
            next_m, next_c, signal = _factor_step(
                mechanism, mrows, crows, mstate, cstate, command
            )
            prediction = prows[pstate, command]
            if prediction["output"] != signal:
                return False
            pending.append((prediction["target"], next_m, next_c))
    return True


def _public_spec(hypothesis):
    return {
        "mechanism": FiniteStateProgram.from_ir(hypothesis.mechanism_ir).model_dump(
            mode="json"
        ),
        "controller": FiniteStateProgram.from_ir(hypothesis.controller_ir).model_dump(
            mode="json"
        ),
    }


def audit_diagnosis(config, hypotheses, result, private_path: Path):
    """Reject public policy, compilation or attribution tampering before private I/O."""
    public = {"public_replay_valid": False, "private_truth_opened": False}
    try:
        if (
            set(config)
            != {"schema", "identification", "baseline_mechanism", "baseline_controller"}
            or config["schema"] != "joint-diagnosis-config-v1"
        ):
            raise ValueError("unregistered joint diagnosis configuration")
        if set(result) != {"schema", "identification", "diagnosis"} or (
            result["schema"] != "joint-diagnosis-run-v1"
        ):
            raise ValueError("unregistered joint diagnosis result")
        joint = tuple(JointHypothesis.model_validate(item) for item in hypotheses)
        candidates = validate_hypotheses(
            joint, config["baseline_mechanism"], config["baseline_controller"]
        )
        core = result["identification"]
        replay = replay_identification(
            config["identification"],
            [ir.model_dump(mode="json") for ir in candidates],
            core,
        )
        if not replay["valid"]:
            raise ValueError(replay["failure"])
        if result["diagnosis"] != project_diagnosis(
            joint, core, config["baseline_mechanism"], config["baseline_controller"]
        ):
            raise ValueError("factor attribution projection mismatch")
        specs = [_public_spec(hypothesis) for hypothesis in joint]
        programs = [
            FiniteStateProgram.from_ir(ir).model_dump(mode="json") for ir in candidates
        ]
        if not all(
            _matches_factors(program, spec) for program, spec in zip(programs, specs)
        ):
            raise ValueError(
                "compiled candidate disagrees with independent factor runtime"
            )
        # Independent evidence filtering: no shared execute/observe implementation.
        remaining, plans = list(range(len(joint))), {}
        for event in core["events"]:
            if event["kind"] == "plan":
                plans[event["query_id"]] = tuple(event["actions"])
            elif event["kind"] == "step":
                record = event["transition"]
                if record["context"] != {
                    "intervention_compliance_scope": "issued-command",
                    "treatment_verification": "unknown",
                } or record["possible_confounders"] != ["unverified-actuator-delivery"]:
                    raise ValueError(
                        "latent delivery falsely presented as verified treatment"
                    )
            elif event["kind"] == "evidence":
                remaining = [
                    index
                    for index in remaining
                    if _factor_rollout(specs[index], plans[event["query_id"]])
                    == tuple(event["outputs"])
                ]
                if remaining != event["remaining"]:
                    raise ValueError(
                        "independent joint-factor evidence filtering mismatch"
                    )
        if remaining != core["remaining_candidates"]:
            raise ValueError("independent final candidate mismatch")
        public["public_replay_valid"] = True
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}

    public["private_truth_opened"] = True
    try:
        truth = json.loads(private_path.read_text(encoding="utf-8"))
        if set(truth) != {
            "schema",
            "mechanism_id",
            "controller_id",
            "mechanism",
            "controller",
        } or (truth["schema"] != "joint-diagnosis-private-v1"):
            raise ValueError("unregistered private factor specification")
        mechanism = FiniteStateProgram.model_validate(truth["mechanism"])
        controller = FiniteStateProgram.model_validate(truth["controller"])
        if any(
            output not in (*mechanism.actions, NO_DELIVERY)
            for output in controller.outputs
        ):
            raise ValueError("private controller has invalid delivery labels")
        actual_valid, prefix, episode, tick = True, [], -1, 0
        for event in core["events"]:
            if event["kind"] == "reset":
                prefix, tick, episode = [], 0, episode + 1
                observed = event["observation"]
                actual_valid &= (
                    observed["episode_id"] == episode
                    and observed["step"] == 0
                    and observed["agent_state"] == {"signal": mechanism.outputs[0]}
                )
            elif event["kind"] == "step":
                record = event["transition"]
                prefix.append(record["executed_action"]["target_id"])
                tick += 1
                signal = _factor_rollout(truth, tuple(prefix))[-1]
                actual_valid &= (
                    record["event"]["details"]["signal"] == signal
                    and record["state_after"]["agent_state"] == {"signal": signal}
                    and record["state_after"]["step"] == tick
                    and record["state_after"]["episode_id"] == episode
                )
        # IDs may only score attribution when the private factor matches its public definition.
        truth_pair = [
            i
            for i, hypothesis in enumerate(joint)
            if hypothesis.mechanism_id == truth["mechanism_id"]
            and hypothesis.controller_id == truth["controller_id"]
            and specs[i]
            == {"mechanism": truth["mechanism"], "controller": truth["controller"]}
        ]
        covered = bool(truth_pair)
        diagnosis = result["diagnosis"]

        def factor_correct(name, baseline):
            status = diagnosis[f"{name}_status"]
            if not covered or status not in ("changed", "unchanged"):
                return None
            return (truth[f"{name}_id"] != baseline) == (status == "changed")

        evaluation = {
            "actual_controls_match_private_reference": bool(actual_valid),
            "private_factor_pair_in_declared_family": covered,
            "true_factor_pair_retained": covered and truth_pair[0] in remaining,
            "identified_behavior_correctly": (
                core["status"] == "conditional_behavior_identified"
                and bool(remaining)
                and all(_matches_factors(programs[i], truth) for i in remaining)
            ),
            "mechanism_change_correctly_resolved": factor_correct(
                "mechanism", config["baseline_mechanism"]
            ),
            "controller_change_correctly_resolved": factor_correct(
                "controller", config["baseline_controller"]
            ),
            "false_mechanism_change": (
                covered
                and truth["mechanism_id"] == config["baseline_mechanism"]
                and diagnosis["mechanism_revision_warranted"]
            ),
            "scope": "known deterministic joint models; no noisy or open-family diagnosis",
        }
        return public | {"valid": bool(actual_valid), "evaluation": evaluation}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}


def run_diagnosis_benchmark(*, output: Path, seed=0, seeds=2, budget=32):
    if (
        type(seed) is not int
        or seed < 0
        or any(type(value) is not int or value < 1 for value in (seeds, budget))
    ):
        raise ValueError("invalid diagnosis seeds/budget")
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]

    def source_hashes():
        return {
            path.relative_to(package).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }

    sources = source_hashes()
    registration = {
        "schema": "h2-development-registration-v1",
        "seed": seed,
        "seeds": seeds,
        "budget": budget,
        "horizon": 5,
        "reset_cost": 1,
        "ordinary_action_cost": 1,
        "alternate_actuation_cost": 3,
        "calibration_cost": 5,
        "agent_rng_seed": 20261008,
        "mechanisms": list(MECHANISMS),
        "controllers": list(CONTROLLERS),
        "alternate_actuation_conditions": [False, True],
        "methods": list(DIAGNOSIS_METHODS),
        "source_sha256": sources,
        "prior": "uniform over exact joint behavior classes, not syntactic factors",
        "split": "fixed development families, randomized labels; not structural holdouts",
        "limitations": [
            "known deterministic controller family",
            "resets include hidden controller state",
            "baseline is supplied public reference knowledge",
            "alternate route is a declared affordance",
            "no structure generation, shadow promotion, stochastic inference or Minecraft",
        ],
    }
    _write_json(output / "registration.json", registration)
    rows = []
    for current_seed, mechanism, controller, bypass, method in itertools.product(
        range(seed, seed + seeds),
        MECHANISMS,
        CONTROLLERS,
        (False, True),
        DIAGNOSIS_METHODS,
    ):
        directory = (
            output
            / f"seed-{current_seed}"
            / mechanism
            / controller
            / f"alternate-{int(bypass)}"
            / method
        )
        directory.mkdir(parents=True)
        task = diagnosis_task(current_seed, mechanism, controller, bypass=bypass)
        config = {
            "schema": "joint-diagnosis-config-v1",
            "baseline_mechanism": task.baseline_mechanism,
            "baseline_controller": task.baseline_controller,
            "identification": {
                "schema": "known-candidate-h1-config-v1",
                "method": method,
                "seed": 20261008,
                "budget": budget,
                "horizon": 5,
                "calibration": list(task.calibration),
                "costs": task.costs,
            },
        }
        hypotheses = [
            hypothesis.model_dump(mode="json") for hypothesis in task.hypotheses
        ]
        _write_json(directory / "config.json", config)
        _write_json(directory / "hypotheses.json", hypotheses)
        result = diagnose_mechanism(
            task.view,
            task.hypotheses,
            task.calibration,
            task.costs,
            baseline_mechanism=task.baseline_mechanism,
            baseline_controller=task.baseline_controller,
            method=method,
            seed=20261008,
            budget=budget,
        )
        _write_json(directory / "run.json", result)
        (directory / "events.jsonl").write_text(
            "".join(
                json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                for event in result["identification"]["events"]
            ),
            encoding="utf-8",
            newline="\n",
        )
        private = directory / "evaluator" / "ground-truth.json"
        private.parent.mkdir()
        _write_json(private, task.private_spec)
        audit = audit_diagnosis(config, hypotheses, result, private)
        _write_json(directory / "audit.json", audit)
        if not audit["valid"]:
            raise RuntimeError(f"independent joint diagnosis audit failed: {directory}")
        rows.append(
            {
                "seed": current_seed,
                "mechanism": mechanism,
                "controller": controller,
                "alternate_actuation": bypass,
                "method": method,
                "status": result["identification"]["status"],
                "spent": result["identification"]["spent"],
                "completed_queries": result["identification"]["completed_queries"],
                "prediction_cache_misses": result["identification"][
                    "prediction_cache_misses"
                ],
                "diagnosis": result["diagnosis"],
                "evaluation": audit["evaluation"],
            }
        )
    if source_hashes() != sources:
        raise RuntimeError("source changed during H2 benchmark; retain partial run")
    summaries = []
    for bypass, method in itertools.product((False, True), DIAGNOSIS_METHODS):
        samples = [
            row
            for row in rows
            if row["alternate_actuation"] == bypass and row["method"] == method
        ]
        summaries.append(
            {
                "alternate_actuation": bypass,
                "method": method,
                "n": len(samples),
                "mean_total_cost": statistics.mean(row["spent"] for row in samples),
                "behavior_identified_correctly": sum(
                    row["evaluation"]["identified_behavior_correctly"]
                    for row in samples
                ),
                "false_mechanism_changes": sum(
                    row["evaluation"]["false_mechanism_change"] for row in samples
                ),
                "mechanism_resolved_correctly": sum(
                    row["evaluation"]["mechanism_change_correctly_resolved"] is True
                    for row in samples
                ),
                "mechanism_resolved_incorrectly": sum(
                    row["evaluation"]["mechanism_change_correctly_resolved"] is False
                    for row in samples
                ),
                "controller_resolved_correctly": sum(
                    row["evaluation"]["controller_change_correctly_resolved"] is True
                    for row in samples
                ),
                "controller_resolved_incorrectly": sum(
                    row["evaluation"]["controller_change_correctly_resolved"] is False
                    for row in samples
                ),
                "mechanism_attribution_ambiguous": sum(
                    row["diagnosis"]["mechanism_status"] == "ambiguous"
                    for row in samples
                ),
                "mechanism_attribution_unresolved": sum(
                    row["diagnosis"]["mechanism_status"] == "unresolved"
                    for row in samples
                ),
            }
        )
    report = {
        "schema": "h2-development-benchmark-v1",
        "registration": registration,
        "rows": rows,
        "summaries": summaries,
        "all_independent_audits_valid": True,
        "scope": "known-family hidden-control development; not novelty or complete H2",
    }
    _write_json(output / "report.json", report)
    _write_json(
        output / "manifest.json",
        {
            path.relative_to(output).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    return report
