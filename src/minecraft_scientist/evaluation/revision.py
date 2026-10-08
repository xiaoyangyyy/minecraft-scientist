"""Public-first structural revision audit, with raw-table independent checks.

Development only: fixed edits, known resettable deterministic controllers. Private
truth is opened after full public replay and independent edit/compile/ledger checks.
Neither the auditor nor its private scores are available to the scientific Agent.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path

from minecraft_scientist.agent.revision import control_runs, revise_online
from minecraft_scientist.discovery.diagnosis import NO_DELIVERY, JointHypothesis
from minecraft_scientist.evaluation.diagnosis import (
    _factor_rollout,
    _factor_step,
    _matches_factors,
    _public_spec,
    _rows,
)
from minecraft_scientist.evaluation.identification import _rollout, _write_json
from minecraft_scientist.protocols import (
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.revision import REVISION_CASES, revision_task

REVISION_METHODS = ("witness_cover", "sequence_ig")
PROTOCOL_FIELDS = {
    "method",
    "seed",
    "budget",
    "horizon",
    "monitor_trials",
    "search_budget",
    "shadow_count",
    "candidate_limit",
}


def _raw_edit(parent, edits):
    """Interpret the declared edit journal, not the production generator."""
    program = copy.deepcopy(parent)
    for edit in edits:
        operation = edit["operation"]
        if operation == "add_state_from_template":
            if set(edit) != {"operation", "template", "state"}:
                raise ValueError("invalid state insertion fields")
            template, new = edit["template"], edit["state"]
            if template not in program["states"] or new in program["states"]:
                raise ValueError("invalid new state/template")
            rows = _rows(program)
            program["states"].append(new)
            for action in program["actions"]:
                row = copy.deepcopy(rows[template, action])
                row["source"] = new
                if row["target"] == template:
                    row["target"] = new
                program["transitions"].append(row)
        elif operation in ("redirect_transition", "change_output"):
            if set(edit) != {"operation", "source", "action", "from", "to"}:
                raise ValueError("invalid row edit fields")
            row = _rows(program)[edit["source"], edit["action"]]
            field = "target" if operation == "redirect_transition" else "output"
            if row[field] != edit["from"]:
                raise ValueError("edit parent value mismatch")
            row[field] = edit["to"]
        else:
            raise ValueError("unregistered edit operation")
    return program


def _bounded_matches_factors(program, spec, horizon):
    """Raw product traversal to H; no production interpreter or color refinement."""
    mechanism, controller = spec["mechanism"], spec["controller"]
    if set(program["actions"]) != set(controller["actions"]):
        return False
    prows, mrows, crows = _rows(program), _rows(mechanism), _rows(controller)
    pending = deque(
        [
            (
                program["initial_state"],
                mechanism["initial_state"],
                controller["initial_state"],
                0,
            )
        ]
    )
    seen = set()
    while pending:
        state, mstate, cstate, depth = pending.popleft()
        if (state, mstate, cstate, depth) in seen or depth == horizon:
            continue
        seen.add((state, mstate, cstate, depth))
        for action in sorted(program["actions"]):
            mnext, cnext, output = _factor_step(
                mechanism, mrows, crows, mstate, cstate, action
            )
            row = prows[state, action]
            if row["output"] != output:
                return False
            pending.append((row["target"], mnext, cnext, depth + 1))
    return True


def _audit_core(run, programs, costs, *, specs=None):
    """Independently check issued costs, frozen predictions and evidence filtering."""
    remaining, plans, spent = list(range(len(programs))), {}, 0

    def prediction(index, word):
        return (
            _factor_rollout(specs[index], word)
            if specs
            else _rollout(programs[index], word)
        )

    for event in run["events"]:
        if event["kind"] == "plan":
            word = tuple(event["actions"])
            plans[event["query_id"]] = word
            expected = {str(i): list(prediction(i, word)) for i in remaining}
            if event["predictions"] != expected or event["remaining"] != remaining:
                raise ValueError("independent frozen prediction mismatch")
            if event["total_cost"] != 1 + sum(costs[a] for a in word):
                raise ValueError("independent planned cost mismatch")
        elif event["kind"] == "reset":
            if event["cost"] != 1:
                raise ValueError("unregistered reset cost")
            spent += 1
        elif event["kind"] == "step":
            record = event["transition"]
            action = record["executed_action"]["target_id"]
            if event["cost"] != costs[action]:
                raise ValueError("independent action cost mismatch")
            spent += costs[action]
            if record["context"] != {
                "intervention_compliance_scope": "issued-command",
                "treatment_verification": "unknown",
            } or record["possible_confounders"] != ["unverified-actuator-delivery"]:
                raise ValueError("latent delivery falsely verified")
        elif event["kind"] == "evidence":
            remaining = [
                i
                for i in remaining
                if prediction(i, plans[event["query_id"]]) == tuple(event["outputs"])
            ]
            if event["remaining"] != remaining:
                raise ValueError("independent candidate filtering mismatch")
    if (
        spent != run["spent"]
        or spent > run["budget"]
        or remaining != run["remaining_candidates"]
    ):
        raise ValueError("independent core ledger mismatch")


def audit_revision(config, bootstrap, result, private_path: Path):
    public = {"public_replay_valid": False, "private_truth_opened": False}
    try:
        if (
            set(config) != {"schema", "protocol", "calibration", "costs"}
            or config["schema"] != "online-revision-config-v1"
        ):
            raise ValueError("unregistered online revision config")
        if set(config["protocol"]) != PROTOCOL_FIELDS:
            raise ValueError("unregistered protocol fields")
        joint = tuple(JointHypothesis.model_validate(item) for item in bootstrap)
        runs = list(control_runs(result))
        controls = [
            event
            for run in runs
            for event in run["events"]
            if event["kind"] in ("reset", "step")
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
                raise ValueError("public control/state mismatch")
            cursor += 1
            current = record.state_after
            return record

        view = AgentWorldView(
            observe=lambda: current, available=available, step=step, begin_trial=reset
        )
        replay = revise_online(
            view,
            joint,
            tuple(config["calibration"]),
            config["costs"],
            **config["protocol"],
        )
        if replay != result or cursor != len(controls):
            raise ValueError("public policy/budget/validation/version replay mismatch")
        specs = [_public_spec(h) for h in joint]
        programs = [h.compile().parameters for h in joint]
        if not all(_matches_factors(p, s) for p, s in zip(programs, specs)):
            raise ValueError("bootstrap compiler disagrees with raw factors")
        _audit_core(result["bootstrap"], programs, config["costs"], specs=specs)
        generation = result["generation"]
        generated_joint = None
        if generation is not None:
            parent = generation["parent_mechanism_ir"]
            for proposal in generation["proposals"]:
                candidate = proposal["ir"]
                if (
                    _raw_edit(parent["parameters"], proposal["edits"])
                    != candidate["parameters"]
                ):
                    raise ValueError("independent edit journal mismatch")
                for key in (
                    "scope",
                    "preconditions",
                    "composition",
                    "latent_variables",
                ):
                    if candidate[key] != parent[key]:
                        raise ValueError("structural proposal changed scope/context")
            controllers = {h.controller_id: h.controller_ir for h in joint}
            generated_joint = tuple(
                JointHypothesis(
                    mechanism_id=p["proposal_id"],
                    controller_id=cid,
                    mechanism_ir=p["ir"],
                    controller_ir=controller,
                )
                for p in generation["proposals"]
                for cid, controller in sorted(controllers.items())
            )
            generated_specs = [_public_spec(h) for h in generated_joint]
            generated_programs = [h.compile().parameters for h in generated_joint]
            if not all(
                _matches_factors(p, s)
                for p, s in zip(generated_programs, generated_specs)
            ):
                raise ValueError("generated compiler disagrees with raw factors")
            if "admitted_joint_indices" in generation:
                anomaly = result["monitoring"][-1]
                word = tuple(
                    next(e["actions"] for e in anomaly["events"] if e["kind"] == "plan")
                )
                outputs = tuple(
                    next(
                        e["outputs"]
                        for e in anomaly["events"]
                        if e["kind"] == "evidence"
                    )
                )
                admitted = [
                    i
                    for i, s in enumerate(generated_specs)
                    if _factor_rollout(s, word) == outputs
                ]
                if admitted != generation["admitted_joint_indices"]:
                    raise ValueError("independent anomaly filtering mismatch")
                if result["discovery"] is not None:
                    _audit_core(
                        result["discovery"],
                        [generated_programs[i] for i in admitted],
                        config["costs"],
                        specs=[generated_specs[i] for i in admitted],
                    )
        old_ir = next(
            (e["ir"] for e in result["theory_events"] if e["version"] == 1), None
        )
        for monitor in result["monitoring"]:
            _audit_core(monitor, [old_ir["parameters"]], config["costs"])
        used, query_ids = set(), set()
        shadow_runs = {}
        for name in ("bootstrap_validation", "replacement_validation"):
            batch = result[name]
            if batch is None:
                continue
            if batch["spent"] != sum(run["spent"] for run in batch["runs"]):
                raise ValueError("shadow ledger mismatch")
            for query, run in zip(batch["plan"]["queries"], batch["runs"]):
                word = tuple(query["actions"])
                if query["prediction"] != list(
                    _rollout(batch["plan"]["candidate_ir"]["parameters"], word)
                ):
                    raise ValueError("shadow batch prediction mismatch")
                plans = [e for e in run["events"] if e["kind"] == "plan"]
                if len(plans) != 1 or tuple(plans[0]["actions"]) != word:
                    raise ValueError("shadow dispatch differs from frozen batch")
                _audit_core(
                    run, [batch["plan"]["candidate_ir"]["parameters"]], config["costs"]
                )
                shadow_runs[id(run)] = word
        for run in runs:
            for event in run["events"]:
                if event["kind"] == "plan":
                    word, qid = tuple(event["actions"]), event["query_id"]
                    if qid in query_ids or (id(run) in shadow_runs and word in used):
                        raise ValueError("reused query ID or discovery/validation word")
                    query_ids.add(qid)
                    used.add(word)
        if (
            result["spent"] != sum(run["spent"] for run in runs)
            or result["spent"] > config["protocol"]["budget"]
        ):
            raise ValueError("online total ledger mismatch")
        public["public_replay_valid"] = True
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        AttributeError,
    ) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}

    public["private_truth_opened"] = True
    try:
        truth = json.loads(private_path.read_text(encoding="utf-8"))
        if (
            set(truth) != {"schema", "boundary_episode", "before", "after"}
            or truth["schema"] != "online-revision-private-v1"
        ):
            raise ValueError("unregistered private revision spec")
        boundary = truth["boundary_episode"]
        if type(boundary) is not int or not 6 <= boundary <= 100:
            raise ValueError("invalid private change boundary")
        for phase in ("before", "after"):
            spec = truth[phase]
            if set(spec) != {"mechanism", "controller"}:
                raise ValueError("unregistered private factor fields")
            mechanism = FiniteStateProgram.model_validate(spec["mechanism"])
            controller = FiniteStateProgram.model_validate(spec["controller"])
            if set(controller.actions) != set(config["costs"]) or any(
                o not in (*mechanism.actions, NO_DELIVERY) for o in controller.outputs
            ):
                raise ValueError("private controller contract mismatch")
        actual_valid, prefix, episode, tick = True, [], -1, 0
        for event in controls:
            if event["kind"] == "reset":
                prefix, tick, episode = [], 0, episode + 1
                spec = truth["after" if episode >= boundary else "before"]
                observation = event["observation"]
                actual_valid &= (
                    observation["episode_id"] == episode
                    and observation["step"] == 0
                    and observation["agent_state"]
                    == {"signal": spec["mechanism"]["outputs"][0]}
                )
            else:
                record = event["transition"]
                prefix.append(record["executed_action"]["target_id"])
                tick += 1
                signal = _factor_rollout(spec, tuple(prefix))[-1]
                actual_valid &= (
                    record["event"]["details"]["signal"] == signal
                    and record["state_after"]["agent_state"] == {"signal": signal}
                    and record["state_after"]["step"] == tick
                    and record["state_after"]["episode_id"] == episode
                )
        horizon = config["protocol"]["horizon"]
        accepted = [e for e in result["theory_events"] if e["status"] == "accepted"]
        bootstrap_correct = bool(accepted) and _bounded_matches_factors(
            accepted[0]["ir"]["parameters"], truth["before"], horizon
        )
        replacement = next((e for e in accepted if e["version"] == 2), None)
        replacement_correct = replacement is not None and _bounded_matches_factors(
            replacement["ir"]["parameters"], truth["after"], horizon
        )
        from minecraft_scientist.evaluation.identification import _exactly_equivalent

        mechanism_changed = not _exactly_equivalent(
            truth["before"]["mechanism"], truth["after"]["mechanism"]
        )
        anomaly = next(
            (r for r in result["monitoring"] if r["status"] == "model_mismatch"), None
        )
        detected_episode = next(
            (
                e["observation"]["episode_id"]
                for e in (anomaly or {}).get("events", ())
                if e["kind"] == "reset"
            ),
            None,
        )
        evaluation = {
            "actual_controls_match_private_reference": bool(actual_valid),
            "bootstrap_behavior_correct_at_declared_horizon": bootstrap_correct,
            "replacement_behavior_correct_at_declared_horizon": replacement_correct,
            "mechanism_behavior_changed": mechanism_changed,
            "false_mechanism_change": not mechanism_changed
            and result["mechanism_change_warranted"],
            "mechanism_attribution_correct": (
                mechanism_changed
                == (result["mechanism_attribution_status"] == "changed")
            )
            if result["mechanism_attribution_status"] in ("changed", "unchanged")
            else None,
            "detected_episode": detected_episode,
            "detection_delay_episodes": None
            if detected_episode is None
            else detected_episode - boundary,
            "post_anomaly_experiment_cost": 0
            if anomaly is None
            else sum(r["spent"] for r in runs[runs.index(anomaly) + 1 :]),
            "total_experiment_cost": result["spent"],
            "scope": "reset queries up to declared H; no unique causal structure or structural holdout claim",
        }
        return public | {"valid": bool(actual_valid), "evaluation": evaluation}
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}


def run_revision_benchmark(*, output: Path, seed=0, seeds=2, budget=240):
    """Freeze before acting, retain every negative case, refuse existing outputs."""
    if (
        type(seed) is not int
        or seed < 0
        or any(type(v) is not int or v < 1 for v in (seeds, budget))
    ):
        raise ValueError("invalid revision seeds/budget")
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
    protocol = {
        "seed": 20261008,
        "budget": budget,
        "horizon": 5,
        "monitor_trials": 10,
        "search_budget": 1000,
        "shadow_count": 3,
        "candidate_limit": 512,
    }
    registration = {
        "schema": "h3-development-registration-v1",
        "seed": seed,
        "seeds": seeds,
        "protocol": protocol,
        "methods": list(REVISION_METHODS),
        "cases": list(REVISION_CASES),
        "source_sha256": sources,
        "boundary_episode": 8,
        "grammar": [
            "single redirect",
            "single output change",
            "template state insertion plus redirect",
        ],
        "bootstrap_joint_count": 9,
        "generated_mechanism_count": 129,
        "generated_joint_count": 387,
        "prior": "uniform over reset behavior classes at H=5, not syntax",
        "split": "two fixed bootstrap templates selected by seed parity; no structural holdout",
        "limitations": [
            "deterministic resettable known controller family",
            "finite edit grammar, not unrestricted synthesis",
            "shadow query coverage is candidate-dependent, not an unbiased random test",
            "full inference compute is not measured by prediction-cache misses",
            "no calibrated posterior confidence, unique causal structure or novelty claim",
            "only matched witness and multi-step IG; adaptive learners and LLM matrix remain open",
            "no H4 transfer, Minecraft/RGB or complete-project acceptance",
        ],
    }
    _write_json(output / "registration.json", registration)
    rows = []
    for current_seed, case, method in itertools.product(
        range(seed, seed + seeds), REVISION_CASES, REVISION_METHODS
    ):
        directory = output / f"seed-{current_seed}" / case / method
        directory.mkdir(parents=True)
        task = revision_task(current_seed, case)
        config = {
            "schema": "online-revision-config-v1",
            "protocol": protocol | {"method": method},
            "calibration": list(task.calibration),
            "costs": task.costs,
        }
        bootstrap = [h.model_dump(mode="json") for h in task.bootstrap]
        _write_json(directory / "config.json", config)
        _write_json(directory / "bootstrap.json", bootstrap)
        result = revise_online(
            task.view,
            task.bootstrap,
            task.calibration,
            task.costs,
            **config["protocol"],
        )
        _write_json(directory / "run.json", result)
        (directory / "events.jsonl").write_text(
            "".join(
                json.dumps(
                    {"stage_run": i, "event": event}, sort_keys=True, allow_nan=False
                )
                + "\n"
                for i, run in enumerate(control_runs(result))
                for event in run["events"]
            ),
            encoding="utf-8",
            newline="\n",
        )
        (directory / "theories.jsonl").write_text(
            "".join(
                json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                for event in result["theory_events"]
            ),
            encoding="utf-8",
            newline="\n",
        )
        private = directory / "evaluator" / "ground-truth.json"
        private.parent.mkdir()
        _write_json(private, task.private_spec)
        audit = audit_revision(config, bootstrap, result, private)
        _write_json(directory / "audit.json", audit)
        if not audit["valid"]:
            raise RuntimeError(
                f"independent revision audit failed: {directory}; retain partial run"
            )
        rows.append(
            {
                "seed": current_seed,
                "case": case,
                "method": method,
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "spent": result["spent"],
                "mechanism_attribution_status": result["mechanism_attribution_status"],
                "bootstrap_validated": result[
                    "bootstrap_independent_validation_completed"
                ],
                "replacement_validated": result[
                    "replacement_independent_validation_completed"
                ],
                "search_evaluations": (result["generation"] or {}).get(
                    "evaluations", 0
                ),
                "evaluation": audit["evaluation"],
            }
        )
    if source_hashes() != sources:
        raise RuntimeError("source changed during H3 benchmark; retain partial run")
    report = {
        "schema": "h3-development-benchmark-v1",
        "registration": registration,
        "rows": rows,
        "all_independent_audits_valid": True,
        "summaries": [
            {
                "method": method,
                "runs": sum(row["method"] == method for row in rows),
                "validated_replacements": sum(
                    row["replacement_validated"]
                    for row in rows
                    if row["method"] == method
                ),
                "correct_replacements_at_H": sum(
                    row["evaluation"][
                        "replacement_behavior_correct_at_declared_horizon"
                    ]
                    for row in rows
                    if row["method"] == method
                ),
                "false_mechanism_changes": sum(
                    row["evaluation"]["false_mechanism_change"]
                    for row in rows
                    if row["method"] == method
                ),
                "unresolved_runs": sum(
                    row["status"] == "unresolved"
                    for row in rows
                    if row["method"] == method
                ),
            }
            for method in REVISION_METHODS
        ],
        "scope": "bounded online structural repair development, not complete H3 or novelty evidence",
    }
    _write_json(output / "report.json", report)
    artifacts = {
        path.relative_to(output).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write_json(
        output / "manifest.json",
        {"schema": "h3-artifact-manifest-v1", "sha256": artifacts},
    )
    return report
