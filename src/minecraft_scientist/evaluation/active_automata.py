"""Public-only policy/fold audit before independent raw private behavior scoring."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from pathlib import Path
from time import perf_counter, process_time

from minecraft_scientist.discovery.active_automata import learn_apartness
from minecraft_scientist.evaluation.identification import _rollout, _write_json
from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TheoryIR,
    TransitionRecord,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.active_automata import AUTOMATA_FAMILIES, automata_task


def _known_witness(nodes, left, right, actions):
    pending = deque([(left, right, ())])
    while pending:
        lp, rp, word = pending.popleft()
        for action in sorted(actions):
            lc, rc = lp + (action,), rp + (action,)
            if lc not in nodes or rc not in nodes:
                continue
            if nodes[lc] != nodes[rc]:
                return word + (action,)
            pending.append((lc, rc, word + (action,)))
    return None


def _check_raw_fold(program, basis, nodes, actions):
    """Independent table construction from acquired prefixes, not fold_hypothesis."""
    rows = {(r["source"], r["action"]): r for r in program["transitions"]}
    labels = [f"basis-{i}" for i in range(len(basis))]
    if program["states"] != labels or program["initial_state"] != labels[0]:
        raise ValueError("independent hypothesis state/basis mismatch")
    for index, prefix in enumerate(basis):
        if prefix not in nodes or any(
            prefix[:i] not in basis for i in range(len(prefix))
        ):
            raise ValueError("independent basis subtree mismatch")
        if any(
            _known_witness(nodes, prefix, other, actions) is None
            for other in basis[:index]
        ):
            raise ValueError("independent basis apartness mismatch")
        for action in actions:
            child = prefix + (action,)
            if child not in nodes:
                raise ValueError("independent basis lacks an observed transition")
            compatible = (
                [child]
                if child in basis
                else [
                    s for s in basis if _known_witness(nodes, child, s, actions) is None
                ]
            )
            if len(compatible) != 1:
                raise ValueError("independent frontier identification mismatch")
            row = rows[labels[index], action]
            if (
                row["target"] != labels[basis.index(compatible[0])]
                or row["output"] != nodes[child]
            ):
                raise ValueError("independent fold target/output mismatch")


def _check_ir_contract(raw, config):
    ir = TheoryIR.model_validate(raw)
    program = FiniteStateProgram.from_ir(ir).model_dump(mode="json")
    scope = {
        "language": "total-mealy-v1",
        "reset": "initial-state",
        "intervention_horizon": config["protocol"]["horizon"],
        "certified_reset_only": True,
        "claim": "provisional finite conformance evidence; not all-query or unique causal identification",
    }
    if (
        ir.scope != scope
        or type(ir.scope["intervention_horizon"]) is not int
        or ir.latent_variables != ("machine_state",)
        or ir.preconditions
        or ir.composition
        or program["actions"] != config["actions"]
        or program["outputs"] != config["outputs"]
    ):
        raise ValueError("independent bounded IR scope/alphabet mismatch")
    return program


def audit_apartness(config, result, private_path):
    public = {"public_replay_valid": False, "private_truth_opened": False}
    try:
        if (
            set(config) != {"schema", "actions", "outputs", "costs", "protocol"}
            or config["schema"] != "apartness-config-v1"
        ):
            raise ValueError("unregistered apartness configuration")
        if set(config["protocol"]) != {
            "budget",
            "horizon",
            "epoch",
            "seed",
            "conformance_count",
            "operation_budget",
        }:
            raise ValueError("unregistered learner protocol")
        controls = [e for e in result["events"] if e["kind"] in ("reset", "step")]
        cursor, current, episode, tick = 0, None, -1, 0
        transition_ids = set()

        def check_observation(observation):
            # These are public resettable-world protocol invariants, not a
            # prediction from the learner or from a private mechanism table.
            if (
                observation.episode_id != episode
                or observation.step != tick
                or set(observation.agent_state) != {"signal"}
                or observation.agent_state["signal"] not in config["outputs"]
                or {o.object_id for o in observation.objects} != set(config["actions"])
                or len(observation.objects) != len(config["actions"])
                or any(
                    o.affordances != (ActionKind.INTERACT,) for o in observation.objects
                )
            ):
                raise ValueError("independent public clock/alphabet mismatch")

        def reset():
            nonlocal cursor, current, episode, tick
            if cursor >= len(controls) or controls[cursor]["kind"] != "reset":
                raise ValueError("reset chronology mismatch")
            current = Observation.model_validate(controls[cursor]["observation"])
            episode, tick = episode + 1, 0
            check_observation(current)
            if current.last_event is not None:
                raise ValueError("independent reset retains a previous event")
            cursor += 1
            return current

        def available():
            return tuple(
                PrimitiveAction(kind=o.affordances[0], target_id=o.object_id)
                for o in current.objects
            )

        def step(action):
            nonlocal cursor, current, tick
            if cursor >= len(controls) or controls[cursor]["kind"] != "step":
                raise ValueError("control chronology mismatch")
            record = TransitionRecord.model_validate(controls[cursor]["transition"])
            if record.executed_action != action or record.state_before != current:
                raise ValueError("public action/state mismatch")
            tick += 1
            check_observation(record.state_after)
            if (
                record.transition_id in transition_ids
                or record.event is None
                or record.event.event_type != "public_signal"
                or set(record.event.details) != {"signal"}
                or record.state_after.agent_state != record.event.details
                or record.state_after.last_event != record.event
                or record.observed_delta != record.event.details
                or record.state_before.objects != record.state_after.objects
                or record.state_before.inventory != record.state_after.inventory
            ):
                raise ValueError("independent public receipt inconsistency")
            transition_ids.add(record.transition_id)
            cursor += 1
            current = record.state_after
            return record

        view = AgentWorldView(
            observe=lambda: current, available=available, step=step, begin_trial=reset
        )
        replay = learn_apartness(
            view,
            tuple(config["actions"]),
            tuple(config["outputs"]),
            config["costs"],
            **config["protocol"],
        )
        if json.dumps(replay, sort_keys=True, allow_nan=False) != json.dumps(
            result, sort_keys=True, allow_nan=False
        ) or cursor != len(controls):
            raise ValueError("public rule/query/budget replay mismatch")
        spent, nodes, sources, plans, acquired, failed = (
            0,
            {(): None},
            {},
            {},
            {},
            set(),
        )
        for event in result["events"]:
            kind = event["kind"]
            if kind == "plan":
                qid, word = event["query_id"], tuple(event["actions"])
                if qid in plans or event["epoch"] != config["protocol"]["epoch"]:
                    raise ValueError("query identity/epoch mismatch")
                if event["total_cost"] != 1 + sum(config["costs"][a] for a in word):
                    raise ValueError("independent planned cost mismatch")
                plans[qid] = {"word": word, "outputs": []}
            elif kind in ("reset", "step"):
                qid = event["query_id"]
                if kind == "reset":
                    if event["cost"] != 1:
                        raise ValueError("independent reset cost mismatch")
                else:
                    record = event["transition"]
                    action = record["executed_action"]["target_id"]
                    if event["cost"] != config["costs"][action]:
                        raise ValueError("independent action cost mismatch")
                    plans[qid]["outputs"].append(record["event"]["details"]["signal"])
                spent += event["cost"]
            elif kind == "control_failure":
                failed.add(event["query_id"])
            elif kind == "evidence":
                qid = event["query_id"]
                word, outputs = plans[qid]["word"], tuple(event["outputs"])
                if qid in failed or outputs != tuple(plans[qid]["outputs"]):
                    raise ValueError("independent retained evidence mismatch")
                acquired[qid] = word
                for i in range(1, len(word) + 1):
                    prefix, output = word[:i], outputs[i - 1]
                    if prefix in nodes and nodes[prefix] != output:
                        raise ValueError(
                            "inconsistent evidence retained in observation tree"
                        )
                    nodes[prefix] = output
                    sources.setdefault(prefix, set()).add(qid)
            elif kind == "cache":
                word = tuple(event["actions"])
                if word not in nodes or event["epoch"] != config["protocol"]["epoch"]:
                    raise ValueError("cache hit lacks same-epoch prior evidence")
        expected = {
            "epoch": config["protocol"]["epoch"],
            "nodes": [
                {
                    "access": list(p),
                    "incoming_output": nodes[p],
                    "evidence_ids": sorted(sources.get(p, ())),
                }
                for p in sorted(nodes, key=lambda p: (len(p), p))
            ],
        }
        if (
            result["observation_tree"] != expected
            or spent != result["spent"]
            or spent > config["protocol"]["budget"]
        ):
            raise ValueError("independent tree/provenance/ledger mismatch")
        for batch in result["conformance"]:
            program = _check_ir_contract(batch["plan"]["candidate_ir"], config)
            for item in batch["plan"]["queries"]:
                word = tuple(item["actions"])
                if item["prediction"] != list(_rollout(program, word)):
                    raise ValueError(
                        "independent conformance frozen prediction mismatch"
                    )
                if item["cost"] != 1 + sum(config["costs"][a] for a in word):
                    raise ValueError("independent conformance cost mismatch")
            if batch["plan"]["total_cost"] != sum(
                q["cost"] for q in batch["plan"]["queries"]
            ):
                raise ValueError("independent conformance batch ledger mismatch")
        if result["candidate_ir"] is not None:
            program = _check_ir_contract(result["candidate_ir"], config)
            _check_raw_fold(
                program, [tuple(p) for p in result["basis"]], nodes, config["actions"]
            )
            for word in nodes:
                if _rollout(program, word) != tuple(
                    nodes[word[:i]] for i in range(1, len(word) + 1)
                ):
                    raise ValueError(
                        "independent candidate disagrees with retained observations"
                    )
        if (
            result["unique_causal_structure_claimed"]
            or result["all_allowed_queries_certified"]
        ):
            raise ValueError(
                "finite conformance falsely certifies unique/all-language recovery"
            )
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
        raw = json.loads(private_path.read_text(encoding="utf-8"))
        truth = FiniteStateProgram.model_validate(raw).model_dump(mode="json")
        if (
            truth["actions"] != config["actions"]
            or truth["outputs"] != config["outputs"]
        ):
            raise ValueError("private/public alphabet mismatch")
        episode, prefix, tick, valid = -1, [], 0, True
        for event in controls:
            if event["kind"] == "reset":
                episode, prefix, tick = episode + 1, [], 0
                observation = event["observation"]
                valid &= (
                    observation["episode_id"] == episode
                    and observation["step"] == 0
                    and observation["agent_state"] == {"signal": truth["outputs"][0]}
                )
            else:
                record = event["transition"]
                prefix.append(record["executed_action"]["target_id"])
                tick += 1
                signal = _rollout(truth, tuple(prefix))[-1]
                valid &= (
                    record["event"]["details"]["signal"] == signal
                    and record["state_after"]["agent_state"] == {"signal": signal}
                    and record["state_after"]["episode_id"] == episode
                    and record["state_after"]["step"] == tick
                )
        correct = None
        if result["candidate_ir"] is not None:
            program = result["candidate_ir"]["parameters"]
            # Independent raw paired-state BFS, depth-limited to public H.
            left = {(r["source"], r["action"]): r for r in program["transitions"]}
            right = {(r["source"], r["action"]): r for r in truth["transitions"]}
            pending = deque([(program["initial_state"], truth["initial_state"], 0)])
            seen, correct = set(), True
            while pending and correct:
                pair = pending.popleft()
                if pair in seen or pair[2] == config["protocol"]["horizon"]:
                    continue
                seen.add(pair)
                for action in config["actions"]:
                    lrow, rrow = left[pair[0], action], right[pair[1], action]
                    if lrow["output"] != rrow["output"]:
                        correct = False
                        break
                    pending.append((lrow["target"], rrow["target"], pair[2] + 1))
        return public | {
            "valid": bool(valid),
            "evaluation": {
                "actual_controls_match_private_reference": bool(valid),
                "provisional_behavior_correct_at_H": correct,
                "scope": "fixed stationary epoch; not causal factor attribution or exact-EQ convergence",
            },
        }
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}


def run_automata_benchmark(
    *,
    output: Path,
    seed=0,
    seeds=2,
    budget=240,
    horizon=5,
    maximum_states=5,
    conformance_count=8,
    operation_budget=200_000,
):
    """Development of the from-scratch baseline, not an H3 superiority study.

    The timing artifact is explicitly nondeterministic and separate from all
    deterministic evidence. Every actual reset, test, failure and input is paid.
    No true program, state count, family, world seed or candidate reaches the
    learner. All cases, including insufficient budgets and false provisional
    predictions, remain in the report.
    """
    if (
        type(seed) is not int
        or seed < 0
        or type(seeds) is not int
        or not 1 <= seeds <= 64
        or type(budget) is not int
        or budget < 1
        or type(horizon) is not int
        or not 1 <= horizon <= 8
        or type(maximum_states) is not int
        or not 2 <= maximum_states <= 8
        or type(conformance_count) is not int
        or not 1 <= conformance_count <= 64
        or type(operation_budget) is not int
        or operation_budget < 1
    ):
        raise ValueError("invalid stationary baseline benchmark protocol")
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]

    def source_hashes():
        return {
            p.relative_to(package).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(package.rglob("*.py"))
        }

    sources = source_hashes()
    protocol = {
        "budget": budget,
        "horizon": horizon,
        "epoch": "stationary-v1",
        "seed": 20261008,
        "conformance_count": conformance_count,
        "operation_budget": operation_budget,
    }
    registration = {
        "schema": "apartness-development-registration-v1",
        "method": "basic-lsharp-finite-conformance",
        "generator": "aal-tables-v1",
        "seed": seed,
        "seeds": seeds,
        "families": list(AUTOMATA_FAMILIES),
        "private_state_counts": list(range(2, maximum_states + 1)),
        "protocol": protocol,
        "source_sha256": sources,
        "agent_inputs": ["actions", "outputs", "costs", "protocol", "capability_view"],
        "reset_cost": 1,
        "each_action_cost": 1,
        "validation": "frozen fresh real conformance queries; no private EQ oracle",
        "split": "synthetic development grid, not a formal structural holdout",
        "limitations": [
            "stationary deterministic resettable behavior learning only",
            "not full Adaptive L#; no reference matching or prior-model reuse",
            "finite conformance is not global or H-complete certification",
            "no control/physical-mechanism factor attribution or online H3 comparison",
            "observation visits are partial counters, not all inference computation",
            "not a novelty, H4, Minecraft/RGB or complete-release result",
        ],
    }
    _write_json(output / "registration.json", registration)
    rows, timings = [], []
    for current_seed, family, states in itertools.product(
        range(seed, seed + seeds),
        AUTOMATA_FAMILIES,
        range(2, maximum_states + 1),
    ):
        directory = output / f"seed-{current_seed}" / family / f"states-{states}"
        directory.mkdir(parents=True)
        task = automata_task(current_seed, family, states)
        config = {
            "schema": "apartness-config-v1",
            "actions": list(task.actions),
            "outputs": list(task.outputs),
            "costs": task.costs,
            "protocol": protocol,
        }
        _write_json(directory / "config.json", config)
        wall, cpu = perf_counter(), process_time()
        result = learn_apartness(
            task.view,
            task.actions,
            task.outputs,
            task.costs,
            **protocol,
        )
        learning_time = {
            "wall_seconds": perf_counter() - wall,
            "cpu_seconds": process_time() - cpu,
        }
        _write_json(directory / "run.json", result)
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
        _write_json(private, task.ground_truth.model_dump(mode="json"))
        wall, cpu = perf_counter(), process_time()
        audit = audit_apartness(config, result, private)
        audit_time = {
            "wall_seconds": perf_counter() - wall,
            "cpu_seconds": process_time() - cpu,
        }
        _write_json(directory / "audit.json", audit)
        if not audit["valid"]:
            raise RuntimeError(
                f"independent apartness audit failed: {directory}; retain partial run"
            )
        key = {"seed": current_seed, "family": family, "private_states": states}
        paid_events = [e for e in result["events"] if e["kind"] in ("reset", "step")]
        rows.append(
            key
            | {
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "spent": result["spent"],
                "paid_queries": result["paid_queries"],
                "paid_actions": sum(e["kind"] == "step" for e in paid_events),
                "cache_hits": result["cache_hits"],
                "observation_visits": result["observation_visits"],
                "conformance_cost": sum(
                    e["cost"]
                    for e in paid_events
                    if e["query_id"]
                    in {
                        p["query_id"]
                        for p in result["events"]
                        if p["kind"] == "plan" and p["reason"] == "conformance"
                    }
                ),
                "candidate_states": None
                if result["candidate_ir"] is None
                else len(result["candidate_ir"]["parameters"]["states"]),
                "evaluation": audit["evaluation"],
            }
        )
        timings.append(
            key
            | {
                "learning_and_real_validation": learning_time,
                "independent_audit_and_private_scoring": audit_time,
            }
        )
    if source_hashes() != sources:
        raise RuntimeError(
            "source changed during baseline benchmark; retain partial run"
        )
    report = {
        "schema": "apartness-development-benchmark-v1",
        "registration": registration,
        "rows": rows,
        "all_independent_audits_valid": True,
        "summary": {
            "runs": len(rows),
            "provisional_conformance_passed": sum(
                r["status"] == "provisional_conformance_passed" for r in rows
            ),
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
        "scope": "from-scratch basic L# adaptation development; not full Adaptive L#, causal attribution, online H3 comparison or formal held-out superiority",
    }
    _write_json(output / "report.json", report)
    _write_json(
        output / "timing.json",
        {
            "schema": "apartness-nondeterministic-timing-v1",
            "rows": timings,
            "scope": "complete learner call (discovery, real conformance, compilation); audit separately; no serialization/setup or external API costs",
        },
    )
    artifacts = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    _write_json(
        output / "manifest.json",
        {
            "schema": "apartness-artifact-manifest-v1",
            "sha256": artifacts,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    return report
