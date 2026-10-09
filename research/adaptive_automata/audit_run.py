"""Stationary adaptive-run audit: raw public ledger + policy replay BEFORE I/O.

Reference/sampler checks do not call the matching/sampling implementations.
Raw folds/predictions and event chronology protect against shared replay defects.
Known failed dispatch is charged but not retained as mechanism evidence. Unknown
delivery, noise and controller-factor attribution remain outside this protocol.
"""

import hashlib
import json
from collections import deque

from audit_matching import check_reference_plan
from audit_random_wp import check_random_wp_plan
from learner import learn_adaptive

from minecraft_scientist.evaluation.active_automata import (
    _check_ir_contract,
    _check_raw_fold,
    _known_witness,
)
from minecraft_scientist.evaluation.identification import _rollout
from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TheoryIR,
    TransitionRecord,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def independent_program(raw, config):
    program = _check_ir_contract(raw, config)
    if raw["scope"]["certified_reset_only"] is not True or canonical(
        raw["parameters"]
    ) != canonical(program):
        raise ValueError("independent bounded IR exact type/program mismatch")
    return program


class PublicLedger:
    def __init__(self, config, result):
        if (
            type(config) is not dict
            or set(config)
            != {"schema", "actions", "outputs", "costs", "references", "protocol"}
            or config["schema"] != "adaptive-config-v1"
        ):
            raise ValueError("unregistered adaptive configuration")
        protocol_fields = {
            "budget",
            "horizon",
            "epoch",
            "seed",
            "conformance_count",
            "operation_budget",
            "reference_operation_budget",
        }
        if set(config["protocol"]) != protocol_fields:
            raise ValueError("unregistered adaptive protocol")
        for name in ("actions", "outputs"):
            alphabet = config[name]
            if (
                type(alphabet) is not list
                or not 1 <= len(alphabet) <= 8
                or any(type(a) is not str or not 1 <= len(a) <= 80 for a in alphabet)
                or len(set(alphabet)) != len(alphabet)
            ):
                raise ValueError("independent public alphabet configuration mismatch")
        protocol = config["protocol"]
        for name, low, high in (
            ("budget", 1, None),
            ("horizon", 1, 8),
            ("seed", 0, None),
            ("conformance_count", 1, 64),
            ("operation_budget", 1, None),
            ("reference_operation_budget", 1, None),
        ):
            value = protocol[name]
            if (
                type(value) is not int
                or value < low
                or (high is not None and value > high)
            ):
                raise ValueError("independent exact protocol configuration mismatch")
        if (
            type(protocol["epoch"]) is not str
            or not 1 <= len(protocol["epoch"]) <= 80
            or type(config["references"]) is not list
            or not 1 <= len(config["references"]) <= 4
        ):
            raise ValueError("independent epoch/reference configuration mismatch")
        if (
            type(config["costs"]) is not dict
            or set(config["costs"]) != set(config["actions"])
            or any(type(c) is not int or c < 1 for c in config["costs"].values())
        ):
            raise ValueError("independent exact cost configuration mismatch")
        if (
            type(result) is not dict
            or result["schema"] != "isolated-adaptive-rules-v3"
            or type(result["events"]) is not list
            or len(result["events"]) > 100_000
        ):
            raise ValueError("unregistered/bounded adaptive result")
        self.config, self.result, self.protocol = config, result, config["protocol"]
        self.actions, self.costs, self.epoch = (
            tuple(config["actions"]),
            config["costs"],
            self.protocol["epoch"],
        )
        self.nodes, self.sources, self.basis = {(): None}, {}, [()]
        self.spent, self.paid_queries, self.cache_hits = 0, 0, 0
        self.active, self.qids, self.receipts = None, set(), set()
        self.current, self.episode, self.tick = None, -1, 0
        self.failed, self.closed, self.latest_program = False, False, None
        self.reference_queue, self.reference_rule = [], None
        self.batches, self.batch, self.conformance_cursor = [], None, 0
        self.controls, self.reference_checks = [], 0

    def snapshot(self):
        return {
            "epoch": self.epoch,
            "nodes": [
                {
                    "access": list(p),
                    "incoming_output": self.nodes[p],
                    "evidence_ids": sorted(self.sources.get(p, ())),
                }
                for p in sorted(self.nodes, key=lambda p: (len(p), p))
            ],
        }

    def check_basis(self, event):
        if event["observation_digest"] != digest(self.snapshot()) or event["basis"] != [
            list(p) for p in self.basis
        ]:
            raise ValueError("independent basis/evidence chronology mismatch")

    def word(self, value):
        if (
            type(value) is not list
            or not 1 <= len(value) <= self.protocol["horizon"]
            or any(type(a) is not str or a not in self.actions for a in value)
        ):
            raise ValueError("independent query alphabet/horizon mismatch")
        return tuple(value)

    def observation(self, raw):
        observation = Observation.model_validate(raw)
        if canonical(observation.model_dump(mode="json")) != canonical(raw):
            raise ValueError("independent public observation type/default mismatch")
        if (
            observation.episode_id != self.episode
            or observation.step != self.tick
            or set(observation.agent_state) != {"signal"}
            or observation.agent_state["signal"] not in self.config["outputs"]
            or {o.object_id for o in observation.objects} != set(self.actions)
            or len(observation.objects) != len(self.actions)
            or any(o.affordances != (ActionKind.INTERACT,) for o in observation.objects)
        ):
            raise ValueError("independent public clock/alphabet mismatch")
        return observation

    def charge(self, event, expected):
        if type(event["cost"]) is not int or event["cost"] != expected:
            raise ValueError("independent exact paid cost mismatch")
        self.spent += expected
        if self.spent > self.protocol["budget"]:
            raise ValueError("independent paid budget exceeded")

    def query_plan(self, event):
        if self.active is not None or self.failed:
            raise ValueError("independent overlapping/post-failure query")
        word = self.word(event["actions"])
        qid = event["query_id"]
        if (
            qid != f"adaptive-query-{self.paid_queries}"
            or qid in self.qids
            or event["epoch"] != self.epoch
        ):
            raise ValueError("independent query identity/epoch mismatch")
        amount = 1 + sum(self.costs[a] for a in word)
        if (
            type(event["total_cost"]) is not int
            or event["total_cost"] != amount
            or amount > self.protocol["budget"] - self.spent
        ):
            raise ValueError("independent query preflight mismatch")
        reason = event["reason"]
        if reason != "conformance" and word in self.nodes:
            raise ValueError(
                "independent cached membership dispatched as paid experiment"
            )
        if self.reference_queue:
            if reason != self.reference_rule or word != self.reference_queue.pop(0):
                raise ValueError("independent reference dispatch order mismatch")
        elif reason in ("R", "AMR", "APS", "AMS"):
            raise ValueError("reference dispatch without frozen guidance")
        prediction = []
        if reason in ("conformance", "counterexample"):
            if self.latest_program is None:
                raise ValueError("predicted query lacks a prior public hypothesis")
            prediction = list(_rollout(self.latest_program, word))
        elif reason not in ("Ex", "S", "R", "AMR", "APS", "AMS"):
            raise ValueError("independent unknown query reason")
        if event["prediction"] != prediction:
            raise ValueError("independent frozen query prediction mismatch")
        intervention = {
            "intervention_id": qid,
            "mechanism": "finite_transducer",
            "controlled_variables": {"reset": True},
            "target_variable": "output_trace",
            "fixed_context": {
                "language": "total-mealy-v1",
                "evidence_epoch": self.epoch,
            },
            "predicted_outcomes": {"hypothesis": prediction}
            if reason in ("conformance", "counterexample")
            else {},
            "stopping_condition": "terminal_event",
        }
        if canonical(event["intervention"]) != canonical(intervention):
            raise ValueError("independent intended intervention mismatch")
        if reason == "conformance":
            if (
                self.batch is None
                or self.batch["status"] != "unresolved"
                or self.batch["plan"]["status"] != "planned"
                or self.batch.get("budget_blocked", False)
                or self.conformance_cursor >= len(self.batch["plan"]["queries"])
            ):
                raise ValueError("independent conformance query has no active plan")
            item = self.batch["plan"]["queries"][self.conformance_cursor]
            if (
                event["actions"] != item["actions"]
                or event["prediction"] != item["prediction"]
            ):
                raise ValueError("independent conformance dispatch order mismatch")
        self.qids.add(qid)
        self.active = {
            "qid": qid,
            "word": word,
            "outputs": [],
            "reset": False,
            "bad": False,
            "reason": reason,
            "prediction": prediction,
            "intervention": intervention,
        }

    def control(self, event):
        q = self.active
        if q is None or event["query_id"] != q["qid"] or q["bad"]:
            raise ValueError("independent control/query chronology mismatch")
        if event["kind"] == "reset":
            if q["reset"] or q["outputs"]:
                raise ValueError("independent duplicate/late reset")
            self.episode, self.tick = self.episode + 1, 0
            self.current = self.observation(event["observation"])
            if self.current.last_event is not None:
                raise ValueError("independent reset retained event")
            q["reset"] = True
            self.paid_queries += 1
            self.charge(event, 1)
        else:
            if not q["reset"] or len(q["outputs"]) >= len(q["word"]):
                raise ValueError("independent unplanned/pre-reset action")
            raw = event["transition"]
            record = TransitionRecord.model_validate(raw)
            if canonical(record.model_dump(mode="json")) != canonical(raw):
                raise ValueError("independent transition type/default mismatch")
            action = q["word"][len(q["outputs"])]
            if (
                record.executed_action
                != PrimitiveAction(kind=ActionKind.INTERACT, target_id=action)
                or record.state_before != self.current
            ):
                raise ValueError("independent public action/state mismatch")
            self.tick += 1
            after = self.observation(raw["state_after"])
            if (
                record.transition_id in self.receipts
                or record.event.event_type != "public_signal"
                or set(record.event.details) != {"signal"}
                or after.agent_state != record.event.details
                or after.last_event != record.event
                or record.observed_delta != record.event.details
                or record.state_before.objects != after.objects
                or record.state_before.inventory != after.inventory
                or canonical(raw["intended_intervention"])
                != canonical(q["intervention"])
                or record.possible_confounders
            ):
                raise ValueError(
                    "independent public receipt/intervention inconsistency"
                )
            self.receipts.add(record.transition_id)
            q["outputs"].append(record.event.details["signal"])
            q["bad"] = (
                not record.intervention_compliance
                or bool(record.controller_failure)
                or not record.event.success
            )
            self.current = after
            self.charge(event, self.costs[action])
        self.controls.append(event)

    def evidence(self, event):
        q = self.active
        if (
            q is None
            or q["bad"]
            or not q["reset"]
            or event["query_id"] != q["qid"]
            or event["outputs"] != q["outputs"]
            or len(q["outputs"]) != len(q["word"])
        ):
            raise ValueError("independent incomplete/failed retained evidence")
        for i, value in enumerate(q["outputs"], 1):
            prefix = q["word"][:i]
            if prefix in self.nodes and self.nodes[prefix] != value:
                raise ValueError("independent inconsistent evidence retained")
            self.nodes[prefix] = value
            self.sources.setdefault(prefix, set()).add(q["qid"])
        if q["reason"] == "conformance":
            self.batch["query_ids"].append(q["qid"])
            self.conformance_cursor += 1
            if q["outputs"] != q["prediction"]:
                self.batch["status"] = "counterexample"
            elif self.conformance_cursor == self.protocol["conformance_count"]:
                self.batch["status"] = "passed"
        self.active = None

    def validate(self):
        event_fields = {
            "plan": {
                "kind",
                "query_id",
                "epoch",
                "reason",
                "actions",
                "prediction",
                "total_cost",
                "intervention",
            },
            "reset": {"kind", "query_id", "cost", "observation"},
            "step": {"kind", "query_id", "cost", "transition"},
            "evidence": {"kind", "query_id", "outputs"},
            "control_failure": {"kind", "query_id"},
            "cache": {"kind", "epoch", "actions", "reason"},
            "reference_plan": {"kind", "plan", "estimated_paid_cost"},
            "basis_promotion": {
                "kind",
                "rule",
                "access",
                "basis",
                "observation_digest",
            },
            "rebuilding_closed": {"kind", "basis", "observation_digest"},
            "hypothesis_plan": {"kind", "candidate_ir", "basis", "observation_digest"},
            "conformance_plan": {"kind", "plan"},
        }
        for event in self.result["events"]:
            if (
                type(event) is not dict
                or type(event.get("kind")) is not str
                or event["kind"] not in event_fields
                or set(event) != event_fields[event["kind"]]
            ):
                raise ValueError("independent event schema mismatch")
            kind = event["kind"]
            if self.failed:
                raise ValueError(
                    "independent events after terminal known control failure"
                )
            if kind == "plan":
                self.query_plan(event)
            elif kind in ("reset", "step"):
                self.control(event)
            elif kind == "evidence":
                self.evidence(event)
            elif kind == "control_failure":
                if (
                    self.active is None
                    or not self.active["bad"]
                    or event["query_id"] != self.active["qid"]
                ):
                    raise ValueError(
                        "independent control failure without failed receipt"
                    )
                self.failed = True
            elif kind == "cache":
                word = self.word(event["actions"])
                if (
                    self.active is not None
                    or word not in self.nodes
                    or event["epoch"] != self.epoch
                    or event["reason"] == "conformance"
                ):
                    raise ValueError(
                        "independent cache lacks prior evidence/is stale validation"
                    )
                if self.reference_queue:
                    if (
                        word != self.reference_queue.pop(0)
                        or event["reason"] != self.reference_rule
                    ):
                        raise ValueError("independent reference cache order mismatch")
                elif event["reason"] in ("R", "AMR", "APS", "AMS"):
                    raise ValueError("independent unplanned reference cache")
                self.cache_hits += 1
            elif kind == "reference_plan":
                if self.active is not None or self.reference_queue:
                    raise ValueError("independent reference plan overlaps a query")
                plan = event["plan"]
                if (
                    plan["basis"] != [list(p) for p in self.basis]
                    or (plan["rule"] == "R") == self.closed
                ):
                    raise ValueError("independent reference basis/phase mismatch")
                check_reference_plan(
                    plan,
                    self.snapshot(),
                    self.config["references"],
                    self.actions,
                    self.costs,
                    event["estimated_paid_cost"],
                )
                if event["estimated_paid_cost"] > self.protocol["budget"] - self.spent:
                    raise ValueError("independent reference batch budget mismatch")
                self.reference_queue = [self.word(w) for w in plan["words"]]
                self.reference_rule = plan["rule"]
                self.reference_checks += 1
            elif kind == "basis_promotion":
                prefix = self.word(event["access"])
                if (
                    self.active is not None
                    or self.reference_queue
                    or event["rule"] not in ("P", "PP")
                    or (event["rule"] == "PP") == self.closed
                ):
                    raise ValueError("independent promotion phase/order mismatch")
                if (
                    prefix not in self.nodes
                    or prefix in self.basis
                    or prefix[:-1] not in self.basis
                    or any(
                        _known_witness(self.nodes, prefix, p, self.actions) is None
                        for p in self.basis
                    )
                ):
                    raise ValueError(
                        "independent promotion lacks observed frontier apartness"
                    )
                self.basis.append(prefix)
                self.check_basis(event)
            elif kind == "rebuilding_closed":
                if self.closed or self.active is not None or self.reference_queue:
                    raise ValueError("independent rebuilding closure mismatch")
                self.check_basis(event)
                self.closed = True
            elif kind == "hypothesis_plan":
                if not self.closed or self.active is not None or self.reference_queue:
                    raise ValueError("independent hypothesis chronology mismatch")
                self.check_basis(event)
                self.latest_program = independent_program(
                    event["candidate_ir"], self.config
                )
                _check_raw_fold(
                    self.latest_program, self.basis, self.nodes, self.actions
                )
            elif kind == "conformance_plan":
                if (
                    self.active is not None
                    or self.reference_queue
                    or self.latest_program is None
                    or (self.batch is not None and self.batch["status"] == "unresolved")
                ):
                    raise ValueError("independent conformance-plan chronology mismatch")
                plan = event["plan"]
                check_random_wp_plan(plan, self.costs)
                if (
                    canonical(plan["hypothesis_program"])
                    != canonical(self.latest_program)
                    or plan["seed"] != self.protocol["seed"] + len(self.batches)
                    or plan["horizon"] != self.protocol["horizon"]
                    or plan["requested_count"] != self.protocol["conformance_count"]
                    or any(
                        list(_rollout(self.latest_program, p))
                        != [self.nodes[p[:i]] for i in range(1, len(p) + 1)]
                        for p in self.nodes
                    )
                ):
                    raise ValueError(
                        "independent tested hypothesis disagrees with prior evidence/protocol"
                    )
                self.batch = {"plan": plan, "query_ids": [], "status": "unresolved"}
                self.batches.append(self.batch)
                self.conformance_cursor = 0
                if (
                    plan["status"] == "planned"
                    and plan["total_cost"] > self.protocol["budget"] - self.spent
                ):
                    # A frozen but unaffordable plan is allowed only with no dispatch.
                    self.batch["budget_blocked"] = True
            else:
                raise ValueError("independent unknown event kind")
        # Observation capacity/noisy-trace stops can occur after paid output,
        # before atomic insertion; no partial prefix may be retained.
        if (
            self.active is not None
            and not self.failed
            and self.result["stop_reason"]
            not in (
                "nonstationary_or_noisy_trace",
                "observation_capacity_exhausted",
                "planned_action_unavailable",
            )
        ):
            raise ValueError("independent unfinished query without declared stop")
        for batch in self.batches:
            if batch.pop("budget_blocked", False) and batch["query_ids"]:
                raise ValueError("independent unaffordable batch dispatched")
        if canonical(self.batches) != canonical(self.result["conformance"]):
            raise ValueError("independent conformance receipt/status mismatch")
        if canonical(self.snapshot()) != canonical(
            self.result["observation_tree"]
        ) or self.result["basis"] != [list(p) for p in self.basis]:
            raise ValueError("independent final tree/provenance/basis mismatch")
        for key, value in (
            ("spent", self.spent),
            ("paid_queries", self.paid_queries),
            ("cache_hits", self.cache_hits),
            ("budget", self.protocol["budget"]),
        ):
            if type(self.result[key]) is not int or self.result[key] != value:
                raise ValueError("independent aggregate ledger mismatch")
        if self.result["reference_digest"] != digest(self.config["references"]):
            raise ValueError("independent owned-reference digest mismatch")
        if (
            any(
                self.result[k] is not False
                for k in (
                    "unique_causal_structure_claimed",
                    "all_allowed_queries_certified",
                    "operation_counters_are_complete_cpu_cost",
                )
            )
            or self.result["reference_models_used"] is not True
        ):
            raise ValueError("independent false scope/cost claim")
        if self.result["candidate_ir"] is not None:
            program = independent_program(self.result["candidate_ir"], self.config)
            _check_raw_fold(program, self.basis, self.nodes, self.actions)
            if (
                self.result["status"] != "provisional_conformance_passed"
                or not self.batches
                or self.batches[-1]["status"] != "passed"
                or canonical(program) != canonical(self.latest_program)
            ):
                raise ValueError("independent unvalidated proposal promotion")
        elif self.result["status"] != "unresolved":
            raise ValueError("independent passing run without a candidate")


def audit_adaptive(config, result, private_path):
    public = {
        "public_replay_valid": False,
        "raw_public_ledger_valid": False,
        "private_truth_opened": False,
        "private_truth_read_attempted": False,
    }
    try:
        ledger = PublicLedger(config, result)
        ledger.validate()
        public["raw_public_ledger_valid"] = True
        cursor, current = 0, None

        def reset():
            nonlocal cursor, current
            if (
                cursor >= len(ledger.controls)
                or ledger.controls[cursor]["kind"] != "reset"
            ):
                raise ValueError("policy replay reset chronology mismatch")
            current = Observation.model_validate(ledger.controls[cursor]["observation"])
            cursor += 1
            return current

        def available():
            return tuple(
                PrimitiveAction(kind=o.affordances[0], target_id=o.object_id)
                for o in current.objects
            )

        def step(action):
            nonlocal cursor, current
            if (
                cursor >= len(ledger.controls)
                or ledger.controls[cursor]["kind"] != "step"
            ):
                raise ValueError("policy replay action chronology mismatch")
            record = TransitionRecord.model_validate(
                ledger.controls[cursor]["transition"]
            )
            if record.executed_action != action or record.state_before != current:
                raise ValueError("policy replay action/state mismatch")
            cursor += 1
            current = record.state_after
            return record

        view = AgentWorldView(
            observe=lambda: current, available=available, step=step, begin_trial=reset
        )
        references = tuple(TheoryIR.model_validate(r) for r in config["references"])
        replay = learn_adaptive(
            view,
            tuple(config["actions"]),
            tuple(config["outputs"]),
            config["costs"],
            references,
            **config["protocol"],
        )
        if cursor != len(ledger.controls) or canonical(replay) != canonical(result):
            raise ValueError("public adaptive policy/query/budget replay mismatch")
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

    public["private_truth_read_attempted"] = True
    try:
        private_text = private_path.read_text(encoding="utf-8")
        public["private_truth_opened"] = True
        truth = FiniteStateProgram.model_validate(json.loads(private_text)).model_dump(
            mode="json"
        )
        if (
            truth["actions"] != config["actions"]
            or truth["outputs"] != config["outputs"]
        ):
            raise ValueError("private/public alphabet mismatch")
        prefix, valid, skipped = (), True, 0
        for event in ledger.controls:
            if event["kind"] == "reset":
                prefix = ()
                valid &= event["observation"]["agent_state"] == {
                    "signal": truth["outputs"][0]
                }
            else:
                record = event["transition"]
                prefix += (record["executed_action"]["target_id"],)
                if (
                    not record["intervention_compliance"]
                    or record["controller_failure"]
                    or not record["event"]["success"]
                ):
                    skipped += 1
                    continue
                valid &= (
                    record["event"]["details"]["signal"] == _rollout(truth, prefix)[-1]
                )
        correct = None
        if result["candidate_ir"] is not None:
            program = result["candidate_ir"]["parameters"]
            rows = [
                {(r["source"], r["action"]): r for r in p["transitions"]}
                for p in (program, truth)
            ]
            pending, seen, correct = (
                deque([(program["initial_state"], truth["initial_state"], 0)]),
                set(),
                True,
            )
            while pending and correct:
                pair = pending.popleft()
                if pair in seen or pair[2] == config["protocol"]["horizon"]:
                    continue
                seen.add(pair)
                for action in config["actions"]:
                    left, right = rows[0][pair[0], action], rows[1][pair[1], action]
                    if left["output"] != right["output"]:
                        correct = False
                        break
                    pending.append((left["target"], right["target"], pair[2] + 1))
        return public | {
            "valid": bool(valid),
            "evaluation": {
                "successful_controls_match_private_reference": bool(valid),
                "known_failed_dispatches_excluded": skipped,
                "provisional_behavior_correct_at_H": correct,
                "raw_reference_plans_checked": ledger.reference_checks,
                "scope": "stationary deterministic trusted-reset behavior only; known terminal failed dispatch charged/excluded, no unknown delivery, factor attribution, noise, online transfer or exact-EQ guarantee",
            },
        }
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        return public | {"valid": False, "failure": str(error), "evaluation": None}
