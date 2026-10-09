"""Bounded edit synthesis, actual crisis monitoring and prospective validation.

The scientific registry publishes only after fresh shadow trials. Temporary H1
provisionals in nested runs are retained but are not scientific promotions.
"""

from minecraft_scientist.agent.identification import identify_mechanism
from minecraft_scientist.discovery.diagnosis import JointHypothesis, validate_hypotheses
from minecraft_scientist.discovery.revision import enumerate_structural_edits
from minecraft_scientist.protocols import TheoryIR, TheoryStatus
from minecraft_scientist.theory.finite_state import FiniteStateProgram, validate_costs
from minecraft_scientist.theory.ir import TheoryInterpreter
from minecraft_scientist.theory.registry import TheoryRegistry
from minecraft_scientist.theory.shadow import validate_shadow


def _bounded(ir, horizon):
    return ir.model_copy(
        deep=True,
        update={
            "scope": ir.scope
            | {
                "intervention_horizon": horizon,
                "certified_reset_only": True,
                "claim": "conditional behavior on reset queries, not unique causal structure",
            }
        },
    )


def control_runs(result):
    """Chronological nested executor results, used by a separate replay auditor."""
    if result["bootstrap"] is not None:
        yield result["bootstrap"]
    for run in (result["bootstrap_validation"] or {}).get("runs", ()):
        yield run
    yield from result["monitoring"]
    if result["discovery"] is not None:
        yield result["discovery"]
    for run in (result["replacement_validation"] or {}).get("runs", ()):
        yield run


def revise_online(
    view,
    bootstrap,
    calibration,
    costs,
    *,
    method="witness_cover",
    seed=20261008,
    budget=240,
    horizon=5,
    monitor_trials=10,
    search_budget=1000,
    shadow_count=3,
    candidate_limit=512,
):
    if (
        method not in ("witness_cover", "sequence_ig")
        or any(
            type(value) is not int or value < 1
            for value in (
                budget,
                horizon,
                monitor_trials,
                search_budget,
                shadow_count,
                candidate_limit,
            )
        )
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("invalid online revision protocol")
    if (
        horizon > 5
        or candidate_limit > 512
        or shadow_count > 16
        or monitor_trials > 100
    ):
        raise ValueError("online development protocol exceeds its declared bounds")
    if type(bootstrap) is not tuple or not bootstrap or len(bootstrap) > 12:
        raise ValueError("bounded bootstrap hypotheses are required")
    bootstrap = tuple(
        JointHypothesis.model_validate(hypothesis.model_dump())
        for hypothesis in bootstrap
    )
    candidates = validate_hypotheses(
        bootstrap, bootstrap[0].mechanism_id, bootstrap[0].controller_id
    )
    costs = dict(costs)
    registry = TheoryRegistry()
    result = {
        "schema": "online-revision-run-v1",
        "method": method,
        "budget": budget,
        "spent": 0,
        "bootstrap": None,
        "bootstrap_validation": None,
        "monitoring": [],
        "generation": None,
        "discovery": None,
        "replacement_validation": None,
        "theory_events": [],
        "status": "unresolved",
        "mechanism_change_warranted": False,
        "unique_causal_structure_claimed": False,
        "bootstrap_independent_validation_completed": False,
        "replacement_independent_validation_completed": False,
        "mechanism_attribution_status": "unresolved",
    }
    used = set()

    def remaining_budget():
        return budget - result["spent"]

    def remember(run):
        result["spent"] += run["spent"]
        used.update(
            tuple(event["actions"])
            for event in run["events"]
            if event["kind"] == "plan"
        )

    def publish(version):
        result["theory_events"].append(version.model_dump(mode="json"))
        return version

    def shadow(ir, prefix):
        batch = validate_shadow(
            view,
            ir,
            costs,
            used,
            budget=remaining_budget(),
            count=shadow_count,
            horizon=horizon,
            seed=seed + 1,
            prefix=prefix,
        )
        for run in batch["runs"]:
            remember(run)
        return batch

    initial = identify_mechanism(
        view,
        candidates,
        calibration,
        costs,
        method=method,
        seed=seed,
        budget=min(40, budget),
        horizon=horizon,
        query_prefix="bootstrap-query-",
    )
    result["bootstrap"] = initial
    remember(initial)
    if initial["status"] != "conditional_behavior_identified":
        result["stop_reason"] = "bootstrap_unresolved"
        return result
    chosen = bootstrap[min(initial["remaining_candidates"])]
    old = _bounded(chosen.compile(), horizon)
    version = publish(
        registry.propose(
            "issued-response",
            old,
            evidence_ids=tuple(
                event["query_id"]
                for event in initial["events"]
                if event["kind"] == "evidence"
            ),
            validation_score=0,
            confidence=0,
            created_by="bootstrap-known-family",
        )
    )
    publish(
        registry.transition(
            version.theory_id, version.version, TheoryStatus.PROVISIONAL
        )
    )
    initial_shadow = shadow(old, "shadow-before-")
    result["bootstrap_validation"] = initial_shadow
    if initial_shadow["status"] != "passed":
        publish(
            registry.transition(
                version.theory_id, version.version, TheoryStatus.REJECTED
            )
        )
        result["stop_reason"] = "bootstrap_validation_failed"
        return result
    publish(
        registry.transition(version.theory_id, version.version, TheoryStatus.ACCEPTED)
    )
    result["bootstrap_independent_validation_completed"] = True
    monitor_words = (
        calibration,
        *(
            calibration + (action,)
            for action in sorted(costs)
            if len(calibration) < horizon
        ),
    )
    anomaly = None
    for index in range(monitor_trials):
        if remaining_budget() < 1:
            result["stop_reason"] = "monitoring_budget_exhausted"
            return result
        trial = identify_mechanism(
            view,
            (old,),
            monitor_words[index % len(monitor_words)],
            costs,
            budget=remaining_budget(),
            horizon=horizon,
            query_prefix=f"monitor-{index}-",
        )
        result["monitoring"].append(trial)
        remember(trial)
        if trial["status"] == "model_mismatch":
            anomaly = trial
            publish(
                registry.transition(
                    "issued-response",
                    1,
                    TheoryStatus.CHALLENGED,
                    counterexample_ids=tuple(
                        event["query_id"]
                        for event in trial["events"]
                        if event["kind"] == "evidence"
                    ),
                )
            )
            break
        if trial["stop_reason"] is not None:
            result["stop_reason"] = trial["stop_reason"]
            return result
    if anomaly is None:
        result["status"], result["stop_reason"] = (
            "no_anomaly_detected",
            "monitoring_limit",
        )
        return result

    anomaly_word = next(
        tuple(e["actions"]) for e in anomaly["events"] if e["kind"] == "plan"
    )
    anomaly_outputs = next(
        tuple(e["outputs"]) for e in anomaly["events"] if e["kind"] == "evidence"
    )
    repair = discover_structural_repair(
        view,
        bootstrap,
        chosen,
        anomaly_word,
        anomaly_outputs,
        tuple(e["query_id"] for e in anomaly["events"] if e["kind"] == "evidence"),
        costs,
        method=method,
        seed=seed,
        budget=remaining_budget(),
        horizon=horizon,
        search_budget=search_budget,
        candidate_limit=candidate_limit,
        validation_reserve=shadow_count * (1 + horizon * max(costs.values())),
    )
    for key in (
        "generation",
        "discovery",
        "mechanism_change_warranted",
        "mechanism_attribution_status",
        "possible_mechanisms",
        "possible_controllers",
    ):
        if key in repair:
            result[key] = repair[key]
    if repair["discovery"] is not None:
        remember(repair["discovery"])
    if repair["status"] != "provisional_structure_identified":
        result["status"], result["stop_reason"] = (
            repair["status"],
            repair["stop_reason"],
        )
        return result
    replacement = TheoryIR.model_validate(repair["candidate_ir"])
    discovery = repair["discovery"]
    new_version = publish(
        registry.propose(
            "issued-response",
            replacement,
            parent_version=1,
            evidence_ids=tuple(
                event["query_id"]
                for event in discovery["events"]
                if event["kind"] == "evidence"
            ),
            validation_score=0,
            confidence=0,
            created_by="bounded-edit-synthesis",
        )
    )
    publish(
        registry.transition(
            "issued-response", new_version.version, TheoryStatus.PROVISIONAL
        )
    )
    validation = shadow(replacement, "shadow-after-")
    result["replacement_validation"] = validation
    if validation["status"] != "passed":
        publish(
            registry.transition(
                "issued-response", new_version.version, TheoryStatus.REJECTED
            )
        )
        result["stop_reason"] = "replacement_validation_failed"
        return result
    publish(
        registry.transition(
            "issued-response", new_version.version, TheoryStatus.ACCEPTED
        )
    )
    publish(registry.transition("issued-response", 1, TheoryStatus.SUPERSEDED))
    result["status"], result["stop_reason"] = "conditional_joint_theory_revised", None
    result["independent_validation_completed"] = True
    result["replacement_independent_validation_completed"] = True
    return result


def discover_structural_repair(
    view,
    bootstrap,
    chosen,
    anomaly_actions,
    anomaly_outputs,
    anomaly_evidence_ids,
    costs,
    *,
    method="witness_cover",
    seed=20261008,
    budget=400,
    horizon=5,
    search_budget=1000,
    candidate_limit=512,
    validation_reserve=48,
):
    """Generate/identify a conditional edit-family candidate; do not promote it.

    Used by the legacy scientific loop and the matched shared lifecycle. Received
    anomaly evidence was paid in the current world; no private replacement table
    or oracle is accepted. Independent final validation belongs to the caller.
    """
    if (
        method not in ("witness_cover", "sequence_ig")
        or type(seed) is not int
        or seed < 0
        or any(
            type(v) is not int or v < 1
            for v in (budget, search_budget, candidate_limit)
        )
        or type(horizon) is not int
        or not 1 <= horizon <= 5
        or candidate_limit > 512
        or type(validation_reserve) is not int
        or validation_reserve < 0
        or type(bootstrap) is not tuple
        or not 1 <= len(bootstrap) <= 12
        or not isinstance(chosen, JointHypothesis)
        or type(anomaly_actions) is not tuple
        or not 1 <= len(anomaly_actions) <= horizon
        or type(anomaly_outputs) is not tuple
        or len(anomaly_outputs) != len(anomaly_actions)
        or type(anomaly_evidence_ids) is not tuple
        or not anomaly_evidence_ids
        or any(type(q) is not str or not q for q in anomaly_evidence_ids)
    ):
        raise ValueError("invalid bounded structural repair context")
    bootstrap = tuple(JointHypothesis.model_validate(h.model_dump()) for h in bootstrap)
    chosen = JointHypothesis.model_validate(chosen.model_dump())
    if chosen not in bootstrap:
        raise ValueError("structural anchor is not a supplied public hypothesis")
    candidates = validate_hypotheses(
        bootstrap, bootstrap[0].mechanism_id, bootstrap[0].controller_id
    )
    program = FiniteStateProgram.from_ir(candidates[0])
    validate_costs(program.actions, costs)
    costs = dict(costs)
    if any(
        type(a) is not str or a not in program.actions for a in anomaly_actions
    ) or any(type(o) is not str or o not in program.outputs for o in anomaly_outputs):
        raise ValueError("structural anomaly outside the public alphabet")
    result = {
        "schema": "structural-repair-discovery-v1",
        "status": "unresolved",
        "spent": 0,
        "stop_reason": None,
        "generation": None,
        "discovery": None,
        "candidate_ir": None,
        "mechanism_change_warranted": False,
        "mechanism_attribution_status": "unresolved",
        "unique_causal_structure_claimed": False,
    }
    search = enumerate_structural_edits(
        chosen.mechanism_ir, evaluation_budget=search_budget
    )
    result["generation"] = {
        "parent_mechanism_ir": chosen.mechanism_ir.model_dump(mode="json"),
        "proposals": [proposal.as_json() for proposal in search.proposals],
        "evaluations": search.evaluations,
        "complete": search.complete,
        "stop_reason": search.stop_reason,
        "discovery_evidence_ids": list(anomaly_evidence_ids),
    }
    if not search.complete:
        result["stop_reason"] = "search_budget_exhausted"
        return result
    controllers = {
        hypothesis.controller_id: hypothesis.controller_ir for hypothesis in bootstrap
    }
    joint = tuple(
        JointHypothesis(
            mechanism_id=proposal.proposal_id,
            controller_id=cid,
            mechanism_ir=proposal.ir,
            controller_ir=controller,
        )
        for proposal in search.proposals
        for cid, controller in sorted(controllers.items())
    )
    actual = {"outputs": list(anomaly_outputs)}
    anomaly_word = anomaly_actions
    programs = tuple(_bounded(hypothesis.compile(), horizon) for hypothesis in joint)
    admitted = tuple(
        index
        for index, ir in enumerate(programs)
        if (
            FiniteStateProgram.from_ir(ir).execute(anomaly_word)[0]
            == tuple(actual["outputs"])
        )
    )
    result["generation"]["admitted_joint_indices"] = list(admitted)
    if not admitted:
        result["status"], result["stop_reason"] = (
            "model_mismatch",
            "no_grammar_candidate_fits",
        )
        return result
    if len(admitted) > candidate_limit:
        result["stop_reason"] = "planning_capacity_exceeded"
        return result
    reserve = validation_reserve
    if (budget - result["spent"]) <= reserve:
        result["stop_reason"] = "discovery_budget_exhausted"
        return result
    discovery = identify_mechanism(
        view,
        tuple(programs[i] for i in admitted),
        anomaly_word,
        costs,
        method=method,
        seed=seed,
        budget=(budget - result["spent"]) - reserve,
        horizon=horizon,
        candidate_limit=candidate_limit,
        equivalence_horizon=horizon,
        query_prefix="revision-query-",
    )
    result["discovery"] = discovery
    result["spent"] += discovery["spent"]
    if discovery["status"] != "conditional_behavior_identified":
        result["stop_reason"] = "structural_discovery_unresolved"
        return result
    survivors = tuple(admitted[index] for index in discovery["remaining_candidates"])
    baseline_id = search.proposals[0].proposal_id
    result["mechanism_change_warranted"] = all(
        joint[index].mechanism_id != baseline_id for index in survivors
    )
    result["possible_mechanisms"] = sorted(
        {joint[index].mechanism_id for index in survivors}
    )
    result["possible_controllers"] = sorted(
        {joint[index].controller_id for index in survivors}
    )
    result["mechanism_attribution_status"] = (
        "changed"
        if result["mechanism_change_warranted"]
        else "unchanged"
        if result["possible_mechanisms"] == [baseline_id]
        else "ambiguous"
    )
    representative = min(
        survivors,
        key=lambda index: (
            joint[index].mechanism_id != baseline_id,
            len(
                next(
                    proposal.edits
                    for proposal in search.proposals
                    if proposal.proposal_id == joint[index].mechanism_id
                )
            ),
            TheoryInterpreter.description_length(programs[index]),
            index,
        ),
    )
    result["candidate_ir"] = programs[representative].model_dump(mode="json")
    result["status"], result["stop_reason"] = "provisional_structure_identified", None
    return result
