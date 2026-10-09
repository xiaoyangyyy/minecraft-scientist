"""Raw edit/factor/query checks for a shared finite promotion protocol."""

from audit_random_wp import check_random_wp_plan
from audit_run import canonical
from audit_transfer import replay_capability
from matched_repair import learn_matched_repair
from paid_transfer import METHODS

from minecraft_scientist.discovery.diagnosis import JointHypothesis
from minecraft_scientist.evaluation.bootstrap import check_identification, coverage_plan
from minecraft_scientist.evaluation.diagnosis import (
    _factor_rollout,
    _matches_factors,
    _public_spec,
)
from minecraft_scientist.evaluation.revision import _raw_edit


def check_matched_repair_public(phase, check_phase, check_query):
    config, result = phase["config"], phase["run"]
    p, policy = config["protocol"], config["policy"]
    if (
        result["schema"] != "matched-repair-run-v1"
        or result["method"] != config["method"]
    ):
        raise ValueError("independent matched repair identity mismatch")
    events, spent, candidate = [], 0, None
    base, structural = result["base_run"], result["structural"]
    if config["method"] in METHODS:
        if structural is not None:
            raise ValueError("baseline received hidden structural search")
        if base is not None:
            check_phase({"config": result["base_config"], "run": base})
            events.extend(base["events"])
            spent += base["spent"]
            if base["status"] == "provisional_conformance_passed":
                candidate = base["candidate_ir"]
    else:
        if base is not None or result["base_config"] is not None:
            raise ValueError("structural method received hidden baseline queries")
        if structural is not None:
            joint = tuple(
                JointHypothesis.model_validate(h)
                for h in config["family"]["hypotheses"]
            )
            anchor = joint[config["anchor_index"]]
            generation = structural["generation"]
            if canonical(generation["parent_mechanism_ir"]) != canonical(
                anchor.mechanism_ir.model_dump(mode="json")
            ) or generation["discovery_evidence_ids"] != [
                config["anomaly"]["query_id"]
            ]:
                raise ValueError("structural search anchor/anomaly provenance mismatch")
            controllers = {h.controller_id: h.controller_ir for h in joint}
            specs, programs = [], []
            for proposal in generation["proposals"]:
                if canonical(
                    _raw_edit(anchor.mechanism_ir.parameters, proposal["edits"])
                ) != canonical(proposal["ir"]["parameters"]):
                    raise ValueError("independent matched raw edit mismatch")
                for cid, controller in sorted(controllers.items()):
                    h = JointHypothesis(
                        mechanism_id=proposal["proposal_id"],
                        controller_id=cid,
                        mechanism_ir=proposal["ir"],
                        controller_ir=controller,
                    )
                    spec, program = _public_spec(h), h.compile().parameters
                    if not _matches_factors(program, spec):
                        raise ValueError("independent matched factor compiler mismatch")
                    specs.append(spec)
                    programs.append(program)
            if generation["complete"]:
                admitted = [
                    i
                    for i, s in enumerate(specs)
                    if list(_factor_rollout(s, config["anomaly"]["actions"]))
                    == config["anomaly"]["observed"]
                ]
                if admitted != generation["admitted_joint_indices"]:
                    raise ValueError(
                        "independent raw anomaly candidate admission mismatch"
                    )
            discovery = structural["discovery"]
            if discovery is not None:
                admitted = generation["admitted_joint_indices"]
                check_identification(
                    discovery,
                    [programs[i] for i in admitted],
                    config["costs"],
                    specs=[specs[i] for i in admitted],
                    horizon=p["horizon"],
                )
                events.extend(discovery["events"])
                spent += discovery["spent"]
            if type(structural["spent"]) is not int or structural["spent"] != spent:
                raise ValueError("independent structural discovery costs mismatch")
            if structural["status"] == "provisional_structure_identified":
                candidate = structural["candidate_ir"]
                survivors = [
                    generation["admitted_joint_indices"][i]
                    for i in discovery["remaining_candidates"]
                ]
                if not any(
                    canonical(candidate["parameters"]) == canonical(programs[i])
                    for i in survivors
                ):
                    raise ValueError(
                        "structural candidate outside surviving raw family"
                    )
                if (
                    candidate["scope"]
                    != {
                        "language": "total-mealy-v1",
                        "reset": "both-initial-states",
                        "treatment": "issued-command; delivery-latent",
                        "intervention_horizon": p["horizon"],
                        "certified_reset_only": True,
                        "claim": "conditional behavior on reset queries, not unique causal structure",
                    }
                    or candidate["latent_variables"]
                    != ["mechanism_state", "controller_state"]
                    or candidate["effects"] != {"output": "transition.output"}
                    or candidate["preconditions"]
                ):
                    raise ValueError(
                        "independent structural candidate reset/causal scope mismatch"
                    )
    batch, coverage = result["random_validation"], result["coverage_validation"]
    final_ids, counterexamples, final_passed = [], [], False
    if batch is not None:
        if candidate is None:
            raise ValueError("matched validation without executable candidate")
        plan = batch["plan"]
        check_random_wp_plan(plan, config["costs"])
        if (
            canonical(plan["hypothesis_program"]) != canonical(candidate["parameters"])
            or plan["seed"] != p["seed"] + 500_000
            or plan["horizon"] != p["horizon"]
            or plan["requested_count"] != p["conformance_count"]
        ):
            raise ValueError("matched final frozen candidate/tester mismatch")
        full, affordable = (
            len(plan["queries"]) == p["conformance_count"],
            plan["total_cost"] <= p["budget"] - spent,
        )
        if (not full or not affordable) and batch["queries"]:
            raise ValueError("matched partial dispatch of unpayable batch")
        if len(batch["queries"]) > len(plan["queries"]):
            raise ValueError("matched final queries exceed frozen batch")
        terminal, expected_status = False, "passed"
        for i, q in enumerate(batch["queries"]):
            if terminal:
                raise ValueError("matched final continued after failure/counterexample")
            outputs, failed = check_query(
                config,
                q,
                plan["queries"][i],
                index=0,
                sample_index=i,
                remaining=p["budget"] - spent,
                epoch="repair-epoch",
                qid=f"repair-check-{i}-0",
                reason="matched-validation",
                reset_episode=q["events"][1]["observation"]["episode_id"],
            )
            spent += q["spent"]
            events.extend(q["events"])
            final_ids.extend(
                e["query_id"] for e in q["events"] if e["kind"] == "evidence"
            )
            if failed or outputs != plan["queries"][i]["prediction"]:
                if not failed:
                    counterexamples.extend(
                        e["query_id"] for e in q["events"] if e["kind"] == "evidence"
                    )
                terminal, expected_status = (
                    True,
                    "stopped" if failed else "counterexample",
                )
        if not full:
            expected_status = "insufficient_samples"
        elif not affordable:
            expected_status = "budget_exhausted"
        elif not terminal and len(batch["queries"]) != p["conformance_count"]:
            raise ValueError("matched incomplete final batch claimed passed")
        if batch["status"] != expected_status:
            raise ValueError("matched independent final batch status mismatch")
        final_passed = expected_status == "passed"
    if coverage is not None:
        if not final_passed:
            raise ValueError("coverage validation after failed random protocol")
        used = {tuple(e["actions"]) for e in events if e["kind"] == "plan"}
        plan = coverage_plan(
            candidate,
            config["costs"],
            used,
            count=policy["shadow_count"],
            horizon=p["horizon"],
            seed=p["seed"] + 500_001,
        )
        if canonical(plan) != canonical(coverage["plan"]):
            raise ValueError("independent matched prospective coverage mismatch")
        if (
            len(plan["queries"]) != policy["shadow_count"]
            or plan["total_cost"] > p["budget"] - spent
        ) and coverage["runs"]:
            raise ValueError("matched unaffordable coverage partially executed")
        paid, terminal = 0, False
        for i, run in enumerate(coverage["runs"]):
            if (
                terminal
                or i >= policy["shadow_count"]
                or run["budget"] != p["budget"] - spent - paid
            ):
                raise ValueError("matched coverage order/remaining budget mismatch")
            check_identification(
                run,
                [candidate["parameters"]],
                config["costs"],
                horizon=p["horizon"],
                equivalence=False,
            )
            plans = [e for e in run["events"] if e["kind"] == "plan"]
            if len(plans) != 1 or plans[0]["actions"] != plan["queries"][i]["actions"]:
                raise ValueError("matched coverage dispatched unplanned word")
            paid += run["spent"]
            events.extend(run["events"])
            final_ids.extend(
                e["query_id"] for e in run["events"] if e["kind"] == "evidence"
            )
            terminal = run["status"] != "conditional_behavior_identified"
            if run["status"] == "model_mismatch":
                counterexamples.extend(
                    e["query_id"] for e in run["events"] if e["kind"] == "evidence"
                )
        if type(coverage["spent"]) is not int or coverage["spent"] != paid:
            raise ValueError("matched coverage actual paid mismatch")
        if coverage["status"] == "passed" and (
            terminal or len(coverage["runs"]) != policy["shadow_count"]
        ):
            raise ValueError("matched incomplete/failed coverage accepted")
        spent += paid
    accepted = final_passed and coverage is not None and coverage["status"] == "passed"
    if (
        canonical(events) != canonical(result["events"])
        or type(result["spent"]) is not int
        or spent != result["spent"]
        or spent > p["budget"]
        or canonical(result["candidate_ir"])
        != canonical(candidate if accepted else None)
        or canonical(result["validation_query_ids"]) != canonical(final_ids)
        or canonical(result["validation_counterexample_ids"])
        != canonical(counterexamples)
        or canonical(result["proposed_ir"]) != canonical(candidate)
        or result["status"]
        != ("provisional_matched_validation_passed" if accepted else "unresolved")
    ):
        raise ValueError("matched publication/flattened evidence/paid cost mismatch")
    if accepted and (
        len(set(final_ids)) != p["conformance_count"] + policy["shadow_count"]
        or set(final_ids)
        & {
            e["query_id"]
            for e in (
                base["events"]
                if base
                else (
                    structural["discovery"]["events"]
                    if structural and structural["discovery"]
                    else []
                )
            )
            if e["kind"] == "evidence"
        }
    ):
        raise ValueError("matched generation/final validation IDs overlap")
    view, consumed = replay_capability(result)
    if (
        canonical(learn_matched_repair(view, config)) != canonical(result)
        or not consumed()
    ):
        raise ValueError("complete matched discovery/validation policy replay mismatch")
