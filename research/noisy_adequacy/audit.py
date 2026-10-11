"""Independent unnormalized raw-IR likelihood/monitor ledger reconstruction.

No monitor class/update, stochastic prediction/filter or Bayes weight update.
Reconstruct every frozen distribution, cumulative likelihood, reset, e-value,
running maximum, sticky rejection and first actual cost from ALL public controls.
IR schema validation and the fixed scope text are shared, not likelihood code.
"""

from fractions import Fraction

from minecraft_scientist.protocols import TheoryIR
from research.noisy_competition.audit import canonical
from research.stochastic_ir.program import StochasticProgram

from .monitor import SCOPE

MAX_STEPS = 256
MAX_BITS = 12000
MAX_BRANCHES = 2_000_000


def encoded(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ValueError("raw monitor coefficient audit bound")
    return {"numerator": value.numerator, "denominator": value.denominator}


def value_record(value):
    return (
        {"kind": "infinite"}
        if value is None
        else {"kind": "finite", "value": encoded(value)}
    )


def verify_monitor(definitions, config, result, *, alpha=Fraction(1, 20)):
    if (
        type(alpha) is not Fraction
        or not 0 < alpha < 1
        or alpha.denominator > 1_000_000
    ):
        raise ValueError("caller-pinned strict alpha required")
    if (
        type(result) is not dict
        or set(result)
        != {
            "schema",
            "agent",
            "monitor",
            "monitor_evidence",
            "first_rejection_actual_cost",
            "planner_changed",
            "family_rejection_is_structure_attribution",
            "accepted_theory",
        }
        or result["schema"] != "online-shadow-family-adequacy-agent-v1"
        or result["planner_changed"] is not False
        or result["family_rejection_is_structure_attribution"] is not False
        or result["accepted_theory"] is not None
    ):
        raise ValueError("invalid shadow-monitor claim/schema")
    if (
        type(definitions) is not list
        or not 1 <= len(definitions) <= 32
        or canonical(definitions) != canonical(result["agent"]["candidate_definitions"])
    ):
        raise ValueError("public fixed model definitions mismatch")
    programs = [
        StochasticProgram.from_ir(TheoryIR.model_validate(d["joint_ir"]))
        for d in definitions
    ]
    if any(
        p.actions != programs[0].actions or p.outputs != programs[0].outputs
        for p in programs
    ):
        raise ValueError("raw monitor requires same public alphabets")
    specs = [p.model_dump(mode="json") for p in programs]
    if (
        type(config.get("costs")) is not dict
        or set(config["costs"]) != set(programs[0].actions)
        or any(
            type(v) is not int or not 1 <= v <= 100 for v in config["costs"].values()
        )
        or type(config.get("budget")) is not int
        or not 1 <= config["budget"] <= 1000
    ):
        raise ValueError("caller-pinned cost/budget contract mismatch")
    keys = [d["key"] for d in definitions]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate public null keys")
    # Unnormalized vectors carry the ENTIRE likelihood across trial boundaries.
    states = [
        {
            e["state"]: Fraction(
                e["probability"]["numerator"], e["probability"]["denominator"]
            )
            for e in spec["initial"]
        }
        for spec in specs
    ]
    alternative, current_e, max_e = Fraction(1), Fraction(1), Fraction(1)
    steps, resets, branches, first_step, actual_cost, first_cost = (
        0,
        0,
        0,
        None,
        0,
        None,
    )

    def snapshot():
        likelihoods = [sum(s.values(), Fraction(0)) for s in states]
        return {
            "model_keys": keys,
            "alpha": encoded(alpha),
            "alternative": "uniform-public-output-v1",
            "steps": steps,
            "resets": resets,
            "prediction_branches": branches,
            "model_likelihoods": [encoded(p) for p in likelihoods],
            "alternative_likelihood": encoded(alternative),
            "maximum_null_likelihood": encoded(max(likelihoods)),
            "e_value": value_record(current_e),
            "running_max_e_value": value_record(max_e),
            "family_rejected": first_step is not None,
            "first_rejection_step": first_step,
            "scope": SCOPE,
        }

    controls = [
        e for e in result["agent"]["evidence"] if e["kind"] in ("reset", "step")
    ]
    ledger = result["monitor_evidence"]
    if type(ledger) is not list or len(ledger) != len(controls) or not ledger:
        raise ValueError("monitor omitted/added actual controls")
    for entry, control in zip(ledger, controls, strict=True):
        if type(control["cost"]) is not int or control["cost"] < 1:
            raise ValueError("actual control cost type mismatch")
        if (
            type(entry) is not dict
            or entry.get("kind") != control["kind"]
            or canonical(entry.get("before")) != canonical(snapshot())
        ):
            raise ValueError("raw monitor prior snapshot/order mismatch")
        before = snapshot()
        actual_cost += control["cost"]
        if control["kind"] == "reset":
            if control["cost"] != 1:
                raise ValueError("reset cost mismatch")
            states = [
                {
                    e["state"]: sum(s.values(), Fraction(0))
                    * Fraction(
                        e["probability"]["numerator"], e["probability"]["denominator"]
                    )
                    for e in spec["initial"]
                }
                for spec, s in zip(specs, states, strict=True)
            ]
            resets += 1
            expected = {
                "kind": "reset",
                "before": before,
                "observation": control["observation"],
                "after": snapshot(),
            }
        else:
            action, output = (
                control["issued_command"],
                control["transition"]["event"]["details"]["signal"],
            )
            if (
                action not in programs[0].actions
                or output not in programs[0].outputs
                or control["cost"] != config["costs"][action]
                or steps >= MAX_STEPS
            ):
                raise ValueError("raw monitor public command/output/cost/step bound")
            predicted, following = [], []
            for key, spec, vector in zip(keys, specs, states, strict=True):
                likelihood = sum(vector.values(), Fraction(0))
                masses, next_states = {}, {}
                for row in spec["transitions"]:
                    if row["action"] == action and vector.get(
                        row["source"], Fraction(0)
                    ):
                        branches += 1
                        mass = vector[row["source"]] * Fraction(
                            row["probability"]["numerator"],
                            row["probability"]["denominator"],
                        )
                        masses[row["output"]] = (
                            masses.get(row["output"], Fraction(0)) + mass
                        )
                        if row["output"] == output:
                            next_states[row["target"]] = (
                                next_states.get(row["target"], Fraction(0)) + mass
                            )
                predicted.append(
                    {
                        "key": key,
                        "null_prefix_possible": bool(likelihood),
                        "output_distribution": {
                            o: encoded(p / likelihood)
                            for o, p in sorted(masses.items())
                        }
                        if likelihood
                        else {},
                    }
                )
                following.append(next_states)
            if branches > MAX_BRANCHES:
                raise ValueError("raw monitor branch audit bound")
            proposal = {
                "atomic_index": steps + 1,
                "issued_command": action,
                "alternative_probability": encoded(
                    Fraction(1, len(programs[0].outputs))
                ),
                "component_predictions": predicted,
            }
            states, alternative = following, alternative / len(programs[0].outputs)
            maximum = max(sum(v.values(), Fraction(0)) for v in states)
            current_e = alternative / maximum if maximum else None
            max_e = (
                None if max_e is None or current_e is None else max(max_e, current_e)
            )
            steps += 1
            if first_step is None and (current_e is None or current_e >= 1 / alpha):
                first_step, first_cost = steps, actual_cost
            expected = {
                "kind": "step",
                "before": before,
                "prediction_before_dispatch": proposal,
                "transition": control["transition"],
                "after": snapshot(),
            }
        if canonical(entry) != canonical(expected):
            raise ValueError(
                "raw monitor likelihood/prediction/reset/rejection mismatch"
            )
    if (
        canonical(result["monitor"]) != canonical(snapshot())
        or type(result["agent"]["spent"]) is not int
        or result["agent"]["spent"] != actual_cost
        or actual_cost > config["budget"]
        or canonical(result["first_rejection_actual_cost"]) != canonical(first_cost)
    ):
        raise ValueError("raw final monitor/actual cost coverage mismatch")
    return {
        "valid": True,
        "atomic_actions": steps,
        "resets": resets,
        "actual_cost": actual_cost,
        "prediction_branches": branches,
        "family_rejected": first_step is not None,
        "first_rejection_actual_cost": first_cost,
        "scope": "Independent fixed-public-model raw likelihood/monitor audit, not an actual-world type-I rate or structure-revision certification",
    }
