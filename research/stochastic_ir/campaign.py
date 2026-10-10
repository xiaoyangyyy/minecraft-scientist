"""Real execution checks with supplied models; not a discovery Agent benchmark."""

from pathlib import Path

from research.compact_campaigns.store import Writer
from research.compact_campaigns.structural import hashes as base_hashes
from research.structural_campaigns.corpus import digest

from .audit import audit_trace
from .fixture import fixture
from .program import StochasticProgram

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "Staged shared-TheoryIR stochastic semantics and supplied-model component execution only; not unknown-controller discovery, noise calibration, active experiments, validated revision, novelty or full H1/H2/H3/H4/Minecraft/RGB acceptance."


def hashes():
    paths = sorted((ROOT / "research" / "stochastic_ir").glob("*.py"))
    paths += [
        ROOT / "research" / "stochastic_ir" / "README.md",
        ROOT / "docs" / "stochastic-ir-contract.md",
        ROOT / "scripts" / "check_stochastic_semantics.py",
    ]
    return base_hashes() | {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def mass_dict(mass):
    return {"numerator": mass.numerator, "denominator": mass.denominator}


def rollout(view, ir, dispatch, ledger):
    """Only capabilities and a declared supplied model; no world/spec argument."""
    model, belief = StochasticProgram.from_ir(ir), None
    for entry in dispatch:
        if entry == ("reset",):
            observation = view.begin_trial()
            belief = model.initial_belief()
            ledger.append(
                {
                    "kind": "reset",
                    "observation": observation.model_dump(mode="json"),
                    "belief": {s: mass_dict(p) for s, p in sorted(belief.items())},
                }
            )
        else:
            command = entry[1]
            prediction = model.predict_step(command, belief=belief)
            frozen = {
                o: mass_dict(sum(states.values())) for o, states in prediction.items()
            }
            item = {
                "kind": "step",
                "issued_command": command,
                "prediction_before_dispatch": frozen,
            }
            ledger.append(item)
            action = next(a for a in view.available_actions() if a.target_id == command)
            record = view.step(action)
            item["transition"] = record.model_dump(mode="json")
            likelihood, belief = model.filter(
                command, record.event.details["signal"], belief=belief
            )
            item["observed_likelihood"] = mass_dict(likelihood)
            item["posterior_belief"] = {
                s: mass_dict(p) for s, p in sorted(belief.items())
            }


def run(*, output, seed=0):
    if type(seed) is not int or not 0 <= seed < 2**64 - 1:
        raise ValueError("campaign seed must allow two bounded adjacent seeds")
    if Path(output).exists():
        raise FileExistsError("never overwrite a stochastic execution run")
    sources = hashes()
    writer = Writer(output)
    registration = {
        "schema": "stochastic-semantics-registration-v1",
        "source_sha256": sources,
        "seed": seed,
        "conditions": [
            {"seed": seed + i, "noise": noise}
            for i in range(2)
            for noise in (False, True)
        ],
        "known_model_supplied": True,
        "unit_costs": {"reset": 1, "atomic_command": 1},
        "scope": SCOPE,
    }
    writer.write("registration.json", registration)
    rows = []
    for condition in registration["conditions"]:
        world, ir, actions, _ = fixture(**condition)
        dispatch = tuple(
            entry
            for _ in range(3)
            for entry in (("reset",), *(("step", a) for a in actions * 4))
        )
        stem = f"seed-{condition['seed']}-noise-{int(condition['noise'])}"
        writer.write(
            f"{stem}/config.json",
            {
                "condition": condition,
                "frozen_dispatch": dispatch,
                "supplied_theory": ir.model_dump(mode="json"),
                "scope": SCOPE,
            },
        )
        ledger = []
        try:
            rollout(world.agent_view(), ir, dispatch, ledger)
            # Freeze public evidence before evaluator-owned truth is obtained.
            writer.write(f"{stem}/events.json", ledger)
            private, trace = world.evaluator_spec(), world.evaluator_trace()
            writer.write(f"{stem}/private.json", private)
            writer.write(f"{stem}/raw-trace.json", trace)
            audit = audit_trace(private, trace, expected_dispatch=dispatch)
            writer.write(f"{stem}/audit.json", audit)
            rows.append(
                {
                    **condition,
                    "resets": audit["resets"],
                    "atomic_actions": audit["atomic_actions"],
                    "actual_unit_cost": audit["resets"] + audit["atomic_actions"],
                    "model_discovery_score": None,
                    "audit_valid": audit["valid"],
                }
            )
        except BaseException as error:
            # Keep partial issued actions/actual receipts and raw private evidence;
            # no finish() means no completion manifest, including source drift.
            if writer.failed:
                # A storage failure cannot be repaired with more writes. Preserve
                # already written/partial files and the original exception.
                raise
            if f"{stem}/events.json" not in writer.artifacts:
                writer.write(f"{stem}/events.json", ledger)
            if f"{stem}/private.json" not in writer.artifacts:
                writer.write(f"{stem}/private.json", world.evaluator_spec())
            if f"{stem}/raw-trace.json" not in writer.artifacts:
                writer.write(f"{stem}/raw-trace.json", world.evaluator_trace())
            writer.write(
                "failure.json",
                {
                    "condition": condition,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "completed_claim": False,
                },
            )
            raise
    if hashes() != sources:
        writer.write(
            "failure.json", {"error_type": "source_drift", "completed_claim": False}
        )
        raise ValueError("stochastic source drift; no completion certificate")
    report = {
        "schema": "stochastic-semantics-report-v1",
        "registration": registration,
        "rows": rows,
        "scope": SCOPE,
        "main_interpreter_integration_completed": False,
        "unknown_controller_discovery_completed": False,
        "probability_calibration_claim": False,
    }
    writer.write("report.json", report)
    writer.finish()
    return report
