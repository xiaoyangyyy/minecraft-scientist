"""Full declared finite-family development matrix, not confirmatory statistics."""

import math
from fractions import Fraction
from pathlib import Path
from time import perf_counter, process_time

from research.compact_campaigns.store import Reader, Writer
from research.stochastic_ir.audit import audit_trace
from research.stochastic_ir.campaign import hashes as upstream_hashes
from research.structural_campaigns.corpus import digest

from .agent import run_agent
from .audit import canonical, verify_public
from .fixture import CONDITIONS, make_world, public_family
from .planning import POLICIES

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "Finite declared 18-model stochastic family/prior development with stationary private conditions; public adaptive-policy and independent raw replay, not unknown-structure discovery, online revision, posterior calibration, formal holdout/statistical/novelty/H4/Minecraft/RGB acceptance."


def hashes():
    paths = sorted((ROOT / "research" / "noisy_competition").glob("*.py"))
    paths += [
        ROOT / "research" / "noisy_competition" / "README.md",
        ROOT / "docs" / "noisy-competition-contract.md",
        ROOT / "scripts" / "run_noisy_competition.py",
    ]
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in paths
    }


def selection(seed):
    if type(seed) is not int or not 0 <= seed < 2**64 - 1:
        raise ValueError("two bounded adjacent seeds required")
    return [
        {
            "seed": seed + i,
            "condition": condition,
            "policy": policy,
            "stem": f"seed-{seed + i}/{condition}/{policy}",
        }
        for i in range(2)
        for condition in CONDITIONS
        for policy in POLICIES
    ]


def support(result, truth):
    endpoints = {}
    threshold = Fraction(9, 10)
    for factor, values in result["factors"].items():
        masses = {
            key: Fraction(v["numerator"], v["denominator"]) for key, v in values.items()
        }
        peak = max(masses.values())
        top = [key for key, probability in masses.items() if probability == peak]
        supported = peak >= threshold and len(top) == 1
        endpoints[factor] = {
            "top_ids": top,
            "peak": {"numerator": peak.numerator, "denominator": peak.denominator},
            "prior_conditional_support": supported,
            "support_correct_if_any": top[0] == truth[f"{factor}_id"]
            if supported
            else None,
            "true_factor_mass": values.get(truth[f"{factor}_id"]),
            "scope": "0.9 conditional posterior support criterion; not frequentist confidence, identifiability or theory acceptance",
        }
    return endpoints


def make_row(item, result, public_audit, raw_audit, truth):
    if (
        public_audit["atomic_actions"] != raw_audit["atomic_actions"]
        or public_audit["resets"] != raw_audit["resets"]
    ):
        raise ValueError("public/raw actual dispatch count mismatch")
    return {
        "seed": item["seed"],
        "condition": item["condition"],
        "policy": item["policy"],
        "status": result["status"],
        "actual_cost": public_audit["actual_cost"],
        "atomic_actions": raw_audit["atomic_actions"],
        "resets": raw_audit["resets"],
        "kernel_calls": result["posterior"]["kernel_calls"],
        "in_public_family": truth["in_public_family"],
        "factor_support": support(result, truth),
        "accepted_theory": None,
    }


def registration_spec(seed, budget, horizon, sources):
    return {
        "schema": "finite-family-noisy-registration-v1",
        "source_sha256": sources,
        "seed": seed,
        "budget": budget,
        "horizon": horizon,
        "selection": selection(seed),
        "policies": list(POLICIES),
        "prior": "equal mass per declared factor combination",
        "support_threshold": {"numerator": 9, "denominator": 10},
        "reset_cost": 1,
        "scope": SCOPE,
    }


def run(*, output, seed=0, budget=96, horizon=4):
    items = selection(seed)
    if (
        type(budget) is not int
        or not 1 <= budget <= 1000
        or type(horizon) is not int
        or not 1 <= horizon <= 4
    ):
        raise ValueError("invalid campaign budget/horizon")
    sources = hashes()
    writer = Writer(output)
    registration = registration_spec(seed, budget, horizon, sources)
    writer.write("registration.json", registration)
    timings = []
    for item in items:
        world, candidates, costs, truth = make_world(item["seed"], item["condition"])
        config = {
            "seed": item["seed"],
            "policy": item["policy"],
            "budget": budget,
            "horizon": horizon,
            "costs": costs,
        }
        stem, ledger = item["stem"], []
        writer.write(f"{stem}/config.json", config)
        wall, cpu = perf_counter(), process_time()
        try:
            result = run_agent(world.agent_view(), candidates, **config, ledger=ledger)
            policy_timing = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            writer.write(f"{stem}/run.json", result)
            # Actual public trajectory is immutable before obtaining truth; keep
            # private evidence, but defer ALL private audits until ALL public
            # policy replays have finished for the complete matrix.
            writer.write(
                f"{stem}/private.json",
                {"world": world.evaluator_spec(), "truth": truth},
            )
            writer.write(f"{stem}/raw-trace.json", world.evaluator_trace())
        except BaseException as error:
            if not writer.failed:
                writer.write(f"{stem}/partial-events.json", ledger)
                writer.write(
                    f"{stem}/partial-private.json",
                    {"world": world.evaluator_spec(), "truth": truth},
                )
                writer.write(f"{stem}/partial-raw-trace.json", world.evaluator_trace())
                writer.write(
                    "failure.json",
                    {
                        "item": item,
                        "type": type(error).__name__,
                        "message": str(error),
                        "completed_claim": False,
                    },
                )
            raise
        timings.append(
            {
                "stem": stem,
                "actual_policy": policy_timing,
            }
        )
        print(f"completed noisy family policy {stem}", flush=True)
    public_checks = {}
    for item in items:
        stem = item["stem"]
        public_checks[stem] = verify_public(
            public_family(item["seed"])[0],
            writer.read(f"{stem}/config.json"),
            writer.read(f"{stem}/run.json"),
        )
    rows = []
    for item in items:
        stem = item["stem"]
        private = writer.read(f"{stem}/private.json")
        raw = audit_trace(
            private["world"],
            writer.read(f"{stem}/raw-trace.json"),
            expected_dispatch=public_checks[stem]["dispatch"],
        )
        writer.write(f"{stem}/audit.json", {"public": public_checks[stem], "raw": raw})
        rows.append(
            make_row(
                item,
                writer.read(f"{stem}/run.json"),
                public_checks[stem],
                raw,
                private["truth"],
            )
        )
    if hashes() != sources:
        writer.write("failure.json", {"type": "source_drift", "completed_claim": False})
        raise ValueError("noisy source drift; no final certificate")
    report = {
        "schema": "finite-family-noisy-report-v1",
        "registration": registration,
        "rows": rows,
        "scope": SCOPE,
        "accepted_theories": None,
        "unmodeled_positive_likelihood_sensor_is_not_guaranteed_detectable": True,
    }
    writer.write("report.json", report)
    writer.write(
        "timing.json",
        {
            "rows": timings,
            "scope": "actual policy compute only; public replay/raw audit/storage/total orchestration not included",
        },
        nondeterministic=True,
    )
    writer.finish()
    return report


def verify_output(directory, *, manifest_sha256):
    reader = Reader(directory, manifest_sha256=manifest_sha256)
    registration = reader.read("registration.json")
    if canonical(registration) != canonical(
        registration_spec(
            registration["seed"],
            registration["budget"],
            registration["horizon"],
            hashes(),
        )
    ):
        raise ValueError("noisy registration/source/complete selection mismatch")
    expected = {"registration.json", "report.json"} | {
        f"{item['stem']}/{leaf}.json"
        for item in registration["selection"]
        for leaf in ("config", "run", "private", "raw-trace", "audit")
    }
    reader.coverage(expected)
    public = {}
    for item in registration["selection"]:
        stem, candidates, costs = (
            item["stem"],
            public_family(item["seed"])[0],
            {a: 1 for a in public_family(item["seed"])[1]},
        )
        config = {
            "seed": item["seed"],
            "policy": item["policy"],
            "budget": registration["budget"],
            "horizon": registration["horizon"],
            "costs": costs,
        }
        if canonical(reader.read(f"{stem}/config.json")) != canonical(config):
            raise ValueError("condition/truth in Agent inputs or config mismatch")
        public[stem] = verify_public(
            candidates, config, reader.read(f"{stem}/run.json")
        )
    rows = []
    for item in registration["selection"]:
        stem = item["stem"]
        world, _, _, truth = make_world(item["seed"], item["condition"])
        private = reader.read(f"{stem}/private.json")
        if canonical(private) != canonical(
            {"world": world.evaluator_spec(), "truth": truth}
        ):
            raise ValueError("private truth differs from frozen construction")
        raw = audit_trace(
            private["world"],
            reader.read(f"{stem}/raw-trace.json"),
            expected_dispatch=public[stem]["dispatch"],
        )
        if canonical(reader.read(f"{stem}/audit.json")) != canonical(
            {"public": public[stem], "raw": raw}
        ):
            raise ValueError("noisy audit mismatch")
        rows.append(
            make_row(item, reader.read(f"{stem}/run.json"), public[stem], raw, truth)
        )
    report = reader.read("report.json")
    if canonical(report) != canonical(
        {
            "schema": "finite-family-noisy-report-v1",
            "registration": registration,
            "rows": rows,
            "scope": SCOPE,
            "accepted_theories": None,
            "unmodeled_positive_likelihood_sensor_is_not_guaranteed_detectable": True,
        }
    ):
        raise ValueError("noisy report/denominator mismatch")
    timing = reader.nondeterministic("timing.json")
    if [v["stem"] for v in timing["rows"]] != [
        i["stem"] for i in registration["selection"]
    ]:
        raise ValueError("noisy timing coverage mismatch")
    if (
        set(timing) != {"rows", "scope"}
        or timing["scope"]
        != "actual policy compute only; public replay/raw audit/storage/total orchestration not included"
    ):
        raise ValueError("noisy timing scope mismatch")
    for row in timing["rows"]:
        if (
            set(row) != {"stem", "actual_policy"}
            or set(row["actual_policy"]) != {"wall_seconds", "cpu_seconds"}
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in row["actual_policy"].values()
            )
        ):
            raise ValueError("invalid actual-policy timing")
    return {
        "valid": True,
        "runs": len(rows),
        "logical_artifacts": len(expected),
        "scope": SCOPE,
    }
