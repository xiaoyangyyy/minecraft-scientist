"""Full real online shadow matrix; fixed-family rejection, not revised theory."""

import math
from fractions import Fraction
from pathlib import Path
from time import perf_counter, process_time

from research.compact_campaigns.store import Reader, Writer, canonical
from research.noisy_competition.audit import verify_public
from research.noisy_competition.belief import ModelBank
from research.noisy_competition.campaign import make_row as base_row
from research.noisy_competition.campaign import selection
from research.noisy_competition.fixture import make_world, public_family
from research.stochastic_equivalence.campaign import hashes as upstream_hashes
from research.stochastic_ir.audit import audit_trace
from research.structural_campaigns.corpus import digest

from .agent import run_monitored_agent
from .audit import verify_monitor
from .monitor import SCOPE, rational

ROOT = Path(__file__).resolve().parents[2]
ALPHA = Fraction(1, 20)


def hashes():
    paths = sorted((ROOT / "research/noisy_adequacy").glob("*.py"))
    paths += [
        ROOT / "research/noisy_adequacy/README.md",
        ROOT / "docs/noisy-adequacy-contract.md",
        ROOT / "scripts/run_noisy_adequacy.py",
    ]
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in paths
    }


def registration(seed, budget, horizon, sources):
    items = selection(seed)
    if (
        type(budget) is not int
        or not 1 <= budget <= 1000
        or type(horizon) is not int
        or not 1 <= horizon <= 4
    ):
        raise ValueError("strict budget/horizon contract required")
    return {
        "schema": "online-shadow-adequacy-registration-v1",
        "source_sha256": sources,
        "seed": seed,
        "budget": budget,
        "horizon": horizon,
        "selection": items,
        "alpha": rational(ALPHA),
        "alpha_scope": "per-trajectory fixed null family; not simultaneous across all 30 runs, new epochs or added model languages",
        "alternative": "uniform-public-output-v1",
        "planner_feedback": False,
        "changed_prior": False,
        "scope": SCOPE,
        "development_not_confirmatory": True,
    }


def config_for(item, spec, costs):
    return {
        "seed": item["seed"],
        "policy": item["policy"],
        "budget": spec["budget"],
        "horizon": spec["horizon"],
        "costs": costs,
    }


def public_checks(candidates, config, result):
    core = verify_public(candidates, config, result["agent"])
    monitor = verify_monitor(
        ModelBank(candidates).definitions(), config, result, alpha=ALPHA
    )
    if (
        core["actual_cost"] != monitor["actual_cost"]
        or core["atomic_actions"] != monitor["atomic_actions"]
        or core["resets"] != monitor["resets"]
    ):
        raise ValueError("shadow/public actual control mismatch")
    return {"agent": core, "monitor": monitor}


def make_row(item, result, public, raw, truth):
    row = base_row(item, result["agent"], public["agent"], raw, truth)
    rejected = result["monitor"]["family_rejected"]
    return row | {
        "family_rejected": rejected,
        "first_rejection_atomic_index": result["monitor"]["first_rejection_step"],
        "first_rejection_actual_cost": result["first_rejection_actual_cost"],
        "false_rejection_of_in_family": rejected and truth["in_public_family"],
        "rejection_of_out_family": rejected and not truth["in_public_family"],
        "final_e_value": result["monitor"]["e_value"],
        "prediction_branches": result["monitor"]["prediction_branches"],
        "posterior_support_flagged_by_family_rejection": {
            f: rejected and s["prior_conditional_support"]
            for f, s in row["factor_support"].items()
        },
        "theory_revised": False,
        "scope": "online shadow diagnostic; unchanged paid experiments, conditional support retained, no registry promotion/replacement or claimed fault attribution",
    }


def report(spec, rows):
    return {
        "schema": "online-shadow-adequacy-report-v1",
        "registration": spec,
        "rows": rows,
        "scope": SCOPE,
        "accepted_theories": None,
        "registry_changed": False,
        "unknown_structure_discovered": False,
    }


def run(*, output, seed=0, budget=96, horizon=4):
    sources = hashes()
    spec = registration(seed, budget, horizon, sources)
    writer = Writer(output)
    writer.write("registration.json", spec)
    timings = []
    for item in spec["selection"]:
        world, candidates, costs, truth = make_world(item["seed"], item["condition"])
        config = config_for(item, spec, costs)
        stem, agent_ledger, monitor_ledger = item["stem"], [], []
        writer.write(f"{stem}/config.json", config)
        wall, cpu = perf_counter(), process_time()
        try:
            result = run_monitored_agent(
                world.agent_view(),
                candidates,
                **config,
                alpha=ALPHA,
                ledger=agent_ledger,
                monitor_ledger=monitor_ledger,
            )
            actual_timing = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            writer.write(f"{stem}/run.json", result)
            writer.write(
                f"{stem}/private.json",
                {"world": world.evaluator_spec(), "truth": truth},
            )
            writer.write(f"{stem}/raw-trace.json", world.evaluator_trace())
        except BaseException as error:
            if not writer.failed:
                writer.write(f"{stem}/partial-agent-events.json", agent_ledger)
                writer.write(f"{stem}/partial-monitor-events.json", monitor_ledger)
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
        timings.append({"stem": stem, "actual_agent_and_online_monitor": actual_timing})
        print(f"completed online shadow adequacy policy {stem}", flush=True)
    checks = {}
    # ALL public core/independent monitor replays precede ANY selected private
    # raw-world audit/scoring. Monitor results never feed the actual planner.
    for item in spec["selection"]:
        stem = item["stem"]
        checks[stem] = public_checks(
            public_family(item["seed"])[0],
            writer.read(f"{stem}/config.json"),
            writer.read(f"{stem}/run.json"),
        )
    rows = []
    for item in spec["selection"]:
        stem = item["stem"]
        private = writer.read(f"{stem}/private.json")
        raw = audit_trace(
            private["world"],
            writer.read(f"{stem}/raw-trace.json"),
            expected_dispatch=checks[stem]["agent"]["dispatch"],
        )
        writer.write(f"{stem}/audit.json", checks[stem] | {"raw": raw})
        rows.append(
            make_row(
                item,
                writer.read(f"{stem}/run.json"),
                checks[stem],
                raw,
                private["truth"],
            )
        )
    if hashes() != sources:
        writer.write("failure.json", {"type": "source_drift", "completed_claim": False})
        raise ValueError("adequacy source drift; no final certificate")
    result = report(spec, rows)
    writer.write("report.json", result)
    writer.write(
        "timing.json",
        {
            "rows": timings,
            "scope": "actual Agent and online shadow monitor together; storage, public replay, independent audit and orchestration excluded; no isolated marginal monitor timing",
        },
        nondeterministic=True,
    )
    writer.finish()
    return result


def verify_output(directory, *, manifest_sha256):
    reader = Reader(directory, manifest_sha256=manifest_sha256)
    spec = reader.read("registration.json")
    if canonical(spec) != canonical(
        registration(spec["seed"], spec["budget"], spec["horizon"], hashes())
    ):
        raise ValueError("adequacy registration/source/full selection mismatch")
    expected = {"registration.json", "report.json"} | {
        f"{i['stem']}/{leaf}.json"
        for i in spec["selection"]
        for leaf in ("config", "run", "private", "raw-trace", "audit")
    }
    reader.coverage(expected)
    checks = {}
    for item in spec["selection"]:
        stem, candidates, actions = (
            item["stem"],
            public_family(item["seed"])[0],
            public_family(item["seed"])[1],
        )
        config = config_for(item, spec, {a: 1 for a in actions})
        if canonical(reader.read(f"{stem}/config.json")) != canonical(config):
            raise ValueError(
                "private condition/truth in Agent inputs or config mismatch"
            )
        checks[stem] = public_checks(
            candidates, config, reader.read(f"{stem}/run.json")
        )
    rows = []
    for item in spec["selection"]:
        stem = item["stem"]
        world, _, _, truth = make_world(item["seed"], item["condition"])
        private = reader.read(f"{stem}/private.json")
        if canonical(private) != canonical(
            {"world": world.evaluator_spec(), "truth": truth}
        ):
            raise ValueError("private stochastic construction mismatch")
        raw = audit_trace(
            private["world"],
            reader.read(f"{stem}/raw-trace.json"),
            expected_dispatch=checks[stem]["agent"]["dispatch"],
        )
        if canonical(reader.read(f"{stem}/audit.json")) != canonical(
            checks[stem] | {"raw": raw}
        ):
            raise ValueError("independent shadow/public/raw audit mismatch")
        rows.append(
            make_row(item, reader.read(f"{stem}/run.json"), checks[stem], raw, truth)
        )
    if canonical(reader.read("report.json")) != canonical(report(spec, rows)):
        raise ValueError("adequacy report/denominator mismatch")
    timing = reader.nondeterministic("timing.json")
    if (
        type(timing) is not dict
        or set(timing) != {"rows", "scope"}
        or timing["scope"]
        != "actual Agent and online shadow monitor together; storage, public replay, independent audit and orchestration excluded; no isolated marginal monitor timing"
        or [r["stem"] for r in timing["rows"]] != [i["stem"] for i in spec["selection"]]
    ):
        raise ValueError("adequacy timing scope/coverage mismatch")
    for row in timing["rows"]:
        values = row.get("actual_agent_and_online_monitor")
        if (
            set(row) != {"stem", "actual_agent_and_online_monitor"}
            or type(values) is not dict
            or set(values) != {"wall_seconds", "cpu_seconds"}
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in values.values()
            )
        ):
            raise ValueError("invalid actual Agent/monitor timing")
    if hashes() != spec["source_sha256"]:
        raise ValueError("adequacy source drift during independent verification")
    return {
        "valid": True,
        "runs": len(rows),
        "logical_artifacts": len(expected),
        "scope": SCOPE,
    }
