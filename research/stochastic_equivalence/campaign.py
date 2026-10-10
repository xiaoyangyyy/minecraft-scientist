"""Complete PUBLIC program-family certificate matrix, not physical trials."""

import itertools
import math
from pathlib import Path
from time import perf_counter, process_time

from research.compact_campaigns.store import Reader, Writer, canonical
from research.noisy_competition.belief import ModelBank
from research.noisy_competition.campaign import hashes as upstream_hashes
from research.noisy_competition.fixture import public_family
from research.structural_campaigns.corpus import digest

from .audit import verify_certificate
from .search import SCOPE, compare

ROOT = Path(__file__).resolve().parents[2]


def hashes():
    paths = sorted((ROOT / "research" / "stochastic_equivalence").glob("*.py"))
    paths += [
        ROOT / "research/stochastic_equivalence/README.md",
        ROOT / "docs/stochastic-equivalence-contract.md",
        ROOT / "scripts/check_stochastic_equivalence.py",
    ]
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in paths
    }


def registration(seed, sources):
    if type(seed) is not int or not 0 <= seed < 2**64 - 1:
        raise ValueError("two adjacent bounded seeds required")
    return {
        "schema": "public-stochastic-equivalence-registration-v1",
        "seed": seed,
        "source_sha256": sources,
        "seeds": [seed, seed + 1],
        "models_per_seed": 18,
        "pairs_per_seed": 153,
        "context": "initial model beliefs; all declared actions",
        "scope": SCOPE,
        "physical_trials": 0,
        "model_prior_reweighted": False,
    }


def public_models(seed):
    bank = ModelBank(public_family(seed)[0])
    return bank.definitions(), tuple(p.to_ir() for p in bank.programs)


def report_rows(seed, certificates):
    groups = [{i} for i in range(18)]
    for (i, j), certificate in certificates.items():
        if certificate["status"] == "equivalent":
            a = next(g for g in groups if i in g)
            b = next(g for g in groups if j in g)
            if a is not b:
                a.update(b)
                groups.remove(b)
    classes = sorted([sorted(g) for g in groups])
    # Invariant checks use ALL pairs, not union-find transitivity alone.
    for (i, j), c in certificates.items():
        same = any(i in group and j in group for group in classes)
        if same != (c["status"] == "equivalent"):
            raise ValueError("public pair equivalence transitivity mismatch")
    return {
        "seed": seed,
        "models": 18,
        "pairs": len(certificates),
        "equivalent_pairs": sum(
            c["status"] == "equivalent" for c in certificates.values()
        ),
        "distinguishable_pairs": sum(
            c["status"] == "distinguishable" for c in certificates.values()
        ),
        "classes": classes,
        "class_count": len(classes),
        "witness_lengths": {
            str(length): sum(
                c["witness"] is not None and len(c["witness"]["actions"]) == length
                for c in certificates.values()
            )
            for length in range(1, 32)
        },
        "scope": "initial public-model behavioral classes, not physical truth classes, finite-budget detectability or posterior calibration",
    }


def make_report(spec, rows):
    return {
        "schema": "public-stochastic-equivalence-report-v1",
        "registration": spec,
        "rows": rows,
        "scope": SCOPE,
        "physical_trials": 0,
        "accepted_theories": None,
    }


def run(*, output, seed=0):
    sources = hashes()
    spec = registration(seed, sources)
    writer = Writer(output)
    writer.write("registration.json", spec)
    summaries, timings = [], []
    for current_seed in spec["seeds"]:
        definitions, models = public_models(current_seed)
        writer.write(f"seed-{current_seed}/public-models.json", definitions)
        certificates = {}
        for i, j in itertools.combinations(range(len(models)), 2):
            stem = f"seed-{current_seed}/pair-{i:02d}-{j:02d}"
            wall, cpu = perf_counter(), process_time()
            try:
                certificate = compare(models[i], models[j])
                search_timing = {
                    "wall_seconds": perf_counter() - wall,
                    "cpu_seconds": process_time() - cpu,
                }
                # Freeze search proof before invoking the independent checker.
                writer.write(f"{stem}/certificate.json", certificate)
                audit = verify_certificate(models[i], models[j], certificate)
                writer.write(f"{stem}/audit.json", audit)
            except BaseException as error:
                if not writer.failed:
                    writer.write(
                        "failure.json",
                        {
                            "pair": stem,
                            "type": type(error).__name__,
                            "message": str(error),
                            "completed_claim": False,
                        },
                    )
                raise
            certificates[i, j] = certificate
            timings.append({"stem": stem, "actual_search": search_timing})
        summaries.append(report_rows(current_seed, certificates))
        print(
            f"completed all 153 public equivalence certificates seed-{current_seed}",
            flush=True,
        )
    if hashes() != sources:
        writer.write("failure.json", {"type": "source_drift", "completed_claim": False})
        raise ValueError("equivalence source drift; no final certificate")
    report = make_report(spec, summaries)
    writer.write("report.json", report)
    writer.write(
        "timing.json",
        {
            "rows": timings,
            "scope": "actual search only; independent proof checking/storage/orchestration excluded",
        },
        nondeterministic=True,
    )
    writer.finish()
    return report


def verify_output(directory, *, manifest_sha256):
    reader = Reader(directory, manifest_sha256=manifest_sha256)
    spec = reader.read("registration.json")
    if canonical(spec) != canonical(registration(spec["seed"], hashes())):
        raise ValueError("equivalence source/registration mismatch")
    pairs = tuple(itertools.combinations(range(18), 2))
    expected = (
        {"registration.json", "report.json"}
        | {f"seed-{s}/public-models.json" for s in spec["seeds"]}
        | {
            f"seed-{s}/pair-{i:02d}-{j:02d}/{leaf}.json"
            for s in spec["seeds"]
            for i, j in pairs
            for leaf in ("certificate", "audit")
        }
    )
    reader.coverage(expected)
    summaries = []
    for s in spec["seeds"]:
        definitions, models = public_models(s)
        if canonical(reader.read(f"seed-{s}/public-models.json")) != canonical(
            definitions
        ):
            raise ValueError("public model family mismatch")
        certificates = {}
        for i, j in pairs:
            stem = f"seed-{s}/pair-{i:02d}-{j:02d}"
            certificate = reader.read(f"{stem}/certificate.json")
            audit = verify_certificate(models[i], models[j], certificate)
            if canonical(reader.read(f"{stem}/audit.json")) != canonical(audit):
                raise ValueError("independent equivalence audit mismatch")
            certificates[i, j] = certificate
        summaries.append(report_rows(s, certificates))
    if canonical(reader.read("report.json")) != canonical(make_report(spec, summaries)):
        raise ValueError("equivalence report/denominator mismatch")
    timing = reader.nondeterministic("timing.json")
    if (
        set(timing) != {"rows", "scope"}
        or timing["scope"]
        != "actual search only; independent proof checking/storage/orchestration excluded"
        or [r["stem"] for r in timing["rows"]]
        != [f"seed-{s}/pair-{i:02d}-{j:02d}" for s in spec["seeds"] for i, j in pairs]
    ):
        raise ValueError("equivalence timing coverage/scope mismatch")
    for row in timing["rows"]:
        if (
            set(row) != {"stem", "actual_search"}
            or type(row["actual_search"]) is not dict
            or set(row["actual_search"]) != {"wall_seconds", "cpu_seconds"}
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in row["actual_search"].values()
            )
        ):
            raise ValueError("invalid actual search timing")
    return {
        "valid": True,
        "public_model_pairs": 306,
        "logical_artifacts": len(expected),
        "physical_trials": 0,
        "scope": SCOPE,
    }
