"""Freeze new graph/edit fixtures and leakage-safe splits, NOT learning results."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.structural_holdouts.audit_corpus import (
    audit_dataset,
    leakage_partition,
    summarize,
)
from research.structural_holdouts.generator import CASES, fixture


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_hashes():
    paths = (
        sorted((ROOT / "research" / "structural_holdouts").glob("*.py"))
        + sorted((ROOT / "research" / "structural_holdouts").glob("*.md"))
        + [Path(__file__)]
    )
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def build(
    *,
    output,
    seed=0,
    base_count=64,
    renamings=2,
    states=(2, 3, 4, 5),
    boundary=20,
    salt="new-structure-split-v1",
):
    if (
        type(seed) is not int
        or seed < 0
        or type(base_count) is not int
        or not 1 <= base_count <= 512
        or type(renamings) is not int
        or not 1 <= renamings <= 4
        or type(states) is not tuple
        or not states
        or len(set(states)) != len(states)
        or any(type(n) is not int or not 2 <= n <= 5 for n in states)
        or type(boundary) is not int
        or not 0 <= boundary <= 1000
        or type(salt) is not str
        or not 1 <= len(salt) <= 256
    ):
        raise ValueError("invalid new-structure corpus protocol")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    config = {
        "seed": seed,
        "base_count": base_count,
        "renamings": renamings,
        "states": list(states),
        "boundary": boundary,
        "salt": salt,
    }
    inputs = [
        {
            "seed": seed,
            "base_index": i,
            "states": states[i % len(states)],
            "case": case,
            "renaming_index": r,
            "boundary": boundary,
        }
        for i, case, r in itertools.product(range(base_count), CASES, range(renamings))
    ]
    registration = {
        "schema": "new-structure-corpus-registration-v1",
        "config": config,
        "fixture_inputs": inputs,
        "source_sha256": sources,
        "leakage_policy": "cost-colored target and all supplied-before-prior structure/behavior share namespaces; before/after pair structure/behavior also grouped. Transitive closure, not seed-based splitting",
        "scope": "new graph/edit development dataset; sampling/analysis/method freeze not yet a completed formal study",
    }
    dump(output / "registration.json", registration)
    records, times = [], []
    for values in inputs:
        start = perf_counter()
        result = fixture(**values)
        fid = result["metadata"]["fixture_id"]
        dump(output / fid / "public.json", result["public"])
        dump(output / fid / "metadata.json", result["metadata"])
        dump(output / fid / "evaluator" / "world.json", result["private"])
        records.append(result["metadata"])
        times.append(
            {"fixture_id": fid, "construction_seconds": perf_counter() - start}
        )
    partitions = leakage_partition(records, salt=salt)
    summary = summarize(records, partitions)
    report = {
        "schema": "new-structure-corpus-report-v1",
        "fixture_count": len(records),
        "independent_group_count": partitions["independent_group_count"],
        "summary": summary,
        "limitations": "renamings are not independent samples; whole prior families may bridge groups; known/prior-covered targets, invisible changes and grammar mismatch retained; no learning or statistical superiority result",
    }
    dump(output / "partitions.json", partitions)
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {"schema": "new-structure-construction-timing-v1", "rows": times},
    )
    if sources != source_hashes():
        raise RuntimeError("corpus source drift; retain partial data without manifest")
    manifest = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    dump(
        output / "manifest.json",
        {
            "schema": "new-structure-corpus-manifest-v1",
            "sha256": manifest,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    try:
        audit = audit_dataset(output)
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
        # output was exclusively created by this invocation; preserve all failed
        # fixtures but remove only the just-created, uncertified manifest.
        (output / "manifest.json").unlink()
        raise
    print(json.dumps(audit, sort_keys=True))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-count", type=int, default=64)
    parser.add_argument("--renamings", type=int, default=2)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--boundary", type=int, default=20)
    parser.add_argument("--salt", default="new-structure-split-v1")
    kwargs = vars(parser.parse_args())
    kwargs["states"] = tuple(kwargs["states"])
    build(**kwargs)
