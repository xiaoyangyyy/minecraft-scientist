"""Freeze and infer movement/settled RGB diagnostics without evaluator access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_motion_common import (
    digest,
    infer_pair,
    inference_config,
    public_inputs,
    runtime_sources,
    validate_acquisition,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=(
            "static-context",
            "consensus",
            "static-consensus",
            "plane-competition",
        ),
    )
    args = parser.parse_args()
    acquisition, report, checks = validate_acquisition(args.route)
    configure, infer, source_registry = inference_config, infer_pair, runtime_sources
    if args.variant is not None:
        from rgb_motion_variants import backend

        variant_backend = backend(args.variant)
        configure, infer, source_registry = (
            variant_backend.inference_config,
            variant_backend.infer_pair,
            variant_backend.runtime_sources,
        )
    config = configure(args.route, acquisition, report, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, sources = public_inputs(args.route), source_registry()
    config["input_sha256"] = {name: digest(path) for name, path in inputs.items()}
    config["source_sha256"] = {name: digest(path) for name, path in sources.items()}
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    path = args.output / "config.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(path)
    results, failure = [], None
    try:
        for index, pair in enumerate(config["pairs"]):
            item = infer(args.route, pair, config, index)
            results.append(item)
            print(
                json.dumps(
                    {
                        "pair_id": pair["id"],
                        "tracks": len(item["tracks"]["track_ids"]),
                        "endpoint_tracks": len(item["endpoint_tracks"]["track_ids"]),
                        "supported": item["motion"]["supported"],
                        "reason": item["motion"]["reason"],
                    }
                ),
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 -- preserve prediction failures
        failure = {"type": type(error).__name__, "message": str(error)}
    stable = (
        digest(path) == frozen
        and all(
            digest(path) == config["input_sha256"][name]
            for name, path in inputs.items()
        )
        and all(
            digest(path) == config["source_sha256"][name]
            for name, path in sources.items()
        )
    )
    output = {
        "config_sha256": frozen,
        "public_acquisition_checks": checks,
        "input_sources_unchanged": stable,
        "completed": failure is None and len(results) == len(config["pairs"]),
        "failure": failure,
        "results": results,
        "navigation_accepted": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not output["completed"] or not stable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
