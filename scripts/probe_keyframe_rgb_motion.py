"""Freeze and run a complete public-only chronological prefix sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_motion_common import digest, public_inputs, validate_acquisition
from keyframe_motion_common import configuration, infer_pair, runtime_sources


def run(route: Path, output: Path, *, seed: int = 0):
    if output.exists():
        raise FileExistsError("refusing to overwrite keyframe experiment")
    acquisition, report, checks = validate_acquisition(route)
    config = configuration(route, acquisition, report, seed)
    inputs, sources = public_inputs(route), runtime_sources()
    config["input_sha256"] = {name: digest(path) for name, path in inputs.items()}
    config["source_sha256"] = {name: digest(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(config_path)
    results, failure = [], None
    try:
        for index, pair in enumerate(config["pairs"]):
            result = infer_pair(route, pair, config, index)
            results.append(result)
            print(
                json.dumps(
                    {
                        "pair_id": pair["id"],
                        "attempts": len(result["selection"]["attempts"]),
                        "selected_offset": result["selection"]["selected_offset"],
                        "terminal_supported": result["selection"]["attempts"][-1][
                            "motion"
                        ]["supported"],
                    }
                ),
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 -- partial failures stay observable
        failure = {"type": type(error).__name__, "message": str(error)}
    stable = (
        digest(config_path) == frozen
        and all(
            digest(path) == config["input_sha256"][name]
            for name, path in inputs.items()
        )
        and all(
            digest(path) == config["source_sha256"][name]
            for name, path in sources.items()
        )
    )
    result = {
        "suite": "chronological_keyframe_development",
        "config_sha256": frozen,
        "public_acquisition_checks": checks,
        "completed": failure is None and len(results) == len(config["pairs"]),
        "input_sources_unchanged": stable,
        "failure": failure,
        "results": results,
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(args.route, args.output, seed=args.seed)
    if not report["completed"] or not report["input_sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
