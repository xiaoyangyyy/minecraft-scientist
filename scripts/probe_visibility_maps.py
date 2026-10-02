"""Freeze and infer all adaptive RGB trials, controls and pre-switch maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import visibility_capture_common as common
from controlled_motion_common import digest


def run(route, output, seed=0):
    if output.exists():
        raise FileExistsError("refusing to overwrite adaptive map inference")
    paths, sources = common.inputs(route), common.sources()
    frozen_inputs = {name: digest(path) for name, path in paths.items()}
    frozen_sources = {name: digest(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    result = {
        "completed": False,
        "failure": None,
        "results": [],
        "navigation_accepted": False,
    }
    manifest = output / "config.json"
    config = {
        "seed": seed,
        "input_sha256": frozen_inputs,
        "source_sha256": frozen_sources,
    }
    try:
        acquisition, raw, _, checks = common.validate_capture(route)
        config.update(common.configuration(route, acquisition, raw, seed))
        manifest.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["config_sha256"] = digest(manifest)
        result["public_acquisition_checks"] = checks
        common.infer(route, raw, config, results=result["results"])
        result["completed"] = True
    except Exception as error:  # noqa: BLE001 -- preserve acquisition and partial inference failures
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if not manifest.exists():
            manifest.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            result["config_sha256"] = digest(manifest)
    result["input_sources_unchanged"] = (
        digest(manifest) == result["config_sha256"]
        and all(digest(path) == frozen_inputs[name] for name, path in paths.items())
        and all(digest(path) == frozen_sources[name] for name, path in sources.items())
    )
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
    result = run(args.route, args.output, args.seed)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "failure": result["failure"],
                "pair_count": len(result["results"]),
            }
        )
    )
    if not result["completed"] or not result["input_sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
