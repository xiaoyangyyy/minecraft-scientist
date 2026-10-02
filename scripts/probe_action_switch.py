"""Freeze and replay original/alternative maps on public switched-action views."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import action_switch_common as common
from controlled_motion_common import digest

from minecraft_scientist.exploration.action_switch_motion import ActionSwitchConfig


def run(route, experiment, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite switched-action experiment")
    paths, sources = common.inputs(route, experiment), common.sources()
    config = common.configuration(route, experiment)
    config.update(
        input_sha256={name: digest(path) for name, path in paths.items()},
        source_sha256={name: digest(path) for name, path in sources.items()},
    )
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    manifest = output / "config.json"
    manifest.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(manifest)
    result = {
        "completed": False,
        "failure": None,
        "config_sha256": frozen,
        "results": [],
        "navigation_accepted": False,
    }
    try:
        common.replay(
            route, experiment, ActionSwitchConfig(), results=result["results"]
        )
        result["completed"] = True
    except Exception as error:  # noqa: BLE001 -- preserve partial experiment evidence
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["inputs_sources_unchanged"] = (
        digest(manifest) == frozen
        and all(
            digest(path) == config["input_sha256"][name] for name, path in paths.items()
        )
        and all(
            digest(path) == config["source_sha256"][name]
            for name, path in sources.items()
        )
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.route, args.experiment, args.output)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "failure": result["failure"],
                "comparisons": dict(
                    Counter(
                        item["action_switch"]["comparison"]
                        for item in result["results"]
                    )
                ),
            }
        )
    )
    if not result["completed"] or not result["inputs_sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
