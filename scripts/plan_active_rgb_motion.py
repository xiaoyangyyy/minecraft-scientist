"""Replay public compound evidence, then preregister a prospective action-pair plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compound_motion_common as compounds
import keyframe_motion_common as keyframes
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, public_inputs

from minecraft_scientist.exploration.active_parallax import (
    ParallaxPolicy,
    homogeneous_direction_samples,
    propose_control_pairs,
)
from minecraft_scientist.exploration.compound_motion import CompoundProbeConfig

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return compounds.runtime_sources() | {
        "active_parallax.py": Path(propose_control_pairs.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "audit_dense_correspondence.py": Path(_numeric_equal.__code__.co_filename),
    }


def inputs(route, experiment):
    paths = (
        list(public_inputs(route).values())
        + [experiment / "config.json", experiment / "report.json"]
        + sorted((experiment / "source-snapshot").glob("*.py"))
    )
    return {path.resolve().relative_to(ROOT).as_posix(): path for path in paths}


def replay_prior(route, experiment):
    acquisition, report, _ = compounds.validate_acquisition(route)
    config = json.loads((experiment / "config.json").read_bytes())
    predictions = json.loads((experiment / "report.json").read_bytes())
    expected = compounds.configuration(route, acquisition, report, config["seed"])
    runtime, public = compounds.runtime_sources(), public_inputs(route)
    valid = (
        predictions["completed"] is True
        and predictions["failure"] is None
        and predictions["input_sources_unchanged"] is True
        and predictions["navigation_accepted"] is False
        and predictions["suite"] == "chronological_keyframe_development"
        and predictions["config_sha256"] == digest(experiment / "config.json")
        and _numeric_equal(
            {
                key: value
                for key, value in config.items()
                if key not in ("input_sha256", "source_sha256")
            },
            expected,
        )
        and config["input_sha256"]
        == {name: digest(path) for name, path in public.items()}
        and config["source_sha256"]
        == {name: digest(path) for name, path in runtime.items()}
        and {path.name for path in (experiment / "source-snapshot").glob("*.py")}
        == set(runtime)
        and all(
            digest(experiment / "source-snapshot" / name) == value
            for name, value in config["source_sha256"].items()
        )
        and [item["pair"] for item in predictions["results"]] == report["pairs"]
    )
    if not valid:
        raise ValueError("prior public prediction declarations or sources are invalid")
    replay = [
        keyframes.infer_pair(route, pair, config, index)
        for index, pair in enumerate(report["pairs"])
    ]
    if not _numeric_equal(replay, predictions["results"]):
        raise ValueError("prior public prefix replay differs")
    samples = homogeneous_direction_samples(replay, report["timeline"])
    return acquisition, samples


def capture_config(acquisition, choices):
    return CompoundProbeConfig.model_validate(
        acquisition.model_dump() | {"handle_pairs": choices["handle_pairs"]}
    )


def build_plan(route, experiment, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite active parallax plan")
    paths, runtime = inputs(route, experiment), sources()
    policy = ParallaxPolicy()
    config = {
        "schema": "active-parallax-plan-v3",
        "source_acquisition": route.resolve().relative_to(ROOT).as_posix(),
        "source_predictions": experiment.resolve().relative_to(ROOT).as_posix(),
        "policy": policy.model_dump(mode="json"),
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "source_sha256": {name: digest(path) for name, path in runtime.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in runtime.items():
        (snapshot / name).write_bytes(path.read_bytes())
    manifest = output / "config.json"
    manifest.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen, result = (
        digest(manifest),
        {"completed": False, "failure": None, "config_sha256": digest(manifest)},
    )
    try:
        acquisition, samples = replay_prior(route, experiment)
        choices = propose_control_pairs(samples, policy=policy)
        result.update(
            completed=True,
            choices=choices,
            capture_config=capture_config(acquisition, choices).model_dump(mode="json"),
        )
    except Exception as error:  # noqa: BLE001 -- preserve failed preregistration/replay
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["inputs_sources_unchanged"] = (
        digest(manifest) == frozen
        and all(
            digest(path) == config["input_sha256"][name] for name, path in paths.items()
        )
        and all(
            digest(path) == config["source_sha256"][name]
            for name, path in runtime.items()
        )
    )
    (output / "plan.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def validate_plan(directory):
    config = json.loads((directory / "config.json").read_bytes())
    plan = json.loads((directory / "plan.json").read_bytes())
    route, experiment = (
        ROOT / config["source_acquisition"],
        ROOT / config["source_predictions"],
    )
    paths, runtime = inputs(route, experiment), sources()
    if (
        config["schema"] != "active-parallax-plan-v3"
        or config["policy"] != ParallaxPolicy().model_dump(mode="json")
        or plan["completed"] is not True
        or plan["failure"] is not None
        or plan["inputs_sources_unchanged"] is not True
        or plan["config_sha256"] != digest(directory / "config.json")
        or config["input_sha256"]
        != {name: digest(path) for name, path in paths.items()}
        or config["source_sha256"]
        != {name: digest(path) for name, path in runtime.items()}
        or {path.name for path in (directory / "source-snapshot").glob("*.py")}
        != set(runtime)
        or any(
            digest(directory / "source-snapshot" / name) != value
            for name, value in config["source_sha256"].items()
        )
    ):
        raise ValueError("active parallax plan or its public provenance is invalid")
    acquisition, samples = replay_prior(route, experiment)
    choices = propose_control_pairs(samples)
    requested = capture_config(acquisition, choices)
    if (
        not _numeric_equal(choices, plan["choices"])
        or requested.model_dump(mode="json") != plan["capture_config"]
    ):
        raise ValueError(
            "active parallax plan does not reproduce public-only action selection"
        )
    if (
        digest(directory / "config.json") != plan["config_sha256"]
        or json.loads((directory / "plan.json").read_bytes()) != plan
        or any(
            digest(path) != config["input_sha256"][name] for name, path in paths.items()
        )
        or any(
            digest(path) != config["source_sha256"][name]
            for name, path in runtime.items()
        )
        or any(
            digest(directory / "source-snapshot" / name) != value
            for name, value in config["source_sha256"].items()
        )
    ):
        raise ValueError("active plan provenance changed during public replay")
    return requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(args.route, args.experiment, args.output)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "failure": result["failure"],
                "pairs": result.get("choices", {}).get("handle_pairs"),
            },
            sort_keys=True,
        )
    )
    if not result["completed"] or not result["inputs_sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
