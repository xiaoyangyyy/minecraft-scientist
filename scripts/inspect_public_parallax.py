"""Replay public shadow evidence and measure its conditional future view diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shadow_motion_common as shadows
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, public_inputs

from minecraft_scientist.exploration.parallax_observability import (
    ParallaxConfig,
    measure_parallax,
)
from minecraft_scientist.exploration.shadow_motion import ShadowMotionEvidence
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


def run(route, experiment, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite parallax diagnosis")
    acquisition, raw, _ = shadows.compounds.validate_acquisition(route)
    config = json.loads((experiment / "config.json").read_bytes())
    report = json.loads((experiment / "report.json").read_bytes())
    expected = shadows.configuration(route, acquisition, raw, config["seed"])
    inputs = public_inputs(route) | {
        "predictions/config.json": experiment / "config.json",
        "predictions/report.json": experiment / "report.json",
    }
    sources = shadows.runtime_sources() | {
        "parallax_observability.py": Path(measure_parallax.__code__.co_filename),
        Path(__file__).name: Path(__file__),
        "audit_dense_correspondence.py": Path(_numeric_equal.__code__.co_filename),
    }
    frozen = {
        path: digest(path) for path in list(inputs.values()) + list(sources.values())
    }
    declared_sources = shadows.runtime_sources()
    if (
        not report["completed"]
        or not report["input_sources_unchanged"]
        or report["failure"] is not None
        or report["navigation_accepted"] is not False
        or report["config_sha256"] != digest(experiment / "config.json")
        or not _numeric_equal(
            {
                key: value
                for key, value in config.items()
                if key not in ("source_sha256", "input_sha256")
            },
            expected,
        )
        or config["source_sha256"]
        != {name: digest(path) for name, path in declared_sources.items()}
        or config["input_sha256"]
        != {name: digest(path) for name, path in public_inputs(route).items()}
        or [item["pair"] for item in report["results"]] != raw["pairs"]
        or {path.name for path in (experiment / "source-snapshot").glob("*.py")}
        != set(declared_sources)
        or any(
            digest(experiment / "source-snapshot" / name) != value
            for name, value in config["source_sha256"].items()
        )
    ):
        raise ValueError("invalid public shadow declarations")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    policy = ParallaxConfig()
    manifest = {
        "policy": policy.model_dump(mode="json"),
        "input_sha256": {name: digest(path) for name, path in inputs.items()},
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "scope": "public development diagnosis; conditional map geometry, not ground-truth accuracy",
    }
    manifest_path = output / "config.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hash, results, failure = digest(manifest_path), [], None
    try:
        for index, pair in enumerate(raw["pairs"]):
            inferred = shadows.infer_pair(route, pair, config, index)
            if not _numeric_equal(inferred, report["results"][index]):
                raise ValueError("public prefix/shadow replay differs")
            shadow = ShadowMotionEvidence.model_validate(inferred["shadow"])
            tracks = (
                None
                if inferred["shadow_tracks"] is None
                else TemporalTracks.model_validate(inferred["shadow_tracks"])
            )
            evidence = measure_parallax(
                shadow,
                tracks,
                np.asarray(config["camera_intrinsics"]),
                seed=config["seed"] + index,
                config=policy,
            )
            results.append({"pair": pair, "parallax": evidence.model_dump(mode="json")})
    except Exception as error:  # noqa: BLE001 -- preserve all partial diagnostics
        failure = {"type": type(error).__name__, "message": str(error)}
    result = {
        "completed": failure is None and len(results) == len(raw["pairs"]),
        "config_sha256": manifest_hash,
        "inputs_sources_unchanged": digest(manifest_path) == manifest_hash
        and all(digest(path) == value for path, value in frozen.items()),
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
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.route, args.experiment, args.output)
    print(
        json.dumps(
            {"completed": result["completed"], "failure": result["failure"]},
            sort_keys=True,
        )
    )
    if not result["completed"] or not result["inputs_sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
