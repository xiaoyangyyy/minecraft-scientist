"""Verify declared duration override and full raw acquisition without telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_motion_common import capture_sources, digest, validate_acquisition

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig

CHECK_NAMES = {
    "single_declared_constructor",
    "process_settings_restored",
    "preregistration_unchanged",
    "source_bytes_unchanged",
    "capture_completed",
}


def audit(acquisition: Path):
    directory = acquisition.with_name(acquisition.name + ".duration-profile")
    try:
        manifest = json.loads((directory / "config.json").read_bytes())
        report = json.loads((directory / "report.json").read_bytes())
        requested = MotionProbeConfig.model_validate(manifest["capture_config"])
        public, _, public_checks = validate_acquisition(acquisition)
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "environments/minestudio/capture_motion_duration.py"
        )
        sources = capture_sources() | {wrapper.name: wrapper}
        hashes = manifest["source_sha256"]
        checks = {
            "profile_checks_complete": report["passed"] is True
            and set(report["checks"]) == CHECK_NAMES
            and all(value is True for value in report["checks"].values()),
            "manifest_identity": digest(directory / "config.json")
            == report["config_sha256"],
            "canonical_source_set": set(hashes) == set(sources)
            and {path.name for path in (directory / "source-snapshot").glob("*.py")}
            == set(sources),
            "sources_current": set(hashes) == set(sources)
            and all(digest(path) == hashes[name] for name, path in sources.items()),
            "source_snapshots_verified": set(hashes) == set(sources)
            and all(
                digest(directory / "source-snapshot" / name) == hashes[name]
                for name in sources
            ),
            "constructor_and_public_config": report["constructor_calls"]
            == [requested.model_dump(mode="json")]
            and public == requested
            and report["failure"] is None,
            "capture_binding": digest(acquisition / "config.json")
            == report["capture_config_sha256"]
            and digest(acquisition / "report.json") == report["capture_report_sha256"],
            "public_acquisition_valid": all(public_checks.values()),
        }
        return {
            "valid": all(checks.values()),
            "checks": checks,
            "failure": None,
            "movement_ticks": list(public.movement_ticks),
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {
            "valid": False,
            "checks": {"duration_profile_execution": False},
            "failure": str(error),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite duration-profile audit")
    result = audit(args.acquisition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
