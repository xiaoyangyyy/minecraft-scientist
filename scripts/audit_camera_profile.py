"""Verify explicit launch camera setting and binding to complete public acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_motion_common import validate_acquisition

from minecraft_scientist.minecraft.client_camera_profile import (
    file_digest,
    options_with_bobbing,
    runtime_ledger,
)


def _audit(directory, acquisition):
    config = json.loads((directory / "config.json").read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    binding = json.loads((directory / "binding.json").read_bytes())
    public_config, _, public_checks = validate_acquisition(acquisition)
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "environments/minestudio/capture_camera_profile.py"
    )
    sources = config["source_sha256"]
    expected_launches = {
        f"launch-options-{index:02d}.txt" for index in range(len(report["launches"]))
    }
    launch_names_valid = {
        item["file"] for item in report["launches"]
    } == expected_launches and {
        path.name for path in directory.glob("launch-options-*.txt")
    } == expected_launches
    checks = {
        "profile_passed": report["passed"] is True
        and set(report["checks"])
        == {
            "launch_options_match",
            "original_runtime_unchanged",
            "supplied_runtime_unchanged",
            "preregistration_unchanged",
            "source_bytes_unchanged",
            "process_settings_restored",
        }
        and all(value is True for value in report["checks"].values()),
        "manifest_identity": file_digest(directory / "config.json")
        == report["config_sha256"],
        "wrapper_identity": file_digest(wrapper)
        == binding["wrapper_sha256"]
        == sources["capture_camera_profile.py"]
        == file_digest(directory / "source-snapshot" / wrapper.name)
        and binding["wrapper_unchanged"] is True,
        "capture_binding": file_digest(acquisition / "config.json")
        == binding["capture_config_sha256"]
        and file_digest(acquisition / "report.json") == binding["capture_report_sha256"]
        and binding["failure"] is None,
        "public_config_matches": public_config.model_dump(mode="json")
        == config["capture_config"],
        "profile_source_snapshots": set(sources)
        == {"client_camera_profile.py", "sdk-launcher.py", "capture_camera_profile.py"}
        and all(
            file_digest(directory / "source-snapshot" / name) == digest
            for name, digest in sources.items()
        ),
        "profile_module_current": file_digest(
            Path(options_with_bobbing.__code__.co_filename)
        )
        == sources["client_camera_profile.py"],
        "options_byte_identity": file_digest(directory / "original-options.txt")
        == config["original_options_sha256"]
        and file_digest(directory / "supplied-options.txt")
        == config["supplied_options_sha256"],
        "single_declared_setting_change": options_with_bobbing(
            (directory / "original-options.txt").read_bytes(), config["enabled"]
        )
        == (directory / "supplied-options.txt").read_bytes(),
        "runtime_ledger": runtime_ledger(directory / "runtime")
        == config["supplied_runtime_sha256"]
        == config["original_runtime_sha256"]
        | {"options.txt": config["supplied_options_sha256"]},
        "all_launches_verified": bool(report["launches"])
        and launch_names_valid
        and all(
            item["matches_supplied"] is True
            and file_digest(directory / item["file"])
            == item["sha256"]
            == config["supplied_options_sha256"]
            for item in report["launches"]
        ),
        "public_acquisition_valid": all(public_checks.values()),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "enabled": config["enabled"],
        "capture_report_sha256": binding["capture_report_sha256"],
        "scope": "verified options supplied at launch; actual motion accuracy remains separately evaluated",
    }


def audit(directory, acquisition):
    try:
        return _audit(directory, acquisition)
    except (ValueError, KeyError, OSError, TypeError) as error:
        return {
            "valid": False,
            "checks": {"complete_profile_evidence": False},
            "failure": str(error),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite camera profile audit")
    result = audit(args.profile, args.acquisition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
