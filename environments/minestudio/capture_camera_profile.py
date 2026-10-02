"""Run the unchanged controlled acquisition with explicit private camera options."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig
from minecraft_scientist.minecraft.client_camera_profile import (
    file_digest,
    isolated_bobbing_profile,
)

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--observation-size", type=int, default=448)
    parser.add_argument("--time", choices=("noon", "midnight"), default="noon")
    parser.add_argument("--weather", choices=("clear", "thunder"), default="clear")
    parser.add_argument("--bobbing", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite camera acquisition")
    config = MotionProbeConfig(
        seed=args.seed,
        observation_size=args.observation_size,
        time_of_day=args.time,
        weather=args.weather,
    )
    import capture_controlled_rgb_motion as capture
    from minestudio.simulator.minerl.env.malmo import InstanceManager, MinecraftInstance

    directory = args.output.with_name(args.output.name + ".client-profile")
    argv = sys.argv
    error = None
    created = False
    manifest = None
    try:
        with isolated_bobbing_profile(
            InstanceManager,
            MinecraftInstance,
            directory=directory,
            enabled=args.bobbing == "enabled",
            capture_config=config.model_dump(mode="json"),
            wrapper_source=Path(__file__),
        ) as manifest:
            created = True
            sys.argv = [
                capture.__file__,
                "--seed",
                str(args.seed),
                "--observation-size",
                str(args.observation_size),
                "--time",
                args.time,
                "--weather",
                args.weather,
                "--output",
                str(args.output),
            ]
            capture.main()
    except (Exception, SystemExit) as failure:  # noqa: BLE001 -- preserve failed acquisition without overwriting existing profiles
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        sys.argv = argv
    if created:
        binding = {
            "wrapper_sha256": manifest["source_sha256"]["capture_camera_profile.py"],
            "wrapper_unchanged": file_digest(Path(__file__))
            == manifest["source_sha256"]["capture_camera_profile.py"],
            "capture_report_sha256": file_digest(args.output / "report.json")
            if (args.output / "report.json").exists()
            else None,
            "capture_config_sha256": file_digest(args.output / "config.json")
            if (args.output / "config.json").exists()
            else None,
            "failure": error,
        }
        (directory / "binding.json").write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"profile_directory": str(directory), "failure": error}))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
