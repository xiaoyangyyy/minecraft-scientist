"""Public profile/plan verification and registered inference for active captures."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import active_capture_common as capture
import compound_motion_common as compounds
import controlled_motion_common as canonical
import keyframe_motion_common as keyframes
import plan_active_rgb_motion as planner

CHECK_NAMES = {
    "single_declared_constructor",
    "process_settings_restored",
    "preregistration_unchanged",
    "source_bytes_unchanged",
    "plan_bytes_unchanged",
    "capture_completed",
}


def validate_acquisition(route):
    directory = route.with_name(route.name + ".active-profile")
    manifest, profile = (
        json.loads((directory / name).read_bytes())
        for name in ("config.json", "report.json")
    )
    plan_directory = planner.ROOT / manifest["plan_directory"]
    plan_manifest = json.loads((plan_directory / "config.json").read_bytes())
    previous = planner.inputs(
        planner.ROOT / plan_manifest["source_acquisition"],
        planner.ROOT / plan_manifest["source_predictions"],
    )
    paths = (
        list(previous.values())
        + list(planner.sources().values())
        + [plan_directory / "config.json", plan_directory / "plan.json"]
    )
    frozen = {path: canonical.digest(path) for path in paths}
    requested = planner.validate_plan(
        plan_directory
    )  # Complete public-only pixel replay.
    public, report, checks = capture.validate_acquisition(route)
    with capture.raw_bindings(canonical, source_function="capture_sources"):
        sources = canonical.capture_sources()
    checks.update(
        {
            "active_profile_checks_complete": profile["passed"] is True
            and set(profile["checks"]) == CHECK_NAMES
            and all(value is True for value in profile["checks"].values())
            and profile["failure"] is None,
            "active_manifest_identity": canonical.digest(directory / "config.json")
            == profile["config_sha256"],
            "active_profile_sources": manifest["source_sha256"]
            == {name: canonical.digest(path) for name, path in sources.items()}
            and {path.name for path in (directory / "source-snapshot").glob("*.py")}
            == set(sources)
            and all(
                canonical.digest(directory / "source-snapshot" / name) == value
                for name, value in manifest["source_sha256"].items()
            ),
            "active_plan_identity": manifest["plan_sha256"]
            == {
                "plan_config": canonical.digest(plan_directory / "config.json"),
                "plan_result": canonical.digest(plan_directory / "plan.json"),
            },
            "active_actual_constructor": public == requested
            and manifest["capture_config"] == requested.model_dump(mode="json")
            and profile["constructor_calls"] == [requested.model_dump(mode="json")],
            "active_capture_binding": profile["capture_config_sha256"]
            == canonical.digest(route / "config.json")
            and profile["capture_report_sha256"]
            == canonical.digest(route / "report.json"),
            "active_prior_replay_inputs_stable": all(
                canonical.digest(path) == value for path, value in frozen.items()
            ),
        }
    )
    if not all(checks.values()):
        raise ValueError(f"active plan/profile verification failed: {checks}")
    return public, report, checks


def runtime_sources():
    return compounds.runtime_sources() | {
        "active_parallax.py": Path(planner.propose_control_pairs.__code__.co_filename),
        "active_capture_common.py": Path(capture.__file__),
        "capture_active_parallax.py": capture.wrapper_path(),
        "plan_active_rgb_motion.py": Path(planner.__file__),
        Path(__file__).name: Path(__file__),
        "probe_active_rgb_motion.py": Path(__file__).with_name(
            "probe_active_rgb_motion.py"
        ),
    }


def configuration(route, acquisition, report, seed):
    config = compounds.configuration(route, acquisition, report, seed)
    profile = route.with_name(route.name + ".active-profile")
    config["active_profile_sha256"] = {
        name: canonical.digest(profile / name)
        for name in ("config.json", "report.json")
    }
    config["scope"] += (
        "; prospective provisional-RGB control pairs; not online replanning or navigation acceptance"
    )
    return config


@contextmanager
def active_pipeline(module):
    expected = {
        "validate_acquisition": canonical.validate_acquisition,
        "configuration": keyframes.configuration,
        "runtime_sources": keyframes.runtime_sources,
    }
    if any(getattr(module, name) is not value for name, value in expected.items()):
        raise ValueError("active inference requires canonical unmodified aliases")
    replacements = {
        "validate_acquisition": validate_acquisition,
        "configuration": configuration,
        "runtime_sources": runtime_sources,
    }
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in expected.items():
            setattr(module, name, value)
