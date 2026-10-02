"""Reuse public acquisition validation and frozen keyframe inference for compounds."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import controlled_motion_common as canonical
import keyframe_motion_common as keyframes

from minecraft_scientist.exploration.compound_motion import compound_bindings


def wrapper_path():
    return (
        Path(__file__).resolve().parents[1]
        / "environments/minestudio/capture_compound_rgb_motion.py"
    )


def validate_acquisition(route):
    manifest = json.loads((route / "config.json").read_bytes())
    if manifest.get("config", {}).get("sequence_kind") != "paired-controls-v1":
        raise ValueError(
            "compound acquisition requires an explicit registered sequence kind"
        )
    with compound_bindings(
        canonical, source_function="capture_sources", wrapper=wrapper_path()
    ):
        return canonical.validate_acquisition(route)


def runtime_sources():
    return keyframes.runtime_sources() | {
        "compound_motion.py": Path(compound_bindings.__code__.co_filename),
        "capture_compound_rgb_motion.py": wrapper_path(),
        Path(__file__).name: Path(__file__),
        "probe_compound_rgb_motion.py": Path(__file__).with_name(
            "probe_compound_rgb_motion.py"
        ),
    }


def configuration(route, acquisition, report, seed):
    config = keyframes.configuration(route, acquisition, report, seed)
    config.update(
        acquisition_kind="paired-controls-v1",
        physical_turn_thresholds={
            "minimum_leg_horizontal_blocks": 0.15,
            "minimum_turn_degrees": 45,
            "maximum_turn_degrees": 135,
        },
    )
    return config


@contextmanager
def compound_pipeline(module):
    originals = {
        name: getattr(module, name)
        for name in ("validate_acquisition", "configuration", "runtime_sources")
    }
    if (
        originals["validate_acquisition"] is not canonical.validate_acquisition
        or originals["configuration"] is not keyframes.configuration
        or originals["runtime_sources"] is not keyframes.runtime_sources
    ):
        raise ValueError("compound pipeline requires canonical unmodified functions")
    try:
        module.validate_acquisition, module.configuration, module.runtime_sources = (
            validate_acquisition,
            configuration,
            runtime_sources,
        )
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
