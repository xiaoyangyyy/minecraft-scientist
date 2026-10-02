"""Process-local source-bound capture aliases for prospective parallax plans."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from minecraft_scientist.exploration.active_parallax import propose_control_pairs
from minecraft_scientist.exploration.compound_motion import compound_bindings


def wrapper_path():
    return (
        Path(__file__).resolve().parents[1]
        / "environments/minestudio/capture_active_parallax.py"
    )


@contextmanager
def raw_bindings(module, *, source_function):
    factory = getattr(module, source_function)
    paths = factory() | {
        "active_parallax.py": Path(propose_control_pairs.__code__.co_filename),
        Path(__file__).name: Path(__file__),
    }
    setattr(module, source_function, lambda: dict(paths))
    try:
        with compound_bindings(
            module, source_function=source_function, wrapper=wrapper_path()
        ):
            yield
    finally:
        setattr(module, source_function, factory)


def validate_acquisition(route):
    import controlled_motion_common as canonical

    with raw_bindings(canonical, source_function="capture_sources"):
        return canonical.validate_acquisition(route)
