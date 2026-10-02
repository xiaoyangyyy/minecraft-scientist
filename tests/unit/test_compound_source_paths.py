"""Decorator wrappers must not substitute stdlib bytes for the executing module."""

import importlib
from functools import wraps
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual pipeline")


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("compound_motion_common")


def test_runtime_ledger_freezes_project_source_not_contextmanager_wrapper(common):
    from minecraft_scientist.exploration import compound_motion

    path = common.runtime_sources()["compound_motion.py"]
    assert path.resolve() == Path(compound_motion.__file__).resolve()
    assert path.name == "compound_motion.py"
    assert b"def compound_pairs" in path.read_bytes()
    # Capture source aliases already freeze the real module via __file__.
    assert (
        path.read_bytes()
        != Path(common.compound_bindings.__code__.co_filename).read_bytes()
    )


def test_source_resolution_survives_additional_decorators(common, monkeypatch):
    from minecraft_scientist.exploration import compound_motion

    original = common.compound_bindings

    @wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "compound_bindings", wrapped)
    assert (
        common.runtime_sources()["compound_motion.py"].resolve()
        == Path(compound_motion.__file__).resolve()
    )
