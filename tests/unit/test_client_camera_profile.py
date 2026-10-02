import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_scientist.minecraft.client_camera_profile import (
    isolated_bobbing_profile,
    options_with_bobbing,
    runtime_ledger,
)


def fake_client(tmp_path, original=b"renderDistance:8\nbobView:true\nfov:0.0\n"):
    source = tmp_path / "shared-runtime"
    source.mkdir()
    if original is not None:
        (source / "options.txt").write_bytes(original)
    (source / "other.txt").write_bytes(b"preserved")
    manager = SimpleNamespace(RUNTIME_DIR=str(source), _instance_pool=[])

    class Instance:
        def __init__(self, working_dir):
            self.working_dir = working_dir

        def _launch_minecraft(self, *args, **kwargs):
            return "fake-launch"

    # Shared pytest caches can retain the other host's filename.
    Instance._launch_minecraft.__code__ = Instance._launch_minecraft.__code__.replace(
        co_filename=str(Path(__file__).resolve())
    )
    return manager, Instance, source


@pytest.mark.parametrize("enabled", [True, False])
def test_options_change_only_declared_line_preserving_crlf(enabled):
    original = b"fov:0.0\r\nbobView:true\r\nrenderDistance:8\r\n"
    value = b"true" if enabled else b"false"
    assert options_with_bobbing(original, enabled) == (
        b"fov:0.0\r\nbobView:" + value + b"\r\nrenderDistance:8\r\n"
    )


def test_missing_options_and_unterminated_lines_are_explicit():
    assert options_with_bobbing(b"", False) == b"bobView:false\n"
    assert options_with_bobbing(b"fov:0", True) == b"fov:0\nbobView:true\n"
    assert options_with_bobbing(b"bobView:true", False) == b"bobView:false"
    with pytest.raises(ValueError, match="duplicate"):
        options_with_bobbing(b"bobView:true\nbobView:false\n", False)
    with pytest.raises(ValueError, match="boolean"):
        options_with_bobbing(b"", 1)


@pytest.mark.parametrize("enabled", [True, False])
def test_launch_profile_is_private_frozen_and_restored(tmp_path, enabled):
    manager, instance, source = fake_client(tmp_path)
    original_runtime, launcher = manager.RUNTIME_DIR, instance._launch_minecraft
    original_ledger = runtime_ledger(source)
    directory = tmp_path / "profile"
    with isolated_bobbing_profile(
        manager,
        instance,
        directory=directory,
        enabled=enabled,
        capture_config={"seed": 201},
        wrapper_source=Path(__file__).resolve(),
    ) as manifest:
        assert manifest["enabled"] is enabled
        assert "capture_camera_profile.py" in manifest["source_sha256"]
        assert Path(manager.RUNTIME_DIR) != source
        assert runtime_ledger(source) == original_ledger
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        assert instance(working)._launch_minecraft(1, option=True) == "fake-launch"
        assert (directory / "launch-options-00.txt").read_bytes() == (
            directory / "supplied-options.txt"
        ).read_bytes()
    report = json.loads((directory / "report.json").read_bytes())
    assert report["passed"] and len(report["checks"]) == 6
    assert manager.RUNTIME_DIR == original_runtime
    assert instance._launch_minecraft is launcher
    assert runtime_ledger(source) == original_ledger


def test_empty_shared_template_stays_empty(tmp_path):
    manager, instance, source = fake_client(tmp_path, original=None)
    directory = tmp_path / "profile"
    with isolated_bobbing_profile(
        manager, instance, directory=directory, enabled=False, capture_config={}
    ):
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        instance(working)._launch_minecraft()
    assert not (source / "options.txt").exists()
    assert (directory / "original-options.txt").read_bytes() == b""
    assert (directory / "supplied-options.txt").read_bytes() == b"bobView:false\n"


def test_exception_restores_settings_and_retains_failed_evidence(tmp_path):
    manager, instance, _ = fake_client(tmp_path)
    original_runtime, launcher = manager.RUNTIME_DIR, instance._launch_minecraft
    directory = tmp_path / "profile"
    with (
        pytest.raises(LookupError, match="original acquisition error"),
        isolated_bobbing_profile(
            manager, instance, directory=directory, enabled=False, capture_config={}
        ),
    ):
        raise LookupError("original acquisition error")
    assert manager.RUNTIME_DIR == original_runtime
    assert instance._launch_minecraft is launcher
    report = json.loads((directory / "report.json").read_bytes())
    assert not report["passed"] and not report["checks"]["launch_options_match"]


def test_wrong_instance_options_fail_before_original_launcher(tmp_path):
    manager, instance, _ = fake_client(tmp_path)
    directory = tmp_path / "profile"
    with (
        pytest.raises(RuntimeError, match="did not copy"),
        isolated_bobbing_profile(
            manager, instance, directory=directory, enabled=False, capture_config={}
        ),
    ):
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        (working / "options.txt").write_bytes(b"bobView:true\n")
        instance(working)._launch_minecraft()
    report = json.loads((directory / "report.json").read_bytes())
    assert not report["passed"] and len(report["launches"]) == 1


@pytest.mark.parametrize("invalid", ["existing", "overlap", "active"])
def test_profile_rejects_unsafe_or_nonfresh_targets(tmp_path, invalid):
    manager, instance, source = fake_client(tmp_path)
    directory = tmp_path / "profile"
    original_runtime, launcher = manager.RUNTIME_DIR, instance._launch_minecraft
    if invalid == "existing":
        directory.mkdir()
        (directory / "owned.txt").write_bytes(b"do not overwrite")
    elif invalid == "overlap":
        directory = source / "profile"
    else:
        manager._instance_pool = [object()]
    with (
        pytest.raises((ValueError, FileExistsError, RuntimeError)),
        isolated_bobbing_profile(
            manager, instance, directory=directory, enabled=False, capture_config={}
        ),
    ):
        pytest.fail("unsafe profile entered capture")
    assert manager.RUNTIME_DIR == original_runtime
    assert instance._launch_minecraft is launcher
    if invalid == "existing":
        assert (directory / "owned.txt").read_bytes() == b"do not overwrite"
        assert not (directory / "config.json").exists()


def test_runtime_copy_budget_is_checked_before_profile_creation(tmp_path):
    manager, instance, source = fake_client(tmp_path)
    for index in range(65):
        (source / f"extra-{index}.txt").write_bytes(b"x")
    directory = tmp_path / "profile"
    with (
        pytest.raises(ValueError, match="budget"),
        isolated_bobbing_profile(
            manager, instance, directory=directory, enabled=False, capture_config={}
        ),
    ):
        pytest.fail("oversized template entered capture")
    assert not directory.exists()


def test_shared_runtime_change_is_not_reported_as_success(tmp_path):
    manager, instance, source = fake_client(tmp_path)
    original_runtime, launcher = manager.RUNTIME_DIR, instance._launch_minecraft
    directory = tmp_path / "profile"
    with (
        pytest.raises(RuntimeError, match="isolation checks"),
        isolated_bobbing_profile(
            manager, instance, directory=directory, enabled=False, capture_config={}
        ),
    ):
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        instance(working)._launch_minecraft()
        (source / "other.txt").write_bytes(b"changed externally")
    assert manager.RUNTIME_DIR == original_runtime
    assert instance._launch_minecraft is launcher
    report = json.loads((directory / "report.json").read_bytes())
    assert not report["checks"]["original_runtime_unchanged"]


def load_wrapper(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "camera_profile_fixture",
        root / "environments/minestudio/capture_camera_profile.py",
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


def test_wrapper_retains_systemexit_failure_and_restores_argv(tmp_path, monkeypatch):
    manager, instance, _ = fake_client(tmp_path)
    wrapper = load_wrapper(monkeypatch)
    output = tmp_path / "acquisition"
    argv = ["capture", "--bobbing", "disabled", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)

    def capture():
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        instance(working)._launch_minecraft()
        raise SystemExit(1)

    monkeypatch.setitem(
        sys.modules,
        "capture_controlled_rgb_motion",
        SimpleNamespace(__file__=__file__, main=capture),
    )
    monkeypatch.setitem(
        sys.modules,
        "minestudio.simulator.minerl.env.malmo",
        SimpleNamespace(InstanceManager=manager, MinecraftInstance=instance),
    )
    with pytest.raises(SystemExit) as error:
        wrapper.main()
    assert error.value.code == 1 and sys.argv is argv
    profile = output.with_name(output.name + ".client-profile")
    binding = json.loads((profile / "binding.json").read_bytes())
    assert binding["failure"]["type"] == "SystemExit"
    assert binding["wrapper_unchanged"] is True
    assert binding["capture_report_sha256"] is None


def test_wrapper_never_overwrites_preexisting_profile(tmp_path, monkeypatch):
    manager, instance, _ = fake_client(tmp_path)
    wrapper = load_wrapper(monkeypatch)
    output = tmp_path / "acquisition"
    profile = output.with_name(output.name + ".client-profile")
    profile.mkdir()
    (profile / "binding.json").write_bytes(b"owned existing evidence")
    monkeypatch.setattr(
        sys, "argv", ["capture", "--bobbing", "disabled", "--output", str(output)]
    )
    monkeypatch.setitem(
        sys.modules,
        "capture_controlled_rgb_motion",
        SimpleNamespace(main=lambda: pytest.fail("existing profile entered capture")),
    )
    monkeypatch.setitem(
        sys.modules,
        "minestudio.simulator.minerl.env.malmo",
        SimpleNamespace(InstanceManager=manager, MinecraftInstance=instance),
    )
    with pytest.raises(SystemExit):
        wrapper.main()
    assert (profile / "binding.json").read_bytes() == b"owned existing evidence"
