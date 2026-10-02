"""Bounded, process-local graphics profiles; never edit shared SDK settings."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options_with_bobbing(original: bytes, enabled: bool) -> bytes:
    if type(enabled) is not bool:
        raise ValueError("view-bobbing setting must be a boolean")
    text = original.decode("utf-8")
    lines = text.splitlines(keepends=True)
    matches = [
        index for index, line in enumerate(lines) if line.split(":", 1)[0] == "bobView"
    ]
    if len(matches) > 1:
        raise ValueError("duplicate bobView options are ambiguous")
    value = "true" if enabled else "false"
    if matches:
        index = matches[0]
        newline = (
            "\r\n"
            if lines[index].endswith("\r\n")
            else "\n"
            if lines[index].endswith("\n")
            else ""
        )
        lines[index] = f"bobView:{value}{newline}"
    else:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        lines.append(f"bobView:{value}\n")
    return "".join(lines).encode("utf-8")


def runtime_ledger(directory: Path) -> dict[str, str]:
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("camera runtime template cannot contain symlinks")
    files = [path for path in paths if path.is_file()]
    if len(files) > 64 or sum(path.stat().st_size for path in files) > 10 * 1024 * 1024:
        raise ValueError("camera runtime template exceeds the explicit copy budget")
    return {
        str(path.relative_to(directory)).replace("\\", "/"): file_digest(path)
        for path in files
    }


@contextmanager
def isolated_bobbing_profile(
    manager,
    instance_class,
    *,
    directory: Path,
    enabled: bool,
    capture_config: dict,
    wrapper_source: Path | None = None,
):
    """Redirect only this process's fresh instance template, restoring on failure.

    Launch options are snapshotted *before* the JVM starts. This verifies supplied
    settings, not a claim that Minecraft honored every renderer option. Evidence
    of its effect still requires independently evaluated real RGB comparison.
    """
    source = Path(manager.RUNTIME_DIR).resolve(strict=True)
    target = directory.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("camera profile must not overlap the shared SDK runtime")
    if directory.exists():
        raise FileExistsError("refusing to overwrite camera profile")
    if getattr(manager, "_instance_pool", ()):
        raise RuntimeError("camera profile requires a fresh isolated instance process")
    original = runtime_ledger(source)
    source_options = source / "options.txt"
    original_options = source_options.read_bytes() if source_options.exists() else b""
    supplied = options_with_bobbing(original_options, enabled)
    directory.mkdir(parents=True, exist_ok=False)
    template = directory / "runtime"
    shutil.copytree(source, template)
    (template / "options.txt").write_bytes(supplied)
    (directory / "original-options.txt").write_bytes(original_options)
    (directory / "supplied-options.txt").write_bytes(supplied)
    launcher = instance_class._launch_minecraft
    sdk_path = Path(launcher.__code__.co_filename)
    sources = {"client_camera_profile.py": Path(__file__), "sdk-launcher.py": sdk_path}
    if wrapper_source is not None:
        sources["capture_camera_profile.py"] = wrapper_source
    snapshot = directory / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    source_hashes = {name: file_digest(path) for name, path in sources.items()}
    manifest = {
        "enabled": enabled,
        "capture_config": capture_config,
        "original_runtime_sha256": original,
        "supplied_runtime_sha256": runtime_ledger(template),
        "original_options_sha256": file_digest(directory / "original-options.txt"),
        "supplied_options_sha256": file_digest(directory / "supplied-options.txt"),
        "source_sha256": source_hashes,
        "scope": "explicit launch graphics setting; not a hidden world rule or navigation acceptance",
    }
    manifest_path = directory / "config.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hash = file_digest(manifest_path)
    launches = []
    old_runtime = manager.RUNTIME_DIR

    def launch(instance, *args, **kwargs):
        path = Path(instance.working_dir) / "options.txt"
        data = path.read_bytes()
        output = directory / f"launch-options-{len(launches):02d}.txt"
        output.write_bytes(data)
        launches.append(
            {
                "file": output.name,
                "sha256": file_digest(output),
                "matches_supplied": data == supplied,
            }
        )
        if data != supplied:
            raise RuntimeError("instance did not copy the declared graphics options")
        return launcher(instance, *args, **kwargs)

    manager.RUNTIME_DIR = str(template.resolve())
    instance_class._launch_minecraft = launch
    try:
        yield manifest
    finally:
        manager.RUNTIME_DIR = old_runtime
        instance_class._launch_minecraft = launcher
        checks = {
            "launch_options_match": bool(launches)
            and all(item["matches_supplied"] for item in launches),
            "original_runtime_unchanged": runtime_ledger(source) == original,
            "supplied_runtime_unchanged": runtime_ledger(template)
            == manifest["supplied_runtime_sha256"],
            "preregistration_unchanged": file_digest(manifest_path) == manifest_hash,
            "source_bytes_unchanged": all(
                file_digest(path) == source_hashes[name]
                for name, path in sources.items()
            ),
            "process_settings_restored": manager.RUNTIME_DIR == old_runtime
            and instance_class._launch_minecraft is launcher,
        }
        report = {
            "config_sha256": manifest_hash,
            "launches": launches,
            "checks": checks,
            "passed": all(checks.values()),
        }
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not report["passed"] and sys.exc_info()[0] is None:
            raise RuntimeError("camera profile failed its launch/isolation checks")
