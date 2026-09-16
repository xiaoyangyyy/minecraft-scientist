"""Pinned, checksum-verified Minecraft Java server lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Self, TextIO

from .backend import discover_java

MINECRAFT_VERSION = "1.21.11"
SERVER_URL = (
    "https://piston-data.mojang.com/v1/objects/"
    "64bb6d763bed0a9f1d632ec347938594144943ed/server.jar"
)
SERVER_SHA1 = "64bb6d763bed0a9f1d632ec347938594144943ed"
SERVER_SIZE = 56_327_581
EULA_URL = "https://www.minecraft.net/en-us/eula"
FABRIC_LOADER_VERSION = "0.19.5"
FABRIC_INSTALLER_VERSION = "1.1.2"
FABRIC_API_VERSION = "0.141.6+1.21.11"
FABRIC_SERVER_URL = (
    "https://meta.fabricmc.net/v2/versions/loader/"
    f"{MINECRAFT_VERSION}/{FABRIC_LOADER_VERSION}/{FABRIC_INSTALLER_VERSION}/"
    "server/jar"
)
FABRIC_LAUNCHER = "fabric-server-launch.jar"
FABRIC_API_FILENAME = f"fabric-api-{FABRIC_API_VERSION}.jar"
RULE_MOD_PREFIX = "scientist-world-rules-"


class MinecraftEulaRequired(RuntimeError):
    """Raised when server preparation is attempted without explicit acceptance."""


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def server_status(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    jar = directory / "server.jar"
    actual_sha1 = file_sha1(jar) if jar.is_file() else None
    eula_path = directory / "eula.txt"
    eula_accepted = False
    if eula_path.is_file():
        eula_accepted = any(
            line.strip().lower() == "eula=true"
            for line in eula_path.read_text(encoding="utf-8").splitlines()
        )
    launcher = directory / FABRIC_LAUNCHER
    launcher_digest = file_sha256(launcher) if launcher.is_file() else None
    launcher_lock = directory / f"{FABRIC_LAUNCHER}.sha256"
    expected_launcher_digest = (
        launcher_lock.read_text(encoding="ascii").strip().lower()
        if launcher_lock.is_file()
        else None
    )
    fabric_launcher_valid = (
        launcher_digest is not None
        and launcher_digest == expected_launcher_digest
        and zipfile.is_zipfile(launcher)
    )
    fabric_mods_valid, fabric_mods = _fabric_mod_status(directory)
    vanilla_ready = actual_sha1 == SERVER_SHA1 and eula_accepted
    return {
        "directory": str(directory),
        "version": MINECRAFT_VERSION,
        "jar_present": jar.is_file(),
        "jar_sha1": actual_sha1,
        "jar_valid": actual_sha1 == SERVER_SHA1,
        "eula_accepted": eula_accepted,
        "properties_present": (directory / "server.properties").is_file(),
        "ready": vanilla_ready,
        "fabric_loader_version": FABRIC_LOADER_VERSION,
        "fabric_launcher_present": launcher.is_file(),
        "fabric_launcher_sha256": launcher_digest,
        "fabric_launcher_valid": fabric_launcher_valid,
        "fabric_mods": fabric_mods,
        "fabric_mods_valid": fabric_mods_valid,
        "fabric_ready": vanilla_ready
        and fabric_launcher_valid
        and fabric_mods_valid,
    }


def prepare_server(
    directory: Path,
    *,
    accept_eula: bool,
    seed: int = 0,
    overwrite_config: bool = False,
) -> dict[str, Any]:
    """Download a pinned vanilla server after explicit EULA acceptance."""

    if not accept_eula:
        raise MinecraftEulaRequired(
            "Minecraft server setup requires explicit EULA acceptance; review "
            f"{EULA_URL} and pass accept_eula=True only if you agree."
        )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jar = directory / "server.jar"
    if jar.exists() and file_sha1(jar) != SERVER_SHA1:
        raise RuntimeError(f"refusing to replace an unrecognized server jar: {jar}")
    if not jar.exists():
        partial = directory / "server.jar.part"
        try:
            urllib.request.urlretrieve(SERVER_URL, partial)
            if partial.stat().st_size != SERVER_SIZE:
                raise RuntimeError("downloaded server jar has an unexpected size")
            if file_sha1(partial) != SERVER_SHA1:
                raise RuntimeError("downloaded server jar failed SHA-1 verification")
            partial.replace(jar)
        finally:
            if partial.exists():
                partial.unlink()

    eula_path = directory / "eula.txt"
    eula_path.write_text(
        f"# Accepted explicitly for Minecraft {MINECRAFT_VERSION}\neula=true\n",
        encoding="utf-8",
    )
    properties_path = directory / "server.properties"
    if overwrite_config or not properties_path.exists():
        properties_path.write_text(
            _server_properties(seed),
            encoding="utf-8",
        )
    return server_status(directory)


def prepare_fabric_server(
    directory: Path,
    *,
    accept_eula: bool,
    mod_bundle: Path,
    seed: int = 0,
    overwrite_config: bool = False,
    overwrite_mods: bool = False,
) -> dict[str, Any]:
    """Prepare the pinned Fabric research server after explicit EULA acceptance."""

    if not accept_eula:
        raise MinecraftEulaRequired(
            "Minecraft server setup requires explicit EULA acceptance; review "
            f"{EULA_URL} and pass accept_eula=True only if you agree."
        )
    mod_bundle = mod_bundle.resolve()
    source_mods = _validate_mod_bundle(mod_bundle)
    prepare_server(
        directory,
        accept_eula=accept_eula,
        seed=seed,
        overwrite_config=overwrite_config,
    )
    directory = directory.resolve()

    launcher = directory / FABRIC_LAUNCHER
    launcher_lock = directory / f"{FABRIC_LAUNCHER}.sha256"
    if launcher.exists():
        if not launcher_lock.is_file():
            raise RuntimeError(f"refusing to trust an unlocked Fabric launcher: {launcher}")
        expected = launcher_lock.read_text(encoding="ascii").strip().lower()
        if file_sha256(launcher) != expected or not zipfile.is_zipfile(launcher):
            raise RuntimeError(f"Fabric launcher failed its recorded checksum: {launcher}")
    else:
        partial = directory / f"{FABRIC_LAUNCHER}.part"
        try:
            urllib.request.urlretrieve(FABRIC_SERVER_URL, partial)
            if not zipfile.is_zipfile(partial):
                raise RuntimeError("downloaded Fabric launcher is not a valid jar")
            digest = file_sha256(partial)
            partial.replace(launcher)
            launcher_lock.write_text(digest + "\n", encoding="ascii")
        finally:
            if partial.exists():
                partial.unlink()

    launcher_properties = directory / "fabric-server-launcher.properties"
    if not launcher_properties.exists():
        launcher_properties.write_text("serverJar=server.jar\n", encoding="ascii")

    mods_directory = directory / "mods"
    mods_directory.mkdir(parents=True, exist_ok=True)
    lock_payload: dict[str, str] = {}
    for source in source_mods:
        destination = mods_directory / source.name
        source_digest = file_sha256(source)
        if destination.exists() and file_sha256(destination) != source_digest:
            if not overwrite_mods:
                raise RuntimeError(f"refusing to replace a different server mod: {destination}")
        shutil.copy2(source, destination)
        lock_payload[source.name] = source_digest
    (directory / "scientist-mods.lock.json").write_text(
        json.dumps(lock_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = server_status(directory)
    if not report["fabric_ready"]:
        raise RuntimeError("Fabric server preparation did not pass integrity checks")
    return report


def run_server(
    directory: Path,
    *,
    min_memory: str = "1G",
    max_memory: str = "3G",
    flavor: str = "auto",
) -> int:
    """Run the prepared server in the foreground and return its exit code."""

    status = server_status(directory)
    launcher = _select_launcher(status, flavor)
    if not status["ready"]:
        raise RuntimeError(
            "server is not ready; run minecraft-server-prepare after reviewing "
            "and accepting the Minecraft EULA"
        )
    completed = subprocess.run(
        _server_command(
            min_memory=min_memory,
            max_memory=max_memory,
            launcher=launcher,
        ),
        cwd=directory.resolve(),
        check=False,
    )
    return completed.returncode


class MinecraftServerProcess:
    """Managed foreground server with a single-line console command channel."""

    def __init__(
        self,
        directory: Path,
        *,
        min_memory: str = "1G",
        max_memory: str = "3G",
        startup_timeout: float = 120.0,
        flavor: str = "auto",
    ) -> None:
        self.directory = directory.resolve()
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.startup_timeout = startup_timeout
        self.flavor = flavor
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._tail: deque[str] = deque(maxlen=100)
        self._reader: threading.Thread | None = None

    @property
    def log_tail(self) -> tuple[str, ...]:
        return tuple(self._tail)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            raise RuntimeError("Minecraft server is already running")
        status = server_status(self.directory)
        if not status["ready"]:
            raise RuntimeError("Minecraft server directory is not prepared")
        launcher = _select_launcher(status, self.flavor)
        self._process = subprocess.Popen(
            _server_command(
                min_memory=self.min_memory,
                max_memory=self.max_memory,
                launcher=launcher,
            ),
            cwd=self.directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self._process.stdout is not None
        self._reader = threading.Thread(
            target=self._read_output,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._reader.start()
        self._wait_until_ready()

    def command(self, value: str) -> None:
        if not self.running or self._process is None or self._process.stdin is None:
            raise RuntimeError("Minecraft server is not running")
        if not value or len(value) > 512 or "\n" in value or "\r" in value:
            raise ValueError("server command must be one non-empty line")
        self._process.stdin.write(value + "\n")
        self._process.stdin.flush()

    def stop(self) -> None:
        if self._process is None:
            return
        if self.running:
            try:
                self.command("stop")
                self._process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
        self._process = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _read_output(self, stream: TextIO) -> None:
        for line in stream:
            cleaned = line.rstrip()
            self._tail.append(cleaned)
            self._lines.put(cleaned)

    def _wait_until_ready(self) -> None:
        import time

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "Minecraft server exited during startup: " + " | ".join(self._tail)
                )
            try:
                line = self._lines.get(
                    timeout=max(0.01, min(1.0, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if "Done (" in line and "For help" in line:
                return
        self.stop()
        raise TimeoutError(
            "Minecraft server startup timed out: " + " | ".join(self._tail)
        )


class GradleMinecraftServerProcess(MinecraftServerProcess):
    """Run the real Fabric dev server from Loom's checksum-backed local cache."""

    def __init__(
        self,
        project_directory: Path,
        *,
        startup_timeout: float = 180.0,
    ) -> None:
        self.project_directory = project_directory.resolve()
        super().__init__(
            self.project_directory / "run",
            startup_timeout=startup_timeout,
            flavor="fabric",
        )

    def start(self) -> None:
        if self.running:
            raise RuntimeError("Minecraft server is already running")
        status = server_status(self.directory)
        if not status["ready"]:
            raise RuntimeError(
                "Gradle dev-server run directory is not prepared with the pinned "
                "server jar and explicit EULA acceptance"
            )
        wrapper = self.project_directory / "gradlew.bat"
        if not wrapper.is_file():
            raise RuntimeError(f"Gradle wrapper is missing: {wrapper}")
        java = discover_java()
        if java is None:
            raise RuntimeError("Java 21 was not found")
        java_tmp = self.project_directory / ".java-tmp"
        java_tmp.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["JAVA_HOME"] = str(java.parent.parent)
        environment["TEMP"] = str(java_tmp)
        environment["TMP"] = str(java_tmp)
        compatibility = " ".join(
            (
                f"-Djava.io.tmpdir={java_tmp}",
                f"-Djdk.net.unixdomain.tmpdir={java_tmp}",
                "-Djava.net.preferIPv4Stack=true",
            )
        )
        environment["JAVA_TOOL_OPTIONS"] = " ".join(
            value
            for value in (environment.get("JAVA_TOOL_OPTIONS", ""), compatibility)
            if value
        )
        self._process = subprocess.Popen(
            [
                str(wrapper),
                "--no-daemon",
                "--project-dir",
                str(self.project_directory),
                "runServer",
            ],
            cwd=self.project_directory,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self._process.stdout is not None
        self._reader = threading.Thread(
            target=self._read_output,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._reader.start()
        self._wait_until_ready()


def create_server_process(
    directory: Path, *, mode: str = "fabric"
) -> MinecraftServerProcess:
    """Select the packaged Fabric launcher or the real Loom development server."""

    if mode == "fabric":
        return MinecraftServerProcess(directory, flavor="fabric")
    if mode == "gradle":
        return GradleMinecraftServerProcess(directory)
    raise ValueError("server mode must be fabric or gradle")


def _server_command(
    *, min_memory: str, max_memory: str, launcher: str = "server.jar"
) -> list[str]:
    for value in (min_memory, max_memory):
        if re.fullmatch(r"[1-9][0-9]*[MG]", value.upper()) is None:
            raise ValueError("memory values must look like 512M or 3G")
    java = discover_java()
    if java is None:
        raise RuntimeError("Java 21 was not found")
    return [
        str(java),
        f"-Xms{min_memory.upper()}",
        f"-Xmx{max_memory.upper()}",
        "-jar",
        launcher,
        "nogui",
    ]


def _select_launcher(status: dict[str, Any], flavor: str) -> str:
    if flavor not in {"auto", "vanilla", "fabric"}:
        raise ValueError("server flavor must be auto, vanilla, or fabric")
    if flavor == "fabric":
        if not status["fabric_ready"]:
            raise RuntimeError("Fabric server is not prepared or failed integrity checks")
        return FABRIC_LAUNCHER
    if flavor == "auto" and status["fabric_ready"]:
        return FABRIC_LAUNCHER
    return "server.jar"


def _validate_mod_bundle(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise RuntimeError(f"Fabric mod bundle does not exist: {directory}")
    api = directory / FABRIC_API_FILENAME
    rules = sorted(directory.glob(f"{RULE_MOD_PREFIX}*.jar"))
    if not api.is_file() or len(rules) != 1:
        raise RuntimeError(
            "Fabric mod bundle must contain the pinned Fabric API and exactly one "
            "scientist-world-rules jar"
        )
    selected = (api, rules[0])
    if not all(zipfile.is_zipfile(path) for path in selected):
        raise RuntimeError("Fabric mod bundle contains an invalid jar")
    return selected


def _fabric_mod_status(directory: Path) -> tuple[bool, dict[str, str]]:
    lock = directory / "scientist-mods.lock.json"
    if not lock.is_file():
        return False, {}
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, {}
    if not isinstance(payload, dict) or not payload:
        return False, {}
    actual: dict[str, str] = {}
    for name, expected in sorted(payload.items()):
        if not isinstance(name, str) or not isinstance(expected, str):
            return False, {}
        path = directory / "mods" / name
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False, actual
        digest = file_sha256(path)
        actual[name] = digest
        if digest != expected.lower():
            return False, actual
    has_api = FABRIC_API_FILENAME in actual
    has_rules = sum(name.startswith(RULE_MOD_PREFIX) for name in actual) == 1
    return has_api and has_rules, actual


def _server_properties(seed: int) -> str:
    values = {
        "difficulty": "normal",
        "enable-command-block": "true",
        "enforce-secure-profile": "false",
        "gamemode": "survival",
        "generate-structures": "true",
        "level-name": "scientist-world",
        "level-seed": str(seed),
        "max-players": "4",
        "motd": "Minecraft Scientist deterministic research server",
        "online-mode": "false",
        "simulation-distance": "5",
        "spawn-animals": "false",
        "spawn-monsters": "false",
        "spawn-npcs": "false",
        "spawn-protection": "0",
        "view-distance": "8",
    }
    return "\n".join(f"{key}={values[key]}" for key in sorted(values)) + "\n"
