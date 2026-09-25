from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from minecraft_scientist.minecraft import server as server_module
from minecraft_scientist.minecraft.server import (
    MinecraftEulaRequired,
    prepare_fabric_server,
    prepare_server,
    server_status,
)


def _jar_bytes(marker: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
        archive.writestr("marker.txt", marker)
    return stream.getvalue()


def test_server_preparation_requires_explicit_eula_acceptance(tmp_path: Path) -> None:
    directory = tmp_path / "server"
    with pytest.raises(MinecraftEulaRequired):
        prepare_server(directory, accept_eula=False)
    assert not directory.exists()


def test_server_preparation_pins_and_verifies_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"test server jar"
    expected_sha1 = hashlib.sha1(payload, usedforsecurity=False).hexdigest()

    def fake_retrieve(_url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    monkeypatch.setattr(server_module, "SERVER_SHA1", expected_sha1)
    monkeypatch.setattr(server_module, "SERVER_SIZE", len(payload))
    monkeypatch.setattr(server_module.urllib.request, "urlretrieve", fake_retrieve)
    directory = tmp_path / "server"

    report = prepare_server(directory, accept_eula=True, seed=73)

    assert report["ready"] is True
    assert (directory / "eula.txt").read_text(encoding="utf-8").endswith("eula=true\n")
    properties = (directory / "server.properties").read_text(encoding="utf-8")
    assert "level-seed=73\n" in properties
    assert "online-mode=false\n" in properties
    assert server_status(directory)["jar_valid"] is True


def test_fabric_preparation_requires_eula_before_creating_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "server"
    with pytest.raises(MinecraftEulaRequired):
        prepare_fabric_server(
            directory,
            accept_eula=False,
            mod_bundle=tmp_path / "missing",
        )
    assert not directory.exists()


def test_fabric_preparation_locks_launcher_and_server_mods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_payload = b"test server jar"
    launcher_payload = _jar_bytes("launcher")
    expected_sha1 = hashlib.sha1(server_payload, usedforsecurity=False).hexdigest()

    def fake_retrieve(url: str, destination: Path) -> None:
        destination.write_bytes(
            launcher_payload
            if url == server_module.FABRIC_SERVER_URL
            else server_payload
        )

    monkeypatch.setattr(server_module, "SERVER_SHA1", expected_sha1)
    monkeypatch.setattr(server_module, "SERVER_SIZE", len(server_payload))
    monkeypatch.setattr(server_module.urllib.request, "urlretrieve", fake_retrieve)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / server_module.FABRIC_API_FILENAME).write_bytes(_jar_bytes("api"))
    (bundle / "scientist-world-rules-0.1.0.jar").write_bytes(_jar_bytes("rules"))

    report = prepare_fabric_server(
        tmp_path / "server",
        accept_eula=True,
        mod_bundle=bundle,
        seed=11,
    )

    assert report["fabric_ready"] is True
    assert report["fabric_launcher_valid"] is True
    assert report["fabric_mods_valid"] is True
    lock = json.loads(
        (tmp_path / "server" / "scientist-mods.lock.json").read_text(encoding="utf-8")
    )
    assert sorted(lock) == [
        server_module.FABRIC_API_FILENAME,
        "scientist-world-rules-0.1.0.jar",
    ]

    installed_mod = tmp_path / "server" / "mods" / "scientist-world-rules-0.1.0.jar"
    installed_mod.write_bytes(_jar_bytes("tampered"))
    assert server_status(tmp_path / "server")["fabric_ready"] is False


def test_server_environment_isolates_windows_java_ipc(tmp_path: Path) -> None:
    directory = tmp_path / "server"

    environment = server_module._server_environment(directory)

    java_tmp = directory.resolve() / ".java-tmp"
    assert java_tmp.is_dir()
    assert environment["TEMP"] == str(java_tmp)
    assert environment["TMP"] == str(java_tmp)
    assert f"-Djava.io.tmpdir={java_tmp}" in environment["JAVA_TOOL_OPTIONS"]
    assert f"-Djdk.net.unixdomain.tmpdir={java_tmp}" in environment["JAVA_TOOL_OPTIONS"]
    assert "-Djava.net.preferIPv4Stack=true" in environment["JAVA_TOOL_OPTIONS"]
