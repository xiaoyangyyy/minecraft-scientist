"""Runtime checks and one-shot connection diagnostics."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .bridge import MinecraftBridgeClient


def discover_java() -> Path | None:
    found = shutil.which("java")
    if found:
        return Path(found)
    root = Path("C:/Program Files/Eclipse Adoptium")
    matches = sorted(root.glob("jdk-21*/bin/java.exe"), reverse=True)
    return matches[0] if matches else None


def check_backend() -> dict[str, Any]:
    java = discover_java()
    node = shutil.which("node")
    node_version = (
        subprocess.check_output([node, "--version"], text=True).strip()
        if node
        else None
    )
    java_version = None
    if java is not None:
        completed = subprocess.run(
            [str(java), "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        java_version = (completed.stderr or completed.stdout).splitlines()[0]
    with MinecraftBridgeClient() as client:
        ping = client.request("ping")
    return {
        "java": str(java) if java else None,
        "java_version": java_version,
        "node": node,
        "node_version": node_version,
        "bridge": ping,
        "ready_for_connection": bool(java and node and ping.get("protocol") == 1),
    }
