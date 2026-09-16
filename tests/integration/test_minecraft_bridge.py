import shutil

import pytest

from minecraft_scientist.minecraft import MinecraftBridgeClient, check_backend


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_mineflayer_jsonl_bridge_ping() -> None:
    with MinecraftBridgeClient(timeout=10) as client:
        response = client.request("ping")
    assert response == {"protocol": 1, "bridge": "mineflayer", "connected": False}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_minecraft_backend_dependencies_are_ready() -> None:
    report = check_backend()
    assert report["ready_for_connection"]
    assert report["java_version"].startswith('openjdk version "21')
    assert report["node_version"].startswith("v22")
