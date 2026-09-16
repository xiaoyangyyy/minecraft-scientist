from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "mod" / "scientist-world-rules"


def test_fabric_mod_versions_and_server_only_manifest() -> None:
    properties = (MOD / "gradle.properties").read_text(encoding="utf-8")
    manifest_template = (MOD / "src/main/resources/fabric.mod.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_template.replace("${version}", "0.1.0"))
    assert "minecraft_version=1.21.11" in properties
    assert "loader_version=0.19.5" in properties
    assert "fabric_api_version=0.141.6+1.21.11" in properties
    assert manifest["environment"] == "server"
    assert "client" not in manifest["entrypoints"]


def test_ground_truth_command_is_privileged_and_agent_has_no_chat_surface() -> None:
    source = (
        MOD
        / "src/main/java/dev/minecraftscientist/worldrules/ScientistWorldRulesMod.java"
    ).read_text(encoding="utf-8")
    bridge = (ROOT / "bridge/mineflayer/server.cjs").read_text(encoding="utf-8")
    assert "Permissions.COMMANDS_ADMIN" in source
    assert 'literal("truth")' in source
    assert 'case "chat"' not in bridge
    assert "eval(" not in bridge


def test_shifted_rule_is_a_real_permutation_not_a_seed_only_label() -> None:
    source = (
        MOD
        / "src/main/java/dev/minecraftscientist/worldrules/ScientistWorldRulesMod.java"
    ).read_text(encoding="utf-8")
    assert "block == Blocks.STONE" in source
    assert "return 2;" in source
    assert "block == Blocks.IRON_ORE" in source
    assert "return 0;" in source
    assert "block == Blocks.DIAMOND_ORE" in source
    assert "return 1;" in source


def test_mod_contains_executable_automaton_and_transform_programs() -> None:
    source = (
        MOD
        / "src/main/java/dev/minecraftscientist/worldrules/ScientistWorldRulesMod.java"
    ).read_text(encoding="utf-8")
    assert "UseBlockCallback.EVENT.register" in source
    assert "ServerTickEvents.END_SERVER_TICK.register" in source
    assert "machineDelay()" in source
    assert "new Transform(Items.OAK_LOG, Items.OAK_PLANKS)" in source
    assert "new Transform(Items.OAK_PLANKS, Items.STICK)" in source
    assert "new Transform(Items.STICK, Items.TORCH)" in source
    assert 'literal("fall")' in source
    assert "fallSafeHeight()" in source
    assert "fallSlope(armor)" in source
