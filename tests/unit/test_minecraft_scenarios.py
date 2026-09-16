from __future__ import annotations

import io
from pathlib import Path

import pytest

from minecraft_scientist.minecraft import (
    FallEquationScenario,
    MinecraftServerProcess,
    MiningArenaScenario,
    ScientistCurriculumScenario,
)


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()

    @staticmethod
    def poll() -> None:
        return None


def test_mining_arena_commands_are_deterministic_and_single_line() -> None:
    first = MiningArenaScenario(username="ScientistBot").commands()
    second = MiningArenaScenario(username="ScientistBot").commands()
    assert first == second
    assert all(command and "\n" not in command for command in first)
    assert "setblock 2 80 0 iron_ore" in first
    assert "give ScientistBot iron_pickaxe 1" in first


def test_mining_arena_rejects_command_injection() -> None:
    with pytest.raises(ValueError, match="player name"):
        MiningArenaScenario(username="bot\nstop").commands()


def test_curriculum_scenario_covers_three_live_symbolic_mechanisms() -> None:
    commands = ScientistCurriculumScenario(seed=19, variant="shifted").commands()
    assert commands[:2] == ("scientist reset 19", "scientist variant shifted")
    assert "setblock 2 80 3 lodestone" in commands
    assert "setblock 5 80 -2 chiseled_copper" in commands
    assert "give ScientistBot raw_copper 4" in commands
    assert "give ScientistBot oak_log 4" in commands
    assert all(command and "\n" not in command for command in commands)


def test_curriculum_scenario_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        ScientistCurriculumScenario(variant="invented").commands()


def test_fall_equation_scenario_has_ten_opaque_physical_fixtures() -> None:
    scenario = FallEquationScenario(seed=31, variant="shifted")
    commands = scenario.commands()
    assert list(scenario.platform_positions()) == list(range(1, 11))
    assert sum(command.endswith("white_concrete") for command in commands) == 10
    assert "give ScientistBot leather_boots 1" in commands
    assert commands[:2] == ("scientist reset 31", "scientist variant shifted")


def test_server_console_accepts_only_one_line(tmp_path: Path) -> None:
    server = MinecraftServerProcess(tmp_path)
    fake = FakeProcess()
    server._process = fake  # type: ignore[assignment]
    server.command("time set day")
    assert fake.stdin.getvalue() == "time set day\n"
    with pytest.raises(ValueError, match="one non-empty line"):
        server.command("say first\nstop")
