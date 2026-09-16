"""Deterministic vanilla command scenarios controlled outside the agent boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class CommandChannel(Protocol):
    def command(self, value: str) -> None: ...


@dataclass(frozen=True, slots=True)
class MiningArenaScenario:
    username: str = "ScientistBot"
    origin_x: int = 0
    origin_y: int = 80
    origin_z: int = 0

    def commands(self) -> tuple[str, ...]:
        if re.fullmatch(r"[A-Za-z0-9_]{1,16}", self.username) is None:
            raise ValueError("username must be a valid Java Edition player name")
        x, y, z = self.origin_x, self.origin_y, self.origin_z
        return (
            "time set day",
            "weather clear",
            f"kill @e[type=item,distance=..64,x={x},y={y},z={z}]",
            f"clear {self.username}",
            f"effect clear {self.username}",
            f"gamemode survival {self.username}",
            f"fill {x - 8} {y - 2} {z - 8} {x + 8} {y + 6} {z + 8} air",
            f"fill {x - 8} {y - 2} {z - 8} {x + 8} {y - 1} {z + 8} stone",
            f"setblock {x + 2} {y} {z - 2} stone",
            f"setblock {x + 2} {y} {z} iron_ore",
            f"setblock {x + 2} {y} {z + 2} diamond_ore",
            f"give {self.username} wooden_pickaxe 1",
            f"give {self.username} stone_pickaxe 1",
            f"give {self.username} iron_pickaxe 1",
            f"tp {self.username} {x} {y} {z} -90 0",
        )

    def apply(self, channel: CommandChannel) -> None:
        for command in self.commands():
            channel.command(command)


@dataclass(frozen=True, slots=True)
class ScientistCurriculumScenario:
    """Resettable live arena covering rule, automaton, and program mechanisms."""

    username: str = "ScientistBot"
    seed: int = 0
    variant: str = "standard"
    origin_x: int = 0
    origin_y: int = 80
    origin_z: int = 0

    def experimental_positions(self) -> tuple[tuple[int, int, int], ...]:
        x, y, z = self.origin_x, self.origin_y, self.origin_z
        return (
            (x + 2, y, z - 4),
            (x + 2, y, z - 2),
            (x + 2, y, z),
            (x + 2, y, z + 3),
            (x + 5, y, z - 2),
            (x + 5, y, z),
            (x + 5, y, z + 2),
        )

    def commands(self) -> tuple[str, ...]:
        if re.fullmatch(r"[A-Za-z0-9_]{1,16}", self.username) is None:
            raise ValueError("username must be a valid Java Edition player name")
        if self.variant not in {"standard", "shifted"}:
            raise ValueError("variant must be standard or shifted")
        x, y, z = self.origin_x, self.origin_y, self.origin_z
        return (
            f"scientist reset {self.seed}",
            f"scientist variant {self.variant}",
            "time set day",
            "weather clear",
            f"kill @e[type=item,distance=..64,x={x},y={y},z={z}]",
            f"clear {self.username}",
            f"effect clear {self.username}",
            f"gamemode survival {self.username}",
            f"fill {x - 8} {y - 2} {z - 8} {x + 8} {y + 6} {z + 8} air",
            f"fill {x - 8} {y - 2} {z - 8} {x + 8} {y - 1} {z + 8} stone",
            f"setblock {x + 2} {y} {z - 4} stone",
            f"setblock {x + 2} {y} {z - 2} iron_ore",
            f"setblock {x + 2} {y} {z} diamond_ore",
            f"setblock {x + 2} {y} {z + 3} lodestone",
            f"setblock {x + 5} {y} {z - 2} chiseled_copper",
            f"setblock {x + 5} {y} {z} chiseled_tuff",
            f"setblock {x + 5} {y} {z + 2} polished_blackstone",
            f"give {self.username} wooden_pickaxe 1",
            f"give {self.username} stone_pickaxe 1",
            f"give {self.username} iron_pickaxe 1",
            f"give {self.username} raw_copper 4",
            f"give {self.username} oak_log 4",
            f"give {self.username} oak_planks 4",
            f"give {self.username} stick 4",
            f"tp {self.username} {x} {y} {z} -90 0",
        )

    def apply(self, channel: CommandChannel) -> None:
        for command in self.commands():
            channel.command(command)


@dataclass(frozen=True, slots=True)
class FallEquationScenario:
    """Physical opaque fixtures for server-side continuous-law interventions."""

    username: str = "ScientistBot"
    seed: int = 0
    variant: str = "standard"
    origin_x: int = 20
    origin_y: int = 80
    origin_z: int = 0

    def platform_positions(self) -> dict[int, tuple[int, int, int]]:
        return {
            height: (
                self.origin_x + 2 + ((height - 1) % 5),
                self.origin_y,
                self.origin_z + (-2 if height <= 5 else 2),
            )
            for height in range(1, 11)
        }

    def commands(self) -> tuple[str, ...]:
        if re.fullmatch(r"[A-Za-z0-9_]{1,16}", self.username) is None:
            raise ValueError("username must be a valid Java Edition player name")
        if self.variant not in {"standard", "shifted"}:
            raise ValueError("variant must be standard or shifted")
        x, y, z = self.origin_x, self.origin_y, self.origin_z
        platform_commands = tuple(
            f"setblock {px} {py} {pz} white_concrete"
            for px, py, pz in self.platform_positions().values()
        )
        return (
            f"scientist reset {self.seed}",
            f"scientist variant {self.variant}",
            f"clear {self.username}",
            f"gamemode survival {self.username}",
            f"fill {x - 2} {y - 2} {z - 5} {x + 9} {y + 3} {z + 5} air",
            f"fill {x - 2} {y - 2} {z - 5} {x + 9} {y - 1} {z + 5} stone",
            *platform_commands,
            f"give {self.username} leather_boots 1",
            f"give {self.username} chainmail_boots 1",
            f"give {self.username} iron_boots 1",
            f"tp {self.username} {x} {y} {z} -90 0",
        )

    def apply(self, channel: CommandChannel) -> None:
        for command in self.commands():
            channel.command(command)
