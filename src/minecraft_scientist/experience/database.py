"""Append-only SQLite storage for observations and controlled interventions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from minecraft_scientist.protocols import TransitionRecord


class ExperienceDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                transition_id TEXT NOT NULL UNIQUE,
                mechanism TEXT NOT NULL,
                compliant INTEGER NOT NULL CHECK (compliant IN (0, 1)),
                controller_failure TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_mechanism "
            "ON transitions(mechanism, sequence)"
        )
        self._connection.commit()

    def append(self, mechanism: str, transition: TransitionRecord) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO transitions "
                "(transition_id, mechanism, compliant, controller_failure, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    transition.transition_id,
                    mechanism,
                    int(transition.intervention_compliance),
                    transition.controller_failure,
                    transition.model_dump_json(),
                ),
            )

    def iter_transitions(
        self, mechanism: str | None = None
    ) -> Iterator[TransitionRecord]:
        if mechanism is None:
            cursor = self._connection.execute(
                "SELECT payload_json FROM transitions ORDER BY sequence"
            )
        else:
            cursor = self._connection.execute(
                "SELECT payload_json FROM transitions "
                "WHERE mechanism = ? ORDER BY sequence",
                (mechanism,),
            )
        for (payload,) in cursor:
            yield TransitionRecord.model_validate_json(payload)

    def count(self, mechanism: str | None = None) -> int:
        if mechanism is None:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM transitions"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM transitions WHERE mechanism = ?",
                (mechanism,),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
