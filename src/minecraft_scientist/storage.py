"""Deterministic run artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from minecraft_scientist.protocols import (
    EvaluationReport,
    ExperimentResult,
    RunConfig,
    SymbolPrimitive,
    TheoryVersion,
)


class RunStore:
    def __init__(self, path: Path, *, overwrite: bool = False) -> None:
        self.path = path
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for filename in (
                "config.json",
                "events.jsonl",
                "theories.jsonl",
                "symbols.json",
                "report.json",
            ):
                candidate = path / filename
                if candidate.exists():
                    candidate.unlink()

    def write_config(self, config: RunConfig) -> None:
        self._write_json("config.json", config)

    def write_experiments(self, experiments: tuple[ExperimentResult, ...]) -> None:
        records = [
            transition
            for experiment in experiments
            for transition in experiment.transitions
        ]
        self._write_jsonl("events.jsonl", records)

    def write_theories(self, theories: tuple[TheoryVersion, ...]) -> None:
        self._write_jsonl("theories.jsonl", theories)

    def write_symbols(self, symbols: tuple[SymbolPrimitive, ...]) -> None:
        self._write_raw_json(
            "symbols.json", [s.model_dump(mode="json") for s in symbols]
        )

    def write_report(self, report: EvaluationReport) -> None:
        self._write_json("report.json", report)

    def _write_json(self, filename: str, value: BaseModel) -> None:
        self._write_raw_json(filename, value.model_dump(mode="json"))

    def _write_jsonl(
        self, filename: str, values: list[BaseModel] | tuple[BaseModel, ...]
    ) -> None:
        payload = "".join(
            json.dumps(
                value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
            )
            + "\n"
            for value in values
        )
        (self.path / filename).write_text(payload, encoding="utf-8")

    def _write_raw_json(self, filename: str, value: object) -> None:
        payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        (self.path / filename).write_text(payload, encoding="utf-8")
