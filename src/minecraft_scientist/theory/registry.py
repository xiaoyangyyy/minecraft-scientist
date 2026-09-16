"""Append-only theory histories with explicit legal status transitions."""

from __future__ import annotations

from collections import defaultdict

from minecraft_scientist.protocols import TheoryIR, TheoryStatus, TheoryVersion


class TheoryRegistry:
    def __init__(self) -> None:
        self._histories: dict[str, list[TheoryVersion]] = defaultdict(list)

    def propose(
        self,
        theory_id: str,
        ir: TheoryIR,
        *,
        evidence_ids: tuple[str, ...],
        validation_score: float,
        confidence: float,
        created_by: str = "heuristic",
        parent_version: int | None = None,
    ) -> TheoryVersion:
        expected = len(self._histories[theory_id]) + 1
        if parent_version is not None and parent_version >= expected:
            raise ValueError("parent version must already exist")
        version = TheoryVersion(
            theory_id=theory_id,
            version=expected,
            ir=ir,
            parent_version=parent_version,
            evidence_ids=evidence_ids,
            validation_score=validation_score,
            confidence=confidence,
            status=TheoryStatus.PROPOSED,
            created_by=created_by,
        )
        self._histories[theory_id].append(version)
        return version

    def transition(
        self,
        theory_id: str,
        version_number: int,
        status: TheoryStatus,
        *,
        counterexample_ids: tuple[str, ...] | None = None,
    ) -> TheoryVersion:
        current = self.get(theory_id, version_number)
        legal = {
            TheoryStatus.PROPOSED: {TheoryStatus.PROVISIONAL, TheoryStatus.REJECTED},
            TheoryStatus.PROVISIONAL: {TheoryStatus.ACCEPTED, TheoryStatus.REJECTED},
            TheoryStatus.ACCEPTED: {TheoryStatus.CHALLENGED},
            TheoryStatus.CHALLENGED: {TheoryStatus.SUPERSEDED, TheoryStatus.ACCEPTED},
            TheoryStatus.SUPERSEDED: set(),
            TheoryStatus.REJECTED: set(),
        }
        if status not in legal[current.status]:
            raise ValueError(f"illegal theory transition: {current.status} -> {status}")
        updated = current.model_copy(
            update={
                "status": status,
                "counterexample_ids": (
                    counterexample_ids
                    if counterexample_ids is not None
                    else current.counterexample_ids
                ),
            }
        )
        self._histories[theory_id][version_number - 1] = updated
        return updated

    def accept_proposal(self, theory_id: str, version_number: int) -> TheoryVersion:
        self.transition(theory_id, version_number, TheoryStatus.PROVISIONAL)
        return self.transition(theory_id, version_number, TheoryStatus.ACCEPTED)

    def get(self, theory_id: str, version_number: int) -> TheoryVersion:
        try:
            return self._histories[theory_id][version_number - 1]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"unknown theory {theory_id}@v{version_number}") from exc

    def latest(self, theory_id: str) -> TheoryVersion:
        if not self._histories[theory_id]:
            raise KeyError(f"unknown theory: {theory_id}")
        return self._histories[theory_id][-1]

    def history(self, theory_id: str) -> tuple[TheoryVersion, ...]:
        return tuple(self._histories.get(theory_id, ()))

    def all_versions(self) -> tuple[TheoryVersion, ...]:
        return tuple(
            version
            for theory_id in sorted(self._histories)
            for version in self._histories[theory_id]
        )
