"""Theory validation helpers kept separate from candidate generation."""

from __future__ import annotations

from collections.abc import Iterable

from minecraft_scientist.protocols import TheoryIR

from .ir import TheoryInterpreter
from .scoring import intervention_error, theory_score


class TheoryValidator:
    @staticmethod
    def validate_tool_theory(
        ir: TheoryIR,
        observations: Iterable[tuple[str, str, bool]],
    ) -> tuple[float, float]:
        rows = list(observations)
        predictions = [
            bool(
                TheoryInterpreter.predict(ir, tool_id=tool, target_id=target)["success"]
            )
            for tool, target, _ in rows
        ]
        outcomes = [outcome for _, _, outcome in rows]
        error = intervention_error(predictions, outcomes)
        return theory_score(ir, predictions, outcomes), 1.0 - error
