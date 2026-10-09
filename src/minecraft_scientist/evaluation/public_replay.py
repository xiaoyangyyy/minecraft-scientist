"""Replay recorded public capabilities without any private-world access.

Policy replay is complemented by independent raw-ledger checks at callers; it
alone cannot rule out shared defects in the policy and recorder.
"""

import json

from minecraft_scientist.protocols import (
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def replay_capability(run):
    controls = [e for e in run["events"] if e["kind"] in ("reset", "step")]
    cursor, current = 0, None

    def reset():
        nonlocal cursor, current
        if cursor >= len(controls) or controls[cursor]["kind"] != "reset":
            raise ValueError("public replay reset chronology mismatch")
        current = Observation.model_validate(controls[cursor]["observation"])
        cursor += 1
        return current

    def available():
        if current is None:
            raise ValueError("public replay actions before reset")
        return tuple(
            PrimitiveAction(kind=o.affordances[0], target_id=o.object_id)
            for o in current.objects
        )

    def step(action):
        nonlocal cursor, current
        if cursor >= len(controls) or controls[cursor]["kind"] != "step":
            raise ValueError("public replay action chronology mismatch")
        record = TransitionRecord.model_validate(controls[cursor]["transition"])
        if record.executed_action != action or record.state_before != current:
            raise ValueError("public replay action/state mismatch")
        cursor += 1
        current = record.state_after
        return record

    return AgentWorldView(
        observe=lambda: current, available=available, step=step, begin_trial=reset
    ), lambda: cursor == len(controls)
