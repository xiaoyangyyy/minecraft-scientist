"""Public policy replay first, then separate raw component execution audit."""

import json

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    Observation,
    PrimitiveAction,
    TransitionRecord,
)

from .agent import run_agent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PublicReplay:
    def __init__(self, events):
        self.controls = [e for e in events if e["kind"] in ("reset", "step")]
        self.index, self.current = 0, None

    def begin_trial(self):
        if (
            self.index >= len(self.controls)
            or self.controls[self.index]["kind"] != "reset"
        ):
            raise ValueError("public replay reset order mismatch")
        payload = self.controls[self.index]["observation"]
        self.current = Observation.model_validate(payload)
        if canonical(self.current.model_dump(mode="json")) != canonical(payload):
            raise ValueError("reset receipt type/canonical mismatch")
        self.index += 1
        return self.current

    def available_actions(self):
        if self.current is None:
            raise ValueError("reset required")
        return tuple(
            PrimitiveAction(kind=ActionKind.INTERACT, target_id=o.object_id)
            for o in self.current.objects
            if ActionKind.INTERACT in o.affordances
        )

    def step(self, action):
        if (
            self.index >= len(self.controls)
            or self.controls[self.index]["kind"] != "step"
        ):
            raise ValueError("public replay action order mismatch")
        payload = self.controls[self.index]["transition"]
        record = TransitionRecord.model_validate(payload)
        if (
            canonical(record.model_dump(mode="json")) != canonical(payload)
            or record.executed_action != action
            or record.state_before != self.current
        ):
            raise ValueError("public replay typed receipt/action/clock mismatch")
        self.current = record.state_after
        self.index += 1
        return record

    def view(self):
        return AgentWorldView(
            observe=lambda: self.current,
            available=self.available_actions,
            step=self.step,
            begin_trial=self.begin_trial,
        )


def verify_public(candidates, config, result):
    replay = PublicReplay(result["evidence"])
    expected = run_agent(
        replay.view(),
        candidates,
        costs=config["costs"],
        budget=config["budget"],
        horizon=config["horizon"],
        policy=config["policy"],
        seed=config["seed"],
    )
    if canonical(result) != canonical(expected) or replay.index != len(replay.controls):
        raise ValueError(
            "public policy/prediction/posterior/denominator replay mismatch"
        )
    spent = sum(e["cost"] for e in result["evidence"] if e["kind"] in ("reset", "step"))
    if spent != result["spent"] or spent > config["budget"]:
        raise ValueError("actual cost mismatch")
    dispatch = tuple(
        ("reset",) if e["kind"] == "reset" else ("step", e["issued_command"])
        for e in replay.controls
    )
    return {
        "valid": True,
        "actual_cost": spent,
        "resets": sum(e[0] == "reset" for e in dispatch),
        "atomic_actions": sum(e[0] == "step" for e in dispatch),
        "dispatch": dispatch,
        "scope": "complete supplied public adaptive policy/evidence replay before private scoring; conditional finite-family posterior, not accepted theory",
    }
