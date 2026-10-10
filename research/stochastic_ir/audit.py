"""Separate raw seeded component replay; no runtime/compiler/world imports."""

import json
import random
from fractions import Fraction
from math import lcm


def audit_trace(spec, trace, *, expected_dispatch):
    """Require evaluator-owned validated spec; compare every public receipt.

    This checks trace execution, not private-file access or planner validity.
    Caller must not supply/rewrite the truth from an Agent-produced report.
    """
    if (
        type(expected_dispatch) is not tuple
        or not 1 <= len(expected_dispatch) <= 10_000
        or type(trace) is not list
        or len(trace) != len(expected_dispatch)
    ):
        raise ValueError("trace must cover the complete frozen dispatch sequence")

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    rng = random.Random(spec["seed"])
    mechanism, controller, sensor = (
        spec[k] for k in ("mechanism", "controller", "sensor")
    )
    controls = {
        (s, a): sorted(
            (
                r
                for r in controller["transitions"]
                if r["source"] == s and r["action"] == a
            ),
            key=lambda r: (r["target"], r["output"]),
        )
        for s in controller["states"]
        for a in controller["actions"]
    }
    sensors = {
        a: sorted(
            (r for r in sensor["transitions"] if r["action"] == a),
            key=lambda r: (r["target"], r["output"]),
        )
        for a in sensor["actions"]
    }
    mr = {(r["source"], r["action"]): r for r in mechanism["transitions"]}
    initial = sorted(controller["initial"], key=lambda r: r["state"])

    def choose(rows):
        denominator = lcm(*(r["probability"]["denominator"] for r in rows))
        point = Fraction(rng.randrange(denominator), denominator)
        cumulative = Fraction(0)
        for row in rows:
            p = row["probability"]
            cumulative += Fraction(p["numerator"], p["denominator"])
            if point < cumulative:
                return row
        raise ValueError("raw replay lost mass")

    def observation():
        return {
            "episode_id": episode,
            "step": step,
            "inventory": {},
            "selected_id": None,
            "agent_state": {
                "signal": min(sensor["outputs"])
                if event is None
                else event["details"]["signal"]
            },
            "objects": [
                {"object_id": a, "affordances": ["interact"], "observable_features": {}}
                for a in sorted(controller["actions"])
            ],
            "last_event": event,
        }

    episode, step, event, resets, actions = -1, 0, None, 0, 0
    for entry, dispatch in zip(trace, expected_dispatch, strict=True):
        if entry.get("kind") == "reset" and set(entry) == {"kind", "observation"}:
            if dispatch != ("reset",):
                raise ValueError("reset outside the frozen dispatch sequence")
            ms, cs = mechanism["initial_state"], choose(initial)["state"]
            episode, step, event = episode + 1, 0, None
            if canonical(entry["observation"]) != canonical(observation()):
                raise ValueError("reset receipt mismatch")
            resets += 1
        elif entry.get("kind") == "step" and set(entry) == {"kind", "transition"}:
            record = entry["transition"]
            issued = record["executed_action"]
            command = issued["target_id"]
            if dispatch != ("step", command):
                raise ValueError("action outside the frozen dispatch sequence")
            if (
                episode < 0
                or issued
                != {
                    "kind": "interact",
                    "target_id": command,
                    "item_id": None,
                    "ticks": 1,
                }
                or command not in controller["actions"]
            ):
                raise ValueError("invalid issued command")
            before = observation()
            delivery = choose(controls[cs, command])
            cs, latent = delivery["target"], spec["idle_output"]
            if delivery["output"] != "_no_delivery_":
                move = mr[ms, delivery["output"]]
                ms, latent = move["target"], move["output"]
            measured = choose(sensors[latent])["output"]
            step += 1
            event = {
                "event_type": "public_signal",
                "target_id": None,
                "success": True,
                "duration": 1,
                "output_id": None,
                "details": {"signal": measured},
            }
            expected = {
                "transition_id": f"episode-{episode}-step-{step}",
                "context": {
                    "intervention_compliance_scope": "issued-command",
                    "treatment_verification": "unknown",
                },
                "state_before": before,
                "intended_intervention": None,
                "executed_action": issued,
                "intervention_compliance": True,
                "state_after": observation(),
                "observed_delta": {"signal": measured},
                "event": event,
                "possible_confounders": [
                    "latent-actuator-delivery",
                    "latent-mechanism-state",
                    "measurement-noise",
                ],
                "controller_failure": None,
                "provenance": {"schema": "stochastic-component-world-v1"},
            }
            if canonical(record) != canonical(expected):
                raise ValueError("public transition/raw component replay mismatch")
            actions += 1
        else:
            raise ValueError("unknown trace event or extra fields")
    if not resets:
        raise ValueError("empty/no-reset trace cannot certify execution")
    return {
        "valid": True,
        "resets": resets,
        "atomic_actions": actions,
        "scope": "complete supplied raw seeded component trace; not planner, probability calibration or formal scientific acceptance",
    }
