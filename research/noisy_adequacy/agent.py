"""Online shadow monitor: never feeds adequacy/evaluator outputs to the planner."""

from fractions import Fraction

from minecraft_scientist.protocols import AgentWorldView
from research.noisy_competition.agent import run_agent
from research.noisy_competition.belief import ModelBank

from .monitor import FamilyAdequacyMonitor


def run_monitored_agent(
    view,
    candidates,
    *,
    costs,
    budget=96,
    horizon=4,
    policy="mechanism_risk",
    seed=0,
    alpha=Fraction(1, 20),
    ledger=None,
    monitor_ledger=None,
):
    bank = ModelBank(candidates)
    monitor = FamilyAdequacyMonitor(
        bank.programs, tuple(c.key for c in candidates), alpha=alpha
    )
    shadow = [] if monitor_ledger is None else monitor_ledger
    if type(shadow) is not list or shadow:
        raise ValueError("fresh caller-owned monitor evidence list required")

    def reset():
        entry = {"kind": "reset", "before": monitor.snapshot()}
        shadow.append(entry)
        observation = view.begin_trial()
        entry["observation"] = observation.model_dump(mode="json")
        monitor.reset_state()
        entry["after"] = monitor.snapshot()
        return observation

    def step(action):
        entry = {"kind": "step", "before": monitor.snapshot()}
        shadow.append(entry)
        # Frozen alternative and component predictions precede actual dispatch.
        entry["prediction_before_dispatch"] = monitor.prepare(action.target_id)
        record = view.step(action)
        entry["transition"] = record.model_dump(mode="json")
        entry["after"] = monitor.finish(record.event.details["signal"])
        return record

    wrapped = AgentWorldView(
        observe=view.observe,
        available=view.available_actions,
        step=step,
        begin_trial=reset,
    )
    result = run_agent(
        wrapped,
        candidates,
        costs=costs,
        budget=budget,
        horizon=horizon,
        policy=policy,
        seed=seed,
        ledger=ledger,
    )
    cumulative_cost, first_cost = 0, None
    for entry in shadow:
        cumulative_cost += (
            1
            if entry["kind"] == "reset"
            else costs[entry["prediction_before_dispatch"]["issued_command"]]
        )
        if first_cost is None and entry["after"]["family_rejected"]:
            first_cost = cumulative_cost
    if cumulative_cost != result["spent"]:
        raise ValueError("monitor actual control cost coverage mismatch")
    return {
        "schema": "online-shadow-family-adequacy-agent-v1",
        "agent": result,
        "monitor": monitor.snapshot(),
        "monitor_evidence": shadow,
        "first_rejection_actual_cost": first_cost,
        "planner_changed": False,
        "family_rejection_is_structure_attribution": False,
        "accepted_theory": None,
    }
