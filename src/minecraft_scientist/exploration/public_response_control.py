"""Bounded opaque-control selection from already validated public novelty outcomes.

Beta counts are a deterministic exploration heuristic, not calibrated probabilities
of physical movement, map correctness or arrival. No private heading is an input.
"""

from fractions import Fraction


def select_public_response(history, *, required_distinct_from=None):
    if required_distinct_from is not None and (
        type(required_distinct_from) is not int or not 0 <= required_distinct_from < 4
    ):
        raise ValueError("distinct future control requires an opaque handle or None")
    history = tuple(history)
    if len(history) > 11:
        raise ValueError("response history exceeds the registered eleven-trial budget")
    counts, novel = [0] * 4, [0] * 4
    previous_handle, consecutive_trials = None, 0
    for trial, record in enumerate(history):
        if type(record) is not dict:
            raise ValueError(
                "response history requires exact chronological public trials"
            )
        handle = record.get("handle")
        if (
            type(handle) is not int
            or not 0 <= handle < 4
            or type(record.get("selected")) is not bool
            or type(record.get("trial")) is not int
            or record["trial"] != trial
        ):
            raise ValueError(
                "response history requires exact chronological public trials"
            )
        counts[handle] += 1
        novel[handle] += int(record["selected"])
        consecutive_trials = consecutive_trials + 1 if handle == previous_handle else 1
        previous_handle = handle
        if consecutive_trials > 4:
            raise ValueError("response history bypassed the four-trial control bound")
    excluded = previous_handle if consecutive_trials == 4 else None
    statistics = [
        {
            "handle": handle,
            "trials": counts[handle],
            "novel_views": novel[handle],
            "score_numerator": novel[handle] + 1,
            "score_denominator": counts[handle] + 2,
            "eligible": handle != excluded and handle != required_distinct_from,
        }
        for handle in range(4)
    ]
    handle = max(
        (
            handle
            for handle in range(4)
            if handle != excluded and handle != required_distinct_from
        ),
        key=lambda handle: (
            Fraction(novel[handle] + 1, counts[handle] + 2),
            -counts[handle],
            -handle,
        ),
    )
    return {
        "handle": handle,
        "observed_trials": len(history),
        "previous_handle": previous_handle,
        "consecutive_trials": consecutive_trials,
        "excluded_handle": excluded,
        "required_distinct_from": required_distinct_from,
        "statistics": statistics,
        "basis": "chronological-public-novelty-only-v1",
    }
