"""First-eligible public stable waiting with a separate bounded recovery ledger.

An exhausted window stays exhausted. Extra windows change the declared control
budget, never pixel support/stability gates or an existing experiment's records.
"""


def settle_with_recovery(result, phase, step, ordinal, stability):
    if phase not in ("initial", "acquisition", "future"):
        raise ValueError("recovery waits require a public experiment phase")
    start, window_start, streak, tick = ordinal(), ordinal(), 0, 0
    recovery = None
    used_at_start = len(result["recovery_wait_windows"])

    def close(status):
        if recovery is not None:
            recovery.update(
                interval=[recovery["interval"][0], ordinal()], status=status
            )

    while True:
        if not step("wait", None):
            close(result["terminal"])
            return False
        tick += 1
        streak = streak + 1 if result["flow_prefixes"][-1]["stable"] else 0
        if tick >= stability.minimum_wait_ticks and streak >= stability.stable_prefixes:
            result["settles"].append(
                {
                    "phase": phase,
                    "interval": [start, ordinal()],
                    "stable_tail": [
                        [ordinal() - 3, ordinal() - 2],
                        [ordinal() - 2, ordinal() - 1],
                        [ordinal() - 1, ordinal()],
                    ],
                    "recovery_windows": len(result["recovery_wait_windows"])
                    - used_at_start,
                }
            )
            close("stable")
            return True
        if ordinal() - window_start < stability.maximum_wait_ticks:
            continue
        close("exhausted")
        result["terminal"] = f"{phase}_wait_exhausted"
        available = len(result["recovery_wait_windows"]) < 2 and ordinal() < 288
        result["failed_wait_windows"].append(
            {
                "phase": phase,
                "interval": [window_start, ordinal()],
                "total_interval": [start, ordinal()],
                "recorded_terminal": result["terminal"],
                "recovery_available": available,
            }
        )
        if not available:
            if ordinal() >= 288:
                result["terminal"] = "budget_exhausted"
            return False
        window_start = ordinal()
        recovery = {
            "phase": phase,
            "interval": [window_start, window_start],
            "source_failed_window": len(result["failed_wait_windows"]) - 1,
            "status": "running",
        }
        result["recovery_wait_windows"].append(recovery)
        result["terminal"] = "recovering"
