# Online fixed-family adequacy shadow monitor

Existing universal-inference e-process, not a novel learner or a replacement
theory. Freeze per-model/uniform-alternative predictions before actual actions;
track cumulative likelihood across every paid reset. Denominator is maximum
fixed-null likelihood, NOT a prior mixture or per-step maximum product.
Rejection is sticky, nonrejection is not adequacy, and no new theory is promoted.

The shadow wrapper executes the unchanged Agent and real AgentWorldView. It
never sends rejection/evaluator results to the planner. Independent raw-IR
monitor reconstruction checks every forecast, likelihood, reset, threshold,
first actual rejection cost and complete trajectory, without monitor/prediction
runtime calls. All public policy/monitor checks precede private raw scoring.

Default complete DEVELOPMENT matrix: two seeds, five stationary conditions,
three policies, budget96/horizon4: 30 real online-monitored Agent trajectories.
Alpha=1/20 is per trajectory under a correctly specified fixed conditional law,
not a simultaneous risk bound across all runs, changing epochs or languages.
These development worlds were inspected previously; this is not a fresh holdout.

Run `uv run python scripts/run_noisy_adequacy.py --output NEW_DIR`.
For full independent verification add `--verify-only --manifest-sha256 PIN`.
See `docs/noisy-adequacy-contract.md`. Unknown structure, shadow promotion,
online registry revision, statistical/structural holdouts, H4 and Minecraft/RGB
remain open. Don't claim zero-error revision from a null accepted-theory field.
