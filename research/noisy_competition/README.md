# Finite-family noisy competition

This extends the existing stochastic TheoryIR execution with competing factor
models, exact model/state Bayes updates and actual public-capability experiments.
Every method gets the same 18 public mechanism/controller/sensor combinations
and prior. Hidden current selection and out-of-family sensor are evaluator-only.
It is **not unknown-controller structure synthesis or an accepted theory**.

Policies: cost-normalized joint mutual information, exact mechanism Bayes-risk
reduction with MI tie-break, and uniform feasible query word/reset selection.
These are established criteria, not novelty claims; the random baseline is not
misnamed uniformly independent atomic actions. Both continuation and paid reset
queries are available. Resets clear latent-state belief but keep model evidence.

Default complete development matrix: 2 naming/seeds x 5 stationary conditions x
3 policies, budget 96 each, horizon 4. All priors, forecasts, plans, resets,
actual transitions, factor beliefs, support and costs are retained. All public
policy replays precede any private scoring; raw component replay is independent.
Positive likelihood under a wrong family is not guaranteed detectable; no
posterior-support criterion automatically promotes or certifies a theory.

Run `uv run python scripts/run_noisy_competition.py --output NEW_DIR`.
Verify with the same command plus `--verify-only --manifest-sha256 PIN`.
See `docs/noisy-competition-contract.md` for limits and unfinished full gates.
