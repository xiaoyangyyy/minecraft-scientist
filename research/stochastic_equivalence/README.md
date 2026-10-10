# Exact public stochastic-program equivalence

An existing weighted-automaton algorithm implemented over the shared rational
TheoryIR, not a novel learner. It compares complete output-prefix probability
series for all finite words, given explicit model state beliefs and an allowed
public action subset. It returns an exact distinguishing event or a reachable
linear-space closure certificate. It does not equate bounded horizon search
failure with global equivalence, and does not equate model equality with truth.

Independent proof checking reconstructs raw branch probabilities, verifies
caller-pinned beliefs/action scope, and checks initial inclusion, reachability,
linear independence, zero terminal values and every operator image. No search
or compiled prediction/filter runtime is invoked by the checker. Shared IR
schema validation is reused; reported search compute is not independently
replayed, and is labelled separately from checker compute.

Run `uv run python scripts/check_stochastic_equivalence.py --output NEW_DIR`.
Default complete PUBLIC proof matrix: two naming seeds, all 153 pairs of the
18 public models per seed. Zero physical trials; no evaluator truth/trajectory
is read. This does not certify any Agent, noisy finite-sample stopping rule,
unknown grammar, causal uniqueness, minimum intervention cost or theory version.
See `docs/stochastic-equivalence-contract.md` for the exact limits and scope.
