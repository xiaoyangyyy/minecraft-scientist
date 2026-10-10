# Stochastic semantics in the shared TheoryIR

This staged isolated runtime extends `finite_transducer` in the existing
`TheoryIR` with a bounded rational stochastic language. It does not create a
second theory format or learner. The default main interpreter deliberately still
rejects this language; integration must follow the current immutable experiment
batch and separate main-package regression. No frozen learner file is edited.

Delivery is latent, control can have persistent hidden modes, and measured
signals can be noisy. Exact public-history filtering conditions joint latent
state; it never silently resets after each action or reads actual delivery/state.
The joint compiler combines a deterministic mechanism, stochastic controller and
memoryless sensor. An independent component world executes real capabilities and
a separate seeded raw auditor checks every frozen dispatch/receipt. Independent
factor-path enumeration checks compiler probabilities, not merely its own filter.

`uv run pytest research/stochastic_ir` checks the contracts. Run actual execution
with `uv run python scripts/check_stochastic_semantics.py --seed 0 --output NEW_DIR`.
All four fixtures deliberately supply the correct model: they verify semantics,
not unknown-controller discovery, statistical calibration or scientific novelty.
Partial failures preserve evidence without a completion manifest. See
`docs/stochastic-ir-contract.md` for limits and open research/system gates.
