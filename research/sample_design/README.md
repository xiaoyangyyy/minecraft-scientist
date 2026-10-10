# Fixed-sample design arithmetic

This isolated offline tool reads **no corpus, sealed data, world, learner output,
API key or experiment score**. It does not alter the frozen 211 learner sources
or 219 analysis dependencies. It plans a binary failure endpoint per assumed IID
sampling unit, not a per-fixture rate falsely treating renamings as independent.

`uv run python scripts/plan_sample_precision.py` prints a deterministic illustrative
grid for ceilings 1/5/10%, alpha 5%, multiplicity 1/4/8, zero/one failures and
marginal power 80% at a good rate one-quarter of the ceiling. None of these values
is selected as a project acceptance standard. The hypothetical n=1/18/27 bounds
are arithmetic examples, NOT confidence bounds for the current corpus/results.

The functions also accept exact `Fraction` inputs. Integer binomial tail tests
prove the least n at a fixed allowance. Powered designs search bounded failure
allowances without rounding decision probabilities. Outward decimal brackets are
only displays. Bounds fail explicitly rather than silently clipping a design.

`uv run pytest research/sample_design` verifies exact independent finite-enumeration
oracles and brute-force minimality. Formal endpoint selection, comparative power,
world sampling, statistical preregistration and the full research/system gates
are still required. See `docs/sample-design-contract.md`.
