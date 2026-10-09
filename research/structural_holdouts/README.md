# Structural holdout construction — evaluator-only development

This is a new isolated development phase, not a completed formal benchmark or
an Agent feature. It does not modify the frozen main-package/shared-H3 runtime.

`identity.py` independently validates raw total Mealy tables (no TheoryIR compiler
or learner), canonicalizes reachable rooted graphs under state/opaque action/output
renaming, preserves action costs and visible reset-output identity, and separately
minimizes exact deterministic behavior before canonicalization. Behavior identity
is unbounded for these finite deterministic reset machines, not a claim about noisy
worlds, continuous contexts, factor identifiability or arbitrary Minecraft actions.
All actions in this contract have otherwise identical public affordances.

Before/after tuples share one action/output renaming, preserving consistent roles.
Unreachable states are ignored conservatively; latent duplicate states can differ
structurally while belonging to the same behavior group. Canonicalization has a
declared permutation cap and refuses to emit a partial digest when exhausted.

`partition_records` merges every connected shared leakage-key group before assigning
development/tuning/evaluation. Callers must declare whether keys cover targets,
revision pairs or other structure relationships. Stable IDs/seed shuffling cannot
separate colliding groups. Components can be imbalanced and are the sampling unit;
additional bridging records require a new dataset freeze. SHA-256 is a reproducible
index, not a mathematical noncollision guarantee: exact canonical bodies are retained
for equality checks. These metadata and private tables never enter Agent inputs.

Still required: new task/combination generator, declared target/pair leakage policy,
independent generation/partition audit, preregistered sampling and statistics, frozen
development/tuning/evaluation sets, all strong baselines/ablation conditions, raw costs
and negative outcomes. No holdout learning or generalization result is claimed yet.

```powershell
uv run pytest research/structural_holdouts -q
```
