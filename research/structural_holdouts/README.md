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

The next construction phase adds `generator.py`, `raw_tables.py`, and
`audit_corpus.py`: genuinely new 2–5-state graphs, parameter/redirect/state-insertion/
composed edits, hidden deterministic controller changes, unchanged and invisible
insertions, and edit-family mismatch. Actual old factor choice is hidden; all methods
receive the same nine before-family alternatives. Target behavior already present
in that family is marked, never silently treated as novel. Only `public_entry` is
passed to learning; fixture inputs/after tables/identities remain evaluator-only.

Splits join target and ALL supplied before-prior graph/behavior identities in the
same namespace, plus before/after pair identities. Each record supports up to 64
leakage keys for this complete prior coverage; the preceding four-file/60-case
identity phase is preserved in `runs/holdout-identity-source-v1` rather than borrowing
its green result for the changed code. Construction audit validates all public
entries before private reads, replays edit journals, checks raw products against
actual world traces in tests, checks canonical bodies on digest reuse, and rejects
grid omission, extra files, source drift and split forgery. Every failed fixture
is retained; failed builds have no certified manifest.

Still required: formal sampling/statistics and METHOD preregistration, held-out
learning experiments, all strong baselines/ablations, unknown/noisy controls, H4 and
Minecraft/RGB/release. A frozen construction dataset is not a generalization result.

```powershell
uv run pytest research/structural_holdouts -q
uv run python scripts/build_structural_corpus.py --output runs/<new-directory>
```
