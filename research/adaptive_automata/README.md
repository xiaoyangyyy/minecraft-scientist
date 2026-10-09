# Ordered adaptive automata prototype

Status: isolated research implementation, **not yet an accepted strong baseline**.
The package and default tests remain frozen while their Linux full regression
runs. These files are not silently substituted into that source freeze. The
older 107-case phase is preserved in `runs/adaptive-rules-source-v1`; its tests
are historical, not evidence for the changed current learner/auditor.

This advances the existing strong-baseline contract, not the entire project
completion claim. H3 structural holdouts, H4/negative transfer, unknown delivery
and noise, fair LLM comparison, controlled Minecraft, natural RGB/navigation,
long recovery and release remain required.

## Implemented scope

- Existing observation tree, paid output queries, fold, consistency and binary
  counterexample processing are reused; no new incompatible theory language.
- Owned public reference IR, common-input edge-level matching with exact rational
  ties, total reference apartness, state covers and distinguishing suffixes.
- One rebuilding/PP phase, then Ex → APS → S → P → AMR → AMS → Eq priority.
  Old reference predictions never become evidence in the current epoch.
- Before-dispatch reference plans bind epoch, public tree, reference digest,
  basis, guidance, input words and pair cost with virtual prefix caching.
- Independent raw public checks validate each rule's applicability and cost.
  Tests reconstruct snapshots from preceding paid plans/evidence, without calling
  the learner's ObservationTree. Shared matching faults are rejected.
- H-conditioned Random-Wp sampling: state-cover prefix, minimum-3 plus geometric
  middle with expected additional length 3, then a fair global/local suffix coin.
  Full out-of-H samples are recorded and rejected, never truncated. Previously
  seen or duplicate admissible words are physically reset/re-executed and paid.
- Independent raw seeded sampler replay verifies components, suffixes, complete
  predictions, rejections, exact types and fresh costs. Shared prediction faults
  are rejected. Partial/unaffordable batches remain unresolved.

## Primary sources and explicit adaptations

[Adaptive L# (FM 2024)](https://arxiv.org/pdf/2406.19714) sections 4–7 and Tables
1/3 guide the ordered rules. Its approximate degree counts **edges**, including
later edges after earlier output mismatches. Empty evidence ties every reference
state here; that is an explicit vacuous convention, not a confidence estimate.

[LearnLib 0.18 RandomWpMethodEQOracle source](https://raw.githubusercontent.com/LearnLib/learnlib/learnlib-0.18.0/oracles/equivalence-oracles/src/main/java/de/learnlib/oracle/equivalence/RandomWpMethodEQOracle.java)
was inspected for the sampling stages, minimum/geometric rule and equal mixture
of global/local suffix testing. This is an independent Python implementation,
not copied Java or a bit-for-bit author reproduction.

Important deviations: canonical pair-BFS suffix sets instead of AutomataLib's
characterizing-set selection; Python MT instead of Java Random; explicit finite
sample/compute bounds; H rejection conditions the unbounded distribution. The
same declared bounded tester must also be used in future from-scratch/adaptive
comparisons. Current main-package basic L# uses a different tester, so comparing
their existing numbers as an algorithm advantage would be invalid.

No unbounded MAT equivalence guarantee is borrowed. A finite passing batch is a
provisional proposal, not unique causal identification or all-H certification.
Matching and tree visit counters are partial work counts, not total CPU cost.

## Verification and next gate

```powershell
uv run pytest research/adaptive_automata -q
uv run ruff check research/adaptive_automata
```

These tests are deliberately outside the default `tests` directory. Exact
source-bound Windows/Linux counts and failures belong in run verification files,
not inferred from this document. Earlier failed XML files must remain visible.

Current `audit_run.py` adds a full stationary policy replay and independent raw
ledger before private I/O. Basis promotions, rebuilding closure and folded
hypotheses are frozen as events. Raw checks enforce input/reset/receipt clocks,
intervention intent, failed/partial query exclusion, per-action costs, actual
caches, epoch provenance, observed apartness, fold tables, conformance order and
candidate validation. A genuine wrong provisional proposal remains a valid run
with a false private behavior score, not an audit failure hidden from the report.

The policy replay uses the same learner; independent raw checks protect specific
shared faults, not every possible algorithm bug. Known terminal failed dispatch
is charged/excluded. Unknown delivery/confounders, nonstationary worlds and causal
factor attribution are not certified under this stationary trusted-reset protocol.

Before strong-baseline acceptance: source-bound cross-platform evidence for this
new phase; paid old-model learning, online epoch wrapper, matched tester/prior/
cost comparisons; prospectively frozen structural splits and transfer evaluation.
The development runner supplies a fixed two-state cycle public prior without
charging prior learning; do not call it an online transfer comparison. See
`docs/adaptive-audit-development.md`. None of this proves novelty or full release.
