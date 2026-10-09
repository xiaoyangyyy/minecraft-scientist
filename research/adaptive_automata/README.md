# Ordered adaptive automata prototype

Status: isolated research implementation, **not yet an accepted strong baseline**.
The package and default tests remain frozen while their Linux full regression
runs. These files are not silently substituted into that source freeze. The
older 107-case phase is preserved in `runs/adaptive-rules-source-v1`, and the
182-case stationary full-audit phase in `runs/adaptive-audit-source-v1`; their
tests are historical, not evidence for the changed current learner/auditor.

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

The current learner accepts zero reference models for a from-scratch ablation,
using the same H-conditioned Random-Wp tester as its adaptive mode. No fabricated
reference or old observations are inserted into the new observation tree.

`paid_transfer.py` learns and independently finitely validates the old context
from actual paid queries before either method receives a public old theory.
Both methods receive the same old evidence and pay its costs; the scratch mode
deliberately ignores the reference in the new context. Reset/action/validation
costs share one total budget. Caches and evidence IDs are qualified by context.
TheoryRegistry records proposed/provisional/accepted snapshots and parent lineage;
accepted here means a passing finite batch, not all-H correctness or calibrated
confidence. Old context theories are not falsely challenged/superseded on an
explicit context switch. This is NOT an unannounced online mutation experiment.

`audit_transfer.py` checks both contexts' raw public ledgers, exact protocol,
paid reference origin, disjoint final validation evidence, registry snapshots
and whole campaign replay BEFORE either private artifact is read. It detaches
the input snapshot before private callbacks. Private scoring reports legitimate
wrong proposals separately from invalid public runs. Same-learner policy replay
and targeted independent raw checks do not certify absence of all shared bugs.

Run the paired development grid with:

```powershell
uv run python scripts/run_paired_transfer_development.py --output runs/<new-directory>
```

The older stationary development runner supplied a free two-state cycle prior;
its numbers are not a matched paid transfer comparison. Current paired development
keeps all failures/unresolved outcomes and reports success-conditioned cost
differences. It does not prove general savings, causal structure, symbol transfer,
statistical significance or innovation. Exact current source-bound outcomes belong
in `docs/paid-transfer-development.md`, not borrowed historical tests.

Before strong-baseline acceptance: cross-platform source-bound evidence, an
unannounced online epoch wrapper and monitoring costs, full matched comparisons,
prospectively frozen structural splits and transfer evaluation. H3/H4, unknown
delivery/noise, Minecraft/RGB/navigation, recovery and full release remain open.
