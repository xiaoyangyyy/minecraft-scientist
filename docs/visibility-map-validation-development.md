# Independent adaptive-control and frozen-map validation

This validates the fresh visibility-aware acquisition, not navigation promotion.
The original raw capture remains byte-unchanged at revision `0b34945`, including
its deliberately deferred formatter-only addition. Runtime/core package sources
remain the V13 98-entry checkpoint.

## Public-first computation

`visibility_capture_common.py` independently verifies the public parent plan,
complete raw source/snapshot/input sets, every PNG/RGB digest, all startup views,
every public action/target/duration/compliance receipt, every causal prefix and
every visibility decision. No SDK installation is needed for this replay: its
entry snapshot is immutable capture provenance, while current project code is
verified separately. Raw integrity booleans are not used instead of recomputation.

Every trial and control remains in the declared 46-record set. Aborted acquisition
retains missing-leg markers instead of inventing movement or dropping a trial.
Chronological keyframe attempts use unchanged geometry/physical error thresholds.
Identity controls retain zero temporal displacement, not a fabricated one-tick
interval. The twelve startup images alone calibrate the public camera.

For each first leg, the original first-supported map and sixteen direction
alternatives are fit using pre-switch pixels. The actual controller's two
post-switch frames check the same surviving original fitting/withheld partitions.
IDs rejected by the controller cannot reappear as geometry evidence. Future pixels
cannot alter original fitting or choose an alternative pose. No-witness is not
uniqueness, unavailable points are not contradictions, and reference failures
cannot be rescued by a passing alternative.

The independent auditor recomputes all 118 prefixes and every future comparison
before opening private positions. Public/source/snapshot and evaluator bytes
remain bound through final scoring. Actual physical turn diversity is measured
after replay and is never a policy input. The single seed is development, not
disjoint world/terrain/condition generalization.

## Complete real result (2026-10-02)

Inputs: `runs/controlled-rgb-visibility-seed201-clear-v1` (172 images/eight probes).
Current inference: `runs/rgb-visibility-maps-seed201-v2`; its independent
`audit.json` passes all **26 checks**. All 126 inference-source entries are frozen.
The earlier V1 inference fails on a return-value unpacking error before map
computation; its empty/failed artifact is retained, never relabeled successful.

| Group | Trials | Selected correct / wrong | Correct terminal moving |
|---|---:|---:|---:|
| Combined moving | 8 | 4 / 0 | 4 |
| Combined settled | 8 | 3 / 0 | 0 |
| First leg | 8 | 2 / 0 | 2 |
| Second leg | 8 | 0 / 1 | 0 |

These ten selected pair records contain nine correct/one wrong results, or six
correct/one wrong **unique intervals**. Overlapping pair groups are not independent
successes. Five/eight actual control pairs are non-collinear under the unchanged
0.15-block/45-to-135-degree evaluator definition. Four correct combined intervals
do not prove route completion or an improvement over a different fixed trajectory.

Both supported first-leg reference maps pass their new withheld pixel tests,
but competing maps also pass: **two ambiguity-survives results and six no-candidate
results**. A conservative ambiguity veto retains no first-leg result. The second
leg 128-to-130 still has a **36.65490872744035-degree** direction error despite
0.146466-degree rotation error and 0.299693-block actual movement. That second-leg
map has no subsequent different-control validation in this two-leg protocol.
Neither earlier point retention nor internal consistency establishes correct pose.

Thirteen actual stationary controls reject translation: seven settle controls,
two rotation controls, two WAIT controls and two identity controls. One nominal
settle control actually moves; it is unsupported, not a stationary negative.
No failed/moving control is omitted from the ledger or reassigned to a favorable
class. No executable replacement pose or navigation promotion is implemented.

## Tests and next action

Sixteen new tests cover complete public replay, private-read ordering, raw/action/
decision/frame/source tampering, missing trials/pairs, prediction tampering,
immutable pre-switch fitting, no identity replenishment, changed bytes during
scoring, partial inference retention and overwrite refusal. Fixtures deliberately
include aborted probes and do not claim physical accuracy. Full regressions pass
**718/two skips on Windows**, **720/two on WSL** and **401/60 in the isolated
default-only offline environment**. Targeted Ruff, format checks for the four new
files and package builds pass. Both V13 core ledgers retain their unchanged 98
source hashes; scientific-core behavior is unchanged. These checks do not
establish the outstanding full-project gates.

Code/trace inspection additionally establishes that every second leg ends at its
two-tick minimum because global anchor retention is already below 0.70 when the
first leg switches. The same old loss therefore triggers second-leg completion
even if the remaining points are relatively stable. Next evaluate additional
loss relative to each phase's starting population, while preserving global
anchor metrics, original IDs, absolute support/depth/validation gates and action
budgets. Then acquire new explicitly versioned trajectories and independently
check actual excitation, ambiguity, second-leg errors and all costs. This is a
public controller-state issue, not tuning a replacement pose to private labels.

More distant/new views and stable landmark lifecycle/refinement may still be
required. Held-out navigation across five additional worlds/multiple terrain/
three conditions, natural science/map integration, long-operation recovery and
full release comparisons remain required; none is removed by this checkpoint.

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_visibility_maps.py runs/controlled-rgb-visibility-seed201-clear-v1 --output runs/new-visibility-maps
/opt/minecraft-scientist-visual/bin/python scripts/audit_visibility_maps.py runs/controlled-rgb-visibility-seed201-clear-v1 runs/new-visibility-maps --output runs/new-visibility-maps/audit.json
```
