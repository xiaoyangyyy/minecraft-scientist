# Bounded competing-map development diagnostic

This diagnostic tests whether a different motion can explain the same public
pixels. It never selects a replacement pose or certifies uniqueness. The full
project still requires reliable goal navigation, held-out worlds/terrain/weather,
natural scientific/map integration and long-operation recovery.

## Fixed hypothesis family and evidence boundary

`exploration/pose_competition.py` takes a frozen locally supported keyframe
candidate and its original training/validation identities. Nineteen fixed trials
cover zero rotation adjustment and +/-0.25, +/-0.5, +/-1 degree about each camera
axis. For each fixed rotation, translation is fitted by robust homogeneous
epipolar equations using **training rays only**. Its sign is selected using
training cheirality only. Validation rays do not select either fit or sign.

Each map must retain all original partition identities under the existing
positive-depth, one-degree parallax and endpoint reprojection criteria. Original
intermediate-frame PnP uses training points only; the distinct original withheld
points must pass the unchanged pixel/coverage thresholds. A plausible explanation
whose camera-center direction differs by at least 20 degrees is an ambiguity
witness. The incomplete grid cannot prove that no other explanation exists.
The public evidence has no executable translation/rotation or navigation-support
fields. Every trial, including failed alternatives, is retained.

This is stricter than merely rerunning minimal-set RANSAC: the family demands
95th-percentile training epipolar error at the fixed 0.5-reference-pixel threshold
and complete original identity retention. Those extra diagnostic demands can
exclude maps, including the training-only zero-offset refit, that passed the old
pose gate. A failed refit is not evidence against the original map's existence.
There is no claim of equivalent acceptance coverage or complete model search.

## Complete public replay before private scoring

`pose_competition_common.py` verifies the active plan/profile, prior prediction
declarations and complete prefix computation. The diagnostic freezes both raw
sessions, old prediction/config/snapshot bytes, active plan/profile snapshots and
all producer/auditor/evaluator source bytes before computation. No private
telemetry file is a policy or diagnostic input.

`audit_pose_competition.py` reconstructs every public competition first, checks
unchanged bytes, then invokes the independently replaying active baseline audit.
Only the baseline's scoring stage reads private positions. Final checks retain
evaluator identity and all public/source bytes through scoring. Invalid artifacts
receive no evaluation. An optional veto comparison only removes original selected
poses; absence of a witness cannot rescue unsupported pairs or replace a pose.

## Development results: the finite family does not repair the real errors

Both existing fresh V2 captures are replayed: development seeds 200/201, noon/clear.
These are not new acquisition or held-out validation. Each retains all 46 pairs,
all 164 original prefixes, and passes **54** independent integrity/replay checks.

Seed 200's thirteen selected pair predictions produce zero ambiguity witnesses.
The ten correct and **three wrong** original predictions all remain. Deduplication
still gives eight image intervals, five correct/three wrong. The wrong intervals
are 60→64, 128→130 and 176→178, with the unchanged direction errors 23.628, 45.389
and 37.599 degrees. The best plausible alternative for 128→130 differs by only
4.023 degrees. Neither other wrong interval has a plausible trial in this family.
No new accuracy or navigation gain follows.

Nine of the thirteen selected pair records have no plausible training-only refit,
even without the 20-degree witness requirement. Many trials fail training residual
or original identity retention before intermediate checks. This makes the negative
search result weak evidence about general observability. It does not establish
unique causality, identify the true motion or justify loosening the old tolerance.

Seed 201 originally supplies no selected pose; the diagnostic produces none and
therefore no witnesses. Its zero useful movement coverage remains a failure, not
successful error prevention. Negative control results remain those of the baseline.
Neither the new diagnostic nor its optional veto is promoted into navigation.

### Fresh weak-anchor V3 acquisition

`runs/public-pose-competition-seed201-clear-v3/audit.json` passes all 54 checks
against the new 225-frame active V3 acquisition. The original 42-check audit
records twelve selected pair predictions, eleven correct/one wrong (nine unique
intervals, eight correct/one wrong). The only ambiguity witness is a -0.5-degree
camera-y rotation adjustment for `compound_settled-06`, interval 168→170: its
direction differs by 20.508553 degrees. The original pose is nevertheless correct
with 12.052553-degree direction error. An optional veto would lose that correct
prediction while leaving the wrong 152→154 direction (32.412640 degrees).
This is a real ambiguity witness, not an accuracy improvement or default gate.
No-witness results still cannot establish uniqueness. All artifacts remain frozen.

## Artifacts and source history

- `runs/public-pose-competition-seed200-clear-v1`: frozen configuration/sources,
  complete public diagnostic and final `audit.json`.
- `runs/public-pose-competition-seed201-clear-v1`: the same complete contract.
- Original images/actions/predictions remain in the V2 active-parallax runs.

These V1 results freeze the V2 selector and the diagnostic helper before import
sorting. Subsequent V3 selector development intentionally makes their current-
source check stale; the original snapshots and successful audits are preserved,
not rewritten as current evidence. Historical replay requires those frozen sources
and the recorded WSL/OpenCV runtime. For fresh current-source active captures:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/inspect_pose_competition.py runs/new-active-capture runs/new-active-predictions --output runs/new-pose-diagnosis
/opt/minecraft-scientist-visual/bin/python scripts/audit_pose_competition.py runs/new-active-capture runs/new-active-predictions runs/new-pose-diagnosis --output runs/new-pose-diagnosis/audit.json
```

All output paths refuse overwrite. No API key or remote model is used. Partial
diagnostic failures retain completed pair records and do not promote a pose.

## Correctness tests and next action

Twelve optional geometry tests include a known-pose distant-plane synthetic witness
with two distinct explanations over 20 degrees apart, an ordinary nonplanar case,
translation equations, withheld corruption, determinism, camera/seed bounds and
schema forgery. The synthetic plane fixture declares a known correct pose; it does
not claim that the baseline learner would select that map. Nine pipeline tests
cover all-public-before-private order, tampering, final byte/identity changes and
partial failures. Three default-only tests cover immutable/lazy schema boundaries.

The initial WSL regression exposed a test-mock filename from shared Windows pytest
bytecode. The mock now declares its executing platform's native source path; no
production source identity or replay gate is weakened. Final full regression and
core release results are recorded in `docs/core-release-verification.md`.

Next require a reference-map sanity replay and a better complete training fit,
then compare wider jointly refined model families using disjoint subsequent views.
Retain every wrong and correct development case and measure both lost positives
and surviving errors. A finite negative search, more abstention or synthetic
witness detection alone cannot establish useful navigation. V3 weak-anchor
resampling is a separate action-design correction, not an accuracy repair.
