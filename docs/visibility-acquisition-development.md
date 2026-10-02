# Visibility-aware causal acquisition

`exploration/visibility_acquisition.py` implements an executable bounded probe,
not another pose acceptance threshold. It chooses anonymous movement handles one
tick at a time from current public RGB tracks. Two distinct handles are requested;
their physical directions remain unknown to the policy.

## Policy and evidence boundaries

- Recompute adjacent, identity-preserving tracks for each consecutive prefix.
  Pixels of common past IDs must remain unchanged; lost IDs cannot reappear in
  the controller population, even if reverse-cycle filtering later accepts them.
- Normalize retention against valid detector points at the original anchor,
  not just points surviving the first action. Early loss must not disappear from
  the denominator. Preserve the original detector/image identity throughout.
- Request a switch after at least two ticks if retention is at most 0.70 or at
  least 0.25 of retained points project near the next frame's central-ROI edge.
  Otherwise switch at the six-tick first-leg limit. Edge margin is twelve pixels
  at the 224-pixel reference scale. These are experimental acquisition heuristics,
  not calibrated survival/pose confidence or physical-direction estimates.
- Request two subsequent distinct frames, finishing the second leg within four
  ticks. Fewer than 32 original tracks or two occupied cells abort, never prove
  a geometric contradiction. Finishing acquisition never accepts a map or pose.
- The executable callback receives only the requested handle and returns an
  RGB frame plus a public one-tick compliance receipt. Mismatched/failed controls
  retain their image/action record but cannot become tracking/theory evidence.
  There are no hidden-state inputs, retries, reseeding or unbounded loops.

The maximum probe uses ten actions/eleven images. Immutable image copies prevent
callbacks from changing the past. Partial control/tracking failures retain all
completed actions, prefixes and decisions. A new anchor requires a separately
recorded new controller, not an identity substitution.

## Existing real-image replay

`scripts/probe_visibility_requests.py` verifies the public acquisition, then
monitors all sixteen movement legs in the corrected-source seed-201 noon/clear
development session. At the first requested different control it stops reading
that trial's subsequent images. If the existing action ends first, the record is
censored. It never treats the old fixed path as a counterfactual switched path.
Whole-session integrity replay is separate from per-trial causal policy dataflow.

`runs/causal-visibility-requests-seed201-v3` completes all sixteen legs; its public
independent replay passes all ten checks. Fifteen legs request a switch at ticks
two to four, all from retention/visibility risk, and one is censored at tick four.
For anchor 152, the policy requests a switch at tick two with **206 original
identities**, versus 352 valid initial points. The old fixed action-switch
diagnostic had zero original points by views 172/176. This establishes an earlier
request opportunity, **not** reliable detection/correction of the 32-degree error.

V1/V2 prototypes remain historical: V1 omitted its input-provider helper from the
source ledger; V2 binds that helper but uses first-action survivors as its initial
denominator. V3 corrects both, retains all trials and freezes the complete current
controller and replay-source ledger. No old result or manifest is overwritten.

## Fresh executable acquisition

`environments/minestudio/capture_visibility_rgb.py` connects the same callback to
the existing public MineStudio adapter, retaining separate evaluator telemetry.
It replays the source-bound public parallax plan before execution, retains the
startup/reference controls and runs eight bounded probes with WAIT controls.
Its preregistered upper bound is **209 frames**, below the 225-frame ceiling of
the earlier fixed development capture. Real rendering/compute costs still need
explicit comparison; fewer declared frames is not an accuracy claim.

The raw ledger freezes all package Python sources, capture/plan helpers, the SDK
entry and public parent input hashes. It records every public transition, RGB
hash, causal prefix, decision, attempted receipt and failed/aborted trial. Raw
acquisition checks are not an independent policy/pose audit and cannot promote
navigation. Fresh capture status and audit must be inspected separately.

The first fresh run, `runs/controlled-rgb-visibility-seed201-clear-v1`, completes
with **172 RGB frames and eight finished probes**. All seven raw integrity checks
pass. First legs take two to five ticks; every second leg takes two. Reported
original identities at completion range from 102 to 240, rather than disappearing
before a much later fixed view. These are raw acquisition/controller records;
independent causal/control replay and original withheld-map/pose accuracy are
still unverified. No navigation acceptance follows.
All 123 registered live source entries and their saved snapshots subsequently
match the current WSL runtime/source paths; this byte check is not policy replay.

The capture wrapper's final package-ledger addition is intentionally kept byte
unchanged while validating this source-bound run. Ruff lint passes; a format-only
check flags that small addition. Format it at the next explicitly versioned
capture boundary rather than rewriting the raw manifest/snapshot or silently
claiming its source still matches after an edit.

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_visibility_rgb.py --plan runs/active-parallax-plan-seed201-source-v4 --output runs/new-visibility-capture
/opt/minecraft-scientist-visual/bin/python scripts/probe_visibility_requests.py runs/controlled-rgb-active-parallax-seed201-source-v4 runs/rgb-active-parallax-prefix-seed201-source-v4 --output runs/new-visibility-requests
/opt/minecraft-scientist-visual/bin/python scripts/probe_visibility_requests.py runs/controlled-rgb-active-parallax-seed201-source-v4 runs/rgb-active-parallax-prefix-seed201-source-v4 --output runs/new-visibility-requests --verify
```

## Verification and next acceptance work

Twenty-eight controller/callback/request-replay tests pass, including real OpenCV
on an anonymous image stream, control failure isolation, no resurrection, prefix
immutability, observation-cost bounds and counterfactual censoring. Two additional
capture-orchestration tests pass independently. Synthetic tests prove contracts,
and the combined thirty-test targeted set also passes in WSL. These tests prove
execution/evidence contracts,
not natural-world accuracy. The controller checkpoint full regressions pass
Windows **700/two skips**, WSL **702/two**, default-only offline **401/58**;
the two later wrapper tests are separately executed, not included in those counts.
Both V13 thirty-seed/two-repeat core releases freeze 98 entries and pass all six
checks; V12 precedes the anchor-denominator correction. Package builds pass.

Next independently replay every fresh adaptive prefix/control receipt before
opening evaluator telemetry. Freeze chronological first-supported original and
alternative maps from pre-switch data, validate original held-out identities
under the actual new views, and report all lost/ambiguous/incorrect results and
costs. A successful request or source audit alone is insufficient. Then validate
fresh disjoint seeds/terrain/conditions and integrate verified map/science/recovery
loops. No full-project requirement or physical error tolerance is reduced.

That complete replay now finishes in `docs/visibility-map-validation-development.md`:
all 26 checks pass, five/eight controls create effective turns, and four/eight
combined local poses are correct. A second-leg direction still errs by 36.654909
degrees. Both supported first-leg maps remain ambiguous after new views, so
navigation is not promoted. Phase-relative loss/excitation and further independent
second-leg validation remain necessary, alongside the full held-out/integration/
long-recovery/release scope.
