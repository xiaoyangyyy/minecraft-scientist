# Public-response keyframe control

The controller now learns which opaque controls have produced supported novel
views from its own public past. Two fresh audited development worlds still fail
to establish a map: one exhausts a stability wait while falling, the other never
acquires enough novel views. This is control adaptation and a diagnosed failure,
not successful mapping, navigation, actual arrival or full-project completion.

## Policy and evidence boundary

`exploration/public_response_control.py` ranks four opaque controls by the exact
rational score `(novel_views + 1) / (completed_trials + 2)`, then fewer observed
trials and opaque index. This Beta-count heuristic is not a calibrated probability
of physical motion, map correctness or arrival. Counts depend only on chronological
completed public trials and the existing novelty gates. Heading labels, evaluator
coordinates and future frames are not inputs. Malformed booleans, reordered trials,
more than eleven trials or a recorded run exceeding four controls reject.

`StableResponsiveKeyframePolicy` retains one-tick acquisition and distinct two-tick
future validation. After four consecutive uses it must choose another control;
after exploring a stalled alternative it may return to a historically useful one.
The future control uses the same past-only ranking, explicitly excluding the last
acquisition control. Its decision is stored after map freezing and before dispatch.
Every choice retains its prefix length, exact statistics and actual frame ordinal.
The original geometry, novelty, stable-tail, consensus, twelve-view/eleven-trial
and 288-action gates remain unchanged. No incomplete fit or passing subset rescues
a rejected proposal.

The explicit API requires both `micro_controls=True` and `responsive_controls=True`.
Capture requires `--stable-keyframes --micro-keyframes --responsive-keyframes` and
uses a separate `responsive-stable-keyframe-camera-projection-v1` registration.
Missing flags reject before allocating output. The auditor reconstructs all public
choices/calibration/controls/fits before reading private poses; changed history,
statistics, future choices, schema or policy cannot reach the private evaluator.

## Fresh results

The two development seeds and rules were declared before dispatch at
`runs/responsive-keyframe-live-registration-v1.json`. No held-out generalization,
paired native-frame identity or statistically significant improvement is claimed.

| Seed | Actions / images | Selected views | Completed past proposals | Terminal |
|---|---:|---:|---:|---|
| 201 | 147 / 148 | 6 | 1, four rejected bootstraps | Acquisition wait exhausted |
| 202 | 194 / 195 | 1 | 0 | Map unavailable |

Each capture freezes 117 executing sources and passes seven capture checks. Each
independent `audit.json` passes three outer/eighteen nested checks and verifies
300/394 artifact hashes. The captures are respectively
`runs/responsive-keyframe-live-seed201-v1` and `-seed202-v1`. Neither contains a
frozen map, future experiment, accepted motion, global gauge, metric map or arrival.
Successful provenance audits do not turn these outcomes into scientific successes.

Seed 201 completes the same first eight trial responses as the preceding micro
experiment, selecting frames 56, 75, 93, 103, 112 and 121. Slots 2/4 lack parallax
or coverage, slot 3 lacks positive-depth support, and slot 5 fails withheld-point
validation; all remain recorded. The controller then returns to useful handle 2
after testing stalled handle 3. Choice nine is retained at frame 130 with eight
completed public outcomes, but its subsequent wait never meets the stable-tail
gate by frame 147. No ninth completed novelty outcome is invented.

The final sixteen public WAIT pairs show substantial scene motion before slowing:
median flow peaks at 19.637 pixels; the last pair is median 0.116/p95 0.603 pixels,
which still fails the unchanged p95 0.5 gate and lacks three stable pairs. After
successful independent chronological replay, evaluator-only positions establish
a five-block vertical fall during frame interval 131→147, with horizontal drift
0.164241 blocks. The controls were compliant; this is not a controller failure.
The hash-bound diagnostic is separately retained at
`runs/responsive-keyframe-wait-failure-diagnostic-v1.json`. Those coordinates were
never available to the controller and do not prove any longer wait would succeed.

Seed 202 observes no novel control response and explores the opaque handles in a
bounded least-observed cycle. All eleven rejected samples remain; no map fitting
is attempted. The new selector does not fix endpoint correspondence acquisition
in this world, nor does adapting action choice alone solve the six-view geometry.

## Failure handling and tests

Stability fitting errors and missing/non-boolean stability summaries now return
`geometry_failed` at stage `stability`, preserving completed action receipts and
prior observations. A successful control followed by a failed pixel solver is
not mislabeled as failed execution. Such a failure cannot pass capture validation.
The previous novelty/bootstrap/future-failure retention and frozen-parent checks
remain; successful base/micro paths keep their existing result shapes.

Both platforms pass 90 focused cases: 25 pure public-response selector tests,
38 acquisition tests and 27 replay/CLI audit tests. These cover identity-independent
control learning, exact prefix decisions, a forced control switch and return,
continued acquisition after rejected past models without increased budgets,
larger unseen future control, immutability and pre-private tamper rejection.
Windows old runtime-auditor regression also passes all 86 cases.

## Release verification

V31 Windows/WSL core releases each pass thirty seeds/two repeats/all eighteen exact
artifacts/all six release checks. All 113 source-ledger entries (110 package files)
are independently rechecked unchanged. Mean experiments remain 53.933333 and
minimum equation transfer savings one. The original seed-zero MVP CLI report
retains SHA-256 `B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Both package builds pass without additional runtime dependencies. Default-only
Linux completes 665 passes/74 optional or platform skips, zero errors/failures in
35.125 seconds. Full V31 regressions are terminal: Windows 1287 passes/two skips
and persistent Linux 1289 passes/two skips, zero failures/errors. Their JUnit
reports are `runs/responsive-keyframe-full-windows-v1.xml` and
`runs/responsive-keyframe-full-persistent-tree-wsl-v1.xml`; the frozen Linux tree's
433 file hashes were unchanged after completion. These verify V31, not subsequent
recovery-control edits.

The exact V31 test/checkpoint snapshot contains 433 files/4,754,260 bytes; archive
SHA-256 is `F1DE37007720992DF550617EB34CB0BDC55BED9AFAC6D4975DA2B8F8236324BF` at
`runs/v31-frozen-test-tree-v1`. Its byte-identical restoration under `/opt` is
recorded at `runs/v31-persistent-frozen-test-tree-v1`; runtime directories were
verified unused before launch. The snapshot predates this document, not the
package/test inputs. No SDK, shared engine, host cleanup policy or prior evidence
was changed. An archive is source provenance, not proof of a passing full suite.

The preceding V30 full suites are now terminal: Windows 1244 passes/two skips,
Linux 1246 passes/two skips, zero failures/errors. The V30 frozen tree has 430
files, all hashes unchanged after completion; Windows's 112-entry core ledger is
also unchanged.
These results remain historical and do not validate V31 changes.

## Next scientific action

Separately preregister bounded recovery from a failed public stability window,
within the original total action budget and unchanged image/depth/withheld-ray
gates. Preserve the exhausted attempt, its rejected models and all subsequent
controls. Test whether a recovered stable view can extend the proposal or instead
requires an explicit new local-map namespace after lost correspondence. Do not
silently extend/restamp this run, erase a negative, assume optical stability is
physical rest, delete bad withheld points or tune controls from privileged
coordinates. All held-out/persistent scientific-loop/long-recovery/release gates
in `docs/quality-roadmap.md` remain open.
