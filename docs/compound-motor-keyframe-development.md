# Anonymous compound motor controls: valid execution, unsuccessful acquisition

Four anonymous cardinal-plus-jump programs are now available under a separate
registration. Both fresh development worlds execute and replay correctly, but each
selects only two views in its first epoch and one independent child anchor. Neither
produces a map proposal; both exhaust their sixteen-tick public settling budget
after using two episode-wide recovery windows. All original completion gates remain
open. This experiment does not establish that jumping improves exploration.

## Explicit registration and opaque boundary

`StableJumpSafeSeedKeyframePolicy` requires all preceding stable/micro/response/
recovery/safe-seed flags plus `--jump-keyframes`. The exact capture schema is
`jump-safe-seed-stable-keyframe-camera-projection-v1`; its policy/result declares
`opaque-four-cardinal-jump-v1`. The existing visual codec already supports compound
movement programs, so no new primitive action kind or SDK/engine patch is needed.

The recorder privately replaces its four cardinal programs with four cardinal-plus-
jump programs. Public object IDs, order, count and visible motion-handle features
are unchanged; native button names are absent from Agent observations. LOOK/WAIT
never include jump. Public-only novelty scores choose controls; no position, heading,
terrain annotation or evaluator-selected winner enters dispatch. The first forty-
eight calibration actions are unchanged. Old recorder/protocol defaults still use
plain cardinal movement and retain their original output contracts.

First-image safe seeds, frozen denominators, sparse selected-view clock, one-tick
acquisition/two-tick future movement, eleven trials per epoch/twenty-two total,
two local epochs, two extra recovery windows and global 288 actions stay unchanged.
The six-view minimum, 32 tracks/four cells/fifty-percent endpoint support, two-pixel
novelty, original WAIT stability and all fitting/positive-depth/withheld-ray/consensus
gates are unchanged. Research-only raw past fitting is not promoted by this change.

## Fresh native results and independent audit

`runs/jump-keyframe-live-registration-v1.json` declares both development seeds before
launch. Sources are frozen during execution and replay. Each capture has 119 source
snapshots, seven passing capture checks, three passing outer audit checks and
eighteen passing nested camera audit checks. Both saved audits are valid, with
330/336 capture artifact hashes. Validity certifies provenance/public reconstruction,
not successful map acquisition or independent goal arrival.

| Seed | Actions / images | Selected frames per epoch | Completed trials | Terminal |
|---|---:|---|---:|---|
| 201 | 162 / 163 | 56, 89; then 123 | 4 + 1 | acquisition_wait_exhausted |
| 202 | 165 / 166 | 56, 111; then 132 | 4 + 1 | acquisition_wait_exhausted |

Both `runs/jump-keyframe-live-seed{201,202}-v1/audit.json` preserve zero proposals,
no frozen/future map, no accepted motion, metric scale, connected gauge or navigation.
Three images across two epochs are not three connected keyframes. No failed sample,
window or model is discarded, and native-byte equality with older runs is not claimed.

Seed 201 retains failed windows 106→122, 124→140 and 146→162. Extra WAITs 122→123
and 140→145 recover stable tails, consuming both allowed recovery windows. The first
recovered endpoint has lost support against its previous selected frame and starts
the sealed child at 123. The next unresolved endpoint ends acquisition at 162.
Seed 202 retains failed windows 90→106, 112→128 and 149→165. Extra WAITs 106→111
and 128→132 recover stable tails; the latter opens the unsupported child at 132.
The final window cannot borrow another recovery allowance.

### Post-audit timing diagnosis, never an online input

After both complete independent replays succeeded, separate evaluator-only analysis
compared the failed tails. In seed 201, body displacement over the final four pairs
decreases through 0.010353/0.005653/0.003086 to zero blocks; corresponding rendered-
camera displacements are 0.018418/0.010920/0.006503/0.003885. In seed 202 the final
pair likewise has zero body displacement but 0.003252 blocks of camera displacement.
The final public p95 flows are still 0.997244/0.871361 actual pixels, failing the
unchanged 0.5-pixel gate. Earlier tail pairs still contain actual body drift.

The hash-bound separate diagnosis is
`runs/jump-keyframe-wait-failure-diagnostic-v1.json`. Player rest and visible stability
must not be conflated. No extra tail images were acquired, so neither the diagnosis
nor extrapolated convergence establishes that a longer wait would pass, acquire
useful geometric views or solve navigation. It does justify next testing a separately
registered settling-time budget for compound controls, still with original pixel
gates and global episode cap. Lowering p95 thresholds or asserting post-terminal
stable prefixes would hide the observed failure rather than repair it.

## Verification scope

Windows controller/compatibility tests pass 128 and replay/CLI tests 70; combined
Linux focused tests pass all 198, zero failures/errors. The seventeen new motor/
controller tests verify anonymous encodings across three seeds, no jump in LOOK/WAIT,
rejection of private pose fields/untyped flags, all four handles without imagined
novelty, parent-protocol separation and unchanged bootstrap requirements. Sixteen
additional audit/CLI cases cover full replay and tamper/missing-flag rejection before
any private scoring. Existing safe-seed, map, recovery and visual-adapter behavior
is checked alongside the new mode.

The initial Windows controller run retains four failed test assertions/44 passes:
three omitted a required anchor argument and one expected an overly specific nested
validation message. Those test bugs were corrected without changing production
behavior. A separate attempted command used a nonexistent test path and ran no tests;
it is not a passing result. Final controller verification is separately saved in
`runs/jump-keyframe-controller-windows-v3.xml`. No earlier report is restamped.

Both V34 core releases pass thirty seeds/two repetitions/all eighteen exact
artifacts/all six checks, with 115 source-ledger entries (112 package files).
Mean experiments remain 53.933333 and minimum equation transfer savings one.
Seed-zero MVP retains byte hash
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`;
the package builds without new runtime dependencies. Default-only isolated Linux
passes 673/77 optional/platform skips, zero failures/errors.

V34 full Windows and frozen persistent Linux regressions are still running, not
reported as passing. Their JUnit targets are `runs/jump-keyframe-full-windows-v1.xml`
and `runs/jump-keyframe-full-persistent-tree-wsl-v1.xml`. Exact source/test/checkpoint
inputs are archived at `runs/v34-frozen-test-tree-v1`: 450 files/4,919,695 bytes,
archive SHA-256
`30c46444fab6160ec3f4ff11b39637446ee5af359f8c31b3ce387c146730cc81`.
Verified restoration is `runs/v34-persistent-frozen-test-tree-v1`. It predates this
document, not the new production/test inputs. Earlier V33 full counts are historical.

The next timing/control experiment must preserve existing failures, declare its
actual WAIT/motor/action budgets before launch, stop from public evidence only,
and pass full source-bound replay and independent camera/body validation. Reliable
actual arrival, five disjoint held-out worlds/multiple terrains/three conditions,
persistent map/scientific-loop integration, hundred-attempt restart/failure recovery,
current baseline/ablation comparisons and consolidated release gates remain open
in `docs/quality-roadmap.md`. The full objective remains active.
