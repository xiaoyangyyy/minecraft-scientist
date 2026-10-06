# Bounded unconnected map-epoch reacquisition (development)

This isolated candidate continues the fresh monitored-map parent described in
`docs/public-map-risk-development.md`. It is not promoted main-source behavior,
reliable navigation, a connected cognitive map or a completed release. The original
excellence requirements in `docs/quality-roadmap.md` remain unchanged.

## Actual continuation, not a reset

`exploration/map_epoch_reacquisition.py` orchestrates the unchanged oriented walking
startup and at most one genuinely new acquisition. The parent must return its
original map failure and bounded public WAIT recovery. The continuation binds:

- the complete sealed parent acquisition, frozen map and last failed prediction;
- the actual final RGB receipt and episode ordinal (which can exceed the historical
  startup-only 48..81 range);
- a fresh unconnected detector/track namespace; and
- the unused portion of the original episode/acquisition/recovery budgets.

The child starts from the returned public image without rerunning calibration,
orientation, reset, teleport or skipped controls. Its tracking, bootstrap,
withheld-ray, consensus and final future-motion gates are unchanged. All parent
failures and every child attempt remain visible; a new local map cannot reinstate
the old map, reuse old landmark IDs, imply known body rest or stitch metric gauges.

Across parent and child, the policy still permits at most 288 actual episode
actions, 22 acquisition trials, two acquisition-local epochs, two additional
acquisition WAIT windows and two passive observation-recovery windows. Calibration
and orientation consume the same clock. Shared counters are not reset per child.
Each passive window retains its original 24-tick cap, eight-tick minimum and three
stable trailing observations. Unknown consumption from a throwing controller
remains unknown; it is not recorded as an invented successful action.

## Native protocol and independent audit

The native flag is `--reacquire-keyframes`, requiring every explicit monitored
walking parent flag, with schema
`reacquiring-monitored-walking-keyframe-camera-projection-v1`. Its startup policy
remains the original monitored policy; the separately registered map-epoch policy
contains the distinct child policy. Old native schemas reject the new policy.

Capture journals preserve the startup parent and every child separately. The
independent auditor first reconstructs calibration, orientation, acquisition,
failure, passive recovery, actual child actions and complete lineage from public
images/receipts. It compares the entire saved journal, including failures and the
shared clock, before opening private poses. A retained child control/solver-error
terminal is not a completed protocol merely because its outer failure field is
null. Child rendered-camera and player-body motion are independently scored only
after all public replay and private byte-verification gates pass. Missing future
evidence stays null, and successful camera scores cannot replace body failures.

## Evidence boundaries

The frozen monitored parent has 526 distinct focused package cases per platform,
plus eight separate driver cases; its two completed fresh development captures
still do not prove recovered maps, future motion or arrival. Those counts do not
cover this changed candidate.

New candidate unit fixtures use real RGB tracking and action chronology with
explicit model/flow/producer dispatch doubles. Native-auditor fixtures are mock
captures, not evidence of successful native geometry. Preserve initial narrow
test receipts separately from subsequent current-source verification. The complete
candidate patch is `research/settled_keyframes/reacquiring_map_candidate.patch`;
its application check is not a full regression or native success claim.

`runs/risk-epoch-source-registration-v1.json` prospectively freezes eighteen
changed/new paths and the complete patch. Its final capture-guard revision passes
forty Windows native-audit/guard cases with zero failures/errors/skips. The earlier
240-case Windows combined receipt belongs to the preceding capture revision and
is retained explicitly as historical, not restamped as current-source coverage.
Both current-source combined runs now terminate zero: 243 cases each on Windows
and Linux, zero failures/errors/skips, with all eighteen frozen source hashes
unchanged. These comprise 203 controller/shared-WAIT cases and forty new native
audit/guard cases. Windows v3 retained 217 passes/26 setup errors because the
explicit basetemp parent was missing; unchanged-source v4 uses an explicitly
created dedicated parent and passes all 243. No failed output is deleted or
restamped. Compatibility and full auditor/full candidate regression remain
separate. Subsequently each platform closes 147 additional compatibility cases,
zero failures/errors/skips, giving 390 distinct current focused cases/platform.
Case-sensitive IDs preserve separate Boolean `True` and string `true` parameters.
Every original frozen source hash remains intact.

`runs/reacquiring-map-live-registration-v1.json` prospectively freezes all four
terminal XML receipts, eighteen changed paths, 117 package/122 native project
sources, the unchanged engine, exact policies, shared budgets and two absent
development output paths. It expressly marks full auditor/full candidate regression
not run. The seed-201 new-protocol JVM has launched under its own 1200-second
timeout; `runs/reacquiring-map-seed201-launch-v1.json` is launch-only, not terminal
capture, child-map success, independent audit or arrival proof.

## First terminal fresh-native result

Seed 201 capture AND independent audit now terminate zero. It consumes 208 actual
actions/209 RGB frames. The startup map is still challenged at 133, with twelve
actual WAITs yielding the original ready anchor at 145. The child genuinely starts
there, uses a new unconnected namespace, selects `[153,162,171,180,189,198]`, fits
one proposal and freezes it at 198. All ten actual future MOVE/WAIT receipts remain
publicly observable; original bootstrap/withheld/consensus/final pixel gates pass
at 208 and the public terminal is `child_validated`.

This is NOT verified useful body motion. The unchanged independent 0.15-block
threshold classifies BOTH rendered and player-body predictions as
`false_translation`: their actual displacements are 0.102769130 and 0.102686592
blocks. Direction errors are null, not zero or success; rotation errors remain
0.038034407/0.037860033 degrees. Metric scale, old-map restoration, connected
gauges, navigation and arrival all remain false/unknown. An accepted pixel model
cannot erase this insufficient-translation negative.

All nine capture checks, three outer and eighteen nested audit checks pass. After
terminal audit, all 124 source snapshots, 122 preregistered current project sources,
422 artifacts and eighteen changed source hashes match. Saved active auditor and
config hashes match too. Preregistered canonical `package/minecraft_scientist/...`
names map one-to-one to capture's existing `package/...` snapshot aliases; original
registration is preserved rather than restamped. See
`runs/reacquiring-map-seed201-post-native-verification-v1.json`.

The prospectively registered seed-202 capture AND independent audit also terminate
zero. It consumes 194 actions/195 images, selects only frame 56, forms zero proposals
or frozen maps, and ends `startup_map_unavailable` without dispatching a child.
No future/motion score exists. Its nine/three/eighteen checks pass; all 124 snapshots,
122 current project sources, 394 artifacts and eighteen source hashes match after
terminal audit. See `runs/reacquiring-map-seed202-post-native-verification-v1.json`.
This is an algorithmic acquisition negative despite a valid experiment/audit.
Both worlds are familiar development seeds in the same clear/noon condition, not
disjoint held-outs or superiority.

## Next actual evidence

The frozen candidate's full Windows auditor now terminates zero: 258 passes,
zero failures/errors/skips, with all eighteen source hashes unchanged. See
`runs/risk-epoch-full-audit-windows-post-verification-v1.json`. Distinct Windows
focused coverage is 608 (203 controller + 147 compatibility + 258 auditor); the
previous forty new audits are included in 258, not counted twice. Linux still has
390 focused cases; its full auditor and both full-package runs remain open. These
are mock/component regressions, not additional native successes.

The next isolated candidate is `docs/four-move-map-epoch-development.md`. It keeps
this entire frozen source/evidence intact and changes only the child future
stimulus length from two to four actual monitored MOVEs, without changing any
original gates/shared budgets or private evaluator thresholds. Its own source
registration and verification are separate; do not apply this parent's coverage
or native results to it.

Preserve this entire source-bound candidate while closing full auditor/candidate
regressions. Then prospectively investigate sufficiently informative public
control excitation and feature-support-sensitive acquisition. Seed 201's small
physical interval cannot be fixed by lowering the evaluator's 0.15-block gate or
selecting a privileged endpoint/direction. Seed 202's no-map result cannot be
hidden behind seed 201's new local map. Any new control policy must retain both
negatives, all actual receipts, original map/withheld/consensus gates, independent
public-first replay and the original global budget.

Full current-source auditor/candidate regression and independently verified useful
body motion remain required before promoting this research path as navigation.
The seed-201 actual new local map is verified; useful physical motion is not.
Even a successful local child will not close goal arrival, disjoint held-out
terrains/conditions, natural science-map integration, 100-attempt operation and
recovery, paired comparisons or consolidated release/hosted CI.
