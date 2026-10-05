# Micro-control keyframe acquisition

This checkpoint tests a different acquisition increment without relaxing any
geometry, stability, novelty or future-validation gate. One development world
reaches six selected views and starts fitting; neither world establishes a map,
future prediction, goal navigation or actual arrival. Full-project gates remain
in `docs/quality-roadmap.md`.

## Registered experiment

`StableMicroKeyframePolicy` uses one MOVE tick per acquisition trial, but retains
the distinct two-tick frozen-future control. The explicit `micro_controls=True`
entry point and `micro-stable-keyframe-camera-projection-v1` schema distinguish
this from the original two-tick policy. An integer/string cannot substitute for
the boolean registration. Canonical JSON comparison rejects a boolean `move_ticks`
masquerading as integer one before any private evaluator read.

All prior gates remain: first eligible WAIT at 8–16 ticks, three stable pairs,
32 tracks/four cells/50% survival, median/p95 stability 0.2/0.5 pixels, two-pixel
novelty, six past keyframes, two supported bootstrap maps, 30 fitting iterations,
the original withheld-ray gates, consensus and nonzero relative translation.
The eleven-trial/288-total-action bounds are unchanged. No private position,
renderer matrix, physical heading or metric block scale selects a view/control.

The two seeds/output paths and conditions were recorded before dispatch at
`runs/micro-keyframe-live-registration-v1.json`. Each capture freezes its executing
package, capture and SDK sources. These are the same two development seeds as the
previous checkpoint, not independent held-out worlds or paired byte-identical
native render streams. The changed controller retains every intervening action.

## Fresh outcomes

| Seed | Actual actions / images | Selected keyframes | Past proposals / bootstrap candidates | Supported map |
|---|---:|---:|---:|---|
| 202 | 194 / 195 | 1 | 0 / 0 | No |
| 201 | 157 / 158 | 6 | 1 / 4 | No |

The captures are `runs/micro-keyframe-live-seed202-v1` and `-seed201-v1`. Both
complete all seven capture checks and independent `audit.json` reports validate
all three outer/eighteen nested checks. All 394/320 artifact hashes are verified.
Audits establish faithful data/replay, not scientific success. Both terminate
`map_unavailable`, with no frozen map, future control, accepted motion or navigation.

Seed 202 retains eleven rejected novelty samples: one-tick inputs still produce
settled displacements around 14–21 pixels for three handles, with insufficient
endpoint track survival. Another handle returns near the anchor but lacks the
required two-pixel novelty. Smaller input does not establish smaller settled
excitation in this world.

Seed 201 selects actual frames 56, 75, 93, 103, 112 and 121, compared with only two
selected views in the historical two-tick run. Its sparse correspondence chain
retains 123 original IDs. All four attempted bootstrap models remain recorded:
slots 2/4 lack triangulation parallax or coverage; slot 3 lacks positive-depth
spatial support; slot 5 fails withheld-point validation. The last model has low
positive-depth ray errors (0.247–0.403 pixels at the public resolution), but only
28 of its 31 withheld landmarks have positive depth in every consumed view.
The existing all-positive-depth gate correctly rejects it; low pixel error is not
sufficient evidence of a physically valid map.

## Numerical-failure retention

Acquisition now records novelty, past-fitting and future-validation exceptions
as `geometry_failed`, retaining all already executed controls and settled samples.
A failed bootstrap retains earlier candidates and the partial proposal digest;
it cannot freeze an incomplete proposal. A future solver failure retains prior
candidate results and verifies the frozen parent's hash. A failure record is not
successful abstention and cannot pass the independent capture audit. Tests inject
tracking, novelty, bootstrap and future-pose failures in both protocols.

## Known-data diagnostics, not live promotion

`research/settled_keyframes/tracking_gap_probe.py` now accepts either registered
capture schema and retains an exhaustive first-loss ledger for every seed. The
first seed-202 interval 56→69 improves from 33.1349% direct survival to 50.9921%
continuous survival, preserving the same detector/gates. Its direct adjacent-cycle
losses decrease from 110 to 24; raw tracking still loses points to texture/ROI
gates. Later original-anchor survival decays and intervals beyond 128 raw frames
are explicitly unavailable, never subsampled. Report:
`runs/micro-keyframe-tracking-loss-seed202-v1`.

`raw_keyframe_fit_probe.py` tracks every raw image in seed 201's already chosen
56→121 past prefix before projecting to the six exact existing sample slots.
Original IDs, loss gates, correlations, cycle errors and selected ordinals remain;
no failed middle image can be skipped. It does not choose new views, open private
poses, fit future pixels, replace a live proposal or execute controls. Both sparse
and raw methods retain every bootstrap at identical fitting settings/seed.

Reports at `runs/micro-keyframe-raw-projection-seed201-v1` and `-v2` exactly replay
the original four sparse models and pass all four provenance checks. V2 adds
explicit withheld-depth diagnostics, not a gate change. Raw projection retains
123 IDs but all four raw candidates also reject. Better local tracking alone
does not repair the map. Each report freezes all executing package/probe bytes;
the earlier version remains intact.

## Verification

Windows and WSL each pass all 47 keyframe/audit tests; the old runtime-auditor
regression passes 86 cases on Windows. Separate research tests pass eight raw
projection, five loss-ledger and six restoration cases. These are not added to
the default full-suite counts. Source and wheel builds pass.

Current V30 core verification on each platform passes thirty seeds/two repeats
and all eighteen identical artifacts/all six release checks. Mean experiments
remain 53.933333; minimum equation-transfer savings remain one. The seed-zero CLI
MVP report preserves SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Default-only Linux passes 640 cases/74 optional or platform skips, zero errors or
failures in 30.480 seconds. Full Windows and persistent-snapshot Linux regressions
are running and must not yet be counted as completed V30 verification.

A 430-file/4,720,387-byte exact source/test/checkpoint snapshot is archived at
`runs/v30-frozen-test-tree-v1`; SHA-256
`040EAEB56D98F6AF28295E717173D0DF2E440B409734C83EB489D9058F61EFB7`.
Its byte-identical persistent restoration is recorded at
`runs/v30-persistent-frozen-test-tree-v1`. The frozen tree predates this checkpoint
document, V2 research-depth reporting and the restoration-helper wording correction;
package/default-suite test inputs match the current V30 source. Current research
scripts and helper tests are verified separately outside the frozen full suite.
No host cleanup policy, SDK, engine JAR or prior evidence was modified.

The preceding V29 persistent full regression is now terminal: 1227 passes/two
skips, zero failures/errors; all 423 original source hashes remain unchanged after
completion. Historical results are not restamped as V30 results.

## Next scientific action

Test bounded public-response action selection: remember which opaque controls
produce supported novel views, distinguish stalls from informative motion, and
seek additional parallax when past geometry rejects instead of merely cycling
least-used handles. Freeze any revised protocol before fresh capture. Do not select
controls from privileged headings, discard failed candidates, lower depth/ray
gates or infer navigation from improved past fitting. Identity-preserving raw
tracking remains a candidate, not an adopted successful map estimator. All
original held-out, persistent scientific-loop, recovery and release gates stay open.
