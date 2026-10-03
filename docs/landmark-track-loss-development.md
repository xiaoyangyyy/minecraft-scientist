# Original-landmark loss diagnosis

This follows `online-landmark-acquisition-development.md`. It changes diagnostic
observability, not the tracking thresholds, acquisition policy or navigation gate.
The broad requirements in `quality-roadmap.md` remain unachieved.

## Implemented loss accounting

`diagnose_frame_chain` runs the same `track_frame_chain` implementation and
returns its unchanged tracks with an exhaustive, disjoint rejection ledger.
Each original detector identity either survives or has exactly one first failing
gate. Schema validation reconstructs every adjacent survival count and rejects
duplicates, missing/surviving identities, invalid indices and ambiguous categories.

The ledger distinguishes seed ROI, forward status/ROI/step size, reverse status/
ROI/adjacent cycle, patch texture/correlation, whole-chain reverse status/ROI and
whole-chain closure. A whole-chain rejection's image index identifies its reverse
target, not the later prefix at which the loss becomes observable. No failed
identity is resurrected, no threshold is relaxed and no extra pixels enter the
default tracker. Tracing is optional and does not replace accepted map evidence.

`scripts/diagnose_visibility_track_losses.py` verifies historical source snapshots,
image bytes, trial ordering and complete prefix schedules. It explicitly loads
the frozen historical tracker and compares every current diagnostic result both
with that tracker and with the recorded public prefix. It replays the recorded
controller decisions with the original maps and partitions, never reading private
position files or accepting a replacement map. Inputs, current diagnostic sources
and preregistration are frozen separately; failure retains its report.

## Completed real-image replay

`runs/public-landmark-loss-diagnosis-seed201-v5` completes all **51 prefixes/eight
trials** from the 180-frame development acquisition. Historical tracking outputs
and all decisions are exactly reproduced on the capture's WSL/OpenCV runtime.
Input/source/preregistration stability checks pass; 97 current diagnostic source
entries are frozen. This is a historical public
RGB replay, not fresh acquisition, independent physical accuracy or navigation.

At anchor 89 the first map has 36 fitting/12 withheld landmarks at tick two.
Tick three loses 31 fitting/all 12 withheld landmarks, leaving five fitting/zero
withheld despite 226 generic survivors. The protected loss ledger is:

| First failing gate | Fitting lost | Withheld lost | Total |
|---|---:|---:|---:|
| Forward ROI | 11 | 2 | 13 |
| Reverse ROI | 2 | 0 | 2 |
| Adjacent forward/backward cycle | 13 | 6 | 19 |
| Patch correlation | 5 | 4 | 9 |
| Total | 31 | 12 | 43 |

All these failures occur in the new adjacent image step, not a retrospective
whole-chain closure. Only fifteen fail an ROI gate; calling all forty-three
out-of-view is unsupported. The ledger identifies rejection mechanisms, not the
physical cause of mismatching pixels: occlusion, repetitive texture, foreground
motion, perspective changes and rendering effects are not yet distinguished.

At anchor 141 the other map retains 69 fitting/22 withheld out of 78/27 by tick
eight. Its fourteen losses are twelve patch-correlation, one adjacent-cycle and
one forward-ROI rejection. This map's later images still do not establish unique
geometry or independent non-collinear validation, as the preceding report states.

Two early diagnostic attempts are preserved: v1 failed on a report-field mismatch;
v2 failed on dynamic-module annotation resolution. V3 completed the same replay;
v4 additionally verifies recorded tracks and full ordered prefix populations.
None of the failed attempts is reported as successful execution evidence.

V5 additionally compiles verified historical source bytes in memory rather than
using a normal source loader. The earlier loader generated a non-source Python
bytecode cache inside the historical snapshot; that cache is retained, and all
declared source/raw/PNG bytes remain unchanged. The new loader neither creates nor
consumes that cache. Tests verify the complete input file tree is unchanged and
that forged bytecode with a valid matching source timestamp/size is not executed.

## Bounded public-only alternatives

Additional read-only WSL command experiments examine the two original maps with
all alternatives reported. They do not consume private pose labels or change
production configuration. Pyramid levels zero/one/two/three leave respectively
**1/6/5/5** original identities at anchor 89/tick three, and **88/91/91/91** at
anchor 141/tick eight. Level zero already loses thirty identities in the original
48-point map before the switch. Reducing pyramid depth is not a supported repair.

A separate one-factor crop-context experiment seeds exactly the original maps'
public first-image pixels and keeps all other tracking thresholds fixed. The
horizontal crop fractions 0.18–0.82, 0.12–0.88 and 0.08–0.92 retain respectively
**5/6/6** original seeds at anchor 89/tick three and **91/90/90** at anchor 141/tick
eight. Wider crops reduce some ROI rejections but increase other inconsistencies;
neither rescues the failing map. These command observations are diagnostics, not
a source-frozen accuracy suite. In particular, changed prefix pixels cannot be
plugged back into the old triangulated map as if its fit remained unchanged.

The installed SDK's bundled example containing `bobView:true` is not evidence of
the actual launch setting. Existing explicit enabled/disabled fresh-session
comparisons in `stationary-consensus-motion-development.md` do not support turning
bobbing off as a repair. No shared SDK/runtime setting is changed here.

## Verification and remaining work

Forty-five targeted tests pass, including all eleven post-seeding loss gates,
unchanged tracking, disjoint accounting, invalid flow ordering, outside-ROI
invariance, all-trial public replay, and forged frames/snapshots/decisions/tracks/
RGB bytes/schedules, read-only historical replay and forged bytecode refusal.
Tests live in `test_temporal_track_diagnosis.py` and
`test_visibility_track_loss_script.py`, with the existing eight tracker tests.
Both V16 scientific-core releases pass all six checks for thirty seeds/two
repeats/eighteen artifacts. Their 99-entry ledgers freeze this tracing change.
Full regressions at the tracing checkpoint pass **785 tests/two skips on Windows**,
**787/two on WSL**, and **427/63 optional or platform skips** in an isolated default-
only offline environment. Both 99-entry core ledgers matched that checkpoint's bytes.
The subsequent script-only cache-safe loading refinement adds one regression
case and passes the 45-test targeted suite; the full suites preceded that final
script refinement. Ruff, formatting, source checks and both package builds pass.
The seed-zero MVP artifact remains byte-identical with SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.

The prior V15 raw/map/core ledgers are historical after changing the tracker
source. The diagnostic explicitly replays historical tracker bytes; it does not
rewrite their manifests or relabel them as current-source acquisition evidence.
The seed-zero MVP result remains unchanged; no new navigation result is claimed.

Subsequent different-control validation is now implemented and independently
audited in `docs/second-leg-shadow-development.md`. The fresh 203-frame capture
rejects its sole second-leg candidate, despite surviving validation landmarks.
V16 and this diagnostic's manifests are historical after that new source boundary;
V17 freezes 100 core entries. Assess joint multi-view refinement/landmark lifecycle where
ambiguous correspondence remains. Do not substitute generic survivor counts,
wider crops, looser cycle thresholds or abstention for useful navigation. After
the repair, fresh disjoint worlds/terrains/conditions, natural science/map
integration, long recovery and consolidated live release/baseline/CI evidence
are still required under the original plan.

## Reproduction

In the dedicated WSL environment, from `/mnt/d/minecraft-scientist`:

```bash
timeout 240 /opt/minecraft-scientist-visual/bin/python scripts/diagnose_visibility_track_losses.py runs/controlled-rgb-landmarks-seed201-clear-v1 --output runs/new-public-landmark-loss-diagnosis
```

The output directory must not exist. No Minecraft process or API key is needed
for this public-image replay. Preserve the completed diagnostic source snapshot
when later package changes establish another provenance boundary.
