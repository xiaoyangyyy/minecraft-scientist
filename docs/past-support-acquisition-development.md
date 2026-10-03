# Public past-support switching: replicated development result

Earlier complementary-control acquisition produces a useful **rendered-camera**
estimate in two fresh same-seed sessions, without weakening any motion/pixel gate.
The fixed-cap reference abstains in both sessions. Simultaneous current-player
direction scoring remains wrong; neither result certifies goal navigation,
held-out generalization, independent confidence or overall project completion.

## Implemented policy and comparison

`exploration/active_causal_map.py` adds the registered strategies `past_support`
(default) and `fixed_cap` (reference). Both retain the same ten-tick acquisition,
two-tick future horizon, support floors, all-bootstrap retention, withheld rays,
and all-passing-map consensus. From the first fitting-eligible prefix, insufficient
past support requests the other acquisition handle rather than waiting until
six ticks. A sufficient prefix freezes before any new-control validation. The
switch happens at most once; its actual public reason/count/time is retained in
`control_decisions`. A fixed-cap strategy still waits until its original cap.

This is a bounded heuristic for experimental excitation, not a guarantee that
different opaque handles produce geometrically independent movement. No private
position, camera matrix or future ray selects the switch or a preferred map.
Failed receipts remain JSON-serializable even with invalid NumPy scalar metadata;
they cannot enter evidence. Tracking gaps cannot enter the acquisition state.

`scripts/compare_active_acquisition.py` freezes its manifest and 111 source entries
before running both strategies in separate fresh Minecraft processes. It retains
both failed and successful conditions, action/fitting costs and timeout logs. Each
capture is independently replayed by `audit_active_causal_map.py` before private
scoring. Policies/seeds must match their declared condition; changing sources
prevents further dispatch but does not silently drop a registered outcome.

The first comparison is `runs/past-support-comparison-seed201-v1`; after observing
it, the **unchanged** protocol is replicated in `...-v2`. Both preserve all six
comparison checks, both conditions, all seven capture checks and three outer/
eighteen nested audit checks per condition. Each capture freezes 108 source entries
including 101 package files. These are four fresh development sessions at seed
201, clear/noon, not four worlds or a held-out evaluation. Sources, engine and
shared SDK stay unchanged. Budget **caps** match; images and actual compute do not.

## Results, without averaging away the counterexample

| Measure | Fixed cap (each run) | Past support (each run) |
|---|---:|---:|
| First-control switch tick | 6 | 3 |
| Attempted past prefixes | 7 | 6 |
| Fitted candidates | 35 | 27 |
| Post-calibration policy actions | 11 | 10 |
| Frozen endpoint / validation endpoint | 57 / 59 | 56 / 58 |
| Future decision | reject | refresh |
| Rendered motion correct | no supported estimate | yes |
| Current-body motion correct | no supported estimate | **no** |

Past-support acquisition retains 121 original tracks at its eight-tick freeze,
with two supported maps (bootstraps 7/8). The two-tick future retains 117 tracks.
Both frozen maps pass every withheld-ray check: 31 held-out tracks, acceptance
1.0, five cells and approximately 1.176/1.179 pixels at the 95th percentile, below
the unchanged three-pixel limit. All passing directions agree within 2.308069
degrees and rotations within 0.184716 degrees. These correlated initializations
are not independent-sample statistical confidence.

The first rendered result has direction error **0.915467°**, rotation error
0.105231° and translation 0.253980 blocks. The replication has direction error
0.915466° and rotation error 0.105232°. Both current-body scores remain wrong:
approximately **64.3907°** direction error with translation 0.233692 blocks.
Rendered coverage is 2/2 for this method and 0/2 for the reference; current-body
correctness is 0/2 for both. Two same-seed trials do not establish a
generalization rate, significance, route success or deterministic live pixels.
All outputs still declare `navigation_accepted=false` and unknown metric scale.

## Measured timing difference, not a replaced score

Evaluator-only decomposition of the first successful interval confirms:

- The rendered camera-base displacement equals the **previous-player** displacement
  at these endpoints (vector difference zero).
- Current-player versus previous-player displacement differs by 0.267166 blocks
  and 69.759589° at this control switch.
- Effective-camera offset change contributes only 0.028352 blocks; rendered versus
  previous-player direction differs by 4.613724°.
- Scoring against previous-player position yields 5.369004° direction error as a
  **diagnostic only**. The registered current-player 64.390716° failure is unchanged.

This agrees with the independently inspected recorder/interpolation behavior in
`docs/runtime-camera-timing-development.md`: position and orientation sampling
must remain separate. Do not blindly shift an entire pose record, calibrate an
agent with private telemetry, replace the original score with the passing
diagnostic, or call current-player control resolved. The public motion estimator
predicts the image camera, not automatically the current physical player.

## Verification

Full frozen regressions pass Windows **995 / two skips** (997 total), WSL
**997 / two skips** (999 total), `runs/past-support-full-{windows,wsl}-v1.xml`.
The isolated default-only profile passes **533 / 70 skips** (603 total), without
Torch, Pillow or OpenCV. The focused package/controller/comparison suite passes
57 cases on Windows. Unit-only fake process tests are not runtime evidence.

Both V22 scientific-core releases pass thirty seeds/two repeats/eighteen identical
artifacts per seed and all six checks. Each freezes 104 entries including 101
package files. These verify their declared structured/reproducibility scopes,
not visual navigation. Changed-file Ruff/format and whitespace checks pass;
existing global lint debt/hosted CI remain separate and unresolved/unverified.

All 104 source hashes match both V22 ledgers after verification. All 111 source
hashes and every persisted artifact hash match both comparison manifests after
the replicated runs. Those integrity checks do not change the physical scores.

## Next gate and unchanged full scope

The next practical gate is **frame-aware continuous control and actual arrival**,
not more private-score selection. Make the image/action sampling contract explicit,
including separate position/orientation phases; evaluate longer and settled
interventions with the same original scores. Then use public geometry in continued
local-map refresh/goal control, independently verify arrival, and retain failures.

Reliable goal navigation, at least five additional disjoint worlds/multiple
terrains/three conditions, natural scientific-map integration, 100 attempts/
recovery, current live baseline comparisons and consolidated CI/release remain
required. The project goal remains active, not complete or blocked.

In the installed optional Linux runtime, use a new output directory:

```bash
timeout 2500 /opt/minecraft-scientist-visual/bin/python scripts/compare_active_acquisition.py --seed 201 --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-past-support-comparison
```

For a single registered condition, pass `--active-causal --acquisition-strategy
past_support` or `fixed_cap` to `capture_runtime_camera.py`, then run the active
auditor. No provider/key, engine edit or terrain edit is required. Prior snapshots,
negative results and accepted route memory are preserved.
