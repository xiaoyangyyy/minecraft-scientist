# Public settling exposes two different continuation failures

The package now implements a source-frozen public MOVE/WAIT continuation and its
independent chronological replay. Three fresh natural-world development sessions
(seeds 201, 202, 203, clear/noon) complete their capture/audit protocols. **None
completes the three settled motion trials; reliable navigation is still open.**
These are sequential development probes, not a preregistered held-out suite or a
significance/generalization claim. Negative and unstarted conditions are retained.

## Public control and evidence contract

`exploration/settled_map_control.py` continues only a validated acquisition. It
reproduces the supplied public image/track/future checkpoint before dispatch,
retains every frozen bootstrap and its original landmark partitions, and never
refits maps, replenishes features, shifts telemetry timestamps, or asks for private
positions. Frame ordinals name returned images, not current-player pose samples.

The declared controls are WAIT followed by three experiments, each consisting of
four one-tick MOVE actions and WAIT. A settle needs at least three waits and two
consecutive stable prefixes, with a maximum eight waits. Stability requires at
least two consistent maps, the same population in both prefixes, at most 0.01
translation relative to each map's **past-only scale**, and 0.2 degrees rotation.
The scale is monocular, not blocks. Existing original-track, withheld-pixel and
all-model consensus gates are unchanged, including the three-pixel 95th-percentile
limit. Failed receipts cannot enter pose evidence. Loss of support aborts; partial
movement endpoints remain in the output without a supported motion prediction.

`capture_runtime_camera.py --active-causal --settled-controls` freezes the distinct
`settled-causal-camera-projection-v1` protocol before launch. `audit_active_causal_map.py`
independently regenerates the acquisition and every requested MOVE/WAIT, exposing
later images only through their chronological action receipts. It compares the
complete result before private scoring. The new 105-frame upper bound does not
enlarge the old active protocol's 61-frame bound or the fixed schedule contract.
Schema/strategy/policy/receipt forgery and late data mutation fail closed.

Original acquisition camera/body scores are retained separately. Every attempted
settled trial is retained, including partial trials; there is no privately chosen
best interval/map. The same 0.15-block translation, 20-degree direction and
five-degree rotation thresholds apply. All navigation flags remain false.

## Actual runtime outcomes

Each run preserves config, public PNGs/actions, native images, evaluator-only
camera/player records, all attempted maps/prefixes, 109 source entries (102
package files), and the auditor source snapshot. The engine/shared SDK and natural
terrain are not edited. Eight capture checks, three outer audit checks and eighteen
nested camera/public-boundary checks pass in every run. Integrity is not accuracy.

| Development seed | Frames | Initial acquisition | Fits | Continuation | WAIT actions | Settled trials completed |
|---|---:|---|---:|---|---:|---:|
| 201 | 63 | validated | 27 | support_aborted | 4 | 0 |
| 202 | 57 | validated | 2 | support_aborted | 3 | 0 |
| 203 | 59 | exhausted | 44 | not_started | 0 | 0 |

Artifacts are `runs/settled-controls-live-seed{201,202,203}-v1`; each contains its
own `report.json`, `active-acquisition.json`, `audit.json`, and `audit.source.py`.
All three conditions, not just the two initially validated ones, remain in scope.

Seed 201 preserves a correct acquisition **rendered-camera** direction estimate
(0.916233 degrees, 0.105209 degrees rotation, 0.253980 blocks) and a wrong **current
body** direction (64.389974 degrees, 0.233692 blocks), over frames 56 to 58.
During WAIT frames 58 to 62, the current player's height decreases from
87.766368 to 86.407395. This is measured downward motion, not merely horizontal
inertia; WAIT cannot be equated with physical rest. Public map gates abort at 62:
86 original tracks remain, but both withheld 95th-percentile errors exceed three
pixels (3.629489 and 3.535923). No movement trial is started.

Seed 202 has a different counterexample. Acquisition frames 51 to 53 predict
current-body direction correctly (4.912995 degrees, 0.086087 degrees rotation,
0.2 blocks). Rendered translation is only 0.098483 blocks, below the unchanged
0.15-block gate, so the supported translation is scored `false_translation`, not
credited as success. Across WAIT frames 53 through 56, every current-player pose
component is unchanged. Nevertheless, one map's withheld-pixel error rises to
3.182983 pixels while the other remains at 1.915792; the population falls below
two and the controller aborts. It does not select the surviving map to obtain a
passing result. This rules out actual falling as a universal explanation for
the continuation failure; map uncertainty/rendered-camera motion still matters.

Seed 203 retains all 44 failed fits within its ten-tick acquisition cap. No eligible
map is frozen, and no WAIT or movement continuation is dispatched or scored.

## Regression and source boundaries

Current V23 offline scientific-core releases each pass thirty seeds, two repeats
and eighteen identical artifacts per seed, with all six checks. Each freezes 105
entries including 102 package sources. Their mean reported experiment count is
53.9333333 and minimum equation-transfer saving is one experiment. These scoped
structured checks do not validate navigation or optional live geometry.

The current isolated default-only WSL profile passes 556 tests with 70 skips,
without Torch, Pillow or OpenCV. Targeted Windows tests pass 76 cases before the
additional successful synthetic replay test, which separately passes; unit fake
geometry/JVM evidence is explicitly not real navigation proof. The new pure
controller has 23 unit cases, including control failures, immutable checkpoints,
support loss, unstable waits, strict budgets and preserved partial endpoints.
Full frozen regressions pass **1028 tests/two skips on Windows** (1030 total)
and **1030/two on WSL** (1032 total), recorded in
`runs/settled-control-full-{windows,wsl}-v1.xml`. All new optional and pure-core
tests are included; full-test success does not certify live navigation.

After verification, all 105 current source hashes match both V23 release ledgers.
For each live capture, all 109 source snapshots and 107 current project source
entries match; the two external SDK entries are verified as capture snapshots,
not claimed as current Windows SDK files. All 130/118/122 persisted capture
artifact hashes respectively match, as do each saved active-auditor snapshot and
both current auditor hashes. No historical source/score is rewritten.

The seed-zero MVP still reports 47 experiments, accuracy/task success 1.0, crisis
delay one, eight repair experiments, description length 1307, three symbols and
thirteen reuses. Package builds and changed-file lint/format checks pass. Existing
global lint debt, hosted CI and current live-fixture/baseline release are separate
outstanding gates. Earlier V22 comparisons/releases remain historical; they are
not restamped with the new source identity.

## Next action and unchanged full scope

Waiting alone with one frozen short-baseline map is insufficient. Add bounded
public-image landing/stability recovery with explicit evidence epochs, followed
by past-only map refresh/reacquisition and independent new-control validation.
Any new features or coordinate gauges need explicit lineage/registration; do not
silently replenish the failed original map or use private height to select
landing time. Maintain both body and rendered-camera scores and retain failures.

That recovery must then enable continued public goal control and independently
verified actual arrival, not become a replacement success criterion. At least
five additional disjoint worlds, multiple terrains/three conditions, adequate
samples/uncertainty, integrated natural scientific discovery/cognitive maps, 100
frontier/experiment attempts with restarts/failures/recovery, current baseline and
symbol/planner comparisons, and consolidated CI/release remain required. The
full project goal remains active, not complete or blocked.

## Reproduce in the installed optional Linux runtime

Use a new output directory; existing capture/audit outputs are refused:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --active-causal --settled-controls --seed 201 --output runs/new-settled-control
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/audit_active_causal_map.py runs/new-settled-control --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-settled-control/audit.json
```

No model/provider/API key is needed. Live pixels and timings are not claimed byte
deterministic; the structured-core exact-repeat claim is a separate contract.
