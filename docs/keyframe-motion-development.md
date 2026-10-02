# Chronological keyframes and duration acquisition (2026-10-02)

This optional RGB development branch tests temporal observability, not completed
navigation. The method is fixed before each run, uses no evaluator input, retains
every candidate prefix and does not tune reliability thresholds. All results here
use development seed 201; broad held-out seed/terrain validation remains open.

## Executable public policy

`exploration/keyframe_motion.py` considers offsets 2, 4, 6, 8, 12, 18, 24 and 31
when within the declared image interval, and always includes its terminal offset.
Each fit starts from the same real anchor and contains every intervening image.
The tracker cannot reseed a lost identity inside that chain. Candidate seeds are
identical for a pair, not searched for a successful RANSAC draw.

The **first** prefix passing existing intermediate-frame PnP and all sixteen fixed
pixel/focal sensitivity trials is latched. Later attempts remain diagnostics and
cannot replace it using future geometry, a better private direction or goal
distance. Schema validation rejects out-of-order/duplicate attempts, missing
terminal evidence, incompatible track/motion counts and forged selection. The
external auditor additionally enforces the complete configured candidate set.

The selected unit translation describes its **local image interval only**. It is
not transported to the action's final frame and is not a metric displacement,
final pose, route completion or goal-arrival claim. The terminal candidate remains
the exact full-window baseline. This is a retrospective public-prefix evaluation,
not an integrated online controller or faster wall-clock navigation.

The producer freezes configuration, all public acquisition inputs, source bytes
and complete prefix results. The independent auditor reconstructs camera/config
and every prefix, including failures, **before** its first evaluator read. It then
scores selected local intervals and terminal intervals separately, preserving all
candidate scores without allowing reselection. The RGB reference calibration
precedes the movement trials; earlier startup/rotation controls use the same
offline calibration and are not presented as an online calibration demonstration.

## Existing image sessions

Each original 201-frame clear/thunder session retains 30 pairs and **92** prefix
fits. Both independent audits pass **33** checks, with no navigation promotion.
Moving and settled endpoints share eight interventions; they are not sixteen
independent motion experiments.

| Acquisition | Correct selected moving local intervals | Correct selected settled local intervals | Correct terminal moving / settled |
|---|---:|---:|---:|
| Noon/clear, 3/6 move ticks | 0/8 | 2/8 | 0/8 / 2/8 |
| Midnight/thunder, 3/6 move ticks | 0/8 | 0/8 | 0/8 / 0/8 |

Clear `settled-00` selects its terminal offset 15. Clear `settled-07` selects
offset 12 rather than terminal 18: six image frames earlier, **not** three and not
a measured compute/control-time improvement. There is no coverage increase on
these sessions. Both retain thirteen actual negative controls and reject them;
one late-WAIT pair actually moves and is a failed positive, not a negative success.

Artifacts are `runs/rgb-keyframe-prefix-seed201-{clear,thunder}-v1`, each containing
config/report/source snapshots and `audit.json`.

## Fresh longer-duration acquisition

`environments/minestudio/capture_motion_duration.py` preregisters a validated
`MotionProbeConfig` with movement durations **6/10** and unchanged 12-frame settle
waits. It delegates to the unchanged natural-spawn capturer through one
process-local constructor override, records the actual constructor call and
restores that alias and argv on success, exceptions and `SystemExit`. Neither the
shared SDK/camera profile nor terrain/control mappings are modified. Raw and
adjacent profile directories must both be new; nested constructor hooks reject.

The adjacent `.duration-profile` freezes the wrapper and canonical capturer
sources before launch, checks five explicit execution/restoration invariants and
binds the raw configuration/report hashes. `audit_motion_duration.py` independently
checks eight profile/source/configuration/binding/public-action conditions without
opening telemetry. This is separate acquisition provenance, not an estimator
input or a substitute for motion scoring.

The fresh noon/clear run completes **229 frames** and all 42 actions. Its profile
and public acquisition pass; both full-window and prefix inference are then
frozen and independently replayed. The full-window audit passes 30 checks, and the
keyframe audit passes 33, retaining all 30 pairs and **108** prefixes.

| Measured interval | Selected local correct motions | Terminal baseline correct motions |
|---|---:|---:|
| Moving | 1/8 | 1/8 |
| Settled | 3/8 | 2/8 |

There are no wrong supported selected local directions or false selected local
translations. Thirteen actual negative controls are rejected by both methods;
one physically moving late-WAIT positive remains unsupported. Moving-04 and
settled-04 select the **same** offset-8 image pose and must not be double-counted
as independent successes. Three unique movement interventions have correct local
poses, not four. All selected cases:

| Pair | Selected / terminal offset | Local direction error | Local rotation error |
|---|---:|---:|---:|
| settled-00 | 8 / 18 | 13.004° | 0.378° |
| settled-03 | 12 / 18 | 2.091° | 0.222° |
| moving-04 | 8 / 10 | 7.078° | 0.925° |
| settled-04 | 8 / 22 | 7.078° | 0.925° |

The extra `settled-00` local pose is correctly scored at offset 8; its terminal
offset 18 still abstains. This is useful bounded local observability, **not** a
recovered final displacement. Independent comparison finds all thirty terminal
track records and poses numerically identical between the prefix and original
full-window producers, with the same seed, pairs, camera and acceptance thresholds.

The longer action schedule changes subsequent physical starting contexts, and
there is only one new rendering session. Do not infer statistical superiority or
a causal duration effect from aggregate comparison with the earlier 3/6 session.
The within-session local/terminal comparison is likewise interval-specific, not
a matched goal-navigation success comparison. No candidate is promoted into memory.

Artifacts:

- `runs/controlled-rgb-motion-seed201-longer-clear-v1` (raw 229-frame capture)
- `runs/controlled-rgb-motion-seed201-longer-clear-v1.duration-profile` (source/configuration binding and audit)
- `runs/controlled-motion-seed201-longer-clear-v1` (full-window report/audit)
- `runs/rgb-keyframe-prefix-seed201-longer-clear-v1` (all prefix results/audit)

## Regression, reproduction and next action

Thirteen policy tests (one optional real-image identity test), ten public-pipeline
tests, twelve duration-wrapper/budget tests and seven duration-audit tests verify
the boundaries. Pipeline geometry fixtures are explicitly **not** real-image
accuracy evidence. Corrupt declarations, missing failed prefixes, changed replay,
forged selector/relaxed gates and changed sources stop before private scoring.
Startup failures preserve partial evidence without reducing the declared budget.
Shared Windows/WSL pytest assertion-cache paths are normalized only in the fixture
mock; production source ledgers are not weakened.

Current regressions: Windows **496 passed/two skips**, WSL **498/two**, clean
default-only **318/41 optional/platform skips**. The current v5 offline scientific
release passes 30 seeds/two repeats/18 identical artifacts on both platforms.

In the installed WSL visual environment, from the project root:

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_motion_duration.py --seed 201 --short-move 6 --long-move 10 --time noon --weather clear --output runs/new-longer-capture
/opt/minecraft-scientist-visual/bin/python scripts/audit_motion_duration.py runs/new-longer-capture --output runs/new-longer-capture.duration-profile/audit.json
/opt/minecraft-scientist-visual/bin/python scripts/probe_keyframe_rgb_motion.py runs/new-longer-capture --output runs/new-prefixes
/opt/minecraft-scientist-visual/bin/python scripts/audit_keyframe_rgb_motion.py runs/new-longer-capture runs/new-prefixes --output runs/new-prefixes/audit.json
```

The acquisition/profile auditor can run on Windows with the optional visual extra;
geometry replay uses the frozen WSL/OpenCV environment. Outputs always refuse
overwrite. Use the original controlled producer/auditor for full-window comparison.

Next test stronger active-parallax/non-collinear trajectories and fresh disjoint
sessions before integrating keyframe execution into the planner. The method still
misses most movements and does not solve thunder tracking. Held-out seed/terrain
generalization, metric/topological motion integration, actual goal arrival,
natural scientific exploration and long-operation recovery remain required. The
bounded local gain does not redefine the full project completion standard.
