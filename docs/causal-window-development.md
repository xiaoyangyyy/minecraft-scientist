# Causal short-window mapping: development checkpoint

Current-prefix validation and explicit local refresh are implemented, but the
fresh replay is still negative: all four horizons yield no accepted estimate.
This is not useful goal navigation, an online game-policy run, or completion.
Rejecting real movement is an accuracy failure, not a successful stationary test.

## Public controller contract

`exploration/causal_map_window.py` contains the reusable `CausalMapWindow`:

1. Fit every bootstrap in a bounded past window, with original disjoint
   fitting/withheld landmarks and unchanged geometric support floors.
2. Accept exactly the next public `TemporalTracks` prefix on each `advance`.
   Preserve earlier pixels/anchor and intersect identities permanently.
3. At the registered two-tick horizon, validate only its current endpoint and
   current surviving identities. Later survival cannot rewrite an earlier gate.
4. Require at least two passing initializations and agreement of **all** passing
   initializations within 10 degrees direction / 2 degrees rotation. No private
   winner or preferred agreeing subgroup is selected. This checks initialization
   stability on correlated data, not independent-sample confidence.
5. Return `refresh` or `reject`, then refuse more input until explicit refresh
   creates a window ending exactly at the current observation. The old window
   remains unchanged; its snapshot digest becomes the child's parent.

Window count, horizon, model count, seeds and fitting budgets are bounded. The
controller imports no evaluator or filesystem API; input contains no coordinates,
hidden rules or renderer matrices. Snapshots return copies and invalid prefixes
preserve prior state. Consensus has unknown metric scale and
`navigation_accepted=false`; local gauges are not stitched into global position.

## Fresh replay and measurement boundary

`runs/runtime-camera-projection-seed201-v4` retains 106 public/native frames,
all 105 fixed actions, six passing acquisition checks and 107 source entries
(100 package files plus seven capture/agent/SDK definitions). It is another
capture on the existing development seed, not another held-out world.

`scripts/probe_causal_windows.py` fits ten past ticks, observes two new ticks,
records the decision, refreshes using only already observed images, then observes
two more ticks. Refresh after rejection is explicitly a recovery diagnostic, not
accepted navigation. The first horizon uses a control distinct from both original
fitting controls. The refreshed window already includes that control in its past,
so the second horizon is temporal holdout, **not** new-control holdout.

All four `(anchor, fitting end, horizon end)` windows remain:
`(48,58,60)`, `(50,60,62)`, `(79,89,91)`, `(81,91,93)`. Each preserves all nine
bootstraps and both one-tick future observations. This replays saved images; it
does not select or stop live game actions. Live action-dispatch integration remains.

The existing public-first runner/auditor now accepts explicit inference, policy,
source and denominator callbacks, reusing the same private-byte freeze and
camera/body scoring rather than creating a weaker evaluator. Both
`runs/causal-windows-seed201-v1/audit.json` and the matched-data reference audit
pass thirteen integrity/replay checks; their nested renderer audits pass eighteen.
Integrity checks are not accuracy or navigation acceptance.

## Result and matched-data reference

| Window | Past tracks | Horizon tracks | Supported past maps / nine | Passing future maps | Controller |
|---|---:|---:|---:|---:|---|
| 48→58→60 | 90 | 65 | 1 | 0 | reject |
| 50→60→62 | 63 | 58 | 0 | 0 | reject |
| 79→89→91 | 31 | 27 | 0 | 0 | reject |
| 81→91→93 | 25 | 23 | 0 | 0 | reject |

Rendered translation is respectively 0.854554, 0.402760, 0.296199 and 0.322558
blocks, each above the unchanged 0.15-block scoring threshold. Coverage is **0/4**;
all unsupported-motion accuracy scores fail. Unsupported poses get no misleading
zero-degree error. Both second-trial windows lack the unchanged 32-track floor.

The old fixed-ten-tick/four-future-frame method is also rerun on the **same capture**
in `runs/causal-baseline-rendered-motion-seed201-v1`. One of eighteen past candidates
is locally supported (rendered direction error 3.335984 degrees); none passes full
future validation. Both methods share images/actions, but the causal method fits
36 versus 18 candidates and evaluates different local intervals. This is not an
equal-compute comparison or evidence of improved accuracy. Earlier partially
positive captures have different images/source boundaries and remain historical.

## Next action and full scope

The failure is still acquiring well-conditioned geometry and retaining original
landmarks, not merely postponing validation for four ticks. Investigate adaptive
public-only acquisition: freeze at the first eligible prefix and deliberately
create parallax if necessary. Preserve every attempted prefix/cost; never select
endpoints by private scores or reduce support/error gates. Then connect the state
machine to bounded live action dispatch and independently validate new trajectories.

Reliable navigation, five additional disjoint worlds/multiple terrains/three
conditions, natural scientific/map integration, 100 attempts/recovery, current
live/baseline comparisons and consolidated CI/release remain open. The broad goal
is active, not complete or blocked.

## Verification and reproduction

Full frozen regressions pass Windows 951/2 skips (953 total), WSL 953/2 skips
(955 total), `runs/causal-window-full-{windows,wsl}-v1.xml`. Both focused
controller/runtime/joint-map suites pass 86/86. Both V20 core releases pass thirty
seeds/two repeats/eighteen identical artifacts per seed and six checks, freezing
103 entries including 100 package files. The MVP CLI, builds and changed-file
Ruff/format pass. These checks certify their declared scopes, not project completion.

The isolated default-only WSL profile passes 495 cases / 70 skips (565 total),
`runs/causal-window-default-only-wsl-v1.xml`, without Torch, Pillow or OpenCV.
The real-geometry test skips there; the pure controller/state/schema tests run.
All 103 V20 source hashes were rechecked on both platform ledgers with no mismatch.

In the existing optional WSL runtime, use new output names:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --seed 201 --output runs/new-causal-capture
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_causal_windows.py runs/new-causal-capture --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-causal-windows
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_causal_windows.py runs/new-causal-capture --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --audit runs/new-causal-windows --output runs/new-causal-windows/audit.json
```

No provider/key is used or historical output overwritten; no engine JAR, shared
SDK or terrain is edited. The default scientific core remains offline.
