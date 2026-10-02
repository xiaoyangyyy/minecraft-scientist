# Plane-aware motion competition (development, 2026-10-02)

This is a tested optional estimator, not accepted navigation. It investigates
whether planar ambiguity contributes to the failed RGB motion coverage, without
assuming that this is the universal cause. All real-image results below are from
development seed 201 and existing sessions, not new held-out acquisition.

## Public model comparison

`exploration/plane_motion.py` fits a homography on a seeded training-ray partition
and retains all finite translation branches. It also proposes a conditioned
essential-model pose fitted on the **same training rays only**. Neither model
family fits selection or shadow rays. Homography decomposition can leave distinct
motion solutions even after positive-depth filtering; see the primary
[OpenCV decomposition contract](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).

The default split is approximately 50% training, 25% selection and 25% shadow,
with at least 20/8/8 points. Larger declared training budgets are preserved rather
than consumed by the fractional holdouts. The training plane requires at least
12 inliers, 50% training support and the existing spatial-coverage threshold.

Both independent partitions must contain at least eight off-plane rays, defined
by symmetric transfer error above three times the unchanged RANSAC threshold.
Selection chooses one branch using clipped standardized epipolar error and
negative-depth penalties. Shadow rays may veto that branch but cannot reselect a
different winner. Both partitions require at least 80% positive-depth,
reprojection-, epipolar- and parallax-valid off-plane points, plus a loss margin of
0.25 against every nonequivalent branch. Only poses within five direction degrees
and one rotation degree are treated as equivalent. Low-parallax competitors are
retained in the loss comparison so their early removal cannot manufacture a unique
solution. Pure rotation, stationary and unsupported planar pairs reject.

The proposed endpoint still needs the original count, coverage and parallax gates,
withheld intermediate-frame PnP, and all sixteen fixed pixel/focal sensitivity
trials. No action direction, simulator coordinate or physical arrival measurement
is an estimator input. Output translation is unit direction, not metric distance.
Diagnostic branches are retained even on failure; they are not executable memory.

`plane-competition` is a registered optional variant. Configuration and complete
input/source snapshots are frozen before inference. Reports include partition IDs,
off-plane IDs, both families' branch poses/losses/acceptance, the selected branch,
selection/shadow margins, all tracks and existing multi-frame/sensitivity evidence.
The same estimator/configuration is used for the candidate and sensitivity trials.
Independent replay checks these diagnostics before opening evaluator telemetry.

## Complete real-image results

Both runs retain all thirty fixed pairs and pass all thirty independent replay
checks. Moving and settled endpoints share eight interventions; they are not
sixteen independent motions.

| Acquisition | Correct validated moving | Correct validated settled | Actual controls rejected |
|---|---:|---:|---:|
| Noon/clear | 0/8 | 0/8 | 13/13 |
| Midnight/thunder | 0/8 | 0/8 | 13/13 |

The bare two-frame clear estimate scores one correct moving endpoint, but no
common-chain estimate survives the full validation path. One nominal late-WAIT
pair actually moves in each session and remains a failed positive; it is excluded
from the thirteen actual negative controls. There is no wrong supported protected
motion and no memory promotion. Zero positive coverage is failure, not navigation
success. The old clear baseline retained two correct settled movements, so this
candidate is not an overall improvement and must not replace it.

Among the sixteen clear moving/settled common chains, nine lack the required
training-plane count/fraction/coverage, two lack independent off-plane points,
four fail selection-ray support, and one lacks final parallax/coverage. Among the
sixteen thunder chains, twelve lack enough rays for the partition budget, three
lack training-plane support, and one fails selection support. These public failure
categories do not identify a universal physical cause. Adding a plane solver alone
does not address the main measured coverage problem.

Artifacts:

- `runs/rgb-motion-plane-competition-seed201-clear-v1/{config,report,audit}.json`
- `runs/rgb-motion-plane-competition-seed201-thunder-v1/{config,report,audit}.json`

## Tests and numerical limits

Fourteen geometry unit tests cover mixed-depth directional recovery, exact replay,
pure plane/rotation/stationary rejection, shadow-only contradiction without
reselection, common intermediate/sensitivity gates, resolution scaling, unrelated
matches, incomplete/forged partitions, invalid parameters, training-only fitting
of both families, custom training budgets, and branch-order invariance.

The wide-plane mixed-depth positive passes the full path on OpenCV 4.8 and 4.13.
A narrower synthetic positive passes sensitivity on 4.13 but abstains on 4.8;
the test independently derives the decision from measured perturbation statistics
and requires the same fixed thresholds, rather than forcing both numerical
backends to accept. Real-image replay stays within its frozen WSL/OpenCV runtime;
there is no cross-version pose-equality claim. No reliability threshold was relaxed.

## Reproduction and next action

From the project root in the installed WSL visual environment:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_controlled_rgb_motion.py runs/controlled-rgb-motion-seed201-clear-v1 --variant plane-competition --output runs/new-plane-clear
/opt/minecraft-scientist-visual/bin/python scripts/audit_controlled_rgb_motion.py runs/controlled-rgb-motion-seed201-clear-v1 runs/new-plane-clear --output runs/new-plane-clear/audit.json
```

Use the thunder acquisition for the second case. Existing destinations are refused.
Older variants' snapshots remain historical when the registration/source set
changes; replay with their frozen code or create a separate new run.

The next navigation action is public keyframe/active-parallax selection and
non-collinear acquisition, not deployment of this unsuccessful estimator. A new
method must improve correctly validated movement coverage and survive disjoint
trajectories before planner integration. Broad seed/terrain generalization,
integrated natural scientific exploration and long-operation recovery remain
required; this bounded diagnostic does not narrow the project completion gates.
