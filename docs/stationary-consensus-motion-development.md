# Stationary context, consensus refinement and camera-profile development

These are optional RGB-geometry development experiments, not a navigation release.
They preserve public-only inference, fixed reliability thresholds and complete
trial retention. All experiments below use development seed 201; none is a new
held-out seed or evidence of broad terrain generalization.

## Implemented methods

`exploration/stationary_context.py` detects points in a three-frame past context.
It crops the world viewport before computing temporal patch range/MAD, initializes
tracking from a temporal median and returns coordinates in the last **real** image.
The synthetic median is never treated as a physical camera observation. Context
tracking uses the native pyramid level so a moving foreground at coarse resolution
cannot steer apparently stationary background seeds. Subsequent intervention
tracking retains the existing multiscale tracker. A declared MOVE/LOOK crossing
the past interval is rejected; future frames and physical telemetry are not inputs.
Lost identities are never reseeded or replaced.

`exploration/consensus_motion.py` refits the calibrated epipolar consensus before
rejecting the initial minimal pose. Three conditioned eight-point fits rescore the
inlier field before pose recovery; degenerate designs and rotation ambiguities
still reject. Conditioning follows centering and radius normalization described
in [Hartley's eight-point paper](https://www.cse.unr.edu/~bebis/CS485/Handouts/hartley.pdf).
Positive depth, spatial coverage, parallax, reprojection, withheld intermediate
points and all sixteen fixed sensitivity trials remain required. The estimator
returns a unit translation direction, not metric distance, consistent with
[OpenCV's pose-recovery contract](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).

`scripts/rgb_motion_variants.py` registers only `static-context`, `consensus` and
`static-consensus`. It freezes the variant/configuration before inference and
uses the same endpoint estimator for the candidate and every sensitivity trial.
The baseline remains the default. Unknown variants, changed configurations and
incomplete public evidence stop before geometry or evaluator scoring.

## Complete variant results

Each run preserves all thirty public pairs and passes thirty independent replay
checks. Moving and settled endpoints share the same eight movement interventions;
they are not sixteen independent trials.

| Variant / acquisition | Validated moving | Validated settled | Actual controls rejected |
|---|---:|---:|---:|
| Consensus / noon-clear | 0/8 | 1/8 | 13/13 |
| Static context / midnight-thunder | 0/8 | 0/8 | 13/13 |
| Static + consensus / midnight-thunder | 0/8 | 0/8 | 13/13 |

The consensus clear run also validates the one late-WAIT pair that actually
moves 0.31 blocks. This does not make it a successful negative control. The old
clear baseline validates two settled movements instead; the new consensus result
does not establish an overall gain. Static selection retains a median of three
common settled tracks in thunder, unchanged from the original baseline. None of
these candidates is promoted into accepted memory or navigation.

Artifacts, under `runs/`, are `rgb-motion-consensus-seed201-clear-v1`,
`rgb-motion-static-seed201-thunder-v1` and
`rgb-motion-static-consensus-seed201-thunder-v1`, each with `audit.json`.

## Explicit private camera-setting comparison

The installed SDK's bundled example mentions `bobView:true`, but inspection of
the **actual** instance runtime template finds no options file. The example
therefore cannot prove the graphics settings used by historical acquisitions.

`minecraft/client_camera_profile.py` makes a bounded private copy of the active
template and changes only `bobView`. It redirects the runtime pointer and launcher
method in this fresh Python process, snapshots actual instance options **before**
the JVM starts, restores both process attributes even on failure and verifies
that the shared SDK template stayed unchanged. Its own source, the SDK launcher
and the capture wrapper are frozen before acquisition. Existing directories are
refused, including on failed startup; neither SDK files nor terrain/physics rules
are edited. Failed acquisition bindings remain explicit failures.

Two fresh natural noon/clear sessions use the unchanged 42-action schedule and
capture 201 frames each. Their independently verified launch profiles pass twelve
checks each; RGB/calibration/motion replay passes thirty each. The consolidated
comparison reruns both component auditors and passes six comparison checks.
All 201 position/yaw/pitch records match exactly, with maximum component difference
zero. The public calibrated horizontal focal is 179.98847 pixels in both cases.

| Supplied setting | Validated moving | Validated settled | Median settled tracks | Actual controls rejected |
|---|---:|---:|---:|---:|
| Bobbing enabled | 0/8 | 2/8 | 131.5 | 13/13 |
| Bobbing disabled | 0/8 | 1/8 | 134.0 | 13/13 |

One nominal late-WAIT pair is physically moving in each case and is a failed
positive, not part of the thirteen successful actual controls. No wrong supported
motion is promoted. The fixed cropped-viewport comparison finds 157/201
nonidentical frames and a median mean absolute RGB difference of 0.361795.
That descriptive image difference does **not** prove the actual renderer flag,
nor isolate bobbing from ordinary cross-session rendering variation. There is
only one session per setting, and the diagnostic band crop is not a complete
hand/foreground segmentation. This pair does not support disabling bobbing as a
navigation fix, statistical superiority, or a general causal conclusion.

Raw runs are `controlled-rgb-motion-seed201-bob-{enabled,disabled}-v1`; adjacent
`.client-profile` directories contain configuration, original/supplied/launch
options, source snapshots, binding, report and audit. Inference runs are
`controlled-motion-bob-{enabled,disabled}-seed201-v1`. Consolidated evidence is
`runs/camera-bobbing-comparison-seed201-v2/audit.json`. The v1 comparison is retained
as historical evidence before that crop-scope comment was corrected.

## Reproduction and remaining work

From the repository root, in the installed WSL visual environment:

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_camera_profile.py --seed 201 --observation-size 448 --time noon --weather clear --bobbing enabled --output runs/new-camera-enabled
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_camera_profile.py --seed 201 --observation-size 448 --time noon --weather clear --bobbing disabled --output runs/new-camera-disabled
/opt/minecraft-scientist-visual/bin/python scripts/probe_controlled_rgb_motion.py runs/new-camera-enabled --output runs/new-camera-enabled-inference
/opt/minecraft-scientist-visual/bin/python scripts/probe_controlled_rgb_motion.py runs/new-camera-disabled --output runs/new-camera-disabled-inference
/opt/minecraft-scientist-visual/bin/python scripts/audit_camera_profile_comparison.py --case runs/new-camera-enabled.client-profile runs/new-camera-enabled runs/new-camera-enabled-inference --case runs/new-camera-disabled.client-profile runs/new-camera-disabled runs/new-camera-disabled-inference --output runs/new-camera-comparison/audit.json
```

For a registered variant, add `--variant consensus` (or either other registered
name) to the producer and replay with `scripts/audit_controlled_rgb_motion.py`.
Each existing target is refused. Full geometry replay uses the frozen OpenCV
runtime; the public launch-profile audit can run on Windows without MineStudio.
Historical source-frozen experiments are preserved, not relabeled as current
evidence after a producer/source change. Replay them using their frozen code and
dependencies, or acquire a separate new run.

The experiments change the next action to better motion observability and
keyframe/model selection, including an explicit check for planar/repeated-texture
ambiguity. That is a hypothesis to test, not a diagnosed universal cause. Candidate
improvements must increase correctly validated movement coverage and survive
fresh disjoint trajectories before planner integration. Multi-seed/terrain RGB
generalization, integrated natural scientific exploration and long-operation
recovery remain required. Regression and the strengthened scientific-core release
verification are documented in `docs/core-release-verification.md`.
