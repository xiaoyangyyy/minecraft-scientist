# RGB geometry development: evidence and limitations

This is an experimental branch of perception, not an accepted replacement for
the audited topological route planner. It is not enabled by default CLI commands.
The overall quality roadmap remains incomplete.

## Motivation and public boundary

The accepted natural seed-201 memory supports A→B→E→F→G route reuse. Its next
appearance-ranked expansion failed physical progress. Evaluator coordinates
cannot be used to select a different candidate. A spatially meaningful signal is
therefore needed before general goal-navigation claims can be expanded.

`visual_geometry.py` uses cropped public RGB, mutual SIFT descriptor matches,
rotation-scan focal calibration, essential geometry and cross-view consensus.
`visual_odometry.py` adds bounded 3D landmarks, recurrent forward/backward optical
flow, accumulated-baseline initialization and independently confirmed descriptor
identity. Subsequent poses require PnP support after refinement. A long temporal
gap cannot silently relocalize from repeated terrain descriptors.

Filtering and CLAHE are applied **after** cropping: HUD pixels outside the world
ROI cannot affect descriptor values through adjacent filters or histogram tiles.
Optical-flow pyramids are also built inside that crop, avoiding false stationary
edges at an artificial full-frame mask. Pixel correspondences retain descriptor
indices; coincident SIFT orientations are not confused by nearest-pixel lookup.

Unsupported positions are `null`, including every pre-initialization frame.
Accepted observations carry an explicit visual-origin frame. Distances are in an
unscaled monocular frame, not Minecraft blocks. Pose support is not proof of pose
accuracy. This branch contains no simulator-coordinate or hidden-FOV policy input.

## Development results (not held-out acceptance)

| Artifact | Observation | Decision |
|---|---|---|
| `runs/visual-geometry-development-v1/report.json` | Focal fit has 234 samples and relative MAD 0.00506, but neither G nor four candidate endpoints has sufficient direct A-image correspondences | Long-baseline bearing rejected |
| `runs/visual-geometry-development-v2-frozen/report.json` | Current cropped preprocessing gives 146 focal samples (relative MAD 0.00461); all five direct-bearing queries are still rejected. Code and PNG inputs are hash frozen | The correspondence bottleneck persists |
| `runs/dense-visual-route-seed201-v1/report.json` | 307 RGB frames on a fresh recognized A→B→E→F→G replay; every primitive and RGB before/after hash aligns | Usable development acquisition |
| `runs/visual-odometry-development-v2/report.json` | Immediate two-frame bootstrap creates 16 landmarks and subsequently loses tracking on 237 frames | No navigation promotion |
| `runs/visual-odometry-development-v5/audit.json` | Recurrent-flow bootstrap supports 20 poses. First eight poses calibrate gauge; subsequent 12 have RMSE 1.6123 and maximum error 2.6979 blocks against evaluator player positions | More pose support is **not** accurate navigation |
| `runs/visual-odometry-development-v8/audit.json` | Descriptor-confirmed bootstrap rejects all 307 frames; independent public replay, calibration, resizing and hash checks pass (16 checks) | Fail-closed behavior verified; navigation still unachieved |
| `runs/dense-visual-route-seed201-448-v1/report.json` | Fresh 448-pixel acquisition stops at node B after 126 frames; controls/hash alignment pass and accepted memory remains unchanged | Resolution shift is a negative route-recognition result |
| `runs/dense-visual-route-seed201-448-v2-normalized/integrity-portable-audit.json` | A fresh reference-scale-normalized replay recognizes all five route nodes, captures 307 frames and passes 24 independent checks. All five physical waypoint errors are 0; accepted memory stays byte-identical | Bounded high-resolution route reuse verified |
| `runs/visual-odometry-seed201-448-v1/audit.json` | Geometry on the accepted 448-pixel sequence also abstains on all 307 frames; 16 replay/integrity checks pass | Higher resolution alone did not establish geometric tracking |

These runs informed the method; none is a held-out seed/terrain evaluation.
The v5 similarity gauge is fitted only on the first eight supported poses, not
on scored poses. RGB camera motion differs slightly from player-position telemetry
(e.g. view bobbing); that caveat does not establish acceptance. The v8 auditor
explicitly reports `navigation_accepted=false`, with no pose-error claim when no
supported poses exist. Odometry runs v3 onward and the second direct-bearing
probe retain source snapshots. The earliest diagnostic records retain hashes only
and are not release-reproduction evidence.

Increasing capture resolution also changes image sampling for remembered edge
features. The new `ResolutionNormalizedSceneEncoder` makes reference pixel scale
explicit while retaining original dense RGB for geometry. Its integer area-pooling
and non-aliasing behavior are tested, and a fresh 448-pixel replay now passes.
No recognition threshold was loosened. This is one bounded resolution-shift result,
not broad viewpoint, condition or terrain generalization.

## Reproduction

Optional geometry tests (no Minecraft or credentials required):

```powershell
uv run --extra geometry pytest tests/unit/test_visual_geometry.py tests/unit/test_visual_pose_evaluation.py tests/unit/test_rgb_resolution.py
```

Dense acquisition requires the installed Linux MineStudio runtime and an accepted
audited source memory. Output directories are never overwritten:

```powershell
wsl -d MinecraftScientist -u root --cd /mnt/d/minecraft-scientist -- /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_visual_odometry_route.py --source-run runs/natural-visual-goal-search-seed201-v1 --output runs/dense-route-new --observation-size 224
wsl -d MinecraftScientist -u root --cd /mnt/d/minecraft-scientist -- /opt/minecraft-scientist-visual/bin/python scripts/probe_visual_odometry.py runs/dense-route-new --output runs/odometry-new/report.json
wsl -d MinecraftScientist -u root --cd /mnt/d/minecraft-scientist -- /opt/minecraft-scientist-visual/bin/python scripts/audit_visual_odometry.py runs/dense-route-new runs/odometry-new/report.json --output runs/odometry-new/audit.json
```

Dense acquisition can be audited directly on Windows without MineStudio, Java,
OpenCV, PyTorch or credentials (only the `visual` extra is required):

```powershell
uv run --extra visual python environments/minestudio/audit_dense_visual_route.py runs/dense-visual-route-seed201-448-v2-normalized --source-run runs/natural-visual-goal-search-seed201-v1 --output runs/dense-visual-route-seed201-448-v2-normalized/audit-new.json
```

That verifier recomputes temporal medians, source/data hashes, all public actions,
route localizations and heading alignments before comparing evaluator waypoints.
It currently requires an accepted goal-search source with independently restarted
waypoint evidence; older frontier-only sources lack that audit path.

The final regression for this checkpoint passes 241 tests on Windows and 243 in
the installed WSL runtime (two skipped on each). Package build and targeted lint
checks pass. A fresh offline MVP benchmark at seed 0 still reports intervention
accuracy 1.0, 47 experiments, detection delay 1, eight repair experiments, three
symbols and 13 reuse records (`runs/geometry-regression-mvp-seed0`). These are
engineering/structured-world regressions, not evidence of geometric navigation.

The producer freezes source bytes before processing and verifies every PNG and
decoded RGB hash. The auditor replays public frames before loading evaluator
coordinates; evaluator gauge parameters never return to policy. Auditing requires
matching geometry runtime source and OpenCV version. Reproduce old results using
their frozen code rather than attributing them to newer source.

## Next required work

Obtain stable, independently verifiable correspondences across real terrain,
turns and condition shifts; diagnose dynamic weather and repeated-texture aliases.
Then validate spatial progress with frozen policy on fresh trajectories before
integrating geometry into the goal-search planner. Keep the existing accepted
topological memory and negative runs intact. Only after that should broader seed,
terrain, longer-operation and combined scientific-exploration gates be claimed.
