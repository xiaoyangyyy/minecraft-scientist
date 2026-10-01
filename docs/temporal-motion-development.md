# Temporal RGB motion and public camera calibration

This is development on seed 201, not held-out navigation acceptance. No pose in
this branch is promoted to the accepted topological map or default planner. The
full scientific, spatial, generalization and long-operation gates remain in
`quality-roadmap.md`.

## Camera resize defect and repair

The installed MineStudio client renders a widescreen image and directly resizes
the full image to the observation dimensions. Its
[official source](https://github.com/CraftJarvis/MineStudio/blob/master/minestudio/simulator/entry.py)
documents these sizes and implements the resize in `_wrap_obs_info`. The installed
file has SHA256 `4f6d71c2ec8a14513ccd244da8728fdbd25b5fb75fb58a1177a3d8e33bceef72`.
The 224 acquisition uses a 640×360 render; the 448 acquisition uses 1280×720.
Stretching either to a square observation does **not** preserve equal x/y focal
lengths. Previous geometry runs incorrectly assumed `fx == fy`.

`camera_intrinsics.py` now represents the public forward model explicitly: native
square pixels, full-frame independent-axis resize, and pixel-grid principal point.
For native dimensions W×H and observation w×h:

```text
fx_observation = f_native * w / W
fy_observation = f_native * h / H
resized_pixel = (native_pixel + 0.5) * scale - 0.5
principal_point = ((w - 1) / 2, (h - 1) / 2)
```

Horizontal focal length is still estimated from public RGB rotation references,
not SDK FOV or evaluator telemetry. Shifting calibration feature coordinates by
half a pixel corrects the old calibrator's center convention without changing its
frozen source. The estimated reference horizontal focal is 90.147002 pixels;
at 224, vertical focal is approximately 160.261337 pixels. At 448 both double,
but the principal point moves from 111.5 to 223.5, not to 223.0.

Legacy acquisitions without render-size metadata require an explicit public
`--render-size`; there is no implicit aspect-ratio guess. Conflicting metadata,
unknown transform modes, nonfinite parameters and non-pinhole matrices fail closed.
Resize/ray-equivalence tests cover anisotropic scaling and exact pixel centers.

The corrected dense matching ablation leaves weights, pairs and thresholds fixed:

| Artifact under `runs/` | Correct MOVE / 8 | Correct controls / 5 | Replay checks |
|---|---:|---:|---:|
| `dense-correspondence-seed201-224-v3-resized-camera/audit.json` | 3 | 4 | 24 |
| `dense-correspondence-seed201-448-v2-resized-camera/audit.json` | 3 | 5 | 24 |

All checks pass, but both runs still contain strongly wrong supported motion.
The 448 result improves from 1/8 to 3/8 and removes its WAIT false translation;
224 remains 3/8 and now misclassifies one LOOK control. This identifies a real
camera-model defect, not a sufficient navigation fix or the sole cause of failures.
Earlier results remain valid historical replays of their frozen implementations,
but their incorrect intrinsics confound attribution solely to the matcher.

## Continuous tracking and independent intermediate checks

`temporal_tracks.py` detects up to 800 corners in the first cropped world ROI,
then preserves those identities over every adjacent frame. Failed identities are
never replaced or resurrected. Before optical-flow pyramids, the image is cropped
and median-filtered; HUD pixels cannot enter filtering or detection. Bounded
forward/backward tracking, texture, patch correlation, step displacement and a
reverse cycle through **all** intermediate images reject inconsistent tracks.
Correlation and cycle error are pixel-consistency statistics, not identity
probabilities. Surviving counts and original seed indices remain in the record.

`multiview_motion.py` retains rotation/degeneracy rejection, then refits the
calibrated essential matrix on the supported epipolar field rather than relying
only on a minimal sample. Exact synthetic tests exposed the minimal-fit problem;
the repair is a constrained eight-point refit, not relaxed acceptance limits.
First/last-frame triangulation provides a candidate map. Each intermediate frame
fits PnP using a deterministic training subset and scores distinct held-out track
IDs for positive depth, coverage and reprojection. Sixteen point/focal perturbation
trials must also pass the declared sensitivity bounds. Rejected evidence exposes
no executable pose. These internal geometric holdouts are not independent physical
navigation ground truth.

## Real replay evidence

Public action endpoints select the same eight MOVE and five control pairs as the
dense baseline. A preregistered three-image window mode includes **every** sliding
window within each selected MOVE: 24 movement windows plus five unchanged controls.
No window is selected using evaluator accuracy; overlapping windows are not
independent samples. Full-chain and shorter-window results are both preserved.

| Artifact under `runs/` | Movements | Correct endpoint candidates | Correct fully validated movements | Correct controls |
|---|---:|---:|---:|---:|
| `temporal-motion-seed201-224-v2-resized-camera/audit.json` | 8 | 0 | 0 | 5/5 |
| `temporal-motion-seed201-448-v2-resized-camera/audit.json` | 8 | 0 | 0 | 5/5 |
| `temporal-windows-seed201-224-v4-resized-camera/audit.json` | 24 | 1 | 0 | 5/5 |
| `temporal-windows-seed201-448-v3-resized-camera/audit.json` | 24 | 0 | 0 | 5/5 |

All four independent audits pass 26 checks. They reproduce every intermediate PNG,
track, camera calibration and multi-view calculation before opening evaluator
telemetry. Complete file sets, alignment, dependency versions, source snapshots,
frozen thresholds and post-replay hash stability are checked. Correctly rejecting
all movement is **not** useful navigation. The one correct short-window endpoint
cannot bypass the full validation/track budget. Both resolutions are fresh
acquisitions of the same route and seed, not broad-world generalization.

Whole-chain movement tracks number only 1–19 at 224 and 2–43 at 448; short windows
retain more points but still lack reliable validated geometry. Large adjacent
survival drops localize the next diagnostic to acquisition/correspondence under
movement. They do not prove that weather, occlusion or camera bobbing is the cause.
Player telemetry approximates the rendered camera and does not itself resolve
these alternatives.

## Reproduction and regression

Use the installed dedicated WSL visual runtime from the repository root. Output
directories/files must be new. Existing optional versions are listed in
`dense-correspondence-development.md`; no API key or remote inference is needed.

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_dense_correspondence.py runs/dense-visual-route-seed201-v1 --checkpoint runs/pretrained-loftr-outdoor-v1/loftr_outdoor.ckpt --sha256 21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec --render-size 640 360 --output runs/dense-camera-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_dense_correspondence.py runs/dense-visual-route-seed201-v1 runs/dense-camera-new --checkpoint runs/pretrained-loftr-outdoor-v1/loftr_outdoor.ckpt --output runs/dense-camera-new/audit.json
/opt/minecraft-scientist-visual/bin/python scripts/probe_temporal_motion.py runs/dense-visual-route-seed201-v1 --reference-config runs/dense-camera-new/config.json --render-size 640 360 --window-frames 3 --output runs/temporal-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_temporal_motion.py runs/dense-visual-route-seed201-v1 runs/temporal-new --reference-config runs/dense-camera-new/config.json --output runs/temporal-new/audit.json
```

Omit `--window-frames 3` for full interventions. For the 448 route use
`runs/dense-visual-route-seed201-448-v2-normalized`; its public report supplies
1280×720 render dimensions. Historical runs require their frozen source snapshots,
not current helpers attributed to old results. Partial/failed exploratory outputs
are retained, including the explicit missing-render-size failure on legacy data.

Windows geometry tests: 328 passed, two skipped. WSL visual tests: 330 passed,
two skipped. Clean default-only tests: 241 passed, 14 optional skips. Targeted Ruff,
wheel/source build and deterministic MVP regression pass. The new MVP report at
`runs/temporal-regression-mvp-seed0/benchmark.json` is byte-identical to the preceding
core report (SHA256 `b95af2e5c152b9f06edc63004ea87fb183c37e6a5a13bff53f21a39a4f48aeb8`):
accuracy/task success 1.0, 47 experiments, detection delay 1, eight revisions,
three symbols/13 reuses, transfer three versus eleven interventions.

Next: controlled real RGB acquisition that separates WAIT, short movement and
settled views, with a fixed action schedule and explicit public camera metadata.
Evaluate static-scene/condition/camera alternatives before changing the estimator.
Require useful correct movement coverage and fresh disjoint validation before
integration; no loosening thresholds or privileged selection of a better heading.
