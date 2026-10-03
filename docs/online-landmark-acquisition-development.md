# Causal protection of fitted map landmarks

This changes the active experiment controller, not navigation acceptance. The
complete requirements in `quality-roadmap.md` remain. It follows the public
map-ID loss diagnosis in `phase-visibility-development.md`, not private-label
tuning of directions or error thresholds.

## Implemented acquisition contract

The capture policy explicitly declares `first_supported_map_v1` alongside
`phase_start_v2`. Raw acquisition schema is `visibility-rgb-acquisition-v2`.
Twelve completed startup RGB yaw views calibrate the same public resize/pinhole
camera as the existing offline estimator; no simulator FOV or pose is consumed.
The algorithm and fixed fitting-seed schedule are frozen before world startup.

At each first-leg prefix of at least two executed ticks, the online fitter uses
only that prefix's permanent original IDs. It removes IDs lost at earlier steps,
even when full-prefix tracking returns them. All unsupported attempts are retained.
The first supported candidate is frozen once; no replacement, reseeding or new
training/withheld partition is allowed. Its digest binds complete fitting evidence.

The controller applies the existing one-step pixel extrapolation/central ROI/
12-reference-pixel margin to the frozen fitting and withheld IDs, in addition to
the generic population. Either risk can request the next distinct public handle.
No new relative-map-loss normalization or looser threshold is introduced.
Protected support below 32 total/12 fitting/8 withheld IDs, or below two withheld
spatial cells, aborts the probe before further actions. Aborts are not map
contradictions, successful experiments or navigation completion.

Global retention, phase-relative retention, 32 generic tracks/two generic cells,
minimum two ticks, six/four maximum ticks and the 209-frame acquisition ceiling
remain unchanged. Protecting a candidate is an experimental use, not proof of its
direction or uniqueness. Distinct handles do not guarantee physical non-collinearity.

## Independent replay and fitting identity

The public-first auditor reconstructs startup calibration, each actual one-tick
receipt, every filtered online attempt, the first-supported freeze, all protected
partitions, controller decisions and aborts. Complete calibration/seed/attempt
metadata and current source/snapshot bytes are checked before private scoring.

First-leg inference uses the actual online attempts, not a retrospective new
keyframe estimate. A separately labeled terminal diagnostic may be appended, but
cannot replace the first supported map. Subsequent direction-search/PnP checks use
the captured fitting seed, even if a different diagnostic CLI seed is supplied.
Incomplete first legs retain missing-leg markers and raw fitting evidence, rather
than fabricating a completed movement. All other groups retain their declared
chronological keyframe schedule and complete trial/control denominators.

Historical raw-V1/V2, earlier visibility-map predictions, and V14 scientific-core
source ledgers are unchanged and now precede this package/script source boundary.
The parent public action-pair plan remains unchanged: seed 201/noon/clear is still
development, not disjoint seed/terrain/condition generalization or matched paths.

## Tests and completed execution evidence

Seventy-three focused tests pass. New coverage includes map-specific warnings
missed by generic visibility, protected-support aborts despite generic survivors,
permanent lost IDs/disjoint partitions, immutable candidate digest, invalid/stale/
invented/unsupported attachments without controller mutation, explicit policy,
current-prefix callback order, filtered fitting, actual synthetic multi-view
geometry, calibration/seed/attempt tampering and no retrospective map replacement.
Policy fixtures and synthetic rays do not establish natural-world accuracy.

Full regressions pass **749 tests/two skips on Windows**, **751/two on WSL**, and
**427/61 optional or platform skips** in the isolated default-only offline
installation. JUnit files are `runs/online-landmark-regression-{windows,wsl,
default-offline}-v1.xml`; focused results are `runs/online-landmark-targeted-v2.xml`.
Both V15 scientific-core releases pass all six checks for thirty seeds/two repeats/
eighteen artifacts and freeze 99 source entries. Current bytes match both ledgers.
The fresh seed-zero MVP report remains byte-identical to the earlier benchmark:
47 reported experiments, task/intervention accuracy 1.0, three symbols/thirteen
reuses, and executable transfer three versus eleven interventions. These are
finite structured-world results, not open-world accuracy or total auxiliary cost.

## Fresh natural-world acquisition and independent audit

`runs/controlled-rgb-landmarks-seed201-clear-v1` completes **180 RGB frames**
(448-by-448 resized from 1280-by-720), **85 public actions**, and all seven raw
integrity gates. Eight trials are retained: seven finish and one aborts. Seventeen
online attempts produce two frozen first-leg maps. The unchanged maximum remains
209 frames; raw provenance freezes 124 project/SDK source entries.

`runs/rgb-landmark-maps-seed201-v1` completes **46 records/128 prefixes**, binds
127 inference source entries and passes all **26 independent audit checks**.
Public calibration, acquisition decisions, fitting identities and source/input
integrity are replayed before privileged pose scoring. Navigation is **not accepted**.

| Group | Trials | Selected correct / wrong | Correct terminal moving |
|---|---:|---:|---:|
| Combined moving | 8 | 3 / 0 | 2 |
| Combined settled | 8 | 4 / 0 | 1 |
| First leg | 8 | 2 / 0 | 2 |
| Second leg | 8 | 1 / 0 | 1 |

Ten selected pair records represent **five unique intervals, five correct/zero
wrong**. Overlapping groups are not independent successes. All fourteen actual
stationary controls reject translation. Six/eight control pairs are physically
non-collinear, but that alone does not validate any supported map's future views.
No wrong supported prefix is observed in this capture; low coverage and changed
paths prevent a matched improvement or generalization claim.

At anchor 89/tick two, generic projected-edge risk is 0.117845 but frozen map
risk is **16/48 = one third**, triggering the switch a tick earlier than the old
generic policy. The map initially has 36 fitting/12 withheld landmarks. One tick
later only five fitting/zero withheld remain despite 226 generic survivors, so
the controller aborts with no holdout validation or invented map contradiction.
The switched physical movement is only 0.015277 blocks: early switching does not
prove informative excitation or rescue the map. The causes of lost tracks are
not yet established; they cannot all be attributed to projected edge risk.

The other first-leg map, 141-to-145, has direction error 7.499209 degrees and keeps
69 fitting/22 withheld IDs in later views. Its optional diagnostic survives, but
the shadow movement is antiparallel (180 degrees) and `no_prefit_witness` has an
incomplete alternative family. It is not unique/non-collinear map validation.
The only supported second-leg map, 159-to-163, has error **19.966090 degrees**,
barely inside the unchanged 20-degree limit. It has no later different-control
validation. Its one/eight coverage is not reliable navigation.

The previous hand calculation of 48/48 endangered landmarks incorrectly used
224-pixel bounds for actual 448-pixel frames. Correct bounds with the scaled margin
give 16/48 in both the historical and fresh public tracks. The controller already
used actual dimensions; documentation is corrected without changing old artifacts.

## Remaining work

Second-leg subsequent different-control validation, genuinely informative active
excitation and useful reliable navigation remain required; this first-leg landmark
protection is not a substitute for those or the broad project completion gates.
Next diagnose per-ID tracking losses and verify the rendered-camera/pose contract
without tuning policies to private labels. Then obtain independent subsequent
views, held-out worlds/terrains/conditions, natural science/map integration,
long-horizon recovery and current live release/baseline/CI evidence. The original
full-project requirements remain intact.
