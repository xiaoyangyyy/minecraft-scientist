# Shared-map settled endpoint development

This is a new counterexample/maintenance revision and a new physical-control
experiment, not a restamping of the shared-anchor checkpoint (`88d1dc1`). Earlier
failed controls, models and gauge links remain immutable. It does not establish
persistent global mapping, reliable goal arrival or full-project completion.

## Public policy and protected evidence

`exploration/shared_map_endpoint.py` consumes the complete public episode through
the parent's last control. It reproduces the original detector-anchor track
history and permanent identity intersection, including the earlier counterexample
frames. Each of the ten original bootstrap records is retained. A separately
declared sixteen-consumed-frame fit revises training landmarks and past poses;
original withheld landmarks keep their original membership and are triangulated
from original past rays only. Neither new consumed holdouts nor future images fit
their own landmarks. All original geometry and validation gates remain unchanged.

The complete proposal freezes before any new control. The trial then executes:

1. WAIT for at least eight and at most sixteen one-tick controls. Declare optical
   stability at the first eligible prefix with three consecutive public image
   pairs satisfying the unchanged flow/support gates.
2. Validate and freeze the initial endpoint at that exact prefix. Later dropout
   cannot rewrite its evidence or select a different starting image.
3. Execute two one-tick controls with a public least-used opaque handle distinct
   from the parent's last handle, then apply the same WAIT rule again.
4. Validate the final endpoint using frozen training points and training-only
   future PnP. Require every originally supported model, the original directional/
   rotational consensus, and a displacement greater than 0.01 of each model's
   consumed trajectory radius. This last unscaled ratio rejects numerical motion;
   it is not a calibrated distance or an arrival threshold.

Whole-map tracking never resurrects a lost original landmark. Independent pair
namespaces measure optical flow without replacing frozen map identities. Losing
map support does not stop waiting early or rescue a passing subset. All actual
actions and partial failures remain recorded. The total episode retains its
200-action/201-image cap; the new trial adds at most 34 actions within that cap.
Tracking is explicitly bounded to 64 frames, not presented as long-horizon SLAM.

Optical stability is **not** assumed to mean a stationary player. The policy only
sees public RGB/action receipts; body/camera telemetry remains evaluator-only.
The independent auditor replays the entire public policy and compares all saved
evidence before reading private data. Only then does it measure camera and player
motion between the two exact frozen endpoints and report physical movement during
the last three publicly stable pairs. Those diagnostics cannot choose endpoints,
rescue candidates, override failures or promote navigation.

## Declared fresh-runtime entry point

```text
environments/minestudio/capture_runtime_camera.py --active-causal --settled-controls
  --recover-map --maintain-map --register-gauges --share-anchor --endpoint-controls
  --seed 202 --output runs/<new-unused-directory>
```

The `shared-endpoint-camera-projection-v1` registration freezes exact policies,
source snapshots and unchanged SDK/engine hashes before launch. The endpoint flag
requires the entire parent protocol. `scripts/audit_active_causal_map.py` verifies
the declared schema, canonical policy, action counts, public replay, raw bytes and
native projection evidence independently. Old captures remain historical and
must not be claimed as fresh results for this new source.

## Verification status

Both V28 offline core releases have completed thirty seeds/two repetitions with
eighteen exact artifacts per seed and all six verification checks passing:
`runs/core-release-seeds0-29-windows-v28/report.json` and
`runs/core-release-seeds0-29-wsl-v28/report.json`. These verify structured scientific
discovery and reproducibility, not Minecraft control or visual generalization.

The focused Windows regression passes 139 cases (29 endpoint, 24 shared-anchor,
86 runtime/auditor) in `runs/shared-endpoint-final-focused-windows-v1.xml`. This
includes frozen-map/initial-endpoint mutation, stationary numerical-motion refusal
and retention of completed WAIT actions after an endpoint solve fails. The isolated
default-only WSL installation passes 640 cases/72 optional or platform skips in
`runs/shared-endpoint-default-only-wsl-v1.xml`, without geometry/train dependencies.
These are engineering tests, not successful physical motion predictions.

Both package builds pass. The seed-zero MVP report at
`runs/shared-endpoint-regression-mvp-seed0/benchmark.json` stays byte-identical,
SHA-256 `B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
The full Windows regression passes 1197 cases/two skips in
`runs/shared-endpoint-full-windows-v1.xml`. The exact frozen V28 Linux tree completes
1199 passes/two skips, zero failures/errors in 752.149 seconds, recorded in
`runs/shared-endpoint-full-frozen-tree-wsl-v1.xml` (terminal process exit zero).
All three fresh runtime audits have completed as recorded below.
Three initial mounted-source WSL profiles terminate nonzero without JUnit and are
not passes. An exact 416-file native source/test/fixture snapshot and durable
verified archive now preserve the V28 bytes. All 416 frozen file hashes are
independently rechecked after the suite, without mismatches. This separate native
profile's pass does not replace the failed mounted-drive profiles or prove later
source revisions. See
`research/settled_keyframes/README.md` for provenance, thread/filesystem diagnostics
and the negative same-evidence optimizer probe.
Earlier focused XML results remain preserved separately. The initial synthetic
fixture run had twenty setup errors because its lateral path left the sensor's
field of view; the fixture's sensor was widened, without changing runtime images,
geometry gates or policy budgets. That failed run is not counted as a pass.

## First independently audited physical outcome

`runs/shared-endpoint-live-seed202-v1` completes 131 actions/132 images, all thirteen
capture checks and three outer/eighteen nested independent audit checks. The
earlier shared version is shadow validated; the new sixteen-view maintenance
proposal retains original supported bootstraps 10/11, both consumed-ray consistent.

The new trial executes 26 actual actions: ten WAITs to first eligible image 115,
two MOVE controls using opaque handle 1, and fourteen WAITs to image 131. Both
publicly selected stable tails have zero actual player translation/rotation in
all three evaluated one-tick intervals. Maximum returned-camera tail movement is
0.002213/0.000385 blocks, with maximum rotations 0.014405/0.002219 degrees. These
are evaluator diagnostics, not policy inputs or a general optical-rest theorem.

**Both frozen endpoint map predictions reject.** At the initial endpoint, original
withheld 95th-percentile errors are 5.976908/6.050534 pixels (43 tracks each); at the
final endpoint they are 9.001402/9.301208 (42 each), exceeding the unchanged three-
pixel gate despite adequate surviving tracks/cells. No subset, new bootstrap or
private timing correction rescues the version. The exact 115→131 interval moves
0.426892 camera blocks and 0.426891 player blocks; neither domain has a supported
prediction. This is an unsupported-motion failure, not correct navigation abstention.

The rest-controlled protocol therefore gives a useful timing boundary, not a
working map. Independent projection decomposition also finds essentially constant
focal length (fx 179.946635…179.946674; fy 319.905122…319.905181), so this particular
failure is not explained by changing focal length. Public calibration is not
replaced with these privileged values. Next investigate observable stable-keyframe
excitation/geometry rather than merely extending WAIT or lowering validation gates.

## Replication and no-map world

`runs/shared-endpoint-live-seed202-v2` independently passes the same thirteen
capture/three outer/eighteen nested audit checks. It selects the same exact
WAIT/MOVE/WAIT intervals, repeats both stationary player tails, and again rejects
all endpoint models. Initial errors are 5.976910/6.050535 pixels; final errors are
9.001324/9.301178. Camera/player displacement magnitudes again agree, but neither
domain has a supported estimate. Only 92 of 132 public image hashes match the
first capture; the native camera and player ledgers are not byte-identical. This
is a same-policy replication, not a paired identical-RGB comparison or a claim of
deterministic real-world acquisition.

`runs/shared-endpoint-live-seed201-v1` completes 86 actions/87 images and all the
same capture/audit checks. Recovery acquires no map and all 49 rolling candidates
across five attempts remain unavailable. There is no shared root, so the new
endpoint trial is explicitly `not_started`, has zero actions and invents no
endpoints, motion, rest diagnostics or evaluated success.

All three captures retain 115 source snapshots and match 113 current project
sources (two SDK sources are external). Their 268/268/178 artifact hashes and all
source/snapshot hashes are rechecked without mismatches. These are development
sessions, not broad held-out terrain/condition evidence. No physical direction
success or full navigation acceptance is claimed.

## Outstanding full-project requirements

Reliable public goal control and independently verified actual arrival remain
required. The project also still needs at least five additional disjoint worlds,
multiple terrains/three conditions with uncertainty estimates, integrated natural
scientific discovery and persistent map reuse, at least 100 frontier/experiment
attempts with restart/failure/recovery, and consolidated current comparisons and
release/CI evidence. See `docs/quality-roadmap.md`; this bounded experiment does not
replace or close any of those gates.
