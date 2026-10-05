# Stable, novel sampled-keyframe acquisition

This checkpoint changes acquisition before fitting, not the acceptance threshold.
Two fresh, independently audited development worlds acquire too few novel views
and therefore no map. It does not establish reliable goal navigation, actual
arrival, persistent global mapping or full-project completion.

## Protocol and evidence boundary

`exploration/stable_keyframe_map.py` uses only public RGB, public camera calibration
and actual one-tick control receipts. The existing 48-action rotation calibration
is followed by the first eligible optically stable prefix: WAIT at least eight,
at most sixteen ticks; require three consecutive pair-local stable observations,
32 tracks, four occupied cells, 50% survival, median flow at most 0.2 and 95th-
percentile flow at most 0.5 pixels. These are the preceding endpoint gates.

Each acquisition trial executes two opaque MOVE controls followed by the same
first-eligible WAIT. Novelty is tested relative to the last selected keyframe:
at least the same support/survival/cell gates and two-pixel median displacement.
Stalled/unsupported views remain recorded, not silently treated as motion or
added to the map. After four same-handle trials, or an unsupported novelty sample,
choose a least-used opaque handle; no private physical heading chooses it.

The first past proposal with at least six selected views and two supported
bootstrap models freezes. All bootstrap candidates from slots two through the
last selected view remain in each attempt, including rejected candidates. The
existing 30-iteration training-only bundle adjustment and three-pixel withheld
gate are unchanged. A distinct least-used control plus the first public stable
future then tests all supported frozen models; no passing-subset rescue, refit,
privileged endpoint choice or metric scale is allowed. Rotation/direction
consensus and nonzero relative translation are necessary, not arrival proof.

`KeyframeIndex` records selected slots' actual image ordinals and raw RGB hashes.
Tracking compares chronologically selected images, not every raw one-tick image;
its explicit sparse namespace **does not claim continuous raw-track identity**.
The complete action/render ledger still preserves every actual intervening tick.
At most eleven acquisition trials/twelve selected views are considered. A new
separate registration permits 288 total episode actions; older capture/audit
registrations keep their 200-action limit.

## Fresh real outcomes

| Development seed | Actual actions / images | Selected stable keyframes | Fitting attempts | Map / future prediction |
|---|---:|---:|---:|---|
| 202 | 203 / 204 | 1 | 0 | Unavailable |
| 201 | 176 / 177 | 2 | 0 | Unavailable |

Sources and evidence are respectively at `runs/stable-keyframe-live-seed202-v1`
and `runs/stable-keyframe-live-seed201-v1`. Both pass all seven capture checks,
three outer and eighteen nested independent audit checks. Their 116 source
snapshots (114 project, two SDK) and 412/358 artifact hashes are independently
rechecked without mismatches. The saved auditor snapshots match current source.

The seed-202 public stable tails have zero player translation, with maximum camera
tail movement 0.001831 blocks. Seed 201 has up to 0.004014 player and 0.009635 camera
blocks within a stable tail. Thus optical stability is not asserted as exact body
rest, even here. These evaluator-only diagnostics cannot select/rescue views.

Seed 202 preserves eleven rejected novelty samples. Direct endpoint matching
retains 26.2%–45.6% of the original detector features: sufficient absolute counts
and displacement but below the unchanged 50% survival gate. Seed 201 selects only
frames 56/101 and likewise fails to gather six views. Neither starts fitting or
invents a successful new-control prediction. This is failed acquisition, not
successful abstention on a stationary target or evidence of better mapping.

The first seed-202 auditor reaches nonempty stability data but fails JSON
serialization of evaluator NumPy arrays. Its original `audit.source.py` and
`audit-serialization-failure.json` remain. An explicit finite/list serializer fixes
the reporting error; `audit-v2.json` is the separate completed audit. No capture,
policy or failed geometry is restamped.

## Reproduction

Use unused output directories in the existing Linux visual runtime:

```text
python environments/minestudio/capture_runtime_camera.py --stable-keyframes
  --seed 202 --output runs/<new-capture>
python scripts/audit_stable_keyframes.py runs/<new-capture>
  --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar
  --output runs/<new-capture>/audit.json
```

This flag is exclusive of the preceding parent/maintenance/endpoint protocol.
The auditor independently reproduces public calibration, every requested control,
first-eligible selection, sample clocks, all attempted models and future decisions
before opening private poses. Tampering, old-schema disguises, altered gates,
invented motion/metric knowledge or late mutation reject and discard scores.

## Engineering verification

Current V29 core releases on Windows/WSL each pass thirty seeds/two repetitions
and all eighteen exact artifacts/all six verification checks. Their 112-entry
ledgers include 109 package Python sources, independently rechecked unchanged.
The seed-zero CLI MVP report retains its earlier SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Source/wheel package builds pass without adding runtime dependencies.

Fifteen keyframe unit tests and thirteen audit tests pass on Windows. The earlier
combined keyframe/runtime profile passes 101 cases; one initial auditor fixture
failure (exception instead of invalid result) remains preserved. The WSL focused
profile passes 34 cases, including seven separate point-refinement research tests.
One earlier profile has five setup errors after its temporary tree disappears;
that XML is not counted as a pass. The later serialization regression is also
included in the frozen full suite.

The default-only Linux installation verifies the exact persistent snapshot:
640 passes/74 optional or platform skips in
`runs/stable-keyframe-default-only-wsl-v2.xml`. It requires no geometry, training,
Minecraft or API provider. Full Windows regression completes 1225 passes/two
skips, zero failures/errors in 1177.34 seconds at
`runs/stable-keyframe-full-windows-v1.xml`. The persistent-snapshot Linux full
suite now completes 1227 passes/two skips, zero failures/errors in 918.562 seconds
at `runs/stable-keyframe-full-persistent-tree-wsl-v2.xml`. All 423 frozen file
hashes were independently rechecked after terminal completion: zero mismatches.
These are V29 results, not regression proof for subsequent micro-control edits.

A 423-file/4,665,376-byte snapshot matches every source/test/checkpoint byte and
V29 ledger entry. Its verified archive is at `runs/v29-frozen-test-tree-v1`, SHA-256
`533D8C163635CEBF7AB74FD5A1A6AF780499A4991F7CE2A08D993D988FECC957`.
The first full Linux run loses its `/tmp` source and test trees during execution
and terminates nonzero without JUnit; earlier default-only launch cannot enter
that missing tree. Startup tmpfiles logs correlate with both disappearances.
All failures remain in `runs/stable-keyframe-runtime-failures-v1.json`.

The archive is restored byte-identically into a fresh task-specific persistent
directory under `/opt/minecraft-scientist-verification`, with no host cleanup
policy changes, overwrite or deletion. Separate declared runtime profiles and
unused test directories are retained. A native snapshot or recovered archive is
provenance, not a green test or successful visual experiment.

## Next scientific action

Known-data diagnostics in `research/settled_keyframes` compare direct endpoint
matching with complete intermediate-frame tracking. In seed 202, interval 56→67
improves survival from 33.3% to 45.4%, still insufficient. Some pair-local intervals
cross 50%, but anchor survival decays; do not stitch different pair namespaces or
infer a valid global map. Seed 201 exhibits similar local improvements but does
not prove six-view map acquisition. The diagnostic does not skip failures, expand
the 128-frame tracking bound, fit a map or use private poses.

Next investigate smaller preregistered control increments and identity-preserving
intermediate tracking with explicit clock/namespace contracts. Keep all failed
controls/candidates and collect fresh subsequent-control evidence before promotion.
Also harden partial acquisition reporting around native numerical fitter errors.
Every original gate in `docs/quality-roadmap.md` remains required, including
five disjoint evaluation seeds/multiple terrains/three conditions, persistent map
and natural scientific-loop integration, long operation/recovery and full release
comparisons. These two development failures do not satisfy those requirements.
