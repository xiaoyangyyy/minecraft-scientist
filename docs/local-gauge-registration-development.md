# Public local-gauge registration development

This work addresses a necessary gap between accurate short-window image motion
and persistent navigation: successive monocular roots have unrelated detector
IDs, origins and scales. It does not equate a passing local link with a globally
registered cognitive map, metric localization or actual arrival.

## Frozen public contract

`exploration/map_gauge_bridge.py` accepts two immutable, past-only map snapshots,
their exact RGB hashes and distinct detector namespaces. Parent/child roots must
have chronological anchors, matching intrinsics/dimensions, all original bootstrap
records and track input hashes. Common absolute image slots must have identical
RGB hashes. Future prefixes/decisions and reused namespaces refuse. Every supported
map's first pose must be identity/origin; its actual bootstrap scale remains
explicit. Original snapshots are never rewritten.

At least three shared image slots are required. Correspondences use mutual nearest
pixels and consistent paths in every shared frame, not numerical ID equality.
At 224-pixel reference width, the first-view distance is at most 0.25 pixels, the
whole-path maximum at most 0.5 pixels, and the nearest-neighbor ambiguity margin
is greater than 0.1 pixels. Gates scale with image width. Ties and ambiguous paths
are rejected. Unmatched original tracks stay unmatched; there is no replenishment.

For each originally supported parent/child bootstrap pair, only associations in
both original training partitions fit a proper positive-scale similarity:

`X_parent = scale * rotation * X_child + translation`.

The closed-form SVD fit requires noncoincident/noncollinear spatial support.
At least twelve training pairs, eight pairs withheld in both original maps and
32 total pairs are required. The fitted transform freezes before the first
withheld 3D point is consumed. Neither cross-partition matches nor failed holdouts
can be recycled into training. Original per-map landmark and pose fits stay fixed.

Validation checks the training/withheld 95th-percentile 3D discrepancy against
0.1 of training spatial spread, both directions of withheld ray reprojection in
every common image, positive depth, at least 0.8 pixel acceptance and two cells.
The original 1.5-reference-pixel limit is unchanged. Mapped overlap poses must
agree within two degrees and 0.1 parent-bootstrap units. All supported bootstrap
pairs are evaluated and retained. One failing pair rejects the link; no privately
scored or passing runner-up can replace the declared first ordered reference pair.
Correlated initializations are not independent statistical samples.

`GaugeSimilarity` executes points, inverse points and world-to-camera poses.
Its scale relates local coordinate gauges, not world meters/blocks. Reflection,
nonpositive scale, improper rotations and nonfinite input refuse. Public result
flags keep global gauge alignment, metric scale and navigation false.

## Live chronological integration

`rolling_map_control.py` has an explicitly optional `gauge_policy`. Default
maintenance records and its earlier registered protocol remain unchanged. When
enabled, the second eligible root attempts registration against the preceding
root's original frozen past snapshot. The link journal, source namespaces and
evidence digest freeze before either second-epoch future action. Every finish
path verifies that later controls did not modify the link. Rejected links do not
rewrite the original accepted/rejected motion records or become navigation.

`capture_runtime_camera.py --active-causal --settled-controls --recover-map
--maintain-map --register-gauges` preregisters the separate
`registered-maintenance-camera-projection-v1` schema and exact policies before
launch. It adds no control action or larger budget: at most 200 actions/201 images.
The eleven capture checks distinguish protocol execution from a supported link.

The independent auditor regenerates all calibration, acquisition, settling,
recovery, maintenance and gauge records chronologically from public inputs before
opening camera/player truth. It compares the complete serialized checkpoint,
not just accepted flags. Forged links/counts, relaxed policies, old-schema disguise
and invented metric scale reject before private reads. Late input mutation drops
every score, including gauge diagnostics. Historical captures remain bound to
their old source snapshots and are not restamped as current-source captures.

## Privileged diagnostics remain evaluation-only

`evaluation/gauge_registration.py` scores every frozen candidate with separate
renderer and current-player pose truth. It uses each actual map bootstrap norm
and evaluator-measured baseline to assess relative scale, rotation and anchor
center. Ground-truth baselines below 0.15 blocks cannot calibrate monocular scale
and return unknown, not a correct score. Block-valued center errors are explicitly
privileged diagnostics, never exposed as agent-known metric localization.

Scoring preserves public rejection even when some candidate parameters look
good privately. It neither chooses a new reference pair nor promotes any link.
The earlier camera/body motion scores and their failures remain separate. A
nearly stationary root-anchor interval is not evidence of long-horizon route
accuracy, and a correct gauge diagnostic cannot substitute for actual arrival.

## Development and verification

The final pure/controller/evaluator focused profile passes 65 cases. Coverage
includes exact known transforms, independent namespace IDs, executable inverse/
poses, withheld failures without parameter changes, degenerate/reflected geometry,
ambiguous paths, complete competitor retention, invalid inputs and every control
checkpoint. Twelve public/native/auditor cases additionally verify unchanged old
maintenance, public-first empty-link replay, and five classes of gauge forgery.
Synthetic geometry and mock engines are not real-runtime link evidence.

An exploratory read of the historical seed-202 v1 public record establishes ten
shared frames and 137 track pairs. Its eight original bootstrap pairs have 46–61
training pairs and ten jointly withheld pairs. **All eight reject** under the
new preregistered policy. Their largest reciprocal 95th-percentile pixel errors
are about 3.076–13.216 pixels, against the unchanged three-pixel limit at width
448. Parent-unit pose-center errors reach 0.07099–0.31173; several training/withheld
point discrepancies also fail. This is development evidence, not an independent
current-source live pass or parameter-selection holdout.

## Fresh live evidence

The parameters freeze before capture. Fresh seed-202 v1 executes 105 actions/
106 images and passes eleven capture checks plus three outer/eighteen nested
independent audit checks. The link freezes between root anchors 90/92 before
the second root's future controls at 104/105. It retains ten shared images,
137 track pairs and all eight supported-bootstrap pair candidates, with the
declared reference `[10, 8]`. **No local link is accepted.**

All pairs fail reciprocal withheld-ray or overlap-pose checks. Their maximum
pixel errors are 3.075844–13.216127, against the fixed three-pixel limit. Their
maximum pose-center differences span 0.070992–0.311733 parent-bootstrap units;
training/withheld 3D discrepancies also reject several pairs. These failures
remain separate from two correct rendered-motion intervals: 101→103 has direction
error 2.595940 degrees and 103→105 has 1.106482. Current-player directions are
wrong at 33.967013/37.336144 degrees. The earlier stationary correct-abstention
and 96→100 unsupported-moving counterexample remain unchanged.

The private gauge diagnostic records all eight candidates. Seven violate the
registered pose/scale tolerances; `[11, 9]` is within them in both domains. Its
rendered relative-scale error is 0.016197, rotation error 0.509756 degrees and
privileged calibrated center error 0.030778 blocks. Nevertheless its public
withheld error 3.075844 exceeds three pixels, and its link stays rejected. It
does not become the declared reference or a private winning-map rescue. Root
anchors are nearly stationary; this diagnostic is not long-trajectory evidence.

The source and negative evidence are retained in `runs/gauge-bridge-live-seed202-v1`.
The second seed-202 session (`runs/gauge-bridge-live-seed202-v2`) completes 79
actions/80 images and passes the same eleven capture/three outer/eighteen nested
audit checks. Settling aborts after three WAITs, recovery finds a new stationary
anchor at 64 but rejects its future motion, and rolling maintenance rejects both
75→77 and 77→79 future motions. Its 12 bridge pairs include five public passes,
but seven failures reject the complete link; the original reference stays `[9, 7]`.
Ten rendered-camera parameter diagnostics and three body diagnostics are within
tolerance, without overriding public failures. Same-seed visual acquisition is
not asserted deterministic. This replication refutes robust consecutive local
motion, not just reliable gauge stitching.

Seed 201 (`runs/gauge-bridge-live-seed201-v1`) completes 86 actions/87 images,
passes the same audit checks, then exhausts all 49 rolling bootstrap fits across
five acquisition attempts. No frozen root/link or accepted future result is
invented. The independent auditor retains this negative evidence.

All three captures preserve 113 source snapshots and match all 111 current project
sources; two SDK snapshots are external. Their 178/216/164 artifact ledgers
recheck with zero mismatches. No source/record or rejection gate changes during
those jobs. The worlds/conditions are development seeds 201/202, clear/noon,
not held-out terrain/condition generalization.

## Engineering checkpoint

Both V26 scientific-core releases pass all thirty seeds/two repetitions/eighteen
exact artifacts and six checks, freezing 109 entries (106 package sources). All
current/snapshot hashes recheck with zero mismatches. The isolated default-only
WSL installation passes 640 cases/70 optional or platform skips without Torch,
Pillow or OpenCV. The full suites use separate verified-unused temporary roots,
avoiding the previous shared pytest cleanup hazard. Windows passes 1128 tests/two
skips and WSL 1130/two, recorded in `runs/gauge-bridge-full-windows-v1.xml` and
`runs/gauge-bridge-full-wsl-v1.xml`. These are source-frozen checkpoint results,
not evidence of broad runtime map/navigation success.

The original seed-zero MVP report remains byte-identical, SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Changed-file lint/format/whitespace checks and package builds pass. The built wheel
includes both new modules and all 106 Python sources. None of these engineering
checks promotes a rejected link or proves the broader project complete.

## Required next outcome and full scope

If fresh links again reject, do not enlarge thresholds or discard the losing
bootstrap pairs. Investigate training-only reconstruction with explicit shared
landmark/anchor constraints across overlapping windows, retaining the independent
local roots as counterexamples. Proposed replacement models must freeze before
new actions and pass independent withheld rays/new-control validation. A nearly
rigid-looking 3D cloud is not proof that separately optimized monocular depths
actually share one coordinate gauge.

This must lead to persistent public trajectories, observable control/endpoint
models and independently verified actual arrival, then integrated natural
scientific experiments and knowledge reuse. The original five additional disjoint
worlds/multiple terrains/three conditions with sufficient uncertainty, at least
100 frontier/experiment attempts with restarts/failures/recovery, current matched
symbol/planner/baseline comparisons and consolidated release gates remain open.
No local-link result or green unit suite proves overall project excellence.
