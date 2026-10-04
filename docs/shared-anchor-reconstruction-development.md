# Shared-anchor reconstruction development

This checkpoint replaces an attempted rigid alignment with a genuinely new
training-only shared-landmark model. It does not prove a persistent global map,
player-motion recovery, goal arrival or overall project completion. The preceding
gauge checkpoint is committed as `32286e6`; its original captures and failures
remain unchanged and are historical after this package-source addition.

## Model and evidence contract

`exploration/shared_anchor_map.py` consumes an immutable original past snapshot
and a same-detector-anchor extension consisting only of already consumed RGB.
All original bootstrap records are required. Original frame/pixel/track hashes,
seed identity, past survival counts and origin/unit-bootstrap gauge must match.
Both detector reinitialization and validation-to-training repartitioning refuse.
An original missing identity can only be lost, never replenished.

Each originally supported model initializes its new consumed-frame poses using
original training landmarks and training-only PnP. Bundle adjustment then jointly
revises those landmarks and past/consumed poses. Frame zero remains the original
identity/origin and the original bootstrap remains a unit gauge. The earlier
model/trajectory is not edited; this is a proposed replacement version.

Only after training optimization freezes are surviving original withheld points
retriangulated. They use **original past rays only**, not consumed extension rays.
Every original and consumed frame undergoes the unchanged withheld-ray check,
including the three-pixel limit at width 448. Extension holdouts therefore cannot
fit either training poses or their own 3D points. This is independent of training
but not an unused world/condition generalization test.

`SharedAnchorWindow` freezes the complete new model before two subsequent controls.
Future PnP uses surviving original training landmarks; neither optimization nor
point triangulation runs on future pixels. Prefix intersection prevents a lost
identity from returning. Every originally supported candidate must pass the new
past and subsequent endpoint validation, with the original motion-consensus gates.
One failure rejects the version; a passing subset cannot rescue it. Unsupported
originals remain explicitly unavailable, not new bootstraps.

This bounded proposal has at most 14 consumed fitting frames and two subsequent
frames. It does not silently drop its anchor, change partitions or keep accepting
frames indefinitely. Extending it further requires a separately registered crisis/
maintenance protocol, not renaming a bounded window as long-horizon mapping.

## Live integration and independent replay

Run the new declared protocol with:

```text
environments/minestudio/capture_runtime_camera.py --active-causal --settled-controls
  --recover-map --maintain-map --register-gauges --share-anchor --seed 202
  --output runs/<new-unused-directory>
```

The capture declares `registered-shared-anchor-camera-projection-v1`, exact policy
and source snapshots before launch. The flag requires the entire parent protocol.
At the second rolling root, it reconstructs from the first root's original anchor
over consumed images, intersecting every preceding future track prefix. It freezes
the new version/hash before the second root's controls, without adding any action
or raising the 200-action/201-image cap. The old independent maps and gauge failure
stay in the record even if a new shared version passes.

The version journal records immutable original lineage, frozen past models and
RGB hashes, chronological future prefixes, decision and terminal descendant.
Its status changes from `proposed` to `shadow_validated` or `rejected`; a history
budget exhaustion rejects without inventing a fitted model. Every finish path
checks that future control callbacks did not mutate the frozen version.

`audit_active_causal_map.py` re-executes the complete public policy and compares
the entire journal before private camera/player reads. Relaxed policies, invented
versions/counts/lineage, old-schema disguise and metric claims reject first.
Private scoring is separate for returned-image camera and current player; invalid
or mutated captures lose every score. The shared result never rewrites an earlier
motion or gauge score. All navigation/global-gauge/metric flags remain false.

## Verified first live outcome

`runs/shared-anchor-live-seed202-v1` completes 79 actions/80 images, twelve capture
checks and three outer/eighteen nested independent audit checks. Original anchor
64 is retained; twelve original frames expand to fourteen consumed frames ending
at 77. All ten original bootstrap records persist. Original supported bootstraps
9/10/11 all pass the new consumed-past withheld check, before controls 78/79.

Their consumed maximum 95th-percentile withheld errors are respectively
2.604054/2.864906/2.840742 pixels. Training costs decrease from
108.007044→94.583598, 135.678860→108.794854 and 138.387404→113.555803. Training
landmark counts are 79/127/126 and each keeps 43 original withheld landmarks.
These are internal/consumed-ray checks, not successful prediction of a new control.

**The new-control shadow validation rejects all three models.** At frame 79,
42 withheld landmarks survive; their 95th-percentile errors are
8.563470/9.994101/9.530533 pixels, well above three. Acceptance fractions are
0.785714/0.761905/0.761905. Training inliers remain 54/92/91 and validation cells
5/4/4, so this is not an absence of minimum surviving track support. The decision
is rejected and no shared version is promoted. Independent evaluation observes
actual 77→79 translation in both domains: 0.288031 camera blocks and 0.280976
player blocks. The unsupported-motion counterexample remains a failure.

This does not show that joint reoptimization solves the original ambiguity: it
shows that consumed-view consistency still fails to transfer to a different opaque
control. It motivates a separately frozen crisis update using these counterexample
training rays, followed by a **new** control holdout. Previously failed validation
points must not become training points or disappear from the failure ledger.
The current bounds do not implement that additional crisis stage yet.

The scene is a development seed with clear/noon settings, not a held-out terrain
or condition. Only nine public frame hashes match the preceding same-seed gauge
capture; neither the native camera nor player ledgers are byte-identical. Do not
claim paired identical-RGB runtime superiority or deterministic visual acquisition.

## Regression checkpoint

The shared geometry/controller plus existing gauge/rolling/runtime focused profile
passes 65 cases on Windows. Twenty-four new geometry/integration cases cover
shared-origin recovery, consumed/future holdout noninterference, immutable roots,
every candidate, lost identities, unchanged action budgets and mutation refusal.
Eight new native/public/auditor cases cover empty-version replay, six forgery
classes and CLI parent-protocol dependencies. Synthetic geometry and fake engines
are explicitly not real-runtime map/navigation success.

Both V27 core releases pass thirty seeds/two repetitions/eighteen exact artifacts
and all six checks, freezing 110 entries (107 package sources). Current/source-
snapshot hashes match. The isolated default-only WSL installation passes 640
cases/71 optional or platform skips, without Torch/Pillow/OpenCV. Wheel/sdist builds
pass; seed-zero MVP `benchmark.json` stays byte-identical to the preceding release,
SHA-256 `B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.

The full Windows regression passes 1160 tests/two skips and WSL passes 1162/two,
in `runs/shared-anchor-full-windows-v1.xml` and `runs/shared-anchor-full-wsl-v1.xml`.
Separate verified-unused temporary roots avoid cleanup races. Source changes stay
frozen through every terminal process and independent audit. Changed-file lint/
format/whitespace checks pass. No green engineering gate promotes a rejected
shared map or proves overall completion.

## Same-policy replication and retained no-map world

`runs/shared-anchor-live-seed202-v2` completes 105 actions/106 images and passes
the same twelve capture/three outer/eighteen nested independent audit checks.
The first rolling root freezes at anchor 90; the shared proposal uses its twelve
original frames plus two consumed frames, then freezes before frames 104/105.
All ten original bootstrap records remain; both original supported bootstraps
10/11 are required and pass. The two fixed future withheld errors are
2.580662/2.646859 pixels, with motion direction/rotation disagreements of
0.536258/0.073818 degrees. The proposal becomes `shadow_validated`.

Independent evaluation of its new 103→105 control interval gives rendered-camera
direction/rotation errors of 0.645666/0.126229 degrees: correct under the registered
tolerances. The simultaneous player direction is wrong at 37.316404 degrees.
The independent local maps retain their own motion scores (2.835630/1.106455
camera directions and 33.690756/37.336025 player directions); the rejected rigid
gauge link and all its candidates remain unchanged. This is a bounded shared-anchor
camera-model success, not reliable player control, metric localization or global
navigation. One success and one failure do not establish robust generalization.

`runs/shared-anchor-live-seed201-v1` completes 86 actions/87 images and the same
capture/audit checks. Recovery cannot acquire a map and all 49 rolling fits across
five attempts fail. The shared-version journal is empty, not an invented success.
All three captures preserve 114 source snapshots, match 112 current project
sources (two SDK snapshots are external), and recheck 178/164/216 artifact hashes
without mismatches. Saved active auditor snapshots match both their report hashes
and current source; the nested runtime auditor source hash also matches.

## Current limited structured ablation

Two identical WSL runs of `benchmark-ablations` use seeds 0–9 and cap 80 in
`runs/shared-anchor-ablations-seeds0-9-wsl-v1` and `-v2`. Their compact reports are
byte-identical, SHA-256
`99460B0D9CAF2279266AF4BDEA43A4721E5308A3E9AFC6E43B69A7D22726A026`.
The unchanged planner checkpoint hash is
`CEBB08E5D939AAEF813B2670D1A615CBE2686151136CF79C05CCD594A9B51CB4`.
All four integrity flags pass. Source identity is the concurrently frozen V27
checkpoint; these commands do not independently snapshot a separate source ledger.

The learned-symbol transfer saves eight experiments per seed and description
length improves by thirteen units. All ten full crises complete versus zero
without repair; post-change accuracy gain is 0.148148. Mean information-gain
savings versus informative random are 0.4 (normal approximate paired 95% interval
−4.340157…5.140157); learned-planner savings versus information gain are 0.7
(−2.026052…3.426052). Both include zero, so neither efficiency-superiority claim
is supported by this sample. Constant effects in this finite protocol family do
not imply zero uncertainty in wider worlds.

The initial Windows learned-ablation command fails because optional `gymnasium`
is absent; it produces no report and is not counted as a pass. The existing WSL
training installation supplies the dependencies, without altering the Windows or
default-only test environments. This comparison is limited to the structured
three-mechanism/cross-world-transform suite, stores compact summaries rather than
complete per-condition traces, and has no fixed-library control. It does not close
the full current four-form/live planner-symbol-baseline release requirement.

## Original scope still open

The next stage must produce reliable public trajectories/control predictions and
independently verified goal arrival, then integrate natural scientific experiments,
theory revision and persistent knowledge reuse. Five additional disjoint worlds,
multiple terrains/three conditions with sufficient uncertainty, at least 100
frontier/experiment attempts with restarts/failure/recovery, matched current
baseline/planner/symbol comparisons and consolidated release/CI evidence are still
required by `docs/quality-roadmap.md`. This bounded module does not substitute for
any of those requirements.
