# Public-support-loss epoch candidate, separate from main V35

This protocol remains an isolated research candidate. Main contains the registered
longer-WAIT protocol, not this epoch change. Windows and persistent Linux full V35
regressions continue against their unchanged 456 declared inputs. This new document
and `research/settled_keyframes/support_loss_candidate.patch` are outside that
manifest, and do not change its source/test inputs or justify new full pass counts.

The six-file candidate patch is preserved against main `3defbb9` and passes
`git apply --check --whitespace=error`. Executing candidate sources remain at
`runs/development-trees/verified-source-txf4f4p8`; the original restoration proof
is `runs/support-loss-development-tree-v1/report.json`. That proof covers the
initial V35 restoration, not subsequent intentionally changed candidate bytes.
Native and core runs separately freeze those changed executing bytes.

## Why this experiment

The preceding longer-WAIT runs complete all eleven stable acquisition trials,
with zero extra recovery windows, but lose public correspondence support on
four/ten trials. Original local-epoch renewal requires an extra WAIT recovery,
so neither run can renew its map after an otherwise compliant stable action.
This experiment separates public map observability from that control-time event.
It does not weaken correspondence support or assume stable pixels mean zero
player/camera motion.

`StableSupportLossKeyframePolicy` registers
`public-support-loss-after-stable-control-v1`. It seals the parent after the first
completed stable acquisition with fewer than the original required 32 tracks,
four cells or 0.5 first-image survival. A new local anchor uses the actually consumed
current image, freezes its own first-image safe seeds, and resets *local* public
control scores. The lost sample stays negative in the sealed parent. Global trial
IDs, the consecutive-control bound and every action receipt remain unchanged.
Gauges are explicitly unconnected; changing namespace is not a loop closure.

The protocol still allows only two local epochs, eleven acquisition trials per
epoch, twenty-two total and 288 actual episode actions. All compound motors,
16/24-frame phase WAIT windows, first-eligible stopping, two episode-wide extra
windows, detector/tracking/fitting/depth/withheld/consensus/future gates remain.
Supported stalls, partial or failed/noncompliant controls and unstable initial
frames cannot trigger renewal. Future validation cannot re-anchor or refit a
frozen past. `--support-loss-keyframes` requires all seven parent flags and uses
its own `support-loss-long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1`
schema. Complete independent public replay precedes any private scoring.

## Tests, not native mapping proof

Fourteen new controller checks pass on Windows and Linux. Mock perception with
real synthetic three-dimensional fitting verifies the distinct parent/new-mode
semantics, sealed parent hash, retained negative, new detector/hash/namespace,
scoped scoring history, future freeze and unchanged global consecutive bound.
Other cases verify no third epoch, eleven child trials, supported stalls not
renewing, failed controls not becoming evidence, initial exhaustion, future WAIT
failure not refitting, an actual partial two-epoch episode ending at action 288,
strict booleans, explicit flag chain, exact budgets and no parent disguise.

One hundred nineteen auditor/CLI checks pass on each platform: all preceding
registered protocols and twenty-seven additional cases exercise public-first
replay and reject forged boundary/schema/motor/detector/phase budgets/lineage
before private access. Existing motor/long-WAIT/seed/adapter/map/recovery/endpoint
compatibility checks pass 144 each. Total focused candidate tests are **277 per
platform**, zero failures/errors; all six changed Python files pass lint.

Both isolated candidate core releases at
`runs/core-release-seeds0-29-{windows,wsl}-support-loss-v1` pass thirty seeds,
two repetitions, eighteen exact artifacts and six checks. They freeze 116 entries
(113 package sources), preserve sources through execution, and retain the same
53.933333 mean reported experiments/minimum one equation-transfer saving. These
prove offline core compatibility and repeatability, not actual Minecraft mapping,
nor completion of the separately running main V35 full suites.

## Fresh native evidence

`runs/public-support-loss-live-registration-v1.json` fixes both development seeds,
source separation, first public loss trigger and all budgets before launch. Each
fresh run preserves 120 source snapshots (118 project entries/two SDK sources),
seven passing capture checks, three passing outer/eighteen nested independent
audit checks. All 118 project hashes still match the executing candidate copy
after capture and replay. The recorded 582/578 artifact hashes match. Both audits
are valid; neither evaluation has a supported map/future-motion score or navigation.

| Seed | Actions / images | Parent selected frames / completed trials | Child selected frames / completed trials | Terminal |
|---|---:|---|---|---|
| 201 | 288 / 289 | 56, 89 / 4 | 123 / 8, then one partial acquisition | budget_exhausted |
| 202 | 286 / 287 | 56 / 1 | 72, 127 / 11 | map_unavailable |

Both retain exactly one unsupported parent→child link and zero failed/extra WAIT
windows. Parent seals occur at frames 123/72 after the actual public support loss.
Each run has three selected images *across two unconnected epochs*, not a validated
three-view map. Neither creates a past proposal, frozen map or future validation.
Seed 201's final compliant MOVE/WAIT prefix is kept up to frame 288; it is not an
invented thirteenth completed trial. In its child all eight completed endpoints
fail support, despite 15–21-frame stable waits. Seed 202's child retains one novel
view at 127, then loses support on every subsequent endpoint.

## What this establishes and what it does not

The public epoch transition works independently of WAIT recovery and respects
the declared failure/provenance budgets. It does **not** establish more useful
native geometry, a connected cognitive map, reliable future motion or arrival.
These are repeated development worlds, not held-outs, native-byte pairs or a
statistical comparison. Original longer-WAIT and earlier failures remain intact.

Next diagnose correspondence identity loss and controllable view diversity on
the complete *public* image/action prefixes, including the unsupported child
endpoints, before preregistering another intervention. Distinguish feature loss,
supported stalls, global cost and local observability. Any alternative detector,
tracking clock, motor program or epoch budget needs its own declared protocol,
unchanged honest geometry/future acceptance and fresh independent validation;
known-image gains cannot be promoted to online success. Further waiting alone or
counting disconnected anchors cannot satisfy the original project requirements.

Reliable independently verified arrival, five disjoint held-out worlds with
multiple terrains/three conditions, scientific/map integration, hundred-attempt
restart/injected-failure recovery, current baseline/ablation comparisons and
consolidated release/hosted CI remain required. The full goal stays active.
