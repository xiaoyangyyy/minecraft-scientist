# First-image safe seeds and complete raw-bridge diagnosis

The new registered detector improves some public correspondence support, but both
fresh development worlds still fail to acquire a supported map. Complete raw-frame
tracking is not a general repair: it improves several endpoint pairs in seed 202,
but accumulates losses in seed 201's recovered local epoch. These are development
results, not held-out generalization, motion prediction, navigation or actual arrival.

## Explicit detector boundary

`StableSafeSeedKeyframePolicy` requires the existing micro/response/recovery flags
and the separate `--safe-seed-keyframes` flag. The capture schema is
`safe-seed-stable-keyframe-camera-projection-v1`; old protocols remain unchanged.
The first-image-only support mask is applied before corner detection, using the
original patch/window margins. Feature budget 800, quality 0.01, spacing, crop,
median filter and all subsequent tracking/geometry gates are unchanged.

This is a distinct seed order/namespace, not filtering old failures or changing an
existing survival denominator. Every flow records its first RGB hash and seed list.
Each local epoch freezes its first-image seeds; past fits and future extension must
use that same read-only array. A returned changed seed count is a geometry failure.
Proposals include a deep copy of the detector declaration before any future images.
The episode-wide 288 actions, eleven trials per epoch/twenty-two total, two local
epochs, two recovery windows, six-view minimum, fifty-percent survival, withheld-ray,
positive-depth and bootstrap-consensus requirements remain unchanged.

## Fresh native evidence

`runs/safe-seed-keyframe-live-registration-v1.json` declares two development seeds
before launch. Each capture has 119 executing source snapshots and seven passing
capture checks. Both independent audits replay the full public controller before
private camera/player scoring, with three outer/eighteen nested passing checks.

| Seed | Actions / images | Selected views per local epoch | Trials | Result |
|---|---:|---|---:|---|
| 201 | 275 / 276 | 6, then 3 | 9 + 11 | One rejected proposal; map unavailable |
| 202 | 185 / 186 | 4 | 11 | No proposal; map unavailable |

Audits at `runs/safe-seed-keyframe-live-seed{201,202}-v1/audit.json` are valid,
with 556/376 capture artifact hashes. Audit validity certifies replay/provenance,
**not** successful mapping. Neither has a frozen map, future experiment, accepted
motion, metric scale, connected global gauge or actual-arrival evidence.

Seed 201's first epoch selects frames 56, 76, 94, 104, 113 and 122. Its proposal
retains all four rejected bootstrap models: slots two/four lack parallax coverage,
slot three lacks positive-depth spatial support, and slot five fails joint withheld
past rays. It has 123 surviving past tracks from 249 frozen first-image seeds.
The failed WAIT window 132→148 remains recorded; extra WAITs 148→151 recover the
first eligible stable tail. The sealed parent ends there. The unconnected child
selects 151, 172 and 194, then exhausts eleven trials through 275. Nine images across
the two epochs are not nine connected keyframes. Seed 202 selects 56, 117, 173 and
185 without a recovery window. Neither reaches an accepted geometric proposal.

## Complete public raw-bridge diagnostic

`research/settled_keyframes/safe_bridge_probe.py` compares every actual original
selected-anchor/sample pair, including all rejected pairs. It checks an already
valid saved audit's attestation fields; evaluator files and private pose values are
not tracking inputs. Each direct diagnosis must exactly reproduce the saved seeds,
RGB hash, tracks, support, novelty decision and flow statistics. The continuous
comparison uses those same first-image seeds and every intervening RGB image;
it cannot skip bad images, redetect points or normalize losses away. Intervals
over 128 images are explicitly unavailable rather than shortened. All original
trials, epoch boundaries, selected frames, losses and namespaces are retained.
Source copies and hashes, and every read public input hash, accompany each report.

The reports are `runs/safe-seed-bridge-probe-seed{201,202}-v1/report.json`.
All twenty/eleven actual pairs replay exactly and all four preservation checks pass.
For seed 201's recovered epoch, raw survival on 194→203 improves from 77/245
(0.314286) to 101/245 (0.412245), still below the original 0.5 gate. Through frame
275 it falls to 55/245 (0.224490). None of the seven original post-194 pairs gains
novelty support. For seed 202, three originally unsupported pairs gain support and
novelty: 56→69 (0.492441→0.682505), 56→82 (0.393089→0.535637), and 117→149
(0.437346→0.626536). But 56→117 loses support (0.501080→0.419006).

This comparison does not replay a different controller: newly selecting frame 69
would change later anchors, response scores and potentially actions. It cannot be
counted as extra connected keyframes or a passing map. Neither saved capture nor
its failed outcome is replaced. Identical native RGB across runs is not claimed.

### Complete past-proposal geometry, separately checked after V33

`research/settled_keyframes/safe_raw_fit_probe.py` extends the diagnosis to every
existing rejected past proposal, not a selected passing subset. It verifies the
sealed proposal, original first-image detector and selected RGB hashes, then
replays sparse past tracks and all original bootstrap models exactly. It tracks
the complete raw prefix with the same frozen seeds and samples the already chosen
view slots without changing identities, original partitions, fitting budgets or
geometry gates. Public inputs and all 112 package-source files are hash bound.

`runs/safe-seed-raw-fit-seed201-v1/report.json` preserves the original six views
56/76/94/104/113/122 and every raw image 56→122. Raw projection retains 124 of
249 frozen seed IDs, versus the original sparse proposal's 123. Bootstrap five
now passes the original past withheld-ray/positive-depth checks: all six views
retain 31 withheld landmarks over five cells with full acceptance; maximum
per-view 95th-percentile ray error is 0.584192 actual pixels. Bootstraps two,
three and four still fail parallax or positive-depth support. Thus only one of
four models passes, below the unchanged requirement of two, and
`raw_past_consensus_candidate` remains false. This is not a unique/correct map,
future-motion result or navigation success. No evaluator pose is used for fitting.

`runs/safe-seed-raw-fit-seed202-v1/report.json` retains zero proposals explicitly;
none is invented. Both reports pass all four preservation checks and retain
source snapshots. Twenty-five research checks pass on Windows/Linux separately
from the earlier frozen V33 full suite. They cover all bootstrap slots, original
IDs/clock/failed middle images, seed-denominator and proposal tampering, no-proposal
captures, private-file guards and unchanged gates. The initial Windows research
run (twenty passes/five fixture-setup errors) is preserved; a removed pytest fixture
import was repaired, registered explicitly and verified in separate successful
reports. Production/package/test inputs of V33 are unchanged by this research.

This evidence supports investigating better controlled geometric view diversity,
not accepting the single passing model. The existing visual adapter already supports
opaque compound movement programs including cardinal movement plus jump; the current
capture exposes only the four plain cardinal programs. A fresh explicitly registered
compound-control experiment can test whether an agent can escape confined terrain
and obtain useful views, without new action kinds, privileged action selection,
changing the SDK or relaxing any geometric acceptance gate.

## Verification and remaining work

Focused Windows/WSL tests pass 146 each (Windows groups 92 and 54); isolated
default-only Linux passes 673/76 skips, zero failures/errors. Twelve safe-seed
tests cover exact research-detector equivalence, original support bounds, failed
future identities, read-only past/future seed sharing, denominator guards and
unchanged gates. Twelve new audit/CLI cases cover positive replay, tamper rejection
before private scoring and missing registration flags. Eighteen additional research
tests pass on each platform, separately from the frozen full suite. They cover all
actual epoch/sample pairs, immutable IDs, failed-middle rejection, budget overflow,
ledger/seed/selection tampering, input mutation and refusing existing output paths.

V33 Windows/WSL core releases each pass thirty seeds/two repeats/all eighteen exact
artifacts/all six checks. Both 115-entry source ledgers (112 package files) match
current bytes. Mean experiments remain 53.933333 and minimum equation-transfer
savings one. Seed-zero MVP bytes retain SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
The package builds with no new runtime dependencies.

Full V33 persistent Linux completes 1349 passes/two skips, zero failures/errors;
all 444 frozen file hashes match after terminal completion. See
`runs/safe-seed-keyframe-full-persistent-tree-wsl-v1.xml` and
`runs/v33-full-linux-post-verification-v1.json`. Full Windows also completes:
1347 passes/two skips, zero failures/errors, with all 444 declared snapshot inputs
and its 115-entry core ledger unchanged after terminal completion. See
`runs/safe-seed-keyframe-full-windows-v1.xml` and
`runs/v33-full-windows-post-verification-v1.json`. The exact
444-file/4,851,791-byte snapshot at
`runs/v33-frozen-test-tree-v1` has archive SHA-256
`61E0297CD15DC83CF3759D70ABEF8FC43B758E54FFEA0C9F8E26EEE3967DE505`.
Its verified persistent restoration is `runs/v33-persistent-frozen-test-tree-v1`.
The frozen tree predates this document and the separately tested bridge research.
Earlier V32 counts are historical, not verification of these production edits.

Next investigate controllable, sufficiently diverse geometric views and whether
past-only raw correspondence improves full proposals. Merely swapping endpoint
tracking cannot repair every failed epoch. Any changed controls or tracking clock
need explicit new registration, fresh complete replay and independent future
camera/player validation; no lower thresholds, passing-subset rescue or privileged
action choices. Reliable actual arrival, five disjoint held-out worlds/multiple
terrains/three conditions, persistent map/scientific-loop integration, hundred-attempt
restart/failure recovery, current baseline/ablation comparisons and consolidated
release gates remain open in `docs/quality-roadmap.md`. The full goal stays active.
