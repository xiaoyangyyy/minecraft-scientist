# Bounded WAIT recovery and sealed local epochs

The first fresh development world recovers an exhausted public stability window
after three extra WAITs and opens a new unconnected local-map namespace. It still
cannot acquire a supported map. The second world does not trigger recovery and
also has no map. This is verified landing/stability recovery, not reliable mapping,
goal navigation, actual arrival, generalization or full-project completion.

## Explicit protocol and unchanged geometry

`StableRecoveringKeyframePolicy` requires explicit micro, response and recovery
flags. Its capture schema is `recovering-stable-keyframe-camera-projection-v1`.
The registered protocol permits two extra sixteen-tick WAIT windows **per episode**,
shared across initial, acquisition and future phases. Exhausted windows, their
original terminal reasons and partial failed controls remain in separate ledgers.
Waiting stops at the first eligible stable tail, including tails spanning a window
boundary; the original minimum eight WAITs/three stable pairs remain unchanged.

Only a recovered acquisition endpoint that fails original correspondence support
may open a second local epoch, before any map freeze. The parent retains all trials,
rejected proposals, namespaces and a sealed hash. The child anchors its own detector
namespace at the recovered public endpoint; the connection is explicitly unsupported.
Neither coordinate gauges nor landmark identities are silently stitched. Mere
stall, unrecovered correspondence loss and future validation cannot reset an epoch.

Response scores use completed trials from the active epoch, recording their original
global trial IDs. Actual consecutive-control counts remain episode-wide; restarting
a namespace cannot bypass the four-trial same-control cap. Future control is selected
from past responses only, after freezing and before dispatch, excluding the last
acquisition control. Future recovery cannot refit or replace the frozen map.

The new registration permits eleven trials **per epoch**, at most twenty-two total,
not the preceding protocol's eleven episode-wide trials. The global 288-action cap
includes the original 48 calibration actions and is unchanged. All tracking, novelty,
positive-depth, withheld-ray and consensus gates remain unchanged; no partial fit or
passing subset rescues a rejected model. Base/micro/response protocols retain their
original successful output shapes and separate registrations.

## Fresh native results

Seeds, budgets and output directories were declared before launch in
`runs/recovering-keyframe-live-registration-v1.json`. Each capture has 118 executing
source snapshots, seven passing capture checks, and a source-bound independent
audit with three outer/eighteen nested checks. Replay reconstructs the complete
public controller and epoch ledger before private camera/player scoring.

| Seed | Actions / images | Selected views per epoch | Completed trials | Result |
|---|---:|---|---:|---|
| 201 | 279 / 280 | 6, then 1 new anchor | 9 + 11 | Recovered; map unavailable |
| 202 | 194 / 195 | 1 | 11 | No recovery needed; map unavailable |

Both `runs/recovering-keyframe-live-seed{201,202}-v1/audit.json` are valid, with
564/394 artifact hashes verified. Neither has a frozen map, future experiment,
accepted motion, metric scale, global gauge or actual-arrival proof. Seven selected
images across seed 201's two epochs are **not** seven connected keyframes.

Seed 201's old namespace retains frames 56, 75, 93, 103, 112 and 121 and all four
rejected bootstrap models. The failed window 131→147 remains exhausted. Extra WAITs
147→150 supply the first eligible three stable pairs, then endpoint support against
frame 121 has zero surviving tracks. The sealed parent ends at 150; the child starts
there and independently consumes eleven more trials through 279. All remain rejected
by the original fifty-percent survival gate. The parent hash remains unchanged.
The same selected ordinals as an earlier capture do not imply native-byte identity:
the fresh past-proposal hashes differ and no paired-byte/statistical claim is made.

After successful replay, evaluator-only player positions show the five-block fall
131→147 and zero displacement in each of the three recovered tail pairs. Maximum
rendered-camera displacement in those pairs is 0.000574699 blocks. These diagnostics
never entered the stopping rule. Subsequent player positions change, so failed
acquisition is not explained solely by nonexecuted movement; camera/body evidence
remains separate and does not certify navigation.

## Public correspondence diagnosis and research-only candidate

`research/settled_keyframes/recovered_epoch_probe.py` preserves the exact original
direct tracks and diagnoses every recovered-epoch sample, without private reads,
new controls, gate changes or replacing the capture. Its original-source report is
`runs/recovering-keyframe-epoch-loss-seed201-v1`. The anchor detects 304 seeds, of
which 55 immediately fail first-image ROI support. Endpoint survivors range 136–147;
cycle and patch-correlation failures account for substantial additional loss.
Continuous tracking improves 150→162 to 0.542763 survival, but anchor-wide survival
declines to 0.286184 by 268. The full 150→279 raw interval has 130 images and is
explicitly unavailable under the unchanged 128-image tracking budget, not truncated.

The separate `safe_seed_detector.py` research candidate detects corners only inside
the first image's already-required patch/window support mask. It does not read a
future image, filter original identities by eventual success, change a survival
denominator after detection or relax tracking gates. Its seed order/namespace is
explicitly different. On the eleven **known, original-anchor** image pairs it detects
256 seeds and retains 141–153 (0.550781–0.597656); all meet the original support gate,
eight also meet the two-pixel novelty gate. Saved at
`runs/recovering-keyframe-safe-seed-probe-seed201-v1`, this is not a sequential
keyframe-selection replay, fitted map, new live controller or independent holdout.
The candidate has not been promoted into the production detector. Old failures
remain failures. It must next pass a new registered full controller/live evaluation.

## Verification and remaining gates

Both platforms pass 126 focused cases: thirteen recovering-controller tests,
eight WAIT-helper tests, twenty-five response-selector tests, thirty-eight existing
acquisition tests and forty-two replay/CLI tests. Coverage includes the first eligible
tail, preserved failed/partial windows, a real synthetic-geometry fit in a new epoch,
scoped scoring, global consecutive-action limits, two-epoch/local-trial caps, total
budget exhaustion, frozen future immutability and pre-private tamper rejection.
The initial Windows controller run retains one failed overly-specific error-message
assertion/56 passes; the corrected verification is separate, not restamped.

Windows/WSL V32 core releases each pass thirty seeds/two repeats/all eighteen exact
artifacts/all six checks, with 114 unchanged source-ledger entries (111 package files).
Mean experiments remain 53.933333 and minimum equation transfer savings one. Seed-zero
MVP report bytes retain SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
The package builds without new runtime dependencies. Default-only persistent Linux
passes 673/75 optional or platform skips, zero failures/errors. The research-only
diagnostic/detector checks pass twenty on each platform, separately from the full suite.

Full V32 persistent Linux regression is terminal: 1325 passes/two skips, zero
failures/errors, with all 437 frozen file hashes unchanged after completion.
The JUnit and post-terminal verification are
`runs/recovering-keyframe-full-persistent-tree-wsl-v1.xml` and
`runs/v32-full-linux-post-verification-v1.json`. Full Windows regression is also
terminal: 1323 passes/two skips, zero failures/errors, with all 377 declared
production/test/fixture inputs and 114 core-ledger entries unchanged after completion.
See `runs/recovering-keyframe-full-windows-v1.xml` and
`runs/v32-full-windows-post-verification-v1.json`. The exact
production/test/checkpoint snapshot has 437 files/4,798,454 bytes, archive SHA-256
`490AF25DF3923319B731E8D8816D7C8B10D56E6C7A0DAB29735226CE2B473D8D` at
`runs/v32-frozen-test-tree-v1`; verified restoration is recorded at
`runs/v32-persistent-frozen-test-tree-v1`. It predates this document and the separately
tested research scripts, not the production/package/test inputs. All full counts
come from terminal JUnit reports, not progress dots. V31 results remain historical.

Next validate safe first-frame detection in a fresh source-bound protocol, without
relaxing geometry, then obtain supported camera/control motion and independently
verified actual arrival. The original five disjoint held-out worlds, multiple
terrains/conditions, persistent map/scientific-loop integration, hundred-attempt
restart/failure recovery, current baseline/ablation comparisons and consolidated
release gates remain open in `docs/quality-roadmap.md`. The full goal stays active.
