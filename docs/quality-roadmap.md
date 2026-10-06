# Project quality and completion plan

The user has requested autonomous planning and implementation until this project
reaches an excellent standard. This plan covers the original theory-acquisition
system, not just the latest navigation experiment. Completion remains unproven
until each required gate has current, inspectable evidence.

## Required outcomes

| Gate | Required behavior and evidence | Current assessment |
|---|---|---|
| Executable scientific discovery | Recover equation, rule, state-machine and program theories through controlled interventions; predict unseen interventions; resolve theory crises through independently validated versions | Current thirty-seed offline releases pass on Windows/WSL with public equation experiment design; live-fixture release revalidation remains |
| Evolving symbols | Promote learned executable abstractions only after independent reuse; demonstrate compression and intervention savings against no-library and fixed-library controls | Current transfer executes learned definitions and independently replays both traces over thirty seeds; fixed-library/planner comparison release remains separately required |
| Honest comparison | Matched budgets, shared evidence, random/context/skill-memory/fixed-symbol/full-system baselines; report per-seed samples, uncertainty and costs | Existing evidence; verify statistical claims and separation of offline/LLM/live results |
| Visual generalization | Anonymous RGB observations, calibrated positive/negative boundaries, ambiguity rejection, held-out sessions/seeds/terrain/conditions; no evaluator labels in policy | Bounded natural scenes and condition shifts verified; broad terrain/seed generalization incomplete |
| Persistent cognitive map | Recover journal and projections after restart; bounded exemplars; executable directed routes; reject false loops; retain unknowns, failures and theory references | Persistent route reuse verified; longer exploration and recovery need expansion |
| Open-world scientific exploration | Select reachable experiments by information gain, execute them in natural worlds, expand/reuse the map, discover mechanisms and revise theories with a fixed budget | Natural navigation and live scientific fixtures remain separate; integration incomplete |
| Reliable goal navigation | Use agent-visible spatial evidence; independently verify arrival; no privileged runner-up selection; compare against appearance-only navigation on held-out trajectories | Appearance-only goal search has a verified counterexample |
| Reproducible engineering | Clean installation, automated portable core checks, explicit optional backends, bounded processes/timeouts, frozen source/data provenance, meaningful tests and documentation | Current offline core verifier passes thirty seeds/two repeats on both platforms; four-job CI is defined and locally parsed, but immutable action pins/hosted execution and full release consolidation remain |

## Execution order

Current development checkpoint: `docs/public-map-risk-development.md`, following
`docs/oriented-walk-keyframe-development.md` and the public future diagnostic.
The isolated component passes 34/49/8 focused cases on each platform, and a known
public replay challenges the map at frame 133/prepares an unconnected anchor at
145. Controller/capture/public-first auditor integration now exists in a distinct
research patch; wider regression and fresh native validation remain required.
No recovered map, body motion, navigation or arrival is claimed. A six-view past map freezes in
one development world, but its future loses all original support; another world
still lacks a map. Complete public future tracking reproduces total support loss
during WAIT. Next verify the registered per-action monitor/passive recovery in
fresh native captures, then acquire a genuinely new local map without relaxing
gates or retrospectively choosing successful endpoints. The
preceding main V35 full release remains separate; none of the required outcomes
above is narrowed to these two development worlds or offline compatibility.

1. Repair the visual navigation evidence. Evaluate correspondence/geometry signals
   from existing RGB scans without supplying evaluator coordinates to the policy.
   Reject low-support or inconsistent geometric estimates. Preregister a candidate
   method before fresh-world validation; retain the appearance-only failure as a
   negative result. Do not tune a direction to its privileged arrival distance.
2. Expand natural-world evaluation. Use disjoint acquisition, calibration and
   evaluation sessions; develop on at most two natural seeds and evaluate on at
   least five additional seeds, multiple terrain classes and three lighting/weather
   conditions. Report retrieval acceptance/rejection, false loop closures, route
   completion, interventions, memory growth and recovery failures. Target at least
   0.90 positive acceptance and 0.95 negative rejection, with confidence intervals
   and sufficient samples; do not silently drop failed seeds or reduce scope.
3. Integrate the scientific and spatial loops. Represent mechanism evidence,
   competing hypotheses and available experiments at map nodes. Travel to a
   reachable experiment, perform controlled interventions, update executable
   theories and learned symbols, then reuse that knowledge on another instance.
   Include unchanged-rule transfer and changed-rule revision in natural-world
   evidence, rather than relying solely on independent resettable fixtures.
4. Validate longer operation and recovery. Run bounded long trajectories of at
   least 100 frontier/experiment attempts, controller restarts and injected
   execution failures. Verify that failed controls cannot become theory evidence,
   known negatives cannot be consumed by memory adaptation, unresolved routes
   remain unresolved, and accepted knowledge survives restart. Record all failure
   categories and investigate them autonomously.
5. Consolidate the engineering. Put reusable perception, planning and persistence
   logic in the package, keep experiment scripts thin, add portable CI, lock
   optional runtime requirements, and provide one release-verification entry point.
   Preserve frozen historical evidence when production code evolves. Fix genuine
   correctness/maintenance problems rather than hiding them behind a narrow gate.
6. Run the release comparison and audit. Reproduce structured mechanisms over at
   least 30 randomized seeds, live mechanism/revision gates, paired symbol/planner
   ablations and the held-out visual/open-world suites. Audit every requirement,
   named artifact and claimed number against the current source and run evidence.
   Document remaining scientific limits explicitly. A green unit suite or one
   natural route does not establish overall completion.

## Evidence discipline

The latest `docs/long-wait-keyframe-development.md` checkpoint registers longer
post-motor WAIT windows with unchanged geometry/global budgets. Both fresh audited
sessions complete eleven acquisition trials with no failed/extra windows, but
each retains two views and no map/navigation. Focused/core repeat gates pass
independently; terminal full V35 Windows passes 1418/Linux 1420 cases, two skips
each, with all 456 frozen inputs unchanged after termination. A later isolated
public-support-loss epoch candidate passes 277 focused tests per platform and
fresh public-first audits, but still has no native map; its results do not replace
its own required full verification or establish goal completion. Public
support can now fail without wait recovery, while the original epoch boundary
explicitly requires recovery. The subsequent complete public bridge diagnostic
retains 46 completed trials across four known development captures; none of its
38 bounded continuous comparisons reaches correspondence support. See
`docs/long-wait-public-bridge-diagnostic.md`. The next separately registered
eight-tick sustained-motor candidate also fails both fresh audited development
worlds, retaining only the initial view, and remains unpromoted. See
`docs/sustained-motor-keyframe-development.md`. Next investigate first-image
texture ambiguity, large-perspective correspondence initialization and public
active viewpoint selection, without gauge stitching or weaker track/fitting
gates. The research-only public startup quality/orientation prototype now passes
58 cases per platform and source-bound known-survey diagnostics, but has not
executed a native observation intervention. Its seed-202 ranking retains the
previous failed initial heading; proxy quality is not future support. See
`docs/public-viewpoint-prototype.md`. Next register complete bounded LOOK/WAIT
chronology and actual post-orientation acquisition anchors before fresh public-
first validation. This diagnoses another
mapping failure; it does not close actual arrival, held-out, integrated-science,
long-operation, comparison or release requirements.

The preceding main-source `docs/compound-motor-keyframe-development.md` checkpoint
registers four opaque cardinal-plus-jump programs. Both fresh independently
audited captures still stop at public stability exhaustion, with no map or
arrival. Terminal V34 full regressions pass Windows 1380/Linux 1382 cases, two
skips each, with all 450 declared inputs unchanged after completion. Post-audit
private timing distinguishes body rest from continued image motion; it cannot
become an online settling decision. Next verify a separately registered longer
post-motor WAIT budget in fresh runs, preserving pixel/geometry gates, first-
eligible stopping and the global 288-action cap. Work is isolated from main V34
sources and needs its own source-bound audit and verification. No original
completion requirement is replaced or closed by that bounded development step.

The preceding `docs/safe-seed-keyframe-development.md` checkpoint registers and freezes
first-image support-masked detector seeds, with unchanged geometry gates. Both
fresh independently audited worlds still lack a supported map or future prediction.
All thirty-one original endpoint pairs replay exactly in public raw-bridge
diagnostics, but continuous tracking gives mixed gains and accumulated losses.
Next investigate sufficient controllable view diversity and full past-proposal
geometry, not a wholesale tracker swap or relaxed thresholds. Completed full V33
regressions and seed-bound audits are engineering evidence, not closure of any
original completion requirement above.

The preceding `docs/recovering-keyframe-acquisition-development.md` checkpoint adds
bounded WAIT recovery and sealed, unconnected local epochs. One of two fresh
audited development worlds recovers its failed stability window after three
extra WAITs; neither acquires a supported map or future navigation evidence.
An exact public rejection ledger identifies invalid first-frame detector seeds
and additional correspondence losses. A research-only first-image safe-mask
detector improves support on known image pairs, without replacing old records
or changing tracking gates. Next preregister that detector for fresh controller
and geometry validation; do not promote known-data support to live map/arrival
success or close any original completion requirement above.

The preceding `docs/responsive-keyframe-acquisition-development.md` checkpoint learns
opaque-control novelty responses and freezes future choices from public past
only. Neither fresh development world supports a map; one returns to a useful
control then exhausts a stability wait during a verified five-block fall. Next
test separately registered bounded public recovery and explicit local-map lineage,
without altering geometric gates or consuming evaluator coordinates online.

The preceding `docs/micro-keyframe-acquisition-development.md` checkpoint uses
one-tick acquisition and an unchanged two-tick future control. Two independently
audited development worlds acquire one/six keyframes, but neither supports a map.
The six-view proposal records every rejected bootstrap, including a small-ray-error
model whose withheld landmarks include negative depths. Complete raw tracking on
the same past images also rejects. Next test bounded public-response action
selection for novel-view/parallax acquisition; do not substitute appearance fit,
green provenance audits or historical tests for future motion and actual arrival.

The preceding `docs/stable-keyframe-acquisition-development.md` checkpoint samples
first-eligible public stable/novel images before fitting, binds sparse slots to
actual global frame ordinals and retains all rejected samples. Two independently
audited development worlds acquire one/two eligible views and no map. Known-data
continuous-track comparisons improve some local survival counts but do not prove
map/future-motion success. Next investigate smaller preregistered control steps,
identity-preserving intermediate tracking and numerical-failure retention before
fresh subsequent-control validation. None closes the original gates above.

The latest `docs/shared-endpoint-control-development.md` checkpoint consumes prior
training-only counterexample/maintenance rays into a separately declared version,
then freezes a public WAIT–MOVE–WAIT trial. Two fresh independently audited
sessions select physically stationary player tails and matching camera/player
displacement magnitudes, yet both reject their endpoint map predictions. A
different world cannot acquire a map and executes no endpoint trial. Next improve
observable stable-keyframe excitation/geometry and new-control validation, then
integrate persistent routes and independently verified actual arrival. Timing
diagnostics, repeated failures and green core tests do not close any gate above.

The latest `docs/shared-anchor-reconstruction-development.md` checkpoint revises
training poses/landmarks in an immutable original-anchor lineage, protects every
withheld identity, freezes before new controls and independently replays three
fresh sessions. One new shared camera model passes (0.65-degree direction), one
fails future rays and another acquires no map; current-player direction remains
wrong. This supports only bounded shared-anchor camera reconstruction. Next address
counterexample-driven map crises and observable camera/control endpoint timing,
then persistent routes and independent actual arrival. A current ten-seed structured
ablation repeats exactly but does not establish planner efficiency superiority or
the complete four-form/live/fixed-library comparison gate.

The latest `docs/local-gauge-registration-development.md` checkpoint independently
replays three fresh registered sessions. All complete gauge links reject; one
same-seed replication rejects both future motions despite an earlier local success.
Private parameter scores never select or rescue a rejected public candidate. Next
implement training-only shared-anchor reconstruction with immutable partitions,
new version lineage and independent subsequent controls. Persistent public routes,
goal control and actual arrival still require direct runtime evidence.

The current `docs/rolling-map-maintenance-development.md` checkpoint retains
failed fits and the moving unsupported parent, replaces local maps using only
consumed public history, and validates two consecutive rendered-camera motion
intervals in one development world. Both registered current-player directions
are wrong; another world exhausts all new fits. Next address observable excitation,
explicit gauge registration and public goal control with independent actual arrival.
These correlated local windows do not establish held-out generalization or close
any original full-project requirement above.

The preceding `docs/public-map-recovery-development.md` checkpoint adds bounded
public optical-flow stability recovery and explicit fresh local-map epochs. Both
development sessions select truly stationary intervals; one new map fails and the
other correctly abstains on a stationary control but loses ray consistency during
actual motion. Next add public stall/observability-aware controls and map maintenance,
retaining that moving failure and independently verifying arrival. Local recovery
does not satisfy global mapping/navigation or any other required outcome above.

The preceding `docs/settled-control-development.md` checkpoint runs the public
MOVE/WAIT controller and independent chronological replay in three fresh natural
worlds. None completes settled motion trials: one loses support while falling,
another loses a map despite stationary current-player telemetry, and the third
exhausts acquisition. Next implement bounded public-image stability/landing
recovery and explicit map refresh with new-control validation, then independent
actual arrival. Do not equate WAIT, pixel consistency or a unit test with physical
rest/navigation. All original completion gates above remain required.

The runtime-camera follow-up in `docs/runtime-camera-timing-development.md`
identifies a measured sampling mismatch between image camera positions and player
telemetry. Next align new geometric evaluation to capture-time camera evidence,
while retaining body scoring and accounting for view bobbing. This changes the
diagnostic priority, not navigation acceptance or any required outcome below.

The subsequent `docs/rendered-camera-motion-development.md` checkpoint now binds
full matrices to RGB and scores camera/body motion separately, with a public
half-pixel convention repair. None of 18 registered candidates passes all future
pixel tests. Next investigate short-horizon rolling maps and original-landmark
support/uncertainty; do not accept gate-rejected diagnostics as navigation success.

`docs/causal-window-development.md` implements actual-prefix validation and
immutable-parent refresh. Its fresh replay rejects all four horizons despite real
motion; rolling refresh alone does not acquire sufficient geometry. Next add
public-only adaptive acquisition and live bounded dispatch, preserving every
attempt and all original gates. The offline causal replay is not an online run.

`docs/active-causal-acquisition-development.md` now verifies first-eligible live
dispatch and complete public-only action replay. Its fresh 35-fit trajectory still
rejects every new-view map test (0/1 supported movements). Live dispatch itself does
not solve acquisition conditioning: next request complementary controls from past
public geometry/visibility rather than a fixed six-tick first leg, and investigate
past-only landmark conditioning without changing gates or dropping failed maps.
All full-scope requirements above remain unchanged.

`docs/past-support-acquisition-development.md` now replicates public-support-triggered
early switching: rendered-camera estimates pass in two development sessions while
fixed-cap controls abstain. Both current-player scores are wrong at the short action
switch because image/body sampling phases differ. Next make the separate
position/orientation timing contract explicit, test longer/settled interventions,
and integrate frame-aware continued map/goal control with independent arrival.
Do not substitute a diagnostic shifted-body score or call this navigation success.

- Agent policy inputs contain RGB/opaque public observations and learned action
  programs. Coordinate telemetry and hidden rules remain evaluator-only.
- Development results are not held-out evaluation. State which failed trajectories
  informed a new method and use fresh disjoint validation before promotion.
- Preserve positive and negative runs, source hashes, raw temporal image samples,
  public actions, privileged evaluation and journal/projection verification.
- A rejected expansion must leave accepted memory unchanged. A tentative node is
  not a resolved transition until independent recognition and arrival checks pass.
- Deterministic structured runs must replay exactly. Real image-rendering variation
  is evaluated statistically; do not claim byte-identical live RGB frames.
- Keep optional external-model comparisons and their cost ledger separate from
  offline defaults. Use existing authorized local resources for implementation.

## Current checkpoint

`ce48e34` provides executable A→B→E→F→G route reuse in seed 201 with fresh-session
recognition, shadow validation and independent audit. The accepted G expansion has
22 passing audit checks. Recursive continuation selected heading 3 using panorama
appearance but failed its physical progress veto; 17 rejection-audit checks prove
that accepted memory remained byte-identical. That counterexample changes the next
engineering action: improve spatially meaningful visual evidence before scaling
claims about goal navigation. The overall project goal remains active.

The subsequent RGB-geometry development is documented in
`docs/visual-geometry-development.md`. Dense acquisition and independent replay
are implemented, but a supported-pose increase failed its disjoint error check.
The stricter bootstrap now abstains on the night/thunder development route.
This improves evidence integrity, not navigation success. A fresh higher-resolution
route failed node-B recognition; reference-scale normalization then passed a new
five-node replay with 24 independently recomputed checks and zero physical waypoint
error. The acquisition audit runs offline on Windows without the simulator stack.
Geometry still abstains on that higher-resolution sequence. Gate 1 remains incomplete.

The correspondence follow-up is in `docs/dense-correspondence-development.md`.
Local pretrained matching supplies more points, but independent replay/scoring
finds only 3/8 correct movements at 224 pixels and 1/8 at 448, including a false
translation in a WAIT control. Sensitivity rejects unstable fits; bidirectional
pixel refinement leaves insufficient geometry. All movement is then abstained,
which is not useful navigation. This changes the next action to short-step
temporal tracking and multi-frame geometry, not relaxed acceptance thresholds.
Existing accepted memory and the full remaining gates stay intact.

The temporal follow-up is in `docs/temporal-motion-development.md`. Inspection of
the installed client identified an incorrect equal-axis camera assumption after
widescreen-to-square resize. The public resize repair improves the 448 dense
baseline to 3/8 correct movements (224 remains 3/8); substantial errors remain.
Identity-preserving adjacent tracks, calibrated essential refitting, held-out
intermediate PnP and sensitivity checks are implemented. Four independently replayed
full/short-window experiments pass 26 provenance/computation checks each, but none
supports reliable movement. The next action is controlled WAIT/short-move/settled
RGB acquisition, not relaxed geometry gates. The full project remains incomplete.

The controlled follow-up in `docs/controlled-motion-development.md` implements a
fixed MOVE/WAIT diagnostic and four natural time/weather acquisitions. All 804
RGB frames align with public actions; four independent component replays pass
30 checks each, and the factorial audit proves identical reported physical paths.
Clear scenes yield 2/8 and 1/8 correct validated post-WAIT motions, thunder scenes
zero; stable point counts collapse under thunder in both lighting conditions.
This is new bounded motion evidence and a weather-related diagnosis, not useful
navigation coverage. Next implement stationary-context dynamic-feature rejection
and investigate robust refinement before minimal-pose rejection, then validate on
fresh disjoint trajectories. Do not drop failed controls or reduce the full scope.

The stationary/consensus/camera follow-up is documented in
`docs/stationary-consensus-motion-development.md`; it does not establish a
navigation gain. The next explicit plane/essential competition, documented in
`docs/plane-motion-development.md`, retains branch ambiguity and independent
training/selection/shadow partitions. Both existing clear/thunder sessions pass
thirty replay checks but yield zero protected moving/settled positives, so the
candidate is not promoted. Most clear failures lack the required training-plane
support; most thunder failures lack enough common rays. Plane decomposition alone
therefore does not solve the measured coverage problem. Next evaluate public
keyframe/active-parallax selection and non-collinear acquisition before any planner
integration; fresh held-out trajectories remain required.

The current v4 offline releases freeze the added optional package source and pass
thirty seeds/two repeats/eighteen exact artifacts on both Windows and WSL. The
four-job core/geometry CI definition is locally parsed, not remotely executed;
GitHub metadata access timed out and no Git remote is configured. Action SHA
pinning, hosted CI, live release revalidation and the full visual/scientific
integration gates remain open. No required gate is removed by these additions.

The chronological-keyframe follow-up in `docs/keyframe-motion-development.md`
implements a fixed public prefix sweep and first-supported latch, with complete
independent replay before private scoring. Old clear/thunder sessions show no
coverage gain. A fresh source-bound 6/10-tick natural acquisition captures 229
images; its 108 prefixes yield three correct settled local intervals versus two
terminal full-window poses, and one moving pose in both methods. These are local
intervals, not improved final displacement or navigation; overlapping endpoints
are not independent successes. Thresholds and accepted memory are unchanged.
The next action remains stronger active-parallax/non-collinear acquisition and
fresh disjoint validation before planner integration. Current v5 core releases
freeze the optional keyframe source and pass on both platforms; full regressions
are 496/498, with clean default-only installation passing 318 tests. All original
requirements remain in scope; broad visual/natural scientific integration and
the other outstanding completion gates stay open.

The distinct-handle compound follow-up in `docs/compound-motion-development.md`
adds two fresh 225-frame acquisitions on the existing development seeds 200/201.
All 328 public prefix fits replay independently; both final audits pass 35 checks.
Only three/four of the eight scheduled trials actually form horizontal turns.
Seed 200 supplies correct combined-motion local intervals in four trials, versus
none in seed 201. A seed-200 single leg passes existing geometric/sensitivity
gates with 26.176-degree direction error, exceeding the fixed tolerance. It is
retained as a counterexample; integrity success is not navigation acceptance.
Next implement independent subsequent-frame map/pose validation rather than
relaxing thresholds or promoting early local intervals as final displacement.
The optional package source is frozen in current v6 thirty-seed offline releases
on both platforms. Regressions are 517/519 tests, default-only 329. All broader
visual, scientific-integration, long-recovery and release requirements remain.

The subsequent-frame veto in `docs/shadow-motion-development.md` is fully
implemented and replayed over both existing compound acquisitions. Original
prefix records remain identical; each final audit passes 36 checks. Nevertheless
the 26.176-degree wrong seed-200 pose still passes, while correct pair predictions
fall from twelve to seven and the only correct seed-201 pose is lost. It is not
promoted. The new requirement is explicit public geometric observability/competing
map discrimination, not merely more time-separated PnP checks: the wrong case's
two future frames lie inside a WAIT. This is an explanatory lead, not proven
unique causality. V7 core releases freeze 92 current sources and pass on both
platforms. All held-out navigation, scientific integration, long-operation and
complete release gates stay open; this negative method does not shrink the goal.

The public-only active-view follow-up is in `docs/active-parallax-development.md`.
V1's fresh session produces zero/eight useful turns, revealing an exploitation
eligibility defect. V2 separates conditional exploitation from single-sample
resampling and anchored unknown exploration. Fresh seed-200 acquisition then
produces six/eight effective turns, but only three/eight correct combined local
poses and three wrong second-leg estimates; all forty-two public replay/provenance
checks pass. Acquisition diversity is improved, not geometry reliability or
navigation. The same V2 policy's second development world yields zero/eight useful
turns and zero useful poses, despite another complete 42-check replay. Unanchored
filling omits the only weak direction hypothesis; coverage/revisiting and terrain
effects must be addressed rather than extrapolating the one-world turning gain.
A public future-parallax diagnostic fails diversity in the retained
old wrong case but would lose most correct poses, so it is not a navigation gate.
Next discriminate competing frozen maps/pose explanations under sufficiently
diverse images and validate before integration; do not tune to evaluator directions
or raise the fixed tolerance. V8 core releases freeze 94 current source entries
on both platforms. Sequential action design must also retest lone provisional
anchors and retain control failures. All full-project requirements remain open
where not proved.

The bounded competing-map follow-up in `docs/pose-competition-development.md`
retains a nineteen-trial training-only rotation/translation family and original
withheld intermediate points. Both complete existing V2 sessions pass 54 replay
checks, but the three wrong seed-200 poses remain and seed 201 has no useful
pose. A synthetic distant-plane witness proves the diagnostic can expose an
ambiguity; it does not prove real accuracy. A negative finite search is explicitly
not uniqueness. Next include a sanity replay of the original reference map and
better jointly refined alternatives under independent subsequent views. Do not
promote the failed diagnostic or loosen the fixed physical error tolerance.
V3 separately resamples lone provisional anchors instead of excluding them;
the source-bound seed-201 fresh acquisition completes with the same 225-frame
budget. Current V9 core releases freeze 95 sources and full regressions pass
604/606 tests on Windows/WSL, default-only 354. All full-project gates remain.

The completed V3 audit passes 42 checks and verifies four/eight effective turns
and four/eight correct combined local poses. One selected second-leg pose remains
wrong by 32.412640 degrees. The fresh bounded competition passes 54 checks but
its only witness vetoes a correct pose, not the surviving error. Retain all
controls (including two physically moving nominal settle controls), improve the
model search and independent acquisition, and do not promote these local poses
as terminal displacement or reliable navigation. The full scope is unchanged.

The direction-constrained follow-up in
`docs/pose-direction-search-development.md` exactly replays all twelve original
V3 maps, then fits sixteen fixed camera-center alternatives from training-only
endpoints. All 54 independent replay/scoring checks pass, but the optional veto
would remove three correct pair records while leaving the 32.412640-degree error.
The wrong interval's alternatives fail training/endpoint gates before withheld
PnP; the cause is not uniquely established. Neither fitted-family completeness
nor no-witness results certify motion. Next test frozen explanations against
genuinely new non-collinear acquisition and investigate tracking/point-retention
boundaries; do not tune a replacement to private labels. V10 core releases freeze
96 source entries; full regressions pass 632/634 on Windows/WSL and 361 default-only.
Held-out navigation, integrated natural science/maps, long recovery and full
release comparison remain open. No required gate is removed.

The source-registration follow-up in `docs/source-ledger-development.md` fixes a
contextmanager wrapper being mislabeled as the compound project's runtime source.
Historical raw capture snapshots independently freeze the real module, but the
inference identity entry was not a correct source check. Decorator unwrapping and
forged-stdlib tests repair it without excluding checks. Corrected prior replay
produces identical predictions and passes 35 checks; a new source-bound unchanged
V3 plan is registered. Previous active/diagnostic manifests are historical after
the helper correction. Current visual acquisition and subsequent competing-map
discrimination remain required alongside all held-out/integration/recovery gates.
Post-correction full regressions pass 635/637 tests on Windows/WSL and 361 in the
clean default-only environment; V10 package/core source hashes remain current.
The corrected-source fresh active capture completes with 225 frames and all six
profile gates; prefix inference/independent scoring remain separate. No new
accuracy or navigation claim follows from that acquisition integrity result.
Its subsequent full 164-prefix inference and 42-check independent audit complete,
retaining four/eight correct combined local poses and the same wrong second-leg
direction. Source-registration repair restores current evidence but does not fix
the geometric counterexample. Action-boundary subsequent views remain next work.

The action-switch follow-up (`docs/action-switch-motion-development.md`) freezes
original/alternative maps before future tracking, retains identical surviving
partitions and completes all 54 independent checks. Its optional veto keeps only
three correct pair records/two unique intervals, discarding eight correct records.
The 32.412640-degree error loses all points and is unavailable, not detected.
No navigation promotion follows. Next implement visibility-aware online control
before point loss, then fresh disjoint validation and stable landmark lifecycle;
do not loosen geometry or declare abstention success. The checkpoint passes full
Windows/WSL regressions 672/674 (two skips each), default-only offline 378/56,
and both V11 thirty-seed/two-repeat/eighteen-artifact core releases (97 entries).
All original held-out/integration/long-recovery/full-release requirements remain.

The executable causal visibility controller now operates on one-tick RGB/receipt
callbacks, preserving frozen-prefix pixels and monotonic identities. Independent
request replay passes ten checks over all sixteen old development movement legs:
fifteen request earlier control, one is censored. Anchor 152 requests at tick two
while 206 original points remain. This is an acquisition opportunity, not repaired
pose accuracy. A fresh natural capture runner is implemented with a 209-frame
ceiling; independent fresh policy/map/pose validation is next, then disjoint-world
generalization and the full integration/recovery/release scope. The checkpoint
passes 700/702 full tests (two skips each), 401/58 default-only offline, 28 targeted
tests and two additional wrapper tests; V13 core releases freeze 98 entries.
Do not replace the broad objective with more diagnostic/unit-test success.

Fresh `controlled-rgb-visibility-seed201-clear-v1` acquisition now actually
executes all eight adaptive probes, completes 172 RGB frames and passes seven
raw integrity checks. Reported original track retention remains 102-to-240 at
completion. This is not yet independent policy/map/pose validation or navigation
accuracy; that complete public-first audit is the next action. Keep the script
bytes frozen through that audit (one minor formatter change remains), then version
further acquisitions without rewriting historical source ledgers.

The public-first independent adaptive-map audit now completes all 46 records/
118 prefixes and passes 26 checks. Five/eight control pairs are physically
non-collinear; four/eight combined local intervals are correct. The 128-to-130
second-leg direction is still wrong by 36.654909 degrees, and both supported
first-leg maps remain ambiguous after subsequent views. Thirteen stationary
controls reject; one moving nominal settle control remains unsupported. No
navigation promotion follows. `docs/visibility-map-validation-development.md`
also records a public controller-state issue: second-leg completion reuses the
already-triggered first-leg global loss criterion. Next test phase-relative new
loss with unchanged IDs/absolute gates/budgets, then fresh excitation and
second-leg map validation. Do not substitute diagnostic green checks for the
full held-out, science/map integration, long-recovery and release requirements.

The phase-relative correction now executes a fresh 188-frame/eight-probe capture;
all second legs reach four ticks. Independent replay passes 26 checks over all
46 records/136 prefixes, but two second-leg directions remain wrong (22.762669/
20.831977 degrees). Only one correct first-leg interval survives the optional
diagnostic, with antiparallel shadow motion and an incomplete no-witness family,
not unique/non-collinear validation. All fourteen stationary controls reject.
The public map-ID trace identifies the next concrete action: at tick two one
candidate has 16/48 projected-at-risk landmarks, while generic edge risk is only
0.117845; by tick three its fitting/withheld support collapses to 7/1. Integrate
online candidate-specific landmark protection and second-leg subsequent-control
validation, then independent fresh accuracy/generalization/integration/recovery.
Do not relax the 20-degree gate or promote generic point retention as navigation.
Full regressions pass 725/727 on Windows/WSL (two skips each), default-only offline
408/60, and both V14 thirty-seed releases (98 source entries). Complete evidence
is in `docs/phase-visibility-development.md`; the full objective remains active.

The 48/48 figure previously reported above was a hand-calculation error using
224-pixel bounds for 448-pixel RGB. The corrected 16/48 still exceeds the existing
0.25 warning threshold; controller code and historical raw artifacts are unchanged.

Online protection now freezes the first supported map from current public RGB
prefixes and aborts when its original fitting/withheld support disappears, even
if generic tracks survive. A fresh 180-frame development capture preserves all
eight trials (seven finished/one aborted), and independent replay passes all 26
checks over 46 records/128 prefixes. Ten selected records represent five unique
intervals, all within the existing error gate; second-leg coverage is only one
of eight and its direction error is 19.966090 degrees against a 20-degree limit.
No matched-path improvement, unique map, held-out success or reliable navigation
is established. Further different-control validation and track-loss diagnosis
remain next, followed by all original generalization/integration/recovery gates.
Full regressions pass Windows 749/WSL 751 (two skips each), default-only offline
427/61; both V15 core releases pass six checks across thirty seeds/two repeats
and freeze 99 source entries. Complete evidence and limits are in
`docs/online-landmark-acquisition-development.md`.

Public per-ID loss tracing now exactly reproduces all 51 historical acquisition
prefixes and decisions. Of 43 lost map landmarks in the aborted trial, fifteen
fail ROI gates, nineteen adjacent-cycle and nine patch-correlation. Reducing
pyramid depth or widening horizontal crops does not restore adequate support in
bounded public-only command checks. This rules out those simple proposed repairs,
not all physical/rendering hypotheses. The next implementation is subsequent
different-control validation of second-leg maps and joint geometry/landmark
lifecycle as needed. All broad requirements remain. Both V16 core releases pass
thirty seeds/two repeats/six checks with 99 source entries; prior V15 visual
artifacts remain historical after the trace-source boundary. See
`docs/landmark-track-loss-development.md`; no navigation promotion follows.

The second-leg different-control follow-up now freezes all past fits and the
training-only competing-map search before new actions. A fresh 203-frame capture
retains all eight trials, including two original aborts; six completed trials add
24 one-tick controls. Independent public-first audit passes 29 checks over all 54
records, but the only second-leg candidate has a 38.119-degree direction error
and fails both protected new-control pixel views. Six plausible alternatives also
fail; no map is promoted. Nine selected records overlap into five unique intervals
(four correct/one wrong), and another later compound prefix is wrong. This changes
the next action from collecting missing second-leg validation to multi-view
estimation/residual diagnosis with separate fitting/selection/future observations.
Do not count this negative development result as held-out navigation success.
Both V17 releases pass thirty seeds/two repeats/six checks with 100 source entries;
full regressions pass Windows 811/WSL 813 (two skips each), default offline 443/64.
See `docs/second-leg-shadow-development.md`. All original generalization,
science/map integration, long recovery, live/baseline release and CI gates remain.

Joint multi-view optimization now reserves landmark partitions before bootstrap
and fits all past camera poses/training points together. Its source-frozen replay
retains 36 initializations/all eight historical trials and passes eleven scoped
public-first component checks. Four internally consistent maps represent three
past intervals, all locally direction-correct; on 77→84 the prior fixed two-view
direction error of 21.754527° becomes 10.302308°. However, one future-pixel-consistent
map still has 33.042212° incremental direction error on 113→115. Aggregate direction
would hide this counterexample. No map is promoted, and this development-only result
does not close navigation/generalization. Next investigate public short-baseline
uncertainty and camera/projection model changes while preserving independent future
validation and all original completion gates. Both V18 releases pass thirty seeds/
two repeats/six checks with 101 source entries; regressions pass Windows 858/WSL 860
(two skips each), default offline 443/66. Full evidence and historical source scopes
are in `docs/joint-multiview-map-development.md`.
