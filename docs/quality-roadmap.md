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
