# Architecture

The project separates an agent-visible capability from a privileged evaluator.
WorldFactory creates both from one private ground-truth object, but ScientistAgent
receives only AgentWorldView. The public view supports observation, feasible action
enumeration, primitive actions, and controlled trial reset.

The fast loop maintains finite, inspectable hypothesis spaces and chooses the next
intervention by information gain, normalized execution cost, and learned controller
reliability. A non-compliant intervention updates only its Beta-style failure-risk
estimate and never filters a world-theory candidate. ExperimentExecutor translates
an intervention into primitive action traces and records prior predictive outcomes
alongside intended versus executed behavior. The
slow loop compiles the surviving hypothesis into TheoryIR, validates it, stores an
append-only version history, detects counterexamples, and branches a revised theory.

All theory forms share one executable IR:

- equation theories represent continuous and piecewise numerical laws;
- rules represent latent tool capability and target requirements;
- state-machine theories represent ordered machine protocols;
- program theories represent transformation mappings and composition;
- abstraction learning extracts repeated validated action structure.

The evaluator alone receives WorldEvaluator. It computes behavioral accuracy over
hidden interventions rather than comparing theory source text.

The live backend preserves the same trust boundary. Mineflayer runs in a separate
Node process and emits semantic protocol data. `MinecraftWorldAdapter` immediately
maps every inventory type, block instance, and entity instance to a seeded opaque
identifier. Only coordinates, counts, physical features, and action affordances
cross into `Observation`; semantic names remain confined to the controller side.

## Learned experiment planner

The training package exposes ToolDiscoveryEnv as a Gymnasium environment. A
procedural curriculum samples hidden theories and off-policy partial evidence states.
A shared utility oracle supplies a distribution over equally optimal experiments
under `information gain - 0.05 * normalized cost - 0.5 * control-failure risk`.
A DeepSets-style policy scores each experiment from local posterior features,
normalized intervention cost, estimated control-failure risk, plus masked global
mean/max context, preserving permutation equivariance across opaque
IDs. The same network scores 9 rule interventions, 15 state-machine interventions,
12 program interventions, or 30 equation interventions. State-machine candidates
encode minimum wait thresholds rather than brittle string equality. Program
candidates jointly infer a processor-labelled Hamiltonian path, expand their opaque
node universe when a live terminal output first appears, and keep the action mask
limited to inputs that are actually available after a trial reset.

The mixed training curriculum contains all four version-space families. The learned
policy is an optional replacement for exact information-gain selection; world
execution, evidence, induction, validation, and evaluation remain unchanged. This
keeps policy-training improvements independent of the Minecraft backend.

## Experience and local causality

`ExperienceDatabase` is an append-only SQLite store keyed by transition ID and
mechanism. It persists the complete `TransitionRecord`, including pre/post state,
intended intervention, controller compliance, confounders, outcome delta, and
provenance. JSONL remains the portable run format; SQLite supports ordered replay and
mechanism queries.

`ExplorationMemory` is a separate durable store for the spatial loop. Its immutable
journal records place observations, supervised place calibration, frontier discovery
and frontier outcomes. Typed materialized projections make restart recovery cheap;
integrity verification re-hashes the journal and compares its latest place/frontier
states with those projections. A place uses a bounded farthest-point exemplar coreset
rather than one view, allowing controlled appearance variation without unbounded
growth. Matching remains fail-closed under ambiguity. Successful frontiers create
directed transitions; repeated unknown-action failures persist and eventually block
that frontier. `plan_next_frontier` first finds a confidence-adjusted known route to a
reachable frontier, then balances uncertainty, novelty, travel cost, execution cost,
risk and empirical failure probability.

The Mineflayer adapter can optionally expose four movement frontier objects by setting
`MinecraftConnectionConfig.exploration_stride`. Their identifiers are seeded opaque
hashes; coordinates and prior-visit flags are public physical features, while compass
direction names are never emitted. A `MOVE` primitive binds the selected object to the
bridge's `move_to` action. This makes real traversal possible through the same public
action contract. Path-search computation has a separate bounded timeout from bridge
I/O. The real-server smoke gate tries at most four anonymous directions and records
each controller failure or resolved transition in `ExplorationMemory`; only a
compliant controller success with sufficient displacement, bounded target error and
a new place cell passes. The clean seed-31 run resolves one frontier in a newly
generated vanilla world. The broader release gate also validates memory and planning
on a hidden procedural graph. These are short traversal and infrastructure gates,
not evidence of long-horizon open-world Minecraft competence.

The `minecraft-explore` runner keeps the vanilla server alive while restarting the
Mineflayer controller and reopening `ExplorationMemory` after a configurable number
of attempts. Each spatial place owns a separate bounded `PerceptualPlaceMemory`
coreset. `AnonymousSceneEncoder` derives a fixed-width descriptor from a world-aligned
local window of public block geometry, hardness and diggability; object IDs, object
order, absolute coordinate values and semantic names do not enter the descriptor.
Public coordinates define the grid anchor, giving whole-cell translation invariance.
The bridge can return up to 2048 nearby blocks so the fixed window is stable under sub-block
reconnect drift. Perceptual views are journaled, projected into SQLite, replayed by
the integrity verifier, and matched by nearest exemplar with fail-closed ambiguity.
Spatial cells currently provide self-supervision and an independent evaluator for
perceptual retrieval. Repeated terrain can still alias, so an unseen frontier never
commits a scene-only loop closure without topological evidence. Learned pixel
embeddings and cross-biome visual calibration remain a separate required layer.

For pixel observations, `RGBSceneEncoder` crops HUD-heavy top, lower and peripheral bands,
area-pools the remaining view to a 16x16 CIELAB grid, normalizes each channel and
quantizes it into 768 integers. `GridVisualController` exposes the serialized
signature, encoder version and dimensionality through the public `Observation`;
`rgb_signature_from_observation` validates that boundary before persistence. The
`modality_place_views` SQLite projection keys each coreset by `(modality, place_id)`
and persists its supervised positive/negative calibration. Journal replay checks
both recorded views and calibration state. Matching uses the per-place radius and
fails closed when two places fall within the ambiguity margin.

The `benchmark-rgb-generalization` gate uses separate real MineStudio episodes for
training and held-out roles. Deterministic brightness and horizontal-translation
perturbations are applied within each role, but no held-out frame participates in
radius fitting. Its independent audit reloads copied pixels rather than trusting
reported signatures. The current representation has controlled-fixture evidence;
broader natural-biome invariance remains unproven.

`RGBPanoramaEncoder` adds an active-view layer without introducing a compass. It
takes the mean and standard deviation of the 768-dimensional single-view signatures
over a complete scan, producing a 1,536-dimensional bag-of-views descriptor. The
descriptor is invariant to cyclic scan start and arbitrary scan ordering. The Agent
infers the number of equal camera interventions in a full turn from RGB recurrence;
evaluator-only yaw telemetry verifies that inference but never enters its policy or
events. The natural-loop gate fits a fail-closed radius from separate restart and
physical-return sessions, persists the calibrated descriptor through the existing
modality journal, then tests a disjoint out-and-back loop and another process
restart. This proves bounded camera-start invariance and pixel-only loop closure on
one untouched generated forest/village seed under fixed conditions. It is not a
learned embedding, a compass, cross-biome localization, or metric SLAM.

The condition-split variant adds `RGBEdgeSceneEncoder` and
`RGBEdgePanoramaEncoder`. The former median-filters luminance, retains quantized
strong-edge occupancy and unsigned orientation, and the latter keeps the same
order-independent panorama aggregation. Formal scans may use an odd number of
temporal RGB samples per heading; the saved view is their pixelwise median, while all
raw samples and hashes remain in the evidence directory so the auditor can rebuild
the aggregation. Calibration samples estimate a maximum-gap radius around already
persisted exemplars but do not silently become new centers in the same operation.
Negative calibration exemplars are persisted as well. After calibration,
`consolidate_modality_view` accepts a candidate only inside an existing local radius;
the new exemplar's radius is capped below its nearest persisted negative by the
original calibration margin. Matching evaluates each exemplar against its own
radius, preventing an accepted weather view from consuming a known negative while
allowing a validated condition-specific prototype to cover held-out weather.
Calibration and consolidation remain explicit hash-chained events replayed by
SQLite integrity verification; legacy states without local radii retain
global-radius semantics.

The accepted seed-200 noon/clear to night/rain run combines five-frame temporal
aggregation with persistent consolidation. The seed-201 ridge follow-up exposes two
anonymous movement programs—forward and forward-plus-jump—and probes the Cartesian
product of six RGB-shortlisted headings and those programs in independent reset
sessions. Pixel reversibility alone selects the program; evaluator coordinates are
read only afterward. The selected probe moves 5.429 blocks and returns within 0.651,
and the held-out night/rain loop moves 5.644 blocks and returns within 0.006. Startup
attempt counts and pre-restart discovery/loop checkpoints make simulator flakes
auditable. The independent audit reconstructs temporal medians, selection scores,
local memory boundaries, public actions and evaluator-only physical metrics. This is
two-seed bounded natural-terrain evidence, not multi-frontier search, cross-biome
localization, long-horizon open-world exploration, survival, or metric SLAM.

`VisualFrontierProgram` serializes relative turn slots, anonymous motion slots,
repeat counts and frame budgets without retaining raw simulator handles or semantic
control names. `align_panorama_heading` cyclically aligns a current ordered scan to
a stored RGB heading reference, so a route program remains executable after restart
when arrival yaw differs. `FrontierPlan.travel_action_roles` carries one program per
known path edge, and `ExplorationMemory.plan_route` returns the same executable form
for any connected pair of known places.

The natural-expedition gate persists only verified forward edges A→B→C→D. A remote
active frontier competes with a high-risk local decoy; planning from A returns the
two-edge travel prefix plus the C intervention. The held-out midnight/thunder
episode discovers D after noon/clear training and night/rain calibration, and a
later process restart independently replays all three edges. Reverse traversal is
intentionally absent: natural-terrain drift prevented a safe homing validation, so
no reverse transition is inferred from action symmetry.

`LocalCausalEngine` accepts `(context, intervention, outcome)` samples. It rejects
controller failures and samples carrying possible confounders before computing a
controlled effect. A directed edge is emitted only when at least two treatment
levels exist inside the fixed context and their outcome means differ by the requested
minimum effect. The full benchmark therefore recovers `height -> damage` while
holding armor fixed and `armor -> damage` while holding height fixed.

## Multi-form symbolic induction

`RepresentationSelector` chooses the minimum expressive representation justified by
the evidence: order-sensitive outcomes select a state machine, repeated composition
selects a program, continuous inputs select an equation, and relational predicates
select a rule.

The equation learner maintains an inspectable version space of hinge laws:

    damage = armor_slope(armor) * max(0, height - safe_height)

Both the safe-height breakpoint and the opaque armor-to-slope assignment are latent.
Predictive entropy chooses interventions, an untested pair provides shadow
validation, and the privileged evaluator measures law recovery over the entire
hidden intervention matrix. A map-only change tests structural transfer; a guaranteed
safe-height and slope permutation triggers a separate equation Theory Crisis and an
append-only v2 replacement.

## Live backend boundary

`bridge/mineflayer/server.cjs` is a JSONL subprocess with a fixed request surface:
connect, observe, act, disconnect, and ping. Its action dispatcher is a whitelist;
no agent-generated JavaScript or `eval` is permitted. The Python bridge correlates
request IDs, applies deadlines, captures bounded stderr, and terminates the child
process on close.

The supported live stack is Minecraft Java 1.21.11, Java 21, Node 22, Mineflayer
4.39.0, and mineflayer-pathfinder 2.4.5. `minecraft.server` pins the official jar's
URL, size, and SHA-1 and refuses unknown existing jars. Fabric preparation uses
exact loader/installer/API versions, records the downloaded launcher's SHA-256,
locks both server mods, and refuses later hash changes. It writes `eula=true` only
after the CLI receives the explicit `--accept-eula` flag.

`MinecraftServerProcess` owns the Java child process, consumes its output on a
dedicated thread, waits for the vanilla `Done` readiness marker, restricts console
input to one command per line, and performs graceful `stop` with terminate/kill
fallbacks. `ScientistCurriculumScenario` emits a deterministic, injection-checked
command sequence that clears a bounded arena and places mining targets, a timed
machine, three transform stations, and the required opaque inventory. The
server-only Fabric mod enforces standard and shifted tool matrices, ordered
place/activate/wait/retrieve state transitions, and a rotated transform program.
Only administrator console commands can reset variants or print ground truth; the
bot has no chat surface and is never an operator.

`minecraft-live-run` wraps the adapter in a privileged trial-reset controller, but
passes only the resettable world view to `ScientistAgent`. The existing version
spaces, information-gain planner, validator, append-only registry, abstraction
learner, and evaluator therefore run without a Minecraft-specific learner. Stable
opaque block IDs are position-scoped so a machine may change physical state without
invalidating the planned target. Sneak-modulated block use distinguishes the public
`activate` primitive from `place_input` and `interact` on the wire.

The live equation path uses a separate physical fixture arena with three opaque
inventory tokens and ten visible intervention heights. An administrator-only Fabric
command applies the seeded hinge law to real server-side player health. The agent
chooses opaque token/height interventions and observes health deltas; the evaluator,
not the agent, retains the seed-to-breakpoint and slope permutation. A shifted
variant changes both parameters, causing the same append-only equation Theory Crisis
and v2 promotion used in the structured full benchmark.

`MinecraftWorldAdapter` translates current bindings into the existing primitive
action vocabulary:

- `select` equips an opaque inventory object;
- `look` applies a camera-only delta toward an opaque screen handle;
- `move` holds one opaque locomotion control for an explicit number of frames;
- `interact` digs an opaque block or attacks an opaque entity;
- `place_input` equips its opaque input and uses an experimental fixture;
- `activate` performs a distinct sneak-modulated fixture activation;
- `wait` advances controller time.

Every action produces a normal `TransitionRecord`, including controller compliance,
opaque inventory deltas, provenance, and failure information. Observations are also
merged into a quantized NetworkX cognitive map whose histograms contain opaque IDs,
not Minecraft names.

## Remaining research layers

The Mineflayer/Fabric path has been executed against the local resettable server;
its measured rule, state-machine, program, equation, mutation, and trained-planner
results are recorded in `docs/completion-audit.md`. Visual perception remains
intentionally separate from the structured protocol path. The optional
`MineStudioVisualAdapter` now formalizes that boundary: `MinecraftSim` supplies RGB
frames, an injected `PerceptionController` emits opaque detections and low-level
actions, and only normalized `Observation`/`TransitionRecord` objects reach the
scientist. Frame bytes become a digest plus dimensions; private controller state is
never serialized. The adapter rejects non-opaque IDs and common semantic-name leaks.
This contract is fake-simulator tested and has also run against the isolated
Linux/JDK8 simulator. Controlled evidence now covers opaque tool selection,
anonymous change-region matching, active camera search, and one-direction
closed-loop visual approach. A reset-controlled motor-calibration suite additionally
tests four independently permuted opaque MOVE handles and selects the target-directed
effect using RGB only. A subsequent continuous-execution fixture uses the learned
effects, a color-mask occupancy signal, and marker-centroid feedback to detour around
one physical obstacle and recenter; evaluator-only coordinates audit collision-free
geometry after the episode. A separate `VisualTopologicalMap` now forms opaque nodes
from quantized RGB landmark signatures and persists successful functional-role action
programs without coordinates or raw motor handles. A preregistered source/held-out
suite reloads one learned route and rebinds it to two independently permuted control
maps; both evaluator paths avoid the obstacle. This is a coordinate-free topological
memory for a controlled two-landmark fixture. These fixtures do not establish
natural-scene object recognition, metric SLAM, arbitrary-layout route search, or
general path planning. The map's second format uses a `MultiDiGraph`, retaining
competing programs between the same landmarks and choosing by confidence-adjusted
cost. Landmark matching rejects equidistant/near-ambiguous candidates. A two-layout
real-simulator suite learns mirror-image obstacle routes into one 3-node/2-edge map,
selects the correct branch in held-out episodes, and rejects an unseen intermediate
signature; it remains a fixed-camera controlled-fixture result.

Format version 3 turns route replay failure into an auditable lifecycle rather than
silently averaging it into a success rate. Active routes may become challenged after
a high-confidence counterexample, be reinstated after independent validation, be
rejected, or be superseded by a distinct program connecting the same landmarks.
Only active routes enter shortest-path planning. Supersession preserves both route
records plus reciprocal parent/replacement IDs, and loading validates those links.
The CLI command `inspect-visual-map` reports the status, evidence counts, failure
streak, cost, functional program and replacement lineage; map formats 1 and 2 are
still accepted with active/default lifecycle fields.

Format version 4 adds optional per-landmark `VisualMatchCalibration` records. A
controlled labeled calibration requires at least two positive signatures and one
negative signature. It sets the acceptance radius to the maximum positive distance
plus a preregistered margin and refuses the calibration unless that boundary remains
another margin inside the nearest negative. `match` uses the fitted radius when
present and otherwise retains the map-level tolerance, so historical maps preserve
their behavior. The CLI reports sample counts, positive spread, nearest-negative
distance, margin and radius. Formats 1–3 remain loadable without synthesized
calibration claims.

The real seed-140 visual-route-crisis suite deliberately uses a coarse 8.25 landmark
tolerance to expose a perceptual alias between mirror obstacle layouts (signature
distance 8.0). This is a controlled stress case: two source executions establish
the old program, a held-out replay supplies a control-valid counterexample, the map
excludes the challenged edge, and a newly induced mirror program must pass a separate
shadow replay before supersession. The final graph keeps both programs and their
lineage. This demonstrates the lifecycle over RGB and real simulator actions; it
does not imply natural-scene change detection, learned tolerance calibration, or
general navigation.

The follow-up adaptive calibration suite uses immutable copies of five already
audited MineStudio episodes with disjoint roles: two training positives, one training
negative, one held-out positive and one held-out negative. The deliberately broad
8.25 baseline accepts the negative at distance 8.0. The fitted radius is 1.0
(positive spread 0.0 plus margin 1.0); it accepts the held-out positive and rejects
the held-out negative and a distance-4 intermediate control. Its independent auditor
re-renders trajectory features from PNG evidence, recomputes the fit, checks every
artifact manifest and reloads the v4 map. Because all positive signatures are
identical in this fixture, this is evidence for supervised boundary persistence and
fail-closed separation—not robustness to natural appearance variation or a learned
embedding.

The seed-143 composition suite exercises the graph rather than a single stored
program. Edge one is learned from one opaque-control source episode and edge two
from another. A held-out episode plans across all three landmarks, executes only the
first edge, re-encodes the resulting RGB view, verifies the intermediate node, and
calls `plan` again from that node before executing edge two. The graph has no direct
start-to-goal edge, and challenging edge two blocks both the midpoint plan and the
composed start plan. This validates two-edge topological composition and boundary
relocalization in one controlled route, not large-scale loop closure or SLAM.

## Shared-evidence external evaluation

`benchmark-external-llm-expanded` constructs candidate interventions and active
evidence entirely through the public `AgentWorldView`. Privileged evaluator truth is
kept in a separate in-memory scoring object and is never serialized into prompts.
Context-only DeepSeek, Voyager-style cross-world memory, and the finite symbolic
posterior are scored against the same tool, machine, and transformation interventions.
Each completed source/target pair is checkpointed before the next paid request; the
usage ledger also records failed structured responses and bounded retries.

## Paired component ablations

`benchmark-ablations` fixes world seed and intervention cap within each pair.
The random selector samples untested informative actions. The theory-repair ablation
retains anomaly detection but leaves the old theory challenged. Library learning is
disabled independently; a separate transfer probe withholds the source's accepted
transform primitive and measures downstream discovery cost. The learned-planner pair
compares the checkpoint against the information-gain selector. The checkpoint hash,
per-seed results, and normal-approximation intervals of paired differences are saved.
These are component-specific pairs, not a factorial experiment. Equal caps do not
mean equal realized experiment counts; symbol validation costs remain included.
