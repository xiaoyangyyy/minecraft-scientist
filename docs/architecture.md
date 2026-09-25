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

The real seed-140 visual-route-crisis suite deliberately uses a coarse 8.25 landmark
tolerance to expose a perceptual alias between mirror obstacle layouts (signature
distance 8.0). This is a controlled stress case: two source executions establish
the old program, a held-out replay supplies a control-valid counterexample, the map
excludes the challenged edge, and a newly induced mirror program must pass a separate
shadow replay before supersession. The final graph keeps both programs and their
lineage. This demonstrates the lifecycle over RGB and real simulator actions; it
does not imply natural-scene change detection, learned tolerance calibration, or
general navigation.

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
