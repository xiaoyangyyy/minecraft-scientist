# Completion Audit

This audit compares the current repository with the original self-evolving symbolic
Minecraft system, not merely with the smaller offline MVP. A green unit test is not
treated as proof of live Minecraft behavior.

| Requirement | Current evidence | Status |
|---|---|---|
| Python 3.11, uv, Git, src layout | `pyproject.toml`, `uv.lock`, package build | Complete |
| Opaque structured world and hidden truth isolation | randomized-ID tests, non-serializable callback-free public capability, privileged evaluator boundary | Complete |
| Ordered tool rule | executable Rule IR, hidden matrix evaluation | Complete in structured world |
| Hidden machine protocol | active state-machine version space and shadow validation | Complete in structured world |
| Transformation program | executable Program IR and held-out validation | Complete in structured world |
| Continuous fall law | piecewise Equation IR, active learner, full hidden matrix evaluation | Complete in structured world |
| Multi-form representation selection | Equation/Rule/State Machine/Program selector tests | Complete for current evidence schema |
| Local causal discovery | controlled effects reject failures/confounders; two recovered causal edges | Complete for current mechanisms |
| Versioned Theory Crisis | challenged/superseded v1 and independently validated v2 | Complete in structured worlds |
| Evolving symbol library | three primitives, namespaced cross-world intervention evidence, generation dependency, MDL gain | Complete for current theory set |
| Experience database | append-only SQLite transition replay plus portable JSONL | Complete |
| Space + knowledge cognitive map | coordinate-backed graph plus versioned coordinate-free visual multi-route graph, ambiguity rejection, routing, persistence | Complete for adapter and two-layout controlled visual fixture layers |
| Learned experiment planner | 15,000-world CUDA curriculum; independent 20,000 mechanism worlds plus 5,000 risk worlds; promoted hash locked | Complete for rule/state-machine/program/equation intervention families |
| Reproducibility | byte-identical JSON/JSONL artifacts for repeated seeds | Complete for structured suites |
| Six research metrics plus task utility | unified `full-benchmark.json` metrics | Complete for structured suites |
| Java/Node/Mineflayer controller | Java 21, Node 22, JSONL bridge, dependency audit | Implemented and locally checked |
| Agent-safe live adapter | opaque inventory/block/entity binding and action translation | Implemented with fake-controller integration tests |
| Managed Fabric server | pinned Minecraft 1.21.11/Fabric versions, verified Mojang JAR, EULA gate, process manager, Gradle/Loom dev launcher | Complete and locally started |
| Deterministic live curriculum | reset restores player/mechanism state; mining/machine/transform fixtures; authoritative packet timing | Complete and live verified |
| Live Minecraft law recovery | `runs/minecraft-live-benchmark-real`: all three discrete mechanisms and equation law recover at 1.0 | Complete on local Fabric server |
| Guaranteed live rule mutation | server-only Fabric mod, admin-only truth, deterministic mining/machine/program shifts, remapped JAR | Complete and live verified |
| Live rule/state/program discovery loop | 42 real interventions; accuracy 1.0; detection delay 1; 3 symbols/13 reuse | Complete on local Fabric server |
| Live continuous equation curriculum | 3 opaque tokens, 10 heights, transfer 1.0, v1 superseded/v2 accepted | Complete on local Fabric server |
| Offline baseline matrix | random, stateless context, Voyager-style skill memory, fixed DSL, full library over 10 seeds | Complete for documented offline scope |
| Context-only/Voyager LLM harness | 20 shared-evidence pairs across tool/machine/transform: context 0.8000, Voyager 0.7280, symbolic 0.9646; 95% CIs and cost ledger | Complete for expanded DeepSeek evaluation |
| MineStudio/visual observation path | opaque RGB adapter; tool-region transfer, camera search, motor discovery, continuous detour, persisted reuse, and two-layout route branching in real Linux simulator | Complete for documented fixtures; natural-scene generalization open |
| Trained planner in live Minecraft loop | `runs/minecraft-live-benchmark-learned`: 47 experiments, all metrics 1.0, checkpoint recorded in both sub-runs | Complete on local Fabric server |

## Current external gates

The component-ablation follow-up passes 79 tests and the ten-seed paired matrix.
Theory repair recovers 0.1481 post-change accuracy; learned symbols remove 13
description units, and transfer of the learned transform program saves eight
interventions. Planner efficiency intervals cross zero in this sample.

1. An OpenAI-provider comparison remains optional and requires separate credentials
   and explicit cost authorization; the external-provider path is measured with
   DeepSeek rather than inferred from an offline substitute.
2. MineStudio now launches in the installed WSL/Linux/JDK8 runtime. Controlled
   grounded change regions, camera search, movement-effect discovery, and one
   obstacle-detour fixture and coordinate-free route reuse across held-out control
   mappings are verified; natural-scene object identity and multi-layout map
   generalization remain open.

The local Fabric and DeepSeek external-LLM paths are empirically verified.
MineStudio visual results remain a separate optional gate and are not substituted
by offline data.

### WSL runtime follow-up

The real production visual adapter completed select/interact/wait with seven
simulator steps and four saved RGB frames (exit code 0). Artifacts are in
`runs/minestudio-adapter-smoke`. The default test suite passed 82 tests.
WAIT now executes the requested frame count and stops on simulator termination.
This proves transport and RGB encoding, not grounded action success: the grid
controller uses screen regions and selectable slot handles, not recognized items.
The controller now returns an empty inventory plus `inventory_observed=false`:
slot handles are not measured items. Earlier smoke artifacts contain the old
placeholder inventory and must not be used as evidence of inventory perception.
Interrupted WAIT interventions are marked noncompliant, and terminal episodes
reject further actions until reset. Eight adapter tests pass under Linux/NumPy 1.26.
The corresponding Windows full regression suite passes all 84 tests; `git diff
--check` reports no whitespace errors.

The controlled creative fixture now builds in the player's loaded chunk. Server
logs identified the earlier failure as unloaded absolute-origin positions.
The corrected run (`runs/minestudio-interaction-probe-loaded`) has zero control
ROI change and 45.4381 intervention ROI MAE; inspected before/after images show
the centered dirt block disappearing. This is one verified action effect, not
learned visual mechanism recovery or a general object detector.

A second creative run on seed 1 with stone also visibly removes the target:
control ROI MAE 0.000651, intervention MAE 46.4181. Evidence:
`runs/visual-probe-seed1-stone`. The parameterized probe refuses output overwrite
and separates privileged fixture metadata from public events.

The survival-mode dirt probe (seed 0, 20 consecutive attack frames) also completed
with control ROI MAE 0 and intervention MAE 44.6003. Artifacts:
`runs/visual-probe-survival-dirt20`. This adds survival-mode action evidence; a
tool/material comparison and held-out predictions remain necessary for mechanism
discovery claims.

Paired survival stone interventions (seed 0, 20 attack frames): empty hand leaves
the block intact (ROI MAE 0); iron pickaxe removes it (59.8166), confirmed from
saved images in `runs/visual-stone20-air` and `runs/visual-stone20-iron_pickaxe`.
This is fixed-duration removal, not a harvest/drop eligibility measurement.
Iron-pickaxe screenshots revealed recipe/item HUD text. The probe now supplies
only the central target crop to perception; full screenshots remain evaluator
artifacts. Earlier event features used full-frame aggregates and are not evidence
of a strict text-free visual input boundary. New pixel-evidence tests (4 passed)
abstain when the control is unstable rather than calling all changes success.

The preregistered seed-2 holdout completed with 2/2 correct predictions and 2/2
coverage. Before/after screenshots were inspected; the SHA256 of preregistered
predictions remains 5b1841cdbb260aa7722530f71a79ab97e1f4f46ac6d071a7c46354d6ce563d6f.
Evidence: `runs/visual-theory-holdout-seed2/report.json`. This is exploratory
fixed-camera appearance prediction with fixture-supplied identity, not autonomous
perception or general discovery. The interpreter now supports mandatory
`required_context` scope checks (five new tests); future generated visual IR
requires survival mode, 20 attack frames and the specific appearance outcome.
The historical preregistration is intentionally not rewritten to include this fix.

### Active visual selection audit

`runs/active-visual-discovery-seeds3-4/report.json` completed with one discovery
experiment and two correct holdout appearance predictions. This diagnostic used
the original structured-world prior: every target is reachable by some tool.
That prior inferred the untested tool from the first failure and is not justified
for unknown visual scenes. It is not final active-discovery acceptance.

The visual learner now uses `include_unreachable=True`, allowing all tools to
fail, while preserving default structured-world behavior. Six focused tests cover
active selection, control abstention, insufficient budgets and this prior defect;
the full Windows suite passes 99 tests. The corrected live runner defaults to fresh
seeds 5/6, freezes scoped predictions before validation, and is not yet run.
Fixture-provided identities and a fixed-camera appearance outcome remain explicit
limitations; no claim of general object perception or full visual mechanism
discovery is made.

The corrected-prior seeds-5/6 run completed with two discovery interventions and
2/2 stable holdout appearance predictions (frozen hash
`b357360b3dd2586cc4d69715e048790d53780c1feda48660562131fbb02f47f4`). However,
manual screenshot inspection found seeds held in the nominal empty-hand arm,
apparently collected from cleared vegetation. This is a control violation despite
correct appearance predictions; do not count it as strict empty-hand/pickaxe
acceptance. The fixture now disables tile drops, removes local dropped-item
entities and clears inventory before equipping. Equipment verification and a
fresh controlled run remain required. Future runs also save runtime/source hashes;
that manifest addition was made after the seeds-5/6 process started and is not
retroactively attributed to its artifacts.

The fresh seeds-7/8 rerun is the accepted controlled visual result. Discovery
tested both opaque tools under the corrected unreachable-target candidate space,
then froze scoped predictions before seed-8 validation. Both holdouts were correct
with full coverage. Manual inspection confirms the empty-hand target remains and
the pickaxe target disappears. All four post-settle equipment traces are valid;
the empty-hand traces contain only `air`, and the pickaxe traces contain the
expected tool after setup. `audit_visual_run.py` verifies all controls and the
frozen prediction hash
`381d48f5170f2914285d20b05298505d5826f6b78d8cb87691ee0b9a3277a873`.
Evidence: `runs/active-visual-discovery-seeds7-8/{report.json,integrity-audit.json}`.

The subsequent seeds-12/13 run closes one more control gap. Instead of directly
equipping the fixture tool, each probe places the candidate in a hotbar slot and
the Agent executes `SELECT` against the slot's opaque visual handle through the
production adapter. The independent audit finds one successful select event in
each of the four logs, validates every post-select equipment trace, verifies the
frozen prediction hash
`6558fa9de4faa87f484c45213a12035320881f1bcac88a4273078ec63804ebd3`, and confirms
1.0 holdout accuracy and coverage. This is now the accepted controlled visual
result. Evidence:
`runs/active-visual-agent-select-seeds12-13/{report.json,integrity-audit.json}`.

The seeds-22/23 run removes the fixed central evidence crop. A HUD-safe world
viewport is partitioned into opaque cells; matched control and intervention
differences are evaluated per cell, with abstention when threshold-scale change
is itself control-unstable. All four grounding artifacts, successful select
events, and equipment traces pass independent audit. Both preregistered holdouts
are correct with full coverage, and prediction SHA256 is
`981759aef3c79308305f71db7477ede40719ac4c1896dd9edd6eaa0707d03d08`.
The discovery loop directly consumes the schema-validated grounding artifact;
the legacy central-crop evidence function is absent from this path. The independent
audit also verifies all six runtime-manifest source hashes against the saved source
snapshot.
Connected changed cells are grouped into anonymous regions. The discovery and
held-out regions differ by one boundary cell but independently recomputed IoU is
0.857, above the preregistered 0.5 threshold. This is now the accepted controlled
visual result. The camera remains fixture aimed, so viewpoint-independent object
identity and navigation remain open. Evidence:
`runs/active-visual-persistent-region-seeds22-23/{report.json,integrity-audit.json}`.

The preregistered seed-30 camera-search suite adds camera-only `LOOK` actions and
tests all three fixture placements (negative, center, and positive yaw). The
pixel-only argmax rule selects 3/3 correctly with 15/15 compliant LOOK transitions.
Independent audit recomputes every score from saved PNGs, verifies case reports
and events, confirms the preregistration hash, and matches all five source snapshot
hashes. This proves controlled active viewpoint search, not unconstrained object
detection, locomotion, or path planning. Evidence:
`runs/camera-search-suite-seed30/{report.json,integrity-audit.json}`.

The preregistered seed-49 navigation suite adds duration-aware opaque `MOVE`
actions and a pixel-only closed-loop stopping rule in a straight fixture corridor.
All 3/3 real MineStudio cases stopped on the first central-salience threshold
crossing after seven four-frame moves. An evaluator-only location channel measured
7.504 blocks of displacement per case. The simulator wrapper strips all position
fields before the adapter, and RGB perception fails closed if any survive.
Independent audit confirms that the privileged source existed while public Agent
logs contain no coordinates; it also recomputes every score from the saved PNGs,
verifies first-crossing behavior, all MOVE durations and compliance, report and
preregistration hashes, source snapshots, and displacement.
This proves controlled one-direction visual approach, not multi-direction
navigation, obstacle avoidance, mapping, or path planning. Evidence:
`runs/navigation-suite-seed49/{report.json,integrity-audit.json}`.

The preregistered seed-62/63 motor-calibration suite removes the known-safe-handle
assumption. Four low-level controls are independently permuted behind opaque MOVE
handles using mapping seeds separate from world seeds. The Agent probes every handle
from reset-identical RGB states, selects the unique maximum visual improvement, and
uses it in a fresh closed-loop approach. Both 2/2 cases matched evaluator-recomputed
positive target-axis progress, displaced 7.504 blocks, and stopped at the first
threshold crossing; selected anonymous indices differed (3 and 1), excluding a
fixed-index shortcut. Independent audit recomputes every calibration and approach
PNG, verifies all action events, private permutations, hashes and privileged-source
presence, and confirms public artifacts contain no direction label or coordinate.
This proves reset-controlled active visual motor-effect discovery, not online
obstacle avoidance, route replanning, or a general spatial cognitive map. Evidence:
`runs/direction-discovery-suite-seed62-map700/{report.json,integrity-audit-v2.json}`.

The preregistered seed-88/89 obstacle-replanning suite adds one physical,
color-coded obstacle and a single continuous execution episode after reset-based
motor calibration. The RGB-only policy detects central occupancy, selects a learned
lateral effect, satisfies a seven-move bypass safety gate, and recenters from marker
centroid feedback. Both 2/2 independent control mappings passed with 7.594 blocks
of goal-axis progress, 2.584 blocks of maximum lateral excursion, and 0.066 blocks
of final lateral offset. Independent audit recomputes all calibration and route PNG
features, verifies events, episode continuity, hashes, privileged-source presence
and public non-leakage, and proves the evaluator path never enters the obstacle
volume. This proves fixture-bound online detour selection, not natural-scene
obstacle recognition, arbitrary-layout search, or a general cognitive map. Evidence:
`runs/obstacle-replanning-suite-seed88-map940/{report.json,integrity-audit.json}`.

The preregistered seed-120/121/122 topological-route suite persists the source
detour as two opaque quantized-appearance landmarks and a 13-step functional-role
program. The map contains neither evaluator coordinates nor source motor handles.
Both 2/2 held-out episodes load the same map, match its source landmark, recalibrate
independently permuted opaque controls, and replay the program with zero online route
exploration. Both paths advance 7.594 blocks, detour 2.584 blocks, return within
0.066 blocks of center, and avoid the obstacle volume. Independent audit derives
the program from source events, recomputes all PNG features and temporal pass events,
verifies each held-out binding and path, and checks strict map loading, hashes, source
snapshots, and public non-leakage. This proves persistence and same-layout topological
route reuse, not natural-scene localization, metric SLAM, arbitrary-layout planning,
or a general cognitive map. Evidence:
`runs/topological-route-reuse-seed120-map1200/{report.json,visual-topology.json,integrity-audit.json}`.

The preregistered seed-140–143 branching suite stores two mirror-image, 11-step
programs for opposite offset-obstacle layouts in one version-2 visual topology.
The graph has two opaque source landmarks, one shared goal and two edges. Held-out
episodes reverse the layout order and independently permute all motor handles; both
select the correct branch from RGB, bind it to current controls, execute with zero
online route exploration, and avoid their distinct obstacle volumes. Each advances
7.594 blocks, detours 1.720 blocks, and returns within 0.066 blocks. A between-layout
signature is rejected rather than assigned to a route. Independent audit validates
the multi-edge map, unknown rejection, every image feature/event/binding/path/hash,
and public non-leakage. This proves two-layout branching in a controlled fixture,
not arbitrary layouts, natural-scene place recognition, or SLAM. Evidence:
`runs/layout-route-branching-seed140-map1400/{report.json,visual-topology.json,integrity-audit.json}`.

The evaluator equipment check now fails closed on empty/missing telemetry as well
as wrong items. The previously recorded Windows regression passed all 140 tests;
the corresponding Linux suite recorded 138 passed and two platform-specific skips.
The offline MVP benchmark
still reports 1.0 hidden intervention accuracy and task success, 47 experiments,
one-step change detection, eight revision experiments, three symbols and 13 reuse
records. Remaining visual scope is explicit: fixture-provided identity and
controlled geometry, one material and narrow interventions, no natural-scene
object recognition, arbitrary-layout navigation, multi-layout map learning, or
general path planning.

## Latest release gate

On 2026-09-16, `scripts/verify.ps1 -Benchmarks` completed with 70 tests, Python
sdist/wheel builds, an npm audit of 92 packages with zero vulnerabilities, a clean
Fabric build, the promoted checkpoint hash lock, and passing MVP/full/ten-seed
baseline suites. The exact-utility and learned-planner combined live Fabric
benchmarks both passed every metric at 1.0; the learned run used 47 experiments.
The learned full suite reported 1.0 hidden accuracy and task success, three promoted
symbols with 13 verified cross-world reuse records, and successful rule and equation
theory revision.

The follow-up DeepSeek integration release gate completed with 73 tests, Python
package builds, an npm audit of 92 packages with zero vulnerabilities, and a clean
Fabric build. Its separate live external baseline used non-thinking structured
responses, provider-stateless storage semantics, public opaque evidence only, and
hard call/token limits.

The 2026-09-23 controlled visual/navigation/replanning gate additionally passed
dependency-lock checking,
Python sdist/wheel builds, WSL backend readiness (MineStudio 1.1.6, Python 3.11.16,
OpenJDK 1.8.0_502), all 130 Windows tests, the full Linux suite with 128 passed and
two platform-specific skips, and the MVP benchmark.
The visual result remains scoped as documented above; it does not replace the
structured-world evidence for hidden state machines or conversion chains.

The 2026-09-24 visual-topology follow-up adds five strict unit tests and the real
seed-120/121/122 source/held-out suite. Its independent integrity audit is valid,
all two replay cases pass, both control-role permutations differ, all evaluator
paths avoid the obstacle, and public artifacts contain no coordinates, direction
bindings, or source motor handles. The resulting regression totals are 135 passed
on Windows and 133 passed plus two platform-specific skips on Linux.

The subsequent two-layout branching gate adds five more map/signature tests and
the real seed-140–143 source/held-out suite. Both source and replay success rates
are 1.0, the unknown signature is rejected, the independent audit is valid, and
the regression totals are 140 passed on Windows and 138 passed plus two skips on
Linux.

The next visual-map revision upgrades new saves to format version 3 and adds a
strict route lifecycle: active, challenged, superseded, and rejected. A
high-confidence replay counterexample removes the route from planning; validated
alternatives retain reciprocal lineage rather than overwriting old evidence. The
public `inspect-visual-map` command exposes status, confidence, failure streak,
cost, functional roles and replacement history. Version-1 and version-2 maps remain
loadable, and historical version-2 branching audits explicitly accept their frozen
format rather than comparing them to the newest writer version.

The preregistered seed-140 visual-route-crisis suite then exercises the complete
format-v3 transition in real MineStudio. Two successful source episodes under
different opaque-control permutations establish the old program at confidence 0.75.
In the mirror layout, the stale route still binds and executes all 11 actions but
only advances 2.20 blocks, fails the visual goal and never passes the obstacle; this
control-valid counterexample moves it to challenged and removes it from planning.
One online repair learns the mirror-image program. A separate seed/mapping shadow
replay advances 7.594 blocks, detours 1.720 blocks, returns within 0.066 blocks of
center, and avoids the evaluator-only obstacle volume. The old record becomes
superseded and the active replacement retains a reciprocal parent link. Independent
audit recomputes every calibration and route PNG, action binding, program, signature,
private path metric, lifecycle transition, source snapshot hash and public leakage
check; `valid` is true. Evidence:
`runs/visual-route-crisis-seed140-map1400/{report.json,visual-topology.json,integrity-audit.json}`.

The corresponding release gate passes 147 tests on Windows and 145 tests plus two
platform-specific skips on Linux. `uv lock --check` and both Python package builds
pass. Two independently written seed-0 MVP benchmark directories contain the same
11 files byte-for-byte and retain 1.0 hidden-intervention accuracy and task success,
47 experiments, one-step change detection, eight revision experiments, description
length 1307, three symbols, and 13 reuse records. The historical version-1 and
version-2 MineStudio topology runs also re-audit as valid under the version-3 loader.
No plaintext API key is present in the repository, and the stale deleted-key copy
has been removed from Windows Credential Manager.

The preregistered seed-143 multi-stage topology suite advances from edge selection
to graph composition. Its three-node map contains only a five-action start-to-midpoint
edge learned from source mapping 1400 and a six-action midpoint-to-goal edge learned
from distinct source mapping 1403; it has no direct start-to-goal edge. Under held-out
mapping 1415, the Agent executes edge one, recomputes and matches midpoint
`view_4db0ee57b62ffc72`, replans from that RGB landmark, executes edge two, and matches
the final landmark. The route advances 7.594 blocks, detours 1.720, finishes within
0.066 of center, avoids the evaluator-only obstacle, and uses zero exploratory moves.
Challenging edge two independently blocks the complete route. The integrity audit
validates the upstream source audit, immutable copied-source manifest, source
independence, all PNG features/actions/private paths, map shape and directionality,
both boundary matches, the midpoint planning decision, hashes and non-leakage;
`valid` is true. Evidence:
`runs/multi-stage-route-seed143-map1415/{report.json,visual-topology.json,integrity-audit.json}`.

The expanded three-mechanism release gate completed with 78 tests. Its 20-pair run
used per-pair durable checkpoints, three bounded JSON retries, 63 total calls, and
305,554 tokens under the configured hard caps. Both exposed credentials are no
longer active; no replacement credential is currently claimed by this audit, and
the repository and run artifacts contain no plaintext key.
