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
| Space + knowledge cognitive map | spatial graph, theory/uncertainty overlays, routing, persistence | Complete at adapter layer |
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
| MineStudio/visual observation path | opaque RGB adapter, grid controller; real Linux simulator reset + five no-op steps and visually inspected RGB image | Runtime verified; grounded visual mechanism discovery not complete |
| Trained planner in live Minecraft loop | `runs/minecraft-live-benchmark-learned`: 47 experiments, all metrics 1.0, checkpoint recorded in both sub-runs | Complete on local Fabric server |

## Current external gates

The component-ablation follow-up passes 79 tests and the ten-seed paired matrix.
Theory repair recovers 0.1481 post-change accuracy; learned symbols remove 13
description units, and transfer of the learned transform program saves eight
interventions. Planner efficiency intervals cross zero in this sample.

1. An OpenAI-provider comparison remains optional and requires separate credentials
   and explicit cost authorization; the external-provider path is measured with
   DeepSeek rather than inferred from an offline substitute.
2. MineStudio now launches in the installed WSL/Linux/JDK8 runtime. Grounded visual
   perception and real visual discovery remain unverified; screen cells are not
   persistent object identities.

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

The evaluator equipment check now fails closed on empty/missing telemetry as well
as wrong items. Windows full regression passes 108 tests; Linux passes all 32
visual adapter/evidence/active-learning/scope tests. The offline MVP benchmark
still reports 1.0 hidden intervention accuracy and task success, 47 experiments,
one-step change detection, eight revision experiments, three symbols and 13 reuse
records. Remaining visual scope is explicit: fixture-provided identity and target
crop, one material, two interventions, fixed camera and no autonomous navigation.

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

The 2026-09-19 controlled visual gate additionally passed dependency-lock checking,
Python sdist/wheel builds, WSL backend readiness (MineStudio 1.1.6, Python 3.11.16,
OpenJDK 1.8.0_502), 108 Windows tests, 32 Linux visual tests, and the MVP benchmark.
The visual result remains scoped as documented above; it does not replace the
structured-world evidence for hidden state machines or conversion chains.

The expanded three-mechanism release gate completed with 78 tests. Its 20-pair run
used per-pair durable checkpoints, three bounded JSON retries, 63 total calls, and
305,554 tokens under the configured hard caps. Both exposed credentials were deleted;
the replacement is stored in Windows Credential Manager and absent from Git and run
artifacts.
