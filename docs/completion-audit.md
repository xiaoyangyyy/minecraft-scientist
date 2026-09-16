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
| MineStudio/visual observation path | opaque RGB adapter, concrete label-free grid controller, isolated manifest, fake simulator tests | Implemented; live visual quality not verified |
| Trained planner in live Minecraft loop | `runs/minecraft-live-benchmark-learned`: 47 experiments, all metrics 1.0, checkpoint recorded in both sub-runs | Complete on local Fabric server |

## Current external gates

The component-ablation follow-up passes 79 tests and the ten-seed paired matrix.
Theory repair recovers 0.1481 post-change accuracy; learned symbols remove 13
description units, and transfer of the learned transform program saves eight
interventions. Planner efficiency intervals cross zero in this sample.

1. An OpenAI-provider comparison remains optional and requires separate credentials
   and explicit cost authorization; the external-provider path is measured with
   DeepSeek rather than inferred from an offline substitute.
2. MineStudio's actual visual simulator requires a separate Linux/JDK8 runtime and
   Minecraft/EULA-compliant setup; this Windows host is not visual-ready.

The local Fabric and DeepSeek external-LLM paths are empirically verified.
MineStudio visual results remain a separate optional gate and are not substituted
by offline data.

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

The expanded three-mechanism release gate completed with 78 tests. Its 20-pair run
used per-pair durable checkpoints, three bounded JSON retries, 63 total calls, and
305,554 tokens under the configured hard caps. Both exposed credentials were deleted;
the replacement is stored in Windows Credential Manager and absent from Git and run
artifacts.
