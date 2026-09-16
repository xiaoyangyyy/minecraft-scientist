# Scientist World Rules Mod

Server-only Fabric mod for deterministic causal experiments on Minecraft 1.21.11.
It exposes administrator-only console commands:

- `scientist reset <seed>` resets controller state to the standard rule;
- `scientist variant standard|shifted` selects the mining requirement matrix;
- `scientist status` prints public controller state to the administrator;
- `scientist truth` logs the privileged ground-truth matrix.

The standard and shifted matrices enforce different mining requirements across
stone, iron ore, and diamond ore. Disallowed tool/block interventions are cancelled;
when a permitted shifted intervention would suppress a vanilla drop, the mod adds
the corresponding resource. The same controller also implements a timed
place/activate/wait/retrieve machine and three ordered transform stations whose
assignments rotate in the shifted world. A console-only fall intervention applies a
seeded piecewise law to server-side health; the shifted world changes its breakpoint
and coefficient assignment. Mineflayer has no generic chat action and is not an
operator, so the agent cannot invoke or observe either privileged endpoint.

Build with Java 21 from PowerShell:

    .\build.ps1

The helper discovers a local JDK, uses a project-local Java socket/temp directory,
and disables the long-lived Gradle daemon. Extra arguments replace the default
`clean build` tasks, for example `./build.ps1 test`.

The resulting remapped server mod is written under `build/libs`. A ready-to-copy
pair containing the mod and its pinned Fabric API dependency is staged under
`build/server-mods`. Building does not start a Minecraft server or write an EULA
acceptance file.

The Gradle wrapper is copied from FabricMC's official 1.21.11 example at commit
`0fe7f6e4605aabdfefd69e1b5f2be8a283026272`. The wrapper jar SHA-256 is
`497c8c2a7e5031f6aa847f88104aa80a93532ec32ee17bdb8d1d2f67a194a9c7`.
