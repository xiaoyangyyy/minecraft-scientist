# Public rolling-map maintenance development

This checkpoint tests continuous local-map replacement after the earlier public
recovery path stops. Its purpose is to recover supported motion during actual
movement, a prerequisite for independently verified arrival. It does not replace
that arrival gate, natural scientific exploration, or held-out evaluation.

## Public controller and evidence boundary

`exploration/rolling_map_control.py` consumes only the completed public recovery
record, its already consumed RGB prefix, calibrated intrinsics and opaque MOVE
receipts. A recovery with no acquired map abstains without any new action. Failed
controls/tracking and invalid seeds/anchors/endpoints refuse before dispatch.
The immutable recovery parent is hashed and rechecked on every terminal path.

The registered policy permits two local epochs, twelve past frames, at most four
additional acquisition ticks per epoch and at most 200 actions across the entire
capture. Every fit records the exact image hashes, its detector namespace, all
bootstrap candidates and the frozen snapshot hash. Negative fits remain in the
record. The original fitting, withheld-pixel, consensus and two-tick validation
gates are unchanged. The first eligible past model freezes before its future
controls; later evidence cannot select or rewrite that past checkpoint.

The controller chooses the least-used public opaque handle, excluding the last
executed MOVE handle. Additional acquisition uses that chosen handle. Validation
chooses a different, less-used handle from the executed history. This is a bounded
novelty heuristic, not a learned world-direction model, a collision classifier or
an information-gain optimum. WAITs have no MOVE handle and add no novelty count.

Each epoch fits a fresh root `CausalMapWindow` from its own past image interval.
The outer epoch journal links the recovery parent and preceding terminal epoch
by SHA-256. Individual roots have distinct detector identities and local gauges;
no landmark stitching, Sim(3) registration or global gauge alignment is claimed.
Both future frames are exposed one at a time after their compliant action
receipts. Frozen landmark identities/partitions remain unchanged within an epoch.
All failed future checks and partial budget/control endpoints are retained.

`terminal=completed` means both bounded epoch protocols ran to their ends, even
if both models were rejected. It is not a physical movement, navigation or goal
success flag. `navigation_accepted` and `global_gauge_aligned` remain false, and
monocular metric scale remains unknown.

## Capture and independent audit

`capture_runtime_camera.py --active-causal --settled-controls --recover-map
--maintain-map` registers `maintenance-causal-camera-projection-v1` before launch.
The independent auditor replays calibration, acquisition, settling, recovery and
every maintenance request chronologically from public RGB/action receipts before
opening private camera/player data. It compares the complete regenerated record,
not just terminal flags. The new schema alone permits 201 frames; earlier schema
bounds and policy registrations remain unchanged. Forged parent hashes, relaxed
geometry/budgets, disguised schemas and invented motion claims reject before
private reads. All invalid audit paths explicitly return no maintenance score.

Camera and current-player motion remain separately scored over each frozen
window's actual two-step future interval. No private winning-map selection,
retiming, reduced translation threshold or removal of failed epochs is allowed.
Original parent and recovery failures remain independently visible. Source and
artifact bytes are frozen through replay/scoring; mutation invalidates results.

## Verification and remaining work

The pure tests cover chronological freeze, explicit lineage, failed past fits,
new namespaces, alternative handle selection, every future prefix, the global
budget endpoint, invalid policy/parent inputs and failed action receipts. Separate
capture/auditor fixtures exercise public-first replay of an exhausted recovery
and reject five classes of forgery. Mock geometry/engine tests are not actual
runtime movement evidence.

## Fresh runtime results

The two development worlds remain seeds 201/202, clear/noon. Neither is a held-out
generalization sample. Both fresh captures pass ten capture checks and independent
audits pass three outer/eighteen nested checks, freezing 111 source entries,
including 104 package files. Runtime images are not claimed byte-deterministic.
Every parent/acquisition/recovery result remains separately recorded.

| Measure | Seed 201 v1 | Seed 202 v1 |
|---|---:|---:|
| Frames / actions | 87 / 86 | 106 / 105 |
| Recovery terminal | map_unavailable | continuation_aborted |
| New maintenance actions | 4 | 5 |
| Maintenance fit attempts / candidates | 5 / 49 | 3 / 30 |
| Completed future validations | 0 | 2 |
| Correct rendered-camera motion intervals | 0 | 2 |
| Correct current-player motion intervals | 0 | 0 |
| Maintenance terminal | map_unavailable | completed |

Seed 201 has no supported maintenance candidate at anchors 72/72/73/74/75. All
49 retained candidates fail training-only bootstrap spatial/depth/parallax or
stationary-ambiguity checks. Its four new MOVE-1 ticks actually displace the player
0.411827 blocks over 82→86 with unchanged height, a post-audit diagnostic only.
Inspecting the public start/end images shows a close dirt wall, trees and little
visible displacement; low parallax/depth ambiguity is consistent with the retained
public fit failures, not proof of an agent-known collision or geometry label.
Movement-command compliance does not imply observable or inferable motion.

Seed 202 preserves the earlier 96→100 unsupported-motion failure. The rolling
controller starts from that stopped public prefix, tries unused MOVE-3 once and
then obtains the first eligible map. Its first attempt has one supported candidate
and is not promoted; the next has two. After past freeze at 101, MOVE-2 validates
101→103. The next fresh root needs no acquisition action, has four supported
candidates, and MOVE-3 validates 103→105. All 30 fits are retained, not just winners.

| Registered domain / interval | 101→103 | 103→105 |
|---|---:|---:|
| Rendered translation (blocks) | 0.169982 | 0.266985 |
| Rendered direction error (degrees) | 2.595411 | 1.106379 |
| Rendered rotation error (degrees) | 0.186465 | 0.250446 |
| Rendered status | correct_supported_motion | correct_supported_motion |
| Current-player translation (blocks) | 0.262864 | 0.302530 |
| Current-player direction error (degrees) | 33.967567 | 37.335694 |
| Current-player status | wrong_supported_motion | wrong_supported_motion |

This is supported rendered-camera motion over two consecutive replacement
windows, not correct current-player motion, a globally aligned trajectory, arrival
or a statistically reliable controller. Shared scene and overlapping history make
the two windows correlated; do not treat them as independent generalization trials.
Both new roots remain unstitched and navigation remains false.

The seed-202 v1 timing diagnostic spans all 106 returned images: renderer camera
base positions differ from current-player eye positions by up to 0.196675 blocks,
but from previous-player eye positions by only 0.000000114 blocks. Captured player
telemetry matches instrumented current-player components exactly; base angles
match both domains in this session. This supports a position-sampling distinction,
not permission to retime registered current-player evaluation or to use evaluator
timing as agent knowledge. The controller still observes only returned RGB.

The independent seed-202 v2 replication also passes all ten capture and three/
eighteen outer/nested audit checks. It uses the same five actions, three attempts/
30 candidates and two future intervals. Rendered direction errors are 2.943707/
1.126835 degrees, both correct; current-player errors are 33.565094/37.365866,
both wrong. Thus both fresh same-seed sessions recover both rendered intervals,
but neither fixes the body-domain mismatch or establishes broad generalization.
Do not claim live RGB determinism from the repeated action schedule.

Runtime evidence is retained in `runs/rolling-map-live-seed201-v1` and
`runs/rolling-map-live-seed202-v{1,2}`. Historical
results in `docs/public-map-recovery-development.md` remain bound to their original
source snapshots; they are not current-source results.

## Engineering checkpoint

The final focused Windows run passes 47 cases. Its initial run preserved five
test failures because rejected audit results omitted the newly introduced score
field; the actual private-read refusal already worked. Invalid returns now
consistently expose null scores for every domain. The original failed XML remains
at `runs/rolling-map-focused-windows-v1.xml`; the corrected run is v2.

Both V25 scientific-core releases pass all thirty seeds/two repetitions/eighteen
exact artifacts and six checks, freezing 107 entries (104 package files). The
current sources and both core snapshots are independently rechecked with zero
mismatches. All three runtime source snapshots and 178/216/216 capture artifact
hashes are also rechecked, with zero mismatches. All 109 current project source
entries per capture and each saved active auditor match too; the two external
SDK entries are retained snapshots, not Windows SDK current-source claims.
These identity checks do not change any physical scores. The isolated
default-only WSL profile passes 593 cases/70 skips, without optional
Torch/Pillow/OpenCV. The seed-zero MVP report retains its historical SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`, and package builds
pass. This does not establish full visual or scientific-map integration.

The full frozen Windows regression passes 1075 cases/two skips (1077 total),
`runs/rolling-map-full-windows-v1.xml`. The isolated-directory WSL full rerun
passes 1077 cases/two skips (1079 total), `runs/rolling-map-full-wsl-v2.xml`.
Changed-file Ruff/format and whitespace checks pass. The built wheel contains all
104 package Python sources, including the new controller. Global historical lint
debt and hosted CI execution remain separate outstanding engineering requirements.

The initial WSL full-suite attempt passes 777 cases/two skips but has 300 setup
errors: its active `/tmp/pytest-of-root/pytest-24` directory disappears during
parallel pytest jobs. The first error loses an existing output manifest and the
other 299 lose the shared parent. Independent integration and affected pipeline
diagnostics pass 25/two skips and ten cases, respectively. Preserve the failed
XML; the full suite passes unchanged with a previously absent, dedicated
`--basetemp=/tmp/minecraft-scientist-rolling-map-full-wsl-v2`. Shared temporary
retention/cleanup is a concurrency hazard, not evidence of a passed suite. Give
parallel pytest jobs distinct, verified-unused temporary directories.

The next required outcome remains supported continuous motion and public goal
control with independently verified actual arrival. Five additional disjoint
worlds/multiple terrains/three conditions with adequate uncertainty, integrated
natural scientific discovery and persistent cognitive maps, at least 100
frontier/experiment attempts with restarts/failures/recovery, matched current
symbol/planner baselines and consolidated release verification remain required.

The next implementation must address these prerequisites together, rather than
increase the count of isolated passing windows:

1. Diagnose inadequate excitation from public ray/flow geometry and choose
   alternative bounded controls when one handle produces insufficient parallax.
   Actual command compliance, surviving feature counts and private displacement
   cannot substitute for public map observability.
2. Register successive gauges using public overlapping-image evidence. New
   detector IDs require explicit mutual/cycle correspondence across common
   absolute image slots; numerical ID equality is not correspondence. Fit only
   designated training associations, independently validate withheld associations
   and reject ambiguity. Every original supported bootstrap must be retained;
   evaluator coordinates cannot select a best similarity transform or scale.
   Local-to-local scale alignment does not give metric world scale.
3. Make control/arrival endpoints explicit and address rendered-camera versus
   current-player timing through public endpoint evidence. A stationary image
   interval is bounded evidence, not permission to shift telemetry to make an
   old score pass. Keep the registered wrong-body results intact.
4. Use the resulting persistent public trajectory to select reachable goals and
   interventions, then independently verify actual physical arrival and scientific
   evidence reuse. Preregister before additional disjoint worlds/conditions and
   compare against the retained appearance-only baseline failures.

These are not implemented by this two-epoch controller. Its real continuous
camera-motion result makes the next gauge/control experiment possible without
claiming that the full trajectory or scientific exploration already exists.

## Reproduction

Use the installed optional Linux visual runtime and a new output directory:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --active-causal --settled-controls --recover-map --maintain-map --seed 202 --output runs/new-rolling-map
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/audit_active_causal_map.py runs/new-rolling-map --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-rolling-map/audit.json
```

Do not overwrite or restamp historical runs. Default scientific-core commands
remain offline and require no Minecraft installation, provider or API key.
