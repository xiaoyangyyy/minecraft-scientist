# Separately registered compound-control settling windows

This is a development checkpoint, not proof of actual goal arrival or broad
held-out navigation. It addresses observed public-image wait exhaustion without
changing any pixel, correspondence, fitting, consensus or future-validation gate.

## Contract and source separation

`StableLongWaitJumpKeyframePolicy` extends the anonymous four-motor registration
with `compound-post-motor-24-frame-windows-v1`. Initial normal/recovery windows
remain sixteen WAITs; post-motor acquisition/future normal/recovery windows allow
twenty-four. The minimum remains eight WAITs and three consecutive stable image
pairs. Stop at the first eligible prefix, not at the maximum window length.
`PostMotorWaitWindow` is a frozen exact-integer timing profile, not a geometry
policy. Actual controls remain exact compliant one-tick receipts. The unchanged
recovery helper shares at most two extra windows across the entire episode and
never borrows actions beyond the global 288-action cap.

`--long-wait-keyframes` requires all six original explicit keyframe/compound flags.
Its schema is `long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1`.
Parent schemas cannot represent it. The public journal binds every accepted,
failed and recovery window to its registered phase budget. Capture and independent
auditors replay the complete action/image/seed/epoch/model chronology before
opening private physical scores. Empty maps or invalid audits cannot yield motion,
metric scale, connected gauges or navigation acceptance.

Main V34 and its live full regressions were preserved while developing this
candidate in an exact source copy under ignored `runs/development-trees`. V34
full tests subsequently terminated Windows 1380/two skips and Linux 1382/two
skips, all 450 declared inputs matching after termination. Those historical tests
do not cover the new protocol. Candidate executing package/capture/auditor bytes
were kept unchanged during its tests, native runs and independent replays, then
merged into main. All 116 candidate core-source entries and 118 project entries
from each native capture match main after that merge. The two SDK source snapshots
are recorded separately and not counted as project files.

## Tests and current offline checks

Candidate controller/profile tests pass sixteen on Windows and Linux. They verify
untextured initial exhaustion retains three sixteen-frame windows, static real
pixels still stop after eight WAITs, and moving real pixel sequences reach the same
first eligible twenty-frame endpoint in both old and new modes. The new mode avoids
the old extra-window ledger in that controlled case. Additional budget tests retain
all three failed twenty-four-frame post-motor windows, cap a genuinely partial
episode at action 288, and reject untyped flags, unregistered profiles and relaxed
budgets before dispatch. Synthetic stability overrides are marked as budget tests,
not native success. The new policy preserves all four geometry/stability configs
and cannot masquerade as its parent.

Auditor/CLI tests pass ninety-two per platform: twenty-two additional cases exercise
positive public replay, parent-schema/motor/detector tampering, exact long-window
registration, saved window-budget tampering and missing-flag rejection before any
private scoring. Existing motor/seed/adapter/map/recovery/endpoint compatibility
tests pass 128 per platform. Total focused candidate checks are 236 per platform.
After an import-only lint correction, main's sixteen controller tests also pass.

Original failed reports remain: controller v1 passes fifteen/fails one per
platform because its test incorrectly used forbidden `anchor=280`. The corrected
test starts from the real registered anchor and reaches the cap through actual
recorded controls, without relaxing production anchor bounds. Windows v2 was a
wrong-directory command that ran no tests, not a passing result. Corrected reports
are `runs/long-window-controller-windows-v3.xml` and `-wsl-v2.xml`.

Both candidate core verifiers complete thirty seeds/two repetitions/all eighteen
exact artifacts/all six checks at `runs/core-release-seeds0-29-{windows,wsl}-long-window-v1`.
They freeze 116 entries (113 package files), remain unchanged through execution
and match main. Mean reported experiments remain 53.933333; minimum equation
transfer savings is one. Seed-zero MVP still has 47 experiments, three symbols,
thirteen reuses and byte hash
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Main builds without new dependencies; default-only Linux passes 673/78 optional/
platform skips, zero failures/errors. New full Windows/persistent-Linux verification
is required separately at `runs/long-window-full-windows-v1.xml` and
`runs/long-window-full-persistent-tree-wsl-v1.xml`; no passing full result is claimed
without terminal process status, JUnit and after-terminal input verification.

## Fresh native evidence

`runs/long-wait-jump-live-registration-v1.json` declared the timing/control budgets,
two development seeds and source separation before launching either session. The
SDK/JAR, clear/noon/creative setup, anonymous handle count and motor durations are
unchanged. No private rest position guided the online policy and no raw fitting
research was promoted. These repeated development seeds are not held-outs or
matched native-byte trajectories, so the results do not establish a statistical
effect size against earlier registrations.

| Seed | Actions / public images | Selected frames | Completed trials | Post-motor WAIT range | Failed / extra windows | Terminal |
|---|---:|---|---:|---:|---:|---|
| 201 | 248 / 249 | 56, 89 | 11 | 15–18 | 0 / 0 | map_unavailable |
| 202 | 269 / 270 | 56, 111 | 11 | 15–21 | 0 / 0 | map_unavailable |

Each run preserves 120 executing source snapshots, seven passing capture checks,
three passing outer and eighteen passing nested independent audit checks. Their
502/544 artifact hashes match, and independent public replays are valid. Both
complete the initial eight WAITs and all eleven first-eligible acquisition waits.
Neither creates a past proposal, frozen map, future motion or accepted navigation.
Public stability success is not a claim that every tail has zero player/camera
displacement; private diagnostics remain separate from online decisions.

## Remaining bottleneck and next intervention

Both acquisitions keep only their original local epoch, despite four/ten completed
trials failing correspondence support. In seed 201 the first lost pair 89→123 has
124/304 tracks (0.407895), six cells and 6.140685 median pixels: visible difference
alone cannot override the unchanged 0.5 survival requirement. In seed 202, after
selecting 111, all eight later trials fail support, eventually retaining only one
of 432 first-image seeds. Returning to the old anchor on another handle is not
new validated view evidence.

The original epoch policy explicitly requires an *extra recovery window* before
opening a new local map after support loss. Longer normal waiting correctly uses
none, leaving this separate map-observability failure untouched. Next investigate
a distinct public-support-loss epoch registration that seals the old map without
pretending gauges are connected. Keep old protocols/tests unchanged, preserve all
old failures and current seeds, and validate new first-image seeds and scoped
control history from fresh runs. A re-anchor diagnostic or extra epoch is not
automatically a map, future prediction or arrival.

All original required gates remain: reliable actual arrival, at least five disjoint
held-out worlds/multiple terrains/three conditions with adequate uncertainty,
natural scientific/map integration, hundred-attempt restart/failure recovery,
current full comparison/ablation suites and consolidated release/hosted-CI
verification. This bounded settling improvement does not close those gates.
