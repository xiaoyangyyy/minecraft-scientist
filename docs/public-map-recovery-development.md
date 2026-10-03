# Public stability recovery and explicit map epochs

The public controller now recovers from a failed map/settle attempt, waits for
visible image stability, and starts a separately versioned local map. Two fresh
natural-world development sessions verify the complete live/public-first replay.
Both chosen stability intervals have zero evaluator-measured player displacement.
**Neither completes reliable navigation or all three supported motion trials.**
This is finite development evidence on seeds 201/202, clear/noon, not held-out
generalization, a statistically established stability detector or global SLAM.

## Implemented contract

`exploration/map_recovery.py` consumes only public calibration, acquisition and
settled results, an RGB image and MOVE/WAIT receipts. Failed controller/tracker
receipts cannot initiate recovery. Completed parents need no recovery. An invalid
parent endpoint, seed or calibration refuses before dispatch. The parent is hashed
and checked for immutability on every terminal path, including exhausted waits.

The policy allows 8–32 one-tick WAITs. Every consecutive image pair gets its own
track namespace; old failed-map points are not replenished. Image stability needs
three consecutive prefixes, at least 32 tracks in four cells, at least 0.5 survivor
fraction, median flow at most 0.2 pixels and 95th-percentile flow at most 0.5 pixels.
Empty/untextured views are unknown, not stable. Receipt/tracker failures and every
executed prefix remain recorded with RGB hashes and original pair-local tracks.

After stability the actual last consumed image becomes a new map anchor. The
child uses seed parent+5 and new detector identities/local coordinates, with an
explicit `parent_sha256` and namespace. No point, pose, past score or acceptance
flag of the old map is rewritten. **Gauges are not stitched and metric scale stays
unknown.** Reacquisition retains the ten-tick cap but raises its first fitting
prefix from three to eight ticks, keeping the original fitting/validation/consensus
gates; actual fitting costs are recorded, not claimed identical to the parent.
Independent different-control validation precedes the unchanged settled trials.

`capture_runtime_camera.py --active-causal --settled-controls --recover-map`
registers `recovery-causal-camera-projection-v1` before launch. The independent
active auditor regenerates every phase from public images/receipts and compares
the entire result before private camera/player bytes are opened. The maximum
193 frames covers the old 105-frame cap plus 32 waits/12 acquisition actions/44
continuation actions; the old active/settled protocols retain their own bounds.
Forged parent lineage, relaxed policy and disguised schema reject before truth.
Original and new camera/body scores are separately retained; late mutations discard
all scores. Private stability measurements are diagnostics, never policy inputs.

## Runtime results

Both complete captures preserve nine capture checks and their independent audits
pass three outer/eighteen nested checks. Each freezes 110 source entries including
103 package files. All raw images, receipts, negative fits, parent and new epoch
records and the audit source remain in
`runs/map-recovery-live-seed{201,202}-v1`. Shared SDK, engine and terrain are unchanged.

| Measure | Seed 201 | Seed 202 |
|---|---:|---:|
| Frames | 83 | 101 |
| Parent continuation | support_aborted | settle_exhausted |
| Recovery WAIT actions | 10 | 8 |
| New image anchor | 72 | 69 |
| New fitted candidates | 24 | 24 |
| New acquisition | exhausted | validated |
| Recovery terminal | map_unavailable | continuation_aborted |
| Completed settled trials / planned | 0 / 3 | 1 / 3 |
| Supported settled translation estimates | 0 | 0 |

Seed 201's public detector rejects the large falling-frame flow (median about
3.56–11.57 pixels, 95th percentile up to 28.25). It selects frames 69→70, 70→71,
71→72: median flow is 0.001080/0.000903/0.000467 pixels, 95th percentile
0.035634/0.033737/0.009163, across eight cells. Independent body step displacement
is zero and maximum rendered-camera step displacement is 0.000074010 blocks.
The later 24 candidates all fail training-only bootstrap gates: insufficient
parallax/coverage, positive-depth support or stationary/rotation ambiguity. They
are not promoted; this is successful scoped stability recovery, not a usable map.

Seed 202 selects public intervals 66→67, 67→68, 68→69. Body displacement is zero;
maximum rendered step displacement is 0.000010949 blocks. Its new map freezes at
79 and validates at 81 with two supported bootstraps. Rendered direction is correct
(1.147981 degrees, 0.153555 degrees rotation, 0.206313 blocks). Body translation
is only 0.083491 blocks, below the **unchanged** 0.15-block gate, so the same
supported prediction is `false_translation` in that domain. Do not average away
the two scores, retime telemetry, lower the gate or select a private winning map.

After eight public geometric WAITs, the child starts at frame 89. Its first control
has four MOVE ticks and three WAITs, ending at 96. Maps disagree in translation
direction by 25.991850 degrees; no translation is supported. Independent scoring
confirms zero body translation and only 0.002697 rendered translation, so this is
a **correct abstention**, not successful movement/route completion. The next
control actually moves (body 0.437138 blocks, camera 0.435791) over 96→100, but
map support is lost. Its partial endpoint and unsupported-motion failures remain.
The third planned handle is not dispatched, not removed from the planned denominator.
At the failed frame 100, 163 original tracks still survive; both maps retain 45
withheld tracks in seven cells and 93/97 training inliers. Their withheld-pixel
95th-percentile errors are 4.764483/4.794050, above three pixels. This is inaccurate
fixed-point ray prediction despite adequate counts, not merely feature exhaustion;
silently dropping rays or lowering the threshold is not a recovery solution.

The earlier parent acquisition results also remain: seed 201 has correct rendered
but wrong current-body direction (~64.39 degrees); seed 202 has a correct body
direction and a rendered translation below the gate. Live pixels are not claimed
deterministic: the current seed-202 parent reaches eight WAITs, unlike the earlier
three-WAIT abort under the preceding source checkpoint.

## Verification boundaries

The new controller has seventeen pure unit cases. Four public/native/auditor
integration cases independently replay all 32 exhausted WAITs without private
reads, retain abstention and reject lineage/policy/schema forgery. Their mock
engine/geometry is not runtime motion evidence. The focused Windows profile
passes 21 cases; an earlier pre-final focused pass has 26 cases from recovery
and settled tests, not a full release.

Both V24 scientific-core releases pass thirty seeds/two repeats/eighteen identical
artifacts per seed and six checks, freezing 106 entries including 103 package
files. Mean reported experiments remain 53.9333333; minimum equation-transfer
saving remains one. The isolated default-only WSL profile passes 573 tests/70
skips (643 total), without Torch/Pillow/OpenCV. Full frozen regressions pass
**1049/two skips on Windows** (1051 total) and **1051/two on WSL** (1053 total),
`runs/map-recovery-full-{windows,wsl}-v1.xml`. The seed-zero MVP benchmark bytes
remain identical; changed-file lint, whitespace and package builds pass.

Post-verification rechecks match all 106 current source hashes in both V24 core
ledgers. Both live captures match all 110 source snapshots, all 108 current project
source entries, and all 170/206 capture artifact hashes. The two external SDK
entries are capture snapshots, not claimed current Windows SDK sources. Saved
active-auditor snapshots and both current auditor hashes match. These checks
do not change physical scores or establish full project completion.

Historical V23 artifacts are preserved with their own sources and scores; they
are not reclassified as current V24 checks. Global lint debt, hosted CI, current
live scientific-fixture releases and consolidated baseline comparisons remain
separate outstanding requirements.

## Next gate and full goal

Actual stationary recovery now enables fresh past-only acquisition, but neither
repeated movement handles nor a frozen map guarantee observable translation.
Next add public stall/low-parallax diagnosis and informative alternative control
selection, followed by map maintenance/refresh during meaningful motion. Preserve
the stationary correct-abstention control and moving unsupported counterexample.
Any gauge registration/new feature generations must be explicit and validated.
The public images show close dirt-wall views, and the stationary control has no
independently measured displacement. Collision is a plausible explanation, not an
agent-provided simulator label. New interventions must respond to observable
stall/uncertainty, not assume every successful command produces movement.

This must lead to public goal control and independently verified actual arrival;
stability recovery does not replace that outcome. The original requirements for
five additional disjoint worlds/multiple terrains/three conditions, adequate
samples/uncertainty, natural scientific discovery/cognitive-map integration, 100
frontier/experiment attempts with restarts/failures/recovery, current symbol/planner
and baseline comparisons, and consolidated CI/release remain open. The goal stays
active, not complete or blocked.

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --active-causal --settled-controls --recover-map --seed 201 --output runs/new-map-recovery
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/audit_active_causal_map.py runs/new-map-recovery --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-map-recovery/audit.json
```

Use new output paths. The optional installed Linux runtime is required for live
capture; default scientific commands remain offline with no provider/API key.
