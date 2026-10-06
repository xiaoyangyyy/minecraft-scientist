# Known frozen-future support-loss diagnosis

This is a retrospective public-pixel diagnostic, not a new successful map or
navigation experiment. It preserves the rejected original result and does not
choose an earlier endpoint, reseed, fit a map, change a geometric threshold or
read evaluator poses. Its implementation is
`research/settled_keyframes/diagnose_frozen_future.py`.

## Source and observation boundary

The input is the complete independently audited walking seed-201 capture at
`runs/oriented-walk-live-seed201-v1`. The original detector freezes 367 seeds in
the first selected public image, frame 77. Original selected past ordinals are
**[77,87,96,105,114,123]**; the consumed settled future is **145**. No views are
reselected. The diagnostic compares these two complete, prespecified schedules:

- Original sparse future: the six original past images plus frame 145.
- Complete future bridge: the same six past images plus **every** frame 124–145.

All tracking gates, seed IDs and denominators are unchanged. The diagnostic's
128-frame bound only accommodates the complete 28-image bridge, rather than
discarding intermediate images to fit the production sparse-view bound. It does
not promote that altered observation schedule into the production map protocol.
The original past and sparse future tracks must replay exactly or diagnosis fails.

Seven saved source files (diagnostic and six tracking dependencies) are bound to
the capture's source snapshots. Thirty-nine input files are hashed before and
after analysis. Audit data is consulted only for saved attestation fields, not
numerical scores. No evaluator-camera, positions or native render matrices are
opened. The original capture and active frozen candidate are not modified.

## Result

The diagnostic terminates zero with four checks true: past replay, future replay,
unchanged public inputs and unchanged source bytes. The original sparse tracker
has **zero** future survivors, exactly reproducing the saved rejection. The full
future bridge also ends at **zero**, not a replacement successful track chain.

| Public frame | Actual requested action | Surviving original seeds |
|---|---|---:|
| 123 | Last past WAIT / frozen map | 181 |
| 124 | First future MOVE | 181 |
| 125 | Second future MOVE | 174 |
| 130 | WAIT | 158 |
| 134 | WAIT | 138 |
| 135 | WAIT | 98 |
| 136 | WAIT | 63 |
| 137 | WAIT | 31 |
| 138 | WAIT | 23 |
| 139 | WAIT | 0 |
| 145 | Original settled future WAIT | 0 |

These counts do not estimate physical displacement or prove the identity of each
feature. The complete past-plus-future rejection ledger is retained: 149 patch-correlation,
89 forward-ROI, 77 adjacent-cycle, 48 forward-status, three step-limit and one
reverse-ROI rejection, totaling all 367 original identities. Viewing the original
public frames at 123/125/134/139/145 shows a large later vertical scene shift near
a block edge; this is qualitative observation, not proof of gravity, body pose,
ground contact or metric depth.

Evidence directory:
`C:/Users/Administrator.DESKTOP-HG1CIML/.codex/storage/minecraft-scientist/oriented-walk-seed201-frozen-future-diagnostic-v1`.
Report SHA-256:
`129c08a4d9f95d829ca2e5b5fe25d1e98fb0ff79af3d58e63fa3dd8869733106`.

## Every consumed future prefix against the frozen models

The expanded, separate v2 diagnosis additionally tests **every frame 124–145**
against all four original frozen bootstraps. It retains the original failed
bootstrap 2 and requires **all** original supported bootstraps 3/4/5, not a new
winning subset. It does not fit or change any 3D points. Each prefix uses the
original selected six-image past plus that particular consumed future frame and
the original fitting/withheld partitions and geometry tests. Every prefix is
retained, including inconsistent/unavailable results; no prefix is selected for
control or substituted for the original settled future at 145.

- Frames **124–132**: all three required models pass their public pixel tests.
- Frame **133**: all three are **inconsistent**, despite 86 retained past IDs.
- Frame **134**: only 28 retained past IDs; all three become **unavailable**.
- Frames **135–145**: zero retained past IDs; all three remain unavailable.

The independent sparse-prefix counts can rise between nearby prefixes because
each is a fresh diagnostic replay from the *same original* first-image identities,
not an online continuous chain or recovery of previously rejected evidence. The
complete raw-future bridge above is separate and loses all identities at 139.
Neither sequence is allowed to replace the original rejection. Pixel consistency
before 133 does not prove a unique map, correct body motion, safety or arrival.

The v2 result terminates zero with the same four preservation/replay checks. All
42 input files and ten saved sources rehash intact, and the active diagnostic
matches its saved source. Evidence is the adjacent
`oriented-walk-seed201-frozen-future-diagnostic-v2` directory; report SHA-256:
`b02831c2651f078f40c0d64bb3e7a63aa7e2a935bd0f4c7ebe727ec5bc6e9515`.
The v1 directory/report remain unchanged.

## Interpretation and next experiment boundary

Skipping all intermediate future views is not the sole cause: using every actual
future image still destroys support. The visible change continues during WAIT,
after the two requested MOVE actions. This narrows the next investigation to
delayed/environment-driven motion, terrain-sensitive control cost and map
observability during settling, without asserting which hidden mechanism caused it.

Next develop a **prospectively registered public observability/control-risk
monitor**, retaining all action receipts and original identities. It should flag
ongoing scene changes and support depletion during both MOVE and WAIT, carry
explicit uncertainty about body/camera/terrain causes, and record an unsupported
transition rather than fabricate a prediction. Any new action duration, safety
response, map-epoch change or integration into planning needs separate tests,
auditor registration and fresh captures. It must not convert frame 125 or another
retrospectively favorable prefix into this experiment's accepted settled future.
Neither this diagnostic nor the two development worlds closes the original
independent-arrival, held-out, integrated-science, longer-recovery, comparison or
consolidated-release requirements.

## Verification

The initial diagnostic tests pass **12 on Windows and 12 on Linux**, zero failures,
errors or skips. They cover complete schedules, unchanged inputs, static-pixel
non-motion, denied reseeding, omitted controls, bad compliance, boolean tick
forgery, changed actual anchors, unordered past views, alternate future endpoints
and output protection. These are synthetic unit fixtures, not native success.
The terminal JUnit files are under the same C-drive spill root:

- `frozen-future-diagnostic-tests-windows-v1.xml`, SHA-256
  `29680d10a68da7b2d0e1800e4dbc1197eb013c79dbdfa80247fe42f9b353a352`.
- `frozen-future-diagnostic-tests-wsl-v1.xml`, SHA-256
  `d38d2c3f89733c4394e8724f82aa75956c82c33fcded5199df3b91c05be5c1b1`.

Both new research files pass Ruff checks and formatting. The candidate protocol
has not changed and no main-package promotion is claimed.

The expanded source now passes **26 tests on each platform**, zero failures,
errors or skips, including full mock-ledger execution, private-numeric sentinel
non-consumption, source/config/image tampering, changes during analysis,
complete-bridge budget refusal, strict clock/action/boolean typing and all-prefix/
all-bootstrap dispatch. The mocked prefix validator is explicitly a selection/
dispatch test double, not valid native geometry. New JUnit receipts:

- `frozen-future-diagnostic-tests-windows-v2.xml`, SHA-256
  `daee40f8f1156e11963074faeafe01e5b65247a05ca473e4ca8576c3bcd89b82`.
- `frozen-future-diagnostic-tests-wsl-v2.xml`, SHA-256
  `8da92ca86ead8e7fe6d61a38eaa07ece7d32c5aa627646b73f127258b5a71634`.
