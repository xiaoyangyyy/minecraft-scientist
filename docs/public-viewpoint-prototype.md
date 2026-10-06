# Public startup view selection and bounded LOOK prototype

This is a research checkpoint, **not a new native control experiment or accepted
map/navigation method**. Main V35 production remains unchanged. It implements
and verifies the next observation intervention rather than changing geometry
acceptance to rescue known failed trajectories.

## First-image-only quality proxy

`research/settled_keyframes/public_view_quality.py` uses the original safe corner
detector, crop, median filter, patch size, texture minimum and 800-feature bound.
Within one public image it compares normalized patches only against other
textured safe corners at least fourteen reference pixels away (twenty-eight at
448 pixels). A patch is a distinctive **proxy** only if a separated competitor
exists and the maximum correlation is below the separately fixed 0.95 threshold.
An absent competitor, untextured view or empty detector cannot invent uniqueness.
All original corner IDs/pixels and patch statistics remain recorded; future
surviving tracks never filter the seed denominator.

Exactly twelve already completed startup images at 15,18,...,48 are ranked.
Eligibility requires at least 32 proxy-distinctive corners and four occupied
cells. Rank by distinctive count, occupied cells, total safe corner count and
finally original ordinal, with exact deterministic tie-breaking. These are
**observation heuristics**, not weaker correspondence, depth or future gates.
The comparison samples only detected corner patches, is not exhaustive or
scale-invariant texture uniqueness, and does not establish physical point identity,
static-world visibility, controllability or future support.

`diagnose_public_survey.py` is now self-contained: it directly binds the five
actually used package modules to both original capture configuration and saved
source snapshots. Seven diagnostic sources (itself, quality and those modules)
and all public input bytes are retained/rechecked. It accepts only the registered
capture schemas and original 48-action startup schedule, requires intact capture/
audit attestations, consumes exactly the twelve unchanged 448-pixel public images,
and refuses escaping paths, changed artifacts or source bytes. Saved audit fields
are attestation only, not numerical policy inputs. No later RGB image, actual
survivor or evaluator position enters ranking.

## Known-data evidence and limits

All four primary diagnostics terminate zero with five passing preservation checks
each, at `runs/public-survey-{long-wait,sustained}-seed{201,202}-v3`. The diagnostic
schema is `known-first-image-public-survey-diagnostic-v2`. The older v1 reports
remain unchanged; they borrowed dependency metadata from the earlier bridge
probe. Removing that otherwise-unrecorded helper import changes provenance, not
the scorer: every full per-view statistic and selected ordinal in v3 exactly
matches its v1 counterpart. The early long-WAIT seed-201 repeat v2 has exactly
the same complete report hash as v1, not a new native validation sample.

| Parent development capture | Chosen past view | Distinctive proxy / original corners | Original final startup view |
|---|---:|---:|---:|
| Long-WAIT 201 | 33 | 350 / 363, eight cells | 48: 242 / 251 |
| Sustained 201 | 33 | 350 / 363, eight cells | 48: 240 / 249 |
| Long-WAIT 202 | 48 | 400 / 462, ten cells | Same chosen view |
| Sustained 202 | 48 | 400 / 462, ten cells | Same chosen view |

The seed-202 result exposes a real limitation: this proxy still prefers the
initial heading whose subsequent registered motor experiments fail. It is not
retuned by future outcomes to force a different choice. Similar ranking across
these correlated development histories is not independent terrain generalization
or proof that the selected view enables a map. No original control/view/outcome
is replaced; all reports explicitly say reorientation was not executed.

Primary report SHA-256:

- Long-WAIT 201: `359a564e3f3d48f6d3f291f48af3bcc6b2eb8c110f93bbc1edc3f459be2d98eb`
- Long-WAIT 202: `9f02f165d85247f3b771c24e87ccb7df8b57c31c07c4590c896131c0ac53dff5`
- Sustained 201: `60f664b33ad4a8b3f39e2a5cea2ba3b22848fd143ef6766f3d5b83cc365a4d62`
- Sustained 202: `ee6509eead54267ce1410c5292442c38e229e1976dd97198a7786a6917c3f9c7`

## Executable bounded control prototype

`public_view_orientation.py` freezes copied startup RGB and ranking before
dispatch. A new typed receipt records actual LOOK/WAIT kind, explicit right-middle
screen-cell targeting, compliance, duration, image hash and contiguous real
ordinal. The chosen startup command phase determines zero through eleven LOOKs,
each followed by two WAITs: at most **33 additional actions**, from frame 48 to
at most frame 81. The counted program does not assert an actual yaw, exact matched
heading, player location, map or arrival. In particular selecting frame 48
executes no new LOOK. No eligible view abstains before dispatch, not an automatic
fallback into movement. Failed, malformed, wrong-target or partially dispatched
controls retain the actual prefix and stop, with no invented completion.

At this original research checkpoint the helper was not yet integrated into
native capture or independent replay. The later isolated implementation and
separate verification are in `docs/oriented-keyframe-development.md`; they do not
change the original prototype results or main V35 production.
Existing acquisition permits anchors only through frame 48; simply hiding LOOKs
outside that clock or resetting its action count would violate the contract.
The next registered native variant must explicitly allow the actual bounded
post-orientation anchor, deduct all survey receipts from the same global 288
actions, and replay every LOOK/WAIT/abstention before map fitting/private scoring.
It must retain the unchanged first-image seed, fitting, consensus and first-
eligible settling/frozen-future gates. No original parent schema may represent
this altered observation chronology. No new native result is claimed here.

The exact V35 source starting point for that integration is restored at
`runs/development-trees/verified-source-2iu7r67h`; its restoration evidence is
`runs/public-survey-development-tree-v1/report.json`. All 456 archived inputs
were verified before allocation. The later candidate intentionally edits that
tree; restoration proves its starting point, not current identity or a test pass.

## Verification

Terminal research suites pass **58 on Windows and 58 on Linux**, zero failures/
errors/skips, at `runs/public-survey-orientation-research-{windows,wsl}-v2.xml`.
The six added research Python files pass Ruff; no default dependency changes.
Thirty-four cases cover deterministic first-image scoring, preserved original
seeds, crop isolation, repeated-patch rejection, absent-competitor abstention,
strict exact profiles/ordinals, first-48-only source-bound input consumption,
forged capture/audit/source/event/image refusal and late-mutation failure.
Twenty-four cases cover exact zero/one/seven/eleven-LOOK programs, actual clocks,
unchanged startup pixels, no-eligible abstention, multiple failure positions,
wrong kind/target/shape/ticks/exceptions, future/untyped selections and frozen
registered costs. Mock ranking isolates receipt timing and does not prove native
heading. Earlier 19/34/24-case reports remain separately preserved.

The preceding sustained-motor candidate's pending Linux auditor session has also
terminated zero: 207 passes, bringing that separate candidate to **395 focused
checks on each platform**. Its six preregistered source hashes still match after
the terminal run. It remains an independently audited **negative** native result,
not promoted into main, and not covered by the historical main V35 full counts.

All original completion requirements in `docs/quality-roadmap.md` remain required,
including reliable actual arrival, disjoint worlds/terrain/conditions, integrated
natural science/map operation, hundred-attempt recovery, baseline/ablation
comparisons and consolidated release/hosted CI. The full goal stays active.
