# Current-production comparison and complete release scope

Tom's September 11 clarification: the goal is to launch the complete encoder
package — MCP, prompt and supporting changes together. The comparison below
is evidence toward that release decision. Its completion is not completion of
the release, and the old merge/deployment gates remain until Tom's launch call.

## Verified production baseline

The running daemon reported source `/Users/tpac/brain`, code fingerprint
`8c33e3f67dc54c06`. The code at that source matched the fingerprint during
export. Its checkout was `codex/contract-host` at `c58f533`; main was in a
different worktree at `e35575c`. Do not infer the deployed code from a branch
name. No source checkout was edited or restarted.

The daemon's effective `s1e` was default, fingerprint `fd28b7b6321f`, with
Sonnet 4.6 / medium and a 111,323-character template. The exported package
includes the generated fields, Arc/Review/closure, native six tools, real
preamble, and limits (12,288 output tokens, five rounds). No gist is deployed.

| Static input | Current production | Frozen V3.2 |
|---|---:|---:|
| System characters, including generated ending | 117,427 | 99,477 |
| Gist characters | 0 | 5,391 |
| Tool JSON characters, same serialization | 26,800 | 26,831 |
| Preamble | Read all, operations in one call | Four worklists, then the required tool call |
| Revision API | Full-field replacement; content-only surgical edits | Value-or-swap across text fields; edge changes on revise |
| Ending | Next reply after tools is final | Inspect results and conditionally repair before final |

## Frozen comparison complete

All nine production encodes completed normally. The [whole-memory comparison](S1E-CURRENT-PRODUCTION-VS-V3-2-RESULTS-2026-09-11.md)
records gains, retained production strengths, limitations and the next release evidence.

Nine new production-package encodes: three repetitions, each with the same
three sequential five-pair windows used by V3.2. The nine completed V3.2 runs
are reused; V3.1 remains a saved reference. Independent repeats have separate
OS processes and isolated copies of the same closed seed. Inputs, source
trace IDs, dates, model settings, limits and initial factual sections match.
Subsequent windows carry that repetition's own actual writes and continuity.

This measures the **deployed encoder package under the shared replay**. It is
not a full-daemon A/B. The common branch mutation engine accepts production's
older API as well as the candidate's wider API; production is shown only its
native schemas. Catalog rendering, journal replay, source-less local dispatch
and absence of S1R/S2/answerer match the prior cell. Therefore this comparison
does not verify production provenance stamping, scheduling, MCP installation,
all shared-reader effects, or downstream retrieval. Nor can its package-level
differences be attributed to prompt wording alone.

Preflight passed with zero model calls. The first attempt could not import an
optional `jsonschema` dependency before executing checks. That unnecessary
added dependency was removed from the new wrapper, with the earlier manifest
preserved; no prompt/schema/model/source input or expected factual comparison
changed. Native tool bytes remain pinned to the verified source export.

Review uses the same whole-memory lens: details and first facts; arcs and
priorities; revisions and preservation; both voices/quotes; behavior/synthesis;
field value, titles, edges and residue; operations/errors and text/output cost.
Do not turn field counts, quotes or source_refs into quotas. This synthetic
design source is not sufficient evidence for all release dimensions.

## What the complete release must contain or settle

| Component | Current state | Remaining release work |
|---|---|---|
| New revise contract and write path | Committed on the encoder branch; exercised by saved candidate writes | Reconcile with current main; verify full replacement, exact swaps, edge updates and preserved fields through real dispatch |
| MCP schema and generic descriptions | New contract committed; reviewed generic descriptions live in frozen eval tools | Integrate the selected descriptions into their real owners, regenerate schemas, run the batch/schema gates, verify installed MCP surface |
| Selected template, gist and section cues | V3.2 frozen and evaluated; current-production comparison complete; V3.3 authored, reviewed and frozen 2026-09-11 with sanity and transfer cells launched (authoring review) | Apply the selected changes against current defaults without overwriting unrelated intervening edits; verify final assembled request |
| Worklist preamble and ending | Branch/eval support exists; latest finishing strategy is in the frozen package | Integrate exactly the selected behavior; review S2-shared closure effects before merge |
| Catalog/edge presentation and shared readers | Branch includes grouped edge rendering, relation age, noise filtering, community presentation and fetch-view changes | Inventory dependent S1/S2/recall effects and run relevant integration checks; the small replay is not this gate |
| Additional proposed mechanisms | Residue/edge IDs in catalog, header/nudge changes, non-tool continuation, trace round text, edge-repair mechanism and confidence/label change were separately gated | Explicitly settle what belongs in this release. “Complete package” does not silently make every parked experiment necessary |
| Release quality | Historical gold/LongMem, V3.1 two-source cell, V3.2 development cell and this production comparison | Compare protected capabilities on broader shared material and downstream recall before declaring overall improvement |
| Deployment | Nothing merged/deployed in this thread | Final concrete diff/review and Tom's launch decision; daemon code and installed MCP both need the correct deployment path |

## Artifacts

- Fixture/export: `eval/fixtures/s1e_production_comparison_2026-09-11/`.
- Results/pins/preflight: `eval/results/s1e_production_comparison_2026-09-11/`.
- Production arm: `6e5a73832c453a4639ca91a5bd83540e53e14c01e2f03c22132b37e022b8250a`.
- V3.2 arm: `15a9770ef2b21aa4939b127e3dcb51792c861ca0850130d0e03dfe7135821031`.
- [Protected capabilities and wider evidence](S1E-RELEASE-REGRESSION-MAP-2026-09-11.md).
- [V3.2 whole-memory review](S1E-V3-2-WHOLE-MEMORY-REVIEW-2026-09-11.md).
