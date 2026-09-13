# S1E pre-eval review — tools, instruction layers and context

The generic tool-description candidate passed the small mechanical probe.
The full prompt still contains two relevant procedural weaknesses: a clean
verdict can escape comparison with evidence, and two remaining instructions
describe closing immediately after a write. The tool change alone does not
remove either weakness. No full encoding cell was run in this review.

## Mechanical probe

Frozen old descriptions **24/24**; generic candidate **24/24**. Each of the
existing eight cases ran three times per arm, using Sonnet 4.6. Arms overlapped;
all calls within an arm ran sequentially. No brain instance, database writes,
tool dispatch, interviews, retries or corrective continuations were involved.

| Existing scenario | Old | New |
|---|---:|---:|
| Revision audit `reason` versus stored `reasoning` | 3/3 | 3/3 |
| Update `reasoning` and supply audit `reason` together | 3/3 | 3/3 |
| Merge intent selects `absorb` with correct direction | 3/3 | 3/3 |
| New-node edge uses `connect_to` | 3/3 | 3/3 |
| Emit an edge once | 3/3 | 3/3 |
| Preserve absorbed node's unique content | 3/3 | 3/3 |
| Relationship verb belongs in an edge, not an op name | 3/3 | 3/3 |
| Archive supplies the node ID | 3/3 | 3/3 |

All 48 generated tool calls also passed their own unchanged JSON schema.
Actual API input tokens: old **99,636**, new **86,580** (13.1% fewer).
Output tokens: old **3,665**, new **3,785**. No cache-read or cache-write tokens
were reported. These are probe usage totals, not whole-encoder savings or a
cost estimate.

The [driver](../eval/fixtures/s1e_tool_descriptions_2026-09-08/run_mechanics.py)
calls the existing `mcp_batch_probe.run_sample` and its existing graders.
The [pin](../eval/results/s1e_tool_mechanics_2026-09-08/manifest.json)
records its source hash, both exact tool definitions and all scenario text.
Each `old|new/<scenario>/repeatN/` contains the exact API request, response and
grade. [Results](../eval/results/s1e_tool_mechanics_2026-09-08/results.json)
and [independent schema review](../eval/results/s1e_tool_mechanics_2026-09-08/schema_review.json)
are the complete ledger for these numbers.

This is a compatibility signal, not evidence of improved coverage. It forces
one `brain_batch` call in a short context. It does not test automatic tool
selection, reads, revise_batch, all six tools, repeated encoding, journal
deferral or 110K-character inputs. In particular, the existing edge cases
exercise new-to-existing edges; they are not a separate sibling-title
resolution test. Both arms reaching the ceiling limits comparative claims.
The production-shaped `mcp_schema_gate.py` remains pending and requires explicit
candidate injection; its current CLI uses the active S2 tool definitions.

## Review of the assembled instruction layers

Inspected the actual frozen V3+cues system, its pre-timeline gist, generated
field reference, Arc/Review, ending strategy, Finishing and all candidate
tool-description edits together. A tool-only comparison keeps all other layers
fixed. API field order must not be described as the model's private ordering;
the earlier [journal/tools audit](S1E-JOURNAL-TOOLS-AUDIT-2026-09-08.md) records
the placement evidence and its limits.

### 1. Verification still inherits the plan's blind spots

The gist asks for field verdicts (`stale`, `clean`, `unread`) and later compares
stale fields/new nodes with successful writes. V3's final strategy also compares
successful writes with intended changes. Neither comparison requires the
encoder to exhibit what an allegedly clean old assertion means against the
new evidence. A mistaken-clean field falls out of the worklist and can pass
that final check untouched.

This interpretation is grounded in the completed sanity captures, not a new
interview: V3+cues repeat 1, window 3 marked the poster's title and reasoning
clean despite the new printing report. Both remained unchanged. The
[saved result](../eval/results/s1e_guide_sanity_2026-09-08/v3_titles/repeat1/window3/result.json)
contains the lists, actual operations and final reply; the
[hand ledger](../eval/results/s1e_guide_sanity_2026-09-08/ADJUDICATION.md)
carries the per-item assessment.

The existing Mira example gets its principal stale verdicts right. It
demonstrates the desired answer but does not demonstrate the comparison that
would catch a misleading clean verdict. The next guide revision should revise
that example and the existing targets procedure: **old assertion → new evidence
→ resulting claim/verdict**, including a clean claim whose unchanged scope is
justified. The final comparison should revisit resulting claims against the
evidence, including claims initially marked clean. This targets the named
mistaken-clean verification mechanism; it is not a request for another reminder
to be thorough. No new guide text was authored here.

### 2. Repair-before-close competes with write-then-close

V3 Cadence permits another write for a missed field or needed repair, and the
ending strategy says to repair supported changes before closing. But the gist
still says **“the reply after the write is the close”**, and the final shared
Finishing block says **“the write's results by the final reply.”** The latter
also correctly defines termination as the first reply with no tool call.

The new tools remove their own one-LLM-round advice and tell the caller to
inspect partial outcomes. They cannot remove this residual conflict in other
layers. The old sanity cell's successful missing-reason repair proves extra
writes are possible; the text conflict is a plausible pressure, not a measured
cause of every missed revision. V3 repeat 3 window 3 explicitly deferred a
known cleanup to a later touch, as documented in the journal/tools audit.

A coherent next guide needs a single sequence: read if necessary, write,
inspect, repair if necessary, then close. Revise the existing gist/cadence/
ending sentences together rather than add a third account of the sequence.
Changes to the shared Finishing contract remain Tom's decision; no shared
runtime text was changed in this review.

### 3. Smaller contract wording issues remain visible

| Topic | Combined-layer finding | Consequence |
|---|---|---|
| Swaps | Actions says “Every field” takes a value or swap; the generated field reference also generalizes. New tool prose correctly limits swaps to writable text fields and gives other fields bare values. | Narrow the existing broad sentence in a future guide/shared-field revision. Schema mechanics are unchanged. |
| Thought | Tools describe an optional, independently revisable interpretation. V3 and the generated field reference still say most nodes need/carry none. | Removing frequency advice from tools did not remove it from the whole request. This is framing, not a hard schema contradiction. |
| Source references | V3 explicitly makes them selective; gist conditions them on the moment being part of the meaning; new tools allow omission. | No universal refs quota was introduced. Replacement/omission/clear semantics agree. |
| Reads | New tools allow IDs from outside the current result set and describe rich versus bounded views. V3/gist already request missing bodies and edge surroundings. | Wording is compatible; this mechanical probe does not establish that missing-node reads will execute. |
| Fields and results | Audit reason, stored reasoning, exact IDs, sibling titles, partial edge outcomes and reference replacement retain consistent mechanics. | No newly introduced incompatibility found in this review; shared-source promotion still needs wider caller checks. |

## Corrected context for the next cell

Authored [context v2](../eval/fixtures/s1e_guide_context_v2_2026-09-08/contrasts_three_windows.json)
and a [versioned runner](../eval/fixtures/s1e_guide_context_v2_2026-09-08/run_sequence.py).
The original fixture, runner, frozen arms and completed captures are intact.

- Each `<other>` now has `speaker="Oren"`. Dialogue and actions are unchanged;
  the fixture no longer asks the model to infer the speaker from a catalog
  containing both Oren's memories and Tom-related background.
- A journal note first produced in the September 8 window renders `since 09-08`,
  even if the real test runs on another date. The adapter changes only that
  displayed date. The real JournalBinding still harvests, counts, resolves and
  pins notes; database transaction timestamps remain real.
- All 12 turns, their 4/3/5 grouping, seed facts, initial residue and predeclared
  criteria are byte-equivalent as parsed values. This remains the same synthetic
  sequence with corrected context, not a new held-out corpus.

The [offline review](../eval/fixtures/s1e_guide_context_v2_2026-09-08/offline_review.json)
used a real journal in an isolated copy of the existing closed eval baseline:
first open ×1 on September 8, repeated open ×2 still since September 8, a new
subject since September 11, then resolution of one subject while the other
stays open. Stored note rows were unchanged by rendering. Runner compilation
and original/new artifact hashes also passed. No model encode was run.

[Fixture diff](../eval/fixtures/s1e_guide_context_v2_2026-09-08/fixture.diff),
[runner diff](../eval/fixtures/s1e_guide_context_v2_2026-09-08/runner.diff),
[context pin](../eval/fixtures/s1e_guide_context_v2_2026-09-08/manifest.json).
The new runner is a deterministic snapshot of the original with eight exact
substitutions, recorded by its author script. It still loads the frozen tools;
candidate-tool injection and the next cell's baseline/preflight pin remain
separate setup work. Every compared arm must receive the corrected context;
do not pool old-context samples into its denominator.

## Next decision

The tools-only variation is mechanically ready for a limited comparison while
holding V3 fixed. My preferred next prompt iteration is to repair the existing
targets/example and reconcile the existing stopping sentences, review and
freeze that distinct guide arm, then run the limited repeated cell before
expanding to the historical corpus and longmem. Tool and guide changes should
have separate control comparisons so a result can tell us which change helped.

No edits to shared tool definitions, production prompts, journal header/nudge,
closure, daemon configuration or live graph were made by these checks. Nothing
was registered, merged, restarted or deployed. Tracked HEAD remains `9a1727f`.
