# S1E guide v2 — behavioral evaluation

**First diagnostic checkpoint complete; full gold/longmem not run.** Tom approved evaluating frozen v2 while an Astra side agent
independently compresses it. He explicitly approved the existing Sonnet API
workflow, including transmission of v2 and selected conversation/eval fixtures,
after automatic network approval review requested that clarification. Nothing
is merged or deployed. This worktree remains at `9a1727f`.

## Candidates and order

V2 template SHA-256:
`4637785918cab686ec9abd29096def0691fb9aaf443b63d893736c4883551edb`.
Gist SHA-256:
`54cf9c77132ce3eac680f9703f71a612293b26117c6886048ba9f001634b3063`.
Astra's compact variants are separate, unevaluated candidates.

1. Capture composition: verify exact template, gist before timeline, lists-first
   preamble, model/settings, full tool schemas, and every request round.
2. Small behavioral diagnostic: retain field repair, preserve first disclosures,
   distinguish observation from interpretation, and maintain facts while a
   developing interpretation changes. Inspect successful writes and next-window
   state; a list by itself is not execution.
3. If the diagnostic supports proceeding, run the existing per-item gold cell
   and longmem comparison, with held-out cases. Existing production, prior best
   candidate, and guide v1 remain separate baselines. Historical scores are in
   the hand ledger and are not recalculated here.

No percentage of nodes carrying source_refs is a success criterion. References
are reviewed for helpful visibility and attribution when used.

## Historical field-repair smoke

Two `bb5b1ef4` repetitions use the existing `encoder_prompt_ab.py` F arm and
historical gold fixture. Writes are intercepted as in the original gold cell;
reads are served from an isolated copy. Captures include full tool schemas in
addition to the existing round payloads. The executed wrapper is preserved at
`eval/fixtures/s1e_guide_v2_2026-09-08/run_gold_smoke.py`
(original invocation: `/private/tmp/run-s1e-v2-gold.py`).

Outputs: `eval/results/s1e_guide_v2_2026-09-08/gold_bb5b1ef4/`.
The initial sandbox-only attempt failed to reach the API; its capture and error
log are retained as `capture_network_failed/` and `network_failed.log`. It is
not a behavioral trial. The approved network retry verified composition.

- Repeat 1: **VOID**, because it read `bb5b1ef4` from its post-capture state.
  Four rounds, two read calls. Do not interpret the scorer's displayed zero as
  a miss. Dump: `s1e-5076cdc2-17-F-run1.json`.
- Repeat 2: **2/2 surfaces by hand** (content and situation), one read call
  of other nodes, four rounds. The automatic scorer displays 1/2 because its
  stale-token rule rejects the retained number `47`; the hand ledger already
  documents this false negative. The emitted situation explicitly says the
  47 ids are a mixed class of corrupted emissions and genuine hard-deletes.
  Its content adds the observed split and states that the full split remains
  unsized. Dump: `s1e-5076cdc2-17-F-run2.json`.

The scoreable sample is only one repeat. One procedural mismatch remains:
its first `targets` list called the target's situation clean, yet the actual
write repaired it. The operation is the score, while the list discrepancy
shows that the final write was not simply executing the first list.

## Fresh sequential diagnostic — criteria fixed before calls

Fixture: `eval/fixtures/s1e_guide_v2_2026-09-08/discovery_sequence.json`.
Runner: `eval/s1e_guide_v2_sequence_probe.py`.
Outputs: `eval/results/s1e_guide_v2_2026-09-08/sequence/`.

Nadia's fictional family archive has first-disclosure possession and schedule
facts, an ownership/loan contrast, a corrected multi-field plan, three distinct
choices that can support a scoped interpretation, and an independently measured
assistant finding. The first window includes an explicitly constructed no-mint
residue note. The second gives counterevidence, a new practice, and an appointment
correction. Neither scene appears in the candidate curriculum.

The probe pins one starting isolated database, then runs guide v1 and v2 on
separate sequential copies. Each arm's second window receives its actual
persisted nodes and harvested journal/arc from its first. Real branch dispatch
applies writes; per-operation results and before/after nodes are retained.
No synthetic success replies or continuation nudges are inserted.

The catalog combines the same frozen historical background with the fictional
nodes rendered by the branch's normal catalog function. The timeline is a
constructed fixture. This tests the encoder under input load and continuity;
it is not an S1R/S2 integration test or a general benchmark score. Background
reads can still expose historical drift; adjudication concerns the new,
controlled facts and seeded claims. Shared runtime gates remain unchanged.

### Observed results — one sequence per guide, not an aggregate score

Both sequences completed. All model calls used `claude-sonnet-4-6`, effort
`medium`, the branch runner, unchanged shared suffix, and complete tool schema
captures. Guide v1 took 3 then 2 rounds; v2 took 2 then 2. Every requested
operation succeeded, including embedded edges. No reads or list-only stop
occurred in these four runs. This shows application to the seeded memories,
not merely valid-looking emitted operations.

| Prior criterion | Guide v1 | Frozen v2 |
|---|---|---|
| French call retained despite initial skip residue | Yes: Luc, Wednesday 07:10, September 16 | Yes, same details |
| Owned recorder/location and temporary loan distinguished | Yes; no duplicate when Roland is mentioned again | Yes; no duplicate when Roland is mentioned again; return-date reasoning wrong |
| Corrected audio plan repaired across title/content/situation/reasoning | Yes; 46 tapes, Farah's index, and planned state preserved | Yes; same still-true details preserved |
| Measured assistant finding retained as diagnosis, not completed fix | Yes | Yes, but added unsupported implementation detail |
| Broader unspoken interpretation across recipe/video/audio | No independent cross-medium interpretation in window 1; encoded ambient-audio rule | No independent cross-medium interpretation in window 1; encoded ambient-audio pattern from two instances |
| Counterevidence and fresh Monday work practice | Retained practice and new provisional insight | Retained practice and explicit framing; revised existing pattern's reasoning to distinguish observed choices from unconfirmed cause |
| Appointment revised to Thursday 07:40 from September 17 | Yes, preserving Luc | Yes, including revised reasoning and event time |
| Generic advice stays out of Nadia's biography | Yes; not minted | Yes; not minted |

The direct audio preference is useful in both arms. Neither first window
discovered the wider link to the recipe choice; the three-domain inference
remains unproven. No penalty is attached to the absence of a `thought` field:
v2's reasoning-only update is a valid maintenance operation. Its more cautious
explanation after counterevidence is a promising observed behavior, not proof
that it would generalize across people or domains.

The first fact criterion is an easy case for both: the schedule is explicitly
recurring, and the equipment belongs to the active topic. V2's explanation
still says the structured recurrence makes it durable. This does **not** prove
that its first-disclosure rule overcomes the longmem single-session gate.
A truly incidental, one-time fact needs its own held-out contrast before
claiming that mechanism is fixed.

### Failures and limits that argue for diagnosis before the large cell

1. **Unfounded precision in v2 becomes next-window authority.** Window 1's
   `a36e6f35` says Friday is `2026-09-12`; that date is Saturday, while the
   coming Friday is September 11. `f01e3bdb` turns “this weekend” into September
   8, a Tuesday. Window 2 then treats the invented September 8 return date as
   past. The error travels from a node into residue instead of being checked.
2. **V2's residue retirement conflicts with its prose.** Its actual window-2
   final reply emits `resolved · preview double-trim fix · still open ...`
   and `resolved · Zoom H5 return · ... no confirmation ... will stay open`.
   These are model outputs, not a renderer interpretation. On subsequent
   contract review, `resolved` retires a handled NOTE; it does not necessarily
   assert that the underlying job is complete. Calling this proven false task
   completion, as the initial summary did, was too strong. Clearing a redundant
   note whose pending work is already stored can be legitimate. The loan line's
   “will stay open” conflicts with retiring that same note, and the invented
   return date remains a separate factual failure. The harvested
   continuity also retains the old resolved Wednesday appointment note while
   the node and newer arc correctly say Thursday. The shared journal contract
   remains unchanged; no mechanism fix was smuggled into this probe.
3. **V2 enriches a measured finding beyond the evidence.** `d05f7082` adds
   “once in its configuration and once again in the export call”. The fixture
   establishes a double trim and exact offsets, but never identifies those
   two code locations. Detail coverage cannot earn credit for invented detail.
   A new edge also calls professional editing Nadia's “trained instincts”;
   routine practice was stated, that psychological explanation was not.
4. **The baseline has faults too.** Guide v1 assigns an exact September 1
   purchase date to “last week”, drops Farah's new Friday availability, and
   initially attributes the deferred fix to Tom in residue although the
   fictional counterpart is Nadia. These are not evidence that all new
   defects were caused by v2; this is one sequence each, without repetitions.

Raw evidence, including successful per-operation results:

- `sequence/v1/window{1,2}/{calls.json,result.json,nodes_after.json,next_continuity.txt}`
- `sequence/v2/window{1,2}/{calls.json,result.json,nodes_after.json,next_continuity.txt}`
- `sequence/{v1,v2}/manifest.json`, `system.txt`, `tools.json`, and each
  `window*/round*.json` pin actual inputs and settings.

Fixture SHA-256: `c2121e4883195e67e53f433ba414dd690fc7ee5482055aa8530427cb7c3f9ba5`.
Both arms started from the same closed snapshot and same seeded ids:
`11cf7d01` audio plan, `1f6b053d` recipe choice, `3e1022d9` video choice.
The snapshot path is in `sequence/baseline.json`; it is an isolated scratch
database, never the live daemon database. The fixture's people and claims
are fictional and were not encoded into the real brain.

**Checkpoint:** v2 has demonstrated fact retention and all-field repair here,
plus an evidence-scoped update after counterevidence. It has not established
an overall win or fixed the original longmem failure. The useful next
discussion is the evidence boundary across facts, interpretations and
residue, alongside Astra's shorter shapes—not adding case-specific dates or
another instruction for each observed miss. Full gold and longmem remain to
be run after this diagnostic review; no pass/ship gate has been declared.

## Size comparison correction

Production's frozen template plus no gist: **110,452 characters**.
V2 template plus gist: **124,298 characters**.
Net: **+13,846 (+12.5%)**. The earlier +7.1% compared only templates against
guide v1, not production. Shared suffix and variable catalog/timeline are excluded.
