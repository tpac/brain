# Reverse pass — the re-authored covered-turn example (2026-09-14)

The V3.8 `example` arm's worked window was re-authored on a science domain after PROBES.md § "Caveat
that governs the read" ruled the first draft contaminated (it was written from the item it was later
probed on). Method: the blind derivation below was taken from the example's shape alone — node count,
ops, degree, fields, moves, absences — and only then checked against the class the example is meant to
teach (AUDIT-COVERED-TURN-2026-09-14.md). Sizes: 8,768 → 10,516 chars (+19.9%). `author.py` and
`static_checks.py template_example.md gist_full.md` both pass on this checkout.

## (a) Blind derivation — what the shape teaches

**Input depicted.** One catalog entry in full (7 rendered fields + one `Edges` line) carrying the tag
`[encoded(me, turn 5)]`; a second node present ONLY as that edge line, body invisible. A five-turn
window: two turns three months old and covered, two covered "just now", one uncovered and routine.
A `<provenance>encoded(me, turn 5): …</provenance>` line sits on the older covering run's last turn and
nowhere else — so the reader can see, on the page, that a later run covered turns 7–8 and kept nothing.

**Counts.**

| dimension | value |
|---|---|
| nodes minted | 1 |
| ops in the write | 3 (2 `revise`, 1 `remember`), one `brain_batch` |
| read rounds | 1 (`get_nodes`, one id) |
| edges written | 3 — one `why` swap on a revise, two new whys on the remember |
| degree of the new node | 2 |
| fields on revise 1 (`4e7b1a92`) | title, content, situation, reasoning, their_raw_quote, event_time, thought, source_refs, connect_to (9) |
| fields on revise 2 (`c1d8f350`) | title, content (2) |
| fields on the remember | type, title, content, situation, question, reasoning, their_raw_quote, event_time, connect_to ×2 |
| Bads | 3, all labelled, all before the lists |
| new-why lengths | 149 and 158 chars; the swapped why resolves to 167 (all inside 120–180) |
| `source_refs` | 2 ids, both copied from depicted `trace=` attributes |

**Field forms shown, in one batch:** a span swap (`title`, `content`, and a `why` on the edge line), a
whole-value replacement where the claim restructured (`situation`, `reasoning`, and the quote —
replaced entire, because a span swap inside someone's sentence would forge it), a bare non-text value
(`event_time`), and two fields the node did not hold before (`thought`, `source_refs`).

**Moves shown.** Read a covered turn against the catalog entry that covers it, and treat the gap as
mine. Treat two dated statements by one speaker as a changed value (swap, old value dated in prose)
rather than an in-window contradiction (no `open` minted). Fetch an edge-only node whose value is
DERIVED from the one being changed, then move the derived number with it (3 min → 2 min, a quarter of
the fitted time constant). Replace a stale quote whole. Move a stale edge `why` through `connect_to`
on the same revise, `old` copied from the depicted edge line. Flag the covered scene with
`source_refs`. Give a new node two honest edges. Keep a `thought` that is true beside a full revise —
the counter-model to the Bad that writes only the thought.

**What it never shows.** No dated successor + `supersedes` (the history branch stays with the temporal
section). No `open`, no `archive`/`disconnect`/`absorb`, no `recall_batch`, no returned-results block,
no second write round or post-write repair, no failed op. No `my_raw_quote`, `emotion`, `confidence`,
`locked`. No node minted for the superseded value. One read round, one write, close.

## (b) Check against intent — flavours C / A / T / R

| element | teaches | flavour |
|---|---|---|
| The covered "just now" turns with no `encoded(me, …)` naming them | the flag records a run; the catalog records memory; empty coverage is invisible unless the reader compares (audit L39, L43) | **C** |
| Bad 1 — "turns 4–8 are covered; turn 9 is routine; `new: none`" | coverage is not a skip ground; the value on the page has no node (audit L273) | **C** |
| Bad 2 — an `open` "12 minutes (turn 4) vs 8 minutes (turn 8), which is correct?" | rule 3 vs rule 4 discriminator when the OLD TEXT is on the page as a covered turn — the audit's finding 1, the exact misfile V3.6 made | **T** (bait supplied by **C**) |
| Bad 3 — `thought` only, "the plume came up again; the single-exponential read is stable" | the confirming covered turn (7) is read, the changing one (8) is not; the anchoring failure with its own bait depicted (audit L79, L111) | **A** |
| `changes` line naming "no encoded(me, …) names it — so the change is mine now" | the reverse pass in one sentence: entry → the covered turns it came from | **C** |
| `targets` marking every surface stale incl. `event_time`, `question clean`, `why→c1d8f350 stale` | the half-revise that a later run reads as clean is what locks the miss in (audit L265) | **A** |
| The swap with "(12 minutes at 18 °C from the 2026-01-15 fit)" in content only | routine parameter change, validity kept in prose, the title asserting only the live value | **T** |
| `fetch` + the derived value following (τ/4) | a dependent whose number is a function of the changed one; edge-only body must be read before it is written | **T**, sweep discipline |
| Closing — "that run's silence is not a ruling, and neither is an arc line that calls the stretch transactional" | a no-mint verdict travels in the Arc, and the next run quotes it (audit finding 2) | **R** |
| The `thought` kept beside a full revise, plus the closing "true and still not the move" | the correct shape of the move Bad 3 truncates | **A** |

Every flavour the audit names for the example carrier (C, A, T, R) has at least one element; **T** and
**C** each have two, **A** three (one Bad, one list line, one counter-model).

## (c) Ledger box row

| law | how this example checks it |
|---|---|
| **T1 / A2** | three labelled Bads against one worked Good, each Bad a move actually made in the measured runs |
| **T4** | one contradiction found and marked below (optionality, not damage) |
| **A1** | the class's behaviour gets worked material, not a rule clause — the audit's finding 4 (no worked window depicted a covered turn) |
| **A3** | placed after Mira's later window, the latest and most concrete example of revising recently encoded nodes |
| **A4** | shape leaks audited on purpose: batch size 3 and new-node degree 2 are above the set's quiet defaults (~1 node, degree ~1.5) and are honest here |
| **A5** | every id (`4e7b1a92`, `c1d8f350`) and every `trace=` is introduced by this example's own excerpt; no placeholders |
| **A6** | a science domain (flume tracer decay, fitted time constant, sampling cadence) — widens the example set away from its personal-logistics and engineering nouns |
| **A7** | the whole reason for the re-authoring: no evaluation item was read (see (e)) |
| **A10 / E11** | the widest revise in the prompt by field count (9), and the only one carrying `source_refs`; `question` is exercised on the remember where a real asking exists, and marked `clean` on the revise where nothing falsified it |
| **E12** | every swap `old` is visible before its op — the title and content spans in the catalog entry, the dependent's content in the `get_nodes` return, the `why` old span on the depicted `Edges` line (enforced by `static_checks.py` check 3) |
| **E13** | the `sweep:` line names two ids against two revises — the half-done case is what the `targets` verdicts and the closing paragraph are written to catch |
| **E15** | new whys 149 / 158 chars and the swapped why resolving to 167, all inside the stated 120–180 band; `source_refs` = 2, inside the stated 1–3 |
| **E17** | inline history sits only in `content`, as a validity interval ("12 minutes at 18 °C from the 2026-01-15 fit") — the legitimate case the law names; the title asserts the live value alone |
| **R-rows (docs/challenges)** | `structure.md` row on the `encoded="true"` gloss (covered turns keep FULL text) — the depiction matches that policy: covered turns render whole text, and the uncovered turn is the only `false` one |

## (d) Contradictions with existing prompt text

1. **L39 gloss vs this example (real, unmarked in the example arm).** The V3.6 gloss the `example` arm
   leaves in place says I reread covered text "for cross-turn patterns and contradictions, **not fresh
   atoms**". This example mints a fresh atom (the demo run) and a changed value from covered turns 7–8.
   By A3 the example wins where they meet, and by B2 it is far later in the document — but the
   contradiction is unmarked, and it is exactly the sentence the `gloss` arm rewrites. Read as
   optionality only if both arms ship; the example arm alone leaves a standing conflict.
2. **L267 Skip, "covered restatements" (optionality, not damage).** Turn 7 *is* a covered restatement,
   and the example treats it correctly — read, not minted, and its confirming force named as the bait
   for Bad 3. So the rule and the example agree on turn 7 while Bad 1 pushes against the same sentence
   for turn 8. Marked as optionality per T4.
3. **`source_refs` before-state is undepictable.** L263 gives `source_refs` REPLACE semantics, and the
   encoder's catalog render carries no refs line at all (checked this session: no `source_refs` in
   `servers/scales/s1/encoder_view.py` or `encode_contract.py`). The depiction is therefore not lossier
   than production — E12 is satisfied — but this is the one field in the batch whose prior value the
   reader cannot see before it is replaced. Recorded as deliberate per A10.5.
4. **Two new type strings.** `measurement` and `protocol` are not in the L101 common-shapes list; that
   list is explicitly open ("material, not a closed menu"), so this exercises the openness rather than
   contradicting it, and it adds two non-personal shapes to what the examples model.
5. **Turn coordinates.** Bad 2 shows "(turn 4) vs (turn 8)" as the wrong thing to STORE, while the
   encoder's own reply lines say "turn 8" as working prose — the same split Mira's lists and L45 model.
   Not a defect; noted so a later reader does not "fix" it.

## (e) Decontamination statement

No evaluation item was read while authoring this example: nothing under `~/AgentsContext/eval-corpus`
or `eval/longmem/data` was opened in this session, and the example was written from the audit's class
description and the V3.6 template's register alone. It differs from a "recurring weekly time changes"
item on every surface a retrieval or a judge could latch onto. **Domain:** a laboratory flume tracer
study — a fitted decay constant and the sampling cadence derived from it — instead of a personal
schedule. **Shape-surface:** the changed quantity is a fitted physical parameter with units whose
change has a stated physical cause (a tank temperature moved from 18 °C to 24 °C), not a preference
moved by choice; the dependent value is derived by an arithmetic rule stated in the edge (interval =
τ/4), not by an offset from a clock time; the incidental new fact is a teaching slot sized by the new
constant, not a meeting placed before an evening commitment; the window's dates are January → April,
not February → May; the conversation date is a Monday and the only named weekday belongs to a future
event. **Vocabulary:** e-folding time, single exponential, rhodamine plume, autosampler, least-squares
fit, set point, tracer run — no time-of-day, no weekday recurrence, no reminder, no alert, no calendar
or task-manager noun anywhere in the example. What is deliberately preserved from the contaminated
draft is the purpose shape only: a covered turn whose changed value no run kept, a derived dependent
reachable only through an edge, three Bads, the per-surface verdicts, and the routine-change swap.
