# S1E prompt — reorganization audit (v-next.7 → v-next.8 candidates)

2026-09-02. Forced full read of `eval/candidate_prompts/s1e_vnext7_wip.md`
(1,660 lines, 114,598 chars: 20 fenced examples = 42,964 chars / 37%; prose
= 71,634 chars). Method: id:71eeff20 — read everything, one audit row per
section, no edits before the artifact exists. Challenge boxes run from
`docs/S1E-CHECKLIST.md`.

## 1. Section map and dependency order (the reader is stateless)

| # | section (lines) | teaches | depends on | forward refs / seams |
|---|---|---|---|---|
| 1 | Opening (1–7) | identity, two registers, many-nodes+edges | — | names remember/revise/connect before any is defined (weight order, id:99810ad5 — deliberate) |
| 2 | What I Receive (9–62) | payload legend: continuity, catalog, timeline, scout_legend, reading order, provenance | node fields (§5), ops (§8) | embeds BEHAVIOR in a legend: anti-twin ×2, "revise if it shifted", "no edge without a why"; timeline sample + turn-coordinate Bad/Good |
| 3 | Scout (64–122) | posture + handoff for ONE facts scout | §2 scout_legend | 60 lines for one Haiku scout; "see Reading the conversation → Emerging patterns" |
| 4 | Reading the conversation (124–223) | 4 correction flavors, emerging patterns, atoms, third-party floor, trace attr | ops (§8): revise_batch, corrects, supersedes, partially_resolves; types (§5): open; Temporal §7 validity intervals | WHEN to revise is taught ~650 lines before HOW; flavor 3 forward-refs Temporal→Validity intervals |
| 5 | Nodes (225–461) | anatomy/5 surfaces, required fields, type, thought, open fields, atomization, source_refs, Flat→Rich | contract list (appended LAST at runtime) | "content replaced on revise / patch — see Actions"; Flat→Rich says see canonical batch below |
| 6 | Edges (463–542) | relation+description/why, Bad/Good, reachability, 120–180 band, recall-work verbs | connect_to (tool desc) | vocabulary delegated to tool description (E10 dereference) |
| 7 | Temporal (544–777) | event_time, time_anchor, sequence, parents, validity intervals, ACL worked example | ops (§8) — example is a full remember batch | "the 97b1f24e shape" (L619) points at an example 600 lines away by id; validity-intervals wording still says "old value preserved in prose" |
| 8 | Actions (779–922) | two reads, remember/revise/connect, every-surface rule, content_edits, E17 rule, REPLACE semantics, batch choice, skip, gate, six defaults | §5, §6 | the revise doctrine sits at 48% depth, mid-bullet, 48 lines long |
| 9 | Cadence + examples (924–1439) | rounds, `sweep:` line, expansive, connect_to target forms, canonical batch, detail/meaning pair, revise ladder, sweep example | everything | mixes PROCESS (rounds/close), TOOL MECHANICS (target forms — belong with §8), and EXAMPLES |
| 10 | Identity examples (1441–1652) | 7 identity/hot-register examples | — | "Sam" ×30 (fictional counterpart, fine) |
| 11 | Closure / What this is (1654–1660) | run close order; identity coda | runtime blocks | — |

**The order problem, stated once:** the reader meets *when to revise* (§4)
before *what a node is* (§5), and *how to revise* (§8) 550 lines after the
fields it revises were defined. Tool mechanics (connect_to target forms) sit
inside the examples section. The payload then adds ~1,400 lines of catalog
between the instructions and the timeline — and the payload carries no
instruction of any kind after the catalog (`<continuity>` → `<node_catalog>` →
`<scout_legend>` → `<timeline>` → end).

## 2. Challenge results on v-next.7

| box | result |
|---|---|
| E1 contradiction | **DEAD-CONFLICT:** Temporal→Validity intervals (L616–620) still says a routine update is a patch with "old value preserved in prose" — the pre-E17 house style; Actions (L827–832) says inline history is for `content` and only where load-bearing. Fix: one sentence, Validity intervals defers to E17. |
| E7 redundancy | anti-twin rule ×4 (L13 twice, L794, L1004); two-reads ×2 (Actions, Cadence); expansive ×3 (gate, defaults, Cadence); sweep discipline ×4 (flavor 3, Actions, sweep line, sweep example); id-copy rule ×3 (L13, target forms, canonical demonstrates). Skip ×2 already marked deliberate. Reorg collapses each to one owner + one deliberate echo. |
| E8 grammar | clean: 0 angle placeholders, 30 curly; 0 ```json fences; 0 "the assistant"; 0 "Operator:" labels. |
| E10 MCP | contract `situation` says "One sentence"; prompt examples average 130 chars, 4/28 multi-sentence; production averages **200 chars**, 23% carry status words. The contract line is the tighter authority (injected last) and is ignored — either the prompt teaches the one-sentence trigger or the contract line changes. |
| E11 rationale census (revise surfaces, v7) | title 2, situation 2, question 1, edge description 2 — balanced vs v6's 3:1. Measured: still under-applied (situation ~50%, edge 0/6). Rationale balance was necessary, not sufficient. |
| E12 before-state | sweep example excerpt renders situation+question for 97b1f24e ✓; auth-rewrite excerpt renders no `situation:` line while production renders it. Half-fixed. |
| E13 graded checks | `sweep:` line lists surfaces ✓. |
| E15 numbers | edge whys in examples n=41: 4 under 80 (invisible by the prompt's own rule), 4 at 80–120, 31 in band, 2 over 180 (max 263). Example titles mean 60 chars vs production mean **109**. |
| A5 leakage | placeholders are curly, one grammar ✓. |
| D5 grounding | "the 97b1f24e shape" (L619) is an internal referent 600 lines early. |

**Measured behavior that no box predicted:** 30 A/B runs — the edge-description
surface was repaired **0/6** by both prompts; each prompt whiffs an entire target
(no revise at all) in ~1 of 6 runs; the run-44 twin is minted 6/6. The revise
REFLEX fires (~90% of runs revise the target); the SWEEP under-applies. The
reflex is taught first, richly (four flavors at L137); the sweep is taught once,
late (L810). Pedagogy matches the measurement.

## 3. Node shape in production (last 200 `encoder:*` nodes, 2026-09-01/02)

| surface | measured | judgment against recall (5 embedded surfaces, id:eb011c95; title+content primary, id:d76cacac) |
|---|---|---|
| title | mean 109 chars (max 164); 25% carry a date, 18% a commit sha, 16% a status word (DONE/SHIPPED/pending/still) | dates and shas are findable handles — keep. Status words in a title are claims that rot (the exact class the sweep repairs). No title-length norm exists in the prompt; examples average 60. |
| situation | 100% present, 98% open with "When" (trigger register — taught, working); mean 200 chars; **23% carry a status word**, 16% a sha | the prompt's own line "when the node has a work-state, the situation carries it" is read as license for verdicts ("it is DONE", "repeats are pending"). A verdict in `situation` is a stale claim waiting — the highest-leverage shape fix: identifiers yes, status no. |
| question / reasoning | 100% / 100% | saturated by prompt + contract; nothing to gain here. |
| thought | 2% | prompt says most nodes carry none — consistent. |
| emotion_label | 0% non-neutral in 200 | total null (also Gate-4). A taught field with zero behavior fails E5 — drop the teaching or move it into a canonical example op. |
| their/my_raw_quote | 28% / 34% | derivation-gated, plausible. |
| event_time | 86% | high — dated nodes dominate (finding/decision/milestone). |
| content | mean 1,304 chars; 40% carry history markers | content is the record — correct home for history (E17). |
| edges | 4.8 per node; **35% are `community_member`/`co_anchored`** (system-written), 14% of descriptions generic ("shared episodic anchor") | recall-surface pollution outside the prompt's reach — S2's, not the encoder's. Encoder relations are specific (grounds, extends, instantiates, supersedes…), 0 `related_to`. |
| type mix | finding 32%, decision 13%, milestone 11%, correction 8% | `milestone` is a status-shaped type; fine as a dated event if the title states the event, not the current state. |

## 4. Proposed order (dependency-ordered for a cold reader)

1. **Who I am** — opening, compressed (identity, two registers, nodes+edges, weight order).
2. **What I receive** — pure legend: the three blocks, timeline sample, provenance/actions semantics, scout notes (Scout section folded to ~12 lines). Behavior moved out.
3. **What a memory is** — Nodes (surfaces, required fields, type, thought, open fields, emotion, atomization, source_refs, Flat→Rich), Edges, Temporal (field-level rules only; the ACL example moves to the examples chapter).
4. **What I do** — Actions: two reads; remember / revise / connect; **the every-surface rule and E17 as the revise definition**; connect_to target forms (moved here from Cadence); batch choice; skip; gate; six defaults.
5. **How I read** — corrections (4 flavors), emerging patterns, atoms, third-party floor — now able to say "revise (Actions)" to a reader who has met revise.
6. **How a run goes** — rounds, `sweep:` line, close order.
7. **Worked examples** — ACL temporal, canonical batch, detail/meaning, revise ladder, sweep.
8. **Identity examples** (unchanged).
9. **Coda.**

**Placement risk (B4, T3), named for the A/B to measure:** moving the correction
flavors from L137 to ~L900 could weaken the reflex that currently fires. The
counter-bet: a reader that already holds the data model and the ops reads the
flavors as *applications*, not as the first thing it hears about revising. The
distilled version halves the distance; the gist before the timeline puts the
sweep rule at the recency position regardless of order.

## 5. The pre-timeline gist (payload, not template)

`_build_user_content` emits `<continuity>`, `<node_catalog>`, `<scout_legend>`,
`<timeline>`. A short free-text block between the legend and the timeline
(constant in `encode_contract.py`, ~700 chars) restates the operating rules at
the recency position: what earns a node (facts, decisions, corrections,
quotes, dates → event_time, opens); read each catalog node as a set of live
claims; revise EVERY surface a falsified claim sits in — title, content,
situation, question, edge descriptions; an `open` the window answered changes
type; don't mint what the catalog holds by id; nothing I write inherits turn
numbers. The A/B harness needs a `--gist` splice for arm F, since F sends the
capture verbatim.

## 6. Third candidate — improvements beyond reorg + distill

- `situation` = trigger, never a verdict (§3): one rule line + the canonical
  batch's `finding` situation kept identifier-rich and status-free.
- Validity intervals E1 fix (§2).
- Title norm: one line — a title states the claim, not its current status;
  ≤ ~100 chars; a date when the claim is dated.
- Emotion pair: either an example op that sets it (canonical `moment` already
  does — production still 0%) or drop the prose paragraph. Recommend: keep the
  example, cut the paragraph (E5).
- The revise definition gets the four-rung ladder INLINE as one sentence each
  (patch one claim / repair reach / walk one node's surfaces / sweep every
  node one event falsified), so the ladder is met where revise is defined, not
  400 lines later.
- Edge-description repair gets its own one-line consequence at the revise
  definition — the surface measured at 0/6.

## 7. Drafts, reviews, and what changed (2026-09-02)

Three candidates in `eval/candidate_prompts/` (all 20 example fences byte-identical to v-next.7, machine-verified):

| file | chars | what it is |
|---|---|---|
| `s1e_vnext8_reorg.md` | 114,729 | v7 text in the §4 order; four seam edits only (E1 fix in Validity intervals, D5 forward-ref by id → by name, two cross-reference rewrites) |
| `s1e_vnext8_distilled.md` | 102,660 | reorg + prose compressed (71.6K → ~60K prose; examples, Bad/Good pairs, FLAT/RICH templates verbatim) |
| `s1e_vnext8_ideas.md` | 104,155 | distilled + §6 improvements: title norm, situation trigger-not-progress rule, `evolution_status` defined in prose, emotion paragraph → one line pointing at the `moment` example, revise definition carries the four-rung ladder and the edge-description consequence, `connect` defined with its in-place-rewrite semantics, same-topic qualifier on the same-batch test, inherited "two revises" count fixed to three |

**Gist:** `ENCODER_GIST` in `servers/scales/s1/encode_contract.py` (1,110 chars), emitted by `_build_user_content` between `<scout_legend>` and `<timeline>`; `eval/encoder_prompt_ab.py --gist` splices it into a frozen capture at the same position. One test caught a real collision — the first wording said "the timeline's `now=`", and the unstampable-timeline test forbids that substring in the body; reworded to "the conversation's date" (correct in the degraded case too). Payload tests 104/104; contract/encode/guardrail tier 513 passed.

**Reviews before any encoder spend:**
- *Information loss (Opus, v7 → distilled):* 20 drops, all restored (category_statement, the Edges-inline bullet, interpret/expand definitions, "any key / invent freely", timeline-id lookup, "nothing to sweep", history-blob constraint, `revise_batch` at point of use, render timestamp line, `event_time_range` start/end, S2 healer + lazy-promotion scope, cross-referencing, substrate keeps episodes, the general revise-don't-twin clause, "Nodes come first", "silently wipes", six minor clauses). Every numeric constraint survived. One flagged item kept deliberately: the E1 fix in Validity intervals — the reviewer read it as contradicting the yoga patch, but that patch's `(was twice a week from 2023-08-11)` is E17's named load-bearing exception; the sentence now says so.
- *Structure (Opus, reorg + ideas):* order verdict — better for a cold reader (ops before run shape and examples; anatomy before example nodes; flavor-3 pointers now resolve backward), at the price of "Reading the conversation" arriving after 200 lines of mechanics. **Load-bearing finding:** the title norm and the trigger-not-verdict rule as first drafted contradicted the worked examples (which re-title with dated events and put asker-state values in situations) — and examples outrank rules. Both narrowed to the node's own *progress verdicts* ('DONE', 'pending', 'awaiting review', 'next step is…'), which is exactly what the 200-node production census flagged and the examples never do. Also fixed from this review: `evolution_status` undefined in prose, "3-anchor bar" used before defined, `connect`'s rewrite semantics stated only where used, the title Bad/Good spliced between the timeline sample and its legend. Left as-is, noted: get_nodes-before-connect vs the sweep's edge-line patch (deliberate optionality, the example argues it); ladder stated at the definition and at the example (reinforcement).
- *Cold-read probes (Sonnet ×3: v7, distilled, ideas):* all eight rule questions answered correctly from every version, with line citations. **The probes cannot separate the candidates** — a cold reader retrieves every rule when asked; the measured defect is under-application while reading a ~280K-char payload. Only encoder runs separate them. All three probes named the same friction: the revise rule's substance is split between its definition and worked examples hundreds of lines later, with the edge-description repair the least anchored piece (still named by the ideas probe after the inline clause was added).

**Next:** small readability run — one run each on the canonical item for v41+gist, distilled, ideas, ideas+gist, plus run-44 on ideas+gist — read the ops for shape before any 3-sample A/B.

## 8. Round-2 measurement (2026-09-02) — the gist is the lever, the template is not

32 encoder runs (arm F, frozen payloads, isolated brain): v41+gist and ideas+gist × 5 items × 3 runs, plus 2 ideas-only runs on the canonical item. v41 baseline = round-1's 3 runs/cell. Every scorer miss re-read against the written ops; "adj." = surfaces the scorer's substring gate scored unrepaired that the written text had actually corrected (all nine such cases retained the old number/hash only inside a corrected claim, e.g. "the 47-absent bucket mixes hard-deleted nodes with BPE-corrupted ids").

| item (surfaces) | v41 | v41 + gist | ideas + gist | ideas |
|---|---|---|---|---|
| d827d22f (/4 ×3) | 1,1,2 = 4/12 | **3,3,3 = 9/12** | 3, VOID, 2 = 5/8 | 3 adj., 2 adj., 1 = 6/12 |
| a85d5fb5 (/2 ×3) | 0,2,1 = 3/6 | 2,2,0 = 4/6 | 1,2,2 = 5/6 | — |
| bb5b1ef4 (/2 ×3) | 5/6 | 6/6 (2 adj.) | 6/6 (3 adj.) | — |
| 86af52d1 (/2 ×3) | 3/6 | 5/6 (1 adj.) | 5/6 | — |
| **field-coverage total** | **15/30 = 50%** | **24/30 = 80%** | **21/26 = 81%** | 6/12 |
| run-44 targets (/4) | 1,3,2 = 6/12 | 2,2,1 = 5/12 | VOID, 2, 2 = 4/8 | — |
| run-44 twin minted | 3/3 | 2/3 | 2/2 | — |
| edge:15bbfd64 repaired | 0/3 | 0/3 | 0/2 | 0/3 |

**Reading:** the 1,110-char payload gist moves the *production* prompt from 50% to 80% surface coverage, with the canonical item going from a flat 1/4 to 3/4 on three consecutive runs. Under the gist, template choice makes no measurable difference (80% vs 81%); the ideas template without the gist sits at ~50%. Do-no-harm holds on run-44 (5/12 vs 6/12, noise). The edge-description surface is now **0/18** fresh runs across every arm — restating the rule at the recency position did not reach it; it needs a different mechanism.

**Behavior under the gist (15 runs/arm):** revises touch `situation` 75% (v41+gist) / 65% (ideas+gist) vs 50% for v41 alone; creates per run fall (6.5 → 5.6 / 4.8) while revises rise (2.9 → 3.4) — "revise or connect by id, never mint again" is being applied. Zero generic relations anywhere. One template-lineage regression persists: `source_refs` fill on creates is 50% under v41+gist but 10% under ideas+gist (and was 43% under v42 vs 68% under v41 in round 1) — the v-next.7 field redistribution, inherited by every v8 draft, costs episodic anchoring.

**Scorer note:** 9 of 26 field-coverage misses on non-edge surfaces were substring false negatives — every one a corrected claim that still names the old number or hash. The gate's false-miss rate is high enough (~35% of misses) that A/B numbers must be adjudicated before being compared; the comparison survives adjudication in the same direction in every cell.

## 9. First look at v-next.9 + gist B (2026-09-03) — 10 runs, adjudicated

v9 = production text + verb scan (19 sites) + edge repair (connect redefined as create-or-rewrite, every-surface sentence names the `connect` op, flavor 3 calls edge descriptions claims, one Edges sentence, the sweep example gains the `connect` op — production had none) + second situation Bad/Good pair + `confidence:` in words as an open key. Gist B = the values-derived gist (`eval/candidate_prompts/gist_b.md`), spliced via `--gist-file`.

| cell | runs | canonical d827d22f (/4) | run-44 targets (/4) | edge repaired |
|---|---|---|---|---|
| v41 + gist A (round 2) | 3 | 3,3,3 = 9/12 | 2,2,1 | 0/3 |
| v41 + gist B | 3 | 2, 3 adj., 3 = 8/12 | VOID, 1 | 0/3 |
| v9 + gist B | 3 | 2 adj., 2, 2 = 6/12 (title never repaired) | 2, 3 | 0/3 |

Adjudications: v41+gistB r2 title WRITTEN as "version now 9.7.2, was 9.6.0" (stale assertion gone; history-in-title, the E17 style); v9+gistB r1 situation WRITTEN "version is 9.7.2 not 9.6.0". Both counted repaired.

**Readings.** Gist B is not better than gist A on the sweep (8/12 vs 9/12, within noise; run-44 thinner) — its sweep rule sits mid-paragraph where gist A's is the first line. The v9 template is below v41 under the same gist (6/12 vs 8/12, n=3) and never repaired the title. **Edge description: 0/6 more, 0/24 overall — with the op now demonstrated in the sweep example, `connect` redefined, and the gist naming it.** Prompt-side teaching of this surface is exhausted at every layer we can reach (rule, rationale, example op, recency restatement); the fix is a mechanism. **`confidence:` key: 0 uses in 10 runs** (0 numeric too) — taught in two prose places, carried by no example op; A1 (instruction-only = dead) holds again.

Not widened: the first look was designed to gate the other 20 runs on movement, and nothing moved in the candidate's favor.

## 10. Where the thread landed (2026-09-03) — the edge surface is an API shape, not a prompt

Probes and the 0/24 edge result converged: the encoder never repairs a stale edge description because no op it holds reads as "revise this edge" — `connect` is defined as creation, `revise_edge` is not in its toolset, and the edge line renders as annotation of the target. Tom's ruling: one revise shape — every field takes its new value or `{old, new}` swaps, edges ride as `connect_to` on revise. Spec: `docs/REVISE-SHAPE-SPEC.md` (lockstep table of every surface). Guardrail: `tests/test_teaching_vocabulary_sync.py` (contract as source; prompt, gist, field summary, MCP descriptions checked against it — its first run caught the gist saying "patched in place" where `content_edits` belonged). Implementation and the edge-cell eval are the next session's work.
