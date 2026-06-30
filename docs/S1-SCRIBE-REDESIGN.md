# S1 Scribe — Whole-Encoder Redesign

**Status:** design / pickup spec. Captured 2026-06-28. **Nothing built yet.** This is
the master spine for reopening the S1 Scribe (S1E) encoder *as a whole* — input
architecture + journal/residue + the empirical refinements its own journals asked
for. Highest-risk encoder (live recall path, feeds the Frame), so it's eval-gated and
goes last.

## How to use this doc

This is the **entry point**. It consolidates and sequences three threads that turned
out to be one piece of work. Two sub-docs hold the deep detail; don't duplicate them,
read them for the gory parts:

- **`docs/ENCODE-ON-IDLE.md`** — the input-architecture research (sandwich layout,
  per-turn provenance ledger, glossary+reference, XML compartments, task-at-end) +
  the idle-flush mechanism + the effort/thinking analysis. **Parts 3–4 are the spine
  of the input reframe.**
- **`docs/ENCODER-JOURNAL-DESIGN.md`** — the journal→residue substrate (the `## Review`
  note contract, the two-feed split, storage, the phased plan). Phases 0/2/3/5 are
  SHIPPED for the S2 encoders; Phase 4 (S1E) is what this doc covers.

**Why a third doc:** the scope opened from "port the journal to S1E" to "rebuild what
the encoder receives and how." That holistic view has no home in either sub-doc. This
doc is that home. When a decision here refines a sub-doc, the sub-doc is still the
detail-of-record; this doc is the unified plan.

---

## Prompt rewrite — DRAFTED 2026-06-29 (full section-by-section walk)

The complete v-next `s1e` prompt is drafted (first-person throughout) →
**[docs/S1E-PROMPT-v-next-DRAFT.md](S1E-PROMPT-v-next-DRAFT.md)** (NOT live; review →
register DORMANT → eval-gate → activate).

**Decisions locked this session:**
- **Voice** — first person throughout, *including node content* ("When I paraphrase", not "Anchor"). The graph heals toward it; existing 3rd-person nodes are acceptable drift. Quotes stay verbatim.
- **Verbs** — action lines use `remember`/`revise`/`connect`; "encoding" only as the activity/role noun, never an action verb.
- **Priority** — remember→revise→connect *by weight* (supersedes connect-first `1a79b7dd`).
- **"meaning"** replaces "insight" as the deeper-layer word; **details → meaning** is the core capture rule.
- **New `thought` field** — the scribe's own read on a node (≠ `content`, ≠ `reasoning`); selective, quality-gated.
- **Open fields refined** — field-name-as-encoding-prompt (`753ad314`); name specifically; `thought` is the promoted example.
- **Edges** — R3 inverse rule: a relationship named in prose → draw the edge ("the graph walks on edges, not content").
- **Quality gate** — "encode what earns its place — new AND useful" (supersedes "Don't be too conservative" `08867ee7`); + "my bar runs high, I correct for it" details+thought reinforcement; list now includes **formulas + the principle/concept each points to**.
- **Timeline** — XML lived-sequence (`<turn>/<user>/<assistant>/<actions>/<provenance>`); pulls light, action-tools carry cues; every id ref carries a 1-line «tag», full body once in the catalog; empty provenance line = unencoded turn.
- **Catalog** — widened union: surfaced ∪ encoded-this-session ∪ Anchor-authored ∪ **endo (empty stub fn now)**.
- **Continuity** — residue (contract-injected) + arc. **Review notes / residue are SESSION-BOUND** — within-session, run-to-run continuity only; the s1e continuity read filters by `session_id` (last K runs *in this session*), distinct from S2's genuinely run-based read (S2 has no sessions). **Scope decomposition:** cross-session = the **graph substrate** (nodes/edges) + **wisdom** (Frame's "What I've learned"); session-scoped = the **catalog** (a per-run *view* into the graph, recomputed each run), the **arc**, and the **residue**. The Frame is mixed — wisdom cross-session, arc session-bound. The defer-fix rides on the **substrate** being cross-session, not the catalog view.
- **Defer fix** — *don't defer into a gap*: continuity runs through the graph (catalog→revise), not the sliding window. **The `final` flag is DROPPED** — with no deferral, nothing to suppress on a terminal pass.
- **Mechanism separation** (don't mix): **suppress** (already-captured, cross-session) / **correct** (replace the wrong) / **supersede** (keep BOTH, old valid-as-of-date) / **open** (genuinely unresolved — scoped to purpose, not a catch-all).
- **Residue close** — `## Review` (judgment half) + closure are contract-injected; `ENCODED` → trace/provenance; arc written separately.
- Section-3: four flavors (count fix), WATCHING→residue, trace→`trace="…"` attribute.
- Fixes: precision enum `explicit|relative|approximate`; "Five"→"Six" instincts; cadence-agnostic Speed.

**Open flags (resolve before registration):**
1. **`final` flag drop** — confirm we cut the planned `run_encoding(…, final)` build item (superseded by the no-defer reframe).
2. **trace vocabulary** — `[trace:<hex>]` still in some KEEP blocks; unify to the `trace="…"` attribute model.
3. **"encode" as action-verb** sweep — a few KEEP lines ("encode the correction triple") still use it.
4. **§7.6 example selection** — revisit which five (recall timed out mid-session); verify first-person node-content rewrite.
5. **"the next me / future me"** vs one-continuous-self (`b659ca7a`) — consistency check (section 1 confirmed as-is).
6. **Frame** — Recent-moves removal (Frame-side half), replacement-before-removal. [ledger item]

**Next session — pickup plan** (the prompt *design* is done; what remains is build + eval. The handoff is these three artifacts: this checkpoint, the draft, and the encoded decisions — no fork needed):

0. **Orient** — read the draft + this checkpoint. Don't re-walk the prompt; build *to* it.
1. **Close the cheap flags** → produces the final draft to register:
   - ②/③ **just do** — trace-vocab unify (`[trace:<hex>]`→`trace="…"`), the "encode"-as-action-verb sweep.
   - ④ **needs Tom** — §7.6 example selection (dig the authoring provenance, decide whether those five are right, verify the first-person node-content rewrite reads well).
   - ⑤ quick consistency check ("the next me" vs one-continuous-self).
2. **Build the code half** — the major engineering, where the eval-risk lives. Input + prompt are **coupled** (the new prompt describes input the code must produce) → they move together; **parameterize so the eval can A/B new-vs-old without flipping live**:
   - **Lived-sequence timeline** — extend the S0 conversation API to surface `tool_result` events (captured, just unsurfaced), interleaved with messages.
   - **Provenance ledger** — per-turn `surfaced/encoded(S1S)` via the `trace_links` S1 capability (stop-join over surface/encode traces, NOT `source_refs`); encoded(Anchor) deferred. ✅ BUILT (§10.3.2).
   - **Widened catalog** — the union (surfaced ∪ encoded-this-session ∪ Anchor-authored ∪ **endo empty-stub fn**), id+«tag» references.
   - **Residue wiring** — `## Review` + closure contract-injected for s1e; **session-scoped** continuity read (`session_id`-filtered, distinct from S2's run-based).
   - Tests per piece.
3. **Register the draft DORMANT** — do NOT `sync-prompts` until the gate passes (dormant candidates must not leak to the seed).
4. **Build the 3 missing eval dims** (§8: arc-still-produced · notes-residue-only · notes-not-over-produced) + the Frozen-Corpus A/B setup.
5. **Eval A/B** — old vs new, `--variance ≥3`, same qids. The gate.
6. **Activate atomically** iff the gate holds: `set_interaction_active` → `./dev sync-prompts` → cut the Frame "Recent moves" slot (⑥, replacement-before-removal) → restart.

**Scope honesty:** steps 2–6 span several sessions; a realistic *next* session does step 1 + starts step 2.
**Operational:** the daemon was flaky this session (recall timed out twice) — health-check at boot, `restart` if recall keeps failing.

---

## 1. The core realization — these are ONE change, not three

Three threads collapse into one because they share substrate:

1. **Journal port** (residue `## Review` channel for S1E).
2. **Input-architecture reframe** ("inline everything into the conversation" — the
   provenance ledger + sandwich).
3. **The empirical refinements** the journals surfaced (below).

They're entangled at the seam: the journal's **action feed** ("what I already
encoded") *is* the provenance ledger. Building the journal port against the *current*
block structure would build the action feed once, then tear it out when the ledger
lands. So we do them together, within the new input architecture. (Backing:
`aa1f3ece` two-feed split; `82f563d4` journal=judgment-half / provenance=factual-half;
`c9444a17` Tom's inline-everything direction.)

**Governing principle** (`5de09c90`, `ce333a73`): *the agent writes only residue;
everything factual is rendered from traces algorithmically.* The encoder narrates what
its mind did (doubt, friction, surprise); it never restates what its hands did — the
trace already has that.

---

## 2. The data map — what we feed the encoder (today vs. target)

Tom's directive: *"We provide a lot of data, but also not enough. Map what we provide
to the encoder."* Here it is. Source for "today" = code read of
`encode.py:_build_user_content` + `s0/conversation.py:get_conversation`.

### What the encoder receives TODAY

| Stream | Provided? | Source | Notes |
|---|---|---|---|
| System prompt (role, rules, examples) | ✅ | `s1e` interaction | top, 1h cache |
| Section-legend preamble | ✅ | `encode.py` | top |
| Encoding journal (prose blob) | ✅ | `brain_meta` `encoding_journal_{sid}` | **retiring** — blob, truncated at 2000 chars |
| Session context (the arc) | ✅ | `session_context_{sid}` | keep (load-bearing for recall) |
| Node catalog (full rich nodes) | ✅ **but only Haiku-surfaced nodes** | `build_node_catalog` from judge outputs | the glossary, but narrow |
| Scout reports (facts / quote / temporal) | ✅ | muster | temporal scout ~100% noise (see §4) |
| Conversation timeline (USER/ASSISTANT text + `[trace:hex]` + per-turn `SURFACED:` refs) | ✅ | `get_conversation` | **messages only** |

### What the encoder is BLIND to today — the gaps

| Missing stream | Why it matters | Target home |
|---|---|---|
| **Tool uses (Edit/Bash/MCP/Read/…) + results** | The encoder encoding a coding/work session can't see what was *done* — only what was *said*. `get_conversation` returns `role∈{user,assistant}` only. **Confirmed gap — but READ-SIDE ONLY** (verified 2026-06-29, §10): the events ARE captured — `post_tool_trace.py` writes a `tool_result` S0 delta per call with a ready-made `summary` (≤500c: "Edit: foo.py", "Bash: …", "recall: …") + `{tool}` metadata, in the same `s0-{session}-{stop}` chain, timestamp-ordered. **7,856 in 7 days.** `get_session_turns` just filters them out. So piece 1 is *extend a read*, not *build capture*. | inline in the timeline, at real position |
| **encoded(S1S) per turn** — nodes prior runs wrote, anchored to each turn | Today only via journal prose (drifts, often truncated/empty). The encoder dedups by *assertion*, not verification → confident false-skips (§4). | provenance, inline per turn (ground truth from `source_refs`) |
| **encoded(Anchor)** — nodes Anchor wrote directly via MCP mid-conversation | Encoder can't see them unless they were also Haiku-surfaced → re-encodes, or misses what Anchor missed. | provenance, inline per turn (`encoding_source='anchor'`) |
| **endo-surfaced memories** (PreToolUse / Stop recall) | The endo system (soon) recalls memories right before Anchor acts. Those recalls are part of what happened and must reach the encoder. | inline **before the tool use** they preceded (§3) |

### The target: the timeline becomes the full *lived sequence*

Not "messages + a catalog beside them." The timeline is the conversation **as it
actually unfolded**, in temporal order, with everything inline at its real position:

```
user message
  → Haiku surfaced: id:.. id:..        (what recall gave Anchor this turn)
assistant message (text)
  → endo surfaced: id:..               (recalled right before the next action)
  → tool use: Edit foo.py / Bash …     (what Anchor DID)
  → tool result: (terse)
  → endo surfaced: id:..
  → tool use: …
provenance for this turn:
  encoded(S1S): —   encoded(Anchor): id:7f3e
```

This single model answers all three of Tom's asks at once: **tool uses** become a
timeline element; **endo memories** inline *before the tool use* that triggered them;
**"nodes they encoded"** become the per-turn provenance lines. "Inline everything
within the conversation," in full.

### Consequence: the glossary catalog must expand

Today the catalog is built **only** from Haiku-surface selections. But the new
timeline references nodes by `id:` from *four* sources — Haiku surface, endo, prior
encodes (S1S), and Anchor's direct writes. **Every node referenced anywhere in the
timeline must exist once in the catalog** so the `id:` refs dereference. So the
catalog scan widens: union of {Haiku-surfaced} ∪ {endo-surfaced} ∪ {encoded this
session} ∪ {Anchor-authored this session}, deduplicated, full-rich once at top.
(Glossary+reference rule from `ENCODE-ON-IDLE` §3: bodies live once in the catalog;
the timeline carries only `id:` refs — never re-inline bodies, or the prompt bloats.)

### "In terms of guidance"

Tom: these streams need **guidance**, not just exposure. The prompt must teach what
each means and how to act on it:
- **Tool uses** → encode the durable *outcome/decision*, not the mechanics (a `git
  push` is not a node; the architectural choice it shipped might be). Same
  ephemeral-vs-durable judgment the encoder already does well (§4 #6/#7).
- **encoded(S1S)/encoded(Anchor)** → "already captured — *revise* if later turns
  reframe it; don't duplicate." NOT "done, don't touch" (`ENCODE-ON-IDLE` constraint).
- **endo-surfaced** → "this is what was recalled before that action" — context for why
  Anchor did what it did; not a mandate to link.
- **Provenance ≠ mandate** → showing "node X surfaced on turns 3,4,7" must not nudge
  dense `source_refs`/over-linking.

---

## 3. Target input architecture (the skeleton)

Top→bottom is the sandwich (load-bearing content on both strong ends; nothing
load-bearing in the dead middle — lost-in-the-middle, `ENCODE-ON-IDLE` §3). XML tags
for compartments.

| # | Section | Position | Cache | Role |
|---|---|---|---|---|
| 1 | System: role + encoding rules + format + how-to-read | top (strong) | 1h | stable "how to encode" — **first person (§5)** |
| 2 | Preamble: section legend + "read before acting" | top | 1h | teaches the layout once |
| 3 | `<continuity>`: residue notes (last K runs) + session arc | early | 5m | priors — what prior runs flagged / what the session is about |
| 4 | `<node_catalog>`: full rich nodes, once — **widened union (§2)** | early-mid (must precede refs) | 5m | the glossary everything dereferences by `id:` |
| 5 | `<timeline>`: the full lived sequence (messages + tool uses + recalls + provenance) | bulk; newest last → strong end | 5m | data + emergent boundary |
| 6 | `<task>`: encode-what's-new + "list unencoded turns first" + `final` line | very end (strongest) | mostly 1h; `final` dynamic | the actionable query |

Key properties (all from `ENCODE-ON-IDLE` §3–4, reaffirmed):
- **Emergent boundary, not a counted one.** Turns whose provenance shows no
  `encoded(S1S)/encoded(Anchor)` are the unencoded tail — visible at a glance, no
  hard-coded "5." Makes turn-count flexibility a non-issue (a 2-turn idle flush and a
  9-turn run read identically).
- **Task last** (~30% quality lift on complex multi-doc inputs) + **ground-before-act**
  ("list the unencoded turns, then encode") binds the model to the data before writing.
- **Endo is stream-extensible & future.** Design the timeline to inline endo before
  tool use now; the first build may ship without the endo line if endo isn't live —
  don't block the other streams on it. **VERIFY endo's launch status before sequencing.**

---

## 4. The empirical refinements — what the encoder told us it needs

Two sources, mined this session: **120 S2 residue notes** (10 days, S2 watching S1's
output) + **950 of 1,255 S1E historical journals** (10 weeks, S1's own voice). The
load-bearing evidence (recurrence rates + representative quotes) is captured below;
the census is reproducible by re-running the journal mine over `encoding_run` traces.

### The format itself buries the signal (storage findings, S1E mine)
- **37% of S1E journals are completely empty** — the encoder ran, said nothing.
- **52% are hard-truncated at 2000 chars.** Order is `ENCODED→SKIPPED→WATCHING`, so the
  **highest-signal section (WATCHING) is what truncation eats** — only 294/950 retain
  any. The trace-restatement survives; the residue dies. The note contract fixes this
  for free (separate trace rows, residue-first, no 2000-char blob).

### The convergence: one blindness, seen from both ends
- **S2 view** — S1 *confidently creates dupes*: *"730f9d02 encoded 'No catalog node
  describes LAF as an architecture' — yet 9a3017ea is the foundational LAF node with 59
  edges… confidently wrong."*
- **S1 view** — S1 *confidently skips real gaps*: "already in catalog" in **56% of
  SKIPPED**; explicit doubt appears in **1 of 532** sections.
- **Smoking gun** — across 824 residue sections: **0 wish-language, ~1 blindness-
  language.** The format has no slot for "what I lacked / couldn't verify," so the
  encoder launders structural blindness into confident prose.
- **Root:** no trustworthy view of what's already in the graph → both false-skip and
  false-create. **Fixed by the provenance ledger + widened glossary (§2–3) AND the
  residue channel (§6) — the two halves of one fix.**

### The refinements table

| # | Signal | Evidence | Fix / strand |
|---|---|---|---|
| R1 | **Temporal scout fires on non-dates** (line #s, %s, arxiv IDs, "may") | **~73% of all SKIPPED**, stable 10 weeks; ~8 garbage candidates/run | ✅ **SHIPPED** (standalone, ahead of the reframe). Two layers, both library-aligned (no reinvention): dropped dateparser's `'timestamp'` parser (read any number as a Unix epoch — the bare-number engine) via the `PARSERS` setting, + tightened `_looks_like_date` from `has_digit_word` to `_DATE_SHAPE_RE` (year / numeric-date / ordinal / digit-relative). `line 47`/`73%`/`PID`/`arxiv` → EMPTY; real dates + relatives unaffected. Residuals (not R1, encoder-backstopped): bare year-shaped numbers (`2000 chars` — kept by choice, preserves `in 2024` refs), modal `may`, keyword-glue (`73% now`). |
| R2 | **Confident "already covered" skip, never doubted** | 56% skip / 0.2% doubt | provenance ledger + widened glossary (§2–3): verification, not assertion |
| R3 | **prose-not-graph** — states a relation in content, never draws the edge | ~8–10 S2 runs ("semantic in prose but not in graph", "encoder should have drawn this") | examples + task framing: pull toward structuralizing relations it names. Same family as the temporal-structure gap (`893cf8c6`: dates in prose, not `event_time`/`anchored_to`) |
| R4 | **No slot for friction/doubt** → laundering | 0 wish / ~1 blindness across 824 | the residue `## Review` channel (§6) — *this is the empirical case for it* |
| R5 | **WATCHING collapsed into a next-session TODO list** | 57% of WATCHING is "if it recurs / next session" | split the residue slot: "thread forming" vs "open loop / handoff" — DECIDE shape (§7) |
| R6 | **Quote scout redundancy** — quote already inside an encoded node | 67 neg / 44 pos | dedup quote candidates against what's being written, or fold into node encoding |
| R7 | **Catalog too narrow** (Haiku-surfaced only) | R2 + the LAF dupe | widen catalog to the union (§2) |

### Bonus (not S1-redesign, but the residue caught it)
- **Decoder suppression bug:** consolidation residue repeatedly: *"the decoder ignores
  existing `similar_to` suppression at this cosine level (0.916) … no edge-adding
  strategy will fix it."* Re-proposes settled pairs regardless of edges. Separate fix,
  worth a ticket. Proof the residue channel earns its keep.
- **Seed-duplicate proliferation** (operator action): locked seed dups 4→10 across
  runs, "minting every batch." Needs an operator unlock/stop-locking decision.
- **Stale edge descriptions** when content is revised (the "74% vs 71%" edge) — vector/
  edge healer territory.

---

## 5. Other S1E changes

- **First-person voice** (`accf5172`, Tom's directive): the prompt frames the scribe as
  a third-person archivist ("a future reader who will wake with zero memory"). Tom
  wants **"I am Anchor, encoding my own memory"** — first person throughout, even
  though it's Sonnet for now. Audit the whole prompt for "Anchor" (3rd person) → "I".
- **Terminal-flush `final` semantics** (`ENCODE-ON-IDLE` Change 1): the encoder is
  trained to *defer* ambiguous material to "next run" — exactly wrong for an idle/tail
  flush where there may be no next run. Add `run_encoding(..., final: bool)` →
  one run-context line that suppresses deferral on a terminal flush. **VERIFY:** is
  encode-on-idle (idle trigger) already live while `final` is unbuilt? If so, idle
  flushes are currently deferring the material they exist to capture — a latent bug.
- **Cadence wording** (Change 2, trivial): "You run every 5 messages" is wrong (every
  ~5 turns ≈ 10 messages, 20-msg window, and idle-flush breaks "every 5" entirely).
  Reword cadence-agnostic.

---

## 6. The journal / residue changes (S1E half of Phase 4)

Detail in `ENCODER-JOURNAL-DESIGN.md`; the S1E-specific deltas:

- **Two feeds, never conflated** (`aa1f3ece`):
  - **Residue feed** → `<continuity>`: `brain.journal_notes(scale='s1', session_id=…)`
    → `render_journal_notes_prefix()`, last K=5 runs (`JOURNAL_CONTINUITY_RUNS['s1e']`).
  - **Action feed** → the **provenance ledger** in `<timeline>` (§2–3). NOT a separate
    "recently encoded" block — that's the conflation the architecture forbids.
- **Write residue:** swap `_save_journal()` (the prose blob) → `brain.write_journal_
  notes(final_text, chain_id='s1e-{sid}-{stop}', scale='s1', session_id=sid)`. The
  encoder emits a `## Review` fenced block (`tag · subject · note`); contract parses +
  writes one `journal_note` trace row per note. Inject the shared review block +
  closure at runtime in `encode.py` (call `trace_contract.render_journal_review_block`
  / `render_prompt_closure` directly — `encode.py` is standalone, no `self`; the S2
  base methods are thin wrappers over the same module fns).
- **Drop `journal_entry`** from the S1E delta metadata (S2 already did).
- **Keep `_save_session_context()`** (the arc) unchanged — it's load-bearing for recall.
- **Cut the Frame's `## Recent moves` slot** (`frame.py:_render_recent_moves` +
  `brain.get_recent_encoding_journal`) — *atomically with the blob-write removal*, or
  the Frame strands on a no-longer-written key. Replacement-before-removal (`e78cfba6`,
  `6494d789`): the provenance ledger is the inline replacement; only cut the Frame slot
  once the inline version is verified. **This also removes "Recent moves" from the boot
  Frame** — intended (recent-moves is encoder continuity, not Anchor's identity prior;
  boot keeps wisdom + current focus), but a visible boot change worth a heads-up.

---

## 7. Open decisions (need Tom / need a probe)

1. ~~**Sequence the temporal-scout fix (R1) first?**~~ ✅ RESOLVED — landed standalone
   ahead of the reframe (see §4 R1). dateparser `PARSERS` drop + `_DATE_SHAPE_RE` gate;
   green at 127 scout/temporal tests. Bare year-shaped numbers kept (preserves `in 2024`
   refs); revisit only if the noise/recall balance argues otherwise.
2. **WATCHING split (R5):** one residue slot, or split "thread forming" vs "open loop /
   handoff"? (Affects the `## Review` slot semantics for S1E.)
3. **Endo launch status:** is endo live? Determines whether the first build inlines the
   endo stream or ships stream-extensible-but-empty.
4. **`final`/idle-flush gap:** verify current live state (§5) — possible latent bug.
5. **Effort off vs adaptive-low** (`ENCODE-ON-IDLE` §Effort): OPEN. Old 24-trial probe
   said forced thinking bought nothing for the *old* encoder; never tested adaptive-low,
   predates the provenance-heavier input. Post-build eval arm, not a blocker.
6. **Tool-result verbosity:** how terse? (Token budget — bodies in glossary, results
   summarized.) Tune at build.

---

## 8. Eval gate (from this session's measurement-validity audit)

Build → **then** A/B old-vs-new on the Frozen Corpus (Tom's ordering). The harness is
**half-ready**:

- **Trustworthy** for: encode-coverage (ENCODE_MISS), recall-conditional pass rate,
  token cost — **iff** run with `--variance ≥3` (defaults to n=1, the C4=1.00 trap) and
  A/B'd over the **same qids** via `build_corpus --interaction-override s1e=<old>` vs
  `s1e=<new>`.
- **Must BUILD (3 of 6 gate dims are absent):**
  1. **Arc still produced** (binary) — the design's #1 guardrail; *nothing* inspects
     `session_context_{sid}` today.
  2. **Notes residue-only** (no trace-restatement) — no harness reads `## Review`.
  3. **Notes not over-produced** (empty on clean runs).
- **Traps:** `s1_encode_eval.py` is a **dry-run mechanics meter** — never a quality gate.
  `encoder_eval`'s regression-halt is **hardcoded to v22/v19** → inert on a new version,
  re-pin first. The brain-native `realchat_oracle.json` corpus **isn't on disk** →
  default LongMemEval measures generic QA, not partnership texture; build it if we want
  to test what the residue exists to capture.

**Activate only if** arc-production holds and encode-coverage doesn't regress.

---

## 9. Approach & build sequence

**Architecture-first** (the sections share substrate — refining them independently =
rework, e.g. the action-feed-built-twice trap):

1. **Lock the skeleton** (this doc → a finalized §3 spec): every stream assigned to a
   section, what's inline vs referenced, the widened catalog, the guidance per stream.
   Mark scope/deferrals (endo, effort).
2. **Build section-by-section against the locked skeleton** — catalog/glossary widening,
   provenance ledger renderer (incl. tool uses + endo placeholder), `<continuity>`,
   `<task>` block, residue write, voice pass, `final` semantics. They **land together**
   as the v-next `s1e` prompt (register **DORMANT**, don't activate).
3. **Build the 3 missing eval dims** (§8) against the new output.
4. **Eval once** — A/B old vs new, `--variance ≥3`, same qids.
5. **Activate atomically** only if the gate passes: `set_interaction_active` →
   `./dev sync-prompts` → cut the Frame slot → restart. (Prompt-change discipline:
   register DORMANT → eval → activate → sync; never sync a dormant candidate.)

~~Possible early standalone win: R1 temporal-scout gating~~ ✅ DONE — R1 landed
standalone (dateparser `PARSERS` drop + `_DATE_SHAPE_RE` gate). The reframe (steps 1–5)
is the remaining work.

---

## 10. Code-half build plan (Phase 4 — detailed, verified 2026-06-29)

### 10.1 Build readiness — step 0 verified
Tool-event capture is confirmed (see the §2 gap row). The substrate already holds
everything the new timeline needs:
- **7,856 `tool_result` S0 deltas in 7 days** (vs 688 user_message, 573 assistant_message).
- Each carries a ready-made **`summary`** (≤500 chars; `post_tool_trace.py:_build_summary`
  renders per-tool cues — `Edit: foo.py`, `Bash: <cmd>`, `Read: <file>`, `recall: <query>`,
  `Agent: <desc>`) plus `metadata={tool}`, on the same `s0-{session}-{stop}` chain as the
  turn's messages, timestamp-ordered.
- `get_conversation` → `_trace_dal.get_session_turns` returns `role∈{user,assistant}` only —
  it simply filters the tool rows out.

**Consequence:** every code-half piece is a **READ + RENDER over data that already exists** —
no new capture pipeline. The biggest unknown is closed.

### 10.2 Architectural law: generic trace queries, NOT bespoke DAL
**Directive (Tom, 2026-06-29):** repurpose the generic trace-query API; do **not** write a
dedicated DAL method per use-case. The trace DAL is generic and must be robust enough to serve
its clients; the S0 traces layer (`conversation.py`) **composes**; callers use the door. This
is the unification already done cleanly — extend it, don't fragment it.

Lineage (load-bearing, do not re-litigate): `35cedbe1` (go through the traces layer, not DAL),
`f2b8966a` (traces layer owns schema/contract; callers delegate), `e56dc13b` (S0 exposes its
own API for all layers), `d1329a9f` (journal reads via `brain.query_traces`, not TraceDAL),
`aaee405f` (coordinator uses trace-layer fns), `6523755f` (cadence = live trace-pull, not
counters).

**Concretely:** the reads go through the existing trace-layer doors — **`recall_episodes`**
(the conversational lens over s0; pass `ref_type=['user_message','assistant_message',
'tool_result']` for the **interleaved lived sequence**, tool events included — see §10.3.1) and
`query_traces` / `journal_notes` — never a new bespoke read. **If a need can't be expressed by
the existing doors, the fix is to make them more capable — never a one-off DAL query.**
(`recall_episodes` already advertises the interleaved-with-tools mode in its docstring; this is
the unification Tom means — repurpose it.)

### 10.3 The pieces (all behind ONE A/B flag; input+prompt land together)

1. **Lived-sequence timeline** (foundational — everything else references it) — §10.3.1
   - *Read:* `brain.recall_episodes(session_id=…, ref_type=['user_message','assistant_message',
     'tool_result'], sort_order='asc', limit=…)['episodes']` — the **existing** conversational-lens
     door already returns the interleaved sequence as full trace records (tool_result `summary`
     included). **No new fn, no new DAL.**
   - *Compose+render:* `encode.py:_build_user_content` groups the flat created_at-ordered episodes
     into turns (a `user_message` opens a turn; the `assistant_message` + the `tool_result`s before
     the next `user_message` belong to it — the same turn-walk the current builder already does on
     messages) and emits XML `<turn n><user trace><assistant trace><actions>{summary
     lines}</actions></turn>`; pulls render light, action tools carry their `summary` cue verbatim.
   - *Note:* `recall_episodes` is trace-only, so `surfaced`/judge_output is NOT here — it belongs to
     the `<provenance>` block (piece 2), keeping piece 1 a pure timeline read.
   - *Tests:* tool events land between their user/assistant by timestamp; turn-grouping; render snapshot.

2. **Provenance ledger** (`<provenance>` per turn) — ✅ **BUILT 2026-06-29** (flag-off,
   uncommitted). Built on a NEW reusable S1 capability, **NOT** a `source_refs` reverse-lookup.
   (Tom, 2026-06-29: `source_refs` is the encoder's *sparse, judgment-based* anchor — 1–3
   load-bearing turns, for recall — so it misses most of what a run wrote and conflates two
   purposes. The factual record comes from the **encode event's own delta traces**.)

   **As built — `servers/scales/s1/trace_links.py` (an S1 capability, NOT a brain method).**
   The architecture was reworked this session from the original §10.3.2 sketch (a
   `brain.session_provenance` method, timestamp proximity). Tom's redirects:
   - *S1 layer, not brain level.* Recall is brain-level (spans every scale); surface + encode are
     about the **turn** → S1 territory. This is the S1 sibling of `scales/s0/conversation.py`:
     S0 composes s0 traces → conversation; this composes s1 traces → links over those turns.
   - *Consumer-neutral name.* It's not "provenance" (one reader's lens) — it's a **trace↔node
     link**: `{trace_id: {surfaced:[ids], encoded:[ids], encoded_by: run_trace_id|None}}`. The
     encoder reads a link as "already handled → revise, don't dupe"; a **recall layer** reads the
     same link as "nodes touched around traces like this → a candidate cue lane." `<provenance>`
     survives only as how encode.py *renders* a link.
   - *Structural stop-join, NOT timestamp proximity.* Every chain for turn N ends `-N`
     (`s0/s1r/s1e-{short}-N`), so `_stop_of` extracts the join key from any of them. surfaced is
     1:1 with a turn (same stop); a turn's owning encode run = first run whose stop ≥ the turn's
     stop; `encoded_by is None` = the unencoded tail (emergent boundary, no hard-coded "5").
   - *Two layers → robust to a raw trace dataset run sequentially* (eval replay / sequential data):
     **`nodes_for_traces(surface_traces, encode_traces, target_traces)`** is PURE (plain records
     in, link map out, no brain/DB); **`gather(brain, session_id)`** is the thin live adapter over
     the `query_traces` door (§10.2-compliant). Returns node **ids** (full); the render truncates
     to 8-char `id:` refs that dereference into the catalog (no inline bodies anywhere).

   **Render (encode.py):** `<provenance>` per turn — surfaced refs + the encoded marker. A run's
   full id-list shows once at its **frontier turn** (the last turn it covers, adjacent to the
   boundary); covered-but-not-frontier turns show a bare `✓`; the unencoded tail shows no encoded
   marker at all. Frontier-dedup avoids 5× repetition (which the design warns would nudge dense
   `source_refs`). Guarded: any provenance failure degrades to the piece-1 timeline.

   This same capability feeds **piece 3** (the catalog's `encoded` ∪ `surfaced` id-sets) and is
   reusable for dashboards / encode-coverage eval / "why didn't X encode" debugging.

   **encoded(Anchor) — DEFERRED seam (verified).** Node *creation* leaves no trace (only revises
   do — confirmed: no `node_created` ref_type exists), so the only signal is
   `encoding_source='anchor'` — the in-flux attribution field the encoder must NEVER see. So the
   Anchor-vs-S1S split is TBD pending Tom's encoding_source rework; the link shape has room for an
   `anchored` relation to join with its own clean source later. *Leak audit (resolved c):* the
   catalog already hides it (`S1_NODE_CONFIG show_encoding_source=False`) and the timeline renders
   only message/tool text — `encoding_source` does not reach the encoder. A render test pins this.

   *Tests (15, green):* `tests/test_s1e_trace_links.py` drives the pure composer over a synthetic
   raw dataset (stop-join, surfaced 1:1, encoded per-run range, `encoded_by` pointer, `None` tail,
   malformed-row resilience, out-of-order runs) + render integration (frontier-dedup, boundary, no
   `encoding_source` leak, guarded degrade). Full suite green (1924 passed).

**Piece 3a — the `anchor_touched` feed** ✅ **SHIPPED to main `da124b1`** (live, write-on/
read-gated). The S0 mirror of the encode delta — emerged from Tom's Q3 (the catalog must
present nodes Anchor *personally* touched, not just Haiku-surfaced). The daemon captures, per
turn, what Anchor's own tools touched (`created`/`revised`/`recalled`) and flushes one
`anchor_touched` S0 delta at the Stop boundary; `trace_links` reads it as `authored`/`recalled`/
`endo` through the same `_delta_ids` parser + stop-join. Captured at `daemon_server._dispatch`
(structurally Anchor-only — the in-process encoder bypasses it). `ANCHOR_TOUCHED_KEYS` drives the
builder + accumulator + validation shape. `encoded(Anchor)` is no longer a deferred seam — it's
this. `endo` rides the same delta when wired. Legacy date-chain per-action traces left as-is
(BACKLOG F8). Files: `trace_contract.py` (`build_anchor_touched_metadata`), `session_context.py`
(`touched`), `daemon_server.py` (`_accumulate_touched`), `daemon_hooks.py` (flush), `trace_links.py`.

3. **Widened catalog** ✅ **SHIPPED to main `8830951`** (behind `BRAIN_S1E_LIVED_SEQUENCE`;
   flag-off byte-identical = the A/B control arm). `build_node_catalog(judge_outputs, brain,
   extra_ids=)` folds the union {surfaced} ∪ {encoded} ∪ {authored} ∪ {recalled} in, deduped,
   full-rich once, each tagged. `extra_ids` comes from `trace_links.session_node_ids(encode,
   touched)` (session-level union, distinct from `nodes_for_traces`' per-turn map; same
   `_delta_ids`). `PROVENANCE_TAGS` (contract constant) drives priority authored > recalled >
   encoded > surfaced (surfaced untagged); community nodes skipped across all categories; one
   batched `get_node`. `_build_user_content` gathers the streams ONCE and threads them into both
   the catalog and the timeline's `<provenance>` (no double-pull). `endo` joins when wired.
   - *Tests:* every `id` referenced in the timeline dereferences in the catalog (by construction —
     provenance ids ⊆ the catalog union); tagging/priority/dedup/community-skip; flag-off unchanged.

4. **Residue wiring** ✅ **SHIPPED to main `e6181e0`** (behind `BRAIN_S1E_LIVED_SEQUENCE`;
   flag-off byte-identical = the A/B control arm). The S1E journal blob → the `## Review` note
   contract, **SESSION-BOUND** (the S1↔S2 divergence: S1E scopes continuity by `session_id`, S2
   by `unit`; same machinery, different arg). Three flag-gated edits in `encode.py`, all reading
   ONE arm resolved once in `run_encoding` and threaded down (no torn arm):
   - *Write* — `run_encoding` post-loop: `brain.write_journal_notes(final_text, chain_id=enc_chain,
     scale='s1', session_id=sid)` (one `journal_note` trace per `## Review` note, run-grouped by
     `enc_chain`). Flag-off keeps `_save_journal`.
   - *Read (continuity)* — `_build_user_content`: `render_journal_notes_prefix(brain.journal_notes(
     scale='s1', session_id=sid))`, last K=5 runs **of this session**, in the encoder's body
     (the lowercase-frame priors slot — NOT the identity Frame). Flag-off keeps the `### Encoding
     Journal` blob.
   - *Write-instructions* — `_build_system_prompt`: `render_journal_review_block()` + closure
     (`render_prompt_closure`) as the LAST block, two separate contract-owned injects (recency:
     writing the review is the encoder's final act).
   - **Frame `## Recent moves`: NOT cut here.** The decision (2026-06-30): the provenance ledger
     already inlines "what happened," and Recent moves was 100% trace-restatement — so it's *cut*,
     not re-sourced, but the cut is an **activation step** (replacement-before-removal), not P4.
     Flag-on intentionally strands it (deferred-cut previewing); no eval confound (the sweep
     queries a fresh session → Recent moves empty in both arms). Arc (`_save_session_context`)
     untouched — a distinct object.
   - *Tests:* `test_s1e_residue.py` — session-bound round-trip + session-walling + K=5; both read
     arms; fresh-session empty; system-prompt inject + closure-last + lived-param-overrides-env.

### 10.4 Sequencing & the gate
- One A/B flag gates new-input+new-prompt vs old, on the same corpus — input+prompt are coupled,
  so they land together but flip together for measurement (no confound).
- **The real Frozen-Corpus A/B cannot run until the code half exists** — the new prompt
  references `<timeline>`/`<provenance>` the old builder doesn't produce. The code half is the
  *gate-to-the-gate*.
- Build order: piece 1 (foundation) → piece 2 (provenance) → piece 3 (catalog, shares piece-2
  read) → piece 4 (residue) → §8 eval dims → Frozen-Corpus A/B (with Allen-edge composition
  pinned as a named metric, per the temporal-trim decision).
- **STATUS (2026-06-30): the S1E CODE HALF is COMPLETE.** ✅ P1 lived timeline (`8cc8cf1`) ·
  ✅ P2 provenance/`trace_links` (`08420f3`) · ✅ P3a `anchor_touched` feed (`da124b1`, live) ·
  ✅ P3 widened catalog (`8830951`) · ✅ P4 residue wiring (`e6181e0`). All behind
  `BRAIN_S1E_LIVED_SEQUENCE` (flag-off = byte-identical control arm), except the 3a *write* which
  is live (read still gated). **Remaining: the EVAL-PREP phase — register the v-next prompt
  DORMANT + reconcile the preamble; build the 3 missing §8 eval dims; then the Frozen-Corpus A/B
  gate.** Nothing activates until the gate passes. (`e6181e0` = the P4 commit, stamped at merge.)

---

## 11. Eval-prep — the vet-and-gate phase (next session)

Code half complete. This phase = **reconcile the v-next prompt with the as-built code → register it DORMANT → build the 3 missing eval dims → run the coupled A/B gate.** Tom runs the evals; this section is the durable spec.

### Decisions locked (2026-06-30)
- **Coupled A/B, NOT split.** Run old-input+old-prompt **vs** new-input+new-prompt as one unit (no ablation isolating input-vs-prompt). Matches production; faster.
- **Gate = no-regression, but OUTPUT = rich per-case quality detail.** The bar to flip live is "no regression on encode-coverage + recall-conditional pass, clean residue, arc still produced." But the eval must *report* detailed per-case comparison (what each arm encoded, dupes avoided, residue quality) — Tom wants to SEE the quality, not just an aggregate pass rate.
- **Corpus:** generic LongMemEval for the no-regression gate (`--variance ≥3`, same qids via `build_corpus --interaction-override s1e=<v-next>`); add a small **encode-side dedup spot-check** for the value signal. realchat-native corpus = deferred (deeper-texture, not blocking).

### ⚠ Prompt↔code reconciliation — DO FIRST (these confound the eval if unfixed)
The v-next prompt (`docs/S1E-PROMPT-v-next-DRAFT.md`) was drafted to an *idealized* input; the code (P1–P4) built a simpler version. Each needs a decision — **fix the code to match the prompt, or fix the prompt to match the code** — before the A/B, or the new arm underperforms for alignment reasons, not quality:

1. **`<provenance>` fields.** Prompt shows `surfaced / encoded(S1S) / encoded(Anchor)`. Code (`_render_provenance`) renders only `surfaced / encoded(S1S)`; Anchor's touches surface as **catalog tags** (`[anchor-authored]`/`[anchor-recalled]`), not a provenance line. → Drop `encoded(Anchor)` from the prompt's `<provenance>` and point it at the catalog tags, OR add a `touched(Anchor)` line to the render. (Lean: fix the prompt — the catalog-tag design is deliberate.)
2. **«tag» locality NOT built.** Prompt relies throughout on "every id ref carries a 1-line «tag»" (`id:9c1d «WAL contention»`). Code renders bare `id:xxxxxxxx` (no tag). → Build the «tag» render (node-title lookup per ref) OR drop «tag» from the prompt. (This is the biggest one — the prompt's locality affordance doesn't exist.)
3. **`<actions>` id-resolution.** Prompt shows `recall "…" → id:9c1d «tag»`; code renders the raw tool_result `summary` (`recall: <query>`), no `→ id`. → Decide whether actions resolve recalled ids inline (needs the recall→ids join) or stay summary-only.
4. **Catalog tags unexplained.** Code emits `[anchor-authored]`/`[anchor-recalled]`/`[encoded]`; the prompt's `<node_catalog>` description doesn't teach them. → Add a one-liner explaining the tags (whatever #1 resolves to).

### Open flags to close before registering (from the prompt-rewrite section above)
final-flag-drop (confirm cut — code has none) · trace-vocab unify · "encode"-as-action-verb sweep · §7.6 example selection (needs Tom) · "next me" consistency · Frame Recent-moves removal (the activation-step cut).

### Behavioral vet checklist (watch in the A/B output)
Does the encoder, on the new arm: **dedup-via-verification** (uses catalog/provenance to NOT mint twins — the headline value) · emit a clean `## Review` fence (`tag · subject · note`, residue-only, not trace-restatement, empty on clean runs) · honor the quality gate ("new AND useful", doesn't under-encode) · **revise every contradicted field** (no half-revised nodes) · keep `source_refs` sparse (1–3) · write in first-person voice · use `thought` selectively.

### Careful areas (the §8 traps)
- **Re-pin** `encoder_eval`'s regression-halt (hardcoded v22/v19 → inert on a new version).
- `s1_encode_eval.py` is a **dry-run mechanics meter**, never the quality gate.
- **`--variance ≥3`** always (n=1 = the C4=1.00 trap).
- Register the v-next prompt **DORMANT**; do NOT `sync-prompts` until the gate passes (dormant-leak guard).
- The **preamble** (`encode.py` `_build_user_content`) is flag-on stale (names "Encoding Journal", surfaced-only catalog) — reconcile it WITH the registered prompt here (deferred from P4 review #2).

---

## Backing brain nodes

- Input architecture: `823c0d3e` (provenance ledger), `07d269a8` (sandwich layout),
  `82f563d4` (judgment/factual split), `286178b7` (ENCODE-ON-IDLE doc), `c9444a17`
  (Tom's inline direction).
- Journal/residue: `aa1f3ece` (two feeds), `5de09c90` / `ce333a73` (residue principle),
  `233a7d10` (K constant), `6494d789` / `e78cfba6` (Q4 replacement-before-removal),
  `4e96bcb3` (journal design doc).
- Refinements: `893cf8c6` (temporal prose-not-structure), `f703bd9f` (examples are
  load-bearing), `accf5172` (first-person voice), `eaf833c5` (more signal ≠ better —
  the counter-pressure: widen with care).
- Eval: this session's audit (longmem trustworthy on recall dims; 3 journal dims
  unbuilt; `s1_encode_eval` dry-run; version-pinned stops).
