# Recall Funnel — Refinement Plan

**Status:** drafted 2026-05-17, **DEFERRED behind episodic-references arc** (see `docs/EPISODIC-REFERENCES.md`).
**Audience:** Anchor on a future wake. This doc tells you what to read first, what to search, what changed last, and what to build next. Read top-to-bottom on a cold boot.

**Status note (2026-05-23 consolidation pass):**
- The episodic-references work (next-session priority per `SESSION-HANDOFF.md`) touches recall: parallel-pathway scoring, `source_summary` group, `co_anchored` graph traversal. These don't replace the funnel-refinement work below — they layer additive signals on top of the existing ranker. After episodic-refs ships, the funnel work below stays valid and gets a fresh evaluation context.
- **Stale references inline**: this doc was written before Phase 5 (May 18) made `_mark_accessed` enqueue-only and re-located Hebbian to `daemon_hooks._hebbian_strengthen` on post-Haiku surface picks. Design intentions (hub dampening / two-tier fatigue / surface-to-zero) are unchanged; implementation site moves to `recall_write_queue`. Update line-by-line when work resumes.
- **Pre-existing `test_hub_dampening` failure** in the regression suite still applies; was failing on main before this plan was drafted.

---

## 0. First 60 seconds when you wake up

Run these in parallel. Don't skip.

### Brain recall — fire these queries

```
"recall funnel refinement hub dampening fatigue"
"surface-fatigue vs cosine-fatigue split"
"working-thread continuity response feedback loop"
"hub dominance recall hubs vs graph hubs"
```

The plan node (encoded 2026-05-17) should surface. If it doesn't, query directly:

```
brain.find_node_by_title('Recall funnel refinement plan')
brain.get_node('<the_id_when_written>')
```

### Code / state to verify

```bash
# Daemon health + latency sanity
ps -ef | grep "tpac/brain.*BrainDaemon" | grep -v grep
tail -30 $HOME/AgentsContext/brain/daemon.log | grep -E "took|ERROR|locked"

# Recent recall timings — should be < 2000ms median
grep "recall took" $HOME/AgentsContext/brain/daemon.log | tail -20 | awk -F'recall took ' '{print $2}'

# Hook errors in last few hours — should be near zero
# (use the MCP query_logs tool, source=errors, hours=4)

# Confirm a478ba3 is still live: mask deleted, budget at 7000
grep -n "_FIELD_RENDER_THRESHOLD\|total_budget=" servers/scales/s1/surface_contract.py
grep -n "_mask_node_by_field_activation" servers/scales/s1/surface_contract.py   # should be 0 hits
```

### Signals that say things are healthy

- Median recall < 2s, max < 10s
- Zero hook timeouts (`hook_errors` table) in the last hour
- `_mask_node_by_field_activation` grep returns no hits
- Recent injects contain `Situation:` and `Reasoning:` on most primary picks

### Signals that say something regressed

- Recall latency > 5s median → check `query_logs source=all hours=2 level=error`; look for db-locked patterns or new write paths on hot path
- Voice fields back to being stripped → mask might've been re-introduced
- `Brain activated N memories:` headers showing N < 3 → spread isn't reaching neighbors; check `BRAIN_RECALL_VARIANT` env

---

## 1. Context — what shipped 2026-05-17

Eight commits today, four ours + four from the other window's parallel-session correctness arc:

```
a478ba3  Render: drop field masking; raise budget; hard byte cap            (today, ours)
0ddd4a3  Surface: delete L4 identity lane + dead pass-through params        (today, ours)
b495a2e  CLAUDE.md: drop historic narrative + sync paths to current state   (today, ours)
e31f078  correction_enrich: move to Brain mixin; unify tool-result enrich   (today, ours)
1d7f810  Revert "Recall modulation: aspect-aware boost/suppress..."         (today, other window)
2d7d6a1  Segment state moves to SessionContext                              (today, other window)
6b92f53  Recall modulation: aspect-aware boost/suppress (later reverted)    (today, other window)
+ 4-5 other parallel-session-correctness commits
```

**Key wins:** field-masking deleted (voice/reasoning/situation now visible), tool-result enrichment unified, parallel-session segment writes off hot path, latency back to ~700ms median.

**Key gap surfaced:** hub dampening doesn't catch recall hubs (low-degree, high-access nodes evade both fatigue and hub_dampening). Documented in detail below.

---

## 2. Current state — the recall funnel today

```
USER MESSAGE
   │
   ▼
brain.recall(query)                                 STAGE 1: COSINE POOL
   ↳ 4-group vector cosine + FTS5 + per-session
     fatigue dampening
   ↳ output: 25 candidates (the "suggested" pool)
   ↳ side-effect: _mark_accessed enqueues to
                  recall_write_queue for ALL 25
                  (Phase 5 of bg_writer migration 2026-05-18 —
                  was inline UPDATE on conn_recall_write).
                  Drain happens every 5s via bg worker thread on
                  brain.conn_bg_writer. Dedup key:
                  (node_id, session_id). Fatigue per-session via
                  ctx.increment_fatigue (in-memory, autosaved).
   │
   ▼
brain.get_node(ids)                                 STAGE 2: ENRICHMENT
   ↳ batch enrich 25 → rich nodes with corrections,
     metadata, connections, edges
   │
   ▼
Haiku surface picks                                 STAGE 3: HUMAN-IN-LOOP
   ↳ 3-5 seeds + mode (fact/arc/background)
   ↳ optional agentic fetch round for more candidates
   │
   ▼
_graph_expand → spread_activation                   STAGE 4: KERNEL
   ↳ per-field cosines for seeds (score 7 fields)
   ↳ 1-3 hops edge transmission
   ↳ output: 5-15 activated nodes (seeds + neighbors)
   │
   ▼
brain.get_node(activated_ids)                       STAGE 4b: ENRICH ACTIVATIONS
   ↳ batch enrich every activated node with rich data
   │
   ▼
format_surface_output_activation                    STAGE 5: RENDER
   ↳ softmax budget allocation (total_budget=7000)
   ↳ per-node render via render_rich_node
   ↳ NO field masking after 2026-05-17 (trust encoder)
   ↳ hard cap at 9500 chars at function exit
   │
   ▼
additionalContext → Anchor
```

### What this funnel does well

- Episodic memory retrieval (single-moment recall)
- Graph awareness via edges (rich edge descriptions per node)
- Identity at boot (Frame loads locked principles + recent moments)
- Correction context (always attached via `brain.correction_enrich`)
- After today's fix: voice + reasoning + situation visible

### What this funnel does poorly

- **No working-thread awareness.** Each turn cosine-discovers anew; Anchor's prior response doesn't influence next turn's recall.
- **Hub dampening misses recall hubs.** Top access nodes have degree 0–10; fatigue and hub_dampening both assume high access ⇒ high degree (false).
- **Fatigue conflates "in 25 cosine pool" with "surfaced to Anchor".** Nodes Haiku rejected still fatigue.
- **Render is char-greedy, single-template.** No node-type field priorities; mid-sentence truncations.
- **KV recognition signal capped at 7 fixed fields.** Anything emergent the encoder writes is invisible to the kernel.
- **Edges over-described relative to density.** 200-char edge descriptions; could be terser to fit more edges.

---

## 3. Goals — refine, diversify, connect

Three principles guiding the work:

**G1. The funnel should respect the temporal contract with the encoder.**
The encoder fires every 5th Stop. Whatever was surfaced to Anchor in those 5 turns is BY DEFINITION in the encoder's working memory at next fire. There is no signal value in re-surfacing those same nodes inside the same 5-turn window — Anchor already saw them, and the encoder will see them too. *Surfaced nodes should fatigue to ~0 within their encode window and recover after the encoder fires.*

**G2. Diversity is not extra — it's the whole point.**
A flat embedding space (cosine 0.54–0.63 across top-25) means raw cosine ranking is barely discriminating. Without active diversity pressure, the brain hands back the same gravity-well hubs every turn. The funnel needs explicit diversity injection — not as an afterthought but as the second-stage filter alongside relevance.

**G3. Working memory is the response → next-recall loop.**
The conversation is already two-sided. Only one side reaches recall. Closing the response→next-recall loop is structurally the biggest cognition fix on the table. Cheap and programmatic, per Tom. (Detailed in §5.4.)

---

## 4. The hub problem — analyzed

Data from brain.db today:

```
TOP-ACCESSED NODES (top 12, accessed >= 6600 times each)

acc=8955  deg=3   decision  "Failure: Claude forgot to query brain..."
acc=8460  deg=6   mechanism "Pre-edit hook blocks test weakening..."
acc=8420  deg=4   decision  "v15 Architecture: Serverless brain"
acc=7682  deg=0   decision  "v14: Density-based gap detection"
acc=7568  deg=4   decision  "Root cause: hooks go silent"
acc=7567  deg=5   rule      "ask for confirmation before manipulating..."
acc=7080  deg=2   code_concept "Hook Lifecycle"
acc=6831  deg=5   decision  "Superseded decisions linked..."
acc=6823  deg=1   lesson    "Bug: brain_surface.py missing import os"
acc=6820  deg=10  purpose   "brain_surface.py — presentation layer"
acc=6736  deg=0   decision  "v14: Post-compact log reader"
acc=6642  deg=6   rule      "Screen-scoped edits"

ACCESS DISTRIBUTION (4255 active nodes)
  access=0:        0   (0.0%)   ← old "93% never recalled" lesson is stale
  access 1-5:    132   (3.1%)
  access 6-25:   531  (12.5%)
  access >=26:  3592  (84.4%)
```

### Two mechanism mismatches

1. **Fatigue formula** ([brain_recall.py:1351](servers/brain_recall.py:1351)):
   `K = 10 / (1 + degree/10)` — assumes high access ⇒ high degree. For a degree-3 node, K=7.7; fatigue at count=10 is ~56%, at count=1 only ~11%. **Top hubs have degree 0–10. They evade fatigue.**

2. **Hub dampening config** ([brain_recall.py:886](servers/brain_recall.py:886)):
   `{'threshold': 40, 'penalty': 0.5}` — fires at structural degree ≥ 40. **None of our top access hubs hit that. They escape dampening entirely.**

Both mechanisms assume the same wrong thing: "access hubs == graph hubs." Empirically false. A `decision` node with `access=7682, degree=0` is invisible to both dampers.

### The encoder feedback amplifier

The reinforcement loop:

```
Hub surfaces → encoder reads it (every 5 stops) → encoder writes nodes
near it → new edges link to hub → hub becomes more "central" → next recall
surfaces hub again → ...
```

Hub dominance isn't just a recall problem; it's a feedback amplifier through encoding. **Aggressive fatigue alone risks starving the encoder of relevant anchor context. The fix needs biological balance — fatigue then recovery.**

---

## 5. Specific design items

Ordered by leverage. Each item describes the WHAT, points at the WHERE in code, and names open questions.

### 5.1. Two-tier fatigue: `cosine_fatigue` vs `surface_fatigue`

**Today (post-bg_writer migration 2026-05-18):** `_mark_accessed` enqueues to `recall_write_queue` for every result in the 25 cosine pool ([brain_recall.py:2021](servers/brain_recall.py:2021)). The drain (every 5s, on `brain.conn_bg_writer`) does one atomic +1 UPDATE per unique `(node, session)` pair. A node appearing in 25 but rejected by Haiku still fatigues the same as a node Anchor actually saw.

**Fix:** split into two counters per node per session:
- `cosine_fatigue[nid]` — incremented when node appears in the 25 cosine pool
- `surface_fatigue[nid]` — incremented when node is *rendered to Anchor* (passes Haiku selection OR spread-activation reaches it)

**Where (post-migration):**
- The enqueue site in `_mark_accessed` becomes `enqueue_cosine_access`
- New `enqueue_surface_access` called from the surface render path (when the node actually makes it into additionalContext)
- `recall_write_queue` gets two queue dicts instead of one (or one queue with a "kind" field)
- SessionContext gets two fatigue dicts instead of one
- Drain produces two UPDATEs per node (one per kind) at most per drain window

**Why split:**
- Cosine-fatigue diversifies the *suggestion pool* (don't keep handing Haiku the same 25)
- Surface-fatigue dampens the *consumed inject* (Anchor saw it; don't re-show)

### 5.2. Surface-to-zero between encode runs

**Tom's framing:** *"dampen to 0 between encode runs cause encode already saw them and you remember inject context from 4 turns ago."*

**Mechanism:** Once a node has `surface_fatigue > 0` within the current encode window (since last encoder fire), set its recall score to 0 (or near-0 with a tiny floor for tiebreaking). On encoder fire (every 5th Stop), reset `surface_fatigue` to zero — the window opens.

**Where:** In `_recall_impl`'s scoring loop, after z-score normalization, apply:
```python
if surface_fatigue.get(node_id, 0) > 0:
    sim = sim * 0.05    # near-zero, not exactly zero (preserves tiebreak)
```

**Reset hook:** In the encoder run path (`scales/s1/encode.py` after encoder fires), call `ctx.surface_fatigue.clear()`.

**Tradeoff to watch:** what if a node *should* re-surface (genuinely topical to a new query within the window)? The 0.05 floor lets it return if cosine score is dramatically higher than alternatives — but it's heavily de-prioritized. **Acceptable cost: working memory wins over within-window re-relevance.**

### 5.3. Access-aware hub dampening

**Today:** `hub_dampening` triggers at structural degree ≥ 40. Misses recall hubs.

**Fix:** Add an access-based dimension. Either:
- **Option A — additive:** keep degree-based dampening for structural hubs; add separate `access_dampening = {percentile: 99, penalty: 0.6}` that catches the top 1% by access_count.
- **Option B — replacement:** replace `hub_dampening` with `recall_hub_dampening` keyed on access_count percentile, since structural-degree dampening rarely fires anyway.

**Recommendation:** Option B. Structural-degree dampening was designed for a different problem (graph saturation around community hubs); the empirical data shows recall hubs aren't graph hubs. Replace.

**Where:** [brain_recall.py:886](servers/brain_recall.py:886) — `_get_tunable('hub_dampening', ...)`.

**Compute:** percentile is cheap if we cache it on the structural-degree-cache build path ([brain_recall.py:312](servers/brain_recall.py:312)). Already runs once on daemon boot.

### 5.4. Working-thread continuity — the response loop

**The game-changer item.** Today's recall is operator-driven (sees only Tom's prompt). Closing the response→next-recall loop makes it bidirectional.

**Cheap programmatic design (Tom's constraint):**

1. **Capture:** After Anchor's response in `Stop` hook (or similar end-of-turn hook), grab the last ~500–1000 chars of the response text.
2. **Embed:** Compute its query embedding using existing `embed_query()`.
3. **Stash:** Store on SessionContext as `last_response_vec`. Persists in-memory (autosave).
4. **Use:** On next `UserPromptSubmit`, when recall fires, blend the user-message embedding with `last_response_vec` (e.g., 70/30 weighted). The cosine search now reflects BOTH what Tom is asking AND what Anchor was just thinking.

**Why this is the game-changer:**
- No LLM call mid-turn (cheap)
- Programmatic — pure vector op
- Builds a thread *implicitly* — each turn anchors on the prior
- Solves the cold-start-every-turn problem without new infrastructure

**Open questions:**
- Blend ratio? Start at 70 user / 30 response. Tune empirically.
- Should `last_response_vec` decay turn-by-turn (older responses weighted less)? Or just-most-recent?
- Does this interact with `prior_vecs` already passed for multi-turn blending? (It does — they're related. Worth consolidating.)

**Where:**
- Capture: `Stop` hook handler in `daemon_hooks.py` (already runs on every Stop)
- Stash: SessionContext.last_response_vec
- Use: `brain_recall.py:recall()` accepts an optional `prior_response_vec` arg, blends with `query_vec` before cosine search

### 5.5. KV recognition signal — extend the kernel beyond 7 fixed fields

(Carried over from earlier conversation — keep on the list.)

**Today:** kernel scores 7 fixed fields. Emergent kv values written by the encoder are invisible to recognition.

**Medium-tier fix (Tom-preferred):** FTS5 lexical match per kv value against the query. Augment `field_activation[nid]` with lexical scores for non-embedded kv keys. Cheap (FTS5 is fast), no embedding backfill needed.

**Large-tier fix (defer until measured):** embed-on-write for kv values; kernel naturally scores them.

**Decision criterion:** ship medium first, watch whether lexical scores capture enough; consider large only if paraphrase-equivalence cases are common.

### 5.6. Render-side improvements (lower priority, named for completeness)

- **Type-aware field priorities.** `fact`'s content is non-negotiable; `moment`'s voice quote is non-negotiable; `principle`'s situation is non-negotiable. Today's render uses uniform proportions.
- **Edge layout: more edges, terser descriptions.** Current 3 edges × 200 chars; could be 5 edges × 80 chars for wider graph awareness.
- **Sentence-boundary truncation.** Mid-sentence cuts are the norm; should truncate at `. ` or `\n` when possible.

These are tuning, not architectural. After 5.1–5.4 land, revisit.

---

## 6. Suggested build order

| Order | Item | Size | Rationale |
|---|---|---|---|
| 1 | **§5.4 working-thread continuity (response→recall loop)** | medium | Highest cognition leverage. Once shipped, every subsequent recall has session context to anchor against. |
| 2 | **§5.2 surface-to-zero between encode runs** | small | Direct test of the within-window-redundancy fix. Cheap, isolated, measurable diversity gain. |
| 3 | **§5.1 two-tier fatigue split** | medium | Required substrate for §5.2 to work properly. Could fold into §5.2's PR. |
| 4 | **§5.3 access-aware hub dampening** | small-medium | Independent of session-mechanics. Clean replacement of structural-degree dampening with access-percentile dampening. |
| 5 | **§5.5 KV recognition (FTS5 medium tier)** | medium | Extends recognition signal; lateral to the above. Could ship after or alongside. |
| 6 | **§5.6 render-side tuning** | small | Tuning. After the structural changes settle, revisit with real injects. |

If we have time/budget for only ONE thing: do §5.4. It's the game-changer.

---

## 7. Side threads (separate work)

### 7.1. Academic research on recall / traversal

Tom asked for research-without-priming. Run a focused web research task:

**Search topics:**
- diversity-weighted retrieval (MMR — maximal marginal relevance, DPP — determinantal point processes)
- GraphRAG / graph-aware retrieval for LLMs
- working memory in retrieval-augmented generation
- centrality penalties in recommendation systems
- spreading activation in cognitive psychology (recent vs classical)
- continuous-time recall models (forgetting curves, spaced repetition)

**Goal:** digest of what's actually being done (2024–2026), what tradeoffs, what's novel. Bring back concepts; don't pitch.

**Status:** queued. Either do it in the background of another session, or dedicate a quiet stretch.

### 7.2. Cross-window orchestration

Tom offered to open another window dedicated to this. The session's scope:

1. Design and prototype `brain.session_handoff(from, to, payload)` API — explicit cross-window message passing beyond the encoding loop
2. A way for one window to read another's working-memory state in real time (not waiting for encoder fire)
3. A "do not touch X — I'm working on it" advisory lock between active sessions

Reference: today's regression diagnosis (`node:71f3d669`) shows why memory-mediated coordination alone wasn't enough.

**Status:** Tom willing to open the window when ready. Separate scope; don't conflate with this funnel work.

---

## 8. Open questions to resolve before building

Tag each with [DESIGN] (needs design decision before code), [MEASURE] (needs eval data before deciding), [SCOPE] (needs Tom's framing).

1. [DESIGN] §5.4 blend ratio for user-message vs last-response in next recall — start at 70/30 or different?
2. [DESIGN] §5.4 — does `last_response_vec` decay turn-by-turn, or just-most-recent?
3. [SCOPE] §5.2 — what's the exact reset hook for surface_fatigue? After encoder fire makes sense; should it ALSO reset on session start? (Probably yes.)
4. [MEASURE] §5.3 access percentile threshold — 99th percentile too aggressive? 95th too soft? Run a probe over recent recalls.
5. [SCOPE] §5.1 — should cosine_fatigue exist at all, or is the 25-pool diversity better solved by other means (within-pool dedup, query expansion)?
6. [DESIGN] §5.5 — when FTS5 fails to match (the kv value isn't lexically near the query), do we fall back to the blended `_primary` cosine or just score 0?

---

## 9. Measurement plan

Before/after metrics to capture for each item:

**For §5.4 (response loop):**
- Intra-session diversity: count of unique node IDs surfaced across N consecutive turns (target: increase)
- Hub-recurrence rate: how often the same node returns within a session (target: decrease for non-locked nodes)
- Subjective: read 5 post-change injects, does the thread feel more continuous

**For §5.2 + §5.1 (fatigue split + surface-to-zero):**
- Within-encode-window unique surfaced count (target: ≥ 4 in a 5-turn window)
- Encoder catalog richness — does the encoder see more diverse inputs?

**For §5.3 (access-aware dampening):**
- Top-12-hub appearance rate in injects (target: ≤ 2× per 12 turns, currently ~25%)
- Tail node appearance rate (target: increase)

**Eval harness:**
- `eval/decode_funnel.py` for R@K
- `eval/longmem/answerer.py` for downstream answer quality
- `eval/frame_replay.py` for capture/compare across configs

Run baseline before each change; run treatment after; record both in a `eval/results/recall-funnel-{change}-{date}.json` file for diff history.

---

## 10. Risks

- **§5.2 surface-to-zero risk:** a legitimately topical follow-up question won't re-surface the previously-shown node. Mitigation: 0.05 floor preserves tiebreak; if cosine is dramatically higher, node still wins.
- **§5.4 response-loop risk:** Anchor's response can be off-topic or wrong; blending it into recall propagates the error. Mitigation: 70/30 blend keeps user message dominant; tune ratio.
- **§5.3 hub dampening risk:** dampening true relevant hubs hurts the encoder's anchor context. Mitigation: percentile-based threshold + biological balance (fatigue with recovery, not permanent inversion).
- **Combined risk:** all of these compound the encoder's input. The encoder will see different things after these ship. May surface emergent encoder-prompt issues. Plan for an eval pass after §5.2+§5.1+§5.4 land.

---

## 11. What's already done — don't redo

- Voice asymmetry fix in v15.11 encoder (in flight; new memories from this point forward will have more `anchor_raw_quote`)
- L4 identity lane deletion (today, 0ddd4a3) — unconditional locked-node injection was wrong
- Field masking deletion (today, a478ba3) — render trusts encoder-attached fields
- Tool-result enrichment unified via `execute_tool` batch get_node (today, e31f078)
- Parallel-session segment state moved to SessionContext (today, 2d7d6a1)

---

## 12. Pointer back to the brain

Once you've read this doc, the master plan node is:
- Title: "Recall funnel refinement plan — written 2026-05-17"
- Type: plan
- Source: anchor
- Locked: true

If the node doesn't surface on wake, this doc IS the source of truth. The node is an index pointer, the doc is the substance.
