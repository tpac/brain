# Temporal Architecture — How the Brain Handles Dates

**Last updated:** 2026-05-11

This doc consolidates the temporal arc from this session into a single reference. It covers the four kinds of dates the brain represents, the layers that handle them, the bugs found and fixed, and the known remaining gaps.

## Four kinds of dates the brain handles

| Date kind | Property of | Volatility |
|---|---|---|
| **Creation date** (`created_at`) | when WE encoded the node | stable |
| **Update date** (`updated_at`) | when WE last revised the node | stable |
| **Event date** (`event_time` kv) | when the EVENT happened in the world | stable |
| **Sequence** (`temporal_sequence` edges, Allen relations) | how events ORDER relative to each other | stable |

System-given dates (creation/update) are always exact. Content-given dates (event, sequence) can be exact, exact-relative ("8 days ago"), fuzzy-exact ("about a week ago"), or fuzzy-relative ("a few weeks before X").

## The three-layer responsibility split

Each integration unit owns one slice of temporal work, at its natural resolution:

### S1 Encoder — capture and anchor
Reads the conversation, extracts dates, resolves relative phrases against the conversation timestamp, writes structured fields.

- Resolves "today / yesterday / last Tuesday / N weeks ago" to ISO at encode time
- Writes `event_time: "<ISO>"` in metadata_kv on any node referring to a specific moment (events, decisions, moments, dated facts) — NOT limited to `event` type
- Creates `time_anchor` nodes ONLY when the date is itself topical (named day, anniversary, hub for 3+ events) — not for bookkeeping dates
- Composes Allen-vocabulary edges (`before`, `after`, `meets`, `met_by`, `during`) between events with clear temporal flow

**Discipline:** preserve fuzziness when the operator gave it; don't fabricate a date when there's no anchor.

### S1 Recall / Surface — temporal translation at query time
Reads the query, decides which retrieval shape fits, translates volatile expressions to stable lookups.

- Routes intent via the agentic toolbox: `recall_recent(window)`, `recall_by_time(start_when, end_when, time_anchor, query)`, `recall_topical(query)`
- Computes deltas at render time: given `event_time` + `brain_now()`, renders "X days ago" relative phrases
- Exposes `event_time` as a structured render line (added 2026-05-11) — answerer doesn't need to parse prose
- Walks `temporal_sequence` edges for "before X" / "after Y" queries

### S2 Healer — temporal skeleton enrichment (designed, not built)
Cross-session background pass that strengthens the temporal graph.

- **Dangling anchor resolution** — node says "before the move" but "the move" wasn't dated at encode time; later it gets dated → healer can now resolve the relative date
- **Implicit sequence edges** — encoder captured two purchase events without linking them; healer detects co-pattern, adds `temporal_sequence` edge
- **Date propagation** — if A `before` B and B has date X, A's date range narrows
- **Cross-session temporal consolidation** — "we discussed X across 4 sessions" → healer builds a timeline node anchoring all four

**Status:** designed in this session, not implemented. Clean slot in the existing healer architecture (same idle cycle, same `Haiku + revise()` machinery, no schema change).

## Bugs found and fixed in this session

### Bug 1 — Temporal scout used wall-clock (FIXED)
[servers/scales/s1/scouts/muster.py:69-70](../servers/scales/s1/scouts/muster.py) defaulted `current_date = _dt.date.today()` when not supplied. [servers/scales/s1/encode.py](../servers/scales/s1/encode.py) didn't pass `current_date` → fallback to NOW.

In eval (replay of historical conversations), the scout resolved "today / yesterday / last Tuesday" against the real wall-clock (2026-05-11) instead of the conversation date (e.g. 2023-03-19). Trace evidence on `gpt4_b0863698`: scout candidates included `2026-05-11, 2026-03-05, 2026-05-10` — all relative phrases mis-anchored to NOW.

**Fix:** [servers/clock.py](../servers/clock.py) — single source of truth for "now" across S1/S2. `brain_now()` (operator wall-clock) + `conversation_now(messages)` (reads `[Current date: ...]` replay prefix or session start). Contract test ([tests/test_clock_contract_sync.py](../tests/test_clock_contract_sync.py)) prohibits direct `datetime.now()` / `date.today()` calls in S1/S2 code; exceptions are marked `# clock-ok`. See brain memories `6d5b789e` (bug) and `dcc093464` (architecture).

### Bug 2 — Encoder under-complies on event_time kv (PARTIALLY FIXED)
v14 / v15.6 / v15.7 prompts all instruct the encoder to write `event_time` in metadata_kv on dated nodes. Compliance: ~0% across multiple runs. Sonnet wrote dates in CONTENT prose but skipped the structured field.

**Root cause** (found via coherence probe): the canonical "five nodes" worked example was stale across multiple rules. Sonnet imitates examples more than rules; the stale example silently overrode the newer rule.

**Fix:** [eval/prompts/s1e_v15_8.txt](../eval/prompts/s1e_v15_8.txt) rebuilt the canonical example to demonstrate event_time on event + moment nodes, voice symmetry (operator + Anchor + third-party), cross-redundant numbers (specific values in raw_quote AND content), proper per-node `connect_to` placement.

**Result:** v15.8 raised event_time kv compliance from 0% (v15.7) to ~5-8% — a 10× improvement. Still far from 100%; suggests prompt-only enforcement has a ceiling. See "future arcs" below.

### Bug 3 — Surface didn't expose event_time to answerer (FIXED 2026-05-11)
Even when encoder writes event_time, the render didn't expose it as a structured field. The kv data lived inside the masked content blob; answerer parsed dates from prose if at all.

**Fix:** [servers/scales/s1/surface_contract.py](../servers/scales/s1/surface_contract.py) `_event_time_line()` — when a node carries event_time, render a dedicated `Event date: <ISO>` line ABOVE content in all three modes (fact / arc / background). Bypasses activation thresholding because temporal anchors are structural, not state-of-mind.

**Design decision: absolute-only, no relative gloss.** First draft rendered `Event date: 2023-05-21 (36mo ago)` — the relative phrase computed against `brain_now()`. Tom flagged this: the relative is anchored to wall-clock now, which is a meaningless reference inside the event_time field (the answerer can derive recency from create/edit dates if it needs to). The relative gloss also recreates the wall-clock-vs-conversation-date ambiguity that Bug 1 eliminated. Final render is absolute ISO only; recency-computation belongs to the answerer's reasoning step, not the render.

### Bug 4 — L3 surface selection collapses on temporal queries (OPEN 2026-05-11)
Post-L4-fix eval revealed the next-layer bottleneck. Example: `gpt4_85da3956` — gold node is present in the recall pool at rank #2, render now exposes its event_time line correctly, but the Haiku surface still returns `selected=[]` for the temporal query. The L4 fix removed the answerer-side parsing burden; it exposed that L3 (surface selection) abstains when the query is dated and the candidate pool is mixed.

**Hypothesis:** the surface prompt's selection heuristic doesn't weight `Event date:` lines the way it weights topical relevance. The new render line is structurally privileged in the answerer's view but invisible to the surface ranker's decision policy.

**Status:** addressed via the `entity_dates` interval index + `recall_by_time` (2026-05-15 evening session — see SESSION-HANDOFF.md). Surface prompt now teaches "dates as entity-selectors, not retrieval filters" via Example 7. Surface tool routes to `recall_by_time(time_anchor='event')` when the query carries dates. Tracked in [BACKLOG.md](BACKLOG.md).

## `entity_dates` interval index (added 2026-05-15 evening)

Closes the "Bug 4" gap above. Single canonical table where every extracted date interval lives, queryable by half-open Unix-second range — the same primitive every production temporal-RAG system uses (SQL:2011 PERIOD, XTDB bitemporal, PostgreSQL `tstzrange`).

**Schema** (`servers/schema.py`):
```sql
CREATE TABLE entity_dates (
  entity_kind TEXT CHECK(entity_kind IN ('node','edge')),
  entity_id TEXT NOT NULL,
  start_ts INTEGER NOT NULL,    -- Unix epoch seconds
  end_ts INTEGER NOT NULL,
  extraction_source TEXT NOT NULL,  -- 'node.title' | 'node.kv:event_time' | 'edge.description' | '_no_dates_found'
  raw_text TEXT,
  created_at TEXT,
  PRIMARY KEY (entity_kind, entity_id, start_ts, end_ts, extraction_source)
);
```

Precision baked into the interval — `"May 2023"` → `(2023-05-01, 2023-05-31)`, `"2023-05-22"` → `(May 22, May 22)`. Every fuzzy/range query collapses to one SQL predicate: `start_ts <= ?query_end AND end_ts >= ?query_start`.

**Extraction** (`servers/temporal_extraction.py`):
- Regex-first, year-required. Five patterns: Q[1-4] YYYY → ISO day → ISO month → MonthName range → MonthName Day Year → MonthName Year.
- No dateparser fallback at extraction time — bare "December" / "May" without year context is intentionally skipped to avoid the current-year-fallback class of bug.
- Scans node `title`, `content`, and KV fields (`event_time`, `event_date`, `when`, `situation`, `reasoning`, `source_context`, `user_raw_quote`, `anchor_raw_quote`).
- Scans edge `description` + `relation`.
- Sentinel row (`extraction_source = '_no_dates_found'`) marks "processed, no dates" so the indexer's left-join doesn't re-scan empty-result entities.
- Cap of 20 intervals per entity to bound pathological inputs.

**Pipeline integration** — runs in `embed_queue` alongside vector backfill. Same async worker, same lock discipline. Boot-time auto-enqueue catches entities lacking rows. Symmetric for nodes and edges.

**Query tool** (`recall_by_time` in `servers/scales/s1/fetch_tools.py`) replaces `recall_by_date`:
- `start_when` / `end_when` natural-language (parsed via `dateparser` at query time only).
- `time_anchor` enum: `event` (default — filters on extracted intervals), `created`, `updated`.
- Optional `query` for semantic AND-combine; 3-tier ranking (query ∩ time → query only → time only).
- Edges → endpoints unwrap: when an edge matches, response includes source + target nodes for context.

**On the live brain at session end**: 19,842 rows total, 555 real intervals, 19,287 sentinels. ~10.8% of nodes and 0.1% of edges have explicit dates. 40 unit tests in `tests/test_temporal_extraction.py`.

## Allen interval algebra adoption

Adopted 9-relation Allen algebra, then trimmed to **5 core + emergent for the rest**:

- `before` / `after` — non-adjacent ordering
- `meets` / `met_by` — adjacent ("right after", "just before")
- `during` — nested

Richer relations (`overlaps`, `contains`, `simultaneous_with`, `starts`, `finishes`) emerge when the encoder needs them; not forced. Per brain memory `0762f572`.

## kv-first / time_anchor-only-when-hub principle

Most dates don't need their own node. The `event_time` kv stamp on the event-bearing node IS the spine. Create a dedicated `time_anchor` node ONLY when:
- The date is itself the TOPIC (named day, anniversary, public event)
- 3+ events already anchor to that date (becoming a hub)
- Operator names the date as a noun ("on March 19, ..."), not adverbially ("yesterday I did X")

S2 healer is the natural promoter — when it observes a date shared across 3+ events, it promotes the kv to a time_anchor node and links them. This matches the brain's lazy-promotion philosophy for types and relations (per memory `6ea4cbac`).

See brain memory `aeb41fc9` for the principle in full.

## Known remaining gaps

### Generative rule compliance ceiling
Sonnet under-complies on positive structural-write rules even with worked examples (~5-8% on event_time, 0% on Allen edges). Either:
- **(a) Code enforcement** — dispatcher detects "node has dated content but missing event_time" and auto-extracts/logs (precedent: `related/related_to` ban via dispatcher per memory `c39b8cc8`)
- **(b) Multi-iteration prompt sharpening** with the agent-introspection probe family — keep adjusting examples until compliance rises
- **(c) Accept the ceiling** — content prose carries dates well enough for some queries

Status: open architectural decision.

### Generic kv field promotion (Tom's note, 2026-05-11)
Current render hard-codes the event_time line. The general pattern: **query-aware kv field promotion** — a temporal query promotes `event_time` / `created_at`; a "what did X say" query promotes `user_raw_quote`; a "why" query promotes `reasoning`. Today only event_time is promoted, hard-coded. Future: generalize via the existing `field_activation` scoring (the "cousin filtering" mechanism Tom referenced) so any kv field can promote when query-relevant.

### Temporal-derivation queries still hard
"How many weeks ago" / "how old was I when" require computing deltas. The render now exposes structured timestamps so the answerer CAN compute, but the answerer still abstains in some cases. v15.8 demonstrated the architecture works (e.g. `gpt4_b0863698` answers "7 days ago"); some items still flip on sampling variance.

### S2 healer temporal enrichment NOT BUILT
Healer currently fills `question` / `situation` / `reasoning` only. Designed extension (date propagation, dangling anchor resolution, implicit sequence edges) is a clean slot, not yet implemented.

## Files involved

| File | Role |
|---|---|
| [servers/clock.py](../servers/clock.py) | `brain_now()` + `conversation_now()` — single source of truth for "now" |
| [servers/scales/s1/scouts/muster.py](../servers/scales/s1/scouts/muster.py) | Passes `current_date` to temporal scout (no longer wall-clock-defaulted) |
| [servers/scales/s1/encode.py](../servers/scales/s1/encode.py) | Calls `conversation_now()` and threads through |
| [servers/scales/s1/scouts/temporal.py](../servers/scales/s1/scouts/temporal.py) | Extracts date candidates, computes ISO from relative phrases against `current_date` |
| [servers/scales/s1/surface_contract.py](../servers/scales/s1/surface_contract.py) | `_event_time_line()` + render integration |
| [eval/prompts/s1e_v15_8.txt](../eval/prompts/s1e_v15_8.txt) | Canonical encoder prompt with temporal section + rebuilt canonical example |
| [tests/test_clock_contract_sync.py](../tests/test_clock_contract_sync.py) | Contract test forbidding direct wall-clock calls |

## Brain memories to consult

- `aeb41fc9` — Temporal anchoring principle (kv-first, time_anchor when hub)
- `6d5b789e` — Bug 1 (wall-clock)
- `dcb5b951` — `brain.now()` architecture
- `9352fcec` — v5 agentic surface eval reframe
- `0762f572` — Allen interval algebra: 5 core + emergent
- `f3940679` — Temporal scout: anchors vs event-relations split
- `b9bf4cd2` — `has_relational_marker` flag mechanism
- `116140c8` — Content prose IS the temporal channel; event_time kv compliance ~0%
- `928f5694` — Canonical example IS the training pattern
- `92b890e7` — Encoder compliance asymmetry (restraint 100%, generative ~0-20%)
- `5e27a23f` — Code enforcement for mandatory metadata fields (dispatcher precedent)
- `6ee9c770` — Agent Introspection probe family
