# S1 Encoder — feed it what Anchor actually touched

**Status:** Spec, not started. The cross-session-leak prerequisite (`session_id` populated on every trace_event) **landed 2026-05-02** — Part A in §design is no longer blocking. Pick up at Part B (new trace emissions).

## Problem

The encoder runs every 5th Stop. It receives:
- Conversation transcript (last 5 turns)
- Node catalog built from `surface_selected` IDs only
- Encoding journal + session arc

This misses every node Anchor engaged with that **wasn't** surface-selected by Haiku. Specifically:

- Anchor authored via `remember` — invisible
- Anchor revised via `revise` — invisible
- Anchor connected via `connect` — invisible (unless an endpoint was surfaced)
- Anchor looked up via `get_node` / `find_node_by_title` — invisible
- Anchor recalled via `recall` / `recall_batch` — results returned but if Haiku didn't select them in the same turn, invisible

Result: the encoder can't build texture around nodes Anchor consciously committed mid-session, can't see the working-memory trail, and risks duplicating principles Anchor already locked.

## What works today

`build_node_catalog` in [encode_contract.py:66](servers/scales/s1/encode_contract.py:66) extracts node IDs from `judge_outputs` (surface_output strings) via regex `id:([a-z0-9_]{6,8})`. It fetches each via `brain.get_node` and renders full rich. Bounded by Haiku's selection (~3-5 per turn × 5 turns = ~25 nodes max).

## Trace coverage map (audited 2026-05-17)

| MCP handler | Trace emitted? | session_id plumbed? |
|---|---|---|
| `_handle_remember` | ✗ none | n/a |
| `_handle_remember_batch` | ✗ none | n/a |
| `_handle_revise` | ✓ via the mutation emitter (`mutations` manifest → `ref_type=node_revised`) | yes (captured at the dispatch chokepoint) |
| `_handle_revise_batch` | likely yes | likely yes |
| `_handle_connect` | ✓ via `_emit_edge_revise_trace` (`ref_type=edge_relation_revised`) | yes |
| `_handle_get_node` / `_handle_get_nodes` | ✗ none | n/a |
| `_handle_recall` (S1R chain) | ✓ but only when called from `hook_recall`; direct MCP recall has no S1R chain | partial |

## The blocker — cross-session leak (handled by other window)

Trace events have an `idx_trace_session` index and a `session_id` column. The dispatcher accepts `session_id` and passes it forward. But every event in production has `session_id=''`. Chain_id encodes session_short (`s1r-43c7e545-17`), so the data is recoverable via regex but not via column query.

The other window of me is fixing this. **This spec depends on that fix landing first** — once the column populates reliably, `WHERE session_id = ?` becomes the foundation for the encoder query.

## Design — bounded, tagged, layered

Encoder gets a second source alongside the existing surface-derived catalog: "what Anchor touched this 5-turn window," sourced from trace_events filtered by `session_id + created_at >= window_start`.

```
Default: include everything from the window — typical 5-turn case is small.
Truncate guardrail: soft cap ~30 total nodes with priority order if exceeded.
```

| Layer | Source | Render | Tag in catalog |
|---|---|---|---|
| L0 — Committed | `node_revised` + `edge_relation_revised` + (new) `node_created` traces | full rich (S1_NODE_CONFIG) | `[anchor-authored]` / `[anchor-revised]` / `[anchor-connected]` |
| L1 — Surfaced | existing `surface_selected` extraction | full rich | (existing, untagged) |
| L2 — Considered | (new) `node_lookup` traces from `get_node` / `get_nodes` | title + situation + 1-line content | `[anchor-looked-up]` |
| L3 — Searched | existing `recall` traces (if direct MCP recall gets a chain) | title only | `[anchor-searched]` |

**Priority when over cap:** drop from L3 → L2 → L1 → L0 (last). Anchor's deliberate commits are the highest-value signal; raw recall results are noise without engagement.

**Deduplication:** a node appearing in multiple layers gets the highest-priority tag, rendered once.

## Implementation plan

### Part A — ✅ DONE (2026-05-02)
Cross-session-leak fix landed. Every trace_event now stamps `session_id` reliably (verifiable by sampling any 5 traces). `WHERE session_id = ?` is the foundation for the encoder query below.

### Part B — new trace emissions (this work)
Add to `daemon_dispatch.py`:

1. `_handle_remember` and `_handle_remember_batch`:
   - Emit `ref_type='node_created'`, `scale='s0'`, `event_type='delta'`
   - `ref_id` = created node ID
   - `summary` = `'created [%s] %s' % (type, title[:60])`
   - `metadata` = `{'encoding_source': args.get('encoding_source'), 'locked': args.get('locked', False)}`
   - Pass `session_id` from args

2. `_handle_get_node` and `_handle_get_nodes`:
   - Emit `ref_type='node_lookup'`, `scale='s0'`, `event_type='observation'`
   - `ref_id` = node ID
   - Lightweight — no metadata bloat (this fires often)
   - Pass `session_id`

### Part C — extend `build_node_catalog`

Change signature to also accept `(brain, session_id, window_start_iso)`:

```python
def build_node_catalog(judge_outputs, brain, session_id=None, window_start=None):
    # Existing: extract IDs from judge_outputs (L1)
    surfaced_ids = _extract_from_judge_outputs(judge_outputs)

    # New: extract from trace_events
    committed_ids = set()    # L0: node_created, node_revised, edge_relation_revised endpoints
    looked_up_ids = set()    # L2: node_lookup
    searched_ids = set()     # L3: recall results

    if session_id and window_start:
        rows = brain._trace_dal.get_by_session_window(
            session_id, window_start,
            ref_types=['node_created', 'node_revised', 'edge_relation_revised',
                       'node_lookup', 'recall'])
        # group by ref_type into the buckets above

    # Layer, tag, dedupe, cap at 30
    return _render_layered_catalog(committed_ids, surfaced_ids,
                                   looked_up_ids, searched_ids, brain)
```

Caller in `encode.py:345` passes `session_id` (already available via `ctx.session_id`) and `window_start` (5-turn lookback or from the last `encoding_run` trace).

## Risks / open decisions

1. **Cap of 30 — gut number.** Should measure typical 5-turn anchor-touch counts on real sessions before committing to a number.
2. **`get_node` traces could be noisy.** If get_node is called 50 times in a window (e.g., a graph audit), the trace table bloats. Mitigation: don't trace if `get_node` is called by another trace producer (caller-pass-through flag), OR cap node_lookup ingestion in the catalog at top-N most-recent.
3. **Locked-node hint.** Encoder needs to know `[anchor-authored]` nodes with `locked=True` are axioms — build around, don't rewrite. Encoder prompt may need a one-line addition explicitly.
4. **Recall trace coverage.** Direct MCP `recall` doesn't always run through S1R; if it doesn't get a trace, L3 will undercount. Decide if direct recall needs its own trace emission too.
5. **Prompt impact on caching.** New catalog sections go in the 5m-cached body, not the 1h preamble. Safe — confirmed during design.

## Why this matters

Today the encoder learns about Anchor's working memory secondhand, through Haiku's relevance bets. Adding the trace-sourced layers gives the encoder direct visibility into what Anchor consciously committed, revised, looked up, and considered. That's the substrate for connections, texture, and refusing to duplicate locked principles.

The mission stays: **the encoder should never have to make a recall to do its job.** This change keeps that invariant — it just widens the pre-packaged context to include more of what Anchor actually did, not what Haiku predicted Anchor cared about.
