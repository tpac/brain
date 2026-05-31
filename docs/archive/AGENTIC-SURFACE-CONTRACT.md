# Agentic Surface — Contract (Milestone-1 → v6 update 2026-05-15)

**Status:** Eval-tested through v6 design. Production opt-in via
`BRAIN_SURFACE_VARIANT=v5_agentic` env var. Live brain stays on surface v4
until explicit activation.

**Why this exists:** the failure-walk on `eval_a_v15_6_2026_05_10` revealed
that ≤15% of failures are encoder-layer; the rest are recall/surface/render.
The quote scout returns silent-partial on 70% of items; surface returns
`selected=[]` on 5 of 13 failures. Encoder-prompt iteration is climbing
noise; the real leverage is making surface agentic with intent-shaped tools.

This contract locks the shape of the new surface BEFORE any code lands.

## v6 update (2026-05-15) — cluster-completion redesign

v5 (the original Milestone-1 design) shipped abstention-biased — Haiku
selected `selected=[]` on items with clearly-relevant candidates in the
top-25. Audit found 7 named contradictions in the v5 prompt (recognize-vs-
retrieve framing fighting the 6 fetch tools; "silence > wrong context"
appearing 3 times; 8 select-0 triggers vs 2 select-prefer triggers).

**v6 ([eval/surface_v6_prompt.txt](../eval/surface_v6_prompt.txt))**
replaces v5 with:

- **Cluster-completion default:** pick 3–5 nodes that frame the topic;
  let the answerer compose. Abstention belongs to the answerer, not surface.
- **Single select-0 trigger:** pure confirmations ("yes", "ok", "thanks").
  Everything else: surface delivers nodes.
- **Tool-empty fallback:** if Round 0 tools return 0 results, pick from
  the original 25 in Round 1. Never abstain because of a tool miss.
- **Parallel tool-use rule with example:** all fetches in ONE round
  (Anthropic API supports multiple `tool_use` blocks per assistant
  message; Haiku does it natively given the rule).
- **`max_rounds=2` (code-enforced):** `_call_surface_agentic` default
  lowered from 3 → 2 in `servers/scales/s1/surface.py`. Belt-and-braces
  for the prompt-level HARD CAP that Haiku occasionally bent.

12-item diverse eval (2026-05-15) showed v6 = 8/12 (67%) on the same
encoder substrate where v5 was 6/12 (50%) — clean +2 items, no
regressions. Direct fixes: `54026fce` (coffee breaks, was ENCODE_MISS),
`8e91e7d9` (siblings, was SURFACE_MISS). One persistent issue
(`75832dbd`) is a render-size bug, not a prompt issue (Haiku hits
`stop=max_tokens` at 8192 tokens before emitting JSON on this item).

**Ship status:** v6 is the candidate. Not yet registered to live brain.
Awaiting v14 + v6 isolation eval to confirm surface-only ship beats
v14 + v4 baseline.

---

## 1. Tool surface — six tools

Each tool is a typed wrapper over an existing `brain.*` primitive. The
**tool name carries the intent** — Haiku picks a tool, that picks the intent
(no separate classifier). All tools return the same candidate shape so the
downstream pipeline (spread activation, render) doesn't need new branches.

| Tool | Wraps | Intent it serves |
|---|---|---|
| `recall_topical` | `brain.recall()` | "Find things similar to X" — current cosine path |
| `recall_recent` | `TraceDAL.get_session_turns()` + node lookup | "What did we work on lately" — continuation queries |
| `recall_by_date` | `brain.filter_nodes(field='created_at', gte/lte=...)` | "What happened on/since/before date" — temporal queries |
| `recall_verbatim` | FTS5 path from `brain_recall.py` | "What did X *say*" — exact-quote lookup |
| `recall_by_aspect` | `brain.aspects` + `brain.filter_nodes(field='type')` | "Show me corrections / principles / open threads" — semantic-family scoped |
| `expand_node` | `traverse()` from `pipeline_contract.py` | "Tell me more about THIS specific thing" — constellation expansion |

### 1.1 Signatures (Python-side; JSON schemas exposed to Haiku in §2)

```python
def recall_topical(brain, query: str, k: int = 25) -> List[Dict]:
    """Topical semantic recall — embeddings + FTS5 union. Current cosine path."""

def recall_recent(brain, session_id: str, window: str = "last 10 hours",
                  k: int = 25) -> List[Dict]:
    """Chronological session-aware recall.

    `window` accepts natural language: 'last 10 hours', 'last 3 turns',
    'today', 'yesterday', 'since last session'. Tool parses to timestamps —
    Haiku doesn't compute dates.
    """

def recall_by_date(brain, when: str, k: int = 25) -> List[Dict]:
    """Date-bounded recall. `when` accepts:
      - 'yesterday', 'today', 'last week', 'this morning'
      - 'on 2026-05-09', 'since 2026-05-01', 'before 2026-04-30'
      - dict {'since': 'YYYY-MM-DD', 'until': 'YYYY-MM-DD'}
    """

def recall_verbatim(brain, phrase: str, k: int = 10) -> List[Dict]:
    """FTS5 verbatim phrase lookup. For 'what did X say' queries where
    the exact wording matters. Skips embedding similarity entirely."""

def recall_by_aspect(brain, aspect: str, recent_first: bool = True,
                     k: int = 25) -> List[Dict]:
    """Recall by semantic family. `aspect` is one of the 14 names:
      identity_bearing, episodic_anchor, active_thread, lesson_insight,
      correction_improvement, contradiction_conflict, ...
    Tool resolves aspect → node_types via brain.aspects."""

def expand_node(brain, node_ref: str, hops: int = 1) -> List[Dict]:
    """Constellation expansion from a known node. `node_ref` is either
    a node_id (8 chars OK) or a fuzzy title match. Returns neighbors at
    `hops` distance."""
```

### 1.2 Return shape (all six tools)

```python
[
    {
        "id": "8charid",
        "title": "...",
        "type": "lesson",
        "score": 0.78,         # tool-specific scoring; topical=cosine, etc.
        "content": "...",      # may be truncated to surface budget
        "kv": {...},           # situation, reasoning, user_raw_quote, anchor_raw_quote
        "source_tool": "recall_recent",  # for trace observability
    },
    ...
]
```

This matches the existing `candidates_data` shape consumed by spread
activation and render. No downstream changes required.

---

## 2. Haiku tool-use flow

### 2.1 Tool definitions exposed to Haiku

Standard Anthropic `tools` parameter. Each tool gets a JSON schema:

```json
{
  "name": "recall_recent",
  "description": "Chronological session-aware recall. Use when the user signals continuation: 'what did we do', 'last session', 'this morning', 'pick up from yesterday'. NOT for topic queries.",
  "input_schema": {
    "type": "object",
    "properties": {
      "window": {
        "type": "string",
        "description": "Natural-language window: 'last 10 hours', 'last 3 turns', 'today', 'yesterday', 'since last session'."
      },
      "k": {"type": "integer", "default": 25}
    },
    "required": ["window"]
  }
}
```

Tool descriptions are the **intent vocabulary** — write them so the
distinction between tools is unambiguous from the docstring alone.

### 2.2 Loop control

- **Max 3 rounds.** A round = Haiku thinks → tool calls execute → Haiku sees results.
- **Parallel tool calls in one round are encouraged.** Anthropic supports
  multiple tool_use blocks per assistant message; the loop executes them
  all then feeds results back together.
- **No cap on tools per round.** Behavioral discipline (in prompt) prevents
  iterating the same query in variations.
- **After max rounds:** Haiku produces a final selection from whatever pool
  accumulated. If still no selection emerges, fall back to cosine pre-seed.
- **Cosine pre-seed remains:** the initial candidate pool comes from
  `daemon_hooks.py` as today. Tools ADD to that pool, don't replace it.

### 2.3 Response shape

```json
{
  "selected": [
    {"id": "8charid", "why": "carries the operator's open question on X", "mode": "fact"},
    {"id": "8charid", "why": "background context for partnership shape", "mode": "background"},
    ...
  ],
  "reason": "..."         // optional, when selected=[]
}
```

- `mode` ∈ `{"fact", "arc", "background"}`. **Default `"arc"`** if omitted (backward-compatible with v4).
- `why` ≤ 80 chars — same as v4 contract.
- `reason` populated only when `selected=[]`. Names what's missing.

---

## 3. Mode-aware render

Render branches on `mode` per node:

| Mode | Behavior |
|---|---|
| `fact` | Emit `title` + full `content` verbatim. NO field masking. NO activation threshold. Used for verbatim facts, exact quotes, specific values. |
| `arc` (default) | Current activation-thresholded path. Field masking by `_FIELD_RENDER_THRESHOLD`. Identity/state-of-mind nodes. |
| `background` | Title + 1-line situation summary only. Low-weight contextual. |

Budget allocation: same softmax distribution as today, but `fact` nodes
get a 1.5× weight bump to ensure their content survives truncation.

---

## 4. Runtime gating

**Feature flag (env var):**

```bash
# Default: agentic path OFF, surface uses v4 path identically to today.
BRAIN_SURFACE_VARIANT=v4    # current behavior

# Opt-in: agentic path ON, uses tools + mode-aware render.
BRAIN_SURFACE_VARIANT=v5_agentic
```

Read once at `_call_surface()` entry. No mid-session toggling.

**Active-version pairing:** when `v5_agentic` is set, surface reads the
`active_version` of the `surface` interaction (now decoupled from
registration — see Step 0). To run v5 in eval, the harness:
1. Registers v5 prompt via `register_interaction` (does NOT activate).
2. Calls `set_interaction_active('surface', <new_version>)` in eval brain.
3. Sets `BRAIN_SURFACE_VARIANT=v5_agentic` for the eval process.

**Production stays on v4** until you set the env var AND flip
`set_interaction_active` on the live brain. Rollback: unset env or
`set_interaction_active` back to v4. One restart away from old behavior.

---

## 5. Trace contract extensions

Three new `ref_type` values under `scale='s1'`, `event_type='K'`:

| ref_type | When emitted | Metadata |
|---|---|---|
| `tool_call` | Per tool invocation (one event per call in a round) | `{tool, args, result_count, latency_ms, round_idx}` |
| `tool_round` | Per round boundary | `{round_idx, tools_called, total_tools, elapsed_ms}` |
| `surface_variant` | Once per surface call | `{variant, prompt_version}` |

These additions register in `servers/trace_contract.py`. Existing
`surface_selection`, `surface_response` event types unchanged.

---

## 6. What this does NOT touch

Explicitly out of scope for Milestone-1:

- **Encoder (S1 Scribe).** Stays on s1e v14.
- **Scouts.** Quote/temporal/facts unchanged. Synthesis stays disabled.
- **S2 maintenance.** Consolidation, community, healer, aspect integration unchanged.
- **Spread activation.** Same algorithm, same params.
- **Frame structure.** Same boot prior, same sections.
- **Brain primitives.** `recall`, `filter_nodes`, FTS5, traverse — single source
  of truth, tools wrap them.
- **Schema.** No new tables, no column additions. (Step 0's `interaction_active`
  table is its own prerequisite, separately landed.)
- **Bi-temporal edges.** Deferred to a later milestone.

---

## 7. Eval contract

The eval is the test. What it must show:

**1. Architecture works end-to-end (smoke gate).**
   - Tools callable, Haiku invokes them, results flow into context
   - No pipeline errors on 5 smoke items
   - Tool-call traces present with non-trivial distribution

**2. Targeted improvement** on 13 v15.6 failures, specifically:
   - L3 RECALL_BURIED (4 items): expect ≥2 pass via tool routing
   - L4 SURFACE_SKIPPED (2 items): expect ≥1 pass via tool routing
   - L5/L6 ANSWERER (3 items, 2 unanalyzed numeric): expect no regression
   - L2 ENCODER (2 items): not expected to improve (encoder unchanged)

**3. Noise-floor-aware comparison.** 3 seeds × 50 items × 2 conditions
(v4 baseline, v5 agentic). Mean pass rate ± std across seeds. The δ
between conditions must exceed within-condition std to be a real signal.

**4. Tool-call distribution.** ≥10% of turns invoke a non-topical tool.
If all turns route to `recall_topical`, intent routing is dead code.

**5. Behavioral signals.**
   - `with_anchor_raw_quote` distribution — no regression from v15.6
   - `open_nodes` count — no regression
   - `mode='fact'` selection rate — should be non-zero (validates the
     fact-render path is being exercised)

---

## 8. File-level placement

| File | Action | Concern |
|---|---|---|
| `servers/scales/s1/fetch_tools.py` | **NEW** | The 6 tools + dispatch loop + JSON schemas |
| `servers/scales/s1/surface.py` | EDIT | Env-gated branch; calls fetch_tools dispatch when v5 |
| `servers/scales/s1/surface_contract.py` | EDIT | Mode-aware render branch in `_render_node_activation` |
| `servers/trace_contract.py` | EDIT | Add 3 new ref_types |
| `eval/longmem/harness.py` | EDIT | Add `--surface-override` + auto-activate after register |
| `eval/surface_v5_prompt.txt` | **NEW** | Prompt v5 source file (NOT registered to live brain) |
| `tests/test_fetch_tools.py` | **NEW** | Per-tool unit tests against IsolatedBrain |
| `tests/test_surface_tool_loop.py` | **NEW** | Tool-use loop behavior tests |

---

## 9. Rollback

If anything breaks:

1. **In code:** unset `BRAIN_SURFACE_VARIANT` (or set to `v4`) → daemon restart → surface back on v4 path identically.
2. **In DB:** `set_interaction_active('surface', 4)` → runtime reads v4 prompt.
3. **At commit level:** revert the agentic-surface commits. No schema migration to undo; the `interaction_active` table from Step 0 is a permanent improvement either way.

---

**Author:** Anchor, 2026-05-10
**Approved direction:** Tom, this session
**Prerequisite landed:** Step 0 — active_version model (this session)
