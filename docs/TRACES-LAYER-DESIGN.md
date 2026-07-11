# Traces Functional Layer — Consolidation Design (Option E)

**Status: PROPOSED** — design agreed in session 2026-07-11, implementation pending go.

## Problem

The traces substrate (`trace_events` + `trace_embeddings` in `brain_logs.db`) has one
DAL (`TraceDAL`) and one vocabulary contract (`trace_contract.py`) — but its
**functional layer is scattered across four homes**:

| Home | Charter of that file | Trace functions living there |
|---|---|---|
| `servers/brain_recall.py` | node recall | `query_traces`, `journal_notes`, `write_journal_notes` |
| `servers/brain_episodes.py` | (whole file) | `recall_episodes`, `_resolve_time_bound` |
| `servers/scales/s0/conversation.py` | "S0 scale layer" | `get_conversation`, `turns_since_last_encode`, `get_conversation_around` + JSONL-fallback helpers |
| `servers/dal_logs.py` (TraceDAL) | primitives | (correct — stays) |

The scatter has a measurable cost: **four operator corrections in one month**
(SessionContext→DAL, journal→DAL, coordinator, scribe_due) plus a 2026-07-11 audit
that found three more DAL bypasses (`s2/base.py` ×2, `s1/surface.py`) and one raw-SQL
scan (`_trace_chain_candidates`). Recurring misplacement is the definition of
"location is not obvious."

## Decision criterion: Generic-LLM natural structure

The maintainer population of this repo is LLM sessions. Placement quality is therefore
measured by three axes:

1. **Training prior** — does the pattern dominate public code, so a model predicts it cold?
2. **One-rule routing** — can the right door be derived without judgment? (Judgment
   calls are exactly where the four corrections happened.)
3. **Local imitation** — models copy the nearest idiom; one repo-wide convention
   outperforms any clever per-layer scheme.

Options rated (see session 2026-07-11 for full discussion):

| Option | Shape | Rating |
|---|---|---|
| A — status quo | semantics→scale module, filters→brain door | 2/5 (the correction record) |
| B — standalone `traces.py` service module | classic db→DAL→service | 4.5/5 |
| C — per-scale trace APIs | full fractal symmetry | 1.5/5 (bespoke ontology, pass-throughs) |
| D — fat TraceDAL as public door | callers hit DAL | 3.5/5 (matches raw LLM instinct, but can't hold embedder composition; breaks DAL-as-primitives convention repo-wide) |
| **E — B packaged as a Brain mixin** | one `brain_traces.py`, exposed as `brain.<method>` | **5/5** |

E wins on axis 3: the repo already teaches `brain.<capability>` as the universal call
shape (`brain.recall`, `brain.remember`, `brain.get_node`, `brain.query_traces`), and
the MCP tool surface mirrors it 1:1. A standalone module (B) would create a second,
competing call idiom.

**The routing rule after E (one line, no judgment):**
> Reading or writing traces through the API? It's a `brain.` method, and it lives in
> `brain_traces.py`. Only `brain_traces.py` touches `TraceDAL`. `trace_contract.py`
> owns the vocabulary. (Dashboard keeps its sanctioned passive-observer read path.)

## Target architecture

```
                 hooks · scribe_due · S1 encode/surface · S2 units · MCP dispatch · eval
                                          │
                                          ▼
                        Brain  ←  brain_traces.py (BrainTracesMixin)
                                  ├─ generic door:   query_traces
                                  ├─ journal:        journal_notes, write_journal_notes
                                  ├─ episodic:       recall_episodes
                                  └─ conversation:   get_conversation,
                                                     turns_since_last_encode,
                                                     get_conversation_around (+ JSONL fallback)
                                          │
        trace_contract.py  ──────────────┤   (vocabulary: ref types, turn classification,
        (feeds both layers)              │    wake marker, journal parse/render)
                                          ▼
                                      TraceDAL (dal_logs.py — indexed primitives)
                                          │
                                          ▼
                                    brain_logs.db (trace_events, trace_embeddings)

        dashboard/ ─ ─ ─ ─ ─ ─ ─ ─ (passive observer: own read path, sanctioned exception)
```

## Move plan (method-by-method)

New file: `servers/brain_traces.py`, class `BrainTracesMixin`. Estimated ~700 lines
(179 episodes + ~130 query/journal + ~380 conversation).

| Function | From | Callers to update |
|---|---|---|
| `query_traces` | `brain_recall.py:540` | none (already `brain.query_traces` / `self.query_traces`) |
| `journal_notes` | `brain_recall.py:576` | none |
| `write_journal_notes` | `brain_recall.py:621` | none |
| `write_session_arc` | `brain_recall.py:707` (journal-family — same trace-residue system) | none |
| `recall_episodes` + `_resolve_time_bound` | `brain_episodes.py` (file dissolves) | none (already `brain.recall_episodes`) |
| `get_conversation` | `s0/conversation.py:56` → **becomes a method** | `brain.py:scribe_due` (→ `self.get_conversation`), `s1/encode.py:_gather_messages` (→ `brain.get_conversation`) |
| `turns_since_last_encode` | `s0/conversation.py:84` → method | `brain.py:scribe_due` (→ `self.turns_since_last_encode`) |
| `get_conversation_around` + private helpers (`_resolve_node_timestamp`, `_find_encoding_session`, `_from_traces_*`, JSONL family) | `s0/conversation.py:104+` → method + private methods | `s2/healer_decoder.py:210` (→ `brain.get_conversation_around`) |

Deletions: `servers/brain_episodes.py`, `servers/scales/s0/conversation.py`
(and `scales/s0/` package if nothing else remains). `BrainRecallMixin` sheds the three
trace functions. `Brain` bases: `BrainEpisodesMixin` → `BrainTracesMixin`.

## What explicitly stays put

- **`TraceDAL`** — the primitives layer, unchanged.
- **`trace_contract.py`** — schema vocabulary + validation; writers need it too, so it
  stays a separate file feeding both the functional layer and the write path.
- **MCP surface** — `dispatch_observability` already calls `brain.query_traces` /
  `brain.recall_episodes`; no schema change, no redeploy needed (verify
  `brain_mcp.py` imports at implementation).
- **Dashboard** — passive-observer exception, untouched.

## Superseded / updated decisions

- Brain node id:71b34243 ("recall_episodes belongs in BrainEpisodesMixin") —
  **generalized**: capability-mixin placement stands; the dedicated file dissolves into
  the consolidated mixin.
- Brain node id:e56dc13b ("S0 API layer serves all layers") — **re-scoped**: the
  service obligation moves to the brain capability layer. Scale packages
  (`scales/s0/s1/s2`) host *integration units*, never data access. Scale is a tag in
  the substrate, not a boundary in the read path.
- `CLAUDE.md` S0 API section — rewrite to point at `brain_traces.py`.

## Phase 0 (recommended): TraceDAL dedup — clean primitives before the move

Intra-DAL audit (2026-07-11, verified spot-checks) found ~70–90 lines of true
duplication inside `dal_logs.py`. Doing this FIRST means the new layer lands on
clean primitives. Sequenced by risk:

1. **`_CANON_COLS` constant** — the canonical 10-column SELECT is copy-pasted 7×
   (5 unqualified: L707/718/788/965/1043 + 2 `te.`-qualified: L880/1456);
   `_row_to_event`'s own docstring documents that the copies have drifted before.
   Zero behavior change, de-risks everything after. (~6–10 lines + kills a
   documented drift class)
2. **Parameterize `get_recent` with `ref_type`/`ref_id`** — `get_by_ref_type`
   (L1006) is `get_recent` (L723) plus two WHERE predicates; identical projection,
   ORDER, LIMIT, index alignment. `get_by_ref_type` becomes a thin wrapper keeping
   its required-ref_type contract. Fold the twice-stated `chain_suffix` fragment
   and the **verbatim-duplicated session-XOR guard** (`get_recent` L748–765 ≡
   `_event_where` L823–836, identical ValueError text) into one WHERE builder
   (`_event_where`, extended with hours/chain_suffix/exclude_ref_types).
   (~50 lines; medium care — hot paths)
3. **`append` delegates to `append_batch([ev])[0]`** — verified byte-identical
   validate→stamp→INSERT sequence; single-event batch is semantically identical
   (same commit gating, same contract raises). (~18 lines; write-path drift hazard
   eliminated)

Explicitly cleared by the audit: no dead methods (Category-B checked), no N+1
loops, all hot paths index-aligned (`find_by_metadata_substring`'s unanchored
LIKE is an intentional forensic scan — needs only a docstring note),
`get_recent_errors` vs `query_logs` share only a JSON-decode helper worth
extracting (~4 lines).

## Pre-flight checklist (verified 2026-07-11)

- **Three grep forms** for dissolved-module callers: import form, dotted
  patch-string form (`patch('servers.scales.s0.conversation…')` —
  `tests/test_daemon_hooks.py:280–282`), and private-symbol form
  (`tests/test_mcp_roundtrip.py:439` imports `brain_episodes._resolve_time_bound`).
  Plus `tests/test_session_context.py` (×2) and
  `eval/oracle_audit/endo_surface_corpus.py`.
- **CLAUDE.md**: exactly one section to rewrite (lines 74–76, "S0 API") — pre-draft
  so docs land in the same commit as code.
- **MCP + dashboard verified clean** — neither imports the dissolved modules:
  daemon restart only, **no redeploy, no new-session**.
- **Merge window**: `brain_recall.py` is the collision hotspot — check
  `self_presence` for active sibling streams (2 were live at spec time, one in
  recall territory) and `self_send` a heads-up before executing.
- **Brain memory**: revise id:71b34243 + id:e56dc13b with supersedes edges at
  implementation time (deliberately, not left to the encoder).
- **Baseline green**: the 2 daemon-recovery reds were fixed on main (`5c0a1fa`).

## Migration steps (each with a check)

1. **Baseline**: full suite green (modulo the 2 known-red daemon-recovery tests on
   main, tracked separately).
2. **Create** `brain_traces.py` from `brain_episodes.py` content; rename class to
   `BrainTracesMixin`; swap into `Brain` bases. Check: import smoke + episodes tests.
3. **Move** `query_traces` / `journal_notes` / `write_journal_notes` out of
   `brain_recall.py`. Check: journal + dispatch tests; grep confirms `brain_recall.py`
   has no trace functions left.
4. **Move** the conversation family; convert functions to methods; update the three
   call sites. **Gotcha**: `_get_conv_dir()` resolves the repo root by `__file__`
   dirname-hops (4 from `scales/s0/`, **2 from `servers/`**) — must be adjusted or the
   JSONL fallback silently dies. Check: scribe/encode/healer targeted tests + an
   explicit `_get_conv_dir` assertion.
5. **Delete** the two dissolved files (+ empty `s0/` package); update
   `scales/s0/__init__.py` or remove. Check: `git grep` both import forms
   (`from servers.scales.s0.conversation import` AND `scales.s0.conversation`) return
   nothing; both entrypoints import-smoke.
6. **Docs**: CLAUDE.md S0 section, this doc → IMPLEMENTED, docstring sweep
   (`scribe_due`, `session_context.py` comments reference the old paths).
7. **Full suite** (import-surface change ⇒ full tier), commit, merge to main, daemon
   restart. MCP redeploy not expected (verify step: `brain_mcp.py` must not import the
   deleted modules).

## Cost accounting (measured 2026-07-11)

~728 lines relocated (167 brain_recall + 179 episodes + 382 conversation); net
savings only ~30–60 lines of merged boilerplate. **This is a routing fix, not a
size reduction** — the payoff is the eliminated correction tax, not line count.

Hidden callers beyond the 3 in `servers/`: `tests/test_mcp_roundtrip.py`,
`tests/test_daemon_hooks.py`, `tests/test_session_context.py`,
`eval/oracle_audit/endo_surface_corpus.py` import the dissolved modules directly —
update in step 5's grep.

## Risks

- **Magnet-file growth** — `brain_traces.py` is now THE home for trace features and
  will grow (dal.py precedent: 4,148 lines before forced split). Watch threshold:
  ~1,200 lines → split by section group; the clean boundary makes that cheap.
- **Stale brain memories** — several reinforced nodes teach "use the S0
  conversation layer"; supersession edges cover the durable record, but
  auto-surfaced stale guidance is a transient confusion source post-migration.

- **JSONL fallback path depth** (step 4 gotcha) — the one silent-failure candidate;
  covered by an explicit check.
- **Function→method conversion** — mechanical, but touches signatures; the targeted
  tests for scribe cadence, encoder gather, and healer conversation-window pin it.
- **Parallel streams** — coordinate the merge window (settled-work-first rule); the
  move touches high-traffic files (`brain.py`, `brain_recall.py`).

## Follow-ons unlocked (not in scope)

- Audit finding #4: `daemon_hooks.py` reaches `get_session_turns` directly and
  re-implements the wake-envelope drop because `get_conversation` doesn't expose
  `with_surfaced` / `exclude_trace_id`. Once `brain_traces.py` owns the function,
  widening it is a natural single-file change; `daemon_hooks` then routes through the
  layer.
- `eval/` and `scripts/` direct-DAL usages can migrate opportunistically to the same
  door.
