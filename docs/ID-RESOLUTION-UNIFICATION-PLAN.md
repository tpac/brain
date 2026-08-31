# Node-ID Resolution Unification — plan, dilemmas, and everything already established

**Status:** PLAN, not started. Written 2026-08-31 by the session that shipped the LAF
survivor-credit fix (merge `ff9ef59`) and the consolidation prompt id widths (merge
`eea8890`). Nothing in "The work" below has been done. Read §2 before proposing
anything — most of the expensive discovery is already done and re-deriving it wastes
the session.

**Deliberation first.** §6 holds live dilemmas the operator has NOT ruled on. Two of
them (D1, D2) decide whether this is a small subtraction or a large refactor. Do not
start Phase 0 before D4 is answered — it changes the review bar.

---

## 1. Why this exists

Turning a node id into a node is done ~30 different ways across the codebase, with
**three incompatible ambiguity policies** and **four liveness policies**. The operator's
framing:

> "The whole 8 chars to 32 char ids is weirdly spread across the code, we either
> support shortcuts or not, supporting it only in some places smells"

and, on the guidance layer specifically:

> "I don't think any MCP consumer should be guided to hand a short prefix of an id.
> They should give the exact id. We can then make it easier on consumers that are
> having a hard time but the guide should not declare prefix"

That second quote is the design principle this plan implements. It is recorded as brain
node **id:173f9853** (guide exact, tolerate quietly, never advertise).

---

## 2. Established facts — verified, do not re-derive

| fact | evidence |
|---|---|
| Node ids are minted **8 hex chars** | `uuid.uuid4().hex[:8]`, `servers/brain.py:549` |
| **Every live node is 8 chars** — 10,284 at time of writing, zero exceptions | verified by query after archiving the legacy cohort |
| The 10 legacy 32-char nodes are **archived** (2026-08-31) | dream-mechanism artifacts, 1 edge total between them; `brain_batch` archive op |
| **Zero prefix collisions** exist, even against archived rows | verified by self-join on `substr(id,1,8)` |
| Trace `[:8]` truncation is therefore **identity** | ~13 render sites listed in §4 |
| `resolve_id` is **prefix + arbitrary pick** | `servers/dal.py:173-180` — `LIKE ?`, `fetchone()`, no `ORDER BY`, no ambiguity check |
| Trace ids are **already exact-8** with no prefix door | `dal.py:37` `_TRACE_ID_HEX`, used with `.fullmatch()` at `:835`, `:874` — the precedent this plan copies |
| A daemon restart does **NOT** refresh MCP proxies | `brain_mcp.py` is the per-session entry script, loaded once; only lazily-imported `servers/*` submodules pick up new code mid-session |

**The invariant is load-bearing and nothing enforces it.** "All node ids are 8 chars" is
what makes exact-match safe. `_generate_id` guarantees it for new nodes, but no test
pins it. **Add that assertion before removing any prefix matching** — otherwise a future
id-scheme change silently breaks every door at once.

---

## 3. The principle, and the test that makes it operational

**Guide exact. Tolerate leniently. Never advertise the leniency.**

The operational refinement — this is what decides each of the ~30 sites without
case-by-case judgment:

> **Tolerance must be unambiguous and loud, never arbitrary and silent.**

| mechanism | verdict |
|---|---|
| `_unique_prefix_match` (`surface.py:613`) — requires uniqueness, scoped to the candidate menu, ≥4 chars, logs `surface_id_fuzzy_recovered` | **honest leniency — KEEP** |
| `resolve_id` `fetchone()` — no ordering, no ambiguity check, no log | **dishonest leniency — DELETE** |

Why guidance matters more than it looks: MCP descriptions shape agent behavior *more
strongly than prompt prose*, and with **cross-caller scope** (brain nodes id:8225980e,
id:807394de). A permissive description doesn't describe leniency — it **teaches**
prefix-emission to every agent that reads it, then the recovery exists to excuse the
sloppiness the description created.

---

## 4. The map (compressed — full detail was scouted, cite these before re-grepping)

### The one real guidance violation
`servers/contract.py:81-82` — inside `BATCH_OP_SPECS`' `connect_to[].title` description,
**live in the MCP tool schema** read at decision time by Anchor, the S1 Scribe, the
consolidation encoder, the community encoder, and the healer:

> "always treated as an id: resolved by unique id prefix, and on a miss dropped loudly,
> never matched as a title"

Recorded as brain node **id:7eb86132**. Secondary: `servers/scales/s1/quality_contract.py:1100`
says "Pass 0 **prefix** lookup" → should read "exact lookup".

**Already fixed (merge `eea8890`):** `consolidation_enrichment_prompt.py` 6-char absorb/
connect example ids, widened to 8 in both the prompt and its candidate
(`eval/candidate_prompts/s2_consolidation_absorb.md`).

### The `len(x) < 16` idiom — 6 verbatim copies + 2 drifted variants
`brain_recall.py:351`, `brain_recall.py:394`, `brain_corrections.py:82`,
`pipeline_contract.py:382`, `surface.py:705`, `healer_encoder.py:378`.
Drifted: `dispatch_common.py:21` (no length gate at all — prefix-matches full ids too),
`dispatch_read.py:174` (inverted shape). Also `eval/edge_selection_eval.py:500`.
The `16` is a magic number straddling 8 vs 32 and lives in no constant.

### Duplicate short→full implementations
`recall_laf.py:532-536` (rebuild) and `:511-517` (incremental) — these two already
disagree: `_row_for` marks a short `_ambig` permanently while the next full rebuild
recomputes from scratch, so a short can be resolvable → permanently unresolvable →
resolvable again depending on rebuild timing. Third copy: `eval/laf/episodic_ops.py:160-179`
(returns a collision count production discards).

**Note:** with all ids 8 chars, `LafV1Engine._short` is *identical* to `_idx`
(`nid[:8] == nid`). It deletes rather than unifies.

### Known ambiguity-policy conflict (same store, four answers)
`resolve_id` → arbitrary, silent · `brain_remember.py:2066-2090` (`connect_to` Pass 0)
→ refuses, loud, terminal · `recall_laf._short` → drops · `filter_nodes(prefix=)` →
returns all matches (legitimate — the caller explicitly opts in and gets every match).

### Silent-miss / dead code
`dispatch_common._resolve_id` returns **the input unchanged** on a miss, so
`dispatch_read.py:101-102`'s `"not found"` branch is unreachable and `get_nodes`
silently drops unknown ids while `get_node` reports them.

### `brain_traces.py:858-859`
`resolve_id(node_id) or resolve_id(node_id[:8])` — for 8-char input the second call is
byte-identical to the first (dead), and for a longer input it *truncates before
resolving*, which under exact-match would bind to a different node. **Delete, don't
leave dead.**

### Raw copies in eval (will never inherit a fix)
~10 `SELECT ... FROM nodes WHERE id LIKE ?` + `fetchone()` across
`eval/oracle_audit/*`, `eval/export_training_data.py:48`, `eval/edge_selection_eval.py:501`,
`scripts/profile_spread.py:61`. Also `eval/capabilities/base.py:587` uses a
**bidirectional** prefix comparison that will keep passing while testing nothing.

### OUT OF SCOPE — different namespaces where prefix matching is legitimate
- **Session ids** — 32-char `uuid4().hex` (`session_context.py:42`); `self_send` prefix
  matching (`brain_mcp.py:690`, `self_channel/signal.py:103`) is load-bearing, keep.
- **Thalamus ids** — `th_` + 8 (`thalamus.py:107,173`).
- **Self-channel message ids** — 12-char (`signal.py:63`).
They share the vocabulary ("short", "prefix"). A sweep will over-reach into them if it
greps by word rather than by namespace.

---

## 5. The sequencing insight (non-obvious — the natural order is the wrong one)

Three deploy surfaces with **different latencies**:

| surface | goes live |
|---|---|
| `servers/*` code | daemon restart — immediate |
| MCP tool descriptions (`contract.py` `BATCH_OP_SPECS`, `brain_mcp.py`) | restart **+ a fresh session** (proxy-side) |
| prompts (`scales/s1|s2/*_prompt.py`) | daemon restart |

- **Enforcement first** → a window where the live schema still promises prefix
  resolution while the code refuses it. Models emit what they were told to; writes fail.
- **Guidance first** → models tighten while code still tolerates. **Zero breakage window.**

**Therefore: guidance before enforcement.** And because the recovery paths log, the gap
between them is measurable rather than guessed — ship guidance, watch whether
prefix-emission actually drops, enforce when clean.

---

## 6. DILEMMAS — operator deliberation required

**D1. Build a unified `widen()` door, or delete the concern?**
The original scout proposed `NodeDAL.widen()` as the single short→full door. But every
live id is 8 chars and trace `[:8]` is identity, so `widen()` would always return its
input — a mechanism supporting nothing. *Prior session's position: don't build it; the
unification is a **subtraction**, not a new abstraction.* This removes most of the
scout's 9-step migration.

**D2. Does `resolve_id` become exact, or disappear?**
An exact `resolve_id` means "does this id exist?", which a plain `SELECT` already does.
*Prior position: it disappears.* Keeping it preserves the shape of the problem — a
general-purpose "resolve any id-ish string" door invites the next caller to reach for it.

**D3. Namespace separation — documentary or structural?**
Four id namespaces share the "short"/"prefix" vocabulary (§4). Do we separate them
structurally (typed id wrappers) or by documentation + discipline? Discipline is exactly
what failed to hold one policy across ~30 node-id doors — which argues for structure —
but that is a far larger change than this plan. *No prior position; genuinely open.*

**D4. Write-door miss policy: refuse, or skip-and-log?** ← **answer before Phase 0**
`brain_batch` today *skips* unresolved `connect_to` targets and logs, deliberately not
failing the batch. `connect_to`'s Pass-0 resolver *refuses* ambiguous ids terminally.
Both defensible; they are inconsistent. *Prior position: batch-skip is a separate
deliberate choice and should NOT be swept into an id refactor* — but if one policy is
wanted, this is where to state it. **This decides whether the plan is a pure subtraction
or also a semantics change, and those deserve different review bars.**

**D5. `dispatch_common._resolve_id`'s "return the input on miss" contract.**
Fixing it makes `get_nodes` report misses it currently swallows — right, but a behavior
change across ~12 write-door call sites that depend on the current contract.

**D6. The eval `WHERE id LIKE ?` copies** — sweep them onto the shared door, or leave
them? They are eval-only, but they are also the reason a fix to `resolve_id` never
reaches eval. *Leaning: sweep, cheap, no production risk.*

---

## 7. Proposed phases (shape only — D1/D2/D4 may reshape this)

**Phase 0 — guidance + instrumentation, no code semantics.**
Reword `contract.py:81-82` and `quality_contract.py:1100`. Add the "ids are 8 chars"
assertion test. Confirm recovery paths log with enough shape to count emission failures.
*Gates:* touching `BATCH_OP_SPECS` requires re-running `eval/mcp_batch_probe.py` +
`eval/mcp_schema_gate.py`, then a daemon restart, then a **fresh session** before the new
description is actually read.

**Phase 1 — measure.** Let sessions turn over; count prefix/short emissions in the logs.
This is the empirical gate for Phase 2 rather than a guess.

**Phase 2 — enforcement.** Delete `resolve_id` (or make it exact, per D2), the `len<16`
idiom's 8 sites, `LafV1Engine._short`/`_ambig`, `eval/laf/episodic_ops._short_to_full`,
and `brain_traces.py:859`. **Keep** `surface.py:613` `_unique_prefix_match` untouched —
it is the honest-tolerance exemplar, and `SURFACE_SELECTED_ID_PATTERN = "^[0-9a-f]{4,8}$"`
should stay as-is (see §8).

**Phase 3 — cleanup.** `dispatch_read` dead branch (per D5), `brain_traces` double
resolve, the eval raw-SQL copies (per D6), `eval/capabilities/base.py:587`.

Phases are independently shippable. Phase 2 has the real blast radius (~15 call sites) and
wants its own code review — but **no recall eval**, because nothing it touches changes
ranking or the candidate pool.

---

## 8. A trap worth not falling into twice

The surface path looks like a blocker and is not. It has **two** mechanisms and only one
is the problem:

1. `_unique_prefix_match` (`surface.py:613`) — menu-scoped, uniqueness-required, logged.
   Handles the *measured* BPE-corruption and leading-zero-drop classes (brain node
   id:cf5de551). **Untouched by this plan.** `surface_prompt.py:99` already guides
   correctly ("the 8-character hex id"), so surface is in fact the best existing example
   of the §3 principle: guide exact, tolerate quietly.
2. `resolve_id(short_id)` (`surface.py:1105`) — corpus-wide arbitrary pick for ids *not*
   in the menu. **This is the bug.** Under exact-match, hallucinated fragments stop
   binding to arbitrary nodes; the leading-zero retry `resolve_id('0'+short)` reconstructs
   a full 8-char id and keeps working.

A prior scout conflated these and reported a hard blocker requiring re-measurement of a
known Haiku failure class. There is no such blocker.

---

## 9. Adjacent findings — context, not scope

- **Corrupted ids are in the permanent trace record.** The LAF orphan reporting shipped
  in `ff9ef59` surfaced ids like `' bd a9e '`, `' 53d 4f '`, `'b2a3cweu'` (non-hex) and
  `'f18e3cd'` (7-char). These are the fossil record of the BPE space-corruption bug
  (brain node id:cf5de551), already root-caused and fixed at source by the constrained-
  decoding schema. They will never resolve, and they inflate the "absent from nodes
  table" bucket in any dead-id measurement. **Not caused by this plan and not fixed by
  it** — but any id-population count must separate them from genuine deletions.
- **`eval/candidate_prompts/*` can re-infect.** The consolidation candidate carried the
  same bug as the shipped prompt. Operator ruling: candidates that shipped without an
  A/B are *not* immutable records — keep them in sync, so the diff itself aids diagnosis.
- **`docs/NODE-PRESENTATION-PIPELINE.md:31-45` documents the `len<16` idiom but its table
  is stale** (lists a removed `daemon_hooks.py:494`, a moved `surface.py:434`, and omits
  `dispatch_read.py:174`). Do not use it as the checklist; use §4.

---

## 10. Pointers

**Brain nodes:** id:173f9853 (the principle) · id:7eb86132 (contract.py violation) ·
id:ebd929b1 (full door map) · id:13c1e0c9 (landscape, superseded by ebd929b1) ·
id:9436e12c (32-char cohort resolved) · id:8225980e + id:807394de (why MCP descriptions
dominate) · id:cf5de551 (BPE id corruption) · id:967809f4 (resolve-forward vs filter) ·
id:d42a49ce (canonical-pull ruling)

**Docs:** `docs/TRACE-NODE-RESOLUTION.md` (the survivor-redirect contract; §5 call-site
map now includes site #9, the LAF episodic lanes)

**Commits:** `ff9ef59` (LAF survivor-credit + orphan reporting) · `eea8890` (consolidation
prompt id widths)

**Repo rules that bite here:** `BATCH_OP_SPECS` changes require `mcp_batch_probe` +
`mcp_schema_gate` · `servers/*` → daemon restart · `brain_mcp.py`/hooks/SKILL.md →
`./redeploy.sh` + new session · run everything through `./dev`.
