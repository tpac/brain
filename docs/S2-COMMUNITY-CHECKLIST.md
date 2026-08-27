# S2 Community — Working Checklist

The instrument for the S2 community visibility + structure work (opened 2026-08-27).
Mirrors `docs/S1E-CHECKLIST.md`'s role: this holds findings, phases, and open
boxes **while the work is in flight**. Current-state design truth belongs in
`docs/S2-COMMUNITY-DESIGN.md` and lands there as each phase closes — not here.

Everything below was measured or read this session unless a row says otherwise.
**Rows marked ⚠ UNVERIFIED are reasoning, not measurement — check before acting.**

---

## 0. Why this exists

The S2 community encoder writes ~55% of its journal about three things it
cannot do. Census over 400 `journal_note` traces (`chain_suffix=community_detection`):

| what the note says | count | share |
|---|---|---|
| metadata debt — `latest_development` stale / "still needs healer" | **152** | 38% |
| "vocabulary artifact" — rejecting a lexically-similar bad proposal | **46** | 12% |
| "unhoused" — this node belongs in a community that doesn't exist | **20** | 5% |

The encoder is not confused. It is blind, under-equipped, and reporting both
accurately into a channel nobody reads.

---

## 1. Findings

| # | finding | evidence |
|---|---|---|
| **F1** | On `add_to_existing` the encoder sees the target community as title + 150 chars, **0 edges, 0 metadata**. A community's edges *are* its members, so it sees no membership at all. | `community_contract.py:275` — `S2CE_COMMUNITY_FORMAT = {content_limit:150, edge_limit:0, metadata_limit:0}`. Every other format in the repo is content 400–None / edges 4–8 / metadata 200–400. |
| **F2** | The proposal's target community is never fetched. `_find_relevant_communities` builds a semantic query from *member titles*, truncates at 500 chars, and recalls 15 communities. Whether the community under decision is in the payload is incidental. | `community_encoder.py:538-560` |
| **F3** | `## MERGE` tells the encoder to REPLACE the surviving community's narrative — from 150 chars of visibility. Measured community content: 1038–1626 chars. Quota 3/run. | prompt `## MERGE`; lengths measured on 5 communities |
| **F4** | `## HEALTH UPDATE` asks "truly dispersed → archive" vs "loose but real → keep" — a **membership judgment from zero members**, whose accept branch archives a whole community. Quota 3/run. More destructive than F3. | prompt `## HEALTH UPDATE` |
| **F5** | `## DRIFT` moves a node between two communities with the encoder seeing neither one's membership. | prompt `## DRIFT` |
| **F6** | `community_members` and `community_key_decisions` have **zero code readers**. `community_members` has exactly one: `reconcile_community_membership`, and only on the **zero-edge** orphan case. `community_latest_development` has one reader (dashboard). | grep over `servers/ dashboard/ docs/`; `dal_graph.py:525-570` |
| **F7** | The prompt routes metadata work to the healer, whose `fields_required` is `['question','situation','reasoning']` — it structurally cannot act. ~100 identical notes 08-15→08-20 already logged with fix "NONE". | `healer_contract.py:27`; prompt line 153; `docs/challenges/structure.md:90` |
| **F8** | The `get_nodes` MCP description is **false for S2 encoders**. `_select_node_config` returns `get_nodes_config` *before* checking `rich` or batch size, so the described batch-size table and `rich=true` are both inert. Its advice "ask for the few you need rather than a large batch" also contradicts the S2CE prompt's "ONE call with ALL the IDs". | `brain_mcp.py:884-907` |
| **F9** | **Growth ratchet.** Per run: 12 adds + 3 merges (which combine) vs 2 drifts (one node each) + ≤3 archives. **No split operation exists.** | `community_contract.py:109-115`; prior gap node `654d0ebf` |
| **F10** | The ratchet, dated: **418 communities archived to date, last one 2026-08-09.** 18 days with zero archives while adds ran. ⚠ Did not separate merge-archives from health-archives. | measured |
| **F11** | Big communities are terminal homes, not umbrellas. Share of members whose **only** community is this one: `bc639843` 56% (206), `2e6986a2` 58% (175), `fe73f0b8` **72%** (164), `eb5bacb0` 59% (144), `fe1d5fd0` 39% (127). | measured via `get_communities_for` |
| **F12** | Size distribution: **781 communities; 258 >10 members, 95 >20, 30 >40; largest 206.** This is why "render the decision set in full" cannot be literal. | measured |
| **F13** | Orphan placement gate is **inert**. `sim = cosine_similarity(emb, centroid)` against `embedding_placement_threshold: 0.50` — raw cosine. Node↔node random-pair raw cosine measured at **0.6929** (`51b87c91`). A 0.50 raw threshold sits below the random baseline. Very likely the cause of F-census's 46 vocabulary-artifact rejections. | `community_decoder.py:1011-1024` |
| **F14** | Inter-community structure is improvised: 12 distinct verbs, mostly singletons, including an invented `sibling_community`. Only `absorbed_into` (20) is systematic — merge lineage. There is **no sanctioned parent/child verb**. | `edge_relations` census |
| **F15** | **11 live `community_member` edges between two communities** — 9 parent communities, all created 2026-06-01→06-18, then it stops. Inflates `community_size`, can set `community_dominant_type='community'`. None of the F11 five is a parent, so that table is unaffected. | measured |
| **F16** | **Three dead columns on `edges`**, all 35,918 rows: `relation` = `'related'` (one value), `edge_type` = `'related'` (one value), `description` = NULL/empty (all). Plus a dead index `idx_edges_type`. `schema.py:161-175` **already declares `edges` without them** and `:456-459` declares only 4 indexes — the install has drifted from its own schema. | measured + `schema.py` |

### The pattern under F4/F7/F9 and the journal census

The encoder can see **four** structural moves it has no operation for: split a
bloated community, seed a community for unhoused nodes, merge two communities it
created itself across runs, and maintain its own metadata. Every one lands in
the journal addressed to a unit that cannot act.

---

## 2. The one-truth dilemma — resolved

There are ~11 render configs feeding one `render_rich_node` (4 `GET_NODES_*`,
4 surface/Haiku, a default, plus `S2CE_COMMUNITY_FORMAT` and `S2CE_NODE_FORMAT`).
"Collapse them" conflates three separable things:

**(a) Disclosure — does a render say what it trimmed?**
Today: per-config, silent. → Make it a **renderer invariant**, not a config option.
**Do it.** One function; every render surface becomes honest at once. Kills the
silent-trim-read-as-complete bug class. `contract.py:696` already enforces this
for bounded *reads* (`truncation_payload` / `truncation_banner`, pinned by
`tests/test_truncation_contract.py`) — renders never joined that contract.
**Bonus: F8 dissolves.** If the renderer always discloses, the MCP description
need not enumerate thresholds at all. Two problems, one fix, and no per-caller
description override (which would be new drift).

**(b) The bespoke community render in `_find_relevant_communities`.**
A second path doing what `get_node` already does, worse. → **Delete it**; fetch
decision-set communities by id through `get_node`. Removes a path instead of
adding one, and the community render improves for the dashboard too.

**(c) Collapsing the 11 configs into one.** → **No.** Budgets differ for real
reasons — S2CE runs 32K `max_tokens` against a 180s client timeout that already
bit at 8 proposals; boot surfaces work to ~10K chars. This is the purist trap.
Configs stay; **disclosure** stops being per-config.

Net: one renderer, one disclosure rule, N budget profiles, zero bespoke paths.

---

## 3. Phases

### P0 — Baselines (before touching anything)

Captured 2026-08-27, in §1: F11 orphan table, F12 size distribution, the 152/46/20
note census, F10 archive dates. Still to capture:

- [ ] **node↔centroid cosine distribution.** ⚠ The 0.6929 figure is node↔node;
      a centroid is an average and has its own distribution. This measurement
      gates F13's fix and is the single highest-value cheap check in the plan.
- [ ] payload chars and rounds per batch (baseline: last full run was 4 rounds
      across 2 batches, 103.7s, 8→3698 tok, `cache_read 71023`).

### P1 — Text only, no behavior risk

- [ ] Drop `community_members` from the NEW COMMUNITY example and metadata list.
- [ ] Drop `community_key_decisions` likewise — zero readers, creation-time stale.
- [ ] Replace it with the standard **`thought`** field. Right field by its own
      definition: *"my own read… a living field — update it when a re-read moves it."*
      The encoder's recurring per-community observations are exactly that.
- [ ] Rewrite prompt line 153 to stop naming the healer.
- [ ] `community_contract.py:210` — document `community_members` as reconcile's
      creation-time orphan seed, and fix the comment (it says "JSON list of node
      IDs"; it is an `"id: title"` comma string, parsed by the regex at
      `dal_graph.py:564`).

Pure deletion where a field leaves — no negation (`d14bef74`).

**Deliberate consequence:** per-community residue moves onto the node; the journal
keeps cross-community and run-level observations only. That is the narrowing
`714e0111` argued for (journals measured 80–90% trace-restatement), but it is a
journal-contract change made on purpose, not a side effect.

**Ruled (Tom, 2026-08-27):** the `thought` field's reach to Anchor is Thalamus's
job, via journals — do **not** design around the community recall-pool exclusion
(`3f135bea`).

### P2 — One truth (§2 (a)+(b))

- [ ] Disclosure becomes a `render_rich_node` invariant ("8 of 164 shown, +156 not shown").
- [ ] Delete the bespoke community render; decision-set fetched **by id** via `get_node`.
- [ ] Correct the `get_nodes` description to stop enumerating thresholds.
- [ ] Community decision-set render: full content + judgment metadata + bounded
      member slice **derived from edges** (newest N + highest weight), never from
      a stored string.

Target shape:

```
[community] "Frame, Journal, and Session Identity…" (id:fe73f0b8)
  Maturity: settled · Size: 164 members
  Content: <full narrative, ~1500 chars>
  Latest development: <…>
  Members (8 of 164 shown, newest first — +156 not shown):
    [finding] "…" (id:…, 3w ago)
```

Cost ≈ 3.6K chars per decision-set community, **flat** — 206 members costs the
same as 6. ~9 per batch ≈ 8K tokens on a ~107K-token run.

⚠ **Widest blast radius in the plan** — `contract.py` is shared by S1E's catalog,
recall surfaces, `get_nodes`, and the dashboard. Merge-tier suite required.

⚠ **UNVERIFIED:** `metadata_limit` is a *count*, not an allowlist. Adding
"show these three keys" may need a `skip_keys`-style affordance rather than a
number. **Read `render_rich_node` before promising the allowlist shape.**

### P3 — Judgment (needs P2)

- [ ] `## ADD TO EXISTING` gains a **conditional** `revise` of
      `community_latest_development` — only when the new member IS the newest
      movement; a mid-arc member gets the connect alone.
- [ ] Size-aware guidance: small/tight → routine; large → is this the *arc* or
      just shared vocabulary; **several proposals in one batch pointing at the
      same large community = a sub-community forming, not five more members.**
- [ ] F3 and F4 resolve as side effects — merge and health targets are
      decision-set, so both finally render properly. **F4 is a correctness fix,
      not a nicety: it currently archives communities blind.**

The rule must be explicit, not example-only: `50f2b7b1` showed example-osmosis
fails for fields the encoder never reaches for, which is exactly this case.

### P4 — Token-aware batching

Current `max_proposals_per_call: 8` is empirical — the comment records that 10
"was hitting the 180s client timeout on batch 1 consistently."

With a flat render: `batch_chars ≈ (distinct decision communities × ~3.6K) +
(proposals × ~0.3K) + context set`.

- [ ] Batch by **distinct decision-set communities under a char budget**;
      proposals secondary. Split proposals are heavy — weight them.
- [ ] Log payload chars per batch so the budget is measured, not guessed.

Note: the flat render already removes most of the size variance. What remains to
budget is how many *distinct* communities a batch touches.

### P5 — Operations for what the encoder can already see

The F9 / unhoused / cross-run-duplicate cluster. **Design before building P2** —
the member-slice render is shared, and I would rather not build it twice.

**Split.** Decoder clusters, encoder names and judges (`e595e444`). **Additive**
(`df292d31`: overlapping communities are valuable; only 100% containment is the
bug) — the child is created, the parent keeps its members. Reuses the
`new_community` write path plus one typed edge; no destructive re-partition.

```
[N] SPLIT CANDIDATE — "Frame, Journal…" (id:fe73f0b8, 164 members)
    Sub-cluster A (23 members, internal 0.71, cross-affinity to B 0.08):
      [finding] "…" (id:…)   ← 8 shown, +15 not shown
    Sub-cluster B (31 members, internal 0.64):  ← 8 shown, +23 not shown
    Remainder: 110 members in neither
```

So yes, showing a split means showing nodes — but only sub-cluster candidates
(~16 titles), never the parent's full membership.

New surface: proposal type, within-community decoder scan (reuses existing
z-score / affinity / tie code scoped to a member set), quota, prompt section,
aspect entry for the parent→child verb.

⚠ **UNVERIFIED:** the parent→child verb probably must join
`non_cohesion_relations` or it distorts `internal_fraction`. Reasoned from
`community_contract.py:88-93`; **not traced through `build_member_adjacency`.**

**Unhoused seeds.** ⚠ Collides with an existing mechanism — `_mark_unplaced_pending`
(`community.py:325-338`) already marks corridor-dropped / quota-deferred /
encoder-skipped nodes unplaceable, and they *"sleep until their 1-hop
neighborhood fingerprint moves"* (Phase-2 rest gate, shipped 2026-06-23). So
unhoused nodes are asleep, not lost. **Understand this interaction before
proposing a seed pool** — this may be a visibility fix, not a new mechanism.

**Cross-run duplicate communities.** `merge_communities` is a *decoder* proposal
type; the encoder cannot initiate one, so it journals
(*"1b3bea95 created today… f3855d26 was created last run on the same topic —
healer should assess"*). `df292d31` diagnosed the intra-batch version; this is
the cross-run version.

### P6 — Decoder precision (F13)

- [ ] Run the node↔centroid baseline measurement (P0).
- [ ] Then either raise the raw threshold into the real discriminating band, or
      centre before comparing — `51b87c91` measured centring turning a +0.07 raw
      gap into +0.23 against ~0.
- [ ] ⚠ **Rejection fingerprints must be invalidated** or the improvement is
      invisible for every pair already seen. `clear_unplaceable_rejections`
      exists in `rejection_table.py` — **not yet traced**; confirm it is the
      right lever and what re-opening costs.

---

## 4. Standalone cleanups (approved, independent of the phases)

### C1 — `edges` dead columns (approved 2026-08-27)

`schema.py:161-175` already declares `edges` with seven columns; `:456-459`
declares four indexes. This install has ten columns and five. The residue is
`relation`, `edge_type`, `description`, and index `idx_edges_type` — all
constant or empty across 35,918 rows. `_backfill_data`'s `from_version < 6`
block still writes to them inside `try/except: pass`, so on a fresh brain those
UPDATEs fail silently forever.

**This is not a schema change — it is converging a drifted install to its own
declared schema.** House pattern exists and has been used twice: v28 and v30
were both DROP COLUMN migrations (`_migrate_v30_project_to_kv` — *"mirroring v28"*).

- [ ] `_migrate_v32_drop_dead_edge_columns(conn)` — drop `idx_edges_type`, then
      DROP COLUMN `relation`, `edge_type`, `description`.
- [ ] Self-detecting idempotency via `PRAGMA table_info`, like
      `_trace_id_column_is_integer`.
- [ ] `BRAIN_VERSION = 32` (currently 31, `schema.py:37`).
- [ ] Remove or neuter the now-dead `from_version < 6` backfill block.

Verified safe: SQLite 3.47.1 (DROP COLUMN needs 3.35+); **no `SELECT * FROM edges`**
anywhere in `servers/`, `dashboard/`, `tests/`, `eval/`.

⚠ **UNVERIFIED:** I grepped for readers of `edges.relation` and found only
`temporal_extraction.py:361-380`, which *labels* its extraction source
`'edge.relation'` but takes values from the caller (`edge_relations`). **I did
not exhaustively audit every `FROM edges` site for a `.relation` read.** Do that
before dropping.

**Why this matters beyond tidiness:** these columns produced a confidently wrong
finding in this very session — a census of community→community edges read
`edges.relation`, got `'related'` × 99, and concluded "no typed inter-community
structure exists." The truth (F14) was 12 verbs in `edge_relations`. Leftovers
cause drift; this one caused it within the hour.

### C2 — 11 community-as-member edges (approved 2026-08-27)

Archive the 11 `community_member` relations where both endpoints are
`type='community'` (F15). `backup_before_destructive` first — no exceptions.
Data, not schema; other installs will not have them. Do it **after** C1 lands so
the two changes do not tangle.

### C3 — Doc merge (ruled 2026-08-27)

Fold `docs/COMMUNITY-METADATA-DENORMALIZATION.md` into
`docs/S2-COMMUNITY-DESIGN.md`; that name survives. Correct all stale claims
**after** the work lands, not before — `c3cb8533`: plan docs go stale against
themselves when step commits edit the files they cite. Both docs currently name
`community_members` and `community_key_decisions` in their field tables; P1
makes both false.

---

## 5. Process — proportionate

S1E's method (numbered stops, per-stop stateless probes, checklist instrument,
R-row recall cross-exams, reverse pass, DORMANT → package eval → activate) was
for a ~90K-char prompt. **S2CE is 8,160 chars with five decision branches.**

**Keep:**
- fresh-eyes, **changes over additions** (`bed31596` — Tom's own catch; it fired
  twice in this session already)
- **behavioral probes, not interviews** — `160a90ac` found introspection and
  emission diverge; probe with a real proposal batch and diff emitted ops
- override → eval → promote via `tests/interaction_override.py` (the one door;
  never hand-roll)

**Drop:** the multi-stop walk, R-rows, coverage matrix.

**Instrument — already live, do not build a new one:**
1. journal notes/run saying "not updated / needs healer" → target **0**
2. the F11 orphan-in-big-community rate → should fall
3. payload chars + rounds per batch → regression guards

- [ ] ⚠ **Read before building anything eval-shaped.** `eval/` already has
      `ab_community_model.py`, `s2_community_decoder_eval.py`,
      `sim_community_journal.py`, `sim_community_structural.py`,
      `diag_community_encode.py`. **None of these were read this session.**
      Also read `docs/S1E-CHECKLIST.md` and the S1E probe method — Tom's note:
      the S1E work is likely more advanced and worth borrowing from.

---

## 6. Rejected paths (do not re-litigate)

| rejected | why |
|---|---|
| Per-caller `get_nodes` description override | manufactures drift; `0d1fca7b` (locked) puts cross-cutting concerns in the shared layer. Correct the shared description instead. |
| Deriving `community_members` structurally every run | 258 communities >10 members, max 206 → writing ~47K-char strings that **nothing reads**. Retire it to a documented creation-time seed instead. |
| Using `community_key_decisions` as the membership signal | it is creation-time stale exactly like `community_members` (F6). Derive the member slice from edges. |
| Collapsing the 11 render configs into one | §2(c) — budgets differ for real reasons. |
| Adding community metadata upkeep to the healer | `healer_contract.py:27`; different input, model tier, and failure mode. The community unit already owns these fields. |
| Opt-in disclosure flag on `render_rich_node` | an opt-in flag is the drift we are removing. |
| Deferring MERGE (F3) as a separate item | it resolves for free once the decision set renders properly. |

---

## 7. Open decisions

- [ ] **Is universal disclosure (§2a) approved?** It is the dramatic half and it
      touches every render surface.
- [ ] **P5 design now or after P1–P4?** Recommendation: **design now, build
      after** — the member-slice render is shared between add and split.
- [ ] Split's parent→child verb: which word, and its aspect entry (a human edit
      to `aspects_v1.json` plus one `REQUIRED_ASPECTS` line).

## 8. Brain nodes from this session

`36c497cf` (metadata_limit:0 root cause) · `d2b4f944` (two fields, opposite
treatments) · `9197faac` (fix plan v2) · `c7adda3c` (scope correction:
`community_members` string ≠ membership) · `4affbe1c` (`_find_relevant_communities`
ignores `community_id`) · `73f2a2b9` (MERGE destructive rewrite) · `a0615a9a`
(misleading healer line) · `7f47fb43` (growth ratchet) · `ef8a5672` (disclosed
member slice design) · `972d4f0c` (the arc community)

Prior art worth pulling: `e595e444` (decoder→encoder two-stage) · `df292d31`
(overlap valuable, containment is the bug) · `654d0ebf` (split gap, already
logged) · `51b87c91` (community geometry / anisotropy) · `3f135bea` (communities
excluded from recall) · `d14bef74` (deletion beats negation) · `50f2b7b1`
(example osmosis fails) · `04ff3d58` + `79b25bac` (MCP mechanics vs prompt
strategy) · `1a98209e` (MCP cross-contamination, open)
