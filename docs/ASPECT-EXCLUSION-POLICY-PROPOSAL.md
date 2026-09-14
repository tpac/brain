# Aspect exclusion policies — inventory, shape ruling, sequencing

**Status:** proposal, revised 2026-09-14 after an architecture review against the call graph and the
brain's prior rulings. Not started.

> **READ THIS FIRST — this is not a new campaign.**
> `docs/ASPECT-OWNERSHIP-ARCH-PLAN.md` **Step 6** ("Name the second concept inside `noise`; unify
> nine exclusion literals") already owns this work, already picked the same seam
> (`AspectRegistry._adopt`), and is **partly shipped** — `structural_exclusions`,
> `traversal_exclusions` and `lineage_relations` are its output. Its own correction block says the
> remaining scope is "the cohesion/adjacency policies". That is exactly what is left here.
> `docs/DAL-BOUNDARY-ARCH-PLAN.md` Step 7 is a third document on the same boundary.
>
> The first version of this document was written without recalling Step 6 and re-derived its
> conclusion. **The fold is DONE (2026-09-14): Step 6 has been re-scoped and now carries the
> corrected inventory, the three rulings, and the A–E ordering.** Step 6 is the executable plan; run
> it there, under that plan's execution posture (id:82f72780: show the code, discuss, then
> implement, per step). This document is kept as the review record — the evidence behind those
> rulings, and what the original proposal got wrong. Nothing here is executable on its own.

## What the review checked

Every row of the original inventory was opened in the source. Live membership was read from the
working copy at `$BRAIN_DB_DIR/aspects_v1.json` (**not** the repo seed — they differ), and live edge
counts from `edge_relations` on the production graph. Prior rulings pulled: `49d734ad` (DAL is
mechanism, brain derives policy), `c8de37c6` (no business logic in the DAL), `cf731a70` (the July
inventory of eight hardcoded lists), `828b86f8` (noise exclusion into `get_node`), `4fd930b6`
(traversal vs structural), `e89a5267` / `40e7125a` (the aspect-name strings), `52cdf2b9` (noise as
veto).

Not traced: the two dashboard copies (Step 6 already rules "leave them" — forced by the
disconnection contract) and the archived module `servers/scales/s2/archive/reclassify.py`.

## Inventory — corrected

Six rows of the original table were wrong. Four described code that no longer exists; one described
tonight's own change backwards; one undercounted.

| # | consumer | policy | where | shape | status |
|---|---|---|---|---|---|
| 1 | node pulls (`get_node` connections → Anchor, recall surface, encoder catalog, healer, consolidation loader) | noise | `aspects.structural_exclusions` | registry attr, read per call | correct |
| 2 | graph dynamics (traverse, spread, `graph_expand`) | noise − `community_member` | `aspects.traversal_exclusions` | registry attr, 3 sites | correct |
| 3 | spread ride-along | lineage − traversal | `aspects.lineage_relations` | derived from the **`structural_lineage` per-aspect fact** | ~~registry attribute~~ — keyed by a boolean fact, not a name list |
| 4 | edge_context text + invalidation | noise | `brain_connections._edge_context_excluded` → installed on `GraphDAL` as a **zero-arg callable** (`brain.py:352`), read through a property | callable indirection + `find_missing(exclude_relations=…)` | ~~"attribute copied at construction… stays a copy"~~ — **inverted**. It is deliberately not a copy |
| 5 | encoder prompt vocabulary | per-aspect opt-in | `Aspect.prompt_visible` in `aspects_v1.json` + a per-member noise drop at `aspects.py:75` | declared JSON fact | ~~`EDGE_ASPECT_PROMPT_SKIP`~~ — **deleted** in Step 4 (`91119f7`) |
| 6 | community typed adjacency (decoder + structural stamper) | generic_relation, noise + literal `community_member` | `ADJACENCY_SKIP_ASPECTS` + `ADJACENCY_EXCLUDED_RELATIONS`, `community_contract.py:164-165` | aspect tuple + verb tuple | line number was `:146`; the verb tuple narrowed 3 → 1 since July |
| 7 | community placeability fingerprint | noise, generic_relation | `relations_in([…])` in `community_decoder.py:223` | inline registry call | correct |
| 7b | **community idle-gate wake filter** | noise, generic_relation | `relations_in([…])` in `community.py:121` | inline registry call | **missing from the original inventory** |
| 8 | community cohesion / auto-archive | 5 literal verbs | `non_cohesion_relations` in `COMMUNITY_DETECTION` | interaction-config verb list | correct — see ruling |
| 8b | **community evidence sample** | 3 literal verbs, inline SQL | `community_decoder.py:1381-1382` | frozen SQL literal | **missing** — a *third* spelling of non-cohesion, matching neither of the other two |
| 9 | consolidation suppression | settlement aspect, fallback 8 verbs | `suppression_relations(brain)` | derived fn + fallback | correct — see ruling |
| 10 | correction dedup on rendered connections | correction_improvement | `relations_in([…])`, `encode_contract.py:283` | inline registry call | the paired "encoder catalog **noise strip**" is **deleted** (`828b86f8`) — that half of the row is gone |
| 11 | absorb edge migration | `community_member` by endpoint types | `absorb_migrates_relation`, `dal_graph.py:102` | rule function | correct |
| 12 | archive exemption | survivor_lineage | `brain_remember.archive_exempt_relations()` | inline registry call, loud on empty | correct |

**The drift story is smaller than the original told it.** All six constants from the July inventory
(`cf731a70`) are gone: `DEFAULT_EXCLUDED_RELATIONS`, `TRAVERSE_EXCLUDED_EDGES`,
`EXCLUDED_EDGE_TYPES`, `INTENTIONAL_EDGE_TYPES`, `LINEAGE_FAMILIES`, `ASPECT_ACCEPTS`. What remains
is three frozen verb lists — rows 8, 8b, and the row-6 tuple — plus one documented fallback.

## Shape ruling — the table is right, and it is Step 6's table

A named policy table at `_adopt` is the correct shape, and it is already the recorded design
(`c314efc9`: "_adopt is the Step 6 precomputation seam"). Three corrections to how the original
stated it:

**1. Two of the six proposed entries must not exist.**
`encoder_vocabulary` is an **opt-in selection** on a per-aspect fact, not an exclusion. Restating it
as `{'skip': [...]}` inverts its default from *quiet* to *visible* — against the conservative
degradation `aspects.py:345-348` chose on purpose. And `aspects.py:63-64` states the limit a
name-keyed table shares: `prompt_visible` "is per-ASPECT, so it cannot say 'this one relation is
machinery'", which is why the per-member noise drop sits beside it. `community_cohesion` is ruled
out below.

**2. Reading through the registry is the invariant, not the table.** The staleness hazard is not
*where the set is declared* — it is *when the consumer binds it*. `_adopt` rebinds on every
classifier cycle, so a consumer that copies at construction goes stale silently. `dal_graph.py:160-171`
records the exact failure: a snapshot "would silently disagree with the backfill filter, which reads
live … deleting vectors nothing would rebuild." So the rule Step 6 should carry is **bind per call,
never at construction** — and where a consumer cannot hold a registry reference (the DAL), the
callable indirection stays. That indirection is the pattern to copy, not an exception to tidy away.

**3. `excluded()` cannot carry one universal empty-registry semantic.** The consumers disagree on
purpose, and each disagreement is reasoned in a comment:
- `brain_connections.py:119` → `frozenset()` on a missing registry, so producer and filter still agree.
- `community_decoder.py:225` → empty set **and logs**, because an empty noise set *includes* noise in
  the fingerprint and churns it.
- `brain_recall.get_node` → unguarded, loud by design (aspects-plan Step 10).
- `brain_remember.archive_exempt_relations` → loud, because a silent empty scrubs `absorbed_into`.

A single accessor with one degradation behaviour would flatten four deliberate choices. Whatever
`excluded()` returns, the **degradation stays at the call site**.

## Rulings on the two "decide, not convert" consumers

### `non_cohesion_relations` — stays literal. Converting it is a live regression.

Its five verbs are `community_member`, `dream_observation`, `dreamed_from` (noise) and `related`,
`related_to` — **2 of `generic_relation`'s 20**. A table keyed by aspect names cannot express
"two of twenty". Converting it to `{'skip': ['noise', 'generic_relation']}` is a strict superset
that newly excludes 19 verbs, including `co_anchored` (3,252 live rows) and `similar_to`.

That inverts the documented purpose. `community_contract.py:93-97`: "counting ALL relations (incl.
`similar_to`) means a community cohesive only via `similar_to` is NOT auto-archived." The set feeds
`_community_disconnected`, whose result routes to `_auto_archive_dead` — archived **in code, with no
encoder round and no rejection fingerprint**. After the conversion, a community held together only by
`similar_to` (which consolidation writes as its KEEP outcome) would be auto-archived instead of
reviewed. `tests/test_community_health_seam.py:105`
(`test_similar_to_cohesive_routes_to_encoder_not_auto_archived`) goes red on it.

**The two dream verbs are not droppable either.** Dreams are paused; the rows are not. Live today:
`dreamed_from` 20, `dream_observation` 19. Dropping them would make a dream-only-linked community
count as cohesive and escape auto-archive.

Second reason to leave it: `non_cohesion_relations` lives in `COMMUNITY_DETECTION`, and `s2_community`
is a registered interaction (`interaction_defaults.py:84`), so the key is override-able per run
today. Moving it to `aspects.py` deletes an override surface.

**What to do instead:** correct the comment (it is a deliberately narrow list, not drift) and fold
row 8b — the frozen 3-verb SQL literal at `community_decoder.py:1381-1382` — into the config key it
should have read all along.

### `suppression_relations` fallback — already decided, already guarded. Close it.

It is a **selection** from a closed aspect (`settlement` is `routable: false`), not an exclusion, and
the literal is a degraded-registry default, not a competing policy. `consolidation_contract.py:168-178`
says so, and `tests/test_s2_consolidation.py:193` (`test_contract_fallback_mirrors_seed_settlement`)
already asserts it equals the seed's membership. There is nothing to decide. The original's claim
that it is config-scoped is also wrong: `CONSOLIDATION` is a plain module constant, absent from
`INTERACTION_DEFAULTS`.

## `ADJACENCY_EXCLUDED_RELATIONS` — redundant today, but do not simply delete it

The claim holds. Both builders apply the verb tuple in SQL and the aspect tuple in Python:

```
SQL   :  AND er.relation NOT IN ('community_member')      # ADJACENCY_EXCLUDED_RELATIONS
Python:  fam = rel_to_fam.get(rel); if fam in skip_fams: continue   # ADJACENCY_SKIP_ASPECTS
```

`community_member` is a noise member, and `_adopt`'s **noise veto** (`aspects.py:556-562`) forces
`_reverse_edge['community_member'] = 'noise'`, so the Python skip catches it whatever the JSON order.
The SQL literal removes nothing the aspect skip would not.

**Why it was written:** it predates the veto. Before `52cdf2b9` (2026-08-23) the primary aspect of a
dual-homed string was first-claimant-by-file-order, so `community_member`'s family depended on where
`noise` happened to sit in the JSON — and the veto's own comment names this consumer as the reason it
was added: "Consumers that skip by primary family rather than by membership — community typed
adjacency is the live one." The literal was the belt to the aspect tuple's braces, and the veto made
the belt redundant. The comment above it claims a *parity* contract, which is a different claim and
still true: the constant is what makes the decoder and the stamper provably read the same thing.

**So: keep it, restate it, and pin it.** It is a SQL-side prefilter that drops ~12,272 live rows
before they reach Python on a graph-wide scan — deleting it is a real cost on the decoder's hot path.
Change the comment to say what it is (a volume prefilter, not a second policy) and add the invariant
as a test: **every verb in `ADJACENCY_EXCLUDED_RELATIONS` must resolve to an aspect in
`ADJACENCY_SKIP_ASPECTS`.** That converts a redundancy into a checked contract, and it fails loudly
if someone adds a verb to the prefilter that the aspect skip does not also remove.

## Found during the review — a live parity break behind the same two constants

The parity contract the comment claims is **already broken**, by the other half of the pair.

- `community_decoder.py:370` builds its relation→family map with `primary_edge_map()` — first
  claimant, plus the noise veto.
- `community_structural.py:78-82` hand-rolls the same map as a dict comprehension — **last claimant
  wins**. This is the exact construction `aspects.py:736-739` warns against: "a hand-rolled
  comprehension lets the LAST claimant win, which silently flips a relation's family when a later
  aspect (settlement) multi-homes it." The decoder was fixed; the stamper was not.

Measured against the live registry: the decoder skips **140** relations, the stamper **98** — they
disagree on **42**, dominated by `similar_to` (`generic_relation` first, `settlement` last; 1,314
live rows). So the decoder drops `similar_to` from typed adjacency and the stamper counts it, and the
stamped `community_internal_fraction` can disagree with the decoder's — which
`community_structural.py:11-16` says is impossible by construction.

The parity test cannot see it: `tests/test_community_structural.py:132` builds the **same**
last-claimant comprehension and feeds it to the decoder's builder, so both sides share the wrong map;
and its fixtures use only `extends`, which is single-homed.

This is a correctness defect, not a consolidation item. It is cheap (one line → `primary_edge_map()`)
and it should not wait behind a policy-table refactor.

## The four aspect names inside `noise` — known, owned, and one of them is new

`noise.edge_relations` in the working copy carries `temporal_sequence`, `extension_refinement`,
`validation_evidence`, `correction_improvement` — four **aspect names** filed as relation verbs.

**What wrote them.** Not the classifier inventing them: an S1 encoder emitted the aspect *name* as a
lazy catch-all relation value, the strings landed in `edge_relations`, and the aspect classifier then
correctly filed those bare strings under `noise` (`e89a5267`, `40e7125a`). Noise is doing its job;
the leak is upstream at the S1 write boundary. The aspect name and the verbs it classifies are not
interchangeable — `temporal_sequence` the aspect is 41 real temporal verbs, and none of them is
affected.

**This is already Step 5's work.** `ASPECT-OWNERSHIP-ARCH-PLAN.md` Step 5 names three of them
verbatim and carries the fix ("fix the three entries whenever this step lands"), using them as the
worked example of why members need provenance records.

**The fourth is the interesting one.** Live counts: `temporal_sequence` 18 rows / 0 active,
`validation_evidence` 4 / 0, `correction_improvement` 1 / 0 — but **`extension_refinement` has zero
rows, ever.** The other three entered `noise` from an observation that has since been archived.
`extension_refinement` entered without any observation at all. That is not a data-hygiene item; it is
a gap at the write door, and it is the argument for Step 5's `count_at` field stated better than the
plan states it: a member with `count_at: 0` and no rationale should never have been accepted
silently.

## What is actually left — now folded into Step 6

**These items now live in `ASPECT-OWNERSHIP-ARCH-PLAN.md` Step 6 as items A–E. Execute there, not
here.** Kept below as the reasoning trail. Ordered by real dependency — the first two do not depend
on the policy table and should not wait for it.

**A. Fix the stamper's relation→family map.** *(⚠ owned by another stream as of 2026-09-14.)* One line: `community_structural.py:78-82` →
`brain.aspects.primary_edge_map()`. Then fix `tests/test_community_structural.py:132` to build the
decoder's side from `primary_edge_map()` too, and add a multi-homed fixture (`similar_to`) so the
test can fail. *Verification:* `tests/test_community_structural.py`, `tests/test_s2_community.py`,
`tests/test_community_health_seam.py`. *Blast radius:* changes stamped `community_internal_fraction`
for communities carrying any of the 42 divergent verbs; re-stamp after. *Depends on:* nothing.
*Respects:* the parity contract in `community_structural.py:11-16` — this restores it.

**B. Collapse row 8b into row 8.** `community_decoder.py:1381-1382`'s frozen
`('community_member','related_to','related')` reads `self.config['non_cohesion_relations']` instead.
Behaviour changes only in that the evidence sample now also skips the two dream verbs. *Verification:*
`tests/test_s2_community.py`. *Depends on:* nothing. *Respects:* leaves the config key literal per the
ruling above.

**C. Pin the adjacency prefilter invariant.** Test: every verb in `ADJACENCY_EXCLUDED_RELATIONS`
resolves through `primary_edge_map()` to an aspect in `ADJACENCY_SKIP_ASPECTS`. Restate the comment.
No behaviour change. *Depends on:* A (so the map both sides use is the same one).

**D. Then, and only then, the table.** Entries: `reads`, `traversal`, `edge_context`,
`community_adjacency`. Four, not six — `encoder_vocabulary` and `community_cohesion` are ruled out
above. `structural_exclusions` / `traversal_exclusions` become aliases; `lineage_relations` keeps its
fact-derived construction and subtracts `excluded('traversal')` (`4fd930b6` — conduction is not
visibility). Each consumer keeps its own degradation behaviour at the call site. *Depends on:* A–C.
*Respects:* `49d734ad`, `c8de37c6` — the DAL still holds no policy; the callable indirection stays.

**E. Ratchet test.** A source grep, like the raw-SQL ratchet: a verb tuple in `servers/` whose
contents are a subset of a policy's resolved set must read the policy. **Do not widen any existing
allowlist to land it** — rows 8 and 11 are deliberate literals and belong in the ratchet's documented
exceptions with their reasons, not in a widened pattern.

**Not in scope, already ruled:** the dashboard copies (forced by the disconnection contract), the
absorb rule function, the archive exemption, and the suppression fallback.
