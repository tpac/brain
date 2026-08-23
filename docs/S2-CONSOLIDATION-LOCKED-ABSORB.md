# S2 Consolidation — Locked Nodes Are Absorbers, Not Absorbed

**Status:** ✅ Locked rules SHIPPED — now in **v7** (2026-06-04, merged `1277a57`),
which uses the `absorb` op, so the "actually merge the dupes" path is now WIRED (not
just a primitive). The locked behavioral rules below (≥2 locked → KEEP, no churn-revise,
contradiction → escalate, locked is always the survivor never absorbed) are live AND
eval-tested: corpus cluster 20 — four locked O/K/Δ duplicates, `pre_class=likely_consolidate`,
cosine 1.0 (maximum merge pressure) → KEEP, correct in v7. See
`docs/archive/session-handoffs/S2-CONSOLIDATION-ABSORB-SESSION-HANDOFF.md` §0. Consolidation is sacred → prompt
changes remain **eval-gated**.

---

## The trigger

```
ERROR archive_guarded: Cannot archive locked node
  s2:consolidation tried to archive 96d2fdf8 "Interactions table: versioned templates"
```

Recurred **14× total, 4× on 2026-05-30** (00:30, 03:37, 17:35, 18:38 UTC). The
deterministic guard in `archive_node` ([brain_remember.py:158](../servers/brain_remember.py)) refused —
**no data lost** — but the recurrence revealed a real gap.

## What's actually happening (layered defense, only the last layer catches it)

1. **Prompt** ([consolidation_enrichment_prompt.py:215](../servers/scales/s2/consolidation_enrichment_prompt.py)) — tells Sonnet
   "LOCKED/CRITICAL must survive, sacred." The encoder even renders a `[LOCKED]`
   flag so Sonnet sees it. **Soft — the LLM occasionally ignores it.**
2. **Cluster-scope archive_guard** ([base.py:291](../servers/scales/s2/base.py)) — checks cluster
   *membership* only, **not lock status** → a locked cluster member passes.
3. **`archive_node` lock guard** — deterministic, caught it. ✅

Root cause: locked-protection in the consolidation pipeline rests entirely on
**prompt compliance**. The deterministic backstop saves the data, but the leak
recurs → churn (wasted Sonnet calls) + a false-error logged on every slip.

## The reframe (the actual fix)

> **A locked node in a cluster is ALWAYS the survivor and the absorb-target.**
> The redundant unlocked neighbors fold *into* it (revise its content
> additively), then get archived. Flip the polarity: locked is not an *obstacle
> the guard protects* — it is the *anchor the cluster collapses toward*.

### Why it's the right altitude
- **Ends the churn** — locked is never an archive target, so the guard rarely fires.
- **Enriches the authoritative node** — it accumulates the unique detail its
  redundant neighbors held, which is exactly what the canonical node should do.
- **Matches the semantics** — locked = canonical = convergence point.
- **The deep why:** locked nodes ARE the identity anchors
  (CLAUDE.md: *"identity is the pattern that accumulated experience anchors into
  place"*). Consolidation flowing *toward* locked nodes is literally experience
  anchoring into identity. This isn't a bug fix — it's making consolidation
  respect the fixed-point structure of the graph. **Locked nodes are the
  attractors; consolidation is gradient descent toward them.**

### The mechanism already exists — no new write-path gate
`revise()` ([brain_remember.py:993](../servers/brain_remember.py)) treats only `{id, created_at, locked}`
as immutable and blocks archive-via-revise on locked nodes. **A locked node's
`content` is fully revisable.** So `revise(locked_id, content=<enriched>)` +
`archive(unlocked_redundant)` works today. Sacred-lock forbids *unlocking* and
*deleting* — not *enriching*.

---

## Generalize it: a survivor-priority ladder (don't special-case locked)

The real principle isn't "locked is special" — it's that survivor selection
should follow an explicit **canonicity ordering**, and consolidation always
absorbs toward the higher-priority node:

```
locked / critical   >   graph-centrality + recall + judge-preference   >   recency
```

Locked is just the unambiguous top of the ladder. Reframing survivor-selection
as a priority ladder (rather than "newest wins, unless locked, unless better
graph position…") makes the whole rule legible and removes the special-case
smell.

## Guardrails (must be explicit in the instruction)

1. **ADDITIVE only.** Absorb unique, *non-contradictory* detail; preserve the
   locked node's core claim. Append/extend — never rewrite.
2. **CONTRADICTION → CORRECTION, not absorption.** If an unlocked neighbor
   *contradicts* the locked node (it's a better/newer version, or disagrees),
   that is supersession/correction — **escalate to the operator**, do not
   silently absorb. Consolidation handles *redundancy*, not *corrections*.
3. **MULTIPLE LOCKED → KEEP + LINK.** Can't pick which sacred node absorbs the
   other. Don't merge — add a `similar_to` edge so the graph records their
   kinship without either being touched.

## Edge cases & risks (the second-cycle catches)

- **Stale-locked vs better-unlocked.** If the unlocked neighbor genuinely
  *supersedes* the locked node, absorbing-into-locked entrenches the inferior
  node as canonical. This is the dangerous case — it must route to the
  correction/supersession path (operator escalation), never be auto-resolved by
  consolidation. **Consolidation must not silently decide a locked node is obsolete.**
- **Gravity wells / accretion.** Under this reframe locked nodes only ever *grow*
  (absorb neighbors, never archived). Mostly healthy — canonical nodes should be
  rich — but it raises the stakes of *locking*: a wrongly-locked node becomes a
  permanent accretion magnet. Implication: **lock deliberately.**
- **Edge migration.** Absorbed neighbors' semantic edges must migrate *into* the
  locked survivor (existing survive-and-absorb edge migration handles this). The
  locked node becomes higher-degree — fine, it's canonical.
- **Content bloat.** Only absorb high-similarity, genuinely-redundant neighbors,
  additively. A loose cluster must not dump into the anchor.

## Policy call (needs operator blessing before implementing)

This lets S2 consolidation **`revise` (mutate) operator-locked node content** —
an expansion of what automated S2 is allowed to touch ("locked" has meant
*untouched* until now). It is additive, attributable (`encoding_source=
s2:consolidation`), and reversible (revision history). **Tom leaning yes** (the
locked node *should* grow with converging knowledge) — confirm before shipping.

---

## The complete fix (4 parts)

1. **PROMPT (primary)** — reframe `consolidation_enrichment_prompt`:
   locked/critical → always survivor + absorb unique non-contradictory detail →
   archive the unlocked; multiple locked → KEEP + LINK; contradiction → correction.
2. **GUARD (backstop, made quiet)** — keep the deterministic locked-archive
   refusal as LLM-slip insurance, but **downgrade `archive_node`'s refusal from
   `_log_error`/ERROR to a warning** — a guard succeeding is not an error.
3. **SUPPRESSION** — when the guard refuses a locked-archive, record a rejection
   fingerprint ([rejection_table.py](../servers/scales/s2/rejection_table.py)) so consolidation stops
   re-clustering + re-proposing the same locked node. Kills the churn.
4. **CLUSTER-SCOPE GUARD (optional, deterministic)** — add a `locked`/`critical`
   check to `base.py` `_make_encoder_dispatch` archive_guard: drop archive ops
   targeting locked nodes the same way it drops out-of-cluster ops. Makes
   protection structural, not prompt-dependent.

## Files

| File | Change |
|---|---|
| `servers/scales/s2/consolidation_enrichment_prompt.py` | the reframe (primary) — the deployed default |
| `servers/brain_remember.py:158` | archive_node locked-refusal → warning severity |
| `servers/scales/s2/base.py:291` | add locked check to cluster-scope archive_guard |
| `servers/scales/s2/rejection_table.py` | record guard-refused archives (suppression) |
| `servers/scales/s2/consolidation_decoder.py` / `_encoder.py` | optionally filter locked from archive candidates upfront |

## Implementation discipline

- **Prompt change** edits `SYSTEM_PROMPT` in
  `consolidation_enrichment_prompt.py` — the code default *is* the deployed
  prompt; the DB holds only per-install overrides.
- **Eval-gate** (consolidation is sacred). A/B the candidate through
  `tests/interaction_override.py` before merging it. Add a locked-cluster case
  to the consolidation eval and verify: (a) locked node never archived, (b)
  locked survivor enriched with the unlocked members' unique detail, (c) no
  contradiction silently absorbed. Merge the default only after it passes.

## Thematic note

This is the **inverse** of the silent-failure work that produced the run_hook
standard: there, real errors were invisible; here, a guard *succeeding* is
logged *as* an ERROR — a false positive polluting the error surface we just
cleaned. Same principle — *"is this really an error?"* — opposite direction.
