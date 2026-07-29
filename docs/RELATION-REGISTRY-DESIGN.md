# Relation-Registry Enrichment — Design Doc

**Status:** DRAFT for section-by-section review (2026-07-29). Not yet implemented.
**Scope:** edge relations only (node-type dimensions deferred). Consumers out of scope.
**Prior art / decisions:** brain nodes id:5beed7fc (the 4-dimension plan), id:86714339
(per-relation algebra research), id:d9c7c5fa (edge types serve two masters),
id:b756d494 (aspects classify strings, don't sense-split), id:be9b09b3 (dimensions
are output not input), id:7609b4d5 (producer open, consumer imposes structure).

---

## 0. Purpose

An edge today carries four things: a **verb** (relation), a **why** (embedded
description), and **source/target**. The verb *implicitly* encodes how the edge
should behave — whether it reinforces or suppresses, whether it's mutual or
directional, whether it chains. A consumer (recall, spread, an inhibition
operator) cannot reconstruct that behavior from the string; it has to already
know that `corrects` suppresses and `similar_to` is mutual.

This doc makes that implicit algebra **explicit and code-consumable** as a small,
closed set of per-verb dimensions, stored in a **relation registry**, derived by
the S2 classifier and validated by contract.

**Why now / why it's not premature:** the research answer to "directed vs
symmetric" is *per-relation algebra, not a global choice* (id:86714339). The
brain already found that the load-bearing edge operation is **inhibition** — "the
half that works" (id:091c0fb9, id:328d2ac3). And an edge-info census confirmed
the *verb's properties*, not raw direction, carry the traversal signal
(id:25a93edb). So the registry isn't speculative metadata — it's the layer many
known consumers already need. Wiring those consumers is a separate effort; this
builds the layer they read.

**The reasoning is in this doc on purpose.** The classifier prompt is a
distillation of Sections 1–2. If the rationale here is muddy, the prompt inherits
the mud. So each dimension is defined by its *semantic story* (what it means) AND
its *functional story* (what a consumer does with it) — because a dimension that
can't tell both stories is not a real dimension.

---

## 1. The dimensions

Four per verb. Each is a **closed value set** (code logic will branch on it), each
tells a semantic and a functional story, and each has a truth/distinctness test.

### 1.1 `symmetry` — does the relation have an actor at all?

- **Values:** `symmetric` | `asymmetric`
- **Semantic:** is `r(a,b)` logically the same claim as `r(b,a)`? `similar_to`,
  `contradicts`, `co_occurred_with` — yes (mutual). `corrects`, `enables`,
  `part_of` — no (one end acts on the other).
- **Functional:** tells the traversal layer whether to **symmetrize** the edge
  (store either way, walk both) or **respect** it as directional. Today the graph
  symmetrizes *everything* blindly (`get_neighbors` is bidirectional); this makes
  that principled.
- **Truth test:** a clean logical test — "does swapping the endpoints preserve the
  claim?" Low erosion risk.

### 1.2 `direction` — which endpoint is the actor?

- **Values:** `source_is_actor` | `target_is_actor` (only meaningful when
  `symmetry = asymmetric`)
- **Semantic:** for a directional edge, which node is the *doer*. `corrects` →
  the corrector is the actor (source). Inverse-voice verbs like `corrected_by` /
  `enabled_by` / `produced_by` → the actor is the **target**.
- **Functional:** this is the frame **sign hangs on**. "There is an inhibitory
  edge between A and B" is inert until you know A inhibits B, not the reverse.
  Direction tells the consumer *which way sign flows*. (This is the correction to
  the Turn-9 "drop direction" mistake — id:bd0fb159: dropping direction because
  its physical *bias* is benign is not the same as it being unnecessary as a
  schema fact. Sign requires it.)
- **NOTE — this is "actor position," not "old vs new."** The old/new framing was
  a write-time heuristic for *resolving* physical direction; it's not the
  dimension. The dimension is intrinsic to the verb's voice.
- **Open encoding choice (§7):** `symmetry` + `direction` can collapse into ONE
  three-valued field `orientation ∈ {symmetric, source_actor, target_actor}`.
  Cleaner (one field, no "N/A when symmetric" awkwardness); the cost is losing the
  explicit symmetric/asymmetric boolean some consumers may want directly.

### 1.3 `sign` — what operation the actor applies to the recipient

- **Values:** `reinforcing (+)` | `inhibitory (−)` | `gating (conditional)`
- **Semantic:** does the actor *support*, *suppress*, or *condition* the
  recipient?
  - `reinforcing` — adds support/evidence/elaboration: `validates`, `extends`,
    `grounds`, `strengthens`, `similar_to`.
  - `inhibitory` — supersedes/opposes/replaces: `corrects`, `supersedes`,
    `contradicts`, `weakens`, `overrides`.
  - `gating` — the recipient is *conditional on* the actor: `enables`,
    `depends_on`, `prerequisite_for`, `requires`. Not "+ support," but "exists /
    is relevant only if."
- **Functional — this is the crux, and it dictates the definition:** sign maps to
  the **algebraic operation a consumer applies**: reinforcing → add activation;
  inhibitory → subtract; gating → multiply/condition. That's why the operation
  vocabulary is exactly three.
- **THE guardrail (answers "did we pick true distinct groups"):** define sign
  **functionally (recall/activation-polarity), never logically.** Example:
  `part_of` is *logically* neutral (a chapter doesn't make the book "more true"),
  but *functionally* reinforcing (recall the book → its parts are relevant). If we
  mix the two definitions, every consumer erodes. Anchoring to function also makes
  the set nearly gap-free: a genuine `neutral` (no recall-polarity at all) is rare
  — pure temporal ordering is the main candidate, and it may fold into
  `reinforcing` (co-activation) too.
- **Erosion watch-point:** the `gating`-vs-`reinforcing` boundary. A dependency
  (`enables`) is conditional, not additive — but a lazy classifier will call it
  reinforcing. The prompt needs a sharp test: "is the recipient *only relevant if*
  the actor holds?" → gating; "does the actor *add weight to* an
  already-standalone recipient?" → reinforcing.

### 1.4 `transitivity` — does the relation chain?

- **Values:** `transitive` | `intransitive`
- **Semantic:** does `A r B` and `B r C` imply `A r C`? `depends_on`, `before`,
  `part_of`, `supersedes` — yes. `corrects`, `similar_to`, `validates` — no
  (similarity famously doesn't chain; a correction of a correction isn't a
  correction of the original).
- **Functional:** the guardrail for **multi-hop spread** — only propagate
  activation along composable chains; a non-transitive edge is single-hop.
- **Truth test:** clean logical test. **Caveat:** it's the *most correlated* with
  the others (transitive verbs skew asymmetric + structural), so it adds the least
  independent information — candidate to defer to v2 (§7).

### 1.5 The derivation heuristic (the generative rule the classifier uses)

Most verbs classify from **two orthogonal reads of the verb itself**:

- **Voice → direction:** is the verb phrased from the *doer's* side
  (`corrects`, `enables` → `source_is_actor`) or the *recipient's* side
  (`corrected_by`, `enabled_by` → `target_is_actor`)?
- **Valence → sign:** does the doer *support* (+), *replace/oppose* (−), or
  *condition* (gating) the recipient?

`symmetry` is the prior: if swapping endpoints preserves the claim, it's
symmetric and direction is moot. `transitivity` is a separate logical check.

This heuristic is what the classifier prompt teaches — not a lookup table, a way
of *reasoning* to the values, so it generalizes to emergent verbs.

---

## 2. Distinctness & truth (the value-erosion guarantee)

Tom's bar: *"make sure what we're choosing is distinct and true, otherwise
consumers will have value erosion."* Three defenses:

**2.1 Orthogonality — the axes carry independent information.** The test is that
every combination is populated by real verbs:

| | reinforcing | inhibitory | gating |
|---|---|---|---|
| **asymmetric** | `extends`, `validates` | `corrects`, `supersedes` | `enables`, `depends_on` |
| **symmetric** | `similar_to` | `contradicts` (mutual) | — (rare) |

`symmetric + inhibitory` (contradicts = mutual weakening) proves sign ≠ direction.
If any needed cell were empty, that axis would be redundant.

**2.2 Anchor to function, not semantics.** (See §1.3.) Every value is defined by
what a consumer *does* with it. This is the single biggest lever against erosion:
functionally-defined `sign` is clean and nearly gap-free; semantically-defined
`sign` breeds a fuzzy `neutral` and endless argument.

**2.3 Vocabulary-QA: clean-value-or-prune.** Some verbs resist a single true
value — `reframes` (replace the old lens = inhibitory, or add a lens =
reinforcing?), `triggers`, `informs`, `affects`. Per id:b756d494 we do **not**
sense-split a string. And a dimension carrying code logic **cannot** hold two
signs. So the resolution is: *a verb that can't take one clean, true value is
evidence the verb is too vague* — discourage/prune it from the vocabulary, don't
soften the dimension to fit it. **The requirement to assign a true value IS the
distinctness test for the vocabulary itself.** The ambiguous tail overlaps
exactly the vague verbs worth cutting anyway. This is what protects consumers:
dimensions stay clean because bad verbs are pruned, not accommodated.

**2.4 Closed list.** Because code branches on these values, the value set per
dimension is fixed and small (§1). New *verbs* are open (the classifier routes
them); new *dimension values* are a deliberate human edit, like adding an aspect.

---

## 3. Structure — the relation registry

**Current state:** `aspects_v1.json` stores `edge_relations` as bare string lists
under each aspect. A verb's only recorded property is which aspect(s) claim it.

**Change:** add a top-level `relations` registry keyed by verb; aspects keep their
string lists as the **grouping** layer (unchanged), referencing the registry.

```jsonc
{
  "relations": {
    "corrects":    { "symmetry": "asymmetric", "direction": "source_is_actor",
                     "sign": "inhibitory",  "transitive": false },
    "enables":     { "symmetry": "asymmetric", "direction": "source_is_actor",
                     "sign": "gating",      "transitive": true },
    "similar_to":  { "symmetry": "symmetric",  "direction": null,
                     "sign": "reinforcing", "transitive": false }
    // ...
  },
  "correction_improvement": { "edge_relations": ["corrects", ...], ... }
  // aspects unchanged — they reference relations by string
}
```

**Why a registry, not inline on the aspect:** these properties are **intrinsic to
the verb, not to the aspect**. A multi-homed verb (`supersedes` lives in
correction_improvement AND hierarchical_structure) has ONE symmetry, ONE sign —
storing them inline per aspect would duplicate and drift. The registry is the
single source; aspects stay about grouping. (This is the "attach the fact to the
relation, surfaced through aspects" conclusion.)

**"Multiple definitions per type?"** Single canonical vector per verb by default.
A genuinely polysemous verb *could* carry a per-sense override — but per
id:b756d494 the string is atomic, so the preferred move is **prune, not split**.
Per-sense override is an escape hatch of last resort, not a first-class pattern.

**Node types:** out of scope for v1. The aspect membership already encodes most of
what a node-type dimension would (episodic=raw ↔ wisdom=distilled). Revisit only
with a specific consumer in mind.

---

## 4. The classifier

The S2 `AspectIntegration` unit already routes new strings into aspects via
Sonnet. It gains a second job: emit the four dimensions per new edge verb.

- **Prompt:** distills Sections 1–2 — the value definitions, the derivation
  heuristic (§1.5), worked examples per value, and the clean-value-or-flag rule.
  It teaches *reasoning to the value*, not a lookup, so emergent verbs classify.
- **Output schema:** per-verb dimension object, closed-list enums, validated at
  the `AspectRegistry` write door (loud-by-default: reject a verb with a
  missing/invalid dimension rather than default it silently).
- **Ambiguity handling:** a verb the classifier can't cleanly value is *flagged*,
  not guessed — surfacing the vocabulary-QA candidates (§2.3) for human review.

---

## 5. Migration / seed curation

The bootstrap is **manual, then eval'd** (Tom's call):

1. **Manual seed pass (next session):** classify the existing ~200 edge verbs into
   the four dimensions — the verb table. The §1.5 heuristic does the bulk; the
   ambiguous tail gets clean-value-or-prune (§2.3).
2. **Classifier eval:** run the classifier against the manual gold; iterate the
   prompt until it reproduces the human classification; then activate it for
   emergent verbs only.

This mirrors how aspects bootstrapped: human-curated seed, classifier for growth.

---

## 6. Contract, tests, loud-by-default

- Extend the aspect contract tests: every edge verb in a `prompt_visible` aspect
  has a registry entry; every entry's values are in the closed enum; no verb
  missing a dimension.
- `AspectRegistry` write-door validates dimensions on every write.
- `sync-prompts` discipline for the classifier prompt (DB-authoritative, seed
  mirrors ACTIVE).
- Self-heal: a fresh brain seeds the registry from the committed baseline.

---

## 7. Open decisions (resolve in review)

1. **`symmetry` + `direction`: two fields or one collapsed `orientation`?**
   (Recommend: decide by which consumers need the plain symmetric/asymmetric
   boolean.)
2. **`transitivity`: v1 or v2?** (It's true and distinct but least independent.)
3. **`sign`: three values (`+`/`−`/gating) or force binary?** (Recommend three —
   gating maps to a real, distinct algebraic operation.)
4. **`neutral` sign: keep as a value, or prove it's empty under the functional
   definition and drop it?**
5. **Per-sense override: allow the escape hatch, or forbid (prune-only)?**

---

## 8. Phasing

- **P1 — Schema + seed curation.** Add the registry; manual-curate the verb table
  (the artifact). Contract tests.
- **P2 — Classifier.** Prompt + output schema; eval against the manual gold;
  activate for emergent verbs.
- **P3 — Backfill + growth.** Any remaining verbs; ongoing classifier routing.
- **(Later, separate) — Consumers.** Inhibition operator, principled
  symmetrize-vs-respect traversal, transitive multi-hop spread,
  query-conditioned aspect/type fields (id:220a2808, id:ba4c59c7). Explicitly out
  of scope here.
