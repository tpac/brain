# Tool-definition interview — brain_batch_v0_live

model=claude-sonnet-4-6 — each lens is an independent stateless call

## Contract recall — required fields per op

## Operations Accepted by `brain_batch`

The tool accepts **six operations**, specified via the `op` field inside each element of the `operations` array.

---

### Top-Level Required Field

| Field | Meaning |
|---|---|
| `operations` | Array of operation objects. Each object must have at minimum an `op` field. |

---

### Per-Operation Required Fields

#### 1. `remember`
- **Required:** `op`
- All other fields (content, title, connect_to, etc.) — **not stated as required** in the schema. The description mentions `connect_to` as optional.

#### 2. `revise`
- **Required:** `op`, `node_id`, `reason`
  - `node_id` — the existing node to update
  - `reason` — audit note for the revision, recorded in trace events; **not** stored on the node itself; distinct from the node field `reasoning`

#### 3. `connect`
- **Required:** `op`
- The description says both IDs "must already exist in the brain" and references `source_id`, `target_id`, `relation`, `description`, and `why` — but **none are listed as required in the schema explicitly**. The schema only requires `op`.

#### 4. `disconnect`
- **Required:** `op`
- No additional required fields stated in the schema or description.

#### 5. `archive`
- **Required:** `op`
- No additional required fields stated.

#### 6. `absorb`
- **Required:** `op`
- The description references `absorbed_id` and `survivor_id` as the core identifiers, but they are **not stated as required in the schema explicitly**.

---

### Summary

The **only field explicitly marked required** at the schema level is:
- `operations` (top-level)
- `op` (per operation item)

For `revise`, `node_id` and `reason` are **explicitly called out as required in the description text**, making them the only per-op required fields stated beyond `op`. All other per-op fields are described functionally but not formally declared required.

## Usage write-out — the reason/reasoning probe

```json
[
  {
    "op": "revise",
    "node_id": "ab12cd34",
    "reasoning": "derived from the v29 trace audit",
    "reason": "rationale was stale after the schema migration"
  }
]
```

**Field-naming doubt:** The definition says `revise` updates fields on the node like any other field, but it never explicitly names the node field that stores "why this node was encoded" — I inferred it is `reasoning` from the parenthetical *"do not confuse it with the node FIELD `reasoning`, which a revise op updates like any other field"*, but the definition does not enumerate all node fields, so I cannot be fully certain `reasoning` is the canonical field name for that rationale rather than, say, `rationale` or `encoding_reason`.

## Trap inventory — what a hurried agent gets wrong

## Traps Ranked by Likelihood

---

### 1. Using `reason` as a node field instead of an audit note (revise op)
**Likelihood: Very High**

A hurried agent will store the `reason` value thinking it persists on the node, or conversely will put their intended audit note into `reasoning` (the actual node field).

> *"requires `node_id` and `reason` (the audit note for the revision, recorded in trace events, NOT stored on the node; do not confuse it with the node FIELD `reasoning`, which a revise op updates like any other field)"*

---

### 2. Using `connect` for an edge involving a newly `remember`ed node
**Likelihood: Very High**

Agent emits a `remember` op and then a separate `connect` op referencing the new node's id in the same batch, not realizing the id doesn't exist at call time.

> *"NEVER use `connect` for an edge involving a new node (its id doesn't exist until this round finishes — forces a wasted second round); use `connect_to` inside the `remember` op instead."*

---

### 3. Double-emitting an edge (both `connect_to` and a separate `connect` op)
**Likelihood: High**

Agent uses `connect_to` inside a `remember` op AND also emits a standalone `connect` op for the same pair, thinking redundancy is safe.

> *"Don't double-emit: an edge already in `connect_to` must NOT also appear as a separate `connect` op for the same pair"*

---

### 4. Inventing op names like `consolidate`, `merge`, `evolve`, `keep`, `skip`
**Likelihood: High**

Agent semantically reasons about what it wants to do and invents a plausible-sounding op name instead of mapping to the six valid ops.

> *"Do NOT invent structural op names like 'consolidate'/'evolve'/'keep'/'skip' — a node merge is the `absorb` op"* and the enum `["absorb","archive","connect","disconnect","remember","revise"]`

---

### 5. Using a relation name as an op type
**Likelihood: High**

Agent emits `{op: 'similar_to', ...}` or `{op: 'supersedes', ...}` treating semantic relation values as op names.

> *"`similar_to`, `corrects`, `supersedes`, `reframes`, `extends`, `grounds`, etc. are values for the `relation` field on a `connect` op, never op types themselves."*

---

### 6. Writing short/generic `why` values
**Likelihood: High**

Agent writes `why: "related"` or `why: "example of"` — plausible-looking but explicitly invalid.

> *"Every edge `why` must be specific (≥30 chars, names the insight between the two nodes); empty/generic `why` ('related', 'connected', 'example of') pollutes the activation kernel"*

**Nothing in the schema enforces this** — no `minLength` constraint exists in `inputSchema`. The definition text warns against it but a hurried agent won't catch it at validation time.

---

### 7. Assuming `absorb` merges content automatically
**Likelihood: High**

Agent uses `absorb` expecting both nodes' content to be combined, losing the absorbed node's content silently.

> *"it is content-DESTRUCTIVE: the survivor KEEPS ITS OWN content — the absorbed node's content is NOT merged and is lost unless you pass a `content` override that folds it in (with an `(id:)` ref)."*

---

### 8. Expecting `connect` to auto-strengthen weight on repeat calls
**Likelihood: Medium**

Agent calls `connect` again on an existing edge to reinforce it, but weight is not incremented.

> *"Does NOT auto-strengthen weight on repeat."*

**Nothing in the schema flags this** — behavior is silent and the definition only states it once.

---

### 9. Sibling title collision with catalog node — resolving to wrong node
**Likelihood: Medium**

Agent `remember`s a node with the same title as an existing catalog node intending to update the catalog node, not realizing a new sibling is created and the catalog node is bypassed.

> *"NEW wins on title collision (a sibling whose title matches a catalog node resolves to the sibling, not the catalog — if you actually meant the catalog node, `revise` it instead of duplicate-title `remember`)"*

---

### 10. Using `relations: [...]` syntax when only one relation exists (or vice versa)
**Likelihood: Medium**

Agent uses `relation`+`why` flat fields when multiple distinct relationships exist (should use `relations` array), or uses the `relations` array for a single relationship unnecessarily — the latter is probably fine but the former drops edges silently.

> *"For one pair carrying multiple distinct relationships, use `relations: [{relation, why}, ...]` in place of `relation`+`why`"*

**The schema does not enforce mutual exclusivity** between `relation`/`why` and `relations` — no `oneOf`/`not` constraint exists.

---

### 11. Using `brain_batch` for a pure single-type batch instead of the specialized tool
**Likelihood: Low-Medium**

Not technically invalid, but contrary to guidance and wastes efficiency.

> *"For pure single-type batches use `remember_batch` / `revise_batch` / `connect_batch`; the moment you have a mix, switch to brain_batch."*

**Nothing in the schema enforces this** — it will execute without error.

## Emphasis audit — word budget vs importance

## What the Definition Spends the Most Words On

The bulk of the text covers **`remember`/`connect_to` edge resolution mechanics** (sibling vs. catalog scope, order-agnostic resolution, title collision/NEW-wins rule, double-emit prohibition) and **`absorb`** (content-destructive behavior, lossless conditions, content override, locked node rules).

---

## Top 3 Over-Weighted Topics

1. **Sibling resolution order / NEW-wins title collision** — Detailed but edge-case behavior; most calls don't involve title collisions between siblings and catalog nodes, so the word investment exceeds practical frequency.

2. **`absorb` content-destructive caveats** — The "survivor keeps its own content / absorbed content is lost" warning is explained repeatedly with multiple conditions; important once, but the repetition inflates word count beyond utility.

3. **Relation names ≠ op names (anti-pattern warning)** — The `similar_to`/`corrects`/`supersedes` example is a useful guard-rail, but devoting a full paragraph to a single invalid-op failure mode is disproportionate to how often that mistake occurs.

---

## Top 3 Under-Weighted Topics

1. **Required fields per op** — The schema lists only `op` as required; nowhere does the definition systematically state what fields each op *requires* (e.g., `remember` needs `title`/`content`, `revise` needs `node_id`+`reason`, `connect` needs `source_id`+`target_id`+`relation`). This is the #1 source of malformed calls.

2. **`connect` idempotency / weight behavior** — "Does NOT auto-strengthen weight on repeat" is mentioned once in passing, but the practical implication (you must explicitly set `weight` if you want it changed) is never clarified, leading to silent semantic drift.

3. **`disconnect` — completely underdescribed** — It is listed as a valid op but receives zero explanation of required fields, behavior, or when to use it vs. `archive`/`absorb`, making correct emission essentially guesswork.

## Prose vs schema — what belongs where

## Bucket A: Duplicates / Could Be Structurally Encoded

These are statements the prose makes that the JSON schema already encodes or could encode with schema changes.

---

**A1. "Six valid op values: 'remember', 'revise', 'connect', 'disconnect', 'archive', 'absorb'"**

The schema already encodes this via the `enum` on `op`. The prose list is fully redundant.

---

**A2. "'revise' updates an existing node — requires `node_id` and `reason`"**

The schema could encode this with `oneOf`/`if-then` discriminated by `op`, making `node_id` and `reason` required properties in the `revise` branch. Currently the schema has no per-op shapes at all — every op item is just `{op: string}`.

**Schema change:** Add discriminated union shapes under `items` using `oneOf` with `const` on `op`, each with its own `required` array.

---

**A3. "'connect' creates OR updates an edge between two EXISTING catalog nodes — both ids must already exist"**

The existence constraint can't be enforced structurally, but the *shape* (requiring `source_id`, `target_id`) absolutely could be encoded in a `connect`-branch schema. The current schema encodes none of the per-op required fields.

**Schema change:** In the `connect` discriminated branch, add `"required": ["source_id", "target_id", "relation"]`.

---

**A4. "For one pair carrying multiple distinct relationships, use `relations: [{relation, why}, ...]` in place of `relation`+`why`"**

This is a field-shape rule. The schema could encode `relations` as an array property with items typed `{relation: string, why: string}`, and could use `oneOf` to express that either `relation`+`why` or `relations` is present (mutually exclusive).

**Schema change:** Add `relations` as a typed array property in the `connect`/`connect_to` branch, with an `oneOf` or `not both` constraint.

---

**A5. "Every edge `why` must be ≥30 chars"**

This is a string-length constraint. The schema could encode `"minLength": 30` on the `why` field in every context it appears (`connect_to`, `connect`, `relations` items).

**Schema change:** Add `"minLength": 30` to the `why` string property wherever it is defined.

---

## Bucket B: Judgment Guidance Only Prose Can Carry

These convey intent, priority, disambiguation, and consequence — not enforceable structure.

---

**B1. "NEW wins on title collision (a sibling whose title matches a catalog node resolves to the sibling, not the catalog — if you actually meant the catalog node, `revise` it instead of duplicate-title `remember`)"**

This is resolution-order policy and a decision heuristic. No schema can express the semantic meaning of "you probably meant X, do Y instead." It requires the caller to understand intent and consequence.

---

**B2. "It is lossless ONLY when the survivor already states the absorbed node's claim, or you write the merged content"**

This is a warning about data loss conditionality — a judgment the caller must apply before choosing whether to pass a `content` override. The schema can mark `content` as optional but cannot encode when omitting it is safe vs. destructive.

---

**B3. "empty/generic `why` ('related', 'connected', 'example of') pollutes the activation kernel and fails to match queries about the relationship"**

The ≥30-char minimum is encodable (see A5), but *why* specificity matters — the downstream semantic consequence for query matching — is judgment guidance. The schema cannot express that "related" is harmful even if it were 30 characters.

---

**B4. "NEVER use `connect` for an edge involving a new node (its id doesn't exist until this round finishes — forces a wasted second round); use `connect_to` inside the `remember` op instead"**

This is a cross-op ordering constraint grounded in system mechanics (id availability timing). It cannot be encoded in schema because it depends on understanding the execution lifecycle and what constitutes a "wasted round."

---

**B5. "Do NOT invent structural op names like 'consolidate'/'evolve'/'keep'/'skip' — a node merge is the `absorb` op; the rest are semantic decisions expressed through which real op you emit. Relation names are NOT op names."**

This addresses a failure mode specific to LLM callers who hallucinate op names from natural language. The enum already rejects invalid ops structurally, but the *reasoning* — that relation values like `similar_to` get mistakenly used as op types — is a conceptual correction no schema constraint can anticipate or explain.

## Blind spots — unanswered usage questions

Here are the unanswered questions that matter for correct usage:

---

## Field Semantics

**1. What fields does `remember` accept beyond `connect_to`?**
The definition mentions `content`, `title`, `confidence`, `situation`, `reasoning` indirectly (via `revise` context), but never enumerates the full field set for a `remember` op. You don't know what's required vs. optional.

**2. What fields does `revise` accept beyond `node_id` and `reason`?**
It says "updates like any other field" but never lists which fields exist or are updatable. You have to infer from context (`content`, `title`, `confidence`, `situation`, `reasoning`) with no guarantee that list is complete.

**3. What fields does `connect` require beyond source/target ids?**
`relation`, `description`, `why` are mentioned but it's unclear which are required vs. optional, and whether `relations: [{relation, why}]` (mentioned for `remember`'s `connect_to`) also applies to standalone `connect` ops.

**4. What fields does `disconnect` require?**
The definition says only that it "removes an edge relation." It doesn't say whether you need source id, target id, relation name, or all three. Without knowing the required fields, you can't emit a valid `disconnect` op.

**5. What fields does `archive` require?**
Only described as "soft-archives a node" — no fields specified (presumably `node_id`, but this is not stated).

**6. What is the difference between `description` and `why` on a `connect` op?**
Both appear in the definition. It's unclear if they are the same field, aliases, or distinct fields serving different purposes.

**7. What does `confidence` accept?**
Is it a float 0–1, an integer, an enum string? Not defined.

**8. What is `situation`?**
Referenced as a field but never explained semantically.

---

## Partial Failure Behavior

**9. What happens when one operation in a batch fails?**
Does the entire batch roll back? Do subsequent ops still run? Do prior ops commit? This is critical — if op 3 of 10 fails, you don't know the state of ops 1–2 or 4–10.

**10. What is returned on failure?**
No error response shape is described. You don't know how to detect which op failed or what error information is available to act on.

**11. What happens if a `connect_to` sibling reference fails to resolve?**
Is the `remember` op itself aborted, or does the node get created without the edge?

---

## Interaction Between Operations

**12. Can `revise` and `remember` target the same title in one batch?**
If a `remember` op creates a node with title X, and a `revise` op in the same batch targets a node also named X (by id), is the `revise` applied to the new sibling or the existing node? The collision rule only addresses `connect_to` resolution, not `revise` targeting.

**13. Can `absorb` and `revise` operate on the same node in one batch?**
E.g., revising the survivor before absorbing into it — is the order guaranteed to matter, and what is the resolved state?

**14. Can `connect` reference a node being created by a `remember` in the same batch?**
The definition explicitly says no (use `connect_to` instead), but it doesn't say what *happens* if you do — silent failure, error, or undefined behavior?

**15. Does sibling resolution for `connect_to` work by title or by a temporary id?**
Only "title" is implied, but if two siblings share the same title (presumably an error), behavior is undefined.

---

## Defaults and Implicit Behavior

**16. What is the default `weight` on a new edge?**
The definition says `connect` does "not auto-strengthen weight on repeat" but never states the initial default weight or whether weight is even a settable field.

**17. What does "sequential" execution mean for ordering guarantees?**
It says "operations run sequentially" but also says sibling resolution is "order-agnostic." Are there any cases where op order matters for correctness beyond the `connect`-on-new-node prohibition?

**18. What does `(id:)` ref syntax mean in the `absorb` content override?**
The definition says to fold in content "with an `(id:)` ref" but never explains the syntax or how it resolves.

**19. What is returned on success?**
No response schema is given. You don't know if you get back the new node ids, edge ids, confirmation per-op, or a single acknowledgment.

**20. What counts as "locked" or "critical" for `absorb` refusal?**
The definition says "locked/critical refused" for the absorbed node but never defines what makes a node locked or critical, so you can't predict when `absorb` will be rejected.
