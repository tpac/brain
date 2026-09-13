# Tool descriptions: the V3.1–V3.3 eval arms (2026-09-08 generic candidate) vs the branch live schemas (deploy path)

Shapes identical for all six tools; only descriptions differ. Left = candidate (what V3.1–V3.3 ran on), right = live (what production ran on and what deploys).

## brain_batch — 14 of 42 descriptions differ
- `/`
  - candidate: Apply remember, revise, connect, disconnect, archive and absorb operations in one batch. Operations run in list order within a transaction; edges from new nodes are resolved after sibling creation. Returns per-operation results and separate connect_to failures. A successful batch response can contain rejected operations; use the returned outcomes to determine which changes took effect. Operation names select actions; relationship verbs belong in relation fields.

For edges involving new nodes, use connect_to on remember with existing node IDs or exact sibling titles. New siblings take precedence on title collisions; a duplicate-title remember creates another node rather than revising the existing one. Do not also emit the same edge as a connect operation. connect_to on revise updates or adds this node's edges to existing IDs. Separate connect operations require two existing IDs and upsert the named relationship. Multiple relationships for a pair can use relations: [{relation, why}, ...].

absorb folds absorbed_id into survivor_id and archives the absorbed node. Edges, source_refs, access counts and metadata transfer, but the survivor keeps its own content unless a content override is supplied. Include any needed absorbed content in that full replacement; content_edits is not supported on absorb. The absorbed node must be archivable; the survivor may be locked. A failed absorb does not commit a partial merge.
  - live: Execute multiple brain operations in one call — the default tool for MIXED batches (remember + revise + connect + archive in any combination), packed into ONE LLM round. For a pure single-type batch use `remember_batch` / `revise_batch` / `connect_batch`; the moment you mix, use brain_batch. Operations run sequentially in one transaction. Per-op required fields and meanings are declared in the schema — emit only the six declared ops: semantic decisions like 'consolidate'/'keep'/'skip' are expressed through which real op you emit, and relation verbs (`similar_to`, `corrects`, `supersedes`, ...) are values for `connect`'s relation field, never op names.

remember + edges: `connect_to` targets resolve in two scopes — SIBLINGS (other remember ops in this same batch, order-agnostic; resolution runs after all siblings are created) and CATALOG (existing nodes by title). NEW wins on title collision: a sibling whose title matches a catalog node resolves to the sibling — if you actually meant the catalog node, `revise` it instead of duplicate-title remember. NEVER use a `connect` op for an edge involving a new node (its id doesn't exist until this round finishes) — that is what connect_to is for. Don't double-emit: an edge already in connect_to must NOT also appear as a separate connect op for the same pair. For one pair carrying multiple distinct relationships, use `relations: [{relation, why}, ...]` in place of `relation`+`why`.

revise + edges: `connect_to` on a revise changes the edge that node already has to `target` (its why or relation, value or swap) — the way to repair a stale edge description; `connect` is for new edges between nodes you are neither creating nor revising.

connect: both ids must already exist in the brain. Idempotent upsert — specified fields update existing rows, unspecified preserve; weight does NOT auto-strengthen on repeat.

absorb IS the real merge — it folds `absorbed_id` INTO `survivor_id`: edges, source_refs, access_count, and metadata transfer automatically and the absorbed is archived. BUT the survivor KEEPS ITS OWN content — the absorbed node's content is lost unless you pass a `content` override that folds it in (with an `(id:)` ref). Lossless ONLY when the survivor already states the absorbed claim, or you write the merged content. The absorbed must be archivable (locked/critical refused); the survivor MAY be locked — you absorb INTO the canonical node.

Every edge `why`/`description` must be specific (>=30 chars, naming the insight between the two nodes) — generic 'related'/'connected'/'example of' pollutes the activation kernel and never matches queries about the relationship.
- `/input_schema/$defs/connect_to_item/properties/relation`
  - candidate: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
  - live: Edge relation, open text — e.g. refines, grounds, corrects, depends_on, supersedes, triggers, implements, anchored_to, during. Invent a specific verb when none fits better. NEVER `related`/`related_to`/empty — generic relations pollute the activation kernel and match no query.
- `/input_schema/$defs/connect_to_item/properties/target`
  - candidate: Existing node: its exact 8-character hex ID from a returned record or supplied context. For remember only, another node created in this batch can be named by its exact title, in any declaration order. New siblings take precedence over title matches; creating a duplicate title does not update the existing node. Hex-shaped values are treated as IDs. Unresolved edge targets are reported and skipped without failing node creation. On revise, only existing node IDs are accepted.
  - live: The other node: for an EXISTING node, its exact 8-char hex id copied verbatim from any visible id: surface (a hex-shaped value is always treated as an id — never matched as a title — and on a miss dropped loudly). On remember only, a node created in this same batch may be named by its exact title (siblings resolve before catalog matches, any declaration order; NEW wins on title collision — to update an existing catalog node use `revise` on its id, not a duplicate-title remember). On revise the target must be an id. Unresolved targets are logged and skipped, never failing the batch.
- `/input_schema/$defs/connect_to_item/properties/why`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
  - live: What the edge MEANS — the insight living between the two nodes, not a summary of either. Embedded for query matching; >=30 chars, or drop the edge.
BAD: "example of the principle" — generic gloss, no insight about which example or why.
GOOD: "the assumption treated concurrent access as a thread-safety question; the correction reframes it as wal-index contention — different failure mode, different fix".
- `/input_schema/$defs/revise_connect_to_item/properties/relation`
  - candidate: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value. A bare value identifies the relationship; required when the pair has multiple relationships. A swap {old, new} renames it while preserving weight and history. If no edge exists, the bare value names the new relation.
  - live: Edge relation, open text — e.g. refines, grounds, corrects, depends_on, supersedes, triggers, implements, anchored_to, during. Invent a specific verb when none fits better. NEVER `related`/`related_to`/empty — generic relations pollute the activation kernel and match no query. On revise a bare relation identifies the edge row (required when the pair carries more than one relation); a swap {old, new} renames it in place — weight and history survive. Also the relation to create when the pair has no edge yet.
- `/input_schema/$defs/revise_connect_to_item/properties/target`
  - candidate: Exact 8-character hex ID of an existing node. Sibling titles are not accepted on revise.
  - live: The other node's exact 8-char hex id — an existing node; sibling titles are not valid on revise.
- `/input_schema/$defs/revise_connect_to_item/properties/why`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule." A bare value replaces the description; a swap {old, new} patches it. A new edge requires a complete description.
  - live: What the edge MEANS — the insight living between the two nodes, not a summary of either. Embedded for query matching; >=30 chars, or drop the edge.
BAD: "example of the principle" — generic gloss, no insight about which example or why.
GOOD: "the assumption treated concurrent access as a thread-safety question; the correction reframes it as wal-index contention — different failure mode, different fix". On revise a bare string replaces the description and a swap {old, new} patches it; required, bare, when the edge is being created.
- `/input_schema/properties/operations/items/oneOf[0]`
  - candidate: Create a node with type, title and content. Also accepts the remember fields, including situation, reasoning, question, thought, event_time, their_raw_quote, my_raw_quote and source_refs. connect_to can address existing IDs or new sibling titles.
  - live: Create a node. Accepts all remember() fields (situation, reasoning, quotes, ...).
- `/input_schema/properties/operations/items/oneOf[0]/properties/content`
  - candidate: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
  - live: Rich content
- `/input_schema/properties/operations/items/oneOf[1]`
  - candidate: Update an existing node by ID. Omitted fields keep their current values. A writable text field accepts its complete new value, a swap {old, new}, or a list of swaps applied in order. Each old span must match exactly once; a missing or ambiguous match rejects the revision. Full replacement discards text omitted from the new value. Other fields take bare values. reason is the required audit note; reasoning is a separate stored field. Immutable id, created_at and locked are skipped with warnings. connect_to can update existing edges or add edges to existing nodes. source_refs replaces the reference list when supplied; omission preserves it, and [] clears it. Results report changes, warnings and failures.
  - live: Update an existing node. Any other key is a field (content, title, situation, question, reasoning, ...). On revise a field takes its NEW VALUE (the whole field replaced) or a swap `{old, new}` — a list of swaps for several spots — that changes only what is stale; `old` is copied VERBATIM from the node as shown and must occur exactly once, or the op fails loudly with the count and nothing is written. Fields not named are untouched. Edges ride as `connect_to` exactly as on remember: on revise an entry changes the edge this node already has to that `target` (its `why`, or its `relation` — value or swap), or creates it if there is none. Non-text fields (confidence, type, event_time, ...) take bare values. `content_edits` is the alias of `content: [swaps]`.
- `/input_schema/properties/operations/items/oneOf[1]/properties/reason`
  - candidate: Required audit note explaining this revision; recorded in history, not stored on the node. Supply reasoning separately to change the node's evidence statement.
  - live: Audit note for this revision — recorded in trace events, NOT stored on the node. Distinct from the node FIELD `reasoning`, which a revise op updates like any other field.
- `/input_schema/properties/operations/items/oneOf[2]/properties/description`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
  - live: What the edge MEANS (>=30 chars)
- `/input_schema/properties/operations/items/oneOf[5]`
  - candidate: Merge absorbed_id into survivor_id, then archive the absorbed node. Accepts complete field overrides; see the tool description for content preservation.
  - live: Lossless merge: fold absorbed INTO survivor. Accepts revise-shape field overrides (content, title, confidence, situation).
- `/input_schema/properties/operations/items/oneOf[5]/properties/content`
  - candidate: Complete survivor content after merging. Required to preserve any absorbed claim not already stated by the survivor; omitted content keeps only the survivor text.
  - live: Merged content override — REQUIRED for losslessness unless survivor already states the absorbed claim

## connect_batch — 4 of 8 descriptions differ
- `/`
  - candidate: Create or update relationships between existing nodes. Each (source_id, target_id, relation) identifies one relationship: supplied fields update it, omitted fields preserve it, and an archived relationship is revived. Repeated calls do not increase weight unless a new weight is supplied. Each description must explain the specific relationship in at least 30 characters. Returns per-connection outcomes.
  - live: Create or update multiple edges in one call. Same idempotent-upsert + field-preservation contract as `connect` — specified fields update on existing rows, unspecified preserve. Each connection entry MUST provide a specific `description` (≥30 chars naming the insight between the two nodes); bare edges with empty descriptions are recall dead weight.
- `/input_schema/properties/connections`
  - candidate: Relationships to create or update between existing nodes.
  - live: Array of connections to create
- `/input_schema/properties/connections/items/properties/description`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
  - live: What the edge MEANS — embedded for recall. Target ≥30 chars. Don't restate node titles. See connect_to.why for BAD/GOOD examples.
- `/input_schema/properties/connections/items/properties/relation`
  - candidate: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
  - live: Open-text verb — specific, never `related`/`related_to`

## get_nodes — 2 of 3 descriptions differ
- `/`
  - candidate: Fetch existing nodes by ID, including IDs obtained outside the current result set. The default view for up to 10 returned nodes includes full content, up to 8 edges with the total edge count, correction summaries and community references. Larger results use a scan view with 800 content characters and 5 edges per node. rich=true requests full content, all edges and full correction fields regardless of batch size. Missing IDs are reported. Recall results use the same default views.
  - live: Get multiple nodes by ID in one call. Two views, picked by count: up to 10 ids render in DETAIL — whole content, the top 8 edges with the total ("Edges (8 of 12)"), corrections at balanced depth, the node's communities as "title" (id) — for nodes you will act on; more than 10 render in SCAN — 800 chars of content, 5 edges — for judging fit across many. rich=true lifts DETAIL to every edge and the full correction K/V (reasoning, raw quotes). A recall renders its results through the same two views, so a recall and a pull of one node read the same.
- `/input_schema/properties/rich`
  - candidate: False uses the batch-size-dependent bounded view. True requests full content, all edges and full correction fields for every returned node.
  - live: Default false → bounded, batch-size-aware view. true → complete view (all edges + heavy correction K/V) for every node. Use sparingly on large batches — it is the firehose.

## recall_batch — 2 of 4 descriptions differ
- `/`
  - candidate: Search memory by meaning for several queries. Returns a ranked result group for each query, with a shared optional field filter and per-query result limit.
  - live: Run multiple recall queries in one call. Returns results for each query.
- `/input_schema/properties/filter`
  - candidate: Field filter applied to every query: {field: {operator: value}}. Operators: exists, equals, in, contains, gte, lte.
  - live: Dict filter applied to all queries. Same format as recall filter.

## remember_batch — 26 of 28 descriptions differ
- `/`
  - candidate: Create multiple memory nodes and return their IDs and per-node outcomes. Each node accepts the remember fields and optional connect_to edges. New-node edges are resolved after all siblings are created, so sibling declaration order does not matter. Node creation and edge creation have separate outcomes; a created node can have an unresolved edge.
  - live: Create multiple nodes in one call. Each node uses the same fields as remember(), plus an optional per-node `connect_to` for typed edges to siblings (in the same batch) and catalog nodes.
- `/input_schema/$defs/connect_to_item/properties/relation`
  - candidate: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
  - live: Edge relation, open text — e.g. refines, grounds, corrects, depends_on, supersedes, triggers, implements, anchored_to, during. Invent a specific verb when none fits better. NEVER `related`/`related_to`/empty — generic relations pollute the activation kernel and match no query.
- `/input_schema/$defs/connect_to_item/properties/target`
  - candidate: Existing node: its exact 8-character hex ID from a returned record or supplied context. For remember only, another node created in this batch can be named by its exact title, in any declaration order. New siblings take precedence over title matches; creating a duplicate title does not update the existing node. Hex-shaped values are treated as IDs. Unresolved edge targets are reported and skipped without failing node creation. On revise, only existing node IDs are accepted.
  - live: The other node: for an EXISTING node, its exact 8-char hex id copied verbatim from any visible id: surface (a hex-shaped value is always treated as an id — never matched as a title — and on a miss dropped loudly). On remember only, a node created in this same batch may be named by its exact title (siblings resolve before catalog matches, any declaration order; NEW wins on title collision — to update an existing catalog node use `revise` on its id, not a duplicate-title remember). On revise the target must be an id. Unresolved targets are logged and skipped, never failing the batch.
- `/input_schema/$defs/connect_to_item/properties/why`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
  - live: What the edge MEANS — the insight living between the two nodes, not a summary of either. Embedded for query matching; >=30 chars, or drop the edge.
BAD: "example of the principle" — generic gloss, no insight about which example or why.
GOOD: "the assumption treated concurrent access as a thread-safety question; the correction reframes it as wal-index contention — different failure mode, different fix".
- `/input_schema/properties/connect_to`
  - candidate: Apply the same relationship from every created node to one existing target; sibling targets are excluded. Node-level connect_to specifies individual edges.
  - live: Batch-level: applies the same edge from EVERY created node to one catalog target. Siblings excluded. For per-node edges, use node-level connect_to.
- `/input_schema/properties/nodes`
  - candidate: Nodes to create, each with type, title, content and optional fields and edges.
  - live: Array of node specs — same fields as remember(), plus optional per-node connect_to.
- `/input_schema/properties/nodes/items/properties/confidence`
  - candidate: Degree of support from 0.0 to 1.0. Hedged, contested or inferred claims use a value below 1.0.
  - live: 0.0-1.0. Set below 1.0 when the claim is hedged, contested, or inferred — recall exposes it and filters select on it. Don't fabricate precision.
- `/input_schema/properties/nodes/items/properties/connect_to`
  - candidate: Relationships from this new node to existing IDs or siblings created in this batch. Sibling titles resolve after all nodes are created. Use this field for new-node edges rather than a separate connect operation, and emit each relationship once.
  - live: Per-node typed edges from THIS node to siblings (created in the same batch) or catalog nodes. Sibling-aware (NEW wins on title collision), order-agnostic, fail-soft. USE THIS for any edge involving a new node — never use a separate `connect` op for new-node edges (`connect` requires ids that don't exist until round 1 finishes, forcing a needless second LLM round). DON'T DOUBLE-EMIT: an edge already in connect_to must NOT also appear as a separate connect op for the same pair. DON'T fake-revise: if the catalog has the title, use `revise` on its id — duplicate-title `remember` + connect_to would resolve to the new sibling (NEW wins) and leave the catalog version stale.
- `/input_schema/properties/nodes/items/properties/content`
  - candidate: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
  - live: Rich content — reasoning, tradeoffs, specifics. On revise: its new value, or `{old, new}` swaps into what is stored (`content_edits` is the alias of the swap list) — a full rewrite must re-author everything the node holds, and dropped details are silent losses.
- `/input_schema/properties/nodes/items/properties/correction_pattern`
  - candidate: Underlying error or behavioral pattern identified by the correction.
  - live: Behavioral pattern behind the correction.
- `/input_schema/properties/nodes/items/properties/emotion`
  - candidate: Signed emotional intensity associated with the memory; pair with emotion_label.
  - live: Emotional charge of the moment — signed; recall reads the magnitude. Pair with emotion_label.
- `/input_schema/properties/nodes/items/properties/emotion_label`
  - candidate: Name of the emotional register, such as satisfaction or frustration.
  - live: Name of the felt register ('satisfaction', 'frustration', ...).
- `/input_schema/properties/nodes/items/properties/event_time`
  - candidate: When the remembered event happened, in ISO 8601; distinct from record creation time. Resolve relative dates against the source date and omit unsupported precision.
  - live: When the remembered thing HAPPENED — ISO 8601, distinct from created_at (when it was written). Resolve relative dates to absolute; leave absent rather than guess. Read by the temporal lane at recall.
- `/input_schema/properties/nodes/items/properties/evolution_status`
  - candidate: Claim lifecycle: active, resolved, validated, confirmed, disproven or dismissed.
  - live: Claim lifecycle once settled: active | resolved | validated | confirmed | disproven | dismissed.
- `/input_schema/properties/nodes/items/properties/locked`
  - candidate: Protection flag. Lock requests from automated provenance are demoted; anchor provenance can create locked nodes.
  - live: Protect from casual revision. Belongs to the interactive session: a write from any automated source (encoder, S2, hooks) has its locked:true demoted at the write boundary. Locking is a rare act.
- `/input_schema/properties/nodes/items/properties/my_raw_quote`
  - candidate: Verbatim words from the agent's own contribution.
  - live: My own exact words — reflections, realizations, insights.
- `/input_schema/properties/nodes/items/properties/question`
  - candidate: Question this memory answers, phrased as a retrieval query.
  - live: The question this node answers, as the other side would ask it — gets its own recall embedding, bridging how it's stored and how it's asked for. Skip when the title already asks it.
- `/input_schema/properties/nodes/items/properties/reasoning`
  - candidate: Stored basis for the claim: evidence, inference, uncertainty and what would change it. Separate from reason, the revision audit note.
  - live: What this claim rests on — how it was established (measured, reported, inferred), how strongly, and what would change it. Written for a reader who has never seen this prompt. NOT revise()'s `reason` param — that is the audit note for a revision, recorded in trace events and never stored on the node.
- `/input_schema/properties/nodes/items/properties/situation`
  - candidate: Recall cue describing when this memory is relevant; used to match future situations.
  - live: When is this knowledge relevant? One sentence. Stored in node_metadata_kv (canonical); a derived _situation embedding row in node_enrichments provides recall scoring. Enrichment text column is deprecated for _situation — kv is the single source of truth.
- `/input_schema/properties/nodes/items/properties/source_context`
  - candidate: Context in which this memory originated.
  - live: Session/context when this was encoded.
- `/input_schema/properties/nodes/items/properties/source_refs`
  - candidate: Optional links to the trace events that support this memory or provide useful scene visibility. Use existing 8-character hex trace IDs from source records or trace markers. Keep references selective; an abstraction without a particular source event may omit them.
  - live: Trace event ids anchoring this node to its originating moments. Each id is an 8-char hex string copied verbatim from the trace markers in your input — the `trace="<hex>"` attribute on timeline turns, or `[trace:<hex>]` markers in conversation renders. Sparse by design: pick 1-3 load-bearing turns per node — the turn(s) whose content is what made this node encodeable. Adjacent context is what graph traversal is for; source_refs are for the moments that GENERATED this node. Leave empty when the node is a multi-session abstraction with no single anchor (pure-synthesis pattern). When content would just rewrite what the source already says clearly, point to the source instead of restating it (the pure-reference pattern). See EPISODIC-REFERENCES.md §7.4 for the full judgment rule.
- `/input_schema/properties/nodes/items/properties/source_turn_id`
  - candidate: Originating message_stream ID for episode linkage.
  - live: message_stream ID that produced this node (episode linkage)
- `/input_schema/properties/nodes/items/properties/their_raw_quote`
  - candidate: Verbatim words from the person or source whose account is being recorded.
  - live: Their exact words — my counterpart's, verbatim.
- `/input_schema/properties/nodes/items/properties/thought`
  - candidate: Optional interpretation, hypothesis or connection beyond the stored account and its evidence. Can be revised independently and is returned beside the memory.
  - live: My own read on the memory — a hunch, a connection, a take the content doesn't carry. Delivered: rendered beside the node at recall and in encoder catalogs. A living field — update it when a re-read moves it; most nodes carry none, and empty is correct.
- `/input_schema/properties/nodes/items/properties/title`
  - candidate: Specific title used for identification and semantic retrieval.
  - live: Specific and scannable — the title is itself an embedded recall vector; specificity is findability.
- `/input_schema/properties/nodes/items/properties/type`
  - candidate: Node category, such as fact, decision, lesson, mechanism, correction or open; open vocabulary.
  - live: Node type (decision, lesson, mechanism, correction, moment, open, ... — open vocabulary, use what fits).

## revise_batch — 25 of 32 descriptions differ
- `/`
  - candidate: Revise multiple nodes, with a node_id and audit reason for each item. Update an existing node by ID. Omitted fields keep their current values. A writable text field accepts its complete new value, a swap {old, new}, or a list of swaps applied in order. Each old span must match exactly once; a missing or ambiguous match rejects the revision. Full replacement discards text omitted from the new value. Other fields take bare values. reason is the required audit note; reasoning is a separate stored field. Immutable id, created_at and locked are skipped with warnings. connect_to can update existing edges or add edges to existing nodes. source_refs replaces the reference list when supplied; omission preserves it, and [] clears it. Results report changes, warnings and failures. Each revision records its own history event.
  - live: Revise multiple brain nodes in one call — one call, many revisions, instead of one call per node. On revise a field takes its NEW VALUE (the whole field replaced) or a swap `{old, new}` — a list of swaps for several spots — that changes only what is stale; `old` is copied VERBATIM from the node as shown and must occur exactly once, or the op fails loudly with the count and nothing is written. Fields not named are untouched. Edges ride as `connect_to` exactly as on remember: on revise an entry changes the edge this node already has to that `target` (its `why`, or its `relation` — value or swap), or creates it if there is none. Non-text fields take bare values; `content_edits` is the deprecated alias of `content: [swaps]`. Immutable fields ({id, created_at, locked}) skipped with warning. Each row emits its own trace event for revision history (queryable via `query_traces` with ref_type='node_revised').
- `/input_schema/$defs/revise_connect_to_item/properties/relation`
  - candidate: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value. A bare value identifies the relationship; required when the pair has multiple relationships. A swap {old, new} renames it while preserving weight and history. If no edge exists, the bare value names the new relation.
  - live: Edge relation, open text — e.g. refines, grounds, corrects, depends_on, supersedes, triggers, implements, anchored_to, during. Invent a specific verb when none fits better. NEVER `related`/`related_to`/empty — generic relations pollute the activation kernel and match no query. On revise a bare relation identifies the edge row (required when the pair carries more than one relation); a swap {old, new} renames it in place — weight and history survive. Also the relation to create when the pair has no edge yet.
- `/input_schema/$defs/revise_connect_to_item/properties/target`
  - candidate: Exact 8-character hex ID of an existing node. Sibling titles are not accepted on revise.
  - live: The other node's exact 8-char hex id — an existing node; sibling titles are not valid on revise.
- `/input_schema/$defs/revise_connect_to_item/properties/why`
  - candidate: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule." A bare value replaces the description; a swap {old, new} patches it. A new edge requires a complete description.
  - live: What the edge MEANS — the insight living between the two nodes, not a summary of either. Embedded for query matching; >=30 chars, or drop the edge.
BAD: "example of the principle" — generic gloss, no insight about which example or why.
GOOD: "the assumption treated concurrent access as a thread-safety question; the correction reframes it as wal-index contention — different failure mode, different fix". On revise a bare string replaces the description and a swap {old, new} patches it; required, bare, when the edge is being created.
- `/input_schema/properties/revisions`
  - candidate: Revisions, each with an existing node_id, audit reason and fields to change.
  - live: List of revisions. Each must have node_id and reason, plus any fields to update — same field semantics as `revise`.
- `/input_schema/properties/revisions/items/properties/confidence`
  - candidate: Degree of support from 0.0 to 1.0. Hedged, contested or inferred claims use a value below 1.0.
  - live: 0.0-1.0. Set below 1.0 when the claim is hedged, contested, or inferred — recall exposes it and filters select on it. Don't fabricate precision.
- `/input_schema/properties/revisions/items/properties/content`
  - candidate: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
  - live: Rich content — reasoning, tradeoffs, specifics. On revise: its new value, or `{old, new}` swaps into what is stored (`content_edits` is the alias of the swap list) — a full rewrite must re-author everything the node holds, and dropped details are silent losses.
- `/input_schema/properties/revisions/items/properties/correction_pattern`
  - candidate: Underlying error or behavioral pattern identified by the correction.
  - live: Behavioral pattern behind the correction.
- `/input_schema/properties/revisions/items/properties/emotion`
  - candidate: Signed emotional intensity associated with the memory; pair with emotion_label.
  - live: Emotional charge of the moment — signed; recall reads the magnitude. Pair with emotion_label.
- `/input_schema/properties/revisions/items/properties/emotion_label`
  - candidate: Name of the emotional register, such as satisfaction or frustration.
  - live: Name of the felt register ('satisfaction', 'frustration', ...).
- `/input_schema/properties/revisions/items/properties/event_time`
  - candidate: When the remembered event happened, in ISO 8601; distinct from record creation time. Resolve relative dates against the source date and omit unsupported precision.
  - live: When the remembered thing HAPPENED — ISO 8601, distinct from created_at (when it was written). Resolve relative dates to absolute; leave absent rather than guess. Read by the temporal lane at recall.
- `/input_schema/properties/revisions/items/properties/evolution_status`
  - candidate: Claim lifecycle: active, resolved, validated, confirmed, disproven or dismissed.
  - live: Claim lifecycle once settled: active | resolved | validated | confirmed | disproven | dismissed.
- `/input_schema/properties/revisions/items/properties/locked`
  - candidate: Immutable on revision; changes to locked are skipped with a warning.
  - live: Protect from casual revision. Belongs to the interactive session: a write from any automated source (encoder, S2, hooks) has its locked:true demoted at the write boundary. Locking is a rare act.
- `/input_schema/properties/revisions/items/properties/my_raw_quote`
  - candidate: Verbatim words from the agent's own contribution.
  - live: My own exact words — reflections, realizations, insights.
- `/input_schema/properties/revisions/items/properties/question`
  - candidate: Question this memory answers, phrased as a retrieval query.
  - live: The question this node answers, as the other side would ask it — gets its own recall embedding, bridging how it's stored and how it's asked for. Skip when the title already asks it.
- `/input_schema/properties/revisions/items/properties/reason`
  - candidate: Required audit note explaining this revision; recorded in history, not stored on the node. Supply reasoning separately to change the node's evidence statement.
  - live: Why this revision — audit note recorded in the trace event, NOT stored on the node. Required. Distinct from the node FIELD `reasoning` (why the node was encoded); to update that field, pass `reasoning` as well.
- `/input_schema/properties/revisions/items/properties/reasoning`
  - candidate: Stored basis for the claim: evidence, inference, uncertainty and what would change it. Separate from reason, the revision audit note.
  - live: What this claim rests on — how it was established (measured, reported, inferred), how strongly, and what would change it. Written for a reader who has never seen this prompt. NOT revise()'s `reason` param — that is the audit note for a revision, recorded in trace events and never stored on the node.
- `/input_schema/properties/revisions/items/properties/situation`
  - candidate: Recall cue describing when this memory is relevant; used to match future situations.
  - live: When is this knowledge relevant? One sentence. Stored in node_metadata_kv (canonical); a derived _situation embedding row in node_enrichments provides recall scoring. Enrichment text column is deprecated for _situation — kv is the single source of truth.
- `/input_schema/properties/revisions/items/properties/source_context`
  - candidate: Context in which this memory originated.
  - live: Session/context when this was encoded.
- `/input_schema/properties/revisions/items/properties/source_refs`
  - candidate: Complete replacement of this node's trace-reference list. Omit to preserve existing references; [] deliberately clears them. Values are existing 8-character hex trace IDs.
  - live: REPLACE semantics: passing this REPLACES the node's existing refs (atomic delete+insert). Omit the field entirely to preserve current refs; pass [] only to deliberately clear them — an empty list is a wipe, not a no-op. Ids are 8-char hex trace ids, same form as on remember.
- `/input_schema/properties/revisions/items/properties/source_turn_id`
  - candidate: Originating message_stream ID for episode linkage.
  - live: message_stream ID that produced this node (episode linkage)
- `/input_schema/properties/revisions/items/properties/their_raw_quote`
  - candidate: Verbatim words from the person or source whose account is being recorded.
  - live: Their exact words — my counterpart's, verbatim.
- `/input_schema/properties/revisions/items/properties/thought`
  - candidate: Optional interpretation, hypothesis or connection beyond the stored account and its evidence. Can be revised independently and is returned beside the memory.
  - live: My own read on the memory — a hunch, a connection, a take the content doesn't carry. Delivered: rendered beside the node at recall and in encoder catalogs. A living field — update it when a re-read moves it; most nodes carry none, and empty is correct.
- `/input_schema/properties/revisions/items/properties/title`
  - candidate: Specific title used for identification and semantic retrieval.
  - live: Specific and scannable — the title is itself an embedded recall vector; specificity is findability.
- `/input_schema/properties/revisions/items/properties/type`
  - candidate: Node category, such as fact, decision, lesson, mechanism, correction or open; open vocabulary.
  - live: Node type (decision, lesson, mechanism, correction, moment, open, ... — open vocabulary, use what fits).

