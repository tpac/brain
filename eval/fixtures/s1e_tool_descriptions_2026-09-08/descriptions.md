# Generic brain tool descriptions — candidate v1

Authoring/review artifact. No guide, journal or runtime change; no behavioral result.

## remember_batch

Create multiple memory nodes and return their IDs and per-node outcomes. Each node accepts the remember fields and optional connect_to edges. New-node edges are resolved after all siblings are created, so sibling declaration order does not matter. Node creation and edge creation have separate outcomes; a created node can have an unresolved edge.

- `/properties/nodes/description`: Nodes to create, each with type, title, content and optional fields and edges.
- `/properties/nodes/items/properties/type/description`: Node category, such as fact, decision, lesson, mechanism, correction or open; open vocabulary.
- `/properties/nodes/items/properties/title/description`: Specific title used for identification and semantic retrieval.
- `/properties/nodes/items/properties/content/description`: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
- `/properties/nodes/items/properties/confidence/description`: Degree of support from 0.0 to 1.0. Hedged, contested or inferred claims use a value below 1.0.
- `/properties/nodes/items/properties/locked/description`: Protection flag. Lock requests from automated provenance are demoted; anchor provenance can create locked nodes.
- `/properties/nodes/items/properties/emotion/description`: Signed emotional intensity associated with the memory; pair with emotion_label.
- `/properties/nodes/items/properties/emotion_label/description`: Name of the emotional register, such as satisfaction or frustration.
- `/properties/nodes/items/properties/evolution_status/description`: Claim lifecycle: active, resolved, validated, confirmed, disproven or dismissed.
- `/properties/nodes/items/properties/source_turn_id/description`: Originating message_stream ID for episode linkage.
- `/properties/nodes/items/properties/situation/description`: Recall cue describing when this memory is relevant; used to match future situations.
- `/properties/nodes/items/properties/question/description`: Question this memory answers, phrased as a retrieval query.
- `/properties/nodes/items/properties/event_time/description`: When the remembered event happened, in ISO 8601; distinct from record creation time. Resolve relative dates against the source date and omit unsupported precision.
- `/properties/nodes/items/properties/reasoning/description`: Stored basis for the claim: evidence, inference, uncertainty and what would change it. Separate from reason, the revision audit note.
- `/properties/nodes/items/properties/thought/description`: Optional interpretation, hypothesis or connection beyond the stored account and its evidence. Can be revised independently and is returned beside the memory.
- `/properties/nodes/items/properties/their_raw_quote/description`: Verbatim words from the person or source whose account is being recorded.
- `/properties/nodes/items/properties/my_raw_quote/description`: Verbatim words from the agent's own contribution.
- `/properties/nodes/items/properties/correction_pattern/description`: Underlying error or behavioral pattern identified by the correction.
- `/properties/nodes/items/properties/source_context/description`: Context in which this memory originated.
- `/properties/nodes/items/properties/source_refs/description`: Optional links to the trace events that support this memory or provide useful scene visibility. Use existing 8-character hex trace IDs from source records or trace markers. Keep references selective; an abstraction without a particular source event may omit them.
- `/properties/nodes/items/properties/connect_to/description`: Relationships from this new node to existing IDs or siblings created in this batch. Sibling titles resolve after all nodes are created. Use this field for new-node edges rather than a separate connect operation, and emit each relationship once.
- `/properties/connect_to/description`: Apply the same relationship from every created node to one existing target; sibling targets are excluded. Node-level connect_to specifies individual edges.
- `/$defs/connect_to_item/properties/target/description`: Existing node: its exact 8-character hex ID from a returned record or supplied context. For remember only, another node created in this batch can be named by its exact title, in any declaration order. New siblings take precedence over title matches; creating a duplicate title does not update the existing node. Hex-shaped values are treated as IDs. Unresolved edge targets are reported and skipped without failing node creation. On revise, only existing node IDs are accepted.
- `/$defs/connect_to_item/properties/title/description`: Deprecated alias of `target`.
- `/$defs/connect_to_item/properties/relation/description`: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
- `/$defs/connect_to_item/properties/why/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
- `/$defs/connect_to_item/properties/relations/description`: Alternative to relation+why when the same pair carries multiple distinct relationships. Each item is {relation, why}.

## connect_batch

Create or update relationships between existing nodes. Each (source_id, target_id, relation) identifies one relationship: supplied fields update it, omitted fields preserve it, and an archived relationship is revived. Repeated calls do not increase weight unless a new weight is supplied. Each description must explain the specific relationship in at least 30 characters. Returns per-connection outcomes.

- `/properties/connections/description`: Relationships to create or update between existing nodes.
- `/properties/connections/items/properties/relation/description`: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
- `/properties/connections/items/properties/description/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
- `/properties/encoding_source/description`: Default provenance tag applied to all connections lacking their own.
- `/properties/chain_id/description`: Trace chain id for cross-event correlation (optional).
- `/properties/session_id/description`: Session id for activity tracking (optional).
- `/properties/reason/description`: Optional batch-level reason recorded in trace events.

## brain_batch

Apply remember, revise, connect, disconnect, archive and absorb operations in one batch. Operations run in list order within a transaction; edges from new nodes are resolved after sibling creation. Returns per-operation results and separate connect_to failures. A successful batch response can contain rejected operations; use the returned outcomes to determine which changes took effect. Operation names select actions; relationship verbs belong in relation fields.

For edges involving new nodes, use connect_to on remember with existing node IDs or exact sibling titles. New siblings take precedence on title collisions; a duplicate-title remember creates another node rather than revising the existing one. Do not also emit the same edge as a connect operation. connect_to on revise updates or adds this node's edges to existing IDs. Separate connect operations require two existing IDs and upsert the named relationship. Multiple relationships for a pair can use relations: [{relation, why}, ...].

absorb folds absorbed_id into survivor_id and archives the absorbed node. Edges, source_refs, access counts and metadata transfer, but the survivor keeps its own content unless a content override is supplied. Include any needed absorbed content in that full replacement; content_edits is not supported on absorb. The absorbed node must be archivable; the survivor may be locked. A failed absorb does not commit a partial merge.

- `/properties/operations/description`: Array of operations. Each object has an 'op' field plus that op's fields — per-op required fields and shapes are declared in the items schema (one branch per op).
- `/properties/operations/items/oneOf/0/properties/type/description`: Node type
- `/properties/operations/items/oneOf/0/properties/title/description`: Specific, scannable title
- `/properties/operations/items/oneOf/0/properties/content/description`: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
- `/properties/operations/items/oneOf/0/properties/connect_to/description`: Typed edges to siblings/catalog — see tool description
- `/properties/operations/items/oneOf/0/description`: Create a node with type, title and content. Also accepts the remember fields, including situation, reasoning, question, thought, event_time, their_raw_quote, my_raw_quote and source_refs. connect_to can address existing IDs or new sibling titles.
- `/properties/operations/items/oneOf/1/properties/node_id/description`: Node to revise
- `/properties/operations/items/oneOf/1/properties/reason/description`: Required audit note explaining this revision; recorded in history, not stored on the node. Supply reasoning separately to change the node's evidence statement.
- `/properties/operations/items/oneOf/1/properties/content/description`: New content, or swaps into the stored content
- `/properties/operations/items/oneOf/1/properties/connect_to/description`: This node's edges to change or add — an entry per target; see the item shape
- `/properties/operations/items/oneOf/1/properties/content_edits/description`: Deprecated alias of `content: [{old, new}, ...]`; passing both is an error.
- `/properties/operations/items/oneOf/1/description`: Update an existing node by ID. Omitted fields keep their current values. A writable text field accepts its complete new value, a swap {old, new}, or a list of swaps applied in order. Each old span must match exactly once; a missing or ambiguous match rejects the revision. Full replacement discards text omitted from the new value. Other fields take bare values. reason is the required audit note; reasoning is a separate stored field. Immutable id, created_at and locked are skipped with warnings. connect_to can update existing edges or add edges to existing nodes. source_refs replaces the reference list when supplied; omission preserves it, and [] clears it. Results report changes, warnings and failures.
- `/properties/operations/items/oneOf/2/properties/source_id/description`: Actor node id (must already exist)
- `/properties/operations/items/oneOf/2/properties/target_id/description`: Acted-upon node id (must already exist)
- `/properties/operations/items/oneOf/2/properties/relation/description`: Open-text verb
- `/properties/operations/items/oneOf/2/properties/description/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
- `/properties/operations/items/oneOf/2/description`: Create/update an edge between two EXISTING catalog nodes.
- `/properties/operations/items/oneOf/3/properties/source_id/description`: Edge source id
- `/properties/operations/items/oneOf/3/properties/target_id/description`: Edge target id
- `/properties/operations/items/oneOf/3/properties/relation/description`: Relation to archive
- `/properties/operations/items/oneOf/3/description`: Soft-archive one relation on an edge; other relations on the same edge survive.
- `/properties/operations/items/oneOf/4/properties/node_id/description`: Node to soft-archive
- `/properties/operations/items/oneOf/4/properties/reason/description`: Why (audit note)
- `/properties/operations/items/oneOf/4/properties/survivor_id/description`: ONLY when a live node REPLACES this one (supersession): its id. Records the redirect lineage (absorbed_into edge) recall walks to the successor. Omit for plain retirement; for merging content use absorb instead.
- `/properties/operations/items/oneOf/4/description`: Soft-archive a node.
- `/properties/operations/items/oneOf/5/properties/survivor_id/description`: Node that remains (may be locked)
- `/properties/operations/items/oneOf/5/properties/absorbed_id/description`: Node folded in + archived (must be archivable)
- `/properties/operations/items/oneOf/5/properties/content/description`: Complete survivor content after merging. Required to preserve any absorbed claim not already stated by the survivor; omitted content keeps only the survivor text.
- `/properties/operations/items/oneOf/5/description`: Merge absorbed_id into survivor_id, then archive the absorbed node. Accepts complete field overrides; see the tool description for content preservation.
- `/$defs/connect_to_item/properties/target/description`: Existing node: its exact 8-character hex ID from a returned record or supplied context. For remember only, another node created in this batch can be named by its exact title, in any declaration order. New siblings take precedence over title matches; creating a duplicate title does not update the existing node. Hex-shaped values are treated as IDs. Unresolved edge targets are reported and skipped without failing node creation. On revise, only existing node IDs are accepted.
- `/$defs/connect_to_item/properties/title/description`: Deprecated alias of `target`.
- `/$defs/connect_to_item/properties/relation/description`: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value.
- `/$defs/connect_to_item/properties/why/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule."
- `/$defs/connect_to_item/properties/relations/description`: Alternative to relation+why when the same pair carries multiple distinct relationships. Each item is {relation, why}.
- `/$defs/revise_connect_to_item/properties/target/description`: Exact 8-character hex ID of an existing node. Sibling titles are not accepted on revise.
- `/$defs/revise_connect_to_item/properties/title/description`: Deprecated alias of `target`.
- `/$defs/revise_connect_to_item/properties/relation/description`: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value. A bare value identifies the relationship; required when the pair has multiple relationships. A swap {old, new} renames it while preserving weight and history. If no edge exists, the bare value names the new relation.
- `/$defs/revise_connect_to_item/properties/why/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule." A bare value replaces the description; a swap {old, new} patches it. A new edge requires a complete description.
- `/$defs/revise_connect_to_item/properties/relations/description`: Alternative to relation+why when the same pair carries multiple distinct relationships. Each item is {relation, why}.
- `/$defs/swap/properties/old/description`: Exact, unique span of the current value
- `/$defs/swap/properties/new/description`: Replacement text

## revise_batch

Revise multiple nodes, with a node_id and audit reason for each item. Update an existing node by ID. Omitted fields keep their current values. A writable text field accepts its complete new value, a swap {old, new}, or a list of swaps applied in order. Each old span must match exactly once; a missing or ambiguous match rejects the revision. Full replacement discards text omitted from the new value. Other fields take bare values. reason is the required audit note; reasoning is a separate stored field. Immutable id, created_at and locked are skipped with warnings. connect_to can update existing edges or add edges to existing nodes. source_refs replaces the reference list when supplied; omission preserves it, and [] clears it. Results report changes, warnings and failures. Each revision records its own history event.

- `/properties/revisions/description`: Revisions, each with an existing node_id, audit reason and fields to change.
- `/properties/revisions/items/properties/node_id/description`: Full node ID to revise
- `/properties/revisions/items/properties/reason/description`: Required audit note explaining this revision; recorded in history, not stored on the node. Supply reasoning separately to change the node's evidence statement.
- `/properties/revisions/items/properties/content_edits/description`: Deprecated alias of `content: [{old, new}, ...]`; passing both is an error.
- `/properties/revisions/items/properties/connect_to/description`: This node's edges to change or add — an entry per target; see the item shape
- `/properties/revisions/items/properties/type/description`: Node category, such as fact, decision, lesson, mechanism, correction or open; open vocabulary.
- `/properties/revisions/items/properties/title/description`: Specific title used for identification and semantic retrieval.
- `/properties/revisions/items/properties/content/description`: Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.
- `/properties/revisions/items/properties/confidence/description`: Degree of support from 0.0 to 1.0. Hedged, contested or inferred claims use a value below 1.0.
- `/properties/revisions/items/properties/locked/description`: Immutable on revision; changes to locked are skipped with a warning.
- `/properties/revisions/items/properties/emotion/description`: Signed emotional intensity associated with the memory; pair with emotion_label.
- `/properties/revisions/items/properties/emotion_label/description`: Name of the emotional register, such as satisfaction or frustration.
- `/properties/revisions/items/properties/evolution_status/description`: Claim lifecycle: active, resolved, validated, confirmed, disproven or dismissed.
- `/properties/revisions/items/properties/source_turn_id/description`: Originating message_stream ID for episode linkage.
- `/properties/revisions/items/properties/situation/description`: Recall cue describing when this memory is relevant; used to match future situations.
- `/properties/revisions/items/properties/question/description`: Question this memory answers, phrased as a retrieval query.
- `/properties/revisions/items/properties/event_time/description`: When the remembered event happened, in ISO 8601; distinct from record creation time. Resolve relative dates against the source date and omit unsupported precision.
- `/properties/revisions/items/properties/reasoning/description`: Stored basis for the claim: evidence, inference, uncertainty and what would change it. Separate from reason, the revision audit note.
- `/properties/revisions/items/properties/thought/description`: Optional interpretation, hypothesis or connection beyond the stored account and its evidence. Can be revised independently and is returned beside the memory.
- `/properties/revisions/items/properties/their_raw_quote/description`: Verbatim words from the person or source whose account is being recorded.
- `/properties/revisions/items/properties/my_raw_quote/description`: Verbatim words from the agent's own contribution.
- `/properties/revisions/items/properties/correction_pattern/description`: Underlying error or behavioral pattern identified by the correction.
- `/properties/revisions/items/properties/source_context/description`: Context in which this memory originated.
- `/properties/revisions/items/properties/source_refs/description`: Complete replacement of this node's trace-reference list. Omit to preserve existing references; [] deliberately clears them. Values are existing 8-character hex trace IDs.
- `/$defs/revise_connect_to_item/properties/target/description`: Exact 8-character hex ID of an existing node. Sibling titles are not accepted on revise.
- `/$defs/revise_connect_to_item/properties/title/description`: Deprecated alias of `target`.
- `/$defs/revise_connect_to_item/properties/relation/description`: Specific relationship verb, such as supports, corrects, depends_on or implements. Use a specific verb rather than related/related_to or an empty value. A bare value identifies the relationship; required when the pair has multiple relationships. A swap {old, new} renames it while preserving weight and history. If no edge exists, the bare value names the new relation.
- `/$defs/revise_connect_to_item/properties/why/description`: Meaning of this particular relationship, used in retrieval. At least 30 characters explaining how these nodes connect, beyond restating their titles. Example: "The revised estimate accounts for the delay missing from the original schedule." A bare value replaces the description; a swap {old, new} patches it. A new edge requires a complete description.
- `/$defs/revise_connect_to_item/properties/relations/description`: Alternative to relation+why when the same pair carries multiple distinct relationships. Each item is {relation, why}.
- `/$defs/swap/properties/old/description`: Exact, unique span of the current value
- `/$defs/swap/properties/new/description`: Replacement text

## get_nodes

Fetch existing nodes by ID, including IDs obtained outside the current result set. The default view for up to 10 returned nodes includes full content, up to 8 edges with the total edge count, correction summaries and community references. Larger results use a scan view with 800 content characters and 5 edges per node. rich=true requests full content, all edges and full correction fields regardless of batch size. Missing IDs are reported. Recall results use the same default views.

- `/properties/node_ids/description`: Array of node IDs to fetch
- `/properties/rich/description`: False uses the batch-size-dependent bounded view. True requests full content, all edges and full correction fields for every returned node.

## recall_batch

Search memory by meaning for several queries. Returns a ranked result group for each query, with a shared optional field filter and per-query result limit.

- `/properties/queries/description`: Array of search queries
- `/properties/filter/description`: Field filter applied to every query: {field: {operator: value}}. Operators: exists, equals, in, contains, gte, lte.
- `/properties/limit/description`: Max results per query (default 5)

