# Encoding/Decoding Pipeline v2

**Date:** 2026-03-23
**Git version:** f27d1dd (main)
**Author:** Claude Opus 4.6 + Tom
**Session:** #10 — Embedding Migration to LLM

---

## 1. OLD FLOW (before session #10 — keyword-dominant)

```
═══════════════════════════════════════════════════════════════
OLD ENCODING — store and forget
Files: brain_remember.py → remember()
Models: Arctic v1.5 (embedding), BART zero-shot (keywords)
Tables: nodes, node_embeddings, node_vectors, doc_freq, edges
═══════════════════════════════════════════════════════════════

Claude calls: remember(type="decision", title="Separate API + Web architecture",
                       content="Glo API REST is single source of truth...")
        │
        ▼
   brain_remember.py → remember()
   ├─ Generate node ID, timestamp
   ├─ Auto-set confidence from TYPE_CONFIDENCE map (decision → 0.80)
   ├─ Extract keywords via BART zero-shot (bart_keywords.py → extract())
   │   └─ "api web architecture separation rest glo"
   ├─ Generate content_summary (first ~100 chars)
   ├─ INSERT into nodes table
   ├─ Embed title+content via Arctic v1.5 (embedder.py → embed())
   │   └─ "Separate API + Web architecture Glo API REST is single source..." → 768-dim vector
   │   └─ INSERT into node_embeddings (node_id, embedding, model, created_at)
   ├─ Store TF-IDF vector (brain_remember.py → _store_tfidf_vector())
   │   └─ Tokenize → compute term frequencies → INSERT into node_vectors + doc_freq
   ├─ Create explicit connections if provided
   ├─ Auto-connect to recently accessed nodes (co_accessed, weight 0.2)
   ├─ Emergent bridging (brain_remember.py → _bridge_at_store_time())
   │   └─ Finds nodes sharing 3+ keywords → creates emergent_bridge edges
   └─ Return {id, type, title, embedding_stored}

   RESULT: 1 node with 1 embedding vector, some auto-edges
   ┌─ primary: "Separate API + Web architecture. Glo API REST is..."  [node_embeddings]
   └─ that's it. ONE vector. ONE chance to match.


═══════════════════════════════════════════════════════════════
OLD RECALL — keyword-first with embedding sprinkled on top
Files: brain_recall.py → recall() (the WRONG function — eval was testing THIS)
       brain_recall.py → recall_with_embeddings() (production, but never benchmarked)
Models: Arctic v1.5 (embedding), TF-IDF (keyword scoring)
Tables: nodes, node_embeddings, node_vectors, doc_freq, edges
═══════════════════════════════════════════════════════════════

   Hook fires: daemon_hooks.py → hook_recall(user_message)
        │
        ▼
   Vocabulary expansion on user_message (regex patterns → resolve_vocabulary())
   Enriched query passed to recall_with_embeddings()
        │
        ▼
   brain_recall.py → recall_with_embeddings(query, limit=8)

   STEP 0.5 — Vocabulary Expansion (again, inside recall)
   brain_recall.py → _expand_query_with_vocabulary(query)
   ├─ Lookup vocabulary nodes for bigrams, then single words
   └─ Append up to 3 expansion terms

   STEP 1 — Query Embedding
   embedder.py → embed(expanded_query) → 768-dim vector via Arctic v1.5

   STEP 2 — Intent Classification
   brain_recall.py → _classify_intent(query)
   ├─ Regex patterns: decision_lookup, reasoning_chain, how_to, etc.
   └─ Returns type_boosts: {node_type: multiplier}

   STEP 3 — Brute-force Cosine Scan
   SELECT ne.node_id, ne.embedding FROM node_embeddings JOIN nodes
   ├─ For each of ~700 nodes:
   │   embedder.py → cosine_similarity(query_vec, node_vec) → score
   └─ Stores in embedding_scores[node_id]

   ❌ NO STEP 3.5 — no enrichment vectors exist

   STEP 4 — Keyword Fallback (runs the ENTIRE old recall() pipeline)
   brain_recall.py → self.recall(query, limit=limit*3, _skip_log=True)
   ├─ _search_keywords() → regex match against titles/content/keywords → seeds
   ├─ TF-IDF scoring → _batch_tfidf_scores() on all activated nodes
   ├─ spread_activation() → 3-hop graph traversal from seeds
   │   └─ For EACH hop: query ALL edges for EACH activated node
   │   └─ Decays by SPREAD_DECAY (0.5) per hop
   │   └─ This is the heaviest computation — O(seeds × hops × edges)
   ├─ Combined scoring per node:
   │   blend = 0.90 × tfidf_score + 0.10 × keyword_relevance
   │   × type_boost × critical_boost × hub_dampening
   │   × Ebbinghaus retention (half-life by type, time-dilated)
   │   × emotion intensity boost
   │   × stability floor
   └─ Returns keyword_scores[node_id] + keyword_nodes[node_id]

   STEP 5 — Union Candidates
   all_candidate_ids = embedding_scores ∪ keyword_scores

   STEP 6 — Score Blending
   ┌──────────────────────────────────────────────────────────┐
   │ if emb + kw:  blended = 0.90 × emb + 0.10 × kw         │
   │ if emb only:  blended = emb                              │
   │ if kw only:   blended = 0.10 × kw  ← CRUSHED            │
   ├──────────────────────────────────────────────────────────┤
   │ × intent type boost (1.0–1.5x by node type)             │
   │ × critical boost (3.0x if locked/critical)               │
   │ × confidence multiplier (0.7–1.05 from node conf)        │
   │ × contextual penalty (0.7x if context mismatch)          │
   │ Filter: score < 0.05 → discard                           │
   └──────────────────────────────────────────────────────────┘

   ❌ NO STEP 6.5 — no graph-augmented recall

   STEP 7 — Hydrate + brain_to_host metadata
   STEP 7.5 — Tiered: top 3 full content, rest truncated to 200 chars
   STEP 8 — Mark accessed (Hebbian learning — co-access edges strengthened)
   STEP 9 — Return results + _embedding_stats
        │
        ▼
   daemon_hooks.py → precision.log_recall() → recall_log table
   brain_voice.py → wrap_for_hook() → [BRAIN] + [BRAIN-To-Tom]
   → injected into additionalContext


   THE CORE PROBLEM:

   Query: "why did we separate the backend from the frontend"
   Node:  "Separate API + Web architecture"

   Arctic embeds them independently:
   query_vec ──→ [0.12, -0.34, 0.56, ...]     ← "backend", "frontend", "separate"
   node_vec  ──→ [0.45, 0.11, -0.23, ...]     ← "API", "Web", "architecture"

   cosine_similarity = 0.58  ← BELOW THRESHOLD

   Same meaning. Different words. Single vector can't bridge the gap.
   The node is INVISIBLE to this query.
```

---

## 2. CURRENT FLOW (session #10 — V5 multi-vector enrichment)

```
═══════════════════════════════════════════════════════════════
CURRENT ENCODING — brain drives, Claude answers, 5 vectors per node
Files: brain_remember.py → remember(), _build_enrichment_prompt(), store_enrichments()
       brain_constants.py → ENRICHMENT_PROMPT_TEMPLATE
       dal.py → EnrichmentDAL, GraphDAL
Models: Arctic v1.5 (embedding), BART zero-shot (keywords)
       + Claude or Gemma 2B (enrichment generation — external to brain)
Tables: nodes, node_embeddings, node_vectors, doc_freq, edges, node_enrichments ★NEW
═══════════════════════════════════════════════════════════════

Claude calls: remember(type="decision", title="Separate API + Web architecture",
                       content="Glo API REST is single source of truth...")
        │
        ▼
   brain_remember.py → remember()
   ├─ [same as OLD: node creation, embedding, TF-IDF, auto-connect, bridging]
   │
   ├─ ★NEW: _build_enrichment_prompt(node_id, title, content)
   │   ├─ dal.py → GraphDAL.get_neighbors_with_context(node_id, limit=5)
   │   │   └─ SQL: edges both directions, ORDER BY weight DESC
   │   │   └─ Joins nodes table to get title, type, keywords, confidence
   │   │   └─ Returns: [{id, type, title, keywords, confidence, relation, weight}]
   │   │
   │   ├─ Formats V5 structured prompt (brain_constants.py → ENRICHMENT_PROMPT_TEMPLATE):
   │   │
   │   │   "The brain found these related memories:
   │   │    - Glo monolith was getting slow (decision, keywords: monolith, performance)
   │   │    - React + Python stack choice (decision, keywords: react, python, frontend)
   │   │    - Frontend team blocked on API deploys (constraint, keywords: deploy, block)
   │   │
   │   │    New node: "Separate API + Web architecture"
   │   │    Content: "Glo API REST is single source of truth..."
   │   │
   │   │    Generate exactly these lines, no explanations:
   │   │    Q: [one question a user would naturally ask that leads to this node]
   │   │    A: [3-5 word phrase using words from the neighbors above]
   │   │    B: [one sentence connecting this node to its most important neighbor]
   │   │    K: [5 comma-separated keywords borrowed from neighbors]"
   │   │
   │   └─ Returns enrichment_prompt in remember() response
   │
   └─ Return {id, enrichment_prompt: "The brain found these...", ...}
        │
        ▼
   Claude reads the enrichment_prompt, fills it in:
     Q: why did we split the backend from the frontend
     A: monolith separation API deploy
     B: This architecture solved the monolith scaling problem that was blocking deploys
     K: backend, frontend, monolith, scaling, deploy
        │
        ▼
   Claude calls: store_enrichments(node_id, question=..., anchor=..., bridge=..., keywords=...)
        │
        ▼
   brain_remember.py → store_enrichments(node_id, Q, A, B, K)
   ├─ For each non-None text:
   │   ├─ embedder.py → embed(text) → 768-dim vector via Arctic v1.5
   │   └─ dal.py → EnrichmentDAL.store(node_id, vector_type, text, embedding, model)
   │       └─ INSERT into node_enrichments (id, node_id, vector_type, text, embedding, model)
   └─ Returns: {enrichments_stored: 4, errors: []}

   RESULT: node now has 5 searchable vectors:
   ┌─ primary:  "Separate API + Web architecture. Glo API REST is..."     [node_embeddings]
   ├─ question: "why did we split the backend from the frontend"           [node_enrichments]
   ├─ anchor:   "monolith separation API deploy"                           [node_enrichments]
   ├─ bridge:   "This architecture solved the monolith scaling problem..." [node_enrichments]
   └─ keywords: "backend frontend monolith scaling deploy"                 [node_enrichments]


═══════════════════════════════════════════════════════════════
CURRENT RECALL — embeddings-first with enrichment scan + graph augmentation
Files: brain_recall.py → recall_with_embeddings()
       dal.py → EnrichmentDAL, GraphDAL
       brain_constants.py (all tuning knobs)
       daemon_hooks.py → hook_recall()
       brain_voice.py → wrap_for_hook()
Models: Arctic v1.5 (query embedding + cosine scan)
Tables: nodes, node_embeddings, node_enrichments ★NEW, node_vectors, edges
═══════════════════════════════════════════════════════════════

   Hook fires: daemon_hooks.py → hook_recall(user_message)
   ├─ Vocabulary expansion on raw user_message (regex → resolve_vocabulary)
   ├─ Precision: evaluate pending followups from PREVIOUS recalls
   └─ Calls recall_with_embeddings(enriched_query, limit=8)
        │
        ▼
   brain_recall.py → recall_with_embeddings(query, limit=8)

   STEP 0.5 — Vocabulary Expansion
   _expand_query_with_vocabulary(query) — same as OLD

   STEP 1 — Query Embedding
   embedder.py → embed(expanded_query) → 768-dim vector via Arctic v1.5

   STEP 2 — Intent Classification
   _classify_intent(query) — same regex patterns → type_boosts

   STEP 3 — Primary Embedding Scan (brute-force cosine)
   SELECT ne.node_id, ne.embedding FROM node_embeddings JOIN nodes
   ├─ For each of ~700 nodes:
   │   cosine_similarity(query_vec, node_vec) → score
   ├─ embedding_scores[node_id] = similarity
   └─ enrichment_hits[node_id] = 'primary'

 ★ STEP 3.5 — Enrichment Vector Scan (NEW)
   SELECT node_id, vector_type, embedding FROM node_enrichments
   WHERE embedding IS NOT NULL
   ├─ For each of ~2,800 enrichment vectors:
   │   cosine_similarity(query_vec, enrichment_vec) → e_score
   │   if e_score > embedding_scores[node_id]:     ← MAX aggregation
   │       embedding_scores[node_id] = e_score      ← REPLACE primary score
   │       enrichment_hits[node_id] = vector_type    ← record which type won
   │
   │   EXAMPLE: query = "why split backend frontend"
   │   Node "Separate API + Web architecture":
   │   ├─ primary:  0.58  (different words — weak)
   │   ├─ question: 0.91  ← WINS ("why did we split the backend from the frontend")
   │   ├─ anchor:   0.72
   │   ├─ bridge:   0.65
   │   └─ keywords: 0.68
   │   Final: 0.91 (was 0.58 — node is now VISIBLE)
   │
   └─ Tracks enrichment_count + enrichment_used for telemetry

   STEP 4 — Keyword Fallback
   self.recall(query, limit=limit*3, _skip_log=True) — runs entire OLD pipeline
   └─ Catches nodes with NO embeddings at all

   STEP 5 — Union Candidates
   all_candidate_ids = embedding_scores ∪ keyword_scores

   STEP 6 — Score Blending — same as OLD
   ┌──────────────────────────────────────────────────────────┐
   │ if emb + kw:  0.90 × emb + 0.10 × kw                   │
   │ × intent type boost × critical boost × confidence        │
   │ × contextual penalty                                     │
   │ Filter: < 0.05 → discard                                 │
   └──────────────────────────────────────────────────────────┘

 ★ STEP 6.5 — Graph-Augmented Recall (NEW)
   dal.py → GraphDAL.get_typed_neighbors()
   ├─ Take top 5 results (GRAPH_AUGMENT_TOP_N=5)
   ├─ For each: query edges for 1-hop neighbors
   │   └─ ONLY intentional edges (21 types in INTENTIONAL_EDGE_TYPES)
   │   └─ SKIP noise edges (co_accessed, emergent_bridge)
   ├─ Neighbor score = parent_score × 0.6 × edge_weight
   ├─ Convergence: neighbor from 2+ parents → ×1.3 boost
   └─ Add to results, re-sort, re-limit

   STEP 7 — Hydrate full node data + _brain_to_host metadata
   STEP 7.5 — Tiered: top 3 full, rest truncated
   STEP 8 — Mark accessed (Hebbian)
   STEP 9 — Return results + _embedding_stats + enrichment stats
        │
        ▼
   daemon_hooks.py:
   ├─ precision.log_recall() → recall_log table (titles, snippets, embedding flag)
   ├─ Segment boundary detection (brain → check_segment_boundary)
   ├─ Aspiration/hypothesis/tension priming check
   ├─ Pending messages drain
   └─ brain_voice.py → wrap_for_hook():
       ├─ [BRAIN] — recall results, graph activity, consciousness → for Claude
       └─ [BRAIN-To-Tom] — reminders, tensions, dreams → for operator
       → injected into additionalContext


   WHAT CHANGED (benchmarked):
   ┌────────────────────┬───────────┬───────────┬─────────┐
   │ Metric             │ OLD       │ CURRENT   │ Delta   │
   ├────────────────────┼───────────┼───────────┼─────────┤
   │ NDCG@10            │ 0.183     │ 0.326     │ +78%    │
   │ MRR                │ 0.167     │ 0.326     │ +95%    │
   │ Passed             │ 27/83     │ 38/83     │ +41%    │
   │ Recall latency     │ ~100ms    │ ~150ms    │ +50ms   │
   │ Vectors per node   │ 1         │ 5         │ +4      │
   │ Storage            │ 2.5MB     │ ~10MB     │ +7.5MB  │
   └────────────────────┴───────────┴───────────┴─────────┘
```

---

## 3. PROPOSED FLOW (next — recall-before-encode + ripple)

```
═══════════════════════════════════════════════════════════════
PROPOSED ENCODING — brain recalls before encoding, ripples backward
Files: brain_remember.py → remember() (modified)
       brain_enrichment.py ★NEW — ripple engine, prompt builder
       brain_constants.py (new ripple tuning knobs)
       dal.py → EnrichmentDAL, GraphDAL (extended)
Models: Arctic v1.5 (embedding)
       Claude (enrichment + impact assessment — already in the loop)
Tables: nodes, node_embeddings, node_enrichments, edges
        brain_telemetry ★NEW (in brain_logs.db — tracks ripple events)
═══════════════════════════════════════════════════════════════

Claude calls: remember(type="lesson", title="API crashed from shared DB connections",
                       content="Production outage: API pods sharing DB pool...")
        │
        ▼
   brain_remember.py → remember()
   ├─ [same: node creation, embedding, TF-IDF, auto-connect]
   │
   ├─ ★ENHANCED: _build_enrichment_prompt(node_id, title, content)
   │   ├─ GraphDAL.get_neighbors_with_context(node_id, limit=5)
   │   │   └─ ALSO searches by embedding similarity (not just edges)
   │   │   └─ Reason: new node may not HAVE edges yet — find semantic neighbors
   │   │
   │   ├─ Returns TWO prompts:
   │   │
   │   │   ENRICHMENT PROMPT (same as current — Q/A/B/K):
   │   │   "The brain found these related memories: ...
   │   │    Generate: Q: ... A: ... B: ... K: ..."
   │   │
   │   │   ★NEW — IMPACT ASSESSMENT PROMPT:
   │   │   "This new memory just arrived:
   │   │    Title: API crashed from shared DB connections
   │   │    Content: Production outage: API pods sharing DB pool...
   │   │
   │   │    These existing memories are related:
   │   │    1. [id:abc] Separate API + Web architecture (decision, conf 0.80)
   │   │    2. [id:def] PostgreSQL connection pooling (mechanism, conf 0.70)
   │   │    3. [id:ghi] Monolith was fine for v1 (decision, conf 0.60)
   │   │
   │   │    For each related memory, answer:
   │   │    [id:abc] VALIDATES | CONTRADICTS | EXTENDS | UNCHANGED? confidence: UP | DOWN | SAME?
   │   │    [id:def] VALIDATES | CONTRADICTS | EXTENDS | UNCHANGED?
   │   │    [id:ghi] VALIDATES | CONTRADICTS | EXTENDS | UNCHANGED?"
   │   │
   │   └─ Returns {enrichment_prompt, impact_prompt}
   │
   └─ Return {id, enrichment_prompt, impact_prompt, ...}
        │
        ▼
   Claude fills in BOTH prompts:

   Enrichment:
     Q: what caused the API crash
     A: shared database connection outage
     B: The DB pool sharing is what made the API/Web separation necessary
     K: database, connection, pool, crash, outage

   Impact:
     [id:abc] VALIDATES — the crash proves the separation was the right call. conf: UP
     [id:def] VALIDATES — connection pooling was the mechanism that failed. conf: UP
     [id:ghi] CONTRADICTS — the monolith SHARED the DB, which caused THIS crash. conf: DOWN
        │
        ▼
   Claude calls: store_enrichments(node_id, Q, A, B, K)  ← same as current
   Claude calls: apply_ripple(node_id, impacts=[...])     ← ★NEW
        │
        ▼
 ★ brain_enrichment.py → apply_ripple(node_id, impacts)
   For each impacted neighbor:
   │
   ├─ [id:abc] VALIDATES, conf UP:
   │   ├─ brain.connect(new_node, abc, "validates", weight=0.8)
   │   ├─ Confidence: 0.80 → 0.85 (+0.05, capped at 1.0)
   │   ├─ ★RE-ENRICH: build new enrichment prompt for abc with NEW neighbor context
   │   │   └─ abc now has a new neighbor (the crash) — its Q/A/B/K should reflect this
   │   │   └─ New question might be: "what proved the API separation was right"
   │   │   └─ Re-embed enrichment vectors for abc
   │   └─ Log to brain_telemetry: {event: 'ripple', source: new_node, target: abc,
   │       action: 'validates', conf_delta: +0.05, re_enriched: true}
   │
   ├─ [id:def] VALIDATES, conf UP:
   │   ├─ brain.connect(new_node, def, "validates", weight=0.7)
   │   ├─ Confidence: 0.70 → 0.75
   │   ├─ RE-ENRICH def
   │   └─ Log to brain_telemetry
   │
   └─ [id:ghi] CONTRADICTS, conf DOWN:
       ├─ brain.connect(new_node, ghi, "contradicts", weight=0.8)
       ├─ Confidence: 0.60 → 0.50 (-0.10)
       ├─ RE-ENRICH ghi — its new enrichment might include:
       │   Q: "was the monolith actually fine"
       │   B: "Contradicted by the shared DB crash — monolith's DB sharing was the root cause"
       ├─ If confidence < 0.3 → flag for operator review (consciousness signal)
       └─ Log to brain_telemetry

   RESULT after encoding "API crashed from shared DB connections":
   ├─ 1 new node with 5 vectors (same as current)
   ├─ 3 edges created (validates ×2, contradicts ×1)
   ├─ 3 existing nodes updated:
   │   ├─ abc: conf 0.80→0.85, re-enriched with crash context
   │   ├─ def: conf 0.70→0.75, re-enriched
   │   └─ ghi: conf 0.60→0.50, re-enriched with contradiction
   ├─ 12 enrichment vectors updated (4 per re-enriched node)
   └─ Telemetry: 3 ripple events logged

   The old nodes are now findable from NEW angles they couldn't be found from before.
   "what proved API separation was right" → matches abc's new question enrichment.


═══════════════════════════════════════════════════════════════
PROPOSED RECALL — same pipeline, but enrichments keep improving
Files: brain_recall.py → recall_with_embeddings() (unchanged)
═══════════════════════════════════════════════════════════════

   The recall pipeline DOES NOT CHANGE.

   Steps 0.5 through 9 stay exactly as they are.
   The improvement comes from the encoding side:

   ├─ More enrichment vectors exist (re-enriched nodes have fresher Q/A/B/K)
   ├─ Better edges (typed: validates, contradicts, extends — not just co_accessed)
   ├─ Confidence scores reflect reality (validated nodes rank higher, contradicted lower)
   └─ Graph augmentation (STEP 6.5) traverses richer, more intentional edges

   The recall pipeline is the READ side. All investment goes into the WRITE side.
   Better data in → better results out. No recall code changes needed.


═══════════════════════════════════════════════════════════════
PROPOSED TELEMETRY — end-to-end visibility
Files: brain_telemetry.py ★NEW
       dal.py → TelemetryDAL
Table: brain_telemetry (in brain_logs.db)
═══════════════════════════════════════════════════════════════

   Every event logged with:
   {event_type, source_id, target_id, action, metadata_json, created_at}

   Event types:
   ├─ 'encode'     — new node created (with enrichment count)
   ├─ 'enrich'     — enrichments stored (Q/A/B/K, which types succeeded)
   ├─ 'ripple'     — impact propagated (validates/contradicts/extends, conf delta)
   ├─ 'recall'     — query served (enrichment_used count, source breakdown)
   ├─ 'precision'  — recall evaluated (useful/not_useful/ask_operator)
   └─ 'error'      — any pipeline failure (with traceback)

   Dashboard query:
   "How many recalls used enrichments this week?"
   SELECT COUNT(*) FROM brain_telemetry
   WHERE event_type = 'recall'
   AND json_extract(metadata, '$.enrichment_used') > 0
   AND created_at > datetime('now', '-7 days')

   "Which nodes were most rippled?"
   SELECT target_id, COUNT(*) as ripple_count
   FROM brain_telemetry WHERE event_type = 'ripple'
   GROUP BY target_id ORDER BY ripple_count DESC

   Nothing silently fails. Every path is instrumented.
```

---

## Summary Table

| | OLD | CURRENT | PROPOSED |
|---|---|---|---|
| **Vectors per node** | 1 | 5 | 5 + re-enriched over time |
| **Encoding awareness** | None — store and forget | Knows neighbors, generates Q/A/B/K | Recalls before encoding, assesses impact, ripples backward |
| **Old nodes improve?** | Never | Never | Yes — re-enriched when new info arrives |
| **Confidence moves?** | Set once, never changes | Set once, never changes | Validated up, contradicted down |
| **Edge quality** | co_accessed, emergent_bridge (noise) | + intentional from SKILL.md | + validates, contradicts, extends (from impact assessment) |
| **Recall changes** | — | +STEP 3.5, +STEP 6.5 | None — all improvement on write side |
| **Telemetry** | recall_log only | + enrichment stats | Full pipeline: encode→enrich→ripple→recall→precision |

---

## Benchmark Data (12 conditions tested this session)

| Condition | NDCG@10 | MRR | Passed | Latency/query | Verdict |
|---|---|---|---|---|---|
| v1.5 control (keyword-only eval bug) | 0.204 | 0.202 | 34/104 | 106ms | Bug: tested wrong function |
| v1.5 control (fixed eval) | 0.183 | 0.167 | 27/83 | ~100ms | True baseline |
| Arctic v2.0 large | 0.198 | 0.197 | 33/104 | ~100ms | Regression — don't switch |
| HyDE + TinyLlama 1.1B | 0.204 | 0.207 | 34/104 | 148ms | No effect — hallucinated |
| HyDE + Gemma 2B | 0.204 | 0.207 | 34/104 | 129ms | No effect — same problem |
| v1.5 + MiniLM reranker | 0.232 | 0.241 | 35/104 | 131ms | Small gain |
| v1.5 + bge-v2-m3 reranker | 0.494 | 0.514 | 61/104 | 4278ms | Too slow |
| v1.5 + gte-modernbert reranker | 0.518 | 0.533 | 61/104 | 2168ms | Too slow |
| v2.0 + gte-modernbert reranker | 0.522 | 0.533 | 62/104 | 2120ms | Too slow |
| **V5 multi-vector (current)** | **0.326** | **0.326** | **38/83** | **~150ms** | **Shipped** |
| V5 benchmark-only (target nodes) | 0.701 | 0.704 | 91/104 | ~150ms | Ceiling (not production) |
