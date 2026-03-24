# Encoding/Decoding Pipeline v2

**Date:** 2026-03-23 (updated 2026-03-24)
**Git version:** 6509d54 (main) → efaaf5c (Session #9 decode pipeline)
**Author:** Claude Opus 4.6 + Tom
**Sessions:** #10-11 — Embedding Migration to LLM, #9 — Identity-First Encoding + Decode Pipeline

---

> **Session #9 Update (2026-03-24):** The encoding problem is solved. Identity + examples + live brain access produces 100%±0% aha capture. Instructions/checklists are killed — they produce compliance, not judgment. The decode side now has a 50-query funnel (51% top-3 after recency boost). See `docs/session-9-handoff.md` for full results.

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

## 3. PROPOSED FLOW — REVISED after benchmarking (ripple KILLED, cues + V6 encoding)

```
⚠️  THE FULL RIPPLE ENGINE WAS DESIGNED, TESTED, AND KILLED IN THIS SESSION.
    15+ benchmark conditions proved it net-negative (-0.0016 NDCG).
    What follows is the REVISED plan based on data.
    See "What We Tested and Killed" section below for the full story.


═══════════════════════════════════════════════════════════════
PROPOSED ENCODING — V6 prompt (N/R/W/C/D fields) + cues-as-edges
Files: brain_remember.py → remember(), _build_enrichment_prompt_v6(), store_enrichments()
       brain_constants.py → ENRICHMENT_PROMPT_TEMPLATE_V6
       dal.py → EnrichmentDAL (new vector types), GraphDAL
Models: Arctic v1.5 (embedding)
       Claude (enrichment + impact cues — already in the loop)
Tables: nodes, node_embeddings, node_enrichments, edges
═══════════════════════════════════════════════════════════════

Claude calls: remember(type="lesson", title="API crashed from shared DB connections",
                       content="Production outage: API pods sharing DB pool...")
        │
        ▼
   brain_remember.py → remember()
   ├─ [same: node creation, embedding, TF-IDF, auto-connect]
   │
   ├─ _build_enrichment_prompt_v6(node_id, title, content)
   │   ├─ GraphDAL.get_neighbors_with_context(node_id, limit=5)
   │   │   └─ ALSO searches by embedding similarity (not just edges)
   │   │   └─ Reason: new node may not HAVE edges yet
   │   │
   │   ├─ Returns enrichment_prompt (V6 — expanded from V5):
   │   │
   │   │   "The brain found these related memories:
   │   │    - Separate API + Web architecture (decision, keywords: api, web)
   │   │    - PostgreSQL connection pooling (mechanism, keywords: pool, db)
   │   │
   │   │    New node: "API crashed from shared DB connections"
   │   │    Content: "Production outage: API pods sharing DB pool..."
   │   │
   │   │    Generate exactly these lines, no explanations:
   │   │    Q: [one question a user would naturally ask that leads to this node]
   │   │    A: [3-5 word phrase using words from the neighbors above]
   │   │    B: [one sentence connecting this node to its most important neighbor]
   │   │    K: [5 comma-separated keywords borrowed from neighbors]
   │   │    N: [what this does NOT mean — a common misunderstanding]
   │   │    R: [3 alternative ways someone might search for this, comma-separated]
   │   │    W: [what this replaces or updates, if anything]
   │   │    D: [what must also be true for this to make sense]
   │   │
   │   │    Impact on related memories (one line per neighbor):
   │   │    [Separate API + Web architecture] VALIDATES | CONTRADICTS | EXTENDS | NO_IMPACT? reason?
   │   │    [PostgreSQL connection pooling] VALIDATES | CONTRADICTS | EXTENDS | NO_IMPACT? reason?"
   │   │
   │   └─ Returns enrichment_prompt in remember() response
   │
   └─ Return {id, enrichment_prompt, ...}
        │
        ▼
   Claude fills it in:
     Q: what caused the API production outage
     A: shared database pool crash
     B: The shared DB pool is what made the API/Web separation necessary
     K: database, connection, pool, crash, outage
     N: This does NOT mean the API itself had bugs — it was the shared DB pool
     R: production outage, database connection failure, API downtime cause
     W: Disproves the assumption that shared DB was fine for our scale
     D: Depends on: having multiple API pods sharing one connection pool

     [Separate API + Web architecture] VALIDATES — crash proves the separation was right
     [PostgreSQL connection pooling] VALIDATES — pooling was the mechanism that failed
        │
        ▼
   Claude calls: store_enrichments(node_id, Q, A, B, K, N, R, W, D)
   Claude calls: store_cues(node_id, impacts=[...])
        │
        ▼
   brain_remember.py → store_enrichments(node_id, ...)
   ├─ For each non-None text (up to 8 fields):
   │   ├─ embedder.py → embed(text) → 768-dim vector via Arctic v1.5
   │   └─ EnrichmentDAL.store(node_id, vector_type, text, embedding)
   └─ Returns: {enrichments_stored: 8, errors: []}
        │
        ▼
   brain_remember.py → store_cues(node_id, impacts)
   ├─ For each impact that is NOT no_impact:
   │   └─ INSERT into edges (source_id=new, target_id=neighbor, relation='validates',
   │      weight=0.8, description='{"reason":"crash proves separation right","date":"2026-03-23"}')
   └─ NO confidence changes. NO re-enrichment. Just edges.

   RESULT: node now has up to 9 searchable vectors + impact cues as edges:
   ┌─ primary:    "API crashed from shared DB connections..."              [node_embeddings]
   ├─ question:   "what caused the API production outage"                  [node_enrichments]
   ├─ anchor:     "shared database pool crash"                             [node_enrichments]
   ├─ bridge:     "The shared DB pool is what made the API/Web..."         [node_enrichments]
   ├─ keywords:   "database connection pool crash outage"                  [node_enrichments]
   ├─ negation:   "This does NOT mean the API itself had bugs..."          [node_enrichments]
   ├─ alias:      "production outage, database connection failure, ..."    [node_enrichments]
   ├─ temporal:   "Disproves shared DB was fine for our scale"             [node_enrichments]
   └─ dependency: "multiple API pods sharing one connection pool"          [node_enrichments]

   Plus 2 cue edges:
   ├─ new_node --validates--> "Separate API + Web architecture"
   └─ new_node --validates--> "PostgreSQL connection pooling"


═══════════════════════════════════════════════════════════════
PROPOSED RECALL — same pipeline + relevance floor + cue surfacing
Files: brain_recall.py → recall_with_embeddings() (minor additions)
       brain_constants.py → RELEVANCE_FLOOR
       brain_voice.py → format cues in output
═══════════════════════════════════════════════════════════════

   Steps 0.5 through 6.5 STAY EXACTLY AS THEY ARE.
   Three additions:

 ★ STEP 6.9 — Relevance Floor (NEW — fixes context bleed)
   ├─ If max(all_blended_scores) < RELEVANCE_FLOOR → return empty results
   │   └─ "birthday" scores 0.85 on enrichment vectors but below floor → empty
   │   └─ "recall pipeline" scores 0.92 on primary embedding → passes floor
   ├─ CRITICAL: this threshold needs sweeping (0.50-0.90 in 0.05 steps)
   │   to find the value where engineering queries pass but personal queries don't
   └─ This is the P0 fix for 100% false positive rate on non-engineering queries

 ★ STEP 7 — Hydrate + Cues (ENHANCED)
   ├─ Same as current: fetch title, content, type, keywords, metadata
   ├─ NEW: GraphDAL.get_cues(node_id) → fetch validates/contradicts/extends edges
   │   └─ SQL: SELECT relation, description, source title FROM edges
   │      WHERE target_id = ? AND relation IN ('validates','contradicts','extends','supersedes')
   │      ORDER BY created_at DESC LIMIT 5
   ├─ Attach cues to result:
   │   node['cues'] = [
   │     {'type': 'validates', 'by': 'API crash from shared DB', 'reason': '...', 'date': '...'},
   │     {'type': 'contradicts', 'by': 'Merged back to monolith v3', 'reason': '...', 'date': '...'}
   │   ]
   └─ Cost: 0.045ms per node (negligible)

 ★ brain_voice.py — Format Cues for Claude (ENHANCED)
   ├─ In [BRAIN] section, after each recalled node:
   │   "Node: Separate API + Web architecture (conf 0.80, decision)
   │    Cues:
   │    ✓ VALIDATED BY: API crash from shared DB (2026-03-23) — proves separation was right
   │    ✗ CONTRADICTED BY: Merged back to monolith v3 (2026-03-25) — reversed decision"
   │
   └─ Claude reads these cues and reasons in context.
      The BRAIN stays dumb and reliable. CLAUDE stays smart and contextual.


═══════════════════════════════════════════════════════════════
PROPOSED TELEMETRY — end-to-end visibility (unchanged from original proposal)
Files: brain_telemetry.py ★NEW
       dal.py → TelemetryDAL
Table: brain_telemetry (in brain_logs.db)
═══════════════════════════════════════════════════════════════

   Event types:
   ├─ 'encode'     — new node created (with enrichment count, vector types stored)
   ├─ 'enrich'     — enrichments stored (which of Q/A/B/K/N/R/W/D succeeded)
   ├─ 'cue'        — impact cue edge stored (validates/contradicts/extends + reason)
   ├─ 'recall'     — query served (enrichment_used, cues_surfaced, relevance_floor_hit)
   ├─ 'precision'  — recall evaluated (useful/not_useful/ask_operator)
   └─ 'error'      — any pipeline failure (with traceback)

   Nothing silently fails. Every path is instrumented.
```

---

## Summary Table

| | OLD | CURRENT (V5) | PROPOSED (V6 + cues) |
|---|---|---|---|
| **Vectors per node** | 1 | 5 | up to 9 (Q/A/B/K/N/R/W/D) |
| **Encoding awareness** | None — store and forget | Knows neighbors, generates Q/A/B/K | + negation, aliases, temporal, dependencies |
| **Old nodes improve?** | Never | Never | Via cue edges (Claude reasons, not the brain) |
| **Confidence moves?** | Set once | Set once | Set once (ripple KILLED — cues replace it) |
| **Edge quality** | co_accessed, emergent | + intentional | + validates, contradicts, extends (cues) |
| **Recall changes** | — | +STEP 3.5, +STEP 6.5 | +STEP 6.9 relevance floor, +cue surfacing |
| **Context bleed** | Not measured | 100% FP rate | Fixed by relevance floor |
| **Telemetry** | recall_log only | + enrichment stats | Full pipeline instrumented |

---

## What We Tested and Killed (15+ conditions, 8 agents)

| Condition | NDCG@10 | MRR | Passed | Verdict |
|---|---|---|---|---|
| **Baseline controls** | | | | |
| v1.5 keyword-only (eval bug) | 0.204 | 0.202 | 34/104 | Bug: tested wrong function |
| v1.5 fixed eval | 0.183 | 0.167 | 27/83 | True baseline |
| v1.5 + V5 enrichments | 0.326 | 0.326 | 38/83 | **Shipped** (+78%) |
| Golden v2 baseline | 0.304 | 0.323 | 73/148 | Current production |
| **Embedding models** | | | | |
| Arctic v2.0 large | 0.198 | 0.197 | 33/104 | ❌ Regression — don't switch |
| **HyDE (query expansion via LLM)** | | | | |
| HyDE + TinyLlama 1.1B | 0.204 | 0.207 | 34/104 | ❌ Hallucinated garbage |
| HyDE + Gemma 2B | 0.204 | 0.207 | 34/104 | ❌ Same — "Glo = online marketplace" |
| **Cross-encoder rerankers** | | | | |
| v1.5 + MiniLM 22M | 0.232 | 0.241 | 35/104 | ❌ Small gain, not worth complexity |
| v1.5 + bge-v2-m3 278M | 0.494 | 0.514 | 61/104 | ❌ +154% but 4.3s/query — too slow |
| v1.5 + gte-modernbert 149M | 0.518 | 0.533 | 61/104 | ❌ Best quality but 2.1s — too slow |
| v2.0 + gte-modernbert | 0.522 | 0.533 | 62/104 | ❌ v2.0 adds nothing on top |
| **Ripple engine** | | | | |
| Ripple only (conf + edges) | 0.304 | 0.315 | 73/148 | ❌ +0.000 — negligible |
| Ripple + re-enrichment | 0.301 | 0.315 | 71/148 | ❌ -0.003 — HARMFUL (noisy vectors) |
| Cues only (typed edges) | 0.304 | 0.323 | 73/148 | ✅ Matches baseline, zero risk |
| **Encoding improvements** | | | | |
| Extra N/R vectors only | **0.313** | **0.331** | **75/148** | ✅ **+0.010 — WINNER** |
| Everything combined | 0.310 | 0.323 | 73/148 | Ripple dilutes N/R gains |
| **Contradiction handling** | | | | |
| Gemma 2B impact assessment | — | — | 5/10 correct | ❌ 50% accuracy, EXTENDS bias |
| **Timing** | | | | |
| Full ripple with Ollama | — | — | — | ❌ 5.9s/encode — unacceptable |
| Full ripple with Claude path | — | — | — | ✅ ~185ms — acceptable |
| **Real conversations** | | | | |
| Engineering queries | 89% precision | — | — | ✅ Good |
| Non-engineering queries | 0% precision | 100% FP | — | ❌ CATASTROPHIC context bleed |

## Critical Discovery: Context Bleed

**Tested:** 85 real conversation queries across 5 simulations (engineering, topic-jumping, context bleed, emotional, segment boundaries).

**Finding:** 100% false positive rate on ALL non-engineering queries. The brain always returns engineering content regardless of what you ask.

| Query | Top Score | Top Result | Relevant? |
|---|---|---|---|
| "birthday" | 0.85 | [vocab] Dimension | ❌ |
| "my cat is sick" | 0.76 | TEST NODE DELETE ME | ❌ |
| "I'm feeling overwhelmed" | 0.87 | brain_surface.py bug | ❌ |
| "help me think through this" | 0.93 | [vocab] Add | ❌ |
| "where did I park my car" | 0.79 | [vocab] Location Service | ❌ |

**Root causes:**
1. No relevance floor — threshold is 0.05, everything passes
2. Enrichment vectors too generic — Gemma 2B anchors like "Add" and "Expand" match any English
3. Vocabulary nodes are universal matchers — single common words with broad enrichments

**This is the P0 blocker.** Must fix before any other improvement matters.
