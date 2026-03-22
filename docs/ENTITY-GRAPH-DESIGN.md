# Entity-Aware Knowledge Graph — Design Document

**Status:** Planned (not yet implemented)
**Created:** 2026-03-21
**Brain nodes:** 6cd874, c0b404, d5d89d, 7713d6 (all locked)

---

## Problem

The brain is a bag of nodes with embedding search. Every node is flat — a title, content, keywords, a type label. Edges exist but are generic weighted connections ("relates_to"). Recall finds nodes by embedding similarity, not by understanding relationships.

This means:
- "Valinor's pitcher" is stored as text, not as a product entity owned by Valinor
- Mentioning "Valinor" doesn't surface its products, people, or decisions
- "The recall screen" and "recall" (the brain mechanism) look the same to embeddings
- System components (daemon, hooks, scorer) aren't tracked as architectural entities

## Vision

Upgrade from document retrieval to a **knowledge graph with typed relationships and graph traversal in recall**.

When you say "Valinor," the brain should:
1. **Recognize** — "Valinor" is a known entity (company)
2. **Traverse** — Valinor → has_product → pitcher, Valinor → friend_of → Tom
3. **Activate** — everything connected lights up, weighted by relevance
4. **Disambiguate** — "pitcher" in Valinor context = product, not baseball

---

## Three Extraction Layers

### Layer 1 — Vocabulary (exists today)

**What:** Term → meaning mappings, context-dependent.
**Examples:** "DAL" = data access layer. "Magic links" = passwordless auth. "Supply adapter" = our abstraction pattern.
**Trigger:** Unknown terms, jargon, abbreviations, project-specific language.
**Storage:** `type='vocabulary'` nodes with definitions and context.

### Layer 2 — Entities (to build)

**What:** Things with identity that persist across conversations.

Tom's entity model is **broader than NLP's named entities**. An entity is anything that has identity:

| Category | Examples |
|----------|----------|
| People | Tom, collaborators, friends |
| Products | Valinor's pitcher, Glo |
| Companies | Valinor, Clerk |
| System components | The daemon, recall scorer, precision loop |
| Architecture | Screens, pages, API endpoints, database tables |
| Concepts-as-things | "The supply adapter pattern", "the hook chain" |

**Key distinction from vocabulary:** Vocabulary maps terms to meanings. Entities ARE things — they have relationships, they change over time, they connect to other entities.

**Same word, different layer:** "recall" as vocabulary = the brain's retrieval mechanism. "The recall screen" as entity = a specific UI component in a specific project.

**Key challenge:** Entity resolution — "Valinor's pitcher" in prompt 1, "the pitcher" in prompt 5, "VP" in prompt 10 must all link to the same entity node.

### Layer 3 — Relationships (to build)

**What:** Typed connections between entities.

**Relationship types (initial set):**

| Type | Meaning | Example |
|------|---------|---------|
| `has_product` | Ownership | Valinor → pitcher |
| `is_a` | Classification | Clerk → auth provider |
| `built_with` | Technology | Glo → Next.js |
| `decided_by` | Decision link | supply adapter → decision node |
| `depends_on` | Dependency | login screen → Clerk |
| `contains` | Composition | dashboard → login screen |
| `implements` | Pattern usage | supply adapter → adapter pattern |
| `friend_of` | Personal | Tom → Valinor person |
| `part_of` | Membership | recall scorer → brain |

**Classification approach:** The LLM classifies relationship type from conversation context. This is free — Claude is already in the conversation. No extra API call needed; the hook extracts it from what was said.

### How layers strengthen each other

- Vocabulary helps entity extraction: knowing "DAL" is a term prevents false entity detection
- Entities give vocabulary context: which project's "DAL"?
- Relationships make recall associative: mention one entity, surface connected ones

---

## What "Encoding" Really Means

Tom's definition of encoding is the **entire capture pipeline**, not just `brain.remember()`:

- Explicit `remember()` calls (guided by Tom or Claude)
- Vocabulary detection and storage
- Entity recognition and linking
- Connection/edge creation between nodes
- Precision evaluation (did recall help?)
- Session synthesis
- Dream connections found during idle

**Tom's encoding patterns** — he encodes more when:
- A decision is made that will matter across sessions
- A correction happens (wrong assumption exposed)
- A new concept or entity enters the conversation
- Something is learned about how the system works

The brain should recognize these patterns and encode proactively.

---

## Current State (v5.4)

What already exists that supports this:

| Component | State | Notes |
|-----------|-------|-------|
| Node types | ✅ Exist | `person`, `project`, `decision`, `concept`, etc. |
| Edge table | ✅ Exists | `(source_id, target_id, relation, weight)` but `relation` is free-text |
| Edge-neighbor discovery | ✅ In suggest() | 1-hop traversal for locked nodes — partial graph walk |
| Vocabulary nodes | ✅ Exist | But conflated with entities |
| Entity extraction | ❌ Missing | Only regex-based vocab detection |
| Entity resolution | ❌ Missing | No coreference or mention linking |
| Typed edges | ❌ Missing | `relation` field exists but unused meaningfully |
| Graph-augmented recall | ❌ Missing | Recall is pure embedding similarity |

---

## Risks

### Dependency risk (spaCy)
- 15MB `en_core_web_sm` model, Cython compiled, platform-specific
- Google Drive sync + macOS + ARM = potential breakage
- If model fails to load → entity extraction dies
- Mitigation: graceful fallback to regex extraction

### Latency risk
- spaCy `nlp()` ~8ms per message, 200ms cold start
- Daemon amortizes cold start, but direct fallback path pays every time
- Hook chain already tight: recall ~50ms + extraction + vocab + precision
- Mitigation: daemon keeps model loaded; budget 8ms is acceptable

### Entity resolution risk
- Coreference resolution is unsolved in NLP
- "Valinor's pitcher" → "the pitcher" → "VP" — same entity?
- Wrong resolution = duplicate nodes or corrupted merges
- Mitigation: conservative resolution (exact match + explicit aliases), not aggressive inference

### Complexity risk
- Who decides relationship types? LLM? Classifier? Heuristics?
- Maintaining a relationship ontology is ongoing work
- Brain's healing/consolidation must understand typed edges
- Mitigation: start with small type set, let it grow organically

### Data model risk
- Existing edges are untyped — migration creates two-class system
- If type ontology is wrong, typed edges are harder to fix than untyped
- Mitigation: existing edges keep generic type; only new edges get typed

---

## Implementation Plan

### Phase 1: Better Extraction (zero new dependencies)
- [x] Identifier splitting: `re.findall` replaces fragile two-pass `re.sub`
- [ ] Sentence splitting: pySBD (MIT, pure Python, 97.9% accuracy)
- [ ] Vocabulary regex: improved patterns (capitalized terms, backticks, acronyms, verb-object, multi-word NPs)
- [ ] Common-word filter: 10K word list to detect domain-specific terms

### Phase 2: Entity Extraction (one dependency: spaCy)
- [ ] Install `spacy` + `en_core_web_sm` (15MB)
- [ ] noun_chunks extraction on every user message
- [ ] Entity node creation with proper types
- [ ] Basic entity resolution (exact match + stored aliases)
- [ ] Fallback to regex if spaCy unavailable

### Phase 3: Typed Edges + Graph Traversal
- [ ] Define initial relationship type set
- [ ] LLM-based relationship classification (from conversation context)
- [ ] Migrate `connect()` to require relationship type
- [ ] Extend suggest() edge-neighbor discovery to use types
- [ ] Graph-augmented recall: 1-hop traversal on embedding matches

### Phase 4: Full Integration
- [ ] Entity-aware consciousness signals ("Valinor mentioned but no recent updates")
- [ ] Entity timeline ("when was Valinor last discussed?")
- [ ] Relationship-based dreaming ("entity X connects to entity Y through Z")
- [ ] Consolidation understands entities (don't merge different entities)

---

## Open Questions

1. **Should entity nodes be a new type or a flag?** New `type='entity'` vs `entity=True` flag on existing types (person, project are already entity-like).

2. **How many relationship types?** Start small (5-6) and grow? Or comprehensive from day one?

3. **Where does classification happen?** In the hook (synchronous, needs to be fast) or async during idle?

4. **How to handle entity death?** Products shut down, people leave. Archive vs delete?

5. **Graph query language?** Do we need structured queries ("find all entities of type product connected to Valinor") or is traversal-augmented embedding search enough?
