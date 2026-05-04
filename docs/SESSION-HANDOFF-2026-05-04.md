# Session handoff — 2026-05-04

Continuation of the Frame Phase 2.5 punch list (`1f68e549`) plus a major
architectural refactor: **the families system unified into aspects**.

Three threads landed: punch list polish, pre/post-compact hook deletion,
and the unified aspects architecture (13-step migration). Read these in
order when resuming.

---

## Shipped this session (in production)

### Pre-aspects polish

| | What | Commit |
|---|---|---|
| 1 | CLAUDE.md update — Frame Phase 2.5 reflection, deTomify, voice consistency with SKILL.md, 451 → 308 lines | `411daf8` |
| 2 | brain_batch description enrichment — sharpened scopes (SIBLINGS vs CATALOG), forward-reference example, empty-`why` anti-pattern, `relations: array` shorthand. Affects all 4 encoders that import brain_mcp.TOOLS | `a8ff65e` |
| 3 | Pre/post-compact hooks deleted — 372-line cleanup. Compaction is now invisible to the brain; next UserPromptSubmit fires hook_recall, Frame is rebuilt from scratch. Drops `s2_node_families` legacy fallback path along the way | `6434125` |

### Unified aspects architecture (13 commits, ~2,470 lines net)

The biggest piece. Replaces two parallel "families" systems
(`s2_node_families` + `s2_edge_families` interactions, `EdgeFamilyIntegration`,
`_FALLBACK_FAMILIES` dict, scattered name-routing in 5 files) with a unified
**aspects** system: aspects-as-nodes (`type='aspect'`), one registry,
one seed JSON, one contract test, one source of truth.

| Step | Commit | What |
|---|---|---|
| 1 | `ff22222` | `brain._log_warning()` infra — non-blocking signal logging |
| 2 | `304f13e` | `Aspect` value object + `AspectRegistry` skeleton (`servers/aspects.py`) |
| 3 | `b4b814e` | `aspects_v1.json` — unified seed for the 14 required aspects |
| 4 | `8d51e8c` | Contract test + caught/fixed `follows` collision (was in 2 aspects) |
| 5a | `8bd5a07` | `MetadataDAL` JSON-encodes list/dict values for clean round-trip |
| 5b | `7e9950a` | Aspect migration module + daemon dispatch + CLI script |
| 6 | `568ebd9` | Wire `AspectRegistry` into `Brain.__init__` — eager validation + auto-heal |
| 7 | `33c2a26` | Migrate `surface_contract.py` to brain.aspects |
| 8 | `7c6bc06` | Migrate consolidation encoder + decoder |
| 9 | `2f823ad` | Migrate `community_decoder.py` |
| 10 | `c78b075` | Drop dead `HEALER_EDGE_FAMILIES` |
| 11 | `cb7bd76` | Migrate `frame.py` — drop families helpers entirely |
| 12 | `6b7c753` | Hard-remove legacy helpers + interaction seed |
| 13 | `cd36efc` | Retrace cleanup — drop `edge_families.py` + dead constants + stale comments |

**Net vs session start:**

- 1 new architectural primitive: `brain.aspects` (AspectRegistry on every Brain)
- 14 required aspects, locked, single source of truth via `aspects_v1.json` ≡ `REQUIRED_ASPECTS`
- 5 consumers migrated (frame, surface, community_decoder, consolidation_encoder/decoder, eval scripts)
- 4 dead modules / dicts removed (edge_families.py, HEALER_EDGE_FAMILIES, EDGE_TYPE_GROUPS, RELATION_TO_GROUP)
- 6 stale comment groups updated to point at brain.aspects
- ~800 lines of dead code deleted; ~2,470 net lines added (most in new tests + registry)
- Pre/post-compact hooks: 4 files deleted, 7 docs/configs updated (372 lines removed)
- 79 new aspect-system tests; 0 regressions across 892 tests

**Tom's brain state post-migration:**

- `./dev python3 scripts/migrate_to_aspects.py` ran cleanly
- 68 aspect-nodes total: 14 required (locked, `anchor:seed_aspects`) + 54 emergent (unlocked, `migration:aspects_emergent`)
- The 54 emergent are the rich taxonomy `EdgeFamilyIntegration` accumulated before being disabled
- Backups: `brain.db.bak-20260503_222400` (script's) + `brain.db.pre-aspects-migration-20260503_222331` (Anchor's)

---

## Architecture, distilled

### Aspect = semantic role of types and/or relations

An aspect is a named semantic role grouping node TYPES and/or edge RELATIONS
under a shared meaning. Some aspects have only `node_types` (`identity_bearing`,
`active_thread`), some only `edge_relations` (`generic_relation`, `noise`),
some both (`correction_improvement` — the unification case: types
`correction`, `bug_lesson` + relations `corrects`, `supersedes`).

### Aspects are nodes (Tom's call)

`type='aspect'` brain nodes. Member lists in `node_metadata_kv` as
JSON-encoded lists (`node_types`, `edge_relations`). Standard encoder fields
(title, content, situation, etc.). Treated like any other node — recall,
revise, lock, archive flow through the standard write path. No special
storage, no parallel system.

This is the same fractal pattern as communities: nodes-about-nodes. Communities
emerge from co-occurrence; aspects assert by classification. Both are abstractions
over individual nodes, both are first-class memories Anchor can recall.

### Two tiers

| Tier | Locked | How seeded | Mutation |
|---|---|---|---|
| **Required** (14) | True | `aspects_v1.json` via `seed_required_aspects` (`anchor:seed_aspects`) | Members can drift; aspect itself can't be archived or renamed (locked + REQUIRED contract) |
| **Emergent** (any number) | False | AspectIntegration discovers from observed types/relations | Free creation/revision by S2; can be locked by anchor or operator deliberately |

### The 14 required aspects

```
identity_bearing       episodic_anchor       active_thread          lesson_insight
correction_improvement extension_refinement  explanation_causation  dependency_flow
contradiction_conflict validation_evidence   hierarchical_structure temporal_sequence
generic_relation       noise
```

These are the names code routes on by string. Test asserts `set(REQUIRED_ASPECTS) ≡ set(aspects_v1.json keys)` — adding a name to one without the other fails.

### One API — `brain.aspects`

```python
# Per-aspect access (attribute style — required raises AspectContractError if missing)
brain.aspects.identity_bearing.node_types
brain.aspects.correction_improvement.edge_relations
brain.aspects.by_name('emergent_xyz')             # Optional[Aspect]

# Reverse lookups
brain.aspects.by_node_type('principle')           # Aspect or None
brain.aspects.by_edge_relation('corrects')        # Aspect or None

# Cross-aspect unions
brain.aspects.types_in(['episodic_anchor', 'lesson_insight'])
brain.aspects.relations_in(['noise', 'generic_relation'])

# Discovery + enumeration
brain.aspects.all()                               # dict[name, Aspect]
brain.aspects.all_with_counts()                   # for list_aspects MCP
brain.aspects.required() / emergent()             # tier filters
brain.aspects.by_dimension('semantic')
brain.aspects.dimensions()                        # set of dimensions present

# Surface-specific (edge enrichment for embeddings)
brain.aspects.relation_meaning_map()              # {relation_str: meaning_text}
brain.aspects.type_meaning_map()                  # {type_str: meaning_text}

# Construction for tests/seeding
AspectRegistry.from_dict(brain, data)             # bypasses _load
```

### Bootstrap path on a fresh brain

```
Brain.__init__(db_path)
  → ensure_schema
  → seed_interactions               (no longer seeds s2_*_families)
  → seed_baby_brain                 (if not skip_embedder)
  → AspectRegistry(self)
       _load: filter_nodes(type='aspect') → empty
       _validate: 14 missing
       _log_warning('aspect_contract', ..., 'auto-healing from seed')
       _auto_heal: seed_required_aspects(brain)
                   → reads aspects_v1.json
                   → 14 brain.remember(type='aspect', locked=True,
                       encoding_source='anchor:seed_aspects', node_types=[...],
                       edge_relations=[...], dimension='semantic',
                       display_label=...) calls
       _load: 14 present, registry populated
```

### Migration path on existing brains (Tom's, etc.)

`./dev python3 scripts/migrate_to_aspects.py`:

```
1. Save (flush WAL)
2. cp brain.db brain.db.bak-{timestamp}
3. daemon dispatch 'migrate_to_aspects'
   → seed_required_aspects (idempotent, skips already-present from auto-heal)
   → migrate_emergent_from_legacy:
       reads s2_node_families + s2_edge_families interactions
       collapses concept overlap (e.g. correction_supersession → correction_improvement)
       creates aspect-nodes for non-required names
       encoding_source='migration:aspects_emergent', locked=False
4. Print result summary
```

The legacy interactions remain readable until manually archived; nothing
reads them after this commit (they're history, not source of truth).

---

## What's still left

### Aspects punch list — 2 of 14 open

| Step | Status | What |
|---|---|---|
| **13** | open | Build `AspectIntegration` (the unified Sonnet maintenance unit replacing the disabled `EdgeFamilyIntegration`) |
| **14** | open | MCP surface — `list_aspects` tool, `filter_nodes(field='aspect')` virtual field, `recall(filter={'aspect':...})` filter |

### Step 13 — AspectIntegration

Designed in Section 5 of the planning conversation. Three files following
the standard S2 pattern:

- `aspect_decoder.py` — scans `nodes.type DISTINCT` + `edge_relations.relation DISTINCT` for unclassified types/relations. Gates on `_has_new_traces()`. Returns proposals dict if work to do; empty if nothing → orchestrator no-ops, encoder never invoked. Same suppression discipline as edge_families had.
- `aspect_encoder.py` — single Sonnet pass classifies BOTH new types AND new relations into the unified aspect taxonomy. Required-aspect guards: never archive locked, never rename required, can revise members within unlocked. Writes via standard `brain_batch` (revise existing, remember new emergent). Trace chain `s2-{YYYYMMDD}-aspect_integration`.
- `aspect_integration.py` — thin orchestrator subclassing the decoder.

Plus:
- Register `s2_aspects` interaction prompt in `interaction_seed.py`
- Add to `coordinator.py` units list
- Add to `tests/test_trace_contract_sync.py::TRACE_WRITER_FILES`
- Tests via IsolatedBrain — proposes correctly, doesn't violate required contract, idempotent

### Step 14 — MCP surface

- `list_aspects(scope='all'|'required'|'emergent', dimension='all'|<name>)` — new tool. Returns counts + previews + dimension catalog. Live (reflects new aspects as they're created). Dimension-aware from day one.
- `filter_nodes` extended — accept virtual `aspect` field. `filter_nodes(field='aspect', include=['active_thread', 'lesson_insight'])` resolves internally to `node_types`.
- `recall` filter dict extended — accept `aspect` key. `recall(query, filter={'aspect': {'in': [...]}})` resolves to type filter.
- `filter_edges` deferred — no real use case yet.
- `get_aspect` shortcut not added — `find_node_by_title` + `get_node` already work since aspects are nodes.

### Frame Phase 2.5 punch list (4 of 14 closed entering this session, more closed now)

Status of the original punch list `1f68e549`:
- ✅ #1 Boot wire-up (closed pre-session)
- ✅ #2 Surface prompt v4 (closed pre-session)
- ✅ #12 SKILL.md identity rewrite + MCP examples (closed pre-session)
- ✅ #12 CLAUDE.md update (closed this session: `411daf8`)
- ✅ #13 session_context leak fix (closed pre-session)
- ✅ #14 brain_batch description enrichment (closed this session: `a8ff65e`)
- (NEW) Pre/post-compact hooks deleted entirely (closed this session: `6434125`)
- (NEW) Aspects unification — replaces #8/#9 from punch list (closed this session: 13 commits)
- ⏳ #3 Wire Frame into encoder (gap-aware encoding)
- ⏳ #4B + #5 Dashboard Frame view + operator boot summary
- ⏳ #6/#7 Cadence + caching split
- ⏳ #10 Encoder session_context format cleanup
- ⏳ #11 Fresh-Claude vs Anchor calibration test

### Recall thread — designed, not built

The deeper recall work from RECALL-OVERVIEW.md §3 still applies — connection
scoring (Step 3.5), agentic Haiku-first recall (7-tool surface), multi-anchor
query decomposition, hybrid retrieval. Aspects are foundational for
multi-dimensional recall in that thread (temporal aspects, provenance
aspects, etc. — the dimension axis is exposed but only `semantic` exists today).

### Pre-existing test failures (not from this session)

8 baseline failures remained throughout — none introduced by this session:
- `test_skill_md_has_anchor_identity` — broken since SKILL.md rewrite (f4bcfd5)
- `test_s1_data_assembly::TestSaveSessionContext` (5 tests)
- `test_recall_quality::test_hub_dampening` (flaky, suite-ordering)
- `test_write_lock_unification::test_concurrent_first_touch_same_id_yields_one_row`

Worth a triage pass when convenient — none aspect-related.

### One thing Tom flagged

> "S2 has tendency to spin process if stopped in the middle"

Captured for later investigation; not addressed this session. Worth
diagnosing separately — could be coordinator + IntegrationUnit lifecycle.

---

## Files / commits to read first when resuming

**Code:**
- `servers/aspects.py` — Registry + Aspect value object
- `servers/aspect_migration.py` — seed + migrate logic
- `servers/scales/s2/aspects_v1.json` — the 14 required aspects with members + meanings
- `servers/scales/s1/frame.py` — cleanest example of consumer migration
- `servers/scales/s2/coordinator.py` — note EdgeFamilyIntegration removed from units list

**Tests:**
- `tests/test_aspects.py` — value object + registry mechanics
- `tests/test_aspects_contract.py` — contract enforcement (REQUIRED ≡ JSON keys, no overlap)
- `tests/test_aspect_registry_wired.py` — auto-heal + Brain integration
- `tests/test_aspect_migration.py` — seed + emergent migration

**Docs (after this session's updates):**
- `CLAUDE.md` — developer guide with aspects section
- `docs/RECALL-OVERVIEW.md` — recall map (aspect terminology)
- `docs/S2-DESIGN.md` — EdgeFamilyIntegration disabled note
- `docs/SESSION-HANDOFF-2026-05-04.md` — this file

**Brain memory worth surfacing:**
- `f129be9a` — "Unified families design: one system with node_types + edge_relations slots per family"
- `0d00a4ee` — "Rename families → aspects across the entire plan"
- `c07eb6e4` — "Required families set: 14 entries locked, seeded from families_v1.json"
- `b42e11d0` — "'follows' collision: present in both hierarchical_structure and temporal_sequence"
- `8c0744b6` — "Delete pre/post compact hooks rather than fix them"
