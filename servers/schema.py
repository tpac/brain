"""
brain — Canonical Database Schema (v21)

SINGLE SOURCE OF TRUTH for every table, column, index, and constraint.

v21 changes (dead table cleanup):
  - Removed 7 dead tables from brain.db: version_history, summaries, projects,
    reasoning_chains, reasoning_steps, prune_archive, project_maps
  - Added 2 production tables: embedding_fidelity, node_communities
  - Removed 14 dead tables from brain_logs.db: access_log, recall_log, miss_log,
    tuning_log, eval_snapshots, suggest_log, curiosity_log, health_log,
    staged_learnings, message_stream, recall_gaps, pending_consolidation,
    brain_telemetry, conflict_log
  - Added hook_errors table to brain_logs.db
  - Cleaned up stale migration code in ensure_logs_schema()

HOW MIGRATION WORKS:
  1. On startup, Brain calls ensure_schema(conn)
  2. ensure_schema creates any missing tables from TABLES
  3. For each existing table, it diffs current columns against TABLES
     and ALTERs in any missing columns
  4. Creates all indexes from INDEXES

HOW TO ADD A NEW COLUMN:
  Add it to the relevant table in TABLES below. That's it.
  ensure_schema will ALTER TABLE ADD it on next startup.

HOW TO ADD A NEW NODE TYPE:
  Edit NODE_TYPES below. ensure_schema will rebuild the nodes table
  with the updated CHECK constraint (SQLite can't ALTER CHECK).

WHAT NOT TO DO:
  Do NOT add migration code in brain.py.
  Do NOT create nodes_vN rebuild tables in brain.py.
  All schema changes go HERE, in this file.
"""

import os
import shutil
import sqlite3
from datetime import datetime, timezone

BRAIN_VERSION = 30  # v30: nodes.project dropped — project is system-stamped kv provenance (node_metadata_kv['project']), not a column.
BRAIN_VERSION_KEY = 'brain_schema_version'

# Oldest brain.db this code can open. v30 shipped 2026-07-03 and every
# install in existence was created at or after it (fresh DBs are built
# directly at BRAIN_VERSION, never migrated up), so the v1..v29 upgrade
# path was unreachable code and is gone.
#
# The floor is what keeps that deletion honest: without it, a v29 DB would
# run an empty migration list and get STAMPED v30 while unmigrated —
# silently mislabelled instead of loudly refused.
MIN_SUPPORTED_VERSION = 30

# Numbered data migrations for brain.db. Empty at v30 — the historical
# ladder was deleted as unreachable. A v31 change adds (31, _migrate_v31)
# here and bumps BRAIN_VERSION; the runner does the rest.
MAIN_MIGRATIONS = []

# brain_logs.db structural version. v1 is the baseline stamp: the table
# shapes that existed when versioning was introduced. Structural changes
# from here get a numbered step in LOGS_MIGRATIONS and bump this.
LOGS_VERSION = 1
LOGS_VERSION_KEY = 'logs_schema_version'

# ─── Allowed node types ───
NODE_TYPES = [
    'person', 'project', 'decision', 'rule', 'concept',
    'task', 'file', 'context', 'intuition', 'procedure',
    'thought', 'object',
    # v4 Code Cognition types
    'fn_reasoning',      # Intent and reasoning behind a function
    'param_influence',   # Parameter with systemic effects across codebase
    'code_concept',      # Semantic unit spanning multiple files/functions (blast radius)
    'arch_constraint',   # What limits what and why
    'causal_chain',      # Regression path: trigger → propagation → failure → root cause
    'bug_lesson',        # General principle extracted from a specific bug
    'comment_anchor',    # Load-bearing comment in code that transfers knowledge
    # v4 Evolution types — forward-facing, describe what is BECOMING
    'tension',           # Contradiction between two existing nodes → must resolve
    'hypothesis',        # Untested belief with confidence score → validate or disprove
    'pattern',           # Meta-observation about recurring behavior → confirm or dismiss
    'catalyst',          # Emotional inflection point that changed direction → permanent
    'aspiration',        # Directional goal without finish line → compass for decisions
    # v4 Self-reflection types — brain looking inward
    'performance',       # Brain's own quality metrics over time (trending, not snapshot)
    'failure_mode',      # Named class of recurring failures with prevention strategy
    'capability',        # What the brain can/cannot do — self-inventory
    'interaction',       # Observed dynamics of human-Claude working relationship
    'meta_learning',     # How the brain learned something — reusable methods
    # v5 Cognitive layer — Claude's own thoughts as first-class data
    'correction',        # Self-correction trace: Claude assumed X, reality was Y, pattern Z
    'validation',        # Positive signal: this approach worked, user confirmed
    'mental_model',      # Claude's understanding of how systems/processes work
    'reasoning_trace',   # Reusable logic chain (not tied to a single decision)
    'uncertainty',       # Where Claude knows it doesn't understand something
    # v5 Engineering memory — kinds of understanding, not code elements
    'purpose',           # What something is and why it exists (system/module/file/function scope)
    'mechanism',         # How something works: flows, algorithms, interactions
    'impact',            # What changes ripple where ("update X → check Y")
    'constraint',        # What must or must not be done (replaces arch_constraint)
    'convention',        # Patterns, utilities, coding style for a codebase
    'lesson',            # What went wrong, root cause, preventive principle
    'vocabulary',        # How operator refers to things → code mapping (shared brain)
    # v6 Consciousness layer — Claude's evolving identity across sessions
    'boot',              # Shapes the boot message — written by previous session's Claude for the next
]

# v8.3: CHECK constraint removed — agents can use any type string.
# NODE_TYPES list kept for documentation and preferred-type guidance.
NODE_TYPE_CHECK = ""  # was: CHECK(type IN (...)) — removed to allow expressive typing

# ─── Canonical table definitions ───
# Each entry: { 'create': SQL, 'columns': { col: default_for_alter } }
# columns dict is used for diff-based migration (add missing columns via ALTER TABLE)
TABLES = {
    'nodes': {
        'create': f"""CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL {NODE_TYPE_CHECK},
            title TEXT NOT NULL,
            content TEXT,
            activation REAL DEFAULT 1.0,
            stability REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 1,
            locked INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            critical INTEGER DEFAULT 0,
            recency_score REAL DEFAULT 0,
            emotion REAL DEFAULT 0,
            emotion_label TEXT DEFAULT 'neutral',
            emotion_source TEXT DEFAULT 'auto',
            confidence REAL DEFAULT NULL,
            personal TEXT DEFAULT NULL,
            personal_context TEXT DEFAULT NULL,
            evolution_status TEXT DEFAULT NULL,
            resolved_at TEXT DEFAULT NULL,
            resolved_by TEXT DEFAULT NULL,
            due_date TEXT DEFAULT NULL,
            content_summary TEXT DEFAULT NULL,
            source_attribution TEXT DEFAULT NULL,
            scope TEXT DEFAULT NULL,
            encoding_version TEXT DEFAULT NULL,
            encoding_source TEXT DEFAULT NULL,
            revised_at TEXT DEFAULT NULL,
            source_turn_id TEXT DEFAULT NULL,
            last_accessed TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        'columns': {
            'id': None, 'type': None, 'title': None, 'content': None,
            'activation': '1.0', 'stability': '1.0', 'access_count': '1',
            'locked': '0', 'archived': '0', 'critical': '0', 'recency_score': '0',
            'emotion': '0', 'emotion_label': "'neutral'",
            'emotion_source': "'auto'",
            'confidence': 'NULL',
            'personal': 'NULL',              # v4: null | 'fixed' | 'fluid' | 'contextual'
            'personal_context': 'NULL',      # v4: qualifier for contextual personal nodes
            'evolution_status': 'NULL',      # v4: 'active' | 'resolved' | 'validated' | 'disproven' | 'confirmed' | 'dismissed'
            'resolved_at': 'NULL',           # v4: when the evolution node was resolved
            'resolved_by': 'NULL',           # v4: node_id of the decision/rule that resolved it
            'due_date': 'NULL',              # v4: ISO timestamp for reminders, scanned at boot
            'content_summary': 'NULL',       # v5: max 200 chars, auto-generated for tiered recall
            'source_attribution': 'NULL',    # v5: user_stated | claude_inferred | session_synthesis | correction | code_reading
            'scope': 'NULL',                 # v5: system | module | file | function | cross-system | cross-file | cross-function
            'encoding_version': 'NULL',      # v6: encoding pipeline version (v5, v6, etc.) — floor adapts to quality
            'encoding_source': 'NULL',       # v7: who created this node. Convention: "category:process". anchor = direct MCP, encoder:sonnet = encoding agent, idle:redistribution/consolidation/etc, hook:boot/compaction. Only 'anchor' can lock.
            'revised_at': 'NULL',            # v8: when this node was last revised via revise()
            'source_turn_id': 'NULL',        # v9 DEPRECATED (2026-05-23): single-ref legacy from removed message_stream table; superseded by node_source_refs (v27, multi-ref). Column kept to avoid touching existing rows; no new writes.
            'last_accessed': 'NULL',
            'created_at': 'NULL', 'updated_at': 'NULL',
        }
    },

    # v22: Physical edge — one row per pair, direction in source/target ordering.
    # source = "actor" node, target = "acted upon" node.
    # Single-direction: no mirror rows. Query both directions with OR.
    'edges': {
        'create': """CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            co_access_count INTEGER DEFAULT 0,
            last_strengthened TEXT,
            created_at TEXT,
            UNIQUE(source_id, target_id)
        )""",
        'columns': {
            'edge_id': None, 'source_id': None, 'target_id': None,
            'weight': '0.5', 'co_access_count': '0',
            'last_strengthened': 'NULL', 'created_at': 'NULL',
        }
    },

    # v22: Semantic edge layer — multiple relations per edge.
    # References edges via edge_id. Open text relation types.
    'edge_relations': {
        'create': """CREATE TABLE IF NOT EXISTS edge_relations (
            edge_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            description TEXT DEFAULT '',
            weight REAL DEFAULT 0.5,
            encoding_source TEXT DEFAULT '',
            decay_rate REAL DEFAULT NULL,
            created_at TEXT,
            archived INTEGER DEFAULT 0,
            archived_at TEXT DEFAULT NULL,
            archived_by TEXT DEFAULT NULL,
            embedding BLOB DEFAULT NULL,
            embedding_model TEXT DEFAULT NULL,
            PRIMARY KEY (edge_id, relation)
        )""",
        'columns': {
            'edge_id': None, 'relation': None,
            'description': "''", 'weight': '0.5',
            'encoding_source': "''",
            'decay_rate': 'NULL', 'created_at': 'NULL',
            # v25: soft-archive columns. PK stays (edge_id, relation);
            # add_relation handles re-archive as un-archive (UPDATE
            # archived=0) rather than requiring a composite PK.
            'archived': '0',
            'archived_at': 'NULL',
            'archived_by': 'NULL',
            # v26: stored edge embedding for surface_spread. Computed at
            # add_relation time from `_compose_enriched_edge_text`
            # (intrinsic — relation + description + family meaning, no
            # partner title). NULL until backfilled by
            # scripts/backfill_edge_embeddings.py or a future write.
            # embedding_model carries the model name so cross-model
            # staleness is detectable on swap.
            'embedding': 'NULL',
            'embedding_model': 'NULL',
        }
    },

    # v27: episodic references — multi-ref pointer from a node to the
    # trace events that anchor it. Sparse by design (typically 1-3 refs
    # per node). `position` preserves the encoder's write order so
    # render can expand primary refs first under budget. `trace_id`
    # is a cross-DB reference to brain_logs.trace_events.id — no
    # SQLite FK (cross-DB FKs aren't enforced); invalid refs degrade
    # gracefully at recall and are cleaned by S2Healer.
    # v29: trace_id is TEXT (8-char hex), matching node id shape.
    'node_source_refs': {
        'create': """CREATE TABLE IF NOT EXISTS node_source_refs (
            node_id    TEXT  NOT NULL,
            trace_id   TEXT  NOT NULL,
            position   INTEGER  NOT NULL DEFAULT 1,
            created_at TEXT,
            PRIMARY KEY (node_id, trace_id),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'node_id': None, 'trace_id': None,
            'position': '1', 'created_at': 'NULL',
        }
    },

    'brain_meta': {
        'create': """CREATE TABLE IF NOT EXISTS brain_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },
    # logs_meta lives in LOG_TABLES (brain_logs.db) — see below. brain_meta
    # is brain.db's counterpart; the two are deliberately separate so each
    # DB carries its own version and can be migrated independently.

    # version_history — REMOVED v21 (dead table)
    # summaries — REMOVED v21 (dead table)

    'emotion_calibration': {
        'create': """CREATE TABLE IF NOT EXISTS emotion_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            user_emotion REAL NOT NULL,
            user_label TEXT NOT NULL,
            context TEXT,
            created_at TEXT,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE SET NULL
        )""",
        'columns': {'id': None, 'node_id': 'NULL', 'user_emotion': None, 'user_label': None,
                    'context': 'NULL', 'created_at': 'NULL'}
    },

    'node_vectors': {
        'create': """CREATE TABLE IF NOT EXISTS node_vectors (
            node_id TEXT NOT NULL,
            term TEXT NOT NULL,
            tf REAL NOT NULL,
            tfidf REAL DEFAULT 0,
            PRIMARY KEY (node_id, term),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {'node_id': None, 'term': None, 'tf': None, 'tfidf': '0'}
    },

    'doc_freq': {
        'create': """CREATE TABLE IF NOT EXISTS doc_freq (
            term TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 1
        )""",
        'columns': {'term': None, 'count': None}
    },

    # projects — REMOVED v21 (dead table)
    # reasoning_chains — REMOVED v21 (dead table)
    # reasoning_steps — REMOVED v21 (dead table)

    'bridge_proposals': {
        'create': """CREATE TABLE IF NOT EXISTS bridge_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            shared_context TEXT DEFAULT '',
            dream_session_id TEXT DEFAULT '',
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending','created','expired','rejected')),
            proposed_at TEXT NOT NULL,
            matures_at TEXT NOT NULL,
            resolved_at TEXT,
            FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {'id': None, 'source_id': None, 'target_id': None,
                    'shared_context': "''", 'dream_session_id': "''",
                    'status': "'pending'", 'proposed_at': None,
                    'matures_at': None, 'resolved_at': 'NULL'}
    },

    # prune_archive — REMOVED v21 (dead table)

    # node_embeddings — REMOVED v23. Migrated to node_enrichments, table dropped.

    # v6→v23: Node vectors — ALL embeddings live here.
    # v6: enrichment vectors (question, anchor, bridge, keywords)
    # v23: also primary (_primary) and situation (_situation) vectors
    # generated at encode time by an LLM. These are searched alongside the primary embedding.
    # See PLAN.md "Embedding Migration to LLM" for design rationale and benchmark results.
    'node_enrichments': {
        'create': """CREATE TABLE IF NOT EXISTS node_enrichments (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            vector_type TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            model TEXT DEFAULT 'nomic-ai/nomic-embed-text-v1.5-Q',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'id': None, 'node_id': None, 'vector_type': None, 'text': None,
            'embedding': 'NULL', 'model': "'nomic-ai/nomic-embed-text-v1.5-Q'",
            'created_at': 'CURRENT_TIMESTAMP',
        }
    },

    # v14: Session activity tracking — replaces in-memory sessionActivity from index.js
    'session_activity': {
        'create': """CREATE TABLE IF NOT EXISTS session_activity (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },

    # DEPRECATED 2026-04-02: Fixed-column metadata. Replaced by node_metadata_kv.
    # Kept for backward compat — migration 008 copies data to KV. Stop writing here.
    # `node_metadata` table REMOVED 2026-05-17 — its fixed-column layout
    # was superseded by `node_metadata_kv` (decision 6bfe45d5, ~2026-04-15).
    # The migration to KV happened ~1 month before this comment; this entry
    # was vestigial schema-create scaffolding that produced an empty table
    # on every fresh brain. Live + clone brains have the table dropped;
    # fresh brains will no longer create it.

    # v21: Key-value metadata — extensible without schema changes
    'node_metadata_kv': {
        'create': """CREATE TABLE IF NOT EXISTS node_metadata_kv (
            node_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (node_id, key),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'node_id': None, 'key': None, 'value': 'NULL',
        }
    },

    # correction_traces — REMOVED v23. Dropped.

    # session_syntheses — REMOVED 2026-05-03 (writer synthesize_session
    # removed 2026-04-13; last reader hook_post_compact_reboot deleted with
    # the pre/post-compact hooks).

    # project_maps — REMOVED v21 (dead table)

    # v21: Embedding fidelity tracking for redistribution
    'embedding_fidelity': {
        'create': """CREATE TABLE IF NOT EXISTS embedding_fidelity (
            node_id TEXT PRIMARY KEY,
            original_embedding BLOB NOT NULL,
            fidelity REAL DEFAULT 1.0,
            last_redistributed TEXT,
            redistribution_count INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'original_embedding': None,
            'fidelity': '1.0',
            'last_redistributed': 'NULL',
            'redistribution_count': '0',
            'pinned': '0',
        }
    },

    # v21: Community detection results
    'node_communities': {
        'create': """CREATE TABLE IF NOT EXISTS node_communities (
            node_id TEXT PRIMARY KEY,
            community_id INTEGER
        )""",
        'columns': {
            'community_id': 'NULL',
        }
    },
    's2_rejections': {
        # Rejection memory for S2 integration units. When an encoder rejects
        # a proposal, its fingerprint is stored here so the decoder can
        # suppress re-proposals of identical input on subsequent runs.
        # Fingerprint captures what the encoder actually judges on (not
        # implementation artifacts) — changes in graph state that would
        # alter the proposal's inputs naturally produce a new fingerprint,
        # letting legitimate re-proposals through.
        # See servers/scales/s2/rejection_table.py for fingerprint logic.
        'create': """CREATE TABLE IF NOT EXISTS s2_rejections (
            fingerprint TEXT PRIMARY KEY,
            integration_unit TEXT NOT NULL,
            proposal_type TEXT NOT NULL,
            proposed_ids TEXT,
            created_at TEXT NOT NULL
        )""",
        'columns': {
            'fingerprint': 'TEXT PRIMARY KEY',
            'integration_unit': 'TEXT NOT NULL',
            'proposal_type': 'TEXT NOT NULL',
            'proposed_ids': 'TEXT',
            'created_at': 'TEXT NOT NULL',
        }
    },

    # Temporal index — interval-based date extraction for nodes and edges.
    # One row per (entity, extracted_date_interval, source). Populated by the
    # embed_queue worker at write time via dateparser scan of title/content/KV
    # for nodes and description/relation for edges. Queried by recall_by_time.
    # Intervals are half-open Unix-second pairs: "May 2023" -> (1682899200,
    # 1685577599); exact dates have start_ts == end_ts. See temporal_extraction.py.
    'entity_dates': {
        'create': """CREATE TABLE IF NOT EXISTS entity_dates (
            entity_kind TEXT NOT NULL CHECK(entity_kind IN ('node','edge')),
            entity_id TEXT NOT NULL,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER NOT NULL,
            extraction_source TEXT NOT NULL,
            raw_text TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_kind, entity_id, start_ts, end_ts, extraction_source)
        )""",
        'columns': {
            'entity_kind': None, 'entity_id': None,
            'start_ts': None, 'end_ts': None,
            'extraction_source': None, 'raw_text': 'NULL',
            'created_at': 'CURRENT_TIMESTAMP',
        }
    },
}

# ─── Canonical indexes ───
INDEXES = [
    # nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation DESC)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_emotion ON nodes(emotion)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)',
    # edges (v22: edge_id PK, source/target for lookups)
    'CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight)',
    'CREATE INDEX IF NOT EXISTS idx_edges_pair ON edges(source_id, target_id)',
    # edge_relations (v22: edge_id FK, relation for type queries)
    'CREATE INDEX IF NOT EXISTS idx_edge_relations_edge ON edge_relations(edge_id)',
    'CREATE INDEX IF NOT EXISTS idx_edge_relations_relation ON edge_relations(relation)',
    # v27: node_source_refs reverse-lookup (engram cohort detection — get all nodes anchored to a given trace)
    'CREATE INDEX IF NOT EXISTS idx_nsr_trace ON node_source_refs(trace_id)',
    # node_vectors
    'CREATE INDEX IF NOT EXISTS idx_vectors_term ON node_vectors(term)',
    'CREATE INDEX IF NOT EXISTS idx_vectors_node ON node_vectors(node_id)',
    # bridge_proposals
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_status ON bridge_proposals(status)',
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_matures ON bridge_proposals(matures_at)',
    # node_embeddings — REMOVED v23 (table deprecated, index not maintained)
    # v15: node_metadata — REMOVED 2026-05-17 (table dropped; KV table is
    # canonical, see decision 6bfe45d5). Indexes idx_metadata_correction
    # and idx_metadata_validated removed alongside.
    # correction_traces — REMOVED v23
    # session_syntheses — REMOVED 2026-05-03
    # v15: nodes scope for engineering memory
    'CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope)',
    # v16: critical flag for safety-important nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_critical ON nodes(critical)',
    # v6 (LLM migration): node_enrichments
    'CREATE INDEX IF NOT EXISTS idx_enrichments_node ON node_enrichments(node_id)',
    'CREATE INDEX IF NOT EXISTS idx_enrichments_type ON node_enrichments(vector_type)',
    # v23: composite index for vector lookups (backfill_vectors, recall scan)
    'CREATE INDEX IF NOT EXISTS idx_enrichments_node_type ON node_enrichments(node_id, vector_type)',
    # s2_rejections — per-unit queries for cleanup/analysis
    'CREATE INDEX IF NOT EXISTS idx_s2_rejections_unit ON s2_rejections(integration_unit)',
    'CREATE INDEX IF NOT EXISTS idx_s2_rejections_created ON s2_rejections(created_at)',
    # entity_dates — interval overlap queries (recall_by_time)
    'CREATE INDEX IF NOT EXISTS idx_entity_dates_start ON entity_dates(start_ts)',
    'CREATE INDEX IF NOT EXISTS idx_entity_dates_end ON entity_dates(end_ts)',
    'CREATE INDEX IF NOT EXISTS idx_entity_dates_kind_id ON entity_dates(entity_kind, entity_id)',
    # brain_telemetry indexes — moved to LOG_INDEXES (brain_logs.db)
]


def _now():
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Versioned migration runner — the ONE mechanism for "code changed,
# migrate this install's data forward".
#
# There is a fleet now (installs other than the author's), so a change
# cannot be applied by hand on one machine. Every versioned stream below
# self-applies at open, exactly once, in order.
#
# Three streams use this, each with its own counter so they move
# independently — structure changes rarely, prompt content often, and
# conflating them would make every prompt edit look like a schema change:
#
#   brain.db      brain_meta.brain_schema_version   BRAIN_VERSION
#   brain_logs.db logs_meta.logs_schema_version     LOGS_VERSION
#   brain_logs.db logs_meta.seed_prompts_version    SEED_PROMPTS_VERSION
#
# Forward-only and idempotent by version guard: a step runs iff the DB's
# stored version is below it. Re-opening a current DB does no writes.
# ═══════════════════════════════════════════════════════════════

def read_schema_version(conn, meta_table: str, version_key: str) -> int:
    """Stored version for a stream, or 0 when never stamped.

    0 covers both a fresh DB and a pre-versioning install — which is
    exactly right: both need every step.
    """
    try:
        cur = conn.execute(
            "SELECT value FROM %s WHERE key = ?" % meta_table, (version_key,))
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def stamp_schema_version(conn, meta_table: str, version_key: str,
                         version: int) -> None:
    """Record a stream as migrated up to `version`."""
    conn.execute(
        "INSERT OR REPLACE INTO %s (key, value, updated_at) "
        "VALUES (?, ?, ?)" % meta_table,
        (version_key, str(version), _now()))


def run_versioned_migrations(conn, meta_table: str, version_key: str,
                             target_version: int, steps, label: str = '') -> int:
    """Run every step the stored version hasn't seen, then stamp.

    `steps` is an ordered [(version, callable(conn))]. A step runs iff
    stored_version < its version. Steps must be idempotent anyway — a
    crash between the last step and the stamp replays them.

    Returns the version found on entry (0 = fresh/pre-versioning), so the
    caller can log or branch on "did this install just migrate".

    A failing step is NOT swallowed: the stamp is skipped so the migration
    retries on the next open rather than marking the DB current on top of
    a half-applied change.
    """
    current = read_schema_version(conn, meta_table, version_key)
    if current >= target_version:
        return current
    for step_version, step_fn in steps:
        if current < step_version:
            step_fn(conn)
            if label:
                print('[brain] %s: applied step v%d' % (label, step_version),
                      flush=True)
    stamp_schema_version(conn, meta_table, version_key, target_version)
    if label and current > 0:
        print('[brain] %s: v%d -> v%d' % (label, current, target_version),
              flush=True)
    return current


def _backup_before_migration(db_path, from_version, to_version):
    """Create a backup before schema migration. Returns backup path or None."""
    if not db_path or from_version == 0 or from_version >= to_version:
        return None
    try:
        backup_path = db_path + '.v%d.bak' % from_version
        if os.path.exists(backup_path):
            return backup_path  # already backed up from a previous attempt
        shutil.copy2(db_path, backup_path)
        print('[brain] Backup created: %s' % backup_path)
        return backup_path
    except Exception as e:
        print('[brain] Backup failed (continuing anyway): %s' % e)
        return None


def ensure_schema(conn, db_path=None):
    """
    The ONLY function that touches table structure.
    Mirrors schema.js ensureSchema() exactly.

    Args:
        conn: SQLite connection
        db_path: Optional path to the DB file (enables pre-migration backup)
    """
    # 1. Create brain_meta first
    conn.execute(TABLES['brain_meta']['create'])

    # 2. Check current schema version (shared reader — same primitive the
    #    logs and prompt streams use, so all three read versions one way)
    current_version = read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY)

    # 2a. Refuse a DB older than the supported floor rather than silently
    #     stamping it current. 0 = fresh DB (built at BRAIN_VERSION) — fine.
    if 0 < current_version < MIN_SUPPORTED_VERSION:
        raise RuntimeError(
            'brain.db is at schema v%d; this build supports v%d and newer. '
            'The v1-v%d upgrade path was removed as unreachable (no such DB '
            'existed). Restore a newer backup, or check out a build from '
            'before the removal to migrate this file forward first.'
            % (current_version, MIN_SUPPORTED_VERSION,
               MIN_SUPPORTED_VERSION - 1))

    # 2b. Backup before migration if version is changing
    backup_path = None
    if current_version > 0 and current_version < BRAIN_VERSION and db_path:
        backup_path = _backup_before_migration(db_path, current_version, BRAIN_VERSION)

    # 3. Get list of existing tables
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cur.fetchall()}

    # 4. For each canonical table: create or diff+alter
    for table_name, spec in TABLES.items():
        if table_name not in existing_tables:
            conn.execute(spec['create'])
            continue

        # Get current columns
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in cur.fetchall()}

        # Add missing columns
        for col_name, default_val in spec['columns'].items():
            if col_name not in existing_cols:
                def_clause = f' DEFAULT {default_val}' if default_val is not None else ''
                try:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name}{def_clause}")
                except Exception:
                    pass

    # 6. Create all indexes
    for idx in INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass

    # 6b. FTS5 full-text search virtual table
    # FTS5 tables use CREATE VIRTUAL TABLE — no ALTER TABLE support, separate from TABLES dict.
    # Porter stemming: "recommending" matches "recommend". Unicode61 for international text.
    try:
        # 3-column FTS5 schema. CREATE VIRTUAL TABLE IF NOT EXISTS never
        # rebuilds an existing table, so a shape change here needs a
        # numbered migration in MAIN_MIGRATIONS that drops and recreates it.
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            title,
            content,
            tokenize='porter unicode61'
        )""")
        # Auto-populate on first run: FTS5 empty but nodes exist
        _fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        _node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
        if _fts_count == 0 and _node_count > 0:
            conn.execute("""
                INSERT INTO nodes_fts (node_id, title, content)
                SELECT id, title, COALESCE(content, '')
                FROM nodes WHERE archived = 0
            """)
            conn.commit()
            print(f"[brain] FTS5 index populated: {_node_count} nodes")
    except Exception as e:
        print(f"[brain] FTS5 setup warning: {e}")

    # 7. Update version (shared stamper — see run_versioned_migrations)
    if current_version < BRAIN_VERSION:
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION)

        try:
            conn.execute(
                "INSERT INTO version_history (version, migration_ts, description, backup_path) VALUES (?, ?, ?, ?)",
                (BRAIN_VERSION, _now(),
                 f'Schema ensured: v{current_version} -> v{BRAIN_VERSION} (serverless Python)',
                 backup_path)
            )
        except Exception:
            pass

        print(f"[brain] Schema ensured: v{current_version} -> v{BRAIN_VERSION}")

    # 8. Numbered data migrations (empty at v30 — see MAIN_MIGRATIONS)
    run_versioned_migrations(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION, MAIN_MIGRATIONS,
                             label='brain schema')

    conn.commit()


# ═══════════════════════════════════════════════════════════════
# LOGS DATABASE — separate from brain.db for isolation
# ═══════════════════════════════════════════════════════════════

# Tables that live in brain_logs.db instead of brain.db.
# These are operational/telemetry tables that grow unbounded and
# don't need referential integrity with the knowledge graph.
LOG_TABLES = {
    # logs_meta — brain_logs.db's version counter, mirroring brain_meta in
    # brain.db. Created FIRST (dict order is insertion order) so the
    # migration runner can read a version before any other table is touched.
    'logs_meta': {
        'create': """CREATE TABLE IF NOT EXISTS logs_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },
    # access_log — REMOVED v21 (dead table)
    'debug_log': {
        'create': """CREATE TABLE IF NOT EXISTS debug_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT NOT NULL,
            source TEXT,
            file_target TEXT,
            suggestions_served INTEGER DEFAULT 0,
            procedures_served INTEGER DEFAULT 0,
            node_ids_served TEXT,
            latency_ms REAL,
            brain_reachable INTEGER DEFAULT 1,
            metadata TEXT,
            created_at TEXT
        )""",
    },
    # recall_log — REMOVED v21 (dead table, replaced by trace_events)
    # miss_log — REMOVED v21 (dead table)
    'dream_log': {
        'create': """CREATE TABLE IF NOT EXISTS dream_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            intuition_node_id TEXT,
            seed_nodes TEXT NOT NULL,
            walk_path TEXT NOT NULL,
            insight TEXT,
            created_at TEXT
        )""",
    },
    # tuning_log — REMOVED v21 (dead table)
    # eval_snapshots — REMOVED v21 (dead table)
    # suggest_log — REMOVED v21 (dead table)
    # curiosity_log — REMOVED v21 (dead table)
    # health_log — REMOVED v21 (dead table)
    # conflict_log — REMOVED v21 (dead table)
    # staged_learnings — REMOVED v21 (dead table)
    # message_stream — REMOVED v21 (dead table, replaced by trace_events)
    # recall_gaps — REMOVED v21 (dead table)
    # pending_consolidation — REMOVED v21 (dead table)
    # brain_telemetry — REMOVED v21 (dead table)

    'hook_errors': {
        'create': """CREATE TABLE IF NOT EXISTS hook_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            hook_name TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'error',
            error TEXT NOT NULL,
            context TEXT DEFAULT '',
            traceback TEXT DEFAULT '',
            surfaced INTEGER DEFAULT 0
        )""",
    },

    # v9.2: Session state — first-class session-scoped data.
    # Replaces scattered in-memory dicts and brain_meta config keys.
    # Key types: 'fatigue' (node_id=node, value=count), 'journal' (node_id='', value=text),
    # 'context' (node_id='', value=journey text). Session-keyed, auto-cleanup by age.
    'session_state': {
        'create': """CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            node_id TEXT NOT NULL DEFAULT '',
            value TEXT,
            updated_at TEXT,
            PRIMARY KEY (session_id, key, node_id)
        )""",
    },

    'interactions': {
        'create': """CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            template TEXT NOT NULL,
            parameters TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT DEFAULT 'anchor',
            parent_version INTEGER,
            UNIQUE(name, version)
        )""",
    },

    # interaction_active — pointer per name to the currently-active version.
    # Separates "register a new version" from "make it live." Without this,
    # any register_interaction call silently changed Anchor's runtime behavior.
    # Now: register inserts a row; set_active flips the pointer; runtime reads
    # the active pointer. Schema added 2026-05-10.
    'interaction_active': {
        'create': """CREATE TABLE IF NOT EXISTS interaction_active (
            name TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            set_at TEXT NOT NULL,
            set_by TEXT NOT NULL
        )""",
    },
    # v29: trace_events.id is TEXT (8-char hex), matching node/edge id shape.
    # Generation: TraceDAL.append() calls secrets.token_hex(4) with collision
    # retry. Historical rows migrate via printf('%08x', old_int) — deterministic
    # so any pre-migration integer reference can still be resolved.
    'trace_events': {
        'create': """CREATE TABLE IF NOT EXISTS trace_events (
            id TEXT PRIMARY KEY,
            chain_id TEXT NOT NULL,
            scale TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ref_type TEXT,
            ref_id TEXT,
            summary TEXT,
            metadata TEXT,
            session_id TEXT,
            interaction_id INTEGER,
            created_at TEXT NOT NULL
        )""",
    },

    # v27: per-trace embeddings for episodic references. One row per
    # unique trace_id (PK); shared by N nodes referencing the same
    # trace. The `text` column stores the rendered string that was
    # fed to the embedder — matches node_enrichments convention;
    # enables diagnostics and detects drift if the trace rendering
    # changes over time.
    # v29: trace_id is TEXT (matches trace_events.id).
    'trace_embeddings': {
        'create': """CREATE TABLE IF NOT EXISTS trace_embeddings (
            trace_id    TEXT    PRIMARY KEY,
            vector      BLOB    NOT NULL,
            text        TEXT,
            model       TEXT,
            created_at  TEXT
        )""",
    },

    # Self channel — directed-signal courier (Phase 2a). One stream sends a
    # message addressed to another live stream (self:<sid>) or self:broadcast;
    # the recipient consumes it once via self_delivered. Pull-based in 2a.
    # Per-message expires_at (resolved by address at send) enforces TTL — readers
    # filter expires_at > now; reap deletes expires_at <= now OR IS NULL. Nullable
    # so the ALTER ADD on the one existing courier succeeds; send() always sets
    # it, so a NULL expiry only ever means a pre-column legacy row — reaped as dead.
    'self_inflight': {
        'create': """CREATE TABLE IF NOT EXISTS self_inflight (
            id TEXT PRIMARY KEY,
            from_session TEXT NOT NULL,
            address TEXT NOT NULL,
            body TEXT NOT NULL,
            refs TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            expires_at TEXT
        )""",
    },
    # One row per (message, recipient) — broadcast fans out, each recipient
    # consumes exactly once (PK guards double-delivery).
    'self_delivered': {
        'create': """CREATE TABLE IF NOT EXISTS self_delivered (
            message_id TEXT NOT NULL,
            to_session TEXT NOT NULL,
            delivered_at TEXT NOT NULL,
            PRIMARY KEY (message_id, to_session)
        )""",
    },

    # Boot observability — the exact `for_claude` text the daemon rendered and
    # served to a session at SessionStart. context_boot is read-only re: the
    # knowledge graph, but it logs what it served here (the same pattern as
    # recall writing a trace from a read path). One row per real boot; the
    # dashboard's Streams→Boot view reads the latest per session. "What
    # actually got to boot." Written by Brain.record_boot_render via
    # _handle_context_boot; never gated on a BRAIN_VERSION bump (new table).
    'boot_renders': {
        'create': """CREATE TABLE IF NOT EXISTS boot_renders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user TEXT DEFAULT '',
            project TEXT DEFAULT '',
            char_count INTEGER DEFAULT 0,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
    },

}

LOG_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_debug_session ON debug_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_debug_type ON debug_log(event_type)',
    'CREATE INDEX IF NOT EXISTS idx_debug_created ON debug_log(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_dream_log_session ON dream_log(session_id)',
    # hook_errors
    'CREATE INDEX IF NOT EXISTS idx_hook_errors_hook ON hook_errors(hook_name)',
    'CREATE INDEX IF NOT EXISTS idx_hook_errors_created ON hook_errors(created_at)',
    # interactions
    'CREATE INDEX IF NOT EXISTS idx_interactions_name ON interactions(name)',
    # trace_events
    'CREATE INDEX IF NOT EXISTS idx_trace_chain ON trace_events(chain_id)',
    'CREATE INDEX IF NOT EXISTS idx_trace_scale ON trace_events(scale)',
    'CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_events(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_events(session_id)',
    # v27: composite covers the embed-worker hot query (find_unembedded):
    # filter by scale + ref_type, sort by created_at DESC, stop early at
    # LIMIT. Without this SQLite materializes all matches into a temp
    # B-tree just to sort — every drain tick. With it the planner walks
    # the index in DESC order and short-circuits after LIMIT matches.
    'CREATE INDEX IF NOT EXISTS idx_trace_scope_created ON trace_events(scale, ref_type, created_at)',
    # journal-note subject pulls: WHERE ref_type='journal_note' AND ref_id=?
    # ORDER BY created_at — the hotspot/subject query (N notes on one node).
    'CREATE INDEX IF NOT EXISTS idx_trace_ref_subject ON trace_events(ref_type, ref_id, created_at)',
    # get_session_turns hot path (hook_recall, once per user prompt): filter by
    # session + scale, walk created_at DESC, stop at LIMIT — same short-circuit
    # pattern as idx_trace_scope_created above. Without it the planner loads
    # every row of the session before sorting.
    'CREATE INDEX IF NOT EXISTS idx_trace_session_created ON trace_events(session_id, scale, created_at)',
    # v9.2: session_state
    'CREATE INDEX IF NOT EXISTS idx_session_state_session ON session_state(session_id)',
    # self channel — drain/peek filter by address + expires_at; reap by expires_at;
    # outbox orders by created_at
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_address ON self_inflight(address)',
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_created ON self_inflight(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_expires ON self_inflight(expires_at)',
    'CREATE INDEX IF NOT EXISTS idx_self_delivered_to ON self_delivered(to_session)',
    # boot_renders — dashboard reads latest-per-session and newest-first
    'CREATE INDEX IF NOT EXISTS idx_boot_renders_session ON boot_renders(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_boot_renders_created ON boot_renders(created_at)',
]


# Numbered structural migrations for brain_logs.db. Empty at v1 — the
# baseline is "whatever shape this DB already had", since the operations
# below are self-detecting and idempotent and have run unversioned for a
# long time. Rewriting proven code to be version-gated would risk the
# working path for no gain; new structural changes ride these rails.
LOGS_MIGRATIONS = []


def ensure_logs_schema(conn):
    """Create all log tables in the logs database (brain_logs.db).

    Also handles column migrations for existing tables via ALTER TABLE,
    and runs the versioned structural migrations (LOGS_MIGRATIONS) that
    every install applies itself at open.
    """
    conn.execute('PRAGMA journal_mode=WAL')

    # logs_meta must exist before the runner can read a version. Cheap and
    # idempotent, so it runs ahead of the main table loop rather than
    # depending on dict ordering staying stable.
    conn.execute(LOG_TABLES['logs_meta']['create'])

    for table_name, spec in LOG_TABLES.items():
        conn.execute(spec['create'])

    _add_column_if_missing(conn, 'trace_events', 'interaction_id', 'INTEGER')

    # Per-message self-channel TTL: add expires_at to the courier. send()
    # stamps it on every new message; any legacy NULL row is swept by reap
    # as dead. (This predates versioning and is idempotent, so it stays
    # unconditional — but note there IS a fleet now, so anything new goes
    # through LOGS_MIGRATIONS instead of an unversioned ALTER.)
    _add_column_if_missing(conn, 'self_inflight', 'expires_at', 'TEXT')

    # Initial population for interaction_active — one-time migration.
    # For brains created before the active-version split, populate the pointer
    # with the current MAX(version) per name so runtime semantics stay
    # byte-identical until someone explicitly registers a new version.
    # Idempotent: INSERT OR IGNORE skips names that already have a pointer.
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            'INSERT OR IGNORE INTO interaction_active (name, version, set_at, set_by) '
            'SELECT name, MAX(version), ?, ? FROM interactions GROUP BY name',
            (now, 'migration:initial_active')
        )
    except Exception:
        pass  # No interactions table yet — fresh brain; seed will populate.

    for idx in LOG_INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass

    # Versioned structural migrations + baseline stamp. Runs after the
    # tables exist so a step can assume current shapes.
    run_versioned_migrations(conn, 'logs_meta', LOGS_VERSION_KEY,
                             LOGS_VERSION, LOGS_MIGRATIONS,
                             label='logs schema')
    conn.commit()


def _add_column_if_missing(conn, table: str, column: str, col_type: str):
    """Add a column to an existing table if it doesn't exist.

    SQLite has no IF NOT EXISTS for ALTER TABLE, so the duplicate-column error is
    the expected idempotent-rerun signal — swallowed silently (it isn't a
    failure). Any OTHER OperationalError (locked DB, disk full) is a real
    migration failure and is logged LOUDLY (stderr→daemon.log) rather than
    hidden, so a column that silently failed to add can't break a feature at
    runtime with no trace. Boot continues regardless — matching every migration
    block in this file. Non-OperationalError exceptions (e.g. a bad col_type)
    propagate: those are dev-time bugs that should fail loud."""
    try:
        conn.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table, column, col_type))
    except sqlite3.OperationalError as e:
        if 'duplicate column' not in str(e).lower():
            print('[brain] schema: ALTER TABLE %s ADD COLUMN %s %s failed: %s'
                  % (table, column, col_type, e))


def migrate_logs_to_separate_db(main_conn, logs_conn):
    """One-time migration: copy log tables from brain.db to brain_logs.db.

    Idempotent — skips tables that already have data in logs_conn.
    After copying, drops the table from main_conn to reclaim space.
    """
    migrated = []
    for table_name in LOG_TABLES:
        # Check if table exists in main DB
        cur = main_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cur.fetchone():
            continue

        # Check if logs DB already has data for this table
        try:
            cur = logs_conn.execute('SELECT COUNT(*) FROM %s' % table_name)
            if cur.fetchone()[0] > 0:
                # Already migrated — just drop from main
                try:
                    main_conn.execute('DROP TABLE IF EXISTS %s' % table_name)
                except Exception:
                    pass
                continue
        except Exception:
            pass

        # Copy data
        try:
            cur = main_conn.execute('SELECT * FROM %s' % table_name)
            rows = cur.fetchall()
            if rows:
                # Get column names from main DB
                col_cur = main_conn.execute('PRAGMA table_info(%s)' % table_name)
                col_names = [r[1] for r in col_cur.fetchall()]
                placeholders = ','.join(['?'] * len(col_names))
                col_list = ','.join(col_names)
                logs_conn.executemany(
                    'INSERT OR IGNORE INTO %s (%s) VALUES (%s)' % (table_name, col_list, placeholders),
                    rows
                )
            # Drop from main DB
            main_conn.execute('DROP TABLE IF EXISTS %s' % table_name)
            migrated.append(table_name)
        except Exception as e:
            print('[brain] Log migration note for %s: %s' % (table_name, e))

    if migrated:
        logs_conn.commit()
        main_conn.commit()
        print('[brain] Migrated %d log table(s) to brain_logs.db: %s' % (len(migrated), ', '.join(migrated)))

    return migrated
