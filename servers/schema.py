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

BRAIN_VERSION = 20  # v20: removed CHECK constraint on nodes.type — agents can use any type string
BRAIN_VERSION_KEY = 'brain_schema_version'

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
            keywords TEXT,
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
            project TEXT,
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
            'keywords': None,
            'activation': '1.0', 'stability': '1.0', 'access_count': '1',
            'locked': '0', 'archived': '0', 'critical': '0', 'recency_score': '0',
            'emotion': '0', 'emotion_label': "'neutral'",
            'emotion_source': "'auto'", 'project': 'NULL',
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
            'source_turn_id': 'NULL',        # v9: message_stream.id that produced this node (episode linkage)
            'last_accessed': 'NULL',
            'created_at': 'NULL', 'updated_at': 'NULL',
        }
    },

    'edges': {
        'create': """CREATE TABLE IF NOT EXISTS edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            relation TEXT DEFAULT 'related',
            co_access_count INTEGER DEFAULT 1,
            stability REAL DEFAULT 1.0,
            last_strengthened TEXT,
            created_at TEXT,
            edge_type TEXT DEFAULT 'related',
            decay_rate REAL DEFAULT NULL,
            description TEXT DEFAULT '',
            PRIMARY KEY (source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'source_id': None, 'target_id': None,
            'weight': '0.5', 'relation': "'related'",
            'co_access_count': '1', 'stability': '1.0',
            'last_strengthened': 'NULL', 'created_at': 'NULL',
            'edge_type': "'related'", 'decay_rate': 'NULL',
            'description': "''",
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

    'node_embeddings': {
        'create': """CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            situation_embedding BLOB,
            situation_text TEXT,
            model TEXT NOT NULL DEFAULT 'snowflake-arctic-embed-m',
            created_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {'node_id': None, 'embedding': None,
                    'situation_embedding': None, 'situation_text': None,
                    'model': "'snowflake-arctic-embed-m'", 'created_at': None}
    },

    # v6: Node enrichments — multi-vector encoding for improved recall.
    # Each node can have multiple enrichment vectors (question, anchor, bridge, keywords)
    # generated at encode time by an LLM. These are searched alongside the primary embedding.
    # See PLAN.md "Embedding Migration to LLM" for design rationale and benchmark results.
    'node_enrichments': {
        'create': """CREATE TABLE IF NOT EXISTS node_enrichments (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            vector_type TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            model TEXT DEFAULT 'snowflake-arctic-embed-m',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'id': None, 'node_id': None, 'vector_type': None, 'text': None,
            'embedding': 'NULL', 'model': "'snowflake-arctic-embed-m'",
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
    'node_metadata': {
        'create': """CREATE TABLE IF NOT EXISTS node_metadata (
            node_id TEXT PRIMARY KEY,
            reasoning TEXT,
            alternatives TEXT,
            user_raw_quote TEXT,
            correction_of TEXT,
            correction_pattern TEXT,
            source_context TEXT,
            confidence_rationale TEXT,
            last_validated TEXT,
            validation_count INTEGER DEFAULT 0,
            change_impacts TEXT,
            created_at TEXT,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'node_id': None, 'reasoning': 'NULL', 'alternatives': 'NULL',
            'user_raw_quote': 'NULL', 'correction_of': 'NULL',
            'correction_pattern': 'NULL', 'source_context': 'NULL',
            'confidence_rationale': 'NULL', 'last_validated': 'NULL',
            'validation_count': '0', 'change_impacts': 'NULL',
            'created_at': 'NULL',
        }
    },

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

    # v15: Self-correction traces — where Claude's model diverges from reality
    'correction_traces': {
        'create': """CREATE TABLE IF NOT EXISTS correction_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            original_node_id TEXT,
            corrected_node_id TEXT,
            claude_assumed TEXT NOT NULL,
            reality TEXT NOT NULL,
            underlying_pattern TEXT,
            severity TEXT DEFAULT 'minor',
            created_at TEXT,
            FOREIGN KEY (original_node_id) REFERENCES nodes(id) ON DELETE SET NULL,
            FOREIGN KEY (corrected_node_id) REFERENCES nodes(id) ON DELETE SET NULL
        )""",
        'columns': {
            'id': None, 'session_id': 'NULL', 'original_node_id': 'NULL',
            'corrected_node_id': 'NULL', 'claude_assumed': None, 'reality': None,
            'underlying_pattern': 'NULL', "severity": "'minor'", 'created_at': 'NULL',
        }
    },

    # v15: Session syntheses — structured knowledge from conversations
    'session_syntheses': {
        'create': """CREATE TABLE IF NOT EXISTS session_syntheses (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            duration_minutes INTEGER,
            decisions_made TEXT,
            corrections_received TEXT,
            inflection_points TEXT,
            mental_model_updates TEXT,
            teaching_arcs TEXT,
            open_questions TEXT,
            created_at TEXT
        )""",
        'columns': {
            'id': None, 'session_id': None, 'duration_minutes': 'NULL',
            'decisions_made': 'NULL', 'corrections_received': 'NULL',
            'inflection_points': 'NULL', 'mental_model_updates': 'NULL',
            'teaching_arcs': 'NULL', 'open_questions': 'NULL',
            'created_at': 'NULL',
        }
    },

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
}

# ─── Canonical indexes ───
INDEXES = [
    # nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_keywords ON nodes(keywords)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation DESC)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_emotion ON nodes(emotion)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)',
    # edges
    'CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight)',
    'CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)',
    # node_vectors
    'CREATE INDEX IF NOT EXISTS idx_vectors_term ON node_vectors(term)',
    'CREATE INDEX IF NOT EXISTS idx_vectors_node ON node_vectors(node_id)',
    # bridge_proposals
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_status ON bridge_proposals(status)',
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_matures ON bridge_proposals(matures_at)',
    # node_embeddings
    'CREATE INDEX IF NOT EXISTS idx_node_embeddings_model ON node_embeddings(model)',
    # v15: node_metadata
    'CREATE INDEX IF NOT EXISTS idx_metadata_correction ON node_metadata(correction_of)',
    'CREATE INDEX IF NOT EXISTS idx_metadata_validated ON node_metadata(last_validated)',
    # v15: correction_traces
    'CREATE INDEX IF NOT EXISTS idx_correction_traces_pattern ON correction_traces(underlying_pattern)',
    'CREATE INDEX IF NOT EXISTS idx_correction_traces_session ON correction_traces(session_id)',
    # v15: session_syntheses
    'CREATE INDEX IF NOT EXISTS idx_session_syntheses_session ON session_syntheses(session_id)',
    # v15: nodes scope for engineering memory
    'CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope)',
    # v16: critical flag for safety-important nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_critical ON nodes(critical)',
    # v6 (LLM migration): node_enrichments
    'CREATE INDEX IF NOT EXISTS idx_enrichments_node ON node_enrichments(node_id)',
    'CREATE INDEX IF NOT EXISTS idx_enrichments_type ON node_enrichments(vector_type)',
    # brain_telemetry indexes — moved to LOG_INDEXES (brain_logs.db)
]


def _now():
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


def _rebuild_nodes(conn, spec):
    """Rebuild nodes table when CHECK constraint changes (new node types)."""
    conn.execute('PRAGMA foreign_keys=OFF')

    # Get current columns
    cur = conn.execute('PRAGMA table_info(nodes)')
    current_cols = [row[1] for row in cur.fetchall()]

    canonical_cols = list(spec['columns'].keys())
    shared_cols = [c for c in current_cols if c in canonical_cols]

    # Build insert columns — use defaults for missing ones
    insert_parts = []
    for c in canonical_cols:
        if c in shared_cols:
            insert_parts.append(c)
        else:
            default = spec['columns'][c]
            insert_parts.append(f"{default} AS {c}" if default is not None else f"NULL AS {c}")

    try:
        conn.execute(spec['create'].replace('nodes', 'nodes_canonical'))
        conn.execute(f"""INSERT OR IGNORE INTO nodes_canonical ({','.join(canonical_cols)})
                        SELECT {','.join(insert_parts)} FROM nodes""")
        conn.execute('DROP TABLE nodes')
        conn.execute('ALTER TABLE nodes_canonical RENAME TO nodes')
    except Exception as e:
        print(f"[brain] nodes rebuild note: {e}")
        try:
            conn.execute('DROP TABLE IF EXISTS nodes_canonical')
        except Exception:
            pass

    conn.execute('PRAGMA foreign_keys=ON')


def _backfill_data(conn, from_version):
    """One-time data backfills for existing brains."""
    if from_version < 8:
        try:
            conn.execute('UPDATE nodes SET confidence = 1.0 WHERE locked = 1 AND confidence IS NULL')
            conn.execute('UPDATE nodes SET confidence = 0.5 WHERE locked = 0 AND confidence IS NULL')
        except Exception:
            pass

    if from_version < 6:
        try:
            conn.execute("UPDATE edges SET edge_type = 'corrected_by' WHERE relation = 'corrected_by'")
            conn.execute("UPDATE edges SET edge_type = 'part_of' WHERE relation = 'part_of'")
            conn.execute("UPDATE edges SET edge_type = 'exemplifies' WHERE relation = 'example_of'")
            conn.execute("UPDATE edges SET edge_type = 'related' WHERE edge_type IS NULL")
        except Exception:
            pass

    if from_version < 15:
        # v15: Generate content_summary for existing nodes that have content
        try:
            cur = conn.execute(
                "SELECT id, title, content FROM nodes WHERE content IS NOT NULL AND content != '' AND content_summary IS NULL"
            )
            for row in cur.fetchall():
                node_id, title, content = row
                # First sentence or first 200 chars
                summary = content
                period_idx = content.find('. ')
                if 0 < period_idx < 200:
                    summary = content[:period_idx + 1]
                elif len(content) > 200:
                    summary = content[:197] + '...'
                conn.execute(
                    "UPDATE nodes SET content_summary = ? WHERE id = ?",
                    (summary, node_id)
                )
            print(f"[brain] v15 backfill: generated content_summary for existing nodes")
        except Exception as e:
            print(f"[brain] v15 backfill note: {e}")


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

    # 2. Check current schema version
    cur = conn.execute(
        f"SELECT value FROM brain_meta WHERE key = ?", (BRAIN_VERSION_KEY,)
    )
    row = cur.fetchone()
    current_version = int(row[0]) if row else 0

    # 2b. Backup before migration if version is changing
    backup_path = None
    if current_version > 0 and current_version < BRAIN_VERSION and db_path:
        backup_path = _backup_before_migration(db_path, current_version, BRAIN_VERSION)

    # 3. Get list of existing tables
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cur.fetchall()}

    # 4. Check if nodes table needs rebuild (CHECK constraint removal)
    nodes_need_rebuild = False
    if 'nodes' in existing_tables:
        cur = conn.execute("SELECT sql FROM sqlite_master WHERE name='nodes'")
        row = cur.fetchone()
        if row:
            current_sql = row[0] or ''
            # Rebuild if the old CHECK constraint is still present
            if 'CHECK(type IN' in current_sql or "CHECK (type IN" in current_sql:
                nodes_need_rebuild = True

    # 5. For each canonical table: create or diff+alter
    for table_name, spec in TABLES.items():
        if table_name not in existing_tables:
            conn.execute(spec['create'])
            continue

        if table_name == 'nodes' and nodes_need_rebuild:
            _rebuild_nodes(conn, spec)
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
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            title,
            content,
            keywords,
            tokenize='porter unicode61'
        )""")
        # Auto-populate on first run: FTS5 empty but nodes exist
        _fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        _node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
        if _fts_count == 0 and _node_count > 0:
            conn.execute("""
                INSERT INTO nodes_fts (node_id, title, content, keywords)
                SELECT id, title, COALESCE(content, ''), COALESCE(keywords, '')
                FROM nodes WHERE archived = 0
            """)
            conn.commit()
            print(f"[brain] FTS5 index populated: {_node_count} nodes")
    except Exception as e:
        print(f"[brain] FTS5 setup warning: {e}")

    # 7. Update version
    if current_version < BRAIN_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)",
            (BRAIN_VERSION_KEY, str(BRAIN_VERSION), _now())
        )

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

    # 8. One-time data backfills
    if current_version > 0 and current_version < BRAIN_VERSION:
        _backfill_data(conn, current_version)

    conn.commit()


# ═══════════════════════════════════════════════════════════════
# LOGS DATABASE — separate from brain.db for isolation
# ═══════════════════════════════════════════════════════════════

# Tables that live in brain_logs.db instead of brain.db.
# These are operational/telemetry tables that grow unbounded and
# don't need referential integrity with the knowledge graph.
LOG_TABLES = {
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

    'signal_queue': {
        'create': """CREATE TABLE IF NOT EXISTS signal_queue (
            id TEXT PRIMARY KEY,
            producer TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            priority REAL DEFAULT 0.5,
            content TEXT NOT NULL,
            content_chars INTEGER DEFAULT 0,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            ttl_seconds INTEGER,
            times_surfaced INTEGER DEFAULT 0,
            max_surfaces INTEGER,
            last_surfaced_at TEXT,
            cooldown_seconds INTEGER,
            dismissed INTEGER DEFAULT 0,
            preempt INTEGER DEFAULT 0
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
    'trace_events': {
        'create': """CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    # signal_queue
    'CREATE INDEX IF NOT EXISTS idx_sq_priority ON signal_queue(dismissed, priority DESC)',
    'CREATE INDEX IF NOT EXISTS idx_sq_producer ON signal_queue(producer)',
    # v9.2: session_state
    'CREATE INDEX IF NOT EXISTS idx_session_state_session ON session_state(session_id)',
]


def ensure_logs_schema(conn):
    """Create all log tables in the logs database (brain_logs.db).

    Also handles column migrations for existing tables via ALTER TABLE.
    """
    conn.execute('PRAGMA journal_mode=WAL')
    for table_name, spec in LOG_TABLES.items():
        conn.execute(spec['create'])

    _add_column_if_missing(conn, 'trace_events', 'interaction_id', 'INTEGER')

    for idx in LOG_INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass
    conn.commit()


def _add_column_if_missing(conn, table: str, column: str, col_type: str):
    """Add a column to an existing table if it doesn't exist.
    SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we catch the error."""
    try:
        conn.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table, column, col_type))
    except Exception:
        pass  # Column already exists — expected for tables created with latest schema


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
