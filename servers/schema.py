"""
brain — Canonical Database Schema (v15)

SINGLE SOURCE OF TRUTH for every table, column, index, and constraint.
Ported from schema.js — identical schema, Python-native implementation.

v15 changes (brain v5 — shared cognitive space):
  - New columns on nodes: content_summary, source_attribution, scope
  - New node types: correction, validation, mental_model, reasoning_trace, uncertainty,
    purpose, mechanism, impact, constraint, convention, lesson, vocabulary
  - New tables: node_metadata, correction_traces, session_syntheses, project_maps
  - tuning_log.old_value/new_value changed from REAL to TEXT (supports JSON)

v14 changes:
  - session_activity table (replaces in-memory tracking from index.js)
  - node_embeddings.model default changes from 'bge-m3' to 'snowflake-arctic-embed-m'
  - Embedding dimension changes from 1024 to 768

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

BRAIN_VERSION = 16
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
]

NODE_TYPE_CHECK = f"CHECK(type IN ({','.join(repr(t) for t in NODE_TYPES)}))"

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

    # access_log — moved to brain_logs.db (see LOG_TABLES)

    'brain_meta': {
        'create': """CREATE TABLE IF NOT EXISTS brain_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },

    'version_history': {
        'create': """CREATE TABLE IF NOT EXISTS version_history (
            version INTEGER NOT NULL,
            migration_ts TEXT NOT NULL,
            description TEXT,
            backup_path TEXT
        )""",
        'columns': {'version': None, 'migration_ts': None, 'description': 'NULL', 'backup_path': 'NULL'}
    },

    'summaries': {
        'create': """CREATE TABLE IF NOT EXISTS summaries (
            cluster_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            node_ids TEXT NOT NULL,
            keywords TEXT,
            last_updated TEXT,
            created_at TEXT
        )""",
        'columns': {'cluster_id': None, 'title': None, 'summary': None, 'node_ids': None,
                    'keywords': 'NULL', 'last_updated': 'NULL', 'created_at': 'NULL'}
    },

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

    # dream_log — moved to brain_logs.db (see LOG_TABLES)

    # recall_log — moved to brain_logs.db (see LOG_TABLES)

    # miss_log — moved to brain_logs.db (see LOG_TABLES)

    # eval_snapshots — moved to brain_logs.db (see LOG_TABLES)

    # tuning_log — moved to brain_logs.db (see LOG_TABLES)

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

    'projects': {
        'create': """CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        'columns': {'id': None, 'name': None, 'description': 'NULL',
                    'created_at': 'NULL', 'updated_at': 'NULL'}
    },

    # suggest_log — moved to brain_logs.db (see LOG_TABLES)

    'reasoning_chains': {
        'create': """CREATE TABLE IF NOT EXISTS reasoning_chains (
            id TEXT PRIMARY KEY,
            decision_node_id TEXT,
            title TEXT NOT NULL,
            trigger_context TEXT,
            full_context TEXT,
            step_count INTEGER DEFAULT 0,
            project TEXT,
            created_at TEXT,
            FOREIGN KEY (decision_node_id) REFERENCES nodes(id) ON DELETE SET NULL
        )""",
        'columns': {'id': None, 'decision_node_id': 'NULL', 'title': None,
                    'trigger_context': 'NULL', 'full_context': 'NULL',
                    'step_count': '0', 'project': 'NULL', 'created_at': 'NULL'}
    },

    'reasoning_steps': {
        'create': """CREATE TABLE IF NOT EXISTS reasoning_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL,
            step_order INTEGER NOT NULL,
            step_type TEXT NOT NULL CHECK(step_type IN ('observation','hypothesis','attempt','evidence','failure','feedback','decision','lesson')),
            content TEXT NOT NULL,
            node_id TEXT,
            created_at TEXT,
            FOREIGN KEY (chain_id) REFERENCES reasoning_chains(id) ON DELETE CASCADE,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE SET NULL
        )""",
        'columns': {'id': None, 'chain_id': None, 'step_order': None,
                    'step_type': None, 'content': None, 'node_id': 'NULL', 'created_at': 'NULL'}
    },

    # curiosity_log — moved to brain_logs.db (see LOG_TABLES)

    # health_log — moved to brain_logs.db (see LOG_TABLES)

    # staged_learnings — moved to brain_logs.db (see LOG_TABLES)

    # debug_log — moved to brain_logs.db (see LOG_TABLES)

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

    'prune_archive': {
        'create': """CREATE TABLE IF NOT EXISTS prune_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL CHECK(item_type IN ('edge','node')),
            item_id TEXT,
            source_id TEXT,
            target_id TEXT,
            weight REAL,
            edge_type TEXT,
            description TEXT DEFAULT '',
            reason TEXT DEFAULT '',
            pruned_at TEXT NOT NULL,
            access_count INTEGER,
            stability REAL,
            emotion REAL,
            created_at TEXT,
            last_accessed TEXT
        )""",
        'columns': {'id': None, 'item_type': None, 'item_id': 'NULL',
                    'source_id': 'NULL', 'target_id': 'NULL', 'weight': 'NULL',
                    'edge_type': 'NULL', 'description': "''", 'reason': "''",
                    'pruned_at': None, 'access_count': 'NULL', 'stability': 'NULL',
                    'emotion': 'NULL', 'created_at': 'NULL', 'last_accessed': 'NULL'}
    },

    'node_embeddings': {
        'create': """CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model TEXT NOT NULL DEFAULT 'snowflake-arctic-embed-m',
            created_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {'node_id': None, 'embedding': None,
                    'model': "'snowflake-arctic-embed-m'", 'created_at': None}
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

    # v15: Node metadata sidecar — rich encoding fields without bloating nodes table
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

    # v15: Project maps — structured engineering data (file inventory, system purpose, etc.)
    'project_maps': {
        'create': """CREATE TABLE IF NOT EXISTS project_maps (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            map_type TEXT NOT NULL,
            content TEXT NOT NULL,
            last_updated TEXT,
            created_at TEXT
        )""",
        'columns': {
            'id': None, 'project': None, 'map_type': None, 'content': None,
            'last_updated': 'NULL', 'created_at': 'NULL',
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
    # summaries
    'CREATE INDEX IF NOT EXISTS idx_summaries_keywords ON summaries(keywords)',
    # node_vectors
    'CREATE INDEX IF NOT EXISTS idx_vectors_term ON node_vectors(term)',
    'CREATE INDEX IF NOT EXISTS idx_vectors_node ON node_vectors(node_id)',
    # reasoning
    'CREATE INDEX IF NOT EXISTS idx_chains_decision ON reasoning_chains(decision_node_id)',
    'CREATE INDEX IF NOT EXISTS idx_chains_project ON reasoning_chains(project)',
    'CREATE INDEX IF NOT EXISTS idx_steps_chain ON reasoning_steps(chain_id)',
    # bridge_proposals
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_status ON bridge_proposals(status)',
    'CREATE INDEX IF NOT EXISTS idx_bridge_proposals_matures ON bridge_proposals(matures_at)',
    # prune_archive
    'CREATE INDEX IF NOT EXISTS idx_prune_archive_date ON prune_archive(pruned_at)',
    'CREATE INDEX IF NOT EXISTS idx_prune_archive_type ON prune_archive(item_type)',
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
    # v15: project_maps
    'CREATE INDEX IF NOT EXISTS idx_project_maps_type ON project_maps(project, map_type)',
    # v15: nodes scope for engineering memory
    'CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope)',
    # v16: critical flag for safety-important nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_critical ON nodes(critical)',
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

    # 4. Check if nodes table needs rebuild (CHECK constraint changed)
    nodes_need_rebuild = False
    if 'nodes' in existing_tables:
        cur = conn.execute("SELECT sql FROM sqlite_master WHERE name='nodes'")
        row = cur.fetchone()
        if row:
            current_sql = row[0] or ''
            for node_type in NODE_TYPES:
                if f"'{node_type}'" not in current_sql:
                    nodes_need_rebuild = True
                    break

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
    'access_log': {
        'create': """CREATE TABLE IF NOT EXISTS access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )""",
    },
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
    'recall_log': {
        'create': """CREATE TABLE IF NOT EXISTS recall_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query TEXT NOT NULL,
            returned_ids TEXT NOT NULL,
            returned_count INTEGER NOT NULL,
            used_ids TEXT,
            used_count INTEGER DEFAULT 0,
            precision_score REAL,
            embeddings_used INTEGER DEFAULT 0,
            recalled_titles TEXT,
            recalled_snippets TEXT,
            assistant_response_snippet TEXT,
            match_method TEXT,
            evaluation_metadata TEXT,
            followup_signal TEXT,
            explicit_feedback TEXT,
            evaluated_at TEXT,
            created_at TEXT
        )""",
    },
    'miss_log': {
        'create': """CREATE TABLE IF NOT EXISTS miss_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            signal TEXT NOT NULL,
            query TEXT,
            expected_node_id TEXT,
            context TEXT,
            created_at TEXT
        )""",
    },
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
    'tuning_log': {
        'create': """CREATE TABLE IF NOT EXISTS tuning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            reason TEXT,
            eval_snapshot_id INTEGER,
            created_at TEXT
        )""",
    },
    'eval_snapshots': {
        'create': """CREATE TABLE IF NOT EXISTS eval_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            recall_precision REAL,
            recall_coverage REAL,
            dream_hit_rate REAL,
            emotion_accuracy REAL,
            avg_boot_relevance REAL,
            total_recalls INTEGER,
            total_misses INTEGER,
            total_dreams INTEGER,
            dreams_accessed INTEGER,
            recommendations TEXT,
            created_at TEXT
        )""",
    },
    'suggest_log': {
        'create': """CREATE TABLE IF NOT EXISTS suggest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            context TEXT,
            suggested_ids TEXT,
            accepted_ids TEXT,
            created_at TEXT
        )""",
    },
    'curiosity_log': {
        'create': """CREATE TABLE IF NOT EXISTS curiosity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            gap_type TEXT NOT NULL,
            target_node_id TEXT,
            prompt TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            created_at TEXT
        )""",
    },
    'health_log': {
        'create': """CREATE TABLE IF NOT EXISTS health_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            check_type TEXT NOT NULL,
            result TEXT,
            actions_taken TEXT,
            created_at TEXT
        )""",
    },
    'conflict_log': {
        'create': """CREATE TABLE IF NOT EXISTS conflict_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            hook_name TEXT NOT NULL,
            rule_node_id TEXT,
            rule_title TEXT,
            claude_action TEXT,
            brain_decision TEXT NOT NULL CHECK(brain_decision IN ('block', 'warn')),
            resolution TEXT CHECK(resolution IN ('pending', 'brain_correct', 'claude_correct', 'scoped_exception', 'dismissed')),
            operator_response TEXT,
            surfaced INTEGER DEFAULT 0,
            created_at TEXT
        )""",
        'columns': {
            'session_id': 'NULL', 'hook_name': None, 'rule_node_id': 'NULL',
            'rule_title': 'NULL', 'claude_action': 'NULL', 'brain_decision': None,
            'resolution': "'pending'", 'operator_response': 'NULL',
            'surfaced': '0', 'created_at': 'NULL'}
    },

    'staged_learnings': {
        'create': """CREATE TABLE IF NOT EXISTS staged_learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'pre_compact',
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','confirmed','dismissed','promoted')),
            confidence REAL NOT NULL DEFAULT 0.2,
            times_revisited INTEGER DEFAULT 0,
            extracted_session TEXT,
            reviewed_session TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
    },
}

LOG_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_access_log_session ON access_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_access_log_node ON access_log(node_id)',
    'CREATE INDEX IF NOT EXISTS idx_debug_session ON debug_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_debug_type ON debug_log(event_type)',
    'CREATE INDEX IF NOT EXISTS idx_debug_created ON debug_log(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_recall_log_session ON recall_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_recall_log_created ON recall_log(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_miss_log_signal ON miss_log(signal)',
    'CREATE INDEX IF NOT EXISTS idx_dream_log_session ON dream_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_curiosity_session ON curiosity_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_health_session ON health_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_staged_status ON staged_learnings(status)',
    'CREATE INDEX IF NOT EXISTS idx_staged_node ON staged_learnings(node_id)',
    'CREATE INDEX IF NOT EXISTS idx_conflict_session ON conflict_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_conflict_surfaced ON conflict_log(surfaced)',
]


def ensure_logs_schema(conn):
    """Create all log tables in the logs database (brain_logs.db)."""
    conn.execute('PRAGMA journal_mode=WAL')
    for table_name, spec in LOG_TABLES.items():
        conn.execute(spec['create'])
    for idx in LOG_INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass
    conn.commit()


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
