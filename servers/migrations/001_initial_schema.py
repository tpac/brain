"""
Migration 001: Initial schema — all tables and indexes for brain.db (v15).

This captures the full schema as it exists at the point of introducing the
numbered migration system. For existing databases that already have all these
tables, the IF NOT EXISTS clauses make this a no-op.
"""

import sqlite3

description = "Create all brain.db tables and indexes (baseline v15)"

# ── Node types (frozen at the time of this migration) ──

NODE_TYPES = [
    'person', 'project', 'decision', 'rule', 'concept',
    'task', 'file', 'context', 'intuition', 'procedure',
    'thought', 'object',
    'fn_reasoning', 'param_influence', 'code_concept',
    'arch_constraint', 'causal_chain', 'bug_lesson', 'comment_anchor',
    'tension', 'hypothesis', 'pattern', 'catalyst', 'aspiration',
    'performance', 'failure_mode', 'capability', 'interaction', 'meta_learning',
    'correction', 'validation', 'mental_model', 'reasoning_trace', 'uncertainty',
    'purpose', 'mechanism', 'impact', 'constraint', 'convention', 'lesson',
    'vocabulary',
]

_NODE_TYPE_CHECK = "CHECK(type IN (%s))" % ','.join(repr(t) for t in NODE_TYPES)


# ── Table definitions ──

TABLES = [
    # brain_meta — must be created first (other logic depends on it)
    """CREATE TABLE IF NOT EXISTS brain_meta (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    )""",

    # nodes
    """CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL %s,
        title TEXT NOT NULL,
        content TEXT,
        keywords TEXT,
        activation REAL DEFAULT 1.0,
        stability REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 1,
        locked INTEGER DEFAULT 0,
        archived INTEGER DEFAULT 0,
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
    )""" % _NODE_TYPE_CHECK,

    # edges
    """CREATE TABLE IF NOT EXISTS edges (
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

    # version_history
    """CREATE TABLE IF NOT EXISTS version_history (
        version INTEGER NOT NULL,
        migration_ts TEXT NOT NULL,
        description TEXT,
        backup_path TEXT
    )""",

    # summaries
    """CREATE TABLE IF NOT EXISTS summaries (
        cluster_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        node_ids TEXT NOT NULL,
        keywords TEXT,
        last_updated TEXT,
        created_at TEXT
    )""",

    # emotion_calibration
    """CREATE TABLE IF NOT EXISTS emotion_calibration (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT,
        user_emotion REAL NOT NULL,
        user_label TEXT NOT NULL,
        context TEXT,
        created_at TEXT,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE SET NULL
    )""",

    # node_vectors (TF-IDF)
    """CREATE TABLE IF NOT EXISTS node_vectors (
        node_id TEXT NOT NULL,
        term TEXT NOT NULL,
        tf REAL NOT NULL,
        tfidf REAL DEFAULT 0,
        PRIMARY KEY (node_id, term),
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""",

    # doc_freq
    """CREATE TABLE IF NOT EXISTS doc_freq (
        term TEXT PRIMARY KEY,
        count INTEGER NOT NULL DEFAULT 1
    )""",

    # projects
    """CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT,
        updated_at TEXT
    )""",

    # reasoning_chains
    """CREATE TABLE IF NOT EXISTS reasoning_chains (
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

    # reasoning_steps
    """CREATE TABLE IF NOT EXISTS reasoning_steps (
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

    # bridge_proposals
    """CREATE TABLE IF NOT EXISTS bridge_proposals (
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

    # prune_archive
    """CREATE TABLE IF NOT EXISTS prune_archive (
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

    # node_embeddings
    """CREATE TABLE IF NOT EXISTS node_embeddings (
        node_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        model TEXT NOT NULL DEFAULT 'snowflake-arctic-embed-m',
        created_at TEXT NOT NULL,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""",

    # session_activity (v14)
    """CREATE TABLE IF NOT EXISTS session_activity (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    )""",

    # node_metadata (v15)
    """CREATE TABLE IF NOT EXISTS node_metadata (
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

    # correction_traces (v15)
    """CREATE TABLE IF NOT EXISTS correction_traces (
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

    # session_syntheses (v15)
    """CREATE TABLE IF NOT EXISTS session_syntheses (
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

    # project_maps (v15)
    """CREATE TABLE IF NOT EXISTS project_maps (
        id TEXT PRIMARY KEY,
        project TEXT NOT NULL,
        map_type TEXT NOT NULL,
        content TEXT NOT NULL,
        last_updated TEXT,
        created_at TEXT
    )""",
]


# ── Index definitions ──

INDEXES = [
    # nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_keywords ON nodes(keywords)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation DESC)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_emotion ON nodes(emotion)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope)',
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
    # node_metadata
    'CREATE INDEX IF NOT EXISTS idx_metadata_correction ON node_metadata(correction_of)',
    'CREATE INDEX IF NOT EXISTS idx_metadata_validated ON node_metadata(last_validated)',
    # correction_traces
    'CREATE INDEX IF NOT EXISTS idx_correction_traces_pattern ON correction_traces(underlying_pattern)',
    'CREATE INDEX IF NOT EXISTS idx_correction_traces_session ON correction_traces(session_id)',
    # session_syntheses
    'CREATE INDEX IF NOT EXISTS idx_session_syntheses_session ON session_syntheses(session_id)',
    # project_maps
    'CREATE INDEX IF NOT EXISTS idx_project_maps_type ON project_maps(project, map_type)',
]


def up(conn: sqlite3.Connection) -> None:
    """Create all brain.db tables and indexes."""
    for sql in TABLES:
        conn.execute(sql)

    for sql in INDEXES:
        try:
            conn.execute(sql)
        except Exception:
            # Index may reference a column that doesn't exist yet in an
            # older DB that will be patched by a later migration.
            pass


def down(conn: sqlite3.Connection) -> None:
    """Drop all tables (dangerous — for testing only)."""
    # Reverse order to respect foreign keys
    table_names = [
        'project_maps', 'session_syntheses', 'correction_traces',
        'node_metadata', 'session_activity', 'node_embeddings',
        'prune_archive', 'bridge_proposals', 'reasoning_steps',
        'reasoning_chains', 'projects', 'doc_freq', 'node_vectors',
        'emotion_calibration', 'summaries', 'version_history',
        'edges', 'nodes', 'brain_meta',
    ]
    for name in table_names:
        conn.execute("DROP TABLE IF EXISTS %s" % name)
