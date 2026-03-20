"""
Migration 002: Initial logs schema — all tables and indexes for brain_logs.db.

This captures the LOG_TABLES schema at the point of introducing the numbered
migration system. Uses IF NOT EXISTS so it's safe for existing databases.
"""

import sqlite3

description = "Create all brain_logs.db tables and indexes (baseline)"


# ── Table definitions (frozen snapshot) ──

TABLES = [
    """CREATE TABLE IF NOT EXISTS access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )""",

    """CREATE TABLE IF NOT EXISTS debug_log (
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

    """CREATE TABLE IF NOT EXISTS recall_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        returned_ids TEXT NOT NULL,
        returned_count INTEGER NOT NULL,
        used_ids TEXT,
        used_count INTEGER DEFAULT 0,
        precision_score REAL,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS miss_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        signal TEXT NOT NULL,
        query TEXT,
        expected_node_id TEXT,
        context TEXT,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS dream_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        intuition_node_id TEXT,
        seed_nodes TEXT NOT NULL,
        walk_path TEXT NOT NULL,
        insight TEXT,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS tuning_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter TEXT NOT NULL,
        old_value TEXT NOT NULL,
        new_value TEXT NOT NULL,
        reason TEXT,
        eval_snapshot_id INTEGER,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS eval_snapshots (
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

    """CREATE TABLE IF NOT EXISTS suggest_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        context TEXT,
        suggested_ids TEXT,
        accepted_ids TEXT,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS curiosity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        gap_type TEXT NOT NULL,
        target_node_id TEXT,
        prompt TEXT NOT NULL,
        resolved INTEGER DEFAULT 0,
        resolved_at TEXT,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS health_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        check_type TEXT NOT NULL,
        result TEXT,
        actions_taken TEXT,
        created_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS staged_learnings (
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
]

INDEXES = [
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
]


def up(conn: sqlite3.Connection) -> None:
    """Create all brain_logs.db tables and indexes."""
    # Note: WAL mode must be set outside of transactions.
    # The migration runner handles PRAGMA journal_mode=WAL separately.
    for sql in TABLES:
        conn.execute(sql)
    for sql in INDEXES:
        try:
            conn.execute(sql)
        except Exception:
            pass


def down(conn: sqlite3.Connection) -> None:
    """Drop all log tables (for testing only)."""
    table_names = [
        'staged_learnings', 'health_log', 'curiosity_log', 'suggest_log',
        'eval_snapshots', 'tuning_log', 'dream_log', 'miss_log',
        'recall_log', 'debug_log', 'access_log',
    ]
    for name in table_names:
        conn.execute("DROP TABLE IF EXISTS %s" % name)
