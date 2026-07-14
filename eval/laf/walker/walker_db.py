"""walker.db schema + connection helpers — §20.2.

The walker DB is a LOCAL BUILD ARTIFACT (never committed). Three data tables:

  turns            — one row per (session, stop): the moment INGREDIENTS.
                     Covers ALL conversational turns in included sessions (a
                     labeled turn's moment stack needs its unlabeled context
                     turns). op_vec/anchor_vec filled by the embed phase.
  candidates       — one row per (session, stop, candidate) for LABELED turns
                     only: outcome/tier/used-next labels + quality flags.
  cand_turn_scores — one row per (session, stop, node, turn-offset j, lane):
                     per-turn lane scores, filled by the scores phase with the
                     production field functions.

Admissibility note (leakage, §20.3): a turn row carries BOTH op_text and
anchor_text. At sweep time, M(t)'s j=0 slot may use op_text ONLY — the turn's
own anchor_text and tool activity happen AFTER the recall being replayed.
j>=1 slots use the full turn record. The schema stores; the sweep enforces.

Sources are opened READ-ONLY (mode=ro URI — the dashboard precedent; never a
writer against live brain DBs).
"""
import os
import sqlite3
from pathlib import Path

WALKER_DIR = Path(__file__).resolve().parent
WALKER_DB = WALKER_DIR / 'walker.db'

DDL = """
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,     -- stop-counter reset segment (resume/compaction boundary);
                                -- the moment stack NEVER crosses an epoch boundary
    seq INTEGER NOT NULL,       -- ts-order position within the epoch (micro-turn index);
                                -- interrupts share a stop, so seq is the true ordering key
    stop INTEGER NOT NULL,
    ts TEXT,                    -- O-row recall moment when labeled, else s0 user_message ts
    op_text TEXT,               -- full operator message (S0)
    anchor_text TEXT,           -- full Anchor response at this stop (S0; j>=1 only)
    query_stored TEXT,          -- O-row 500-char query (join-sanity witness)
    op_vec BLOB,                -- embed phase
    anchor_vec BLOB,            -- embed phase
    labeled INTEGER NOT NULL DEFAULT 0,
    -- phi(M) activity features (j=0 admissibility: computed DURING the turn,
    -- so only j>=1 slots may use tool/anchor-derived features)
    op_len INTEGER, has_code INTEGER, has_question INTEGER,
    tool_result_count INTEGER, files_touched INTEGER,  -- files_touched = Edit/Write/NotebookEdit tool results (proxy)
    gap_seconds REAL, turns_since_start INTEGER,
    project TEXT,
    flags TEXT,                 -- JSON list: 'interrupted' (O row, s0 never recorded — op_text
                                -- is the 500-char O query), 'no_recall' (s0 turn, recall failed)
    PRIMARY KEY (session_id, epoch, seq)
);
CREATE TABLE IF NOT EXISTS candidates (
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    cand_short TEXT NOT NULL,   -- 8-char id as traced
    node_id TEXT,               -- resolved full id (NULL = unresolved)
    outcome TEXT,               -- 'selected' | 'dropped' | NULL (floored never pooled)
    tier TEXT,                  -- 'picked' | 'pooled_dropped' | 'floored'
    fetched_by TEXT,            -- tool name (post-2026-07-02 rows only)
    used_next_1 INTEGER, used_next_3 INTEGER,
    rank_in_pool INTEGER, pool_score REAL,
    node_created_at TEXT, node_revised_after_turn INTEGER,
    flags TEXT,
    PRIMARY KEY (session_id, epoch, seq, cand_short)
);
CREATE TABLE IF NOT EXISTS cand_turn_scores (
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    j INTEGER NOT NULL,         -- turn offset 0..K by seq (within the SAME epoch)
    lane TEXT NOT NULL,         -- maxsim view name | 'sit' | 'idf'
    score REAL,
    PRIMARY KEY (session_id, epoch, seq, node_id, j, lane)
);
CREATE TABLE IF NOT EXISTS build_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def brain_db_dir():
    return Path(os.environ.get('BRAIN_DB_DIR') or str(Path.home() / 'AgentsContext' / 'brain'))


def open_ro(path):
    return sqlite3.connect('file:%s?mode=ro' % path, uri=True)


def open_logs_ro():
    return open_ro(brain_db_dir() / 'brain_logs.db')


def open_brain_ro():
    return open_ro(brain_db_dir() / 'brain.db')


def open_walker():
    conn = sqlite3.connect(WALKER_DB)
    conn.executescript(DDL)
    return conn


def fresh_walker():
    """Delete + recreate walker.db — the walker is a rebuildable artifact."""
    if WALKER_DB.exists():
        WALKER_DB.unlink()
    return open_walker()
