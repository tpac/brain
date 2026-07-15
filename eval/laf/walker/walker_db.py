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
    op_trace_id TEXT,           -- trace_events.id (HEX STRING — INTEGER affinity would coerce
                                -- ids like '6e46…' via scientific notation and mangle the join)
    anchor_trace_id TEXT,       -- trace_events.id of the assistant_message
    op_vec BLOB,                -- embed phase: trace_embeddings join, or local for untraced_legacy
                                -- (DOCUMENT-side: 'search_document:' + 'Tom: <text>' render —
                                -- the j>=1 moment-context representation, = live trace matrix)
    anchor_vec BLOB,            -- embed phase (document-side)
    q_vec BLOB,                 -- QUERY-side embedding of op_text[:500] — what production
                                -- scores the j=0 prompt with ('search_query:' prefix, no
                                -- speaker token); labeled turns only
    op_vec_source TEXT,         -- 'store' | 'local_untraced' | NULL (pending drain)
    labeled INTEGER NOT NULL DEFAULT 0,
    -- phi(M) activity features (j=0 admissibility: computed DURING the turn,
    -- so only j>=1 slots may use tool/anchor-derived features)
    op_len INTEGER, has_code INTEGER, has_question INTEGER,
    tool_result_count INTEGER, files_touched INTEGER,  -- files_touched = Edit/Write/NotebookEdit tool results (proxy)
    gap_seconds REAL, turns_since_start INTEGER,
    project TEXT,
    flags TEXT,                 -- JSON list (see extract.py taxonomy): 'untraced_legacy' (O row,
                                -- s0 never recorded — pre-2026-06-08; op_text is the 500-char O
                                -- query), 'superseded' (shares its stop and is NOT the survivor —
                                -- the turn holding the response; steering/interrupt/notification;
                                -- no_recall turns never flagged), 'text_disagree' (structurally
                                -- paired s0, op/query mismatch — never labeled),
                                -- 'no_recall' (s0 turn, no O row — register_only or hook miss)
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
    -- WIDE: one row per (labeled turn, candidate, offset j); lanes are named
    -- columns (~970k rows × 16 REAL vs 14.5M skinny rows). j indexes the turn
    -- at seq-j in the SAME epoch. Sources: _op = that turn's operator message
    -- vector/text, _anchor = its Anchor response. ADMISSIBILITY (§20.3): at
    -- j=0 only *_op lanes are usable — the turn's own anchor/activity happen
    -- AFTER the recall being replayed; the sweep enforces this.
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seq INTEGER NOT NULL,       -- the LABELED turn being scored
    node_id TEXT NOT NULL,
    j INTEGER NOT NULL,         -- 0..K turn offset (seq-j exists, same epoch)
    v_title_op REAL, v_primary_op REAL, v_high_meta_op REAL,
    v_other_meta_op REAL, v_edge_context_op REAL, v_question_op REAL,
    sit_op REAL, idf_op REAL,
    v_title_anchor REAL, v_primary_anchor REAL, v_high_meta_anchor REAL,
    v_other_meta_anchor REAL, v_edge_context_anchor REAL, v_question_anchor REAL,
    sit_anchor REAL, idf_anchor REAL,
    PRIMARY KEY (session_id, epoch, seq, node_id, j)
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
