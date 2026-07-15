"""Walker phase 2 — fill turn vectors. INCREMENTAL by design (Tom 2026-07-14:
never redo what has a vector).

Vectors are never recomputed here if the substrate holds them:
  • op_vec / anchor_vec fill from `trace_embeddings` by trace id — the SAME
    stored vectors live episodic scoring uses (measured==shipped).
  • Turns whose trace simply isn't drained yet (April backlog, worker window
    opened 2026-07-14) stay NULL, are COUNTED as pending, and fill on the
    next run — this phase never wipes, only fills.
  • The ONLY local embedding is for turns that will NEVER appear in the
    store: untraced_legacy micro-turns (the prompt has no s0 row — only the
    O row preserved it; pre-2026-06-08 Stop-time write, see extract.py).
    Rendered with the worker's exact recipe ('%s: %s' % (identity, text),
    document prefix, same model) and flagged op_vec_source='local_untraced'
    so the sweep can run a with/without sensitivity arm (live recall's trace
    matrix won't hold these either).

Run:  ./dev python3 eval/laf/walker/embed.py
"""
import json
import sys
from pathlib import Path

from walker_db import open_walker, open_logs_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

HUMAN_IDENTITY = 'Tom'   # matches trace metadata human_identity (verified: zero
                         # unembedded dialogue rows lack identity, 2026-07-14)
FETCH_BATCH = 500


def fill_from_store(walker, logs, col, tid_col, c, stamp_source=False):
    pending = walker.execute(
        "SELECT rowid, %s FROM turns WHERE %s IS NULL AND %s IS NOT NULL"
        % (tid_col, col, tid_col)).fetchall()
    sql = ("UPDATE turns SET %s=?, op_vec_source='store' WHERE rowid=?" % col
           if stamp_source else "UPDATE turns SET %s=? WHERE rowid=?" % col)
    filled = 0
    for i in range(0, len(pending), FETCH_BATCH):
        batch = pending[i:i + FETCH_BATCH]
        tids = [t for _, t in batch]
        rows = logs.execute(
            "SELECT trace_id, vector FROM trace_embeddings WHERE trace_id IN (%s)"
            % ','.join('?' * len(tids)), tids).fetchall()
        vec_of = {tid: vec for tid, vec in rows}
        updates = [(vec_of[t], rowid) for rowid, t in batch if t in vec_of]
        walker.executemany(sql, updates)
        filled += len(updates)
    c[col + '_filled_store'] = filled
    c[col + '_pending_drain'] = len(pending) - filled


def embed_untraced(walker, c):
    rows = walker.execute(
        "SELECT rowid, op_text FROM turns WHERE op_vec IS NULL AND op_trace_id IS NULL"
        " AND flags LIKE '%untraced_legacy%' AND op_text != ''").fetchall()
    if not rows:
        c['op_vec_local_untraced'] = 0
        return
    from servers import embedder
    embedder.load_model()          # fresh process — daemon loads this at boot
    texts = ['%s: %s' % (HUMAN_IDENTITY, t) for _, t in rows]
    vectors = embedder.embed_batch(texts, kind='document')
    if not vectors or len(vectors) != len(rows):
        print('FATAL: embed_batch returned %d vectors for %d texts'
              % (len(vectors) if vectors else 0, len(rows)))
        sys.exit(2)
    updates = [(vec, rowid) for (rowid, _), vec in zip(rows, vectors) if vec is not None]
    walker.executemany(
        "UPDATE turns SET op_vec=?, op_vec_source='local_untraced' WHERE rowid=?",
        updates)
    c['op_vec_local_untraced'] = len(updates)


def embed_query_side(walker, c):
    """q_vec for labeled turns: the QUERY-side vector production scores the
    prompt with — 'search_query:' prefix over op_text[:500] (the production
    recall-query cap), no speaker token. The trace vector (document-side,
    'Tom: '-prefixed full render) is a DIFFERENT point in the asymmetric-
    prefix space — reusing it for j=0 was the walker bug the replay-sanity
    check caught (maxsim-only rho 0.16 vs live). Incremental: fills NULLs."""
    rows = walker.execute(
        "SELECT rowid, op_text FROM turns WHERE labeled=1 AND q_vec IS NULL"
        " AND op_text != ''").fetchall()
    if not rows:
        c['q_vec_embedded'] = 0
        return
    from servers import embedder
    embedder.load_model()
    texts = [t[:500] for _, t in rows]
    vectors = embedder.embed_batch(texts, kind='query')
    if not vectors or len(vectors) != len(rows):
        print('FATAL: q_vec embed_batch returned %d for %d texts'
              % (len(vectors) if vectors else 0, len(rows)))
        sys.exit(2)
    updates = [(v, r) for (r, _), v in zip(rows, vectors) if v is not None]
    walker.executemany("UPDATE turns SET q_vec=? WHERE rowid=?", updates)
    c['q_vec_embedded'] = len(updates)          # actual writes, not attempts (review F6)
    c['q_vec_embed_failed'] = len(rows) - len(updates)


def main():
    walker = open_walker()
    logs = open_logs_ro()
    c = {}
    # op_vec pass carries op_vec_source='store'; anchor has no source column
    # (store is its only source — untraced turns have no anchor at all).
    fill_from_store(walker, logs, 'op_vec', 'op_trace_id', c, stamp_source=True)
    fill_from_store(walker, logs, 'anchor_vec', 'anchor_trace_id', c)
    embed_untraced(walker, c)
    embed_query_side(walker, c)
    c['op_vec_still_null'] = walker.execute(
        "SELECT count(*) FROM turns WHERE op_vec IS NULL").fetchone()[0]
    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('embed_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()
    walker.close()
    logs.close()
    print('embed phase — counters:')
    for k in sorted(c):
        print('  %-28s %d' % (k, c[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
