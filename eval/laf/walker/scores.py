"""Walker phase 3 — per-(turn, candidate, offset) lane scores (§20.2).

Computes, for every LABELED turn t and candidate n, the raw lane scores
against each of the K+1 most recent turns in t's epoch (offset j: 0=the
prompt itself, 1=previous turn, ...). Every moment-shape hypothesis in the
§20.5 sweep is then pure arithmetic over this table.

Measured==shipped: cosine lanes use the SAME stored vectors production
scores with (turn vectors from trace_embeddings via the embed phase; node
view vectors from node_enrichments), decoded/normalized by recall_laf._unit;
the idf lane calls recall_laf.idf_scores (the production formula, pure) fed
AS-OF corpus state — title_tok/title_df restricted to nodes created before
turn t (per-turn df via bisect over per-token creation timestamps).

Text used for idf is capped at 500 chars — the production RECALL QUERY cap
(pipeline_contract 'user_message_query': 500), so at j=0 idf sees exactly what
production saw. Note the trace VECTORS embed the full turn render (the 500
cap on trace_embeddings.text is storage-only) — production itself carries
this query-cap vs full-turn-vector asymmetry.

INCREMENTAL: turns already scored under the current LANES_VERSION are
skipped; --rebuild wipes the table first. Missing inputs (April vectors
pending drain, nodes without a view vector) leave NULL cells and are counted.

Run:  ./dev python3 eval/laf/walker/scores.py [--rebuild]
"""
import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import (open_walker, open_brain_ro, lane_columns,
                       check_lane_schema, scores_table_ddl, lanes_version,
                       EXTRACT_VERSION as EXPECTED_EXTRACT_VERSION)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _unit, idf_scores, MAXSIM_VIEWS  # noqa: E402

K_MAX = 8
TEXT_CAP = 500                  # trace-embedding recipe cap — keep idf consistent
# Version stamp is DERIVED from the production view list (walker_db) — a config
# retune changes the stamp mechanically and the mismatch gate below demands
# --rebuild; no human has to remember to bump anything.
LANES_VERSION = lanes_version(MAXSIM_VIEWS)
VIEWS = list(MAXSIM_VIEWS) + ['_situation']   # 6 maxsim views + sit
LANE_COLS = lane_columns(MAXSIM_VIEWS)        # single-sourced (walker_db)


def load_node_vectors(braindb, node_ids):
    """{view: {node_id: unit np.array}} for the candidate node set."""
    out = {v: {} for v in VIEWS}
    ids = list(node_ids)
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        for nid, vt, blob in braindb.execute(
                "SELECT node_id, vector_type, embedding FROM node_enrichments "
                "WHERE vector_type IN (%s) AND node_id IN (%s)"
                % (','.join('?' * len(VIEWS)), ','.join('?' * len(batch))),
                VIEWS + batch):
            if blob:
                vec = _unit(blob)
                if vec is not None:
                    out[vt][nid] = vec
    return out


def build_asof_idf(braindb):
    """Structures for as-of idf: per-token sorted creation timestamps + node
    title tokens + all-node sorted timestamps. Import the production tokenizer
    pieces via recall_laf's own regex/stopwords (idf_scores tokenizes the
    QUERY side itself; the title side must match its token shape)."""
    from servers.recall_laf import _IDF_TOK
    from servers.brain_constants import _TITLE_BOOST_STOPWORDS
    node_tokens, all_created = {}, []
    token_created = defaultdict(list)
    for nid, title, created in braindb.execute(
            "SELECT id, title, created_at FROM nodes WHERE archived=0"):
        toks = frozenset(t for t in _IDF_TOK.findall((title or '').lower())
                         if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS)
        node_tokens[nid] = toks
        if created:
            all_created.append(created)
            for t in toks:
                token_created[t].append(created)
    all_created.sort()
    for lst in token_created.values():
        lst.sort()
    return node_tokens, token_created, all_created


def asof_tok_df(cand_ids, node_tokens, token_created, all_created, turn_ts):
    """(title_tok, title_df, n_titles) restricted to corpus state at turn_ts.
    title_tok rows = candidate INDEXES (idf_scores keys rows by index)."""
    title_tok = {}
    tokens_needed = set()
    for i, nid in enumerate(cand_ids):
        toks = node_tokens.get(nid, frozenset())
        title_tok[i] = toks
        tokens_needed |= toks
    title_df = {t: bisect.bisect_left(token_created[t], turn_ts)
                for t in tokens_needed if t in token_created}
    n_titles = bisect.bisect_left(all_created, turn_ts)
    return title_tok, title_df, max(n_titles, 1)


def main():
    rebuild = '--rebuild' in sys.argv
    walker = open_walker()
    # phase-consistency stamp: refuse a turns table built by a different
    # extract than this code expects (wrong-science hardening — the artifact
    # must prove its provenance, or the sweep measures the wrong thing)
    ev = walker.execute(
        "SELECT value FROM build_meta WHERE key='extract_version'").fetchone()
    if not ev or ev[0] != EXPECTED_EXTRACT_VERSION:
        raise SystemExit(
            'walker.db extract_version=%s but scores expects %s — rebuild the '
            'walker (extract.py, then embed.py) before scoring.'
            % (ev[0] if ev else 'MISSING (pre-stamp build)', EXPECTED_EXTRACT_VERSION))
    if rebuild:
        walker.execute('DROP TABLE IF EXISTS cand_turn_scores')
        walker.commit()
    walker.executescript(scores_table_ddl(MAXSIM_VIEWS))
    check_lane_schema(walker, MAXSIM_VIEWS)   # unreachable-by-design assert

    # turns: (sess, epoch) -> seq -> (op_vec, anchor_vec, op_text, anchor_text, q_vec)
    # op/anchor vecs are DOCUMENT-side (the j>=1 moment context, = live trace
    # matrix); q_vec is the QUERY-side vector production scores the j=0 prompt
    # with — the j=0 'op' source uses q_vec, never the document-side vector.
    turns = defaultdict(dict)
    for sess, epoch, seq, opv, av, opt, at, qv in walker.execute(
            "SELECT session_id, epoch, seq, op_vec, anchor_vec, op_text, anchor_text, q_vec FROM turns"):
        turns[(sess, epoch)][seq] = (
            _unit(opv) if opv else None, _unit(av) if av else None,
            (opt or '')[:TEXT_CAP], (at or '')[:TEXT_CAP],
            _unit(qv) if qv else None)

    # labeled turns + their candidates (resolved only)
    cand_by_turn = defaultdict(list)
    for sess, epoch, seq, node_id in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates WHERE node_id IS NOT NULL"):
        cand_by_turn[(sess, epoch, seq)].append(node_id)
    turn_ts_of = {(s, e, q): ts for s, e, q, ts in walker.execute(
        "SELECT session_id, epoch, seq, ts FROM turns WHERE labeled=1")}

    done = set()
    if not rebuild:
        prior_version = walker.execute(
            "SELECT value FROM build_meta WHERE key='scores_lanes_version'").fetchone()
        has_rows = walker.execute(
            "SELECT 1 FROM cand_turn_scores LIMIT 1").fetchone() is not None
        if prior_version and prior_version[0] != LANES_VERSION:
            print('FATAL: existing scores are %s, current code is %s — rerun with --rebuild'
                  % (prior_version[0], LANES_VERSION))
            return 2
        if has_rows and not prior_version:
            # rows without a stamp = an interrupted build of UNKNOWN semantics
            # (the stamp used to be written only at completion — lean review
            # 2026-07-15). Resuming over them could mix score semantics in one
            # table and then stamp it as current — refuse instead.
            print('FATAL: cand_turn_scores has rows but no scores_lanes_version '
                  'stamp (interrupted build?) — rerun with --rebuild.')
            return 2
        done = {k for k in walker.execute(
            "SELECT DISTINCT session_id, epoch, seq FROM cand_turn_scores")}
    # Stamp AT START, not completion: every committed batch below is thereby
    # provenance-covered; an interrupted run resumes under the same verified
    # stamp instead of bypassing the gate as unstamped.
    walker.execute("INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
                   ('scores_lanes_version', LANES_VERSION))
    walker.commit()

    braindb = open_brain_ro()
    all_nodes = {n for cands in cand_by_turn.values() for n in cands}
    node_vecs = load_node_vectors(braindb, all_nodes)
    node_tokens, token_created, all_created = build_asof_idf(braindb)
    braindb.close()

    c = defaultdict(int)
    c['candidate_nodes_unique'] = len(all_nodes)
    for v in VIEWS:
        c['node_vec_coverage_%s' % v.strip('_')] = len(node_vecs[v])

    ins_cols = 'session_id, epoch, seq, node_id, j, ' + ', '.join(LANE_COLS)
    ins_sql = ('INSERT OR REPLACE INTO cand_turn_scores (%s) VALUES (%s)'
               % (ins_cols, ','.join('?' * (5 + len(LANE_COLS)))))
    rows_buf = []

    for (sess, epoch, seq), cand_ids in sorted(cand_by_turn.items()):
        if (sess, epoch, seq) in done:
            c['turns_skipped_done'] += 1
            continue
        turn_ts = turn_ts_of.get((sess, epoch, seq))
        if turn_ts is None:
            c['turns_missing_ts'] += 1
            continue
        epoch_turns = turns[(sess, epoch)]
        # per-view candidate matrices (NaN row when node vector missing)
        mats = {}
        for v in VIEWS:
            m = np.full((len(cand_ids), 768), np.nan, dtype=np.float32)
            for i, nid in enumerate(cand_ids):
                vec = node_vecs[v].get(nid)
                if vec is not None:
                    m[i] = vec
            mats[v] = m
        title_tok, title_df, n_titles = asof_tok_df(
            cand_ids, node_tokens, token_created, all_created, turn_ts)

        for j in range(0, K_MAX + 1):
            src = epoch_turns.get(seq - j)
            if src is None:
                break                      # epoch boundary — achieved window ends
            op_vec, anchor_vec, op_text, anchor_text, q_vec = src
            op_j_vec = q_vec if j == 0 else op_vec
            cells = {}
            for prefix, tvec, ttext in (('op', op_j_vec, op_text),
                                        ('anchor', anchor_vec, anchor_text)):
                if tvec is not None:
                    for v in MAXSIM_VIEWS:
                        cells['v_%s_%s' % (v.strip('_'), prefix)] = mats[v] @ tvec
                    cells['sit_%s' % prefix] = mats['_situation'] @ tvec
                else:
                    c['j_missing_%s_vec' % prefix] += 1
                if ttext:
                    cells['idf_%s' % prefix] = idf_scores(
                        ttext, title_tok, title_df, len(cand_ids),
                        n_titles=n_titles)
            for i, nid in enumerate(cand_ids):
                vals = []
                for col in LANE_COLS:
                    arr = cells.get(col)
                    x = None
                    if arr is not None:
                        x = float(arr[i])
                        if x != x:         # NaN → NULL (missing node vector)
                            x = None
                    vals.append(x)
                rows_buf.append((sess, epoch, seq, nid, j, *vals))
        c['turns_scored'] += 1
        if len(rows_buf) >= 20000:
            walker.executemany(ins_sql, rows_buf)
            walker.commit()
            c['rows_written'] += len(rows_buf)
            rows_buf = []

    if rows_buf:
        walker.executemany(ins_sql, rows_buf)
        c['rows_written'] += len(rows_buf)
    # (scores_lanes_version stamped at START — see the gate block)
    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('scores_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()
    walker.close()

    print('scores phase — counters:')
    for k in sorted(c):
        print('  %-30s %d' % (k, c[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
