"""Episodic substrate rebuild for the composition test (Tom, 2026-07-18) —
computes the two new episodic channels for ALL labeled turns x candidates and
caches them in a versioned table, so the composition fit and all downstream
refinement (gain sweeps, hop/decay variants) run cheaply on top.

Channels at the current-message cue (op j0 = q_vec):
  pick_rec    picked u dropped (recalled ~25)   [Move 1]
  enc_graph   encoded u 1-hop graph neighbors   [Tom's graph-hop]

Non-destructive: writes a NEW table cand_turn_episodic_ext, leaves the
production cand_turn_episodic (pick_sel/enc) untouched.

Params (the iteration knobs — re-run to sweep a variant):
  --hop N        graph-hop depth for enc_graph (default 1)
  --decay F      neighbor score = source * F per hop (default 1.0)

Run:  ./dev python3 eval/laf/walker/episodic_ext_build.py
Pool60: BRAIN_DB_DIR=.../0a9baa/pooled WALKER_OUT_DIR=.../0a9baa/walker ./dev python3 ...
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit                   # noqa: E402
from episodic_roles import (build_role_map, roles_at, _stop_of,      # noqa: E402
                            TOP_MOMENTS, WINDOW)
from tests.isolated_brain import IsolatedBrain                       # noqa: E402

EXT_VERSION = 'ext-v1-hop1'


def _arg(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv \
        else default


def channels(eng, role_map, sims, as_of, trace_created, adj, hop, decay):
    """{'pick_rec': {row:score}, 'enc_graph': {row:score}} for one query."""
    sims = np.where(trace_created <= as_of, sims, -np.inf)
    k = min(TOP_MOMENTS * 3, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    moments = {}
    for i in top:
        chain = eng._tr_meta[i][0] or ''
        stop = _stop_of(chain)
        parts = chain.split('-')
        sess = eng._tr_meta[i][1]
        if stop is None or not sess or len(parts) < 3:
            continue
        key = (sess, parts[1], stop)
        s = float(sims[i])
        if s > moments.get(key, 0.0):
            moments[key] = s
        if len(moments) >= TOP_MOMENTS:
            break
    prec = {}
    enc_ids = {}
    for (sess, _short, stop), s in moments.items():
        rec = role_map.get(sess)
        if rec is None:
            continue
        picked, encoded, dropped = set(), set(), set()
        for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
            p, e, d = roles_at(rec, ws, as_of)
            picked |= p
            encoded |= e
            dropped |= d
        for nid in (picked | dropped):
            r = eng._resolve(nid)
            if r is not None:
                prec[r] = max(prec.get(r, 0.0), s)
        for nid in encoded:
            enc_ids[nid] = max(enc_ids.get(nid, 0.0), s)
    # graph-hop: spread enc across hop levels with decay
    egraph = {}
    for nid, s in enc_ids.items():
        r = eng._resolve(nid)
        if r is not None:
            egraph[r] = max(egraph.get(r, 0.0), s)
    frontier = dict(enc_ids)
    for _ in range(hop):
        nxt = {}
        for nid, s in frontier.items():
            for nb in adj.get(nid, ()):
                ns = s * decay
                if ns > nxt.get(nb, 0.0):
                    nxt[nb] = ns
        for nid, s in nxt.items():
            r = eng._resolve(nid)
            if r is not None and s > egraph.get(r, 0.0):
                egraph[r] = s
        frontier = nxt
    return prec, egraph


def main():
    hop = _arg('--hop', 1, int)
    decay = _arg('--decay', 1.0, float)
    walker = open_walker()
    turn_q = {}
    for s, e, q, qv, ts in walker.execute(
            "SELECT session_id, epoch, seq, q_vec, ts FROM turns "
            "WHERE labeled=1"):
        turn_q[(s, e, q)] = (_unit(qv) if qv else None, ts)
    cand_by_turn = defaultdict(list)
    for s, e, q, nid in walker.execute(
            "SELECT c.session_id, c.epoch, c.seq, c.node_id FROM candidates c "
            "JOIN turns t ON t.session_id=c.session_id AND t.epoch=c.epoch "
            " AND t.seq=c.seq WHERE t.labeled=1 AND c.node_id IS NOT NULL"):
        cand_by_turn[(s, e, q)].append(nid)

    walker.execute('DROP TABLE IF EXISTS cand_turn_episodic_ext')
    walker.execute(
        'CREATE TABLE cand_turn_episodic_ext ('
        ' session_id TEXT, epoch INTEGER, seq INTEGER, node_id TEXT,'
        ' pick_rec REAL, enc_graph REAL,'
        ' PRIMARY KEY (session_id, epoch, seq, node_id))')

    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        role_map = build_role_map(env.brain)
        adj = defaultdict(set)
        for s, t in env.brain.conn.execute(
                "SELECT e.source_id, e.target_id FROM edges e "
                "JOIN nodes ns ON ns.id=e.source_id "
                "JOIN nodes nt ON nt.id=e.target_id "
                "WHERE ns.archived=0 AND nt.archived=0"):
            adj[s].add(t)
            adj[t].add(s)

        buf, done = [], 0
        for key in sorted(cand_by_turn):
            qv, ts = turn_q.get(key, (None, None))
            if qv is None:
                continue
            prec, egraph = channels(eng, role_map, trace_mat @ qv, ts,
                                    trace_created, adj, hop, decay)
            for nid in cand_by_turn[key]:
                r = eng._resolve(nid)
                pr = prec.get(r, 0.0) if r is not None else 0.0
                eg = egraph.get(r, 0.0) if r is not None else 0.0
                buf.append((key[0], key[1], key[2], nid, pr, eg))
            done += 1
            if len(buf) >= 20000:
                walker.executemany(
                    'INSERT OR REPLACE INTO cand_turn_episodic_ext VALUES '
                    '(?,?,?,?,?,?)', buf)
                walker.commit()
                buf = []
        if buf:
            walker.executemany(
                'INSERT OR REPLACE INTO cand_turn_episodic_ext VALUES '
                '(?,?,?,?,?,?)', buf)
        walker.execute("INSERT OR REPLACE INTO build_meta VALUES "
                       "('episodic_ext_version', ?)",
                       ('%s-hop%d-decay%g' % (EXT_VERSION, hop, decay),))
        walker.commit()
        nz_pr = walker.execute("SELECT COUNT(*) FROM cand_turn_episodic_ext "
                               "WHERE pick_rec>0").fetchone()[0]
        nz_eg = walker.execute("SELECT COUNT(*) FROM cand_turn_episodic_ext "
                               "WHERE enc_graph>0").fetchone()[0]
        tot = walker.execute("SELECT COUNT(*) FROM "
                             "cand_turn_episodic_ext").fetchone()[0]
        print('built cand_turn_episodic_ext: %d turns, %d rows '
              '(hop=%d decay=%g)' % (done, tot, hop, decay))
        print('  pick_rec nonzero %d (%.0f%%) · enc_graph nonzero %d (%.0f%%)'
              % (nz_pr, 100 * nz_pr / max(1, tot), nz_eg,
                 100 * nz_eg / max(1, tot)))
    walker.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
