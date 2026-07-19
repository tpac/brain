"""Comprehensive episodic-reach decision test (Tom, 2026-07-18: "do all the
testing... perhaps some combination between the two episodic").

Four episodic channels, scored per candidate by moment similarity, support-z:
  pick_sel   surfaced (Haiku picks)            [production]
  pick_rec   picked u dropped (recalled ~25)   [Move 1]
  enc        encoded (created u revised)       [production]
  enc_graph  enc u 1-hop graph neighbors       [Tom's graph-hop]
plus unions pick_rec|enc_graph and all-four.

The bar is INDEPENDENCE FROM COSINE: reach@5 on the CURRENT-MISS golds (the
ones the maxsim field buried) — not the easy ones. Also complementarity on
current-miss (does enc_graph reach golds pick_rec doesn't?) and support.

Run:  ./dev python3 eval/laf/walker/episodic_reach_probe.py
Pool60: BRAIN_DB_DIR=.../0a9baa/pooled WALKER_OUT_DIR=.../0a9baa/walker ./dev python3 ...
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (LafV1Engine, _unit, _zscore,         # noqa: E402
                                _zscore_support)
from q1_sweep import load, gate_provenance                          # noqa: E402
from episodic_roles import (build_role_map, roles_at, _stop_of,      # noqa: E402
                            TOP_MOMENTS, WINDOW)
from tests.isolated_brain import IsolatedBrain                       # noqa: E402

CH = ['pick_sel', 'pick_rec', 'enc', 'enc_graph']


def gold_flags(turns):
    """{turn_key: (gold_idx, is_current_miss)} for val gold turns
    (gold soft >= 90th pctile). current_miss = COSINE failed = the gold's
    rank under the raw current-message maxsim lane (op j0) is >=5. Corpus-
    agnostic (no fitted split); the cosine-independence bar the test needs."""
    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)
    out = {}
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any():
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] < hi:
            continue
        nc = len(td.cands)
        cz = _zscore(td.op['maxsim'][:, 0], nc)
        miss = int((cz > cz[g]).sum()) >= 5
        out[td.key] = (g, miss)
    return out, hi


def channels(eng, role_map, sims, as_of, trace_created, adj):
    """{channel: {row: score}} for one query vector."""
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
    psel, prec, enc = {}, {}, {}
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
        recalled = picked | dropped
        for nid in picked:
            r = eng._resolve(nid)
            if r is not None:
                psel[r] = max(psel.get(r, 0.0), s)
        for nid in recalled:
            r = eng._resolve(nid)
            if r is not None:
                prec[r] = max(prec.get(r, 0.0), s)
        for nid in encoded:
            r = eng._resolve(nid)
            if r is not None:
                enc[r] = max(enc.get(r, 0.0), s)
            enc_ids[nid] = max(enc_ids.get(nid, 0.0), s)
    enc_graph = dict(enc)
    for eid, s in enc_ids.items():
        for nb in adj.get(eid, ()):
            r = eng._resolve(nb)
            if r is not None and enc_graph.get(r, 0.0) < s:
                enc_graph[r] = s
    return {'pick_sel': psel, 'pick_rec': prec, 'enc': enc,
            'enc_graph': enc_graph}


def reaches(chan_scores, cands, g, eng):
    """gold in top-5 under this channel (support-z over the pool)?"""
    nc = len(cands)
    v = np.zeros(nc)
    for i, nid in enumerate(cands):
        r = eng._resolve(nid)
        if r is not None and r in chan_scores:
            v[i] = chan_scores[r]
    if v[g] == 0 and (v != 0).sum() == 0:
        return False, 0
    z = _zscore_support(v, nc)
    return int((z > z[g]).sum()) < 5, int((v != 0).sum())


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    qvec = {}
    for s, e, q, qv, ts in walker.execute(
            "SELECT session_id, epoch, seq, q_vec, ts FROM turns "
            "WHERE labeled=1"):
        qvec[(s, e, q)] = (_unit(qv) if qv else None, ts)
    walker.close()

    flags, hi = gold_flags(turns)
    tmap = {td.key: td for td in turns}

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

        # per (bucket, channel) reach counts + support; plus current-miss
        # reach sets for complementarity
        buckets = {'all': 0, 'miss': 0}
        reach = {b: {c: 0 for c in CH} for b in buckets}
        sup = {b: {c: 0 for c in CH} for b in buckets}
        miss_hit = {c: set() for c in CH}          # turn keys reached (miss)
        for key, (g, miss) in flags.items():
            td = tmap.get(key)
            qv, ts = qvec.get(key, (None, None))
            if td is None or qv is None:
                continue
            ch = channels(eng, role_map, trace_mat @ qv, ts, trace_created, adj)
            buckets['all'] += 1
            if miss:
                buckets['miss'] += 1
            for c in CH:
                r, sp = reaches(ch[c], td.cands, g, eng)
                reach['all'][c] += int(r)
                sup['all'][c] += sp
                if miss:
                    reach['miss'][c] += int(r)
                    sup['miss'][c] += sp
                    if r:
                        miss_hit[c].add(key)

        print('val gold turns %d (current-miss %d) soft hi=%.2f'
              % (buckets['all'], buckets['miss'], hi))
        for b in ('all', 'miss'):
            n = buckets[b]
            print('\n=== %s golds (n=%d) ===' % (b.upper(), n))
            print('channel       reach@5      avg support')
            for c in CH:
                print('  %-11s %4.0f%% (%d)   %.1f'
                      % (c, 100 * reach[b][c] / max(1, n), reach[b][c],
                         sup[b][c] / max(1, n)))
        # combinations on current-miss (union reach)
        nm = buckets['miss']
        u_pr_eg = len(miss_hit['pick_rec'] | miss_hit['enc_graph'])
        u_all = len(miss_hit['pick_sel'] | miss_hit['pick_rec']
                    | miss_hit['enc'] | miss_hit['enc_graph'])
        print('\n=== CURRENT-MISS combinations (union reach, n=%d) ===' % nm)
        print('  pick_rec u enc_graph  %4.0f%% (%d)' % (100 * u_pr_eg / max(1, nm), u_pr_eg))
        print('  all four              %4.0f%% (%d)' % (100 * u_all / max(1, nm), u_all))
        print('  enc_graph beyond pick_rec: %d golds  |  pick_rec beyond enc_graph: %d'
              % (len(miss_hit['enc_graph'] - miss_hit['pick_rec']),
                 len(miss_hit['pick_rec'] - miss_hit['enc_graph'])))
        print('  enc_graph beyond enc (graph-hop gain): %d'
              % len(miss_hit['enc_graph'] - miss_hit['enc']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
