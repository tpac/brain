#!/usr/bin/env python3
"""SPREAD GLANCE — a read-only look at what spreading activation does to a pull.

For each query: show the recall candidates BEFORE spread (the seeds), then run
the production `_graph_expand` and show AFTER — node_activation (seeds vs newly
lit neighbors), the per-hop wave trace, and field_activation (which FIELDS of
the dimmer neighbors surfaced). This is the "collect fields from neighbors that
cousin-match, neighbors light up to a lesser degree" mechanism, made visible.

Daemon-safe: runs against an IsolatedBrain copy, never the live DB.
Usage: ./dev python3 eval/oracle_audit/spread_glance.py
"""
import os, sys
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                            # noqa: E402
from servers.scales.s1.surface import _graph_expand     # noqa: E402

QUERIES = [
    "endo surface awareness suppression — don't re-show what's already surfaced",
    "when did EX.CO first come up in our conversation",
    "spread activation recall architecture — fields from neighbors",
]
N_SEEDS = 5      # proxy for Haiku's picks
N_SHOW_BEFORE = 8


def short(s):
    return (s or '')[:8]


with IsolatedBrain() as env:
    b = env.brain

    def titles_for(ids):
        if not ids:
            return {}
        qmarks = ','.join('?' * len(ids))
        rows = b.conn.execute(
            "SELECT id, title, type FROM nodes WHERE id IN (%s)" % qmarks, list(ids)
        ).fetchall()
        return {short(r[0]): (r[1], r[2]) for r in rows}

    for q in QUERIES:
        print("\n" + "=" * 92)
        print("QUERY: %s" % q)
        print("=" * 92)

        qv = np.frombuffer(embedder.embed_query(q), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)

        out = b.recall(query=q, limit=25)
        results = out.get('results', []) if isinstance(out, dict) else (out or [])
        cand_ids = [(r.get('id') or r.get('node_id')) for r in results]
        cand_titles = {short(r.get('id') or r.get('node_id')):
                       (r.get('title', ''), r.get('type', '')) for r in results}

        seeds = cand_ids[:N_SEEDS]
        seed_short = {short(s) for s in seeds}

        print("\n── BEFORE spread — recall candidates (top %d; first %d are the spread seeds) ──"
              % (N_SHOW_BEFORE, N_SEEDS))
        for i, cid in enumerate(cand_ids[:N_SHOW_BEFORE]):
            t, ty = cand_titles.get(short(cid), ('?', '?'))
            tag = "SEED " if i < N_SEEDS else "     "
            print("  %s#%-2d [%-11s] %s" % (tag, i + 1, ty, t[:66]))

        res = _graph_expand(b, seeds, query_vec=qv)
        node_act = res.get('node_activation', {})
        field_act = res.get('field_activation', {})
        trace = res.get('trace', [])

        # titles for every node that lit up (seeds + new neighbors)
        all_lit = list(node_act.keys())
        tmap = titles_for(all_lit)
        for k, v in cand_titles.items():       # prefer recall titles where present
            tmap.setdefault(k, v)

        print("\n── AFTER spread — node_activation (sorted; SEED = started lit, +NEW = neighbor that lit up) ──")
        n_new = sum(1 for nid in node_act if short(nid) not in seed_short)
        print("   %d seeds → %d nodes activated (%d new neighbors)\n"
              % (len(seeds), len(node_act), n_new))
        for nid, act in sorted(node_act.items(), key=lambda kv: -kv[1])[:16]:
            sid = short(nid)
            t, ty = tmap.get(sid, ('?', '?'))
            tag = "SEED" if sid in seed_short else "+NEW"
            bar = '█' * int(round(act * 20))
            print("  %.3f %-4s %-22s %s" % (act, tag, bar, t[:54]))

        print("\n── the wave (per-hop trace) ──")
        for st in trace:
            print("  hop %d: new_nodes=%-3s edges_considered=%-4s transmitted=%-4s max_act=%.3f"
                  % (st.get('step', -1), st.get('new_nodes', '?'),
                     st.get('edges_considered', '?'), st.get('edges_transmitted', '?'),
                     st.get('max_act', 0.0)))

        print("\n── field_activation — which FIELDS lit up on the brightest NEW neighbors ──")
        new_sorted = [(nid, a) for nid, a in sorted(node_act.items(), key=lambda kv: -kv[1])
                      if short(nid) not in seed_short]
        for nid, act in new_sorted[:3]:
            sid = short(nid)
            t, _ = tmap.get(sid, ('?', '?'))
            fa = field_act.get(nid, {})
            top_fields = sorted(fa.items(), key=lambda kv: -kv[1])[:5]
            print("  +NEW %.3f  %s" % (act, t[:58]))
            for fname, fv in top_fields:
                print("        %-16s %.3f" % (fname, fv))
