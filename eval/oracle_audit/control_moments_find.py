#!/usr/bin/env python3
"""MOMENT FINDER — surfaces real S0 conversation moments to write the control-fails questions FROM
(grounded in actual turns, not invented from nodes). Stratifies across the modes Tom named so the
30 questions span easy->hard:
  TRIGGER  — procedural turns ("let's commit", "push", "next") -> should the brain surface process memory?
  TOPIC    — entity/subject turns (ex.co, adcp, recall, encoder...) -> the right cluster
  HEAVY    — dense-discussion turns (many nearby nodes / thick community) -> precision under abundance
  REMOTE   — turns whose nearest node is ISOLATED (low degree) -> reach (can recall even find it?)
  EPISODE  — oldest turns ("the research a while back") -> temporal/episodic reach

Per candidate: session, date, role, text snippet, nearest node (title + DEGREE + community size).
Daemon-safe (IsolatedBrain). Dumps moments_candidates.json + prints the menu I author from.
Usage: ./dev python3 eval/oracle_audit/control_moments_find.py
"""
import sys, json, re, random
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402

random.seed(7)
SAMPLE = 500
TRIGGER = re.compile(r"\b(commit|push|ship|deploy|merge|let'?s|next|wrap|should we|kk|go ahead|done)\b", re.I)
ENTITY = re.compile(r"(ex\.?co|adcp|ad context|springserve|multicall|shachar|kevel|vast|prebid|"
                    r"\bgam\b|ctv|yield|recall|encoder|surface|daemon|fatigue|community|trace)", re.I)


def _norm(blobs):
    vs, idx = [], []
    for i, b in enumerate(blobs):
        if not b:
            continue
        v = np.frombuffer(b, dtype=np.float32); n = np.linalg.norm(v)
        if n:
            vs.append(v / n); idx.append(i)
    return (np.vstack(vs) if vs else np.zeros((0, 768), np.float32)), idx


with IsolatedBrain() as env:
    b = env.brain
    model = embedder.stats.get('model_name') or None
    nrows = b._vec_dal.get_all_vectors(vector_types=['_primary'], model=model)
    node_mat, nidx = _norm([r['embedding'] for r in nrows])
    node_ids = [nrows[i]['node_id'] for i in nidx]

    def ntitle(nid):
        r = b.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return (r[0] if r else '?')[:48]

    def degree(nid):
        return b.conn.execute("SELECT count(*) FROM edges WHERE source_id=? OR target_id=?", (nid, nid)).fetchone()[0]

    trows = b.logs_conn.execute(
        """SELECT te.trace_id, te.vector, te.text, ev.ref_type, ev.session_id, ev.created_at
           FROM trace_embeddings te LEFT JOIN trace_events ev ON ev.id = te.trace_id""").fetchall()
    s0 = [r for r in trows if r[1] and r[3] in ('user_message', 'assistant_message')]
    if len(s0) > SAMPLE:
        s0 = random.sample(s0, SAMPLE)
    tmat, tidx = _norm([r[1] for r in s0])
    meta = [{'tid': s0[i][0], 'text': (s0[i][2] or ''), 'role': s0[i][3], 'sess': (s0[i][4] or '?')[:8],
             'date': (s0[i][5] or '?')[:10]} for i in tidx]

    cand = []
    for j, m in enumerate(meta):
        tv = tmat[j]
        sims = node_mat @ tv
        top = int(np.argmax(sims))
        ncos = float(sims[top])
        density = int((sims > 0.55).sum())
        nn = node_ids[top]
        m.update({'ncos': round(ncos, 2), 'density': density, 'near': nn,
                  'trig': bool(TRIGGER.search(m['text'])), 'ent': bool(ENTITY.search(m['text']))})
        cand.append(m)

    today = max(c['date'] for c in cand)

    def show(tag, items, k=6):
        print("\n── %s ──" % tag)
        out = []
        for c in items[:k]:
            d = degree(c['near'])
            csize = 0
            try:
                comms = b._graph.get_communities_for([c['near']]) or {}
                cl = comms.get(c['near']) if isinstance(comms, dict) else comms
                if cl:
                    cid = cl[0]['id'] if isinstance(cl[0], dict) else cl[0]
                    csize = len(b._graph.get_community_members(cid) or [])
            except Exception:
                csize = 0
            c['near_deg'] = d; c['near_comm'] = csize
            print("  [%s %s %-9s] %s" % (c['sess'], c['date'], c['role'][:4], c['text'][:74].replace('\n', ' ')))
            print("        ~%s (%s) deg=%d comm=%d  dens=%d cos=%.2f"
                  % (c['near'][:8], ntitle(c['near']), d, csize, c['density'], c['ncos']))
            out.append(c)
        return out

    users = [c for c in cand if c['role'] == 'user_message']
    picked = {}
    picked['TRIGGER'] = show('TRIGGER (procedural user turns)', [c for c in users if c['trig'] and c['ncos'] > 0.4][:20])
    picked['TOPIC'] = show('TOPIC (entity/subject user turns)', sorted([c for c in users if c['ent']], key=lambda x: -x['density'])[:20])
    picked['HEAVY'] = show('HEAVY (dense discussion)', sorted(cand, key=lambda x: -x['density'])[:20])
    picked['REMOTE'] = show('REMOTE (nearest node isolated)', sorted([c for c in cand if c['ncos'] > 0.5], key=lambda x: degree(x['near']))[:20])
    picked['EPISODE'] = show('EPISODE (oldest turns)', sorted(users, key=lambda x: x['date'])[:20])

    with open(f'{ROOT}/eval/oracle_audit/moments_candidates.json', 'w') as f:
        json.dump({k: v for k, v in picked.items()}, f, indent=2, default=str)
    print("\nwrote moments_candidates.json — author 30 questions across these modes")
