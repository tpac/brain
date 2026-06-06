#!/usr/bin/env python3
"""WHY is query #11 buried? — mechanistic probe (no recall changes).

Phase-1b of docs/HANDOFF-RECALL-NORMALIZATION.md. The diagnostic showed #11
("what did we do on the last session we work on ex.co?") is the one live
burial: nearest EX.CO node at rank 36 (cosine 0.68), invisible to the ~30-cand
surfacer. This probe answers WHICH mechanism, by looking at the actual winners:

  1. Dump #11's top-35 with rank / score / cosine / DEGREE / source / is_EXCO / title.
     -> degree-driven (hubs win)  => hub-dampening is the lever
     -> cosine-driven (closer win) => z-score / contrastive is the lever
     -> topic-irrelevant winners   => query-dilution (the discriminating token is diluted)
  2. FTS5 lexical-bridge test: does "ex.co" (+ variants) match the EX.CO nodes at all?
     (#11 contains "ex.co" yet no EX.CO node surfaced via fts5 — tokenization?)
  3. Query-dilution test: re-run with ex.co-FOCUSED queries; if EX.CO leaps up,
     the full query's generic "last session / work" terms are diluting the signal.

Usage: ./dev python3 eval/oracle_audit/burial_why.py   (isolated copy, never live)
"""
import os
import sys
import json

ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(os.path.join(ROOT, 'eval/oracle_audit/meshed_top10.json')))
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}

Q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)
Q2 = next(it['prompt'] for it in CORPUS if it['rank'] == 2)


def _bust(b):
    if hasattr(b, '_recall_cache'):
        try:
            b._recall_cache.clear()
        except Exception:
            pass


def rows(brain, query, limit, deg):
    _bust(brain)
    out = brain.recall(query=query, limit=limit)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    r = []
    for i, n in enumerate(res, 1):
        nid = (n.get('id') or n.get('node_id') or '')[:8]
        r.append({
            'rank': i, 'id': nid,
            'score': n.get('effective_activation'),
            'cos': n.get('embedding_similarity'),
            'deg': deg.get(n.get('id') or n.get('node_id') or '', 0),
            'src': n.get('_discovery') or n.get('_source') or '?',
            'exco': nid in KNOWN_EXCO,
            'title': (n.get('title') or '')[:46],
        })
    return r


def best_exco_rank(r):
    return next((x['rank'] for x in r if x['exco']), None)


with IsolatedBrain() as env:
    brain = env.brain
    brain._ensure_structural_degree_cache()
    deg = brain._structural_degree_cache
    print("\n=== WHY #11 (isolated, %d nodes) ===" % env.node_count())

    # 1. #11 top-35 — what is actually winning?
    r11 = rows(brain, Q11, 35, deg)
    print("\n--- #11 top-35  '%s'" % Q11[:70])
    print("  %-5s %-9s %-6s %-6s %-5s %-9s %-4s %s" %
          ("rank", "id", "score", "cos", "deg", "src", "EXCO", "title"))
    for x in r11:
        print("  %-5d %-9s %-6s %-6s %-5d %-9s %-4s %s" % (
            x['rank'], x['id'],
            ("%.3f" % x['score']) if isinstance(x['score'], (int, float)) else '-',
            ("%.3f" % x['cos']) if isinstance(x['cos'], (int, float)) else '-',
            x['deg'], x['src'], 'YES' if x['exco'] else '', x['title']))

    # degree contrast: winners (top-30, non-exco) vs the buried exco nodes
    win_deg = [x['deg'] for x in r11[:30] if not x['exco']]
    import statistics as st
    if win_deg:
        print("\n  top-30 non-EXCO winners: degree median=%d mean=%.1f max=%d  (n=%d)" %
              (int(st.median(win_deg)), st.mean(win_deg), max(win_deg), len(win_deg)))

    # 2. FTS5 lexical-bridge test
    print("\n--- FTS5 bridge: does the literal token reach EX.CO nodes?")
    fts = getattr(brain, '_fts', None)
    for tok in ['ex.co', 'exco', 'EX.CO', 'ex co', '"ex.co"']:
        try:
            hits = fts.search(tok, 30) if fts else []
            hit_exco = [h[:8] for h in hits if h[:8] in KNOWN_EXCO]
            print("  %-9s -> %2d hits, %d are EX.CO  %s" %
                  (tok, len(hits), len(hit_exco), hit_exco[:6]))
        except Exception as e:
            print("  %-9s -> ERROR %s" % (tok, e))

    # 3. Query-dilution test
    print("\n--- query-dilution: best EX.CO rank under different phrasings")
    for label, q in [
        ("FULL #11", Q11),
        ("ex.co", "ex.co"),
        ("ex.co company", "ex.co company"),
        ("last session ex.co work", "last session we worked on ex.co"),
        ("ex.co ad server", "ex.co ad server CTV"),
    ]:
        rr = rows(brain, q, 100, deg)
        ber = best_exco_rank(rr)
        n_top30 = sum(1 for x in rr[:30] if x['exco'])
        print("  %-26s best EX.CO rank=%-5s  EX.CO in top30=%d" %
              (label, str(ber or '—'), n_top30))

    # contrast: #2 (the healthy one) top-12
    r2 = rows(brain, Q2, 12, deg)
    print("\n--- #2 (HEALTHY contrast) top-12  '%s'" % Q2[:55])
    print("  %-5s %-9s %-6s %-5s %-4s %s" % ("rank", "id", "cos", "deg", "EXCO", "title"))
    for x in r2:
        print("  %-5d %-9s %-6s %-5d %-4s %s" % (
            x['rank'], x['id'],
            ("%.3f" % x['cos']) if isinstance(x['cos'], (int, float)) else '-',
            x['deg'], 'YES' if x['exco'] else '', x['title']))
