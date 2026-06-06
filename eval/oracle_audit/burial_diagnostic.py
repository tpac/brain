#!/usr/bin/env python3
"""Thin-cluster burial DIAGNOSTIC — buckets WHY each EX.CO node misses.

This is the Phase-1 measurement from docs/HANDOFF-RECALL-NORMALIZATION.md (thread #0):
convert "we think it's burial" into "it's Case A / Case B / deep, by these numbers".
It does NOT change recall — pure observation on an ISOLATED brain copy.

For each EX.CO-class query it runs recall twice:
  - at the PRODUCTION limit (25): does any EX.CO node surface? (the burial symptom)
  - at a HIGH limit (so the [:limit] cut hides nothing): where does each known EX.CO
    node actually sit, and via which discovery path?

Buckets per known EX.CO node:
  OK       — surfaces within the production limit (no burial)
  CASE_A   — fts5_only / no embedding match (rides the 0.20 passthrough) AND rank > prod limit
             => killed by the [:limit] cut before the floor-bypass. Fix = fts5 reservation.
  CASE_B   — embedding match present but rank > prod limit => outranked by brain-dev hubs.
             Fix = degree/embedding normalization (hub-dampening modulator / z-score).
  DEEP     — absent even at the high limit => cosine never reached it (encode/embedding-severe).

Usage:  ./dev python3 eval/oracle_audit/burial_diagnostic.py
Never touches live: IsolatedBrain copies brain.db to a temp dir.
"""
import os
import sys
import json

ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(os.path.join(ROOT, 'eval/oracle_audit/meshed_top10.json')))

# Known EX.CO / ad-tech node ids — the "should surface" set (from the oracle + sweeps).
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}
EXCO_SRCS = {'EXCO-recall', 'B1-20'}

PROD_LIMIT = 25      # what S1R candidate recall uses
DEEP_LIMIT = 400     # high enough that the [:limit] cut hides nothing reachable


def _bust(brain):
    if hasattr(brain, '_recall_cache'):
        try:
            brain._recall_cache.clear()
        except Exception:
            pass


def recall_rows(brain, query, limit):
    """Return list of (id, title, discovery, emb_sim, score) in rank order."""
    _bust(brain)
    out = brain.recall(query=query, limit=limit)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    rows = []
    for r in res:
        nid = (r.get('id') or r.get('node_id') or '')[:8]
        rows.append((
            nid,
            (r.get('title') or '')[:48],
            r.get('_discovery') or r.get('_source') or '?',
            r.get('embedding_similarity'),
            r.get('effective_activation'),
        ))
    return rows


def index_of(rows, nid):
    for i, row in enumerate(rows, 1):
        if row[0] == nid:
            return i, row
    return None, None


def bucket(rank_prod, rank_deep, row_deep):
    if rank_deep is None:
        return 'DEEP'
    if rank_prod is not None:
        return 'OK'
    _, _, disc, emb_sim, _ = row_deep
    has_emb = bool(emb_sim) and emb_sim > 0
    if disc == 'fts5_only' or not has_emb:
        return 'CASE_A'
    return 'CASE_B'


with IsolatedBrain() as env:
    brain = env.brain
    print("\n=== BURIAL DIAGNOSTIC (isolated copy, %d nodes) ===" % env.node_count())
    print("prod_limit=%d  deep_limit=%d" % (PROD_LIMIT, DEEP_LIMIT))

    exco_items = [it for it in CORPUS if it['src'] in EXCO_SRCS]
    tally = {'OK': 0, 'CASE_A': 0, 'CASE_B': 0, 'DEEP': 0}

    for it in exco_items:
        q = it['prompt']
        print("\n" + "=" * 78)
        print("[#%d %s] %s" % (it['rank'], it['src'], q[:90].replace("\n", " ")))

        prod = recall_rows(brain, q, PROD_LIMIT)
        deep = recall_rows(brain, q, DEEP_LIMIT)

        # best EX.CO rank at production limit (the burial symptom)
        best_prod = next((i for i, row in enumerate(prod, 1) if row[0] in KNOWN_EXCO), None)
        print("  prod top-%d size=%d | best EX.CO rank @prod: %s | deep size=%d"
              % (PROD_LIMIT, len(prod), str(best_prod or '— none —'), len(deep)))

        print("  %-9s %-6s %-7s %-9s %-9s  %s" %
              ("nid", "rankP", "rankD", "disc", "emb_sim", "title"))
        for nid in sorted(KNOWN_EXCO):
            rp, _ = index_of(prod, nid)
            rd, rowd = index_of(deep, nid)
            b = bucket(rp, rd, rowd)
            tally[b] += 1
            disc = rowd[2] if rowd else '-'
            emb = rowd[3] if rowd else None
            emb_s = ("%.3f" % emb) if isinstance(emb, (int, float)) else '-'
            ttl = rowd[1] if rowd else ''
            # only print the informative rows (found somewhere) + a compact note for absent
            if rd is not None or rp is not None:
                print("  %-9s %-6s %-7s %-9s %-9s  %-7s %s"
                      % (nid, str(rp or '-'), str(rd or '-'), disc, emb_s, b, ttl))
        n_deep = sum(1 for nid in KNOWN_EXCO if index_of(deep, nid)[0])
        print("  (%d/%d known EX.CO nodes reachable within deep_limit)" % (n_deep, len(KNOWN_EXCO)))

    print("\n" + "=" * 78)
    print("BUCKET TALLY across %d EX.CO queries × %d known nodes:"
          % (len(exco_items), len(KNOWN_EXCO)))
    for k in ('OK', 'CASE_A', 'CASE_B', 'DEEP'):
        print("  %-7s %d" % (k, tally[k]))
    print("\nFix mapping: CASE_A -> fts5 reservation | CASE_B -> hub-dampening/z-score"
          " | DEEP -> PPR/encode")
