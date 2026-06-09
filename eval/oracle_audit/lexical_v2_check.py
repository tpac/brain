#!/usr/bin/env python3
"""LEXICAL-V2 LANE CHECK (porter-bm25) — proves the arm-gated lexical lane: (1) control
byte-identical, (2) porter search_scored returns real bm25 scores for entity terms, (3) Group-A
activation lifts the literal-match node. Daemon-safe (IsolatedBrain). Reports numbers, no asserts.

NOTE: recall's 5s result cache is NOT keyed by BRAIN_RECALL_ARM, so we clear it between arm flips
(the real eval runs each arm in its own process). Usage: ./dev python3 eval/oracle_audit/lexical_v2_check.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

CORPUS = {q['id']: q for q in json.load(open(f'{ROOT}/eval/oracle_audit/recall_corpus_v2.json'))['queries']}


def _clear(b):
    if hasattr(b, '_recall_cache'):
        try: b._recall_cache.clear()
        except Exception: pass


def top_ids(b, query, n=25):
    out = b.recall(query=query, limit=n)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    return [(r.get('id') or r.get('node_id')) for r in res]


def rank(ids, nid):
    p = [i[:8] for i in ids]
    return (p.index(nid) + 1) if nid in p else None


with IsolatedBrain() as env:
    b = env.brain
    L1 = CORPUS['L1']['query_rich']
    C2 = CORPUS['C2']['query_rich']

    os.environ.pop('BRAIN_RECALL_ARM', None)
    print("=== (1) CONTROL byte-identical (arm unset) ===")
    _clear(b); ctrl_L1 = top_ids(b, L1)
    _clear(b); ctrl_C2 = top_ids(b, C2)
    _clear(b); ctrl_L1b = top_ids(b, L1)
    print("  control L1 stable across runs:", ctrl_L1 == ctrl_L1b)

    print("\n=== (2) porter search_scored returns real bm25 scores for entity terms ===")
    for tok in ('ex.co', 'springserve', 'multicall'):
        hits = b._fts.search_scored(tok, 10)
        print("  '%-12s' -> %d scored hits, top relevance=%s"
              % (tok, len(hits), (round(hits[0][1], 2) if hits else None)))

    print("\n=== (3) ARM A activation (additive porter-bm25) ===")
    os.environ['BRAIN_RECALL_ARM'] = 'A'
    _clear(b); a_L1 = top_ids(b, L1)
    _clear(b); a_C2 = top_ids(b, C2)
    os.environ.pop('BRAIN_RECALL_ARM', None)
    print("  L1 8359cf1d (literal SpringServe/dynamic-pricing) rank — control: %s  armA: %s"
          % (rank(ctrl_L1, '8359cf1d'), rank(a_L1, '8359cf1d')))
    print("  L1 62a9f30d (multicall OFF)               rank — control: %s  armA: %s"
          % (rank(ctrl_L1, '62a9f30d'), rank(a_L1, '62a9f30d')))
    new_in_L1 = [i[:8] for i in a_L1 if i[:8] not in {x[:8] for x in ctrl_L1}]
    print("  armA admitted into L1 top-25 (not in control):", new_in_L1 or 'none')
    print("  C2 control top-5 stable (armA vs control):",
          [i[:8] for i in ctrl_C2[:5]] == [i[:8] for i in a_C2[:5]],
          "(answer-presence, not rank-match, is the real metric — eval/longmem)")
