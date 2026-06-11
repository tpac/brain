#!/usr/bin/env python3
"""HELD-OUT validation: BRAIN_TITLE_BOOST arms on recall_corpus_v2 (10 queries × 2
phrasings, gold defined 2026-06 — NEVER seen during idf/idf2 calibration, which used
control_corpus.json only). Guards against corpus overfit before flipping the default.

Scores node_gold_primary coverage in top-5/top-25 per arm, per phrasing style
(terse vs rich), with gold_extended counted for top-5 gold density.
Usage: ./dev python3 eval/oracle_audit/ab_title_boost_v2corpus.py"""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

D = json.load(open(f'{ROOT}/eval/oracle_audit/recall_corpus_v2.json'))
QS = D['queries']
ENV = 'BRAIN_TITLE_BOOST'
ARMS = ['add', 'idf', 'idf2']
STYLES = ['query_terse', 'query_rich']


def bust(b):
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()


def run_arm(b, arm):
    os.environ[ENV] = arm
    bust(b)
    per_style = {s: {'fails': 0, 'tot': 0, 'in5': 0, 'in25': 0, 'n': 0, 'dens': []}
                 for s in STYLES}
    detail = []
    for q in QS:
        gold = [g[:8] for g in q.get('node_gold_primary', [])]
        if not gold:
            continue
        helpful = {g[:8] for g in q.get('node_gold_extended', [])} | set(gold)
        for style in STYLES:
            text = q.get(style)
            if not text:
                continue
            out = b.recall(query=text, limit=25)
            ids = [(r.get('id') or r.get('node_id'))[:8]
                   for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
            t5, t25 = set(ids[:5]), set(ids[:25])
            e5 = sum(1 for g in gold if g in t5)
            e25 = sum(1 for g in gold if g in t25)
            fail = any(g not in t5 for g in gold)
            ps = per_style[style]
            ps['fails'] += 1 if fail else 0
            ps['tot'] += len(gold); ps['in5'] += e5; ps['in25'] += e25; ps['n'] += 1
            ps['dens'].append(sum(1 for n in ids[:5] if n in helpful) / max(len(ids[:5]), 1))
            detail.append((q['id'], style.replace('query_', ''), len(gold), e5, e25, fail))
    os.environ.pop(ENV, None)
    return per_style, detail


with IsolatedBrain() as env:
    b = env.brain
    out = {arm: run_arm(b, arm) for arm in ARMS}

    print("\n=== HELD-OUT: recall_corpus_v2 (%d queries × 2 phrasings) ===" % len(QS))
    print("%-5s %-6s  %-9s  %-13s  %-13s  %-9s" % ("arm", "style", "fails/N", "top5", "top25", "dens5"))
    for arm in ARMS:
        per_style, _ = out[arm]
        for s in STYLES:
            ps = per_style[s]
            if not ps['n']:
                continue
            print("%-5s %-6s  %-9s  %-13s  %-13s  %-9s" % (
                arm, s.replace('query_', ''),
                "%d/%d" % (ps['fails'], ps['n']),
                "%d/%d (%.0f%%)" % (ps['in5'], ps['tot'], 100.0 * ps['in5'] / max(ps['tot'], 1)),
                "%d/%d (%.0f%%)" % (ps['in25'], ps['tot'], 100.0 * ps['in25'] / max(ps['tot'], 1)),
                "%.0f%%" % (100.0 * sum(ps['dens']) / max(len(ps['dens']), 1))))

    # per-query changes idf2 vs add
    base = {(d[0], d[1]): d for d in out['add'][1]}
    print("\n--- per-query changes idf2 vs add (g5 = gold_primary in top-5) ---")
    any_flip = False
    for d in out['idf2'][1]:
        qid, style, ng, e5, e25, fail = d
        b0 = base.get((qid, style))
        if b0 and (e5 != b0[3] or e25 != b0[4]):
            any_flip = True
            arrow = "↑" if e5 > b0[3] else ("↓" if e5 < b0[3] else "·")
            print("  #%-3s %-5s  g5 %d→%d %s  g25 %d→%d  (of %d gold)"
                  % (qid, style, b0[3], e5, arrow, b0[4], e25, ng))
    if not any_flip:
        print("  (no changes)")

    # smoke: stopword-only + empty-ish queries must not crash and idf2 must no-op
    print("\n--- smoke: degenerate queries under idf2 ---")
    os.environ[ENV] = 'idf2'
    for smoke_q in ("what did we do?", "the", "???"):
        bust(b)
        try:
            r = b.recall(query=smoke_q, limit=5)
            n = len(r.get('results', []) if isinstance(r, dict) else r or [])
            print("  %-18r -> ok (%d results)" % (smoke_q, n))
        except Exception as e:
            print("  %-18r -> EXCEPTION: %s" % (smoke_q, e))
    os.environ.pop(ENV, None)
