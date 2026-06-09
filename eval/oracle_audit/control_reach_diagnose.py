#!/usr/bin/env python3
"""REACH DIAGNOSIS — for the essential gold Control MISSED (not in top-5), classify HOW recoverable it
is, to decide if the compositional gap is real or already-closed-downstream:
  1hop    = a graph-neighbor of one of Control's top-5 hits  -> the brain's spread activation (Layer 3,
            post-surface) likely ALREADY surfaces it -> gap is partly a pre-spread measurement artifact
  fts     = keyword search finds it                          -> lexical-reachable
  2hop    = neighbor-of-neighbor                             -> deeper spread would reach
  UNREACH = none of the above                                -> a REAL recall gap needing a new mechanism

Reads gold from control_corpus.json, reruns Control time-scoped. Daemon-safe (IsolatedBrain).
Usage: ./dev python3 eval/oracle_audit/control_reach_diagnose.py
"""
import sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


with IsolatedBrain() as env:
    b = env.brain

    def neigh(n8):
        rows = b.conn.execute("SELECT target_id FROM edges WHERE source_id LIKE ? "
                              "UNION SELECT source_id FROM edges WHERE target_id LIKE ?",
                              (n8 + '%', n8 + '%')).fetchall()
        return {x[0][:8] for x in rows}

    tally = {'1hop': 0, 'fts': 0, '2hop': 0, 'UNREACH': 0}
    total_missed = 0
    per = []
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        cutoff = cutoff_for(q)
        elig = None
        if cutoff:
            elig = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (cutoff,)).fetchall()}
        filt = {"created_at": {"lte": cutoff}} if cutoff else None
        try:
            out = b.recall(query=q['query'], limit=25, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=25)
        ctrl = [(r.get('id') or r.get('node_id'))[:8]
                for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
        if elig is not None:
            ctrl = [n for n in ctrl if n in elig]
        top5 = ctrl[:5]
        missed = [n for n in ess if n not in set(ctrl[:5])]
        if not missed:
            continue
        hop1 = set()
        for s in top5:
            hop1 |= neigh(s)
        hop2 = set()
        for n in list(hop1)[:60]:
            hop2 |= neigh(n)
        fts = {n[:8] for n, _ in b._fts.search_scored(q['query'], 25)}
        cls = {'1hop': 0, 'fts': 0, '2hop': 0, 'UNREACH': 0}
        for n in missed:
            total_missed += 1
            if n in hop1:
                tally['1hop'] += 1; cls['1hop'] += 1
            elif n in fts:
                tally['fts'] += 1; cls['fts'] += 1
            elif n in hop2:
                tally['2hop'] += 1; cls['2hop'] += 1
            else:
                tally['UNREACH'] += 1; cls['UNREACH'] += 1
        per.append((q['id'], q['mode'], len(missed), cls))
        print("#%-4s %-8s missed=%d  1hop=%d fts=%d 2hop=%d UNREACH=%d"
              % (q['id'], q['mode'], len(missed), cls['1hop'], cls['fts'], cls['2hop'], cls['UNREACH']))

    print("\n=== of %d MISSED essential nodes, how recoverable ===" % total_missed)
    for k in ('1hop', 'fts', '2hop', 'UNREACH'):
        print("  %-8s %3d  (%.0f%%)" % (k, tally[k], 100.0 * tally[k] / max(total_missed, 1)))
    spread_reach = tally['1hop'] + tally['2hop']
    print("  => graph-reachable (1hop+2hop, spread-activation territory): %d (%.0f%%)"
          % (spread_reach, 100.0 * spread_reach / max(total_missed, 1)))
    print("  => TRULY UNREACHABLE (real recall gap): %d (%.0f%%)"
          % (tally['UNREACH'], 100.0 * tally['UNREACH'] / max(total_missed, 1)))
