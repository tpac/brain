#!/usr/bin/env python3
"""TOP-25 DIFF — the Phase-A eval. Real pipeline, flag OFF vs ON, all 12 corpus queries.

For each query: run the REAL brain.recall(limit=25) with BRAIN_TRACE_CHAIN off, then on, and show
exactly WHAT ENTERS the 25 (the trace-chain rescues) and WHAT DROPS (the displaced baseline tail),
with titles + discovery tags so we can judge relevance. This is the deterministic, no-Haiku Phase-A
comparison (docs/DUAL-STORE-EVAL-HANDOFF.md §2 Phase A).

Never touches live (IsolatedBrain copies the DBs). Usage:
  ./dev python3 eval/oracle_audit/trace_chain_top25_diff.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
EXCO_RANKS = {2, 11, 12}
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}
GOLD = {'174fd960'}   # #11's target node ("EX.CO ambient recall failure")


def tag(nid):
    p = nid[:8]
    if p in GOLD:
        return 'GOLD'
    if p in KNOWN_EXCO:
        return 'EXCO'
    return '    '


with IsolatedBrain() as env:
    brain = env.brain

    def title(nid):
        t = brain.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return (t[0] if t else '?')[:50]

    def recall_pool(query, flag_on):
        if flag_on:
            os.environ['BRAIN_TRACE_CHAIN'] = '1'
        else:
            os.environ.pop('BRAIN_TRACE_CHAIN', None)
        if hasattr(brain, '_recall_cache'):
            try: brain._recall_cache.clear()
            except Exception: pass
        out = brain.recall(query=query, limit=25)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        return [((r.get('id') or r.get('node_id')), r.get('_discovery')) for r in res]

    tot_entered = tot_dropped = tot_entered_relevant = 0
    for it in sorted(CORPUS, key=lambda x: x['rank']):
        rank, q = it['rank'], it['prompt']
        grp = 'EX.CO' if rank in EXCO_RANKS else 'ctrl '
        off = recall_pool(q, False)
        on = recall_pool(q, True)
        off_ids = [i for i, _ in off]
        on_ids = [i for i, _ in on]
        off_set, on_set = set(off_ids), set(on_ids)
        entered = [(i, d) for i, d in on if i not in off_set]     # what the lane added
        dropped = [i for i in off_ids if i not in on_set]          # what got displaced
        tot_entered += len(entered)
        tot_dropped += len(dropped)
        relevant_in = sum(1 for i, _ in entered if i[:8] in (KNOWN_EXCO | GOLD))
        tot_entered_relevant += relevant_in

        print("\n" + "=" * 84)
        print("#%-2d [%s] %s" % (rank, grp, q[:62]))
        print("    off pool: %d  |  on pool: %d  |  entered: %d (%d relevant)  |  dropped: %d"
              % (len(off_ids), len(on_ids), len(entered), relevant_in, len(dropped)))
        if entered:
            print("    ── ENTERED (what the lane added) ──")
            for i, d in entered:
                print("        [%s] %-10s %s %s" % (tag(i), (d or '?')[:10], i[:8], title(i)))
        if dropped:
            print("    ── DROPPED (displaced baseline tail) ──")
            for i in dropped:
                # show the discovery the dropped node had in the OFF run
                dd = dict(off).get(i)
                print("        [%s] %-10s %s %s" % (tag(i), (dd or '?')[:10], i[:8], title(i)))

    print("\n" + "=" * 84)
    print("SUMMARY across 12 queries:")
    print("  total entered: %d  (of which gold/EX.CO-relevant: %d, %.0f%%)"
          % (tot_entered, tot_entered_relevant, 100.0 * tot_entered_relevant / max(tot_entered, 1)))
    print("  total dropped: %d" % tot_dropped)
    print("  READ: 'entered' relevance on EX.CO queries = rescue quality; 'entered' on controls +")
    print("  'dropped' importance = the displacement cost (selection-level safety is Phase B/Haiku).")
    print("=" * 84)
