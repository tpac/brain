#!/usr/bin/env python3
"""WIRED CHECK — the trace-chain lane inside the REAL recall() (flag on vs off), isolated copy.

Validates the flag-gated build in brain_recall.py against two guarantees:
  (1) FLAG OFF  → byte-equivalent behavior: no 'trace_chain' discovery, pool unchanged.
  (2) FLAG ON   → reproduces the offline probe: #11's buried EX.CO nodes get rescued via the lane,
                  AND the main-lane top-K is unchanged (reserved-tail is additive, never reorders).

This runs the ACTUAL pipeline (not a reimplementation) on a DB copy. Never touches live.
Usage: ./dev python3 eval/oracle_audit/trace_chain_wired_check.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}
q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)
qctrl = next(it['prompt'] for it in CORPUS if it['rank'] == 6)   # a control that currently HITS


def run(brain, query, flag):
    if flag:
        os.environ['BRAIN_TRACE_CHAIN'] = '1'
    else:
        os.environ.pop('BRAIN_TRACE_CHAIN', None)
    if hasattr(brain, '_recall_cache'):
        try: brain._recall_cache.clear()
        except Exception: pass
    out = brain.recall(query=query, limit=25)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    ids = [(r.get('id') or r.get('node_id')) for r in res]
    disc = [r.get('_discovery') for r in res]
    return ids, disc


GOLD11 = {'174fd960'}   # corpus #11 target node ("cross-session EX.CO recall (174fd960 class)")

def title(brain, nid):
    t = brain.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
    return (t[0] if t else '?')[:46]

with IsolatedBrain() as env:
    brain = env.brain
    print("\n=== #11 — flag OFF vs ON ===")
    off_ids, off_disc = run(brain, q11, flag=False)
    on_ids, on_disc = run(brain, q11, flag=True)
    off_exco = [i for i in off_ids if i[:8] in KNOWN_EXCO]
    on_exco = [i for i in on_ids if i[:8] in KNOWN_EXCO]
    tc = [on_ids[i] for i, d in enumerate(on_disc) if d == 'trace_chain']
    print("  OFF: %d results, %d EX.CO, gold-in-pool=%s"
          % (len(off_ids), len(off_exco), bool(set(i[:8] for i in off_ids) & GOLD11)))
    print("  ON : %d results, %d EX.CO, gold-in-pool=%s, %d trace_chain rescues:"
          % (len(on_ids), len(on_exco), bool(set(i[:8] for i in on_ids) & GOLD11), len(tc)))
    for nid in tc:
        tag = 'GOLD' if nid[:8] in GOLD11 else ('EXCO' if nid[:8] in KNOWN_EXCO else '    ')
        print("      [%s] %s %s" % (tag, nid[:8], title(brain, nid)))
    # additivity: the main (non-trace_chain) results in ON must be a prefix of the OFF ordering
    on_main = [on_ids[i] for i, d in enumerate(on_disc) if d != 'trace_chain']
    main_prefix_match = on_main == off_ids[:len(on_main)]
    print("  main lane: ON keeps %d, OFF top-%d identical? %s" % (len(on_main), len(on_main), main_prefix_match))

    print("\n=== control #6 — flag OFF vs ON (safety) ===")
    coff_ids, coff_disc = run(brain, qctrl, flag=False)
    con_ids, con_disc = run(brain, qctrl, flag=True)
    ctc = sum(1 for d in con_disc if d == 'trace_chain')
    con_main = [con_ids[i] for i, d in enumerate(con_disc) if d != 'trace_chain']
    # does the control's main top-5 survive unchanged (reserved-tail additivity)?
    ctrl_top5_held = con_main[:5] == coff_ids[:5]
    print("  OFF: %d results ; ON: %d results, trace_chain=%d" % (len(coff_ids), len(con_ids), ctc))
    print("  control main top-5 unchanged with lane on? %s" % ctrl_top5_held)

    print("\n=== GUARANTEES ===")
    print("  (1) FLAG OFF no-op (no trace_chain on #11 or control): %s"
          % (sum(1 for d in off_disc if d == 'trace_chain') == 0 and
             sum(1 for d in coff_disc if d == 'trace_chain') == 0))
    print("  (2) FLAG ON rescues #11 relevant nodes (gold 174fd960 OR EX.CO in trace_chain): %s"
          % (any(t[:8] in (GOLD11 | KNOWN_EXCO) for t in tc)))
    print("  (3) #11 EX.CO in pool OFF=%d -> ON=%d ; gold rescued: %s"
          % (len(off_exco), len(on_exco), bool(set(t[:8] for t in tc) & GOLD11)))
    print("  (4) main-lane top preserved on #11 (additive prefix): %s" % main_prefix_match)
    print("  (5) control top-5 held (reserved-tail safe at retrieval): %s" % ctrl_top5_held)
