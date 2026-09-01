#!/usr/bin/env python3
"""Targeted burial check: EP5 ("what did we do last session on ex.co?") — does the
MAX enrichment fix lift the EX.CO nodes that avg2 buries? Prints the rank of each
target node under each arm (time-scoped to the moment). Daemon-safe (IsolatedBrain)."""
import os, sys, json, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QUERY = "what did we do on the last session we worked on ex.co?"
CUTOFF = "2026-04-22T23:59:59+00:00"   # episode moment (ex.co last-session was 04-21..22)
TARGETS = ['b3b6ce2a', 'dabb3078',     # corpus gold_essential
           'b8b8370b', '7b14f270',     # corpus gold_helpful
           '8359cf1d']                  # burial-handoff flagship ("EX.CO CTV kit")
ARMS = ['avg2', 'max', 'maxbonus']


def ranks(b, arm, timed=True):
    os.environ['BRAIN_ENRICH_SCORE'] = arm
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()
    filt = {"created_at": {"lte": CUTOFF}} if timed else None
    out = b.recall(query=QUERY, limit=300, filter=filt)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    ids = [(r.get('id') or r.get('node_id'))[:8] for r in res]
    os.environ.pop('BRAIN_ENRICH_SCORE', None)
    pos = {}
    for t in TARGETS:
        pos[t] = (ids.index(t[:8]) + 1) if t[:8] in ids else None
    return pos, len(ids)


def _delta(armpos):
    for t in TARGETS:
        a, m = armpos['avg2'][t], armpos['max'][t]
        verdict = ""
        if a is None and m is not None:
            verdict = "RESCUED (absent→%d)" % m
        elif a is not None and m is not None and m < a:
            verdict = "lifted %d→%d" % (a, m)
        elif a is not None and m is not None and m > a:
            verdict = "dropped %d→%d" % (a, m)
        elif a == m:
            verdict = "unchanged"
        print("  %s  %s" % (t, verdict))


with IsolatedBrain() as env:
    b = env.brain
    # confirm targets exist + not archived at cutoff
    print("=== target nodes (exist? archived?) ===")
    for t in TARGETS:
        row = b.conn.execute(
            "SELECT id, type, COALESCE(archived,0), substr(title,1,55) FROM nodes WHERE id = ?",
            (t,)).fetchone()
        print("  %s  %s" % (t, (("[%s arch=%s] %s" % (row[1], row[2], row[3])) if row else "NOT FOUND")))

    for timed in (True, False):
        label = "time-scoped <= %s" % CUTOFF[:10] if timed else "UNTIMED (full brain)"
        print("\n=== rank by arm (%s) ===" % label)
        armpos = {}
        for arm in ARMS:
            pos, n = ranks(b, arm, timed=timed)
            armpos[arm] = pos
            print("  %-9s pool=%d  %s" % (arm, n,
                  "  ".join("%s:%s" % (t, (pos[t] if pos[t] is not None else '—')) for t in TARGETS)))
        print("  -- avg2 → max delta (lower=better; — = absent) --")
        _delta(armpos)
