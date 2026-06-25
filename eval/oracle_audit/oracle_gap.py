#!/usr/bin/env python3
"""ORACLE-GAP — the falsifiable minimum test of "recall is prediction".

Thesis (ca840441): the best recall at turn N is the one you'd write AFTER seeing the next
turn. If true, recalling against the FUTURE (Anchor's actual next response) should surface
the knowledge the conversation went on to PRODUCE better than recalling against the PRESENT
(the user's message at N) does. If present≈future, prediction buys nothing — thesis dead.

For ~30 (user-msg N, assistant-response N) pairs from OLD sessions (frozen IsolatedBrain):
  present_pool = recall(user_msg_N)            # what we do today
  oracle_pool  = recall(assistant_resp_N)      # "the recall I'd write having seen my next move"
  target       = nodes created/revised in (t_N, t_N+WINDOW]   # the knowledge the stretch PRODUCED
                 (Tom's point 3 — created/revised, NOT surfaced; non-circular)
Measure per turn:
  - DIVERGENCE: top-K overlap(present, oracle). Low overlap → future wants different nodes.
  - HEADROOM:   target-coverage@K, oracle minus present. Positive → foresight surfaces produced
                knowledge the present misses = prediction has something real to learn.

Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/oracle_gap.py
"""
import sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

N_PAIRS = 30
K_SET = (8, 25)
WINDOW_H = 6          # hours after turn N to call "produced knowledge"
MIN_OLD_DAYS = 7      # only sample turns older than this (avoid this-session contamination)


def content_of(meta, summary):
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return summary or ''
    if isinstance(meta, dict):
        return meta.get('content') or summary or ''
    return summary or ''


def hours_between(a, b):
    # crude hour delta from ISO strings 'YYYY-MM-DDTHH:MM:SS'
    def h(s):
        s = str(s)
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})', s)
        if not m:
            return None
        y, mo, d, hh, mm = map(int, m.groups())
        return ((y * 365 + mo * 30 + d) * 24) + hh + mm / 60.0
    ha, hb = h(a), h(b)
    return None if ha is None or hb is None else hb - ha


with IsolatedBrain() as env:
    b = env.brain
    lc, bc = b.logs_conn, b.conn

    # pull s0 dialogue turns ordered within session
    rows = lc.execute(
        "SELECT session_id, ref_type, summary, metadata, created_at FROM trace_events "
        "WHERE scale='s0' AND ref_type IN ('user_message','assistant_message') "
        "ORDER BY session_id, created_at"
    ).fetchall()

    # build (user N, assistant-response N) pairs: a user_message immediately followed by an assistant_message
    pairs = []
    for i in range(len(rows) - 1):
        s0, rt0, sum0, meta0, ca0 = rows[i]
        s1, rt1, sum1, meta1, ca1 = rows[i + 1]
        if s0 != s1:
            continue
        if rt0 == 'user_message' and rt1 == 'assistant_message':
            u = content_of(meta0, sum0)
            a = content_of(meta1, sum1)
            if len(u) > 12 and len(a) > 40:
                pairs.append((ca0, u, a))

    # newest sampled turn must be older than MIN_OLD_DAYS — approximate by dropping the last chunk
    pairs.sort(key=lambda p: p[0])
    # take an evenly-strided sample across history (skip the most recent ~15% as "this-era")
    cut = int(len(pairs) * 0.85)
    pool_pairs = pairs[:cut]
    stride = max(1, len(pool_pairs) // (N_PAIRS * 2))
    sampled = pool_pairs[::stride]

    def nodes_in_window(t0):
        # produced knowledge: created OR revised within (t0, t0+WINDOW_H]
        out = set()
        for col in ('created_at', 'revised_at'):
            for (nid, ts) in bc.execute(
                "SELECT id, %s FROM nodes WHERE %s > ? AND COALESCE(archived,0)=0" % (col, col),
                (t0,)).fetchall():
                dh = hours_between(t0, ts)
                if dh is not None and 0 < dh <= WINDOW_H:
                    out.add(nid[:8])
        return out

    def ranked(query):
        out = b.recall(query=query, limit=100)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        return [(r.get('id') or r.get('node_id'))[:8] for r in res]

    agg = {k: {'present': 0, 'oracle': 0, 'target_tot': 0, 'overlap': 0.0} for k in K_SET}
    n_used = 0
    print("turn | target | " + " | ".join("K=%d pres/orac" % k for k in K_SET))
    print("-" * 70)
    for (t0, umsg, aresp) in sampled:
        if n_used >= N_PAIRS:
            break
        target = nodes_in_window(t0)
        if not target:
            continue
        present = ranked(umsg)
        oracle = ranked(aresp)
        if not present or not oracle:
            continue
        n_used += 1
        line = ["%s" % str(t0)[5:10], "%d" % len(target)]
        for k in K_SET:
            ps, os_ = set(present[:k]), set(oracle[:k])
            pc = len(target & ps)
            oc = len(target & os_)
            agg[k]['present'] += pc
            agg[k]['oracle'] += oc
            agg[k]['target_tot'] += len(target)
            agg[k]['overlap'] += len(ps & os_) / k
            line.append("%d/%d" % (pc, oc))
        print(" | ".join(line))

    print("\n=== ORACLE GAP (n=%d turn-pairs, window=%dh) ===" % (n_used, WINDOW_H))
    print("Does recalling against the FUTURE (oracle) surface produced-knowledge that recalling")
    print("against the PRESENT misses? coverage = target-nodes-hit / target-total\n")
    for k in K_SET:
        a = agg[k]
        tot = max(a['target_tot'], 1)
        pcov = 100.0 * a['present'] / tot
        ocov = 100.0 * a['oracle'] / tot
        print("  K=%-2d  present-coverage %5.1f%%   oracle-coverage %5.1f%%   GAP %+5.1f pp   |  present/oracle top-K overlap %.0f%%"
              % (k, pcov, ocov, ocov - pcov, 100.0 * a['overlap'] / max(n_used, 1)))
    print("\nREAD: big positive GAP + low overlap → the future wants materially different, better")
    print("nodes than the present → prediction-recall has real headroom. ~zero GAP → thesis dead.")
