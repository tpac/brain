#!/usr/bin/env python3
"""AREA-3 TOOL-USE AUDIT: mine production s1r traces for Haiku tool behavior.

Reads a /tmp COPY of brain_logs.db (never the live file). For every s1r chain
in the window: K event (surface_selected) carries tool_trace + selected ids;
the chain's O event carries the cosine candidate pool → off-25 attribution.

Reports: fire rate, rounds, per-tool stats (calls/results/latency/errors),
off-25 selection share, duplicate-query waste, and a pre/post-idf2 split.
Usage: ./dev python3 eval/oracle_audit/ab_tool_use_audit.py [days]"""
import os, sys, json, shutil, sqlite3, glob, re
from collections import Counter, defaultdict

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
IDF2_LIVE_AT = "2026-06-12T14:40:00+00:00"   # idf2 merge + daemon restart (approx)

SRC = os.path.expanduser("~/AgentsContext/brain/brain_logs.db")
DST = "/tmp/brain_logs_audit_copy.db"
for ext in ("", "-wal", "-shm"):
    s = SRC + ext
    if os.path.exists(s):
        shutil.copy(s, DST + ext)

conn = sqlite3.connect("file:%s?mode=ro" % DST, uri=True)
conn.row_factory = sqlite3.Row

from datetime import datetime, timedelta, timezone
cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat()

rows = conn.execute(
    """SELECT chain_id, event_type, ref_type, ref_id, metadata, created_at, session_id
       FROM trace_events
       WHERE scale='s1' AND created_at > ? AND chain_id LIKE 's1r-%'
       ORDER BY created_at""", (cutoff,)).fetchall()

chains = defaultdict(dict)
for r in rows:
    if r['event_type'] == 'K' and r['ref_type'] == 'surface_selected':
        chains[r['chain_id']]['K'] = r
    elif r['event_type'] == 'O':
        chains[r['chain_id']].setdefault('O', r)

stats = {
    'pre':  defaultdict(float),
    'post': defaultdict(float),
}
tool_stats = defaultdict(lambda: {'calls': 0, 'results': 0, 'lat': [], 'errors': 0,
                                  'zero': 0, 'last_seen': ''})
dup_queries = 0
topical_queries = []
off25_examples = []

n_chains = 0
for cid, ev in chains.items():
    K = ev.get('K')
    if not K:
        continue
    try:
        meta = json.loads(K['metadata']) if isinstance(K['metadata'], str) else (K['metadata'] or {})
    except Exception:
        continue
    if meta.get('surface_variant') != 'v5_agentic':
        continue
    n_chains += 1
    era = 'post' if K['created_at'] >= IDF2_LIVE_AT else 'pre'
    s = stats[era]
    s['n'] += 1

    tt = meta.get('tool_trace') or []
    calls = [c for rnd in tt for c in (rnd.get('tool_calls') or [])]
    rounds_used = sum(1 for rnd in tt if rnd.get('tool_calls'))
    s['fired'] += 1 if calls else 0
    s['total_calls'] += len(calls)
    s['rounds'] += len(tt)

    qs_this = []
    for c in calls:
        t = tool_stats[c.get('tool', '?')]
        t['calls'] += 1
        t['last_seen'] = max(t['last_seen'], K['created_at'][:16])
        rc = c.get('result_count', 0) or 0
        t['results'] += rc
        t['zero'] += 1 if rc == 0 else 0
        if c.get('latency_ms'):
            t['lat'].append(c['latency_ms'])
        if c.get('error'):
            t['errors'] += 1
        q = (c.get('args') or {}).get('query', '')
        if q:
            qs_this.append(q)
            if c.get('tool') == 'recall_topical':
                topical_queries.append(q)
    # near-duplicate queries within one recall (waste signal)
    for i in range(len(qs_this)):
        for j in range(i + 1, len(qs_this)):
            a, b = set(qs_this[i].lower().split()), set(qs_this[j].lower().split())
            if a and b and len(a & b) / len(a | b) > 0.6:
                dup_queries += 1

    # off-25 attribution: selected ids not in the O candidate pool
    selected = []
    try:
        selected = [x[:8] for x in json.loads(K['ref_id'])]
    except Exception:
        pass
    O = ev.get('O')
    pool = set()
    if O:
        try:
            ometa = json.loads(O['metadata']) if isinstance(O['metadata'], str) else (O['metadata'] or {})
            for key in ('candidates', 'candidate_ids', 'results'):
                v = ometa.get(key)
                if isinstance(v, list):
                    pool = {(x.get('id') if isinstance(x, dict) else str(x))[:8] for x in v}
                    break
            if not pool:
                ids = re.findall(r'[0-9a-f]{8}', O['ref_id'] or '')
                pool = set(ids)
        except Exception:
            pass
    if selected and pool:
        off = [x for x in selected if x not in pool]
        s['sel_total'] += len(selected)
        s['sel_off25'] += len(off)
        s['recalls_with_off25'] += 1 if off else 0
        s['recalls_all_off25'] += 1 if len(off) == len(selected) else 0
        s['off25_measured'] += 1
        if off and len(off25_examples) < 5:
            off25_examples.append((cid, len(off), len(selected)))

print("\n=== TOOL-USE AUDIT — %d v5_agentic recalls in last %.0f days ===" % (n_chains, DAYS))
print("%-22s %10s %10s" % ("", "pre-idf2", "post-idf2"))
for label, key, denom in (
        ("recalls", 'n', None),
        ("fired >=1 tool", 'fired', 'n'),
        ("avg tool calls/recall", 'total_calls', 'n'),
        ("off-25 measured", 'off25_measured', None),
        ("selected off-25 share", 'sel_off25', 'sel_total'),
        ("recalls w/ off-25 pick", 'recalls_with_off25', 'off25_measured'),
        ("recalls ALL off-25", 'recalls_all_off25', 'off25_measured')):
    vals = []
    for era in ('pre', 'post'):
        s = stats[era]
        v = s.get(key, 0)
        if denom:
            d = s.get(denom, 0) or 1
            vals.append("%.0f%% (%d/%d)" % (100.0 * v / d, v, s.get(denom, 0)) if denom != 'n' or key != 'total_calls' else "")
            if key == 'total_calls':
                vals[-1] = "%.1f" % (v / d)
        else:
            vals.append("%d" % v)
    print("%-22s %10s %10s" % (label, vals[0], vals[1]))

print("\n--- per-tool (whole window) ---")
print("%-16s %6s %9s %7s %7s %9s %17s" % ("tool", "calls", "avg-res", "zeros", "errors", "avg-ms", "last-seen"))
for tool, t in sorted(tool_stats.items(), key=lambda x: -x[1]['calls']):
    print("%-16s %6d %9.1f %7d %7d %9.0f %17s" % (
        tool, t['calls'], t['results'] / max(t['calls'], 1), t['zero'], t['errors'],
        sum(t['lat']) / max(len(t['lat']), 1), t['last_seen']))

print("\nnear-duplicate queries within one recall: %d" % dup_queries)
print("\n--- sample recall_topical reformulations (last 8) ---")
for q in topical_queries[-8:]:
    print("  %s" % q[:90])

# Latency decomposition from recall_log phase-timing lines (the timing string
# lives in the `source` column for hook_phase_timing events)
print("\n--- hook_recall phase decomposition (debug_log, window) ---")
prows = conn.execute(
    """SELECT source, event_type FROM debug_log
       WHERE created_at > ? AND (source LIKE '%surface_haiku%' OR event_type LIKE '%phase%')
       ORDER BY created_at DESC LIMIT 300""", (cutoff,)).fetchall()
tot, hk, n = [], [], 0
for r in prows:
    m = re.search(r'total:(\d+)ms.*surface_haiku:(\d+)ms', (r['source'] or '') + ' ')
    if m:
        tot.append(int(m.group(1))); hk.append(int(m.group(2))); n += 1
if n:
    import statistics
    print("  n=%d  total: med=%.0fms p90=%.0fms   surface_haiku: med=%.0fms p90=%.0fms (%.0f%% of total)" % (
        n, statistics.median(tot), sorted(tot)[int(0.9 * n) - 1],
        statistics.median(hk), sorted(hk)[int(0.9 * n) - 1],
        100.0 * sum(hk) / max(sum(tot), 1)))
conn.close()
