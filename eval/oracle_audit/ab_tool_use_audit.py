#!/usr/bin/env python3
"""SURFACE JUDGMENT SCOREBOARD: mine production s1r traces for Haiku judgment quality.

Reads a /tmp COPY of brain_logs.db (never the live file). For every s1r chain
in the window: K event (surface_selected) carries tool_trace + telemetry +
selected ids; the chain's O event carries the cosine candidate pool.

Reports (the judgment-quality triad + loop economics):
  - fire rate, rounds, per-tool stats (calls/results/latency/errors)
  - FETCH-PRECISION per tool: fetched candidates that won selection
    (needs tool_calls[].result_ids — traces written after 2026-07-02;
    older traces fall back to the off-25 aggregate)
  - off-25 selection share (tool-sourced picks, works on all traces)
  - recall_by_time window discipline: discussed-anchor share, sub-session
    windows ("last 5 minutes" class — the conversation already has these)
  - loop economics: cache hit share, forced-finalize / empty-tool_use rates
  - drift warnings: floor-all-drop, topical-zero-raw, cache-miss counts

Usage: ./dev python3 eval/oracle_audit/ab_tool_use_audit.py [days] [--split ISO]
  --split: optional ISO timestamp; report splits pre/post (e.g. a deploy)."""
import os, sys, json, shutil, sqlite3, re
from collections import Counter, defaultdict

args = [a for a in sys.argv[1:]]
SPLIT_AT = None
if '--split' in args:
    i = args.index('--split')
    SPLIT_AT = args[i + 1]
    del args[i:i + 2]
DAYS = float(args[0]) if args else 7.0

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

ERAS = ('pre', 'post') if SPLIT_AT else ('window',)


def era_of(created_at):
    if not SPLIT_AT:
        return 'window'
    return 'post' if created_at >= SPLIT_AT else 'pre'


stats = {e: defaultdict(float) for e in ERAS}
tool_stats = defaultdict(lambda: {'calls': 0, 'results': 0, 'lat': [], 'errors': 0,
                                  'zero': 0, 'fetched': 0, 'fetched_sel': 0,
                                  'calls_with_pick': 0, 'calls_with_ids': 0,
                                  'last_seen': ''})
dup_queries = 0
topical_queries = []
by_time_windows = Counter()      # (anchor, sub_session?) -> count
elapsed = {'fired': [], 'nofire': []}

# Sub-session window: the conversation buffer / recently-surfaced block
# already carries this — a discussed-anchor fetch here is the judge
# re-fetching what it holds (the wrong-reason-turn class).
_SUB_SESSION_RE = re.compile(
    r'just now|this conversation|last\s+\d+\s*(minute|min\b|m\b|turn)'
    r'|last\s+[1-3]\s*h(our)?s?\b')

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
    s = stats[era_of(K['created_at'])]
    s['n'] += 1

    tt = meta.get('tool_trace') or []
    calls = [c for rnd in tt for c in (rnd.get('tool_calls') or [])]
    s['fired'] += 1 if calls else 0
    s['total_calls'] += len(calls)
    s['api_rounds'] += meta.get('rounds') or len(tt)
    s['forced_finalize'] += sum(1 for rnd in tt if rnd.get('forced_finalize'))
    s['empty_tool_use'] += sum(
        1 for rnd in tt
        if rnd.get('stop_reason') == 'tool_use'
        and not rnd.get('tool_calls') and not rnd.get('forced_finalize'))

    # Loop economics from the flat telemetry block (post cost-telemetry ship)
    if meta.get('rounds'):
        if (meta.get('rounds') or 0) >= 2:
            s['multi_round'] += 1
            if (meta.get('cache_read_tokens') or 0) > 0:
                s['cache_hits'] += 1
                s['cache_read_total'] += meta['cache_read_tokens']
    if meta.get('elapsed_ms'):
        elapsed['fired' if calls else 'nofire'].append(meta['elapsed_ms'])

    # Selected set (8-char shorts from K.ref_id)
    selected = []
    try:
        selected = [x[:8] for x in json.loads(K['ref_id'])]
    except Exception:
        pass
    sel_set = set(selected)

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
        # FETCH-PRECISION: result_ids (admitted) that won selection.
        rids = c.get('result_ids')
        if rids is not None:
            t['calls_with_ids'] += 1
            picked = [x for x in rids if x in sel_set]
            t['fetched'] += len(rids)
            t['fetched_sel'] += len(picked)
            t['calls_with_pick'] += 1 if picked else 0
        cargs = c.get('args') or {}
        q = cargs.get('query', '')
        if q:
            qs_this.append(q)
            if c.get('tool') == 'recall_topical':
                topical_queries.append(q)
        if c.get('tool') == 'recall_by_time':
            anchor = cargs.get('time_anchor', 'event')
            win = ('%s %s' % (cargs.get('start_when') or '',
                              cargs.get('end_when') or '')).strip().lower()
            sub = bool(_SUB_SESSION_RE.search(win))
            by_time_windows[(anchor, sub)] += 1

    # near-duplicate queries within one recall (waste signal)
    for i in range(len(qs_this)):
        for j in range(i + 1, len(qs_this)):
            a, b = set(qs_this[i].lower().split()), set(qs_this[j].lower().split())
            if a and b and len(a & b) / len(a | b) > 0.6:
                dup_queries += 1

    # off-25 attribution: selected ids not in the O candidate pool
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
        s['off25_measured'] += 1


def _pct(v, d):
    return "%.0f%% (%d/%d)" % (100.0 * v / d, v, d) if d else "n/a"


print("\n=== SURFACE JUDGMENT SCOREBOARD — %d v5_agentic recalls, last %.0f days ==="
      % (n_chains, DAYS))
if SPLIT_AT:
    print("split at %s" % SPLIT_AT)
hdr = "".join("%16s" % e for e in ERAS)
print("%-28s%s" % ("", hdr))
for label, key, denom in (
        ("recalls", 'n', None),
        ("fired >=1 tool", 'fired', 'n'),
        ("avg tool calls/recall", 'total_calls', 'n'),
        ("avg api rounds/recall", 'api_rounds', 'n'),
        ("forced-finalize", 'forced_finalize', 'n'),
        ("empty tool_use rounds", 'empty_tool_use', 'n'),
        ("cache hit (multi-round)", 'cache_hits', 'multi_round'),
        ("selected off-25 share", 'sel_off25', 'sel_total'),
        ("recalls w/ off-25 pick", 'recalls_with_off25', 'off25_measured')):
    vals = []
    for era in ERAS:
        s = stats[era]
        v, d = s.get(key, 0), s.get(denom, 0) if denom else 0
        if denom:
            vals.append("%.1f" % (v / (d or 1)) if key in ('total_calls', 'api_rounds')
                        else _pct(v, d))
        else:
            vals.append("%d" % v)
    print("%-28s%s" % (label, "".join("%16s" % v for v in vals)))

import statistics
for lbl, arr in (('fired', elapsed['fired']), ('no-fire', elapsed['nofire'])):
    if arr:
        arr.sort()
        print("latency %-8s p50=%5dms  p90=%5dms  n=%d"
              % (lbl, statistics.median(arr), arr[int(0.9 * len(arr)) - 1], len(arr)))

print("\n--- per-tool: cost AND judgment (whole window) ---")
print("%-16s %6s %8s %6s %7s %8s | %9s %10s %12s" % (
    "tool", "calls", "avg-res", "zeros", "errors", "avg-ms",
    "precision", "conversion", "(ids-cover)"))
for tool, t in sorted(tool_stats.items(), key=lambda x: -x[1]['calls']):
    prec = "%.0f%%" % (100.0 * t['fetched_sel'] / t['fetched']) if t['fetched'] else "—"
    conv = "%.0f%%" % (100.0 * t['calls_with_pick'] / t['calls_with_ids']) if t['calls_with_ids'] else "—"
    cover = "%d/%d" % (t['calls_with_ids'], t['calls'])
    print("%-16s %6d %8.1f %6d %7d %8.0f | %9s %10s %12s" % (
        tool, t['calls'], t['results'] / max(t['calls'], 1), t['zero'], t['errors'],
        sum(t['lat']) / max(len(t['lat']), 1), prec, conv, cover))
print("  precision = fetched candidates that won selection; conversion = calls")
print("  with >=1 pick. Both need result_ids (traces from 2026-07-02 on).")

if by_time_windows:
    print("\n--- recall_by_time window discipline ---")
    total_bt = sum(by_time_windows.values())
    for (anchor, sub), n in by_time_windows.most_common():
        tag = "SUB-SESSION" if sub else "cross-session"
        print("  %-10s %-13s %5d  (%.0f%%)" % (anchor, tag, n, 100.0 * n / total_bt))
    sub_total = sum(n for (a, sub), n in by_time_windows.items() if sub)
    print("  sub-session share: %s  <- wrong-reason fires; v12 target ~0"
          % _pct(sub_total, total_bt))

print("\nnear-duplicate queries within one recall: %d" % dup_queries)
print("\n--- sample recall_topical reformulations (last 8) ---")
for q in topical_queries[-8:]:
    print("  %s" % q[:90])

# Drift-warning streams (the tripwires shipped 2026-07-02)
print("\n--- drift warnings (debug_log, window) ---")
wrows = conn.execute(
    """SELECT source, COUNT(*) AS n, MAX(created_at) AS last
       FROM debug_log
       WHERE created_at > ? AND source IN (
         'surface_floor_dropped_all', 'fetch_topical_zero_raw',
         'surface_cache_miss', 'surface_forced_finalize',
         'surface_empty_tool_use', 'surface_id_fuzzy_recovered',
         'surface_unknown_selected_id')
       GROUP BY source ORDER BY n DESC""", (cutoff,)).fetchall()
if wrows:
    for r in wrows:
        print("  %-28s %5d   last=%s" % (r['source'], r['n'], (r['last'] or '')[:16]))
else:
    print("  (none — either healthy or pre-deploy traces)")

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
    print("  n=%d  total: med=%.0fms p90=%.0fms   surface_haiku: med=%.0fms p90=%.0fms (%.0f%% of total)" % (
        n, statistics.median(tot), sorted(tot)[int(0.9 * n) - 1],
        statistics.median(hk), sorted(hk)[int(0.9 * n) - 1],
        100.0 * sum(hk) / max(sum(tot), 1)))
conn.close()
