"""Role-coverage audit — go/no-go for the enc-lane role expansion (connected/authored).

Before rebuilding the episodic substrate with the two new role sets, measure
whether the trace record can power a door-1 eval at all:

  A. AVAILABILITY — edge_relation_revised (s1, encoder chains) and
     anchor_touched (s0) volume by ISO week: the seams shipped 2026-06-18 /
     2026-06-30, so any moment older than that contributes nothing new.
  B. MAGNITUDE — per encode run: |created ∪ revised| vs |connected endpoints
     NOT in created∪revised| (the pre-existing resonant nodes Tom's point is
     about). Per anchor_touched turn: authored/recalled counts. How much
     role mass the expansion actually adds where data exists.
  C. EXAM COVERAGE — door-1 turns (corpus-v2 cue valids, quality era
     ≥2026-05-11): how many fall after each seam, and what share of the
     surface-moment pool (the population the enc lane's top-15 draws from,
     as-of each turn) is post-seam.

Read-only everywhere (open_logs_ro / verdicts jsonl / walker.db turns).
Run:  ./dev python3 eval/laf/walker/role_coverage_audit.py
"""
import json
from collections import Counter, defaultdict

from walker_db import open_logs_ro, open_walker, OUT_DIR

EDGE_SEAM = '2026-06-18'      # 7f43c2d — directional edge traces ship
TOUCHED_SEAM = '2026-06-30'   # da124b1 — anchor_touched feed ships
QUALITY_CUTOFF = '2026-05-11' # 7333b0d8 — quality corpus era


def week(ts):
    return ts[:10] if not ts else '%s-W%02d' % __import__('datetime').datetime.fromisoformat(ts.replace('Z', '+00:00')).isocalendar()[:2]


def load_meta(row):
    try:
        return json.loads(row or '{}')
    except ValueError:
        return {}


def main():
    logs = open_logs_ro()

    # ── A+B: edge events on encoder chains ──
    edge_rows = logs.execute(
        "SELECT chain_id, metadata, created_at FROM trace_events "
        "WHERE ref_type='edge_relation_revised' AND scale='s1'").fetchall()
    edges_by_week = Counter()
    endpoints_by_chain = defaultdict(set)
    for chain, meta_raw, ts in edge_rows:
        edges_by_week[week(ts)] += 1
        m = load_meta(meta_raw)
        for k in ('source_id', 'target_id'):
            if m.get(k):
                endpoints_by_chain[chain].add(m[k])

    runs = logs.execute(
        "SELECT chain_id, metadata, created_at FROM trace_events "
        "WHERE ref_type='encoding_run' AND scale='s1' AND event_type='delta'").fetchall()
    run_stats = []          # (ts, n_created_revised, n_connected_new)
    for chain, meta_raw, ts in runs:
        m = load_meta(meta_raw)
        cr = set((m.get('created') or []) + (m.get('revised') or []))
        conn = endpoints_by_chain.get(chain, set()) - cr
        run_stats.append((ts, len(cr), len(conn)))

    # ── B: anchor_touched ──
    touched = logs.execute(
        "SELECT metadata, created_at FROM trace_events "
        "WHERE ref_type='anchor_touched' AND scale='s0'").fetchall()
    touched_by_week = Counter()
    t_auth = t_rec = t_any = 0
    for meta_raw, ts in touched:
        touched_by_week[week(ts)] += 1
        m = load_meta(meta_raw)
        a = len((m.get('created') or [])) + len((m.get('revised') or []))
        r = len(m.get('recalled') or [])
        t_auth += a
        t_rec += r
        t_any += 1 if (a or r) else 0

    # ── C: moment pool + door-1 exam ──
    moments = [r[0] or '' for r in logs.execute(
        "SELECT created_at FROM trace_events "
        "WHERE ref_type='surface_selected' AND scale='s1'")]
    moments.sort()

    verds = [json.loads(x) for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()]
    door1_keys = {v['key'] for v in verds
                  if v.get('verdict') == 'valid' and v.get('stratum') == 'cue'}
    w = open_walker()
    ts_by_key = {'%s/%d/%d' % (s, e, q): t for s, e, q, t in
                 w.execute('SELECT session_id, epoch, seq, ts FROM turns')}
    door1_ts = sorted(t for k, t in ts_by_key.items() if k in door1_keys and t)
    quality = [t for t in door1_ts if t >= QUALITY_CUTOFF]

    import bisect
    def pool_share(turn_ts, seam):
        """Share of the surface-moment pool visible at turn_ts that is ≥ seam."""
        n_vis = bisect.bisect_right(moments, turn_ts)
        if not n_vis:
            return 0.0
        n_post = n_vis - bisect.bisect_left(moments, seam, 0, n_vis)
        return n_post / n_vis

    print('=== A. AVAILABILITY (events/week) ===')
    for wk in sorted(set(edges_by_week) | set(touched_by_week)):
        print('  %s  edge_relation_revised(s1): %4d   anchor_touched: %4d'
              % (wk, edges_by_week.get(wk, 0), touched_by_week.get(wk, 0)))

    post_runs = [(c, n) for ts, c, n in run_stats if ts and ts >= EDGE_SEAM]
    print('\n=== B. MAGNITUDE ===')
    print('  encode runs total: %d; post-seam (>=%s): %d' % (len(run_stats), EDGE_SEAM, len(post_runs)))
    if post_runs:
        n_with = sum(1 for _, n in post_runs if n)
        print('  post-seam runs with >=1 connected-old endpoint: %d (%.0f%%)'
              % (n_with, 100.0 * n_with / len(post_runs)))
        print('  per-run means: created∪revised %.1f, connected-old %.1f'
              % (sum(c for c, _ in post_runs) / len(post_runs),
                 sum(n for _, n in post_runs) / len(post_runs)))
    print('  anchor_touched turns: %d (non-empty: %d); authored ids: %d, recalled ids: %d'
          % (len(touched), t_any, t_auth, t_rec))

    print('\n=== C. EXAM COVERAGE (door-1 = cue valids) ===')
    print('  door-1 turns with walker ts: %d; quality era (>=%s): %d'
          % (len(door1_ts), QUALITY_CUTOFF, len(quality)))
    for seam, name in ((EDGE_SEAM, 'edge seam'), (TOUCHED_SEAM, 'touched seam')):
        after = [t for t in quality if t >= seam]
        shares = sorted(pool_share(t, seam) for t in after)
        med = shares[len(shares) // 2] if shares else 0.0
        mean = sum(shares) / len(shares) if shares else 0.0
        print('  %s (%s): turns after seam %d/%d; as-of pool share >= seam among those:'
              ' median %.1f%% mean %.1f%% (expected post-seam moments in top-15: ~%.1f)'
              % (name, seam, len(after), len(quality),
                 100 * med, 100 * mean, 15 * mean))
    print('\n  moment pool total: %d surface moments (%s .. %s)'
          % (len(moments), moments[0][:10] if moments else '-', moments[-1][:10] if moments else '-'))


if __name__ == '__main__':
    main()
