"""Backfill + audit for the enc-lane role expansion (connected / authored).

One script, one report, one decision point (plan step 1, 2026-07-29):

  A. REACH     — how far back each backfill source extends (edge rows /
                 anchor node rows vs the trace seams 06-18 / 06-30).
  B. ATTRIBUTION — connected: edge row → owning encode run via next-run-end
                 bisect (Scribe is single-flight, so the first run ending
                 after the edge's created_at is the run that wrote it).
                 authored: anchor node → (session, stop) via latest s0
                 user_message trace at-or-before the node's created_at.
  C/D. VALIDATION — post-seam, trace ground truth exists for BOTH roles:
                 backfill vs edge_relation_revised endpoint sets per run,
                 and vs anchor_touched created-sets per turn. Micro P/R.
  E. ROLE MASS — connected-old ids per run (the pre-existing resonant nodes)
                 vs created∪revised; authored volume.
  F. COLLINEARITY — the kill-switch stat: share of connected-old already in
                 the same window's `picked` role (encoder catalogs are built
                 from surfaced nodes, so conn may duplicate the pick lane).
                 Unique mass = ids in no existing role.
  G. EXAM COVERAGE — door-1 quality turns: as-of moment-pool share ≥ the
                 BACKFILL floor (vs the 13.8% trace-seam figure).

Read-only. Sources: $BRAIN_DB_DIR (point at the snapshot) + walker.db turns
+ corpus_v2_verdicts.jsonl.
Run:  BRAIN_DB_DIR=~/AgentsContext/brain-snap-roles-20260729 \
      ./dev python3 eval/laf/walker/role_backfill_audit.py
"""
import bisect
import json
from collections import defaultdict
from datetime import datetime

from walker_db import (open_brain_ro, open_logs_ro, open_walker, OUT_DIR,
                       brain_db_dir)

EDGE_SEAM = '2026-06-18'
TOUCHED_SEAM = '2026-06-30'
QUALITY_CUTOFF = '2026-05-11'


def norm(ts):
    """Comparable key across the two stored formats ('Z' vs '+00:00')."""
    return (ts or '').replace('Z', '+00:00')[:26]


def secs(a, b):
    """b − a in seconds; None when either side won't parse."""
    try:
        pa = datetime.fromisoformat(norm(a))
        pb = datetime.fromisoformat(norm(b))
        return (pb - pa).total_seconds()
    except ValueError:
        return None


def stop_of(chain):
    tail = str(chain or '').rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def meta(raw):
    try:
        return json.loads(raw or '{}')
    except ValueError:
        return {}


def pct(part, whole):
    return '%.1f%%' % (100.0 * part / whole) if whole else 'n/a'


def quantiles(xs, qs=(0.5, 0.9, 1.0)):
    xs = sorted(xs)
    return [xs[min(int(q * (len(xs) - 1)), len(xs) - 1)] for q in qs] if xs else []


def main():
    print('BRAIN_DB_DIR = %s' % brain_db_dir())
    brain = open_brain_ro()
    logs = open_logs_ro()

    # ── encode runs (attribution targets) ──
    runs = []          # (end_key, chain, session, cr_set, elapsed_s, raw_ts)
    for chain, sess, m_raw, ts in logs.execute(
            "SELECT chain_id, session_id, metadata, created_at FROM trace_events "
            "WHERE ref_type='encoding_run' AND scale='s1' AND event_type='delta'"):
        m = meta(m_raw)
        cr = set((m.get('created') or []) + (m.get('revised') or []))
        runs.append((norm(ts), chain, sess, cr, (m.get('elapsed_ms') or 0) / 1000.0, ts))
    runs.sort(key=lambda r: r[0])
    ends = [r[0] for r in runs]

    # ── A+B: encoder edge rows → runs ──
    edge_rows = brain.execute(
        "SELECT er.created_at, e.source_id, e.target_id FROM edge_relations er "
        "JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE er.encoding_source = 'encoder:sonnet'").fetchall()
    edge_rows.sort(key=lambda r: norm(r[0]))
    conn_floor = norm(edge_rows[0][0])[:10] if edge_rows else '-'

    by_run = defaultdict(set)      # run idx → backfilled endpoint set
    gaps, unattributed, within_elapsed = [], 0, 0
    for ts, src, tgt in edge_rows:
        i = bisect.bisect_left(ends, norm(ts))
        if i >= len(runs):
            unattributed += 1
            continue
        by_run[i].update((src, tgt))
        g = secs(ts, runs[i][5])
        if g is not None:
            gaps.append(g)
            if g <= runs[i][4] + 60:
                within_elapsed += 1

    print('\n=== A. REACH ===')
    print('  encoder edge rows: %d, floor %s (trace seam was %s)'
          % (len(edge_rows), conn_floor, EDGE_SEAM))
    anchor_nodes = brain.execute(
        "SELECT id, created_at FROM nodes WHERE encoding_source='anchor'").fetchall()
    print('  anchor nodes: %d, floor %s (trace seam was %s)'
          % (len(anchor_nodes), norm(min(n[1] for n in anchor_nodes))[:10]
             if anchor_nodes else '-', TOUCHED_SEAM))

    print('\n=== B. ATTRIBUTION ===')
    print('  edges attributed: %d/%d (unattributed: %d — no later run)'
          % (len(edge_rows) - unattributed, len(edge_rows), unattributed))
    print('  gap edge→run-end (s): p50/p90/max = %s' % quantiles(gaps))
    print('  within run elapsed window: %s' % pct(within_elapsed, len(gaps)))

    # ── C: connected validation vs trace ground truth (post-seam) ──
    trace_ep = defaultdict(set)    # chain → endpoint set from edge events
    for chain, m_raw in logs.execute(
            "SELECT chain_id, metadata FROM trace_events "
            "WHERE ref_type='edge_relation_revised' AND scale='s1'"):
        m = meta(m_raw)
        for k in ('source_id', 'target_id'):
            if m.get(k):
                trace_ep[chain].add(m[k])
    tp = fp = fn = 0
    n_cmp = 0
    for i, (end_key, chain, _s, _cr, _el, _ts) in enumerate(runs):
        if end_key < EDGE_SEAM or (chain not in trace_ep and i not in by_run):
            continue
        got, want = by_run.get(i, set()), trace_ep.get(chain, set())
        if not got and not want:
            continue
        n_cmp += 1
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
    print('\n=== C. CONNECTED VALIDATION (runs ≥ %s, n=%d) ===' % (EDGE_SEAM, n_cmp))
    print('  micro precision %s  recall %s  (tp=%d fp=%d fn=%d)'
          % (pct(tp, tp + fp), pct(tp, tp + fn), tp, fp, fn))

    # ── D: authored attribution + validation ──
    s0 = [(norm(ts), sess, stop_of(chain)) for ts, sess, chain in logs.execute(
        "SELECT created_at, session_id, chain_id FROM trace_events "
        "WHERE scale='s0' AND ref_type='user_message'")]
    s0 = [t for t in s0 if t[2] is not None]
    s0.sort()
    s0_keys = [t[0] for t in s0]
    auth_at = defaultdict(set)     # (session, stop) → node ids
    ambiguous = a_unattr = 0
    for nid, ts in anchor_nodes:
        k = norm(ts)
        i = bisect.bisect_right(s0_keys, k) - 1
        if i < 0:
            a_unattr += 1
            continue
        auth_at[(s0[i][1], s0[i][2])].add(nid)
        for j in range(i - 1, -1, -1):     # different session within 10 min?
            d = secs(s0[j][0], k)
            if d is not None and d > 600:
                break
            if s0[j][1] != s0[i][1]:
                ambiguous += 1
                break
    touched_at = {}                 # (session, stop) → created set (ground truth)
    for sess, chain, m_raw in logs.execute(
            "SELECT session_id, chain_id, metadata FROM trace_events "
            "WHERE ref_type='anchor_touched' AND scale='s0'"):
        st = stop_of(chain)
        if st is not None:
            touched_at[(sess, st)] = set(meta(m_raw).get('created') or [])
    atp = afp = afn = 0
    for key, want in touched_at.items():
        got = auth_at.get(key, set())
        atp += len(got & want)
        afp += len(got - want)     # backfill said created here, trace disagrees
        afn += len(want - got)
    print('\n=== D. AUTHORED (anchor nodes → turn) ===')
    print('  attributed: %d/%d (no-prior-trace: %d); cross-session ambiguity: %s'
          % (len(anchor_nodes) - a_unattr, len(anchor_nodes), a_unattr,
             pct(ambiguous, len(anchor_nodes))))
    print('  vs anchor_touched.created (%d turns): precision %s recall %s '
          '(tp=%d fp=%d fn=%d)'
          % (len(touched_at), pct(atp, atp + afp), pct(atp, atp + afn),
             atp, afp, afn))

    # ── E+F: role mass + collinearity ──
    picks_at = defaultdict(set)    # (session, stop) → picked shorts
    for sess, ref, chain in logs.execute(
            "SELECT session_id, ref_id, chain_id FROM trace_events "
            "WHERE ref_type='surface_selected' AND scale='s1'"):
        st = stop_of(chain)
        try:
            ids = json.loads(ref or '[]')
        except ValueError:
            ids = []
        if st is not None and isinstance(ids, list):
            picks_at[(sess, st)].update(ids)
    picks_sess = defaultdict(set)
    for (sess, _st), ids in picks_at.items():
        picks_sess[sess].update(ids)

    run_stop_by_idx = {i: stop_of(r[1]) for i, r in enumerate(runs)}
    prev_stop = {}
    by_sess_runs = defaultdict(list)
    for i, r in enumerate(runs):
        if run_stop_by_idx[i] is not None:
            by_sess_runs[r[2]].append(i)
    for sess, idxs in by_sess_runs.items():
        idxs.sort(key=lambda i: run_stop_by_idx[i])
        for pos, i in enumerate(idxs):
            prev_stop[i] = run_stop_by_idx[idxs[pos - 1]] if pos else 0

    mass_cr, mass_conn, n_runs_conn = [], [], 0
    in_win = in_sess = uniq = tot = 0
    for i, ids in by_run.items():
        cr = runs[i][3]
        conn_old = ids - cr
        mass_cr.append(len(cr))
        mass_conn.append(len(conn_old))
        if conn_old:
            n_runs_conn += 1
        sess, stop = runs[i][2], run_stop_by_idx.get(i)
        if stop is None:
            continue
        win = set()
        for ws in range(prev_stop.get(i, 0), stop + 1):
            win |= picks_at.get((sess, ws), set())
        for nid in conn_old:
            short = nid[:8]
            tot += 1
            if short in win:
                in_win += 1
            elif short in picks_sess.get(sess, set()):
                in_sess += 1
            else:
                uniq += 1
    print('\n=== E. ROLE MASS (attributed runs, n=%d) ===' % len(by_run))
    print('  per-run mean: created∪revised %.1f, connected-old %.1f; runs with conn: %s'
          % (sum(mass_cr) / max(len(mass_cr), 1),
             sum(mass_conn) / max(len(mass_conn), 1),
             pct(n_runs_conn, len(by_run))))
    print('\n=== F. COLLINEARITY (connected-old, n=%d ids) ===' % tot)
    print('  in same-window picked: %s   elsewhere-in-session picked: %s   '
          'UNIQUE (no pick role): %s'
          % (pct(in_win, tot), pct(in_sess, tot), pct(uniq, tot)))
    a_tot = sum(len(v) for v in auth_at.values())
    a_uniq = sum(1 for (sess, st), ids in auth_at.items() for nid in ids
                 if nid[:8] not in picks_sess.get(sess, set()))
    print('  authored: %d ids, %s not picked anywhere in their session'
          % (a_tot, pct(a_uniq, a_tot)))

    # ── H: unstamped-pool recovery (time-window attribution, calibrated) ──
    # encoder edge stamping starts late June; pre-06-08 encoder edges sit in
    # the (empty) encoding_source bucket alongside unstamped anchor/legacy
    # writes. Attribution: a row whose created_at falls inside a run's
    # [end − elapsed − 60s, end + 60s] window is encoder-written (Scribe is
    # single-flight). Calibration: TP rate = stamped encoder rows inside
    # their window (should match §B's 98.9%); FP rate = stamped NON-encoder
    # rows inside ANY window (chance collision — writers active near runs).
    def inside_window(ts_raw):
        k = norm(ts_raw)
        i = bisect.bisect_left(ends, k)
        if i >= len(runs):
            return None
        g = secs(ts_raw, runs[i][5])
        return i if (g is not None and g <= runs[i][4] + 60) else None

    def rate_inside(where):
        rows = brain.execute(
            "SELECT created_at FROM edge_relations WHERE %s" % where).fetchall()
        n_in = sum(1 for (ts,) in rows if inside_window(ts) is not None)
        return n_in, len(rows)

    print('\n=== H. UNSTAMPED-POOL RECOVERY (calibrated time-window) ===')
    tp_n, tp_d = rate_inside("encoding_source='encoder:sonnet'")
    print('  TP calibration (stamped encoder inside window): %s (%d/%d)'
          % (pct(tp_n, tp_d), tp_n, tp_d))
    for cls, where in (
            ('hebbian', "encoding_source='recall:hebbian'"),
            ('s2:*', "encoding_source LIKE 's2:%'"),
            ('anchor', "encoding_source IN ('anchor','anchor:seed')"),
            ('co_anchored', "encoding_source='dispatch:co_anchored'")):
        n_in, n_d = rate_inside(where)
        print('  FP calibration (%s inside window): %s (%d/%d)'
              % (cls, pct(n_in, n_d), n_in, n_d))
    # Relation-verb filter: the mechanical writers are single-verb (hebbian =
    # co_accessed, dispatch = co_anchored), S2/legacy structural edges are
    # community_member / emergent_bridge, and related/related_to is the
    # pre-v22 auto_connect-era generic bucket (cosine-automatic, not encoder
    # judgment). Excluding them leaves open semantic verbs — encoder-style.
    MECHANICAL = ('co_accessed', 'co_anchored', 'community_member',
                  'emergent_bridge', 'related', 'related_to')
    empty_rows = brain.execute(
        "SELECT er.created_at, e.source_id, e.target_id FROM edge_relations er "
        "JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE COALESCE(er.encoding_source,'') = '' "
        "AND er.relation NOT IN (%s)"
        % ','.join('?' * len(MECHANICAL)), MECHANICAL).fetchall()
    rec_by_month = defaultdict(int)
    rec_floor = None
    n_rec = 0
    for ts, src, tgt in empty_rows:
        if inside_window(ts) is not None:
            n_rec += 1
            m = norm(ts)[:7]
            rec_by_month[m] += 1
            if rec_floor is None or norm(ts) < rec_floor:
                rec_floor = norm(ts)
    print('  unstamped semantic-verb rows: %d; recovered as encoder: %d (floor %s)'
          % (len(empty_rows), n_rec, (rec_floor or '-')[:10]))
    print('  recovered by month: %s'
          % ', '.join('%s:%d' % kv for kv in sorted(rec_by_month.items())))
    # residual contamination bound: unstamped anchor-style writes that land
    # inside windows by chance — the anchor FP rate applied to the non-window
    # share tells us how much of the recovered set could be non-encoder.
    print('  outside-window semantic rows (non-encoder or failed-run): %d'
          % (len(empty_rows) - n_rec))

    # ── G: exam coverage at the backfill floor ──
    moments = sorted(norm(r[0]) for r in logs.execute(
        "SELECT created_at FROM trace_events "
        "WHERE ref_type='surface_selected' AND scale='s1'"))
    verds = [json.loads(x) for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()]
    door1 = {v['key'] for v in verds
             if v.get('verdict') == 'valid' and v.get('stratum') == 'cue'}
    w = open_walker()
    q_ts = sorted(norm(t) for s, e, q, t in w.execute(
        'SELECT session_id, epoch, seq, ts FROM turns')
        if '%s/%d/%d' % (s, e, q) in door1 and t and norm(t) >= QUALITY_CUTOFF)
    for floor, name in ((conn_floor, 'connected (stamped floor)'),
                        ((rec_floor or conn_floor)[:10],
                         'connected (with unstamped recovery)'),
                        (QUALITY_CUTOFF, 'authored (full era)')):
        shares = []
        for t in q_ts:
            n_vis = bisect.bisect_right(moments, t)
            if n_vis:
                shares.append(
                    (n_vis - bisect.bisect_left(moments, floor, 0, n_vis)) / n_vis)
        shares.sort()
        med = shares[len(shares) // 2] if shares else 0.0
        print('\n=== G. EXAM COVERAGE — %s ===' % name)
        print('  door-1 quality turns n=%d; median as-of pool share ≥ %s: %.1f%% '
              '(~%.1f of top-15 moments)'
              % (len(q_ts), floor, 100 * med, 15 * med))


if __name__ == '__main__':
    main()
