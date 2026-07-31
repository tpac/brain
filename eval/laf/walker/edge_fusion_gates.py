"""T1 echo control + T2 provenance-marked per-lane reach — the two gates.

T1 ECHO CONTROL. The surviving cross-lane signal is exposed to echo: `pick`/
`enc` derive from Haiku's past selections, so "gold adjacent to an
episodically-lit node" may just mean "gold is in the thread Haiku was
surfacing." Three populations, same degree-matched estimator as
edge_fusion_audit.py:
  ALL              — every rankable valid turn (the audited baseline)
  NOT-PICKED-BEFORE— golds whose short id appears in NO surface_selected trace
                     of their session at or before the turn (as-of honest)
  NEVER-PICKED     — golds picked nowhere in their session, ever (strict)
plus an ENC-ONLY variant where the episodic group drops `pick` entirely.
KILL: complementary matched-diff collapsing below +5pp ⇒ the adjacency effect
is echo. KILL for cross-lane: similarity/hebbian sign flips vanishing.

T2 PROVENANCE-MARKED REACH (Tom's addition). Per LANE, seed from that lane's
OWN top-25 (not the mix) and hop over edges, marking every field entry:
  organic   — the gold is already in the lane's top-25, no hop needed
  hop1      — reached by one edge from an organic seed
  hop2      — reached only from a hop1 node (traversed-from-traversed: the
              double-count the fusion's off-diagonal term must exclude)
Reported with FANOUT so efficiency is visible (rescues per 100 nodes added) —
a lane that rescues by dragging in 400 nodes has not earned a diagonal block.
Complementary-verb-only vs all-verb walks are both scored (the audit says
complementary carries the signal).
KILL: no lane's hop1 efficiency beating maxsim's ⇒ diagonal blocks are just
enrichment renamed.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/edge_fusion_gates.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_brain_ro, open_logs_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from edge_fusion_census import LAM, TOP_N, LIT_Z, verb_class, iso   # noqa: E402

REPORT = OUT_DIR / 'edge_fusion_gates.md'
BAND = (6, 25)
CELLS = ('complementary', 'hebbian', 'similarity', 'corrective_strict')
DEG_LO, DEG_HI = 0.7, 1.4
SEED_K = 25          # per-lane seed set (pool-size convention)
LANES_T2 = ('maxsim', 'sit', 'idf', 'pick', 'enc', 'mh')


def norm_ts(ts):
    return (ts or '').replace('Z', '+00:00')[:26]


def main():
    aspects = json.loads(
        (Path(__file__).resolve().parents[3] /
         'servers/scales/s2/aspects_v1.json').read_text())
    corrective_all = set(
        (aspects.get('correction_improvement') or {}).get('edge_relations') or [])

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    n_nodes = idx['n_nodes']

    # ── graph ──
    b = open_brain_ro()
    adj = defaultdict(list)
    partners = defaultdict(set)
    for src, tgt, rel, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        vc = verb_class(rel, corrective_all)
        adj[si].append((ti, vc, created))
        adj[ti].append((si, vc, created))
        partners[si].add(ti)
        partners[ti].add(si)
    b.close()
    deg = {k: len(v) for k, v in partners.items()}

    # ── pick history per session (echo control) ──
    logs = open_logs_ro()
    picks = defaultdict(list)          # session -> [(created_norm, {shorts})]
    for sess, ref, created in logs.execute(
            "SELECT session_id, ref_id, created_at FROM trace_events "
            "WHERE ref_type='surface_selected' AND scale='s1'"):
        try:
            ids = json.loads(ref or '[]')
        except ValueError:
            ids = []
        if isinstance(ids, list) and ids:
            picks[sess].append((norm_ts(created), {str(x)[:8] for x in ids}))
    logs.close()
    print('pick history: %d sessions' % len(picks))

    turns, _enr, n = D.build_corpus('2026-05-11')
    gains = np.array([A.GAINS[ln] for ln in A.LANES])

    # ── per-turn precompute (shared by T1 and T2) ──
    rows = []
    for t in turns:
        U = np.flatnonzero(t['alive'])
        if U.size < 20:
            continue
        Z = np.column_stack([t['zl'][ln][U] for ln in A.LANES])
        f0 = Z @ gains
        if not np.isfinite(f0).any() or f0.std() <= 1e-9:
            continue
        zf0 = (f0 - f0.mean()) / f0.std()
        zmh = zn(t['mh'])[U]
        mix = LAM * zf0 + (1.0 - LAM) * zmh
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        order_local = np.argsort(-fin)
        top_nodes = U[order_local[:TOP_N]]
        gpos = np.flatnonzero(top_nodes == t['gr'])
        gold_rank = int(gpos[0]) + 1 if gpos.size else None

        # lane z arrays over the full master (−inf outside alive)
        lz = {}
        for ln in A.LANES:
            arr = np.full(n_nodes, -np.inf)
            arr[U] = t['zl'][ln][U]
            lz[ln] = arr
        arr = np.full(n_nodes, -np.inf)
        arr[U] = zmh
        lz['mh'] = arr

        sess = t['key'].split('/')[0]
        gold_short = master[t['gr']][:8]
        tdt = t.get('turn_dt')
        tkey = norm_ts(tdt.isoformat() if tdt else '')
        picked_before = any(gold_short in s for c, s in picks.get(sess, ())
                            if not tkey or c <= tkey)
        picked_ever = any(gold_short in s for _c, s in picks.get(sess, ()))
        rows.append({'t': t, 'U': U, 'top': top_nodes, 'gold_rank': gold_rank,
                     'lz': lz, 'gr': int(t['gr']), 'turn_dt': tdt,
                     'picked_before': picked_before, 'picked_ever': picked_ever,
                     'stratum': t['stratum']})
    print('turns prepared: %d' % len(rows))

    # ── T1: degree-matched diffs under population / lane-def variants ──
    def matched_diffs(subset, episodic_lanes):
        acc = defaultdict(lambda: [[], []])
        for r in subset:
            if r['gold_rank'] is None or not (
                    BAND[0] <= r['gold_rank'] <= BAND[1]):
                continue
            lz, U = r['lz'], r['U']
            lit = {'current': np.maximum.reduce(
                       [lz[x] for x in ('maxsim', 'sit', 'idf')]),
                   'episodic': np.maximum.reduce(
                       [lz[x] for x in episodic_lanes]),
                   'history': lz['mh']}
            tdt = r['turn_dt']

            def feats(ni):
                out = set()
                for (oi, vc, created) in adj.get(int(ni), ()):
                    edt = iso(created)
                    if tdt and edt and edt > tdt:
                        continue
                    for gname, arr in lit.items():
                        if arr[oi] >= LIT_Z:
                            out.add((vc, gname))
                return out

            band_nodes = [int(x) for i, x in enumerate(r['top'], 1)
                          if BAND[0] <= i <= BAND[1]]
            bf = {ni: feats(ni) for ni in band_nodes}
            gd = deg.get(r['gr'], 0)
            pool = [ni for ni in band_nodes if ni != r['gr']
                    and DEG_LO * gd <= deg.get(ni, 0) <= DEG_HI * gd]
            if len(pool) < 3:
                continue
            for vc in CELLS:
                for gname in lit:
                    g = 1.0 if (vc, gname) in bf[r['gr']] else 0.0
                    m = float(np.mean([1.0 if (vc, gname) in bf[ni] else 0.0
                                       for ni in pool]))
                    acc[(vc, gname)][0].append(g)
                    acc[(vc, gname)][1].append(m)
        return acc

    rng = np.random.default_rng(20260729)
    pops = [('ALL (audited baseline)', rows, ('pick', 'enc')),
            ('NOT-PICKED-BEFORE', [r for r in rows if not r['picked_before']],
             ('pick', 'enc')),
            ('NEVER-PICKED (strict)', [r for r in rows if not r['picked_ever']],
             ('pick', 'enc')),
            ('ALL, ENC-ONLY episodic', rows, ('enc',))]
    L = ['# T1 echo control + T2 provenance-marked reach', '',
         '## T1. Degree-matched diffs (band %d–%d) under echo controls' % BAND,
         '', '| population | matched turns | cell | episodic | current | history |',
         '|---|---|---|---|---|---|']
    t1 = {}
    for pname, subset, epi in pops:
        acc = matched_diffs(subset, epi)
        nmatch = len(acc[(CELLS[0], 'episodic')][0]) if acc else 0
        t1[pname] = (nmatch, acc)
        for vc in CELLS:
            cells = []
            for gname in ('episodic', 'current', 'history'):
                g, m = acc[(vc, gname)]
                if not g:
                    cells.append('—')
                    continue
                d = np.asarray(g) - np.asarray(m)
                # ONE generator for the whole table (hoisted): re-seeding per
                # cell made every cell's CI draw the identical resample
                # indices, so the 12 intervals were perfectly correlated
                # rather than independent.
                boots = np.array([
                    d[rng.integers(0, d.size, d.size)].mean()
                    for _ in range(2000)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                cells.append('%+.1f [%+.1f, %+.1f]%s'
                             % (100 * d.mean(), 100 * lo, 100 * hi,
                                ' **' if lo * hi > 0 else ''))
            L.append('| %s | %d | %s | %s |'
                     % (pname, nmatch, vc, ' | '.join(cells)))
        print('  T1 %s: n=%d' % (pname, nmatch))

    # ── T2: per-lane provenance-marked reach on the MISS population ──
    misses = [r for r in rows
              if r['gold_rank'] is not None and r['gold_rank'] > 5]
    stat = {ln: defaultdict(int) for ln in LANES_T2}
    fan = {ln: defaultdict(list) for ln in LANES_T2}
    for r in misses:
        tdt = r['turn_dt']
        gr = r['gr']
        for ln in LANES_T2:
            arr = r['lz'][ln]
            cand = r['U']
            vals = arr[cand]
            k = min(SEED_K, cand.size)
            sel = cand[np.argsort(-vals)[:k]]
            seeds = set(int(x) for x in sel)
            stat[ln]['turns'] += 1
            if gr in seeds:
                stat[ln]['organic'] += 1
                continue
            for verbset, tag in ((('complementary',), 'comp'), (None, 'all')):
                frontier, hit1 = set(), False
                for si in seeds:
                    for (oi, vc, created) in adj.get(si, ()):
                        if verbset and vc not in verbset:
                            continue
                        edt = iso(created)
                        if tdt and edt and edt > tdt:
                            continue
                        if oi in seeds:
                            continue
                        frontier.add(oi)
                        if oi == gr:
                            hit1 = True
                stat[ln]['hop1_%s' % tag] += 1 if hit1 else 0
                if hit1:
                    # NEW REACH vs SORTING: a rescue whose gold was already
                    # inside the mix's top-25 is a reorder, not new reach.
                    key = 'newreach' if r['gold_rank'] > 25 else 'sorting'
                    stat[ln]['hop1_%s_%s' % (tag, key)] += 1
                fan[ln]['hop1_%s' % tag].append(len(frontier))
                # hop2: only from the hop1 frontier (traversed-from-traversed)
                f2, hit2 = set(), False
                for si in frontier:
                    for (oi, vc, created) in adj.get(si, ()):
                        if verbset and vc not in verbset:
                            continue
                        edt = iso(created)
                        if tdt and edt and edt > tdt:
                            continue
                        if oi in seeds or oi in frontier:
                            continue
                        f2.add(oi)
                        if oi == gr:
                            hit2 = True
                stat[ln]['hop2_%s' % tag] += 1 if (hit2 and not hit1) else 0
                fan[ln]['hop2_%s' % tag].append(len(f2))

    L += ['', '## T2. Per-lane provenance-marked reach — miss population (n=%d)'
          % len(misses), '',
          'organic = gold already in that lane\'s own top-%d (no hop). '
          'hop1 = reached by one complementary edge from an organic seed. '
          'hop2 = reached ONLY from a hop1 node (traversed-from-traversed — '
          'the excludable double-count). EFF = rescues per 100 nodes of '
          'fan-out.' % SEED_K, '',
          '| lane | organic | hop1 comp | of which NEW REACH | sorting | '
          'fanout | EFF | hop1 all-verbs | EFF | hop2 extra | fanout |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for ln in LANES_T2:
        s, f = stat[ln], fan[ln]
        nt = max(s['turns'], 1)

        def eff(key):
            fm = float(np.mean(f[key])) if f[key] else 0.0
            return (100.0 * s[key] / nt / fm) if fm > 0 else 0.0
        L.append('| %s | %d (%.0f%%) | %d | **%d** | %d | %.0f | %.2f | %d '
                 '| %.2f | %d | %.0f |'
                 % (ln, s['organic'], 100.0 * s['organic'] / nt,
                    s['hop1_comp'], s['hop1_comp_newreach'],
                    s['hop1_comp_sorting'],
                    float(np.mean(f['hop1_comp']) or 0), eff('hop1_comp'),
                    s['hop1_all'], eff('hop1_all'),
                    s['hop2_comp'], float(np.mean(f['hop2_comp']) or 0)))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
