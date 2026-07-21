"""ARBITRATION, part 1 — node/turn/lane CHARACTERISTICS of the +15.9pp
prize (Tom's challenge, 2026-07-21: 'did you really look at the nodes?').

The TIE-FAIR remix map (miss_anatomy 398e08fb) prices REACHABLE at 317
misses whose gold is already top-5 in some held field — lane-oracle
ceiling 46.6% @5 vs 30.7% baseline. Claiming it needs per-turn TRUST. Before
building any gate: WHAT distinguishes the turns/nodes where field X holds
the gold? This probe mines the contrast:

For every turn, find the gold's best-ranking held field (5 slots + 5 msg-0
lanes + M_h). Group turns by that winner; per group, contrast against the
mix-hit population on:
  node side:  gold age at turn · type · degree · title length
  turn side:  cue length · has_question · session position
  lane side:  winning lane's z at gold · margin over the lane's #2 ·
              lane support size · the lane's top-1 z (spike strength —
              the would-be gate feature)
Also the ARBITRATION HONESTY numbers: lane-oracle @5 (ceiling), and per
winning-field the FALSE-TRUST cost — on turns where that field's top-1 is
NOT the gold, how often would trusting it displace a mix hit? (gate
feasibility = spike-strength separation between true and false trusts).

Run:    ./dev python3 eval/laf/walker/arbitration_probe.py
Pool60: BRAIN_DB_DIR=... WALKER_OUT_DIR=... (brain reads need the dir)
"""
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star, GRID                       # noqa: E402
from layer_readout_probe import lane_z                               # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
LANE_CACHE = OUT_DIR / 'lane_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'


def brain_conn():
    db = os.environ.get('BRAIN_DB_DIR') or \
        str(Path.home() / 'AgentsContext' / 'brain')
    return sqlite3.connect('file:%s/brain.db?mode=ro' % db, uri=True)


def rank_in(f, gr):
    """TIE-FAIR rank (mid-rank among ties) — see miss_anatomy.rank_in;
    strict-> inflated sparse-lane ranks (review BLOCKER 2026-07-21)."""
    if f is None or not np.isfinite(f[gr]):
        return None
    fin = np.where(np.isfinite(f), f, -np.inf)
    greater = int((fin > f[gr]).sum())
    ties = int((fin == f[gr]).sum())
    return greater + (ties - 1) / 2.0 + 1


def iso_days(a, b):
    try:
        return (datetime.fromisoformat(a.replace('Z', '+00:00'))
                - datetime.fromisoformat(b.replace('Z', '+00:00'))).days
    except Exception:
        return None


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    lanes_mm = np.load(LANE_CACHE, mmap_mode='r')
    slots, lanes = idx['slots'], idx['lanes']
    S = {s: i for i, s in enumerate(slots)}
    n = idx['n_nodes']
    master = idx['master']

    walker = open_walker()
    tmeta = dict(((s, e, q), (ln or 0, hq or 0, ts)) for s, e, q, ln, hq,
                 ts in walker.execute(
                     "SELECT session_id, epoch, seq, op_len, has_question, "
                     "ts FROM turns WHERE labeled=1"))
    walker.close()

    conn = brain_conn()
    deg = Counter()
    for src, tgt in conn.execute(
            "SELECT e.source_id, e.target_id FROM edges e "
            "JOIN nodes ns ON ns.id=e.source_id "
            "JOIN nodes nt ON nt.id=e.target_id "
            "WHERE ns.archived=0 AND nt.archived=0"):
        deg[src] += 1
        deg[tgt] += 1
    nmeta = dict((nid, (typ, cat, len(title or '')))
                 for nid, typ, cat, title in conn.execute(
                     "SELECT id, type, created_at, title FROM nodes "
                     "WHERE archived=0"))
    conn.close()

    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr >= 0 and tt.ro is not None and tt.mh is not None \
                and not np.isnan(tt.fields[0]).all():
            turns.append((t, tt))
    per_l = {l: 0 for l in GRID}
    for _t, tt in turns:
        for l, r in lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr).items():
            per_l[l] += int(r <= 5)
    lam_s = max(GRID, key=lambda l: per_l[l])

    FIELD_NAMES = (['F0', 'f1', 'a1', 'f2', 'a2']
                   + ['lane_' + ln for ln in lanes] + ['M_h'])
    groups = defaultdict(list)       # winner field → feature dicts
    hits_feats = []
    oracle5 = 0
    false_trust = defaultdict(lambda: [0, 0])   # field → [false, total]
    spike_true, spike_false = defaultdict(list), defaultdict(list)

    for t, tt in turns:
        gr = tt.gr
        f0z, mhz = zn(tt.fields[0]), zn(tt.mh)
        rk = lambda_star(f0z, mhz, gr, grid=np.array([lam_s]))
        mix_rank = min(rk.values()) if rk else None
        if mix_rank is None:
            continue
        L = lanes_mm[t['row']].astype(np.float32)
        mx = L[S['op0'], lanes.index('maxsim')]
        alive = np.isfinite(mx)
        z0 = {ln: lane_z(L[S['op0'], li], ln, alive, n)
              for li, ln in enumerate(lanes)}
        held = {}
        for nm, f in zip(('F0', 'f1', 'a1', 'f2', 'a2'), tt.fields):
            held[nm] = f
        for ln in lanes:
            held['lane_' + ln] = z0[ln]
        held['M_h'] = tt.mh

        ranks = {k: rank_in(f, gr) for k, f in held.items()}
        rvals = {k: v for k, v in ranks.items() if v is not None}
        best_k = min(rvals, key=rvals.get) if rvals else None
        best = rvals.get(best_k, 10 ** 9)
        oracle5 += int(min(best, mix_rank) <= 5)

        # false-trust bookkeeping: per field, is its top-1 the gold?
        for k, f in held.items():
            if f is None or np.isnan(f).all():
                continue
            top1 = int(np.nanargmax(np.where(np.isfinite(f), f, -np.inf)))
            z1 = float(f[top1])
            zvals = np.sort(f[np.isfinite(f)])
            margin = z1 - (zvals[-2] if len(zvals) > 1 else z1)
            false_trust[k][1] += 1
            if top1 == gr:
                spike_true[k].append((z1, margin))
            else:
                false_trust[k][0] += 1
                spike_false[k].append((z1, margin))

        # feature record
        gid = master[gr]
        typ, cat, tl = nmeta.get(gid, ('?', None, 0))
        oplen, hq, ts = tmeta.get(tt.key, (0, 0, ''))
        wl = best_k if best <= 5 else None
        feats = {
            'age': iso_days(ts, cat) if (cat and ts) else None,
            'type': typ, 'deg': deg.get(gid, 0), 'title_len': tl,
            'cue_len': oplen, 'has_q': hq,
            'wz': (float(held[wl][gr]) if wl and held[wl] is not None
                   and np.isfinite(held[wl][gr]) else None) if wl else None,
        }
        if mix_rank <= 5:
            hits_feats.append(feats)
        elif wl is not None:
            groups[wl].append(feats)

    n_t = len(turns)
    print('turns %d · λ=%.2f · mix@5 %.1f%% · LANE-ORACLE @5 %.1f%% '
          '(the arbitration ceiling)'
          % (n_t, lam_s, 100 * len(hits_feats) / n_t, 100 * oracle5 / n_t))

    def agg(rows, key, fn=np.nanmedian):
        v = [r[key] for r in rows if r.get(key) is not None]
        return fn(np.array(v, dtype=float)) if v else float('nan')

    print('\n== who holds the gold on REACHABLE misses — node/turn '
          'characteristics vs mix-hits ==')
    print('%-12s %5s  %7s %6s %7s %8s %6s  %s'
          % ('winner', 'n', 'age_md', 'deg', 'title', 'cue_len', 'has_q',
             'top types'))
    ref = ('mix-hits', hits_feats)
    for name, rows in [ref] + sorted(groups.items(),
                                     key=lambda kv: -len(kv[1])):
        tc = Counter(r['type'] for r in rows).most_common(3)
        print('%-12s %5d  %6.0fd %6.0f %7.0f %8.0f %5.0f%%  %s'
              % (name, len(rows), agg(rows, 'age'), agg(rows, 'deg'),
                 agg(rows, 'title_len'), agg(rows, 'cue_len'),
                 100 * agg(rows, 'has_q', np.nanmean),
                 ' '.join('%s:%d' % kv for kv in tc)))

    print('\n== gate feasibility: spike strength, TRUE trust vs FALSE '
          'trust (field top-1 == gold vs not) ==')
    print('%-12s %10s %12s %12s %12s' % ('field', 'P(top1=gold)',
                                         'z1 true', 'z1 false',
                                         'margin true/false'))
    for k in FIELD_NAMES:
        ft = false_trust.get(k)
        if not ft or not ft[1]:
            continue
        st, sf = spike_true.get(k) or [], spike_false.get(k) or []
        zt = np.median([z for z, _m in st]) if st else float('nan')
        zf = np.median([z for z, _m in sf]) if sf else float('nan')
        mt = np.median([m for _z, m in st]) if st else float('nan')
        mf = np.median([m for _z, m in sf]) if sf else float('nan')
        print('%-12s %9.1f%% %12.2f %12.2f %8.2f / %.2f'
              % (k, 100 * (ft[1] - ft[0]) / ft[1], zt, zf, mt, mf))
    return 0


if __name__ == '__main__':
    sys.exit(main())
