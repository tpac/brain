"""MISS ANATOMY (Tom's Q5+Q6+Q8, 2026-07-21): for every miss of the static
mix, decompose WHERE the gold actually lives across everything we hold, and
reverse-engineer how to get there — or learn that we can't.

Per miss (gold outside top-5 of the λ-mix), the gold's rank in:
  - the 5 slot fields (F0, f1, a1, f2, a2) — composed cache
  - the 5 msg-0 lanes (maxsim/sit/idf/pick/enc) — raw lane cache, z'd
  - M_h and the mix itself
then classify by the BEST rank any held field/lane gives it:
  REACHABLE   best <= 5     a remix could have surfaced it — meshing/λ problem;
                            the argmin field is the reverse-engineering map
  ALMOST      best <= 25    in candidate range somewhere — selection problem
  BURIED      best <= 100   present but weak — calibration/normalization
  BARELY      best  > 100   weak EVERYWHERE we look — NOT a meshing problem:
                            Tom's Q6 tripwire — back to optimizing LAF itself
                            (lanes, views, embedding, encode-side)

Plus the Q8 half: stratified CASE STORIES (msg → gold → rank table → who
displaced it and which lane powers each displacer → rank-vs-λ trajectory)
written to OUT_DIR/miss_anatomy.md for joint mechanism-hunting.

Machinery: Turn (per-msg kernel), lane_z, lambda_star — imported, never
re-implemented. Pool60 NEEDS BRAIN_DB_DIR (node metadata reads the brain).

Run:    ./dev python3 eval/laf/walker/miss_anatomy.py
Pool60: BRAIN_DB_DIR=~/AgentsContext/eval-corpus/0a9baa/pooled \
        WALKER_OUT_DIR=~/AgentsContext/eval-corpus/0a9baa/walker ...
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker, open_brain_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import GAINS                                           # noqa: E402
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star, GRID                       # noqa: E402
from layer_readout_probe import lane_z                               # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
LANE_CACHE = OUT_DIR / 'lane_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
REPORT = OUT_DIR / 'miss_anatomy.md'
CLASSES = (('REACHABLE', 5), ('ALMOST', 25), ('BURIED', 100),
           ('BARELY', 10 ** 9))
STORY_PER_CLASS = 6
LAM_TRAJ = (0.0, 0.25, 0.5, 0.75, 1.0)
SNIP = 200


def rank_in(f, gr):
    """TIE-FAIR rank: mid-rank among ties. Strict `>` gave a gold tied
    with 200 nodes at a sparse lane's top value 'rank 1' — 42% of misses
    had >50-node ties at their best lane, inflating REACHABLE 36→24% and
    understating BARELY 0.1→8.8% (review BLOCKER, 2026-07-21). Continuous
    fields are unaffected (ties are measure-zero there)."""
    if f is None or not np.isfinite(f[gr]):
        return None
    fin = np.where(np.isfinite(f), f, -np.inf)
    greater = int((fin > f[gr]).sum())
    ties = int((fin == f[gr]).sum())          # includes the gold itself
    return greater + (ties - 1) / 2.0 + 1


def classify(best):
    for name, cap in CLASSES:
        if best <= cap:
            return name
    return 'BARELY'


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
    optext = dict(((s, e, q), (op or '', ts)) for s, e, q, ts, op in
                  walker.execute("SELECT session_id, epoch, seq, ts, "
                                 "op_text FROM turns WHERE labeled=1"))
    walker.close()

    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr >= 0 and tt.ro is not None and tt.mh is not None \
                and not np.isnan(tt.fields[0]).all():
            turns.append((t, tt))

    # corpus static λ (same frame as derivation_audit)
    per_l = {l: 0 for l in GRID}
    for _t, tt in turns:
        for l, r in lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr).items():
            per_l[l] += int(r <= 5)
    lam_s = max(GRID, key=lambda l: per_l[l])
    print('turns %d · static λ=%.2f (%.1f%% @5)'
          % (len(turns), lam_s, 100 * per_l[lam_s] / len(turns)))

    # ── pass: per-miss anatomy ──────────────────────────────────────────
    rows = []                        # one record per miss
    n_miss = n_hit = 0
    for t, tt in turns:
        gr = tt.gr
        f0z, mhz = zn(tt.fields[0]), zn(tt.mh)
        rk = lambda_star(f0z, mhz, gr, grid=np.array([lam_s]))
        mix_rank = min(rk.values()) if rk else None
        if mix_rank is None:
            continue
        if mix_rank <= 5:
            n_hit += 1
            continue
        n_miss += 1
        L = lanes_mm[t['row']].astype(np.float32)
        mx = L[S['op0'], lanes.index('maxsim')]
        alive = np.isfinite(mx)
        z0 = {ln: lane_z(L[S['op0'], li], ln, alive, n)
              for li, ln in enumerate(lanes)}
        ranks = {}
        for nm, f in zip(('F0', 'f1', 'a1', 'f2', 'a2'), tt.fields):
            ranks[nm] = rank_in(f, gr)
        for ln in lanes:
            ranks['lane_' + ln] = rank_in(z0[ln], gr)
        ranks['M_h'] = rank_in(tt.mh, gr)
        held = {k: v for k, v in ranks.items() if v is not None}
        best_k, best = (min(held, key=held.get), min(held.values())) \
            if held else (None, 10 ** 9)
        # λ trajectory of the gold
        traj = {l: (min(lambda_star(f0z, mhz, gr,
                                    grid=np.array([l])).values()
                        or [None]))
                for l in LAM_TRAJ}
        # displacers: top-3 of the mix field + their powering lane.
        # Attribution covers BOTH mix sides (review MEDIUM 2026-07-21):
        # λ·zn(F0)[d] decomposes into λ·gain·z_lane[d]/σ(F0) terms (zn is
        # affine; the shared shift drops out of the per-node argmax), and
        # the Moment side enters as an 'M_h' pseudo-lane at (1−λ)·mhz[d].
        both = np.isfinite(f0z) & np.isfinite(mhz)
        mixf = np.where(both, (1 - lam_s) * mhz + lam_s * f0z, -np.inf)
        top3 = np.argsort(-mixf)[:3]
        sig0 = float(np.nanstd(tt.fields[0])) or 1.0
        disp = []
        for d in top3:
            contrib = {ln: float(lam_s * GAINS[ln]
                                 * (z0[ln][d] if np.isfinite(z0[ln][d])
                                    else 0.0) / sig0) for ln in lanes}
            contrib['M_h'] = float((1 - lam_s) * mhz[d]) \
                if np.isfinite(mhz[d]) else 0.0
            disp.append((int(d), max(contrib, key=contrib.get)))
        rows.append({'key': tt.key, 'gr': gr, 'mix_rank': mix_rank,
                     'ranks': ranks, 'best': best, 'best_k': best_k,
                     'cls': classify(best), 'strong': tt.strong,
                     'traj': traj, 'disp': disp,
                     'row': t['row'], 'ts': t.get('ts', '')})

    print('hits %d · misses %d' % (n_hit, n_miss))

    # node metadata for golds + displacers
    need = {master[r['gr']] for r in rows}
    need.update(master[d] for r in rows for d, _ in r['disp'])
    bro = open_brain_ro()
    meta = {}
    for chunk in (list(need)[i:i + 500] for i in range(0, len(need), 500)):
        for nid, title, typ, cat in bro.execute(
                'SELECT id, title, type, created_at FROM nodes '
                'WHERE id IN (%s)' % ','.join('?' * len(chunk)), chunk):
            meta[nid] = (title, typ, cat)
    bro.close()

    # ── summary ─────────────────────────────────────────────────────────
    print('\n== miss classes (best rank across ALL held fields/lanes) ==')
    print('  %-10s %6s %6s   %s' % ('class', 'n', 'share', 'of strong-tier '
                                    'misses'))
    ns_miss = sum(r['strong'] for r in rows)
    for name, _cap in CLASSES:
        sub = [r for r in rows if r['cls'] == name]
        st = sum(r['strong'] for r in sub)
        print('  %-10s %6d %5.0f%%   %5.0f%%'
              % (name, len(sub), 100 * len(sub) / max(1, n_miss),
                 100 * st / max(1, ns_miss)))

    print('\n== REACHABLE: which held field reaches the gold '
          '(the remix map) ==')
    reach = [r for r in rows if r['cls'] == 'REACHABLE']
    for k, c in Counter(r['best_k'] for r in reach).most_common():
        print('  %-12s %4d (%.0f%%)' % (k, c, 100 * c / max(1, len(reach))))

    print('\n== BARELY (Q6 tripwire): gold age + type ==')
    barely = [r for r in rows if r['cls'] == 'BARELY']
    ages, typs = [], Counter()
    for r in barely:
        title, typ, cat = meta.get(master[r['gr']], ('?', '?', None))
        typs[typ] += 1
        if cat and r['ts']:
            d = iso_days(r['ts'], cat)
            if d is not None:
                ages.append(d)
    if ages:
        a = np.array(ages)
        print('  age: median %.0fd · <1d %.0f%% · >7d %.0f%%'
              % (np.median(a), 100 * (a < 1).mean(), 100 * (a > 7).mean()))
    print('  types: ' + ' · '.join('%s %d' % kv
                                   for kv in typs.most_common(6)))
    print('\n== displacer power (which lane powers the mix top-3 on '
          'misses) ==')
    for ln, c in Counter(ln for r in rows
                         for _d, ln in r['disp']).most_common():
        print('  %-8s %5d (%.0f%%)' % (ln, c, 100 * c / (3 * n_miss)))

    # ── case stories ────────────────────────────────────────────────────
    def story(r):
        gid = master[r['gr']]
        title, typ, cat = meta.get(gid, ('?', '?', None))
        key = tuple(r['key'])
        op, ts = optext.get(key, ('', r['ts']))
        age = iso_days(ts, cat) if (cat and ts) else None
        lines = ['### %s · %s · mix rank %d · class %s'
                 % ('/'.join(map(str, key)), (ts or '')[:16],
                    r['mix_rank'], r['cls']),
                 '**Tom:** %s' % op[:SNIP].replace('\n', ' '),
                 '**GOLD [%s]** %s — age %sd' % (typ, title, age),
                 '**gold rank per field:** '
                 + ' · '.join('%s %s' % (k, v if v is not None else '—')
                              for k, v in r['ranks'].items()),
                 '**rank vs λ:** '
                 + ' '.join('%.2f:%s' % (l, r['traj'][l])
                            for l in LAM_TRAJ),
                 '**displaced by:**']
        for d, ln in r['disp']:
            t2, ty2, _c2 = meta.get(master[d], ('?', '?', None))
            lines.append('  - [%s] %s ← powered by %s' % (ty2, t2, ln))
        return '\n'.join(lines)

    out = ['# Miss anatomy — %s · λ=%.2f · %d misses of %d turns\n'
           % (OUT_DIR, lam_s, n_miss, n_hit + n_miss)]
    for name, _cap in CLASSES:
        sub = sorted([r for r in rows if r['cls'] == name],
                     key=lambda r: r['best'])
        take = (sub[:STORY_PER_CLASS // 2]
                + sub[len(sub) // 2:len(sub) // 2 + STORY_PER_CLASS // 2]) \
            if len(sub) > STORY_PER_CLASS else sub
        out.append('\n## %s — %d misses\n' % (name, len(sub)))
        out += [story(r) for r in take]
    REPORT.write_text('\n\n'.join(out))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
