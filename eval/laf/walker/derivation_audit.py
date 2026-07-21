"""The DERIVATION AUDIT (Tom's ask, 2026-07-20 eve): what derives from
what, what influences what — and which of our CONSTANT parameters deserve
λ-ification (per-event derivation) vs are legitimately constant.

The λ arc found one constant (the msg0/Moment blend) hiding +7.6pp of
per-turn plasticity. This probe asks the same question of EVERY constant
in the composition, plus maps the redundancy structure of the lanes/slots
themselves. Two halves:

  1. INFLUENCE / DERIVABILITY MAP
     - per-lane (msg 0): solo reach, ablation drop, R²(lane | other lanes)
       — a lane with high R² is derivable from the others (its constant
       gain is doing redundant work); a lane with low R² and real ablation
       drop is an independent signal source.
     - per-slot (Moment): ablation/solo of f1/a1/f2 inside M_h, and
       R²(F0 | history slots) — 'how much of msg 0 is already in the
       Moment' as a field-level quantity, correlated against λ* (readout
       candidate).

  2. CONSTANTS ORACLE AUDIT (the λ-pattern, generalized)
     For each constant: per-turn oracle over a small variant grid vs the
     static best — headroom = evidence the constant should be DERIVED,
     flat = evidence it is legitimately constant.
       λ        21-point mix grid (reference row — the known +7.6pp)
       γ        history decay {0.25, 0.5, 0.75, 1.0}
       slots    drop-one / solo of (f1, a1, f2) inside M_h
       gain_ln  ×{0, 0.5, 2} per lane inside F0's compose (one at a time)
     Scored in the λ-mix frame at the corpus's static λ* (pre-pass), gold
     rank over the full alive field, reach@5.

Machinery: lane_z + msg0 spot-parity (layer_readout_probe), wsum
(field_mesh_probe), lambda_star/zn/plateau_of (lambda_probe), Turn
(mesh_fit_probe) — never re-implement (the wsum rule).

Run:    ./dev python3 eval/laf/walker/derivation_audit.py
Pool60: WALKER_OUT_DIR=~/AgentsContext/eval-corpus/0a9baa/walker ... (same)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import GAINS                                           # noqa: E402
from field_mesh_probe import wsum                                    # noqa: E402
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, spearman, lambda_star, plateau_of, GRID  # noqa: E402
from layer_readout_probe import lane_z, PARITY_TOL                   # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
LANE_CACHE = OUT_DIR / 'lane_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
GAMMA = 0.5                       # per-MSG decay step (aee1772e kernel)
GAMMAS = (0.25, 0.5, 0.75, 1.0)
MULTS = (0.0, 0.5, 2.0)           # one-at-a-time gain multipliers
SPOT_PARITY = 20                  # recomposition spot-gate (proven substrate)


def rank1(f0z, mhz, gr, lam):
    """Gold rank of the λ-mixed field at one λ (via THE λ machinery)."""
    rk = lambda_star(f0z, mhz, gr, grid=np.array([round(float(lam), 4)]))
    return min(rk.values()) if rk else None


def r2_of(y, X):
    """R² of y ~ [1, X] over rows where everything is finite."""
    m = np.isfinite(y)
    for c in X:
        m &= np.isfinite(c)
    if m.sum() < 30:
        return np.nan
    yv = y[m]
    if yv.std() < 1e-9:
        return np.nan
    Xb = np.column_stack([np.ones(m.sum())] + [c[m] for c in X])
    beta, *_ = np.linalg.lstsq(Xb, yv, rcond=None)
    res = yv - Xb @ beta
    return float(1.0 - (res ** 2).sum() / ((yv - yv.mean()) ** 2).sum())


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    lanes_mm = np.load(LANE_CACHE, mmap_mode='r')
    slots, lanes = idx['slots'], idx['lanes']
    S = {s: i for i, s in enumerate(slots)}
    n = idx['n_nodes']

    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr >= 0 and tt.ro is not None and tt.mh is not None \
                and not np.isnan(tt.fields[0]).all():
            turns.append((t, tt))
    print('turns %d' % len(turns))

    # ── pre-pass: corpus static λ (the frame every audit scores in) ────
    per_l = {l: 0 for l in GRID}
    for _t, tt in turns:
        rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr)
        for l, r in rk.items():
            per_l[l] += int(r <= 5)
    lam_s = max(GRID, key=lambda l: per_l[l])
    print('static λ* = %.2f (%.1f%% @5) — the audit frame'
          % (lam_s, 100 * per_l[lam_s] / len(turns)))

    # ── main pass ──────────────────────────────────────────────────────
    lane_stats = {ln: {'solo': [], 'ablate': [], 'r2': []} for ln in lanes}
    slot_stats = {s: {'ablate': [], 'solo': []}
                  for s in ('a1', 'f1', 'a2', 'f2')}
    r2_f0_hist, base_ranks = [], []
    oracle = {}                       # audit name → list of per-turn ranks
    lam_mid, lam_dec, gam_mid = [], [], []   # aligned with base_ranks
    parity_worst = 0.0

    def orank(name, r):
        oracle.setdefault(name, []).append(r if r is not None else 10 ** 9)

    for ti, (t, tt) in enumerate(turns):
        f0, f1, a1, f2, a2 = tt.fields
        gr = tt.gr
        f0z, mhz = zn(f0), zn(tt.mh)
        rb = rank1(f0z, mhz, gr, lam_s)
        if rb is None:
            continue
        base_ranks.append(rb)
        rk_l = lambda_star(f0z, mhz, gr)
        lo, hi, mid = plateau_of(rk_l)
        lam_mid.append(mid)
        lam_dec.append(hi - lo <= 0.5)

        L = lanes_mm[t['row']].astype(np.float32)
        z0 = {}
        mx = L[S['op0'], lanes.index('maxsim')]
        alive = np.isfinite(mx)
        for li, ln in enumerate(lanes):
            z0[ln] = lane_z(L[S['op0'], li], ln, alive, n)
        # spot parity: standard-gain recomposition ≡ cached composed F0
        if ti < SPOT_PARITY:
            rec = np.zeros(n)
            for ln in lanes:
                rec += GAINS[ln] * np.where(np.isfinite(z0[ln]),
                                            z0[ln], 0.0)
            rec[~alive] = np.nan
            both = np.isfinite(rec) & np.isfinite(f0)
            d = float(np.abs(rec[both] - f0[both]).max()) if both.any() \
                else 0.0
            assert d < PARITY_TOL, 'PARITY FAIL |Δ| %.4f at %s' \
                % (d, tt.key)
            parity_worst = max(parity_worst, d)

        # ---- 1a. lane influence (msg 0)
        for ln in lanes:
            zl = z0[ln]
            if np.isfinite(zl).any() and np.nanstd(zl) > 1e-9:
                rs = rank1(zn(zl), mhz, gr, lam_s)
                lane_stats[ln]['solo'].append(
                    (rs or 10 ** 9) <= 5)
            v = np.zeros(n)
            for l2 in lanes:
                if l2 == ln:
                    continue
                v += GAINS[l2] * np.where(np.isfinite(z0[l2]), z0[l2], 0.0)
            v[~alive] = np.nan
            ra = rank1(zn(v), mhz, gr, lam_s)
            lane_stats[ln]['ablate'].append((ra or 10 ** 9) <= 5)
            lane_stats[ln]['r2'].append(
                r2_of(z0[ln], [z0[l2] for l2 in lanes if l2 != ln]))

        # ---- 1b. slot influence + R²(F0 | history)
        # per-MSG weights (aee1772e): a1=γ, f1=γ², a2=γ³, f2=γ⁴
        def have(x):
            return x is not None and not np.isnan(x).all()
        havef1, havea1, havef2 = have(f1), have(a1), have(f2)
        parts = {'a1': (GAMMA, a1) if havea1 else None,
                 'f1': (GAMMA ** 2, f1) if havef1 else None,
                 'a2': (GAMMA ** 3, a2) if have(a2) else None,
                 'f2': (GAMMA ** 4, f2) if havef2 else None}
        for s in slot_stats:
            if parts[s] is None:
                continue
            drop = wsum([p for k, p in parts.items()
                         if p is not None and k != s])
            if drop is not None:
                rd = rank1(f0z, zn(drop), gr, lam_s)
                slot_stats[s]['ablate'].append((rd or 10 ** 9) <= 5)
            solo = wsum([parts[s]])
            rs = rank1(f0z, zn(solo), gr, lam_s)
            slot_stats[s]['solo'].append((rs or 10 ** 9) <= 5)
        r2_f0_hist.append(r2_of(
            f0, [x for x in (f1, a1, a2, f2) if have(x)])
            if (havef1 or havea1 or havef2) else np.nan)

        # ---- 2. constants oracle audit
        orank('λ (mix, K=21)', min(rk_l.values()) if rk_l else None)
        # γ audit (per-msg powers)
        best_g, gam_ranks = None, {}
        for g in GAMMAS:
            mv = wsum([(g, a1), (g ** 2, f1), (g ** 3, a2), (g ** 4, f2)])
            if mv is None:
                continue
            r = rank1(f0z, zn(mv), gr, lam_s)
            if r is not None:
                gam_ranks[g] = r
                best_g = r if best_g is None else min(best_g, r)
        orank('γ (decay, K=4)', best_g)
        if gam_ranks:
            bg = min(gam_ranks.values())
            pl = [g for g, r in gam_ranks.items() if r == bg]
            gam_mid.append(float(np.median(pl)))
        else:
            gam_mid.append(np.nan)
        # slot-weight audit: baseline + drop-one + solo (within M_h)
        cands = [rb]
        for s in slot_stats:
            if parts[s] is None:
                continue
            drop = wsum([p for k, p in parts.items()
                         if p is not None and k != s])
            if drop is not None:
                r = rank1(f0z, zn(drop), gr, lam_s)
                if r is not None:
                    cands.append(r)
            r = rank1(f0z, zn(wsum([parts[s]])), gr, lam_s)
            if r is not None:
                cands.append(r)
        orank('slots (drop/solo, K=9)', min(cands))
        # per-lane gain audit (F0 side, one at a time)
        for ln in lanes:
            cands = [rb]
            for m in MULTS:
                v = np.zeros(n)
                for l2 in lanes:
                    g = GAINS[l2] * (m if l2 == ln else 1.0)
                    v += g * np.where(np.isfinite(z0[l2]), z0[l2], 0.0)
                v[~alive] = np.nan
                if np.nanstd(v) < 1e-9:
                    continue
                r = rank1(zn(v), mhz, gr, lam_s)
                if r is not None:
                    cands.append(r)
            orank('gain_%s (×0/½/2, K=4)' % ln, min(cands))

    nb = len(base_ranks)
    base5 = 100 * np.mean(np.array(base_ranks) <= 5)
    print('spot parity (%d turns): worst |Δ| %.2e OK' % (SPOT_PARITY,
                                                         parity_worst))

    print('\n== 1a. lane influence map (msg 0; reach@5 in the λ-mix '
          'frame, baseline %.1f%%) ==' % base5)
    print('  %-8s  solo@5    ablated@5   Δablate    R²(lane|others)'
          % 'lane')
    for ln in lanes:
        st = lane_stats[ln]
        solo = 100 * np.mean(st['solo']) if st['solo'] else np.nan
        abl = 100 * np.mean(st['ablate'])
        r2 = np.nanmean(st['r2'])
        print('  %-8s  %5.1f%%     %5.1f%%     %+5.1fpp       %.3f'
              % (ln, solo, abl, abl - base5, r2))

    print('\n== 1b. slot influence map (inside M_h; baseline %.1f%%) =='
          % base5)
    print('  %-4s  drop@5    solo@5' % 'slot')
    for s in ('a1', 'f1', 'a2', 'f2'):
        st = slot_stats[s]
        print('  %-4s  %5.1f%%    %5.1f%%'
              % (s, 100 * np.mean(st['ablate']) if st['ablate'] else np.nan,
                 100 * np.mean(st['solo']) if st['solo'] else np.nan))
    r2h = np.array(r2_f0_hist)
    lam_mid = np.array(lam_mid)
    lam_dec = np.array(lam_dec)
    print('  R²(F0 | history slots): mean %.3f · → λ* Spearman %+0.3f '
          '(decisive turns)'
          % (np.nanmean(r2h), spearman(r2h[lam_dec], lam_mid[lam_dec])))

    print('\n== 2. constants oracle audit (frame: λ=%.2f static; '
          'n=%d) ==' % (lam_s, nb))
    print('  %-24s static@5   oracle@5   headroom' % 'constant (grid)')
    order = ['λ (mix, K=21)', 'γ (decay, K=4)', 'slots (drop/solo, K=9)'] \
        + ['gain_%s (×0/½/2, K=4)' % ln for ln in lanes]
    for name in order:
        rr = np.array(oracle.get(name, []))
        if not len(rr):
            continue
        o5 = 100 * np.mean(rr <= 5)
        print('  %-24s  %5.1f%%     %5.1f%%    %+5.1fpp'
              % (name, base5, o5, o5 - base5))

    gm = np.array(gam_mid)
    m = np.isfinite(gm)
    print('\n  γ*-mid distribution (n=%d): %s'
          % (m.sum(), ' '.join('%.2f:%d' % (g, int((gm[m] == g).sum()))
                               for g in sorted(set(gm[m])))))
    print('  γ* → λ* Spearman (decisive): %+0.3f'
          % spearman(gm[lam_dec & m], lam_mid[lam_dec & m]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
