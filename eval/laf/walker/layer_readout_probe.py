"""The λ-derivation hunt, step 1 (§21 NEXT, 2026-07-20): per-LAYER readouts.

Question: no COMPOSED-field readout tracks λ* (|ρ|≤0.14) — does the signal
live one level down, in the raw lanes (maxsim/sit/idf/pick/enc) that the
composed field sums away? Three threads:
  1. per-(slot, layer) peak/conc → λ* Spearman — does specificity live in
     idf/sit rather than composed peak?
  2. msg-0 CROSS-LAYER structure — do cosine and episodic layers point at
     the same region? (disagreement → defer to the Moment?)
  3. the CONC-INVERSION autopsy — conc_F0 anti-predicts λ* (−0.141 live):
     WHICH lane's concentration drives it, and is the peak lexical
     (idf-dominated top-50 share = the peaked-but-wrong-spike hypothesis)?

Substrate: lane_cache.npy [turns × 5 slots × 5 lanes × nodes] RAW pre-gain
pre-z values + field_cache.npy composed fields (same frozen index). λ* via
lambda_probe.lambda_star (never re-implement — the wsum rule); composed
mesh via mesh_fit_probe.Turn (wsum inside).

SELF-GATE (hard-fail): the composed field is RECONSTRUCTED from the raw
lanes (same z routing as field_cache_build.compose_slot: support-z for the
sparse zero seas, current-z dense, alive = finite(maxsim)) and compared to
field_cache.npy on the first turns — if the mask/z reconstruction drifts
from the build's, this probe dies instead of printing quotable-but-wrong
correlations.

Run:    ./dev python3 eval/laf/walker/layer_readout_probe.py
Pool60: WALKER_OUT_DIR=~/AgentsContext/eval-corpus/0a9baa/walker ... (same)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import zscore_variant                        # noqa: E402
from q1_sweep import GAINS                                           # noqa: E402
from field_mesh_probe import conc, align, topset                     # noqa: E402
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, spearman, lambda_star, plateau_of       # noqa: E402
from soft_usage import auc                                           # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
LANE_CACHE = OUT_DIR / 'lane_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
SPARSE = ('pick', 'enc', 'idf')       # the zero seas — support-z (build meta)
PARITY_TOL = 5e-3                     # float32 storage round-trip headroom
TOPK = 50                             # composed-peak region for lane shares


def lane_z(raw, lane, alive, n):
    """One lane's z-field, replicating field_cache_build.compose_slot's
    routing exactly: sparse lanes zero-filled then support-z, dense lanes
    as-stored under current-z; dead nodes NaN.

    float64 is LOAD-BEARING, not style: the cache stores float32 but the
    build z-scored the engine's float64 arrays. A tied support (all values
    equal — pool60 pick lanes) has float64 std exactly 0.0 → build zeroes
    the lane via the >1e-9 guard; float32 std of the same tie is ~6e-8
    (mean-rounding ulp) → guard passes → z=±1 garbage across the tie. Any
    future derivation over lane_cache must z in float64 or ties explode."""
    raw = raw.astype(np.float64)
    if lane in SPARSE:
        col = np.where(np.isfinite(raw), raw, 0.0)
        z = zscore_variant(col, n, mask=alive, kind='support')
    else:
        z = zscore_variant(raw, n, mask=alive, kind='current')
    z[~alive] = np.nan
    return z


def msg0_readouts(tt, zf, lanes):
    """The msg-0 readout vector: per-lane peak/conc, cross-layer pair
    structure, lane shares of the composed peak region — plus the composed
    baselines from Turn.ro. THE readout definition; lambda_fit_probe
    imports this so the fit sees exactly what the table reported."""
    ro = dict(tt.ro)                                 # composed baselines
    z0 = {ln: zf.get(('op0', ln)) for ln in lanes}
    present = [ln for ln in lanes if z0[ln] is not None
               and np.isfinite(z0[ln]).any()
               and np.nanstd(z0[ln]) > 1e-9]
    for ln in lanes:
        z = z0[ln]
        ok = z is not None and np.isfinite(z).any()
        ro['peak_%s' % ln] = float(np.nanmax(z)) if ok else np.nan
        ro['conc_%s' % ln] = conc(z) if ok else np.nan
    pair = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            pair[(a, b)] = align(z0[a], z0[b])
    ro['agr_cos_sit'] = pair.get(('maxsim', 'sit'), np.nan)
    epi = [v for k, v in pair.items()
           if 'maxsim' in k and (k[0] in ('pick', 'enc')
                                 or k[1] in ('pick', 'enc'))]
    ro['agr_cos_epi'] = float(np.mean(epi)) if epi else np.nan
    ro['agr_cos_idf'] = pair.get(('idf', 'maxsim'),
                                 pair.get(('maxsim', 'idf'), np.nan))
    ro['disagree'] = (1.0 - float(np.nanmean(list(pair.values())))
                      if pair else np.nan)
    ro['n_lanes'] = float(len(present))
    # lane share of the composed peak region (top-50 of composed F0)
    top = sorted(topset(tt.fields[0], TOPK))
    contrib = {ln: (GAINS[ln] * np.where(np.isfinite(z0[ln][top]),
                                         z0[ln][top], 0.0)).sum()
               if z0[ln] is not None else 0.0 for ln in lanes}
    tot = sum(contrib.values())
    for ln in lanes:
        ro['shr_%s' % ln] = (contrib[ln] / tot if abs(tot) > 1e-9
                             else np.nan)
    return ro


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    lanes_mm = np.load(LANE_CACHE, mmap_mode='r')
    slots, lanes = idx['slots'], idx['lanes']
    S = {s: i for i, s in enumerate(slots)}
    n = idx['n_nodes']
    assert lanes_mm.shape[1:] == (len(slots), len(lanes), n), \
        'lane cache shape mismatch vs index'
    print('turns %d · slots %s · lanes %s · n_nodes %d'
          % (len(idx['turns']), slots, lanes, n))

    lam_mid, decisive, per_turn = [], [], []
    slot_ro = {(s, ln, r): [] for s in slots for ln in lanes
               for r in ('peak', 'conc')}
    parity_worst, parity_checked = 0.0, 0

    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.ro is None or tt.mh is None \
                or np.isnan(tt.fields[0]).all():
            continue
        ranks = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr)
        if not ranks:
            continue
        lo, hi, mid = plateau_of(ranks)
        lam_mid.append(mid)
        decisive.append(hi - lo <= 0.5)

        L = lanes_mm[t['row']].astype(np.float32)   # [slots × lanes × n]
        zf = {}                                      # (slot, lane) → z-field
        for si, sl in enumerate(slots):
            mx = L[si, lanes.index('maxsim')]
            if np.isnan(mx).all():                  # slot absent this turn
                for ln in lanes:
                    slot_ro[(sl, ln, 'peak')].append(np.nan)
                    slot_ro[(sl, ln, 'conc')].append(np.nan)
                continue
            alive = np.isfinite(mx)
            for li, ln in enumerate(lanes):
                z = lane_z(L[si, li], ln, alive, n)
                zf[(sl, ln)] = z
                fin = np.isfinite(z)
                slot_ro[(sl, ln, 'peak')].append(
                    float(np.nanmax(z)) if fin.any() else np.nan)
                slot_ro[(sl, ln, 'conc')].append(conc(z))
            # -- self-gate: composed reconstruction ≡ field_cache
            #    (every turn — z-fields exist anyway; fail fast, named)
            rec = np.zeros(n)
            for ln in lanes:
                rec += GAINS[ln] * np.where(
                    np.isfinite(zf[(sl, ln)]), zf[(sl, ln)], 0.0)
            rec[~alive] = np.nan
            stored = fields[t['row'], si].astype(np.float64)
            both = np.isfinite(rec) & np.isfinite(stored)
            assert (np.isfinite(rec) == np.isfinite(stored)).all(), \
                'PARITY: alive-mask mismatch at %s slot %s' % (tt.key, sl)
            d = float(np.abs(rec[both] - stored[both]).max()) \
                if both.any() else 0.0
            assert d < PARITY_TOL, \
                'PARITY FAIL |Δ| %.4f at %s slot %s' % (d, tt.key, sl)
            parity_worst = max(parity_worst, d)
        parity_checked += 1

        per_turn.append(msg0_readouts(tt, zf, lanes))

    print('self-gate: composed reconstruction parity (%d turns, all '
          'slots) worst |Δ| %.2e  OK' % (parity_checked, parity_worst))

    lam = np.array(lam_mid)
    dec = np.array(decisive)
    print('turns %d · decisive %d (%.0f%%)'
          % (len(lam), dec.sum(), 100 * dec.mean()))

    def col(k):
        return np.array([r.get(k, np.nan) for r in per_turn])

    # ── 1. per-(slot, layer) readouts → λ* ─────────────────────────────
    print('\n== 1. per-(slot, layer) readouts → λ* (Spearman, decisive '
          'turns) ==')
    print('  %-8s %-7s  peak      conc' % ('slot', 'lane'))
    for sl in slots:
        for ln in lanes:
            pk = np.array(slot_ro[(sl, ln, 'peak')])
            cc = np.array(slot_ro[(sl, ln, 'conc')])
            print('  %-8s %-7s %+0.3f    %+0.3f'
                  % (sl, ln, spearman(pk[dec], lam[dec]),
                     spearman(cc[dec], lam[dec])))

    # ── 2. msg-0 readout table: λ* Spearman + AUC(λ*≤0.3) + group means ─
    keys = ['peak_F0', 'conc_F0', 'peak_Mh', 'conc_Mh', 'ov_F0_F1']
    keys += ['peak_%s' % ln for ln in lanes] + ['conc_%s' % ln
                                                for ln in lanes]
    keys += ['agr_cos_sit', 'agr_cos_epi', 'agr_cos_idf', 'disagree',
             'n_lanes'] + ['shr_%s' % ln for ln in lanes]
    lo_g = dec & (lam <= 0.3)
    md_g = dec & (lam > 0.3) & (lam < 0.7)
    hi_g = dec & (lam >= 0.7)
    print('\n== 2. msg-0 readouts → λ* (decisive n=%d; groups λ*≤0.3 '
          'n=%d · mid n=%d · ≥0.7 n=%d) =='
          % (dec.sum(), lo_g.sum(), md_g.sum(), hi_g.sum()))
    print('  %-13s Spearman  AUC(λ≤.3)   mean λlo   mean mid   mean λhi'
          % 'readout')
    rows = []
    for k in keys:
        v = col(k)
        sp = spearman(v[dec], lam[dec])
        m = np.isfinite(v)
        a = (auc(v[m & lo_g], v[m & (dec & ~lo_g)])
             if (m & lo_g).sum() >= 10 else np.nan)
        rows.append((abs(sp) if np.isfinite(sp) else 0, k, sp, a,
                     np.nanmean(v[lo_g]), np.nanmean(v[md_g]),
                     np.nanmean(v[hi_g])))
    for _, k, sp, a, ml, mm, mh in sorted(rows, reverse=True):
        print('  %-13s %+0.3f     %s      %8.3f   %8.3f   %8.3f'
              % (k, sp, ('%0.3f' % a) if np.isfinite(a) else '  -  ',
                 ml, mm, mh))

    # ── 3. conc-inversion autopsy: what is a concentrated msg-0 made of ─
    cf = col('conc_F0')
    m = dec & np.isfinite(cf)
    ter = np.nanpercentile(cf[m], [33, 67])
    lo_c, hi_c = m & (cf <= ter[0]), m & (cf >= ter[1])
    print('\n== 3. conc-inversion autopsy (decisive; conc_F0 terciles: '
          'low n=%d vs high n=%d) ==' % (lo_c.sum(), hi_c.sum()))
    print('  %-13s  low-conc   high-conc' % 'mean of')
    for k in (['lam'] + ['shr_%s' % ln for ln in lanes]
              + ['conc_%s' % ln for ln in lanes]
              + ['agr_cos_epi', 'agr_cos_idf', 'disagree', 'n_lanes',
                 'peak_F0']):
        v = lam if k == 'lam' else col(k)
        print('  %-13s %9.3f  %9.3f'
              % ('λ*' if k == 'lam' else k,
                 np.nanmean(v[lo_c]), np.nanmean(v[hi_c])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
