"""2D pivot conditioning — refine the failed 1D agreement gate.

gated_mesh.py showed agreement(F0, Fhist) measures REDUNDANCY, not
pivot: low agreement mixes true pivots (peaked F0, new region — history
poisons) with anaphora (flat F0, no signal — history rescues).

Hypothesis: peakedness(F0) separates them.
  peaked + low-ρ  = PIVOT     → moment should hurt here
  flat   + low-ρ  = ANAPHORA  → moment should help most here
  high-ρ (either) = REDUNDANT → mild effect

Validation only — no gate is built until this table says the cells
separate. Conditioning: tercile(ρ) × tercile(peakedness of F0), cells
report mean Δ target-rank (moment − j0-only), %hurt, %helped, and where
the four named eyeball MOMENT-HURT cases land.

Run:  ./dev python3 eval/laf/walker/pivot_2d.py
Out:  pivot_2d.md
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import load, gate_provenance, weights                 # noqa: E402
from moment_grids import cfg_for, message_fields, mesh              # noqa: E402
from gated_mesh import agreements, HURT_CASES                       # noqa: E402

CFG = cfg_for(8, 0.7)
REPORT = OUT_DIR / 'pivot_2d.md'


def peakedness(f0):
    """top1 − median of the finite F0 activations (z-composed units):
    how confidently the current message points at ONE place."""
    fin = f0[np.isfinite(f0)]
    if fin.size < 4:
        return np.nan
    return float(np.max(fin) - np.median(fin))


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()

    w = weights(CFG)
    rows = []
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any():
            continue
        F, ww = message_fields(td, CFG, w)
        rhos, _ = agreements(F)
        fin = rhos[np.isfinite(rhos)]
        if not fin.size:
            continue
        tgt = int(np.nanargmax(td.soft))
        if td.soft[tgt] < 0.5:
            continue
        s1 = mesh(F, ww, 'linear')
        s0 = mesh(F[:, :1], ww[:1], 'linear')

        def rank_of(s, i):
            o = np.argsort(-np.where(np.isfinite(s), s, -np.inf))
            return int(np.where(o == i)[0][0]) + 1
        rows.append({'key': td.key, 'rho': float(fin.mean()),
                     'peak': peakedness(F[:, 0]),
                     'd': rank_of(s1, tgt) - rank_of(s0, tgt)})

    rows = [r for r in rows if np.isfinite(r['peak'])]
    rho_t = np.percentile([r['rho'] for r in rows], [33.3, 66.7])
    pk_t = np.percentile([r['peak'] for r in rows], [33.3, 66.7])

    def tri(v, t):
        return 0 if v <= t[0] else (1 if v <= t[1] else 2)

    lines = ['# pivot_2d — ρ(F0,Fhist) × peakedness(F0) conditioning',
             '',
             '- Δ = moment − j0-only target rank; negative = moment '
             'helped. n=%d val turns.' % len(rows), '',
             '| | ρ low (disagree) | ρ mid | ρ high (redundant) |',
             '|---|---|---|---|']
    grid = {}
    for pb in range(3):
        cells = []
        for rb in range(3):
            sub = [r for r in rows
                   if tri(r['rho'], rho_t) == rb
                   and tri(r['peak'], pk_t) == pb]
            d = np.array([r['d'] for r in sub])
            grid[(pb, rb)] = sub
            cells.append('Δ%+.1f · hurt %d%% · help %d%% · n=%d'
                         % (d.mean(), 100 * (d > 2).mean(),
                            100 * (d < -2).mean(), len(sub))
                         if len(sub) else '—')
        label = ('F0 flat (anaphora?)', 'F0 mid',
                 'F0 peaked (pivot?)')[pb]
        lines.append('| **%s** | %s | %s | %s |'
                     % (label, *cells))
    lines.append('')
    named = []
    for r in rows:
        if (r['key'][0][:8],) + tuple(r['key'][1:]) in HURT_CASES:
            named.append('%s → ρ-tercile %d, peak-tercile %d (Δ%+d)'
                         % (r['key'][0][:8], tri(r['rho'], rho_t) + 1,
                            tri(r['peak'], pk_t) + 1, r['d']))
    lines.append('- named MOMENT-HURT cases: %s' % '; '.join(named))
    lines.append('- validation rule: PIVOT cell (peaked × ρ-low) must '
                 'show hurt >> help relative to ANAPHORA cell (flat × '
                 'ρ-low); otherwise the 2D gate is dead too.')

    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
