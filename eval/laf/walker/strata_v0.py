"""Context-dependence stratification v0 (Tom, 2026-07-21: Moment-golds
don't belong in bare-cue evaluation). Mechanical proxy:
  CUE-SUFFICIENT   gold in F0's top-25 (the bare msg alone reaches it)
  MOMENT-DEPENDENT not cue-sufficient, gold in M_h's top-25 (only the
                   conversation reaches it)
  NEITHER          hard / weak-signal / mislabeled-echo candidates
Reports stratum sizes, per-stratum mix reach@5, cue-length profile, and
strong-tier composition — the honest per-door metrics.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/tpac/brain/eval/laf/walker')
from walker_db import OUT_DIR, open_walker                           # noqa: E402

sys.path.insert(0, '/Users/tpac/brain')
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star                             # noqa: E402

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.65

idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
S = {s: i for i, s in enumerate(idx['slots'])}

walker = open_walker()
oplen = dict(((s, e, q), (ln or 0)) for s, e, q, ln in walker.execute(
    "SELECT session_id, epoch, seq, op_len FROM turns WHERE labeled=1"))
walker.close()


def rank_tf(f, gr):
    if f is None or not np.isfinite(f[gr]):
        return None
    fin = np.where(np.isfinite(f), f, -np.inf)
    return int((fin > f[gr]).sum()) + (int((fin == f[gr]).sum()) - 1) / 2.0 \
        + 1


strata = {k: {'n': 0, 'hit': 0, 'strong': 0, 'lens': []}
          for k in ('CUE-SUFF', 'MOMENT-DEP', 'NEITHER')}
for t in idx['turns']:
    if t.get('skipped'):
        continue
    tt = Turn(t, fields, S)
    if tt.gr < 0 or tt.ro is None or tt.mh is None \
            or np.isnan(tt.fields[0]).all():
        continue
    r_f0 = rank_tf(tt.fields[0], tt.gr)
    r_mh = rank_tf(tt.mh, tt.gr)
    if r_f0 is not None and r_f0 <= 25:
        st = 'CUE-SUFF'
    elif r_mh is not None and r_mh <= 25:
        st = 'MOMENT-DEP'
    else:
        st = 'NEITHER'
    rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr,
                     grid=np.array([LAM]))
    hit = bool(rk) and min(rk.values()) <= 5
    d = strata[st]
    d['n'] += 1
    d['hit'] += int(hit)
    d['strong'] += int(tt.strong)
    d['lens'].append(oplen.get(tt.key, 0))

tot = sum(d['n'] for d in strata.values())
print('λ=%.2f · %d turns' % (LAM, tot))
print('%-11s %6s %7s %9s %10s %12s' % ('stratum', 'n', 'share',
                                       'mix r@5', '%strong', 'cue_len md'))
for k, d in strata.items():
    print('%-11s %6d %6.0f%% %8.1f%% %9.0f%% %10.0f'
          % (k, d['n'], 100 * d['n'] / tot, 100 * d['hit'] / max(1, d['n']),
             100 * d['strong'] / max(1, d['n']), np.median(d['lens'])))
print('\nshort cues (<30 chars) by stratum:')
for k, d in strata.items():
    print('  %-11s %4d short (%.0f%% of stratum)'
          % (k, sum(1 for x in d['lens'] if x < 30),
             100 * sum(1 for x in d['lens'] if x < 30) / max(1, d['n'])))
