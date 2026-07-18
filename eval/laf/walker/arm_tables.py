"""§20.18 arm tables — definitive_fit.json weights → engine moment_gains blobs.

An arm is a gain table, not code (the recall_laf moment wiring resolves every
lane's gain through the table). This is the ONE converter from the fit's
'{lane}·{op|anchor}{j}' weight keys to the engine's '{lane}_{o|a}{j}' config
keys, so Leg B arm configs and the eventual Stage-3 K-store flip are generated
from the frozen fit — never hand-typed, never refit on an eval corpus.

Engine-expressible arms (S_content family — content lanes only; S_full's
pick/enc SLOT weights are not wired, so S_full/A1s stays a Leg-A-only arm):
  A0f  fitted-K0: o0 keys + pick/enc zeros, moment_K=0 (no stack pulled)
  A1   S_content full table @ K=8 (THE hypothesis)
  A1t  A1 with |w| < 0.10 dropped (pre-registered trim robustness check)
  A1a  additive: j≥1 keys only, production j0/pick/enc gains untouched

Every table pins z_norm='current' — the fit z-scored lanes with production's
_zscore (definitive_fit.py line ~57). The fit's M_e_f term is engine fatigue,
which runs downstream of scores() in brain_recall — deliberately NOT in the
table (its scope question is ba05383d, sequenced after the value check).

Run:  ./dev python3 eval/laf/walker/arm_tables.py          # print all arms
      ./dev python3 eval/laf/walker/arm_tables.py A1        # one arm's JSON
"""
import json
import re
import sys
from pathlib import Path

FIT = Path(__file__).resolve().parent / 'definitive_fit.json'
TRIM_ABS = 0.10               # pre-registered (§20.18): A1t drops |w| < this
K = 8                         # the fit's slot horizon
CONTENT_LANES = ('maxsim', 'sit', 'idf')
_KEY = re.compile(r'^(%s)·(op|anchor)(\d+)$' % '|'.join(CONTENT_LANES))


def s_content_weights():
    d = json.loads(FIT.read_text())
    w = d['weights']['S_content']
    out = {}
    for k, v in w.items():
        m = _KEY.match(k)
        if not m:                      # M_e_f and any non-lane term: not a
            continue                   # gain-table entry (see module doc)
        lane, side, j = m.group(1), m.group(2), int(m.group(3))
        if side == 'anchor' and j == 0:
            raise ValueError('a0 in the fit — the temporal-leak rule (W3) '
                             'says anchor joins at j>=1 only; refusing')
        out['%s_%s%d' % (lane, 'o' if side == 'op' else 'a', j)] = round(v, 4)
    if not any(k.endswith('_o0') for k in out):
        raise ValueError('no o0 keys in S_content weights — wrong fit file?')
    return out


def arm(name):
    w = s_content_weights()
    zeros = {'pick': 0.0, 'enc': 0.0}    # third consecutive quality null
    if name == 'A0f':
        table = {k: v for k, v in w.items() if k.endswith('_o0')}
        return {'moment_K': 0, 'moment_gains': {**table, **zeros},
                'z_norm': 'current'}
    if name == 'A1':
        return {'moment_K': K, 'moment_gains': {**w, **zeros},
                'z_norm': 'current'}
    if name == 'A1t':
        table = {k: v for k, v in w.items() if abs(v) >= TRIM_ABS}
        return {'moment_K': K, 'moment_gains': {**table, **zeros},
                'z_norm': 'current'}
    if name == 'A1a':
        table = {k: v for k, v in w.items() if not k.endswith('_o0')}
        return {'moment_K': K, 'moment_gains': table, 'z_norm': 'current'}
    if name == 'A1stk':
        # stack-only: o0 gains zeroed — the walker-faithful ENDO shape
        # (moment_grids stop_weights: at stop time the cue text is the
        # just-finished response, already the stack's freshest member;
        # scoring it again at j0 double-counts = self-echo)
        table = {k: (0.0 if k.endswith('_o0') else v) for k, v in w.items()}
        return {'moment_K': K, 'moment_gains': {**table, **zeros},
                'z_norm': 'current'}
    if name == 'A1k3':
        # depth-trimmed exploratory table (2026-07-18 dev20 depth cells:
        # seq4+ hurts, Leg A soft_r identical to A1) — slots j>3 dropped,
        # same frozen weights otherwise
        table = {k: v for k, v in w.items()
                 if int(re.search(r'\d+$', k).group()) <= 3}
        return {'moment_K': 3, 'moment_gains': {**table, **zeros},
                'z_norm': 'current'}
    raise ValueError('unknown arm %r (A0f/A1/A1t/A1a/A1k3/A1stk)' % name)


ARMS = ('A0f', 'A1', 'A1t', 'A1a', 'A1k3', 'A1stk')


def main():
    names = sys.argv[1:] or ARMS
    for name in names:
        cfg = arm(name)
        mg = cfg['moment_gains']
        slots = sorted({k.rsplit('_', 1)[-1] for k in mg
                        if '_' in k and k.split('_')[0] in CONTENT_LANES})
        print('── %s: %d gain keys, slots %s' % (name, len(mg), slots))
        print(json.dumps(cfg, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
