"""Q1 shuffle control (§20.5, mandatory) — run on the top configs BEFORE the
grid is read as a verdict.

CLAIM UNDER ATTACK: the moment-stack gain is real signal from THIS
conversation's history. THE ARTIFACT IT WOULD BE INSTEAD: a norm/length
effect — any extra vectors, from anywhere, lifting AUC by changing score
statistics. TEST: rebuild each labeled turn's history (j≥1, op AND anchor)
from RANDOM OTHER-SESSION turns, seeded; keep j=0 real; recompute the lanes
fresh (content lanes via the engine's node matrices, idf via _idf_asof at
the turn's own as_of, episodic via the engine-parity role-map path); compose
with the IDENTICAL production math (q1_sweep.compose). Shuffled history must
NOT beat K0 on the same turn subset. If it does, the config is dead
regardless of its grid numbers — and that's the headline.

Bounded as registered: top-3 DISTINCT-shape configs + N_RANDOM random
configs, on a seeded sample of labeled turns.

Run:  ./dev python3 eval/laf/walker/shuffle_control.py
Exit: 0 = control holds; 1 = shuffled history beat K0 somewhere (headline).
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker, WALKER_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, MAXSIM_VIEWS, _unit   # noqa: E402
import q1_sweep                                                    # noqa: E402
from q1_sweep import (GAINS, TurnData, compose, stack_messages,    # noqa: E402
                      weights, configs, auc)
from episodic_roles import (build_role_map, episodic_from_sims,    # noqa: E402
                            K_MAX)

SEED = 20260715
N_TURNS = 400
N_RANDOM_CFGS = 5
TOL = 0.005                    # shuffled may not beat K0 by more than this
REPORT = WALKER_DIR / 'shuffle_control.md'


def top_distinct(results, k=3):
    """Top-k configs with DISTINCT weight shapes (exp0.5/pow1.0 are identical
    at K=1 — dedup by the actual weight vector + axes)."""
    seen, out = set(), []
    for r in sorted(results, key=lambda r: -r['d_val']):
        if r['name'] == 'K0':
            continue
        cfg = next(c for c in configs() if c['name'] == r['name'])
        key = (tuple(np.round(weights(cfg), 6)), cfg['comp'], cfg['agg'],
               cfg['texts'], cfg['me'])
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
        if len(out) >= k:
            break
    return out


def build_shuffled(sample, donors, rng, eng, role_map, trace_mat,
                   trace_created, n):
    """TurnData clones with j≥1 history replaced by random OTHER-session
    donor turns; j=0 stays real; lanes recomputed fresh through production
    code. Extracted for reuse by the P3 fitted-model shuffle (p3_eval) —
    behavior identical to the Q1 run."""
    shuffled = []
    for td in sample:
        rows = np.array([eng._idx.get(nid, -1) for nid in td.cands])
        ok = rows >= 0
        sh = TurnData(td.key, td.ts, td.cands, td.flagged)
        sh.sel, sh.soft, sh.fat = td.sel, td.soft, td.fat
        for ln in GAINS:              # j=0 stays REAL (stored, cross-checked)
            sh.op[ln][:, 0] = td.op[ln][:, 0]
        for j in range(1, K_MAX + 1):
            dsess = dov = dav = dopt = dat = None
            while True:               # donor from a DIFFERENT session
                dsess, dov, dav, dopt, dat = rng.choice(donors)
                if dsess != td.key[0]:
                    break
            for kind, vec, text in (('op', dov, dopt),
                                    ('anchor', dav, dat)):
                tgt = sh.op if kind == 'op' else sh.anchor
                if vec is None:
                    continue
                with np.errstate(all='ignore'):
                    ms = np.nanmax(np.stack(
                        [eng._mats[vt][:n] @ vec for vt in MAXSIM_VIEWS]),
                        axis=0)
                sit = eng._mats['_situation'][:n] @ vec
                idf = eng._idf_asof(text, n, td.ts) if text else None
                ep = episodic_from_sims(eng, role_map, trace_mat @ vec,
                                        td.ts, trace_created)
                pick = np.zeros(n)
                enc = np.zeros(n)
                for r_, (p, e) in ep.items():
                    pick[r_], enc[r_] = p, e
                for i, r_ in enumerate(rows):
                    if not ok[i]:
                        continue
                    tgt['maxsim'][i, j] = ms[r_]
                    tgt['sit'][i, j] = sit[r_]
                    if idf is not None:
                        tgt['idf'][i, j] = idf[r_]
                    tgt['pick'][i, j] = pick[r_]
                    tgt['enc'][i, j] = enc[r_]
        shuffled.append(sh)
    return shuffled


def load_donors(walker):
    """Donor pool: real non-machine turns from every session (op/anchor
    vectors + capped texts) — history stand-ins for the shuffle."""
    donors = []
    for sess, opv, av, opt, at, flags in walker.execute(
            "SELECT session_id, op_vec, anchor_vec, op_text, anchor_text, "
            "flags FROM turns WHERE op_vec IS NOT NULL "
            "AND flags NOT LIKE '%machine_turn%'"):
        donors.append((sess, _unit(opv), _unit(av) if av else None,
                       (opt or '')[:500], (at or '')[:500]))
    return donors


def main():
    rng = random.Random(SEED)
    results = json.loads((WALKER_DIR / 'q1_sweep_full.json').read_text())
    tops = top_distinct(results, 3)
    pool = [c for c in configs() if c['K'] > 0
            and c['name'] not in {t['name'] for t in tops}]
    arms = tops + rng.sample(pool, N_RANDOM_CFGS)
    k0 = configs()[0]

    walker = open_walker()
    q1_sweep.gate_provenance(walker)
    turns = q1_sweep.load(walker)
    donors = load_donors(walker)
    walker.close()

    sample = rng.sample(turns, min(N_TURNS, len(turns)))
    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        role_map = build_role_map(env.brain)
        n = eng._n
        shuffled = build_shuffled(sample, donors, rng, eng, role_map,
                                  trace_mat, trace_created, n)

    # same-subset AUCs: K0, real config, shuffled config
    def subset_auc(tds, cfg):
        w = weights(cfg)
        sel, drp = [], []
        for td in tds:
            s = q1_sweep.score_turn(td, cfg, w)
            sel.append(s[td.sel])
            drp.append(s[~td.sel])
        return auc(np.concatenate(sel), np.concatenate(drp))

    k0_auc = subset_auc(sample, k0)
    lines = ['# shuffle_control — §20.5 mandatory gate', '',
             '- sample: %d labeled turns (seed %d); donors: other-session '
             'non-machine turns; j=0 real, j≥1 donor-replaced, lanes fresh '
             'through production code' % (len(sample), SEED),
             '- K0 AUC on this subset: **%.4f**' % k0_auc, '',
             '| config | real AUC | shuffled AUC | shuffled − K0 | verdict |',
             '|---|---|---|---|---|']
    failed = []
    for cfg in arms:
        real = subset_auc(sample, cfg)
        shuf = subset_auc(shuffled, cfg)
        d = shuf - k0_auc
        ok_ = d <= TOL
        if not ok_:
            failed.append(cfg['name'])
        lines.append('| %s | %.4f | %.4f | %+.4f | %s |'
                     % (cfg['name'], real, shuf, d,
                        'holds' if ok_ else 'SHUFFLE WINS — config dead'))
    lines.append('')
    lines.append('**Overall: %s**' % (
        'CONTROL HOLDS — history gain is not a norm/length artifact'
        if not failed else
        'SHUFFLE BEAT K0 on: %s — those configs are dead regardless of grid '
        'numbers; THIS IS THE HEADLINE' % ', '.join(failed)))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
