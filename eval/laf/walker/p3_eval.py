"""P3.2 — pre-declared evaluations on the fitted models (§20.13).

Runs the engine-side (blind-judged) and control evaluations that p3_fit's
rank-leg table can't provide. Arms evaluated: A_full_picked (the registered
primary) and F_soft_ablate (the quality candidate the ablation matrix
surfaced: soft-target, content-only — its soft_r 0.430 ≈ B's 0.427, so the
quality signal needs no pick/enc). Static references K0 / Q1-winner are
recomputed in-run on the identical substrate.

Evaluations (numbering = §20.13 P3.2):
  (2) gold-24 miss-class deltas — near_miss + lane_buried should shrink;
      **unreachable must NOT move (leak canary)**. Classes are computed with
      the IDENTICAL lane attribution as q1_reverse (winner-config lane
      percentiles — substrate properties, model-independent); only the
      blend rank changes per arm. Any unreachable-class node entering an
      arm's top-25 is flagged explicitly (the leak smell).
  (3) tier placement re-run (blind-judged tiers, q1_tiers table shape).
  (4) shuffle control on the FITTED models: donor-replaced j≥1 history
      (shuffle_control.build_shuffled, same seed policy) must not beat the
      SAME coefficients restricted to j0 (empty history ≡ j0-restriction —
      the fitted model's K0-equivalent) by more than TOL.
  (1)(5) live in p3_fit.md (rank-leg AUC vs statics; soft_r column).
  (6) ship gate = P1 discipline — runs at ACTIVATION time (H4 + frame_replay
      + latency), not here; both deliverables stay DORMANT until then.

M_e_f is omitted on the engine leg (gold sessions are walker-EXCLUDED by
manifest — no fatigue replay state exists; Q1 measured M_e flat) and on the
shuffle leg (donor history carries no fatigue semantics); the j0-restricted
reference drops it identically, so the comparison is like-for-like.

Run:  ./dev python3 eval/laf/walker/p3_eval.py
Out:  p3_eval.md, p3_eval.json
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, GOLD_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit, _zscore          # noqa: E402
import q1_sweep                                                      # noqa: E402
from q1_sweep import GAINS, compose, stack_messages, weights, configs  # noqa: E402
from q1_tiers import WINNER, TIERS                                   # noqa: E402
from q1_reverse import lane_attribution, classify_miss               # noqa: E402
from reach_leg import load_cues, cue_fields, rank_rows               # noqa: E402
from p3_fit import FEATURES, LANES, SLOTS, slot_raw, turn_features    # noqa: E402
from soft_usage import auc                                            # noqa: E402
from shuffle_control import build_shuffled, load_donors               # noqa: E402
from episodic_roles import build_role_map                             # noqa: E402

ARMS = ('A_full_picked', 'F_soft_ablate')
SEED = 20260715
N_SHUFFLE_TURNS = 400
TOL = 0.005
EYEBALL_CUES = 2
REPORT = OUT_DIR / 'p3_eval.md'
OUT = OUT_DIR / 'p3_eval.json'


def load_coefs():
    fit = json.loads((OUT_DIR / 'p3_fit.json').read_text())
    out = {}
    for arm in ARMS:
        coef = fit['results'][arm]['coef']
        out[arm] = {f: v for f, v in coef.items() if f != 'M_e_f'}
    return out


def fitted_field_score(op, an, coef, n, node_mask):
    """Engine-side fitted score: Σ coef · z(lane-slot) over the masked field
    — the identical 'current' substrate the fit ran on."""
    s = np.zeros(n)
    for f, w in coef.items():
        ln, sl = f.split('·')
        td_like = type('X', (), {'op': op, 'anchor': an})
        z = _zscore(slot_raw(td_like, ln, sl), n, mask=node_mask)
        s += w * z
    return s


def gold_evaluations(coefs):
    """(2) miss classes + (3) tier placement, per arm + statics."""
    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    cues = load_cues()
    gold = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())
    k0 = configs()[0]
    win = next(c for c in configs() if c['name'] == WINNER)
    arms = {'k0_static': None, 'winner_static': None}
    arms.update(coefs)
    placements = {a: {t: [] for t in TIERS} for a in arms}
    miss_class = {a: Counter() for a in arms}
    unreach_leaks = {a: [] for a in arms}
    eyeball = []
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        titles = dict(env.brain._nodes.conn.execute(
            'SELECT id, title FROM nodes'))
        w_win = weights(win)
        for ci, cue in enumerate(cues):
            q0 = _unit(embedder.embed_query(cue['text']))
            node_mask, tm = eng._asof_masks(cue['cutoff'], n)
            op, an = cue_fields(eng, tm, cue, q0)
            tiers = gold[cue['cue_id']]['tiers']
            golds = {it['node_id'] for t in TIERS for it in tiers.get(t, [])}
            # shared lane attribution (imported from q1_reverse — one home,
            # eligible-universe percentiles; code-review 2026-07-16)
            mats_w, ww = {}, None
            for ln in GAINS:
                mats_w[ln], ww = stack_messages(op[ln], an[ln], w_win, win)
            _lz, lane_pct = lane_attribution(mats_w, ww, n, node_mask)
            msg_cos = {'j0-op': op['maxsim'][:, 0],
                       'j1-op': op['maxsim'][:, 1],
                       'j1-anchor': an['maxsim'][:, 1]}
            # per-arm score + rank
            for arm in arms:
                if arm == 'k0_static':
                    mats0, ww0 = {}, None
                    w0 = weights(k0)
                    for ln in GAINS:
                        mats0[ln], ww0 = stack_messages(op[ln], an[ln],
                                                        w0, k0)
                    s = compose(mats0, ww0, k0, n, mask=node_mask)
                elif arm == 'winner_static':
                    s = compose(mats_w, ww, win, n, mask=node_mask)
                else:
                    s = fitted_field_score(op, an, coefs[arm], n, node_mask)
                order = rank_rows(s, node_mask)
                rank_of = {r: i + 1 for i, r in enumerate(order)}
                short_rank = {eng._master[r][:8]: i + 1
                              for i, r in enumerate(order)}
                for t in TIERS:
                    for it in tiers.get(t, []):
                        placements[arm][t].append(
                            short_rank.get(it['node_id']))
                # miss classes (shared attribution, arm-specific rank)
                for t in TIERS:
                    for it in tiers.get(t, []):
                        row = eng._resolve(it['node_id'])
                        if row is None:
                            miss_class[arm]['not_in_field'] += 1
                            continue
                        rk = rank_of.get(row)
                        best_ln = max(GAINS, key=lambda l: lane_pct[l][row])
                        best_pct = lane_pct[best_ln][row]
                        unreach_sub = best_pct < 95.0
                        if rk is not None and rk <= 25:
                            if unreach_sub:
                                unreach_leaks[arm].append(
                                    (cue['cue_id'], it['node_id'], rk))
                            continue
                        mc = {k: float(v[row]) if np.isfinite(v[row])
                              else None for k, v in msg_cos.items()}
                        cls = classify_miss(rk, best_ln, best_pct, mc)
                        miss_class[arm][cls.split(':')[0]] += 1
                if arm == 'F_soft_ablate' and ci < EYEBALL_CUES:
                    rows = ['## eyeball · F_soft_ablate · %s' % cue['cue_id']]
                    for i, r in enumerate(order[:10]):
                        rows.append('%2d. [%s] %s%s' % (
                            i + 1, eng._master[r][:8],
                            (titles.get(eng._master[r])
                             or eng._master[r])[:80],
                            ' ◀ TIERED' if eng._master[r][:8] in golds
                            else ''))
                    eyeball.append(rows)
    return placements, miss_class, unreach_leaks, eyeball


def shuffle_on_fitted(coefs):
    """(4) donor-shuffled j≥1 history vs the same coefficients' j0
    restriction — fitted-model shuffle control."""
    rng = random.Random(SEED)
    walker = open_walker()
    q1_sweep.gate_provenance(walker)
    turns = q1_sweep.load(walker)
    donors = load_donors(walker)
    walker.close()
    sample = rng.sample(turns, min(N_SHUFFLE_TURNS, len(turns)))
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
        shuffled = build_shuffled(sample, donors, rng, eng, role_map,
                                  trace_mat, trace_created, eng._n)

    def arm_auc(tds, coef, j0_only=False):
        sel, drp = [], []
        idx = {f: i for i, f in enumerate(FEATURES)}
        for td in tds:
            X = turn_features(td)
            s = np.zeros(len(td.cands))
            for f, w in coef.items():
                if j0_only and not f.endswith('j0-op'):
                    continue
                s += w * X[:, idx[f]]
            sel.append(s[td.sel])
            drp.append(s[~td.sel])
        return auc(np.concatenate(sel), np.concatenate(drp))

    out = {}
    for arm, coef in coefs.items():
        real = arm_auc(sample, coef)
        shuf = arm_auc(shuffled, coef)
        j0 = arm_auc(sample, coef, j0_only=True)
        out[arm] = {'real': real, 'shuffled': shuf, 'j0_restricted': j0,
                    'holds': bool(shuf <= j0 + TOL)}
    return out


def tier_table(placements):
    lines = ['| arm | tier | n | top-1 | top-5 | top-25 | median |',
             '|---|---|---|---|---|---|---|']
    data = {}
    for arm, tp in placements.items():
        for t in TIERS:
            rs = tp[t]
            known = [r for r in rs if r is not None]
            row = {'n': len(rs),
                   'top1': sum(1 for r in known if r <= 1),
                   'top5': sum(1 for r in known if r <= 5),
                   'top25': sum(1 for r in known if r <= 25),
                   'median': float(np.median(known)) if known else None}
            data.setdefault(arm, {})[t] = row
            lines.append('| %s | %s | %d | %d | %d | %d | %s |'
                         % (arm, t, row['n'], row['top1'], row['top5'],
                            row['top25'],
                            '%.0f' % row['median'] if row['median'] else '—'))
    return lines, data


def main():
    coefs = load_coefs()
    lines = ['# p3_eval — P3.2 pre-declared evaluations (§20.13)', '',
             '- arms: %s; M_e_f omitted engine-side (gold sessions walker-'
             'excluded; Q1 measured M_e flat)' % ', '.join(ARMS), '']

    print('gold-24 leg (tiers + miss classes, 4 arms)...')
    placements, miss_class, unreach_leaks, eyeball = gold_evaluations(coefs)
    lines.append('## (3) tier placement (blind-judged)')
    tl, tier_data = tier_table(placements)
    lines.extend(tl)
    lines.append('')
    lines.append('## (2) miss classes (gold+silver not in top-25; shared '
                 'q1_reverse attribution)')
    all_cls = ('near_miss', 'lane_buried', 'moment_seen', 'unreachable',
               'weak_everywhere', 'not_in_field')
    lines.append('| arm | ' + ' | '.join(all_cls) + ' |')
    lines.append('|' + '---|' * (len(all_cls) + 1))
    for arm, mc in miss_class.items():
        lines.append('| %s | ' % arm + ' | '.join(
            str(mc.get(c, 0)) for c in all_cls) + ' |')
    lines.append('')
    canary = {arm: len(v) for arm, v in unreach_leaks.items()}
    for arm, leaks in unreach_leaks.items():
        if leaks and arm not in ('k0_static', 'winner_static'):
            lines.append('- **LEAK CANARY %s**: unreachable-substrate nodes '
                         'in top-25: %s' % (arm, leaks))
    lines.append('- leak canary (unreachable-substrate nodes ranked into '
                 'top-25): %s' % json.dumps(canary))
    lines.append('')

    print('shuffle control on fitted models...')
    shuf = shuffle_on_fitted(coefs)
    lines.append('## (4) shuffle control (fitted models; reference = same '
                 'coefficients, j0-restricted)')
    lines.append('| arm | real AUC | shuffled AUC | j0-restricted | verdict |')
    lines.append('|---|---|---|---|---|')
    for arm, m in shuf.items():
        lines.append('| %s | %.4f | %.4f | %.4f | %s |'
                     % (arm, m['real'], m['shuffled'], m['j0_restricted'],
                        'holds' if m['holds'] else
                        'SHUFFLE WINS — moment gain is artifact'))
    lines.append('')
    for rows in eyeball:
        lines.extend(rows)
        lines.append('')

    OUT.write_text(json.dumps({'tiers': tier_data,
                               'miss_class': {a: dict(c) for a, c in
                                              miss_class.items()},
                               'unreach_leaks': unreach_leaks,
                               'shuffle': shuf}))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
