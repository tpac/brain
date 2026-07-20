"""Turn-influence probes — Tom's three asks (2026-07-17):

1. PLACEMENT — improvement in Haiku selections at top-1/5/25:
   pool-restricted over ALL val turns (@1/@3/@5 + MRR; @25 ≡ pool), and
   full-field over a ~120-turn sample through the engine matrices
   (true @1/@5/@25 among every eligible node).
2. TURN INFLUENCE, not lane tuning — arms vary only the turn dimension:
   K=0 vs winner (K1) vs anchor-only history vs op-only history.
3. INHIBITION of selected nodes — running fatigue replayed from labels
   (f ← β·f + 1[selected], score − δ·f), δ sweep on top of K0 and winner.

Plus the talky-user check: j1-op / j1-anchor marginal contribution
conditioned on op_len quartiles (does the operator side wake up when the
operator writes long?).

Run:  ./dev python3 eval/laf/walker/moment_influence.py
Out:  moment_influence.md, moment_influence.json
"""
import json
import sys
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import (GAINS, ME_BETA, load, gate_provenance, configs,  # noqa: E402
                      score_turn, weights, compose, stack_messages)
from soft_usage import auc                                             # noqa: E402
from episodic_roles import K_MAX                                       # noqa: E402
from reach_leg import stack_rows, rank_rows, TEXT_CAP                  # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
DELTAS = (0.1, 0.3, 0.6, 1.0)
N_SAMPLE = 120
J_LIMIT = 2                      # full-field slots j<=2 — all arms are K<=1
SEED = 20260717
REPORT = OUT_DIR / 'moment_influence.md'
OUT = OUT_DIR / 'moment_influence.json'


# ---------------------------------------------------------------- part 1

def variant(td, drop_op_hist=False, drop_anchor=False):
    """TurnData view with history slots masked (turn-dimension arms only)."""
    op, an = td.op, td.anchor
    if drop_op_hist:
        op = {ln: m.copy() for ln, m in op.items()}
        for m in op.values():
            m[:, 1:] = np.nan
    if drop_anchor:
        an = {ln: np.full_like(m, np.nan) for ln, m in an.items()}
    return SimpleNamespace(key=td.key, ts=td.ts, cands=td.cands, op=op,
                           anchor=an, sel=td.sel, soft=td.soft,
                           flagged=td.flagged, era_post=td.era_post,
                           val=td.val, fat=td.fat)


def pool_metrics(turns, cfg, tdmap=None, delta_override=None):
    """Placement of selected nodes within the pool + soft reads (val)."""
    w = weights(cfg)
    top1 = top3 = top5 = nsel = nturn = 0
    rr = []
    sx, sy, s_at_1 = [], [], []
    sel_p, drp_p = [], []
    for td in turns:
        if not td.val:
            continue
        t = tdmap(td) if tdmap else td
        s = score_turn(t, cfg, w)
        if delta_override is not None:
            s = s - delta_override * td.fat
        if not td.sel.any() or td.sel.all():
            continue
        order = np.argsort(-s)
        ranks = np.empty(len(s), dtype=int)
        ranks[order] = np.arange(1, len(s) + 1)
        rsel = np.sort(ranks[td.sel])
        nturn += 1
        nsel += len(rsel)
        top1 += int(rsel[0] == 1)
        top3 += int((rsel <= 3).sum())
        top5 += int((rsel <= 5).sum())
        rr.append(1.0 / rsel[0])
        sel_p.append(s[td.sel])
        drp_p.append(s[~td.sel])
        if np.isfinite(td.soft[order[0]]):
            s_at_1.append(td.soft[order[0]])
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            sx.append(s[m])
            sy.append(td.soft[m])
    return {
        'sel_at_1': top1 / nturn,
        'sel_in_3': top3 / nsel,
        'sel_in_5': top5 / nsel,
        'mrr': float(np.mean(rr)),
        'auc': auc(np.concatenate(sel_p), np.concatenate(drp_p)),
        'soft_r': float(np.corrcoef(np.concatenate(sx),
                                    np.concatenate(sy))[0, 1]),
        'soft_at_top1': float(np.mean(s_at_1)),
        'n_turns': nturn,
    }


def op_len_conditioning(walker, turns, cfg_win):
    """Δsoft_r / ΔAUC of the j1-op and j1-anchor slots, per op_len
    quartile — the talky-user check."""
    lens = dict(((s, e, q), l) for s, e, q, l in walker.execute(
        "SELECT session_id, epoch, seq, op_len FROM turns WHERE labeled=1"))
    vals = sorted(lens[td.key] for td in turns if td.val and td.key in lens)
    qs = [vals[int(len(vals) * f)] for f in (0.25, 0.5, 0.75)]

    def bucket(td):
        v = lens.get(td.key, 0)
        return sum(v > q for q in qs)          # 0..3

    w = weights(cfg_win)
    out = []
    for b in range(4):
        sub = [td for td in turns if td.val and bucket(td) == b]
        rows = {}
        for name, tdmap in (
                ('full', None),
                ('no_j1op', lambda td: variant(td, drop_op_hist=True)),
                ('no_anchor', lambda td: variant(td, drop_anchor=True))):
            sx, sy, sel_p, drp_p = [], [], [], []
            for td in sub:
                t = tdmap(td) if tdmap else td
                s = score_turn(t, cfg_win, w)
                if td.sel.any() and not td.sel.all():
                    sel_p.append(s[td.sel])
                    drp_p.append(s[~td.sel])
                m = np.isfinite(td.soft) & np.isfinite(s)
                if m.sum() > 2:
                    sx.append(s[m])
                    sy.append(td.soft[m])
            rows[name] = {
                'auc': auc(np.concatenate(sel_p), np.concatenate(drp_p)),
                'soft_r': float(np.corrcoef(np.concatenate(sx),
                                            np.concatenate(sy))[0, 1])}
        out.append({
            'bucket': b, 'n': len(sub),
            'len_range': '%d-%d' % (
                (0 if b == 0 else qs[b - 1]),
                (qs[b] if b < 3 else max(vals))),
            'd_j1op_auc': rows['full']['auc'] - rows['no_j1op']['auc'],
            'd_j1op_soft': rows['full']['soft_r'] - rows['no_j1op']['soft_r'],
            'd_anchor_auc': rows['full']['auc'] - rows['no_anchor']['auc'],
            'd_anchor_soft': (rows['full']['soft_r']
                              - rows['no_anchor']['soft_r']),
        })
    return out


# ---------------------------------------------------------------- part 2

def full_fatigue(walker):
    """(session, epoch, seq) → {node_id: f} at recall time (pre-accrual),
    replayed exactly like q1_sweep.precompute_fatigue but keeping the FULL
    per-node dict for full-field inhibition."""
    picked_of = defaultdict(set)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE outcome='selected' AND node_id IS NOT NULL"):
        picked_of[(sess, epoch, seq)].add(nid)
    snap = {}
    cur_sess, f = None, defaultdict(float)
    for sess, epoch, seq, ts, labeled, flags in walker.execute(
            "SELECT session_id, epoch, seq, ts, labeled, flags FROM turns "
            "ORDER BY session_id, ts"):
        if sess != cur_sess:
            cur_sess, f = sess, defaultdict(float)
        if 'machine_turn' in (flags or ''):
            continue
        for k in list(f):
            f[k] *= ME_BETA
            if f[k] < 1e-6:
                del f[k]
        if labeled:
            snap[(sess, epoch, seq)] = dict(f)
        for nid in picked_of.get((sess, epoch, seq), ()):
            f[nid] += 1.0
    return snap


def sample_turns(walker, rng):
    """Val-slice labeled turns with ≥1 selected node — sampled rows carry
    what the full-field pass needs."""
    rows = walker.execute(
        "SELECT t.session_id, t.epoch, t.seq, t.stop, t.ts, t.op_text, "
        "t.q_vec FROM turns t WHERE t.labeled=1 AND t.ts >= '2026-06-01' "
        "AND t.q_vec IS NOT NULL AND EXISTS (SELECT 1 FROM candidates c "
        "WHERE c.session_id=t.session_id AND c.epoch=t.epoch AND "
        "c.seq=t.seq AND c.outcome='selected' AND c.node_id IS NOT NULL)"
    ).fetchall()
    sel_of = defaultdict(set)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE outcome='selected' AND node_id IS NOT NULL"):
        sel_of[(sess, epoch, seq)].add(nid)
    idx = rng.permutation(len(rows))[:N_SAMPLE]
    return [(rows[i], sel_of[tuple(rows[i][:3])]) for i in idx]


def turn_fields(eng, trace_mask, session, stop, ts, op_text, q0):
    """cue_fields, limited to slots j<=J_LIMIT (all part-2 arms are K<=1)."""
    from servers.recall_laf import MAXSIM_VIEWS, DEFAULT_CONFIG
    n = eng._n
    cfg = dict(DEFAULT_CONFIG)
    slots = {(0, 'op'): ((op_text or '')[:TEXT_CAP], q0)}
    for (j, kind), v in stack_rows(eng._brain_ref, session, stop,
                                   before_ts=ts).items():
        if (j, kind) == (0, 'op') or j > J_LIMIT:
            continue
        slots[(j, kind)] = v
    op = {ln: np.full((n, K_MAX + 1), np.nan) for ln in GAINS}
    an = {ln: np.full((n, K_MAX + 1), np.nan) for ln in GAINS}
    for (j, kind), (text, vec) in slots.items():
        tgt = op if kind == 'op' else an
        with np.errstate(all='ignore'):
            tgt['maxsim'][:, j] = np.nanmax(
                np.stack([eng._mats[vt][:n] @ vec for vt in MAXSIM_VIEWS]),
                axis=0)
        tgt['sit'][:, j] = eng._mats['_situation'][:n] @ vec
        if text:
            tgt['idf'][:, j] = eng._idf_asof(text, n, ts)
        pick, enc = eng._episodic_vectors(eng._brain_ref, vec, cfg, n,
                                          as_of=ts, trace_mask=trace_mask)
        tgt['pick'][:, j] = pick
        tgt['enc'][:, j] = enc
    for ln in an:            # j0-anchor = the turn's own future response —
        an[ln][:, 0] = np.nan    # never a cue, even for direct consumers
    return op, an


def field_metrics(rank_of_sel):
    r = np.array(rank_of_sel)
    return {'sel_at_1': float((r == 1).mean()),
            'sel_in_5': float((r <= 5).mean()),
            'sel_in_25': float((r <= 25).mean()),
            'median_rank': float(np.median(r)), 'n_sel': len(r)}


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    lines = ['# moment_influence — placement, turn arms, inhibition', '']

    # -------- part 1: pool-restricted, all val turns
    cfg_k0 = configs()[0]
    cfg_win = next(c for c in configs() if c['name'] == WINNER)
    arms = [('K0', cfg_k0, None, None),
            ('winner K1', cfg_win, None, None),
            ('winner anchor-only hist', cfg_win,
             lambda td: variant(td, drop_op_hist=True), None),
            ('winner op-only (no anchor)', cfg_win,
             lambda td: variant(td, drop_anchor=True), None)]
    for d in DELTAS:
        arms.append(('K0 + inhibit δ=%.1f' % d, cfg_k0, None, d))
        arms.append(('winner + inhibit δ=%.1f' % d, cfg_win, None, d))
    part1 = {}
    lines.append('## pool placement of Haiku selections (val turns; '
                 '@25 ≡ pool size)')
    lines.append('| arm | sel@1 | sel-in-3 | sel-in-5 | MRR | AUC | '
                 'soft_r | soft@top1 |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for name, cfg, tdmap, d in arms:
        m = pool_metrics(turns, cfg, tdmap, d)
        part1[name] = m
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f | %.4f | %.3f | '
                     '%.3f |' % (name, m['sel_at_1'], m['sel_in_3'],
                                 m['sel_in_5'], m['mrr'], m['auc'],
                                 m['soft_r'], m['soft_at_top1']))
        print('part1', name, 'done')
    lines.append('')

    # -------- talky-user check
    lines.append('## op_len conditioning — j1-op / anchor value per '
                 'operator-message-length quartile (winner arm)')
    lines.append('| quartile | op_len | n | Δj1-op AUC | Δj1-op soft_r | '
                 'Δanchor AUC | Δanchor soft_r |')
    lines.append('|---|---|---|---|---|---|---|')
    cond = op_len_conditioning(walker, turns, cfg_win)
    for c in cond:
        lines.append('| Q%d | %s | %d | %+.4f | %+.3f | %+.4f | %+.3f |'
                     % (c['bucket'] + 1, c['len_range'], c['n'],
                        c['d_j1op_auc'], c['d_j1op_soft'],
                        c['d_anchor_auc'], c['d_anchor_soft']))
    lines.append('')
    print('op_len conditioning done')

    # -------- part 2: full-field sample through the engine
    rng = np.random.default_rng(SEED)
    sample = sample_turns(walker, rng)
    fat_snap = full_fatigue(walker)
    walker.close()
    print('full-field sample: %d turns' % len(sample))

    from servers.recall_laf import LafV1Engine, _unit
    from tests.isolated_brain import IsolatedBrain
    f_arms = ['K0', 'winner K1', 'winner anchor-only hist',
              'winner + inhibit δ=0.3']
    ranks = {a: [] for a in f_arms}
    unresolved = 0
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        w0, ww_win = weights(cfg_k0), weights(cfg_win)
        for i, ((sess, epoch, seq, stop, ts, op_text, qb), sel_ids) \
                in enumerate(sample):
            q0 = _unit(qb)
            if q0 is None:
                continue
            node_mask, trace_mask = eng._asof_masks(ts, n)
            op, an = turn_fields(eng, trace_mask, sess, stop, ts,
                                 op_text, q0)
            sel_rows = [r for r in (eng._resolve(nid) for nid in sel_ids)
                        if r is not None]
            if not sel_rows:
                unresolved += 1
                continue
            fvec = np.zeros(n)
            for nid, fv in fat_snap.get((sess, epoch, seq), {}).items():
                r = eng._resolve(nid)
                if r is not None:
                    fvec[r] = fv

            def field_score(cfg, w, drop_op_hist=False, delta=None):
                opx = op
                if drop_op_hist:
                    opx = {ln: m.copy() for ln, m in op.items()}
                    for m in opx.values():
                        m[:, 1:] = np.nan
                mats, ww = {}, None
                for ln in GAINS:
                    mats[ln], ww = stack_messages(opx[ln], an[ln], w, cfg)
                s = compose(mats, ww, cfg, n, mask=node_mask)
                if delta:
                    s = s - delta * fvec
                return s
            for name, s in (
                    ('K0', field_score(cfg_k0, w0)),
                    ('winner K1', field_score(cfg_win, ww_win)),
                    ('winner anchor-only hist',
                     field_score(cfg_win, ww_win, drop_op_hist=True)),
                    ('winner + inhibit δ=0.3',
                     field_score(cfg_win, ww_win, delta=0.3))):
                order = rank_rows(s, node_mask)
                pos = np.empty(n, dtype=int)
                pos[order] = np.arange(1, n + 1)
                ranks[name].extend(int(pos[r]) for r in sel_rows)
            if (i + 1) % 20 == 0:
                print('  field %d/%d' % (i + 1, len(sample)))

    part2 = {a: field_metrics(r) for a, r in ranks.items()}
    lines.append('## full-field placement of Haiku selections '
                 '(%d sampled val turns, all eligible nodes)' % len(sample))
    lines.append('- selected nodes unresolvable in today\'s brain '
                 '(archived since): %d turns skipped' % unresolved)
    lines.append('| arm | sel@1 | sel@5 | sel@25 | median rank | n sel |')
    lines.append('|---|---|---|---|---|---|')
    for a in f_arms:
        m = part2[a]
        lines.append('| %s | %.3f | %.3f | %.3f | %.0f | %d |'
                     % (a, m['sel_at_1'], m['sel_in_5'], m['sel_in_25'],
                        m['median_rank'], m['n_sel']))
    lines.append('')

    OUT.write_text(json.dumps({'pool': part1, 'op_len': cond,
                               'field': part2}, indent=1))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
