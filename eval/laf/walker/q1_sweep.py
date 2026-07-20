"""Q1 sweep — rank leg + controls (§20.5, H2-LOCKED grid; runs AFTER Tom's
final H2 look; building and control-validation are pre-approved harness work).

QUESTION Q1: does any moment shape beat K=0 (prompt-only) on BOTH reach and
rank? This file owns the RANK leg (judge-label AUC + soft-usage correlation
over the walker tables) and the harness controls. The gold-24/LongMemEval
REACH legs run through the engine's as_of path (separate runner; the
pre-declared aggregate combines both legs).

Grid (verbatim from §20.5, enumerated in one place — configs()):
  K {0,1,2,4,8} × decay {exp γ .3/.5/.7/.9, power α 1/2, uniform}
  × composition {turnsum, turnmax} × aggregation point {lane, zsum}
  × texts {op, op+anchor} × M_e {off, δ .1/.3}
K=0 collapses every other axis → ONE baseline config.
Lanes: maxsim (nanmax over the 6 content views), sit, idf, pick, enc —
episodic rides the identical grid via cand_turn_episodic (engine-parity
proven by that build's self-check). Gains are the shipped P1 statics
(DEFAULT_CONFIG) — Q1 fits nothing.

Weight normalization note: all metrics are within-turn (pool z-scoring is
shift/scale invariant per turn), so decay weights need no normalization —
recorded so nobody "fixes" it into a bug.

Full-field caveat (health check 5): pool z here is over ~24 candidates,
not the engine's full node field — the known rank-leg proxy; the reach
legs run the real full-field engine. Both legs together are the verdict.

M_e — PINNED AT H2 (Tom, 2026-07-15): 2′, surfaced-only running fatigue.
    f ← β·f + 1[picked-into-context]      score' = score − δ·f   (z-space)
β=0.7 fixed; session-scoped replay from STORED labels (picked = the nodes
actually injected into Anchor's context — availability management, "no need
to restate"). DROPPED nodes carry no automatic within-session signal (one
pass-over is not a relevance verdict); the cross-moment drop-rate stays a
future P3+ lane. Per-node only — no neighbor spread. KEY PROPERTY: f depends
only on stored labels, never on config scores, so the fatigue table is
precomputed ONCE and every M_e config is base − δ·f. Decay ticks on every
non-machine turn row (lived turn steps), session-ordered by ts across
epochs; f accrues only at labeled turns (recall events carry picks).

CONTROLS (mandatory, run first — `--controls`):
  coverage  turns with NO prior turns must score EXACTLY the K=0 config
            under every moment shape (invariant, asserted here)
  shuffle   moment stack from random OTHER-session turns must not beat
            K=0 — runs on the top-3 configs AFTER the grid (bounded)
  positive  the ±1-turn episodic reproduction lives in the reach runner
            (gold-side), not here.

Run:  ./dev python3 eval/laf/walker/q1_sweep.py --controls   (harness gate)
      ./dev python3 eval/laf/walker/q1_sweep.py --grid       (after H2 look)
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import (open_walker, EXTRACT_VERSION, EMBED_VERSION,
                       lanes_version, OUT_DIR)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (MAXSIM_VIEWS, _zscore, DEFAULT_CONFIG,  # noqa: E402
                                zscore_variant)
from soft_usage import SOFT_USAGE_VERSION, auc                        # noqa: E402
from episodic_roles import EPISODIC_VERSION, K_MAX                    # noqa: E402

ERA_SPLIT = '2026-06-08'
VAL_SPLIT = '2026-06-01'          # rank leg: April–May reference / June+ verdict
GAINS = {k[5:]: float(v) for k, v in DEFAULT_CONFIG.items()
         if k.startswith('gain_') and k != 'gain_proj'}
V_OP = ['v_%s_op' % v.strip('_') for v in MAXSIM_VIEWS]
V_AN = ['v_%s_anchor' % v.strip('_') for v in MAXSIM_VIEWS]
REPORT = OUT_DIR / 'q1_sweep.md'


def gate_provenance(walker):
    """Every input table must carry the stamp its builder's CURRENT code
    expects — the sweep reads nothing it can't prove."""
    stamps = dict(walker.execute("SELECT key, value FROM build_meta"))
    expect = {'extract_version': EXTRACT_VERSION,
              'embed_version': EMBED_VERSION,
              'scores_lanes_version': lanes_version(MAXSIM_VIEWS),
              'soft_usage_version': SOFT_USAGE_VERSION,
              'episodic_roles_version': EPISODIC_VERSION}
    bad = {k for k, v in expect.items() if stamps.get(k) != v}
    if bad:
        raise SystemExit('q1_sweep: unproven inputs (%s) — rebuild the '
                         'missing phase, never bypass.' % ', '.join(sorted(bad)))
    chk = json.loads(stamps.get('episodic_self_check', '{}'))
    if chk.get('worst_abs_delta', 1) > 1e-6:
        raise SystemExit('q1_sweep: episodic self-check stamp missing/failed')


def configs():
    """The H2-locked grid, one row per config. K=0 is the single baseline."""
    decays = ([('exp', g) for g in (0.3, 0.5, 0.7, 0.9)] +
              [('pow', a) for a in (1.0, 2.0)] + [('uniform', None)])
    out = [{'name': 'K0', 'K': 0, 'decay': ('exp', 0.0), 'comp': 'turnsum',
            'agg': 'lane', 'texts': 'op', 'me': ('off', 0.0)}]
    for K, dec, comp, agg, texts, me in itertools.product(
            (1, 2, 4, 8), decays, ('turnsum', 'turnmax'), ('lane', 'zsum'),
            ('op', 'op+anchor'),
            (('off', 0.0), ('fatigue', 0.1), ('fatigue', 0.3))):
        out.append({'name': 'K%d-%s%s-%s-%s-%s-me%s' % (
            K, dec[0], ('' if dec[1] is None else dec[1]), comp, agg,
            texts.replace('+', ''), me[1] or '0'),
            'K': K, 'decay': dec, 'comp': comp, 'agg': agg,
            'texts': texts, 'me': me})
    return out


def weights(cfg):
    """w_j for j=0..K (unnormalized — within-turn metrics are scale-free)."""
    K = cfg['K']
    fam, p = cfg['decay']
    j = np.arange(K + 1, dtype=float)
    if K == 0:
        return np.ones(1)
    if fam == 'exp':
        return p ** j
    if fam == 'pow':
        return (j + 1.0) ** -p
    return np.ones(K + 1)


ME_BETA = 0.7                     # H2-pinned fatigue decay (2′)


class TurnData:
    """One labeled turn's lane tensors: lane → [n_cand × (K_MAX+1)] float
    (NaN = missing), per text source; plus labels and class tags."""
    __slots__ = ('key', 'ts', 'cands', 'op', 'anchor', 'sel', 'soft',
                 'flagged', 'era_post', 'val', 'fat')

    def __init__(self, key, ts, cands, flagged):
        self.key, self.ts, self.cands, self.flagged = key, ts, cands, flagged
        nc = len(cands)
        shape = (nc, K_MAX + 1)
        self.op = {ln: np.full(shape, np.nan) for ln in GAINS}
        self.anchor = {ln: np.full(shape, np.nan) for ln in GAINS}
        self.sel = np.zeros(nc, dtype=bool)
        self.soft = np.full(nc, np.nan)
        self.fat = np.zeros(nc)           # per-candidate fatigue f at this turn
        self.era_post = ts >= ERA_SPLIT
        self.val = ts >= VAL_SPLIT


def precompute_fatigue(walker, turns_by_key):
    """The 2′ fatigue table: for every labeled turn, each candidate's f value
    at recall time (BEFORE the turn's own picks accrue). Session-scoped,
    lived-order (ts across epochs); decay β ticks on every non-machine turn
    row; f accrues +1 on the nodes PICKED at each labeled turn. Config-
    independent by construction — computed once, applied as base − δ·f."""
    rows = walker.execute(
        "SELECT session_id, epoch, seq, ts, labeled, flags FROM turns "
        "ORDER BY session_id, ts").fetchall()
    picked_of = defaultdict(set)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE outcome='selected' AND node_id IS NOT NULL"):
        picked_of[(sess, epoch, seq)].add(nid)
    cur_sess, f = None, defaultdict(float)
    for sess, epoch, seq, ts, labeled, flags in rows:
        if sess != cur_sess:
            cur_sess, f = sess, defaultdict(float)
        if 'machine_turn' in (flags or ''):
            continue                          # not a turn — no decay tick
        for k in list(f):
            f[k] *= ME_BETA
            if f[k] < 1e-6:
                del f[k]
        key = (sess, epoch, seq)
        td = turns_by_key.get(key)
        if td is not None:
            td.fat[:] = [f.get(nid, 0.0) for nid in td.cands]
        for nid in picked_of.get(key, ()):    # accrue AFTER scoring the turn
            f[nid] += 1.0


def load(walker):
    """walker.db → [TurnData], one pass per table."""
    turn_meta = {}
    flags_by_turn = {}
    for sess, epoch, seq, ts, flags in walker.execute(
            "SELECT session_id, epoch, seq, ts, flags FROM turns"):
        flags_by_turn[(sess, epoch, seq)] = bool(json.loads(flags or '[]'))
        turn_meta[(sess, epoch, seq)] = ts
    labeled = {k for (k,) in
               ((r[:3],) for r in walker.execute(
                   "SELECT session_id, epoch, seq FROM turns WHERE labeled=1"))}
    # a turn's stack is flagged if the turn OR any of its K_MAX predecessors
    # in the same (sess, epoch) carries a flag — the 36%-contamination class
    def stack_flagged(key):
        sess, epoch, seq = key
        return any(flags_by_turn.get((sess, epoch, seq - j), False)
                   for j in range(0, K_MAX + 1))

    cand_rows = defaultdict(list)
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, outcome FROM candidates "
            "WHERE node_id IS NOT NULL"):
        if row[:3] in labeled:
            cand_rows[row[:3]].append((row[3], row[4]))
    soft = {}
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, soft_max "
            "FROM soft_usage WHERE soft_max IS NOT NULL"):
        soft[row[:4]] = row[4]

    turns = {}
    for key, cands in cand_rows.items():
        td = TurnData(key, turn_meta[key], [c[0] for c in cands],
                      stack_flagged(key))
        td.sel[:] = [c[1] == 'selected' for c in cands]
        td.soft[:] = [soft.get((*key, nid), np.nan) for nid, _ in cands]
        turns[key] = td
    idx_of = {key: {nid: i for i, nid in enumerate(td.cands)}
              for key, td in turns.items()}

    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, j, %s, sit_op, idf_op,"
            " %s, sit_anchor, idf_anchor FROM cand_turn_scores"
            % (', '.join(V_OP), ', '.join(V_AN))):
        key, nid, j = row[:3], row[3], row[4]
        td = turns.get(key)
        if td is None:
            continue
        i = idx_of[key].get(nid)
        if i is None:
            continue
        nv = len(V_OP)
        opv = row[5:5 + nv]
        sit_o, idf_o = row[5 + nv], row[6 + nv]
        anv = row[7 + nv:7 + 2 * nv]
        sit_a, idf_a = row[7 + 2 * nv], row[8 + 2 * nv]
        with np.errstate(all='ignore'):
            td.op['maxsim'][i, j] = np.nanmax(
                np.array(opv, dtype=float)) if any(
                    v is not None for v in opv) else np.nan
            td.anchor['maxsim'][i, j] = np.nanmax(
                np.array(anv, dtype=float)) if any(
                    v is not None for v in anv) else np.nan
        td.op['sit'][i, j] = np.nan if sit_o is None else sit_o
        td.op['idf'][i, j] = np.nan if idf_o is None else idf_o
        td.anchor['sit'][i, j] = np.nan if sit_a is None else sit_a
        td.anchor['idf'][i, j] = np.nan if idf_a is None else idf_a
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, j, pick_op, enc_op,"
            " pick_anchor, enc_anchor FROM cand_turn_episodic"):
        key, nid, j = row[:3], row[3], row[4]
        td = turns.get(key)
        if td is None:
            continue
        i = idx_of[key].get(nid)
        if i is None:
            continue
        for ln, v_op, v_an in (('pick', row[5], row[7]),
                               ('enc', row[6], row[8])):
            td.op[ln][i, j] = np.nan if v_op is None else v_op
            td.anchor[ln][i, j] = np.nan if v_an is None else v_an
    precompute_fatigue(walker, turns)
    return list(turns.values())


def compose(mats, ww, cfg, n, gains=None, mask=None, znorm=None):
    """THE grid composition — shared by the rank leg (pool rows) and the
    reach leg (full-field rows); one implementation, no cross-leg drift.

    mats: {lane: [n × M] message columns, NaN = missing}; ww: [M] weights.
    agg='lane': reduce each lane over messages, THEN z + gain-sum.
    agg='zsum': z + gain-sum per MESSAGE, then reduce over messages.
    reduce is turnsum (nan-weighted-sum) or turnmax (nan-weighted-max);
    all-NaN rows stay NaN.

    mask: the as_of node mask (reach leg) — z-statistics must run over the
    masked universe EXACTLY like the engine's scores(as_of=) z-loop, or the
    per-lane affine scales drift and the composed ORDER diverges from
    production (base-parity check catches this). Rank leg passes None
    (the pool is its own universe).

    znorm: P3.0 lane-normalizer variant ('current'/'support'/'rank') —
    routed through the production zscore_variant dispatch; default is
    bit-identical to the Q1 runs. When the kwarg is not given explicitly,
    cfg['z_norm'] is honored (the PRODUCTION channel for this knob) so a
    replayed K-store config can never be silently scored as 'current'
    under a variant label (code-review 2026-07-16).
    """
    gains = gains or GAINS
    if znorm is None:
        znorm = cfg.get('z_norm', 'current')

    def zs(x):
        return zscore_variant(x, n, mask=mask, kind=znorm)

    def reduce(mat):
        with np.errstate(all='ignore'):
            if cfg['comp'] == 'turnsum':
                v = np.nansum(mat * ww, axis=1)
            else:
                v = np.nanmax(mat * ww, axis=1)
        v[np.all(np.isnan(mat), axis=1)] = np.nan
        return v

    if cfg['agg'] == 'lane':
        zsum = np.zeros(n)
        for ln, g in gains.items():
            zsum += g * zs(reduce(mats[ln]))
        return zsum
    n_msg = next(iter(mats.values())).shape[1]
    per_msg = np.full((n, n_msg), np.nan)
    for col in range(n_msg):
        acc = np.zeros(n)
        any_data = np.zeros(n, dtype=bool)
        for ln, g in gains.items():
            x = mats[ln][:, col]
            acc += g * zs(x)
            any_data |= np.isfinite(x)
        per_msg[:, col] = np.where(any_data, acc, np.nan)
    return reduce(per_msg)


def stack_messages(op_mat, anchor_mat, w, cfg):
    """Lane matrix + weights for one lane under a config's K/texts axes.

    TEMPORAL-LEAK RULE (coverage control caught this): the j=0 anchor is
    the turn's OWN response — it does not exist at recall time and it IS
    the soft-usage label. Anchor messages join the stack at j≥1 ONLY
    (previous turns' responses; v4 attach order guarantees an anchor at
    seq-j predates the prompt at seq). texts collapses at K=0."""
    K = cfg['K']
    m = op_mat[:, :K + 1]
    if cfg['texts'] == 'op+anchor' and K >= 1:
        return (np.concatenate([m, anchor_mat[:, 1:K + 1]], axis=1),
                np.concatenate([w, w[1:]]))
    return m, w


def score_turn(td, cfg, w, gains=None, znorm=None):
    """Config → per-candidate scores for one turn (pool z, production
    _zscore). A message = (offset j, source op|anchor); texts=op+anchor
    puts BOTH sources in the stack as separate messages with weight w_j —
    turnsum sums them, turnmax maxes over them (never a pre-summed pair).

    NOTE (shuffle control): the rank-leg shuffle needs candidate × random-
    OTHER-session-turn cosines, which the walker columns don't hold — it is
    a fresh (bounded) compute on the top-3 configs after the grid, not a
    score_turn mode.
    """
    nc = len(td.cands)
    mats, ww = {}, None
    for ln in GAINS:
        mats[ln], ww = stack_messages(td.op[ln], td.anchor[ln], w, cfg)
    base = compose(mats, ww, cfg, nc, gains=gains, znorm=znorm)
    kind, delta = cfg['me']
    if kind != 'off':
        # 2′ (H2-pinned): surfaced-only running fatigue, z-space subtraction.
        # δ is calibrated in 'current' z-space (unbounded lane z); variant
        # norms re-scale base (rank bounds it to ~±√3·Σgains) so the same δ
        # carries a different relative weight — cross-norm comparisons of
        # me≠off configs are confounded. Refuse loudly until δ is
        # recalibrated per norm (code-review 2026-07-16).
        if znorm not in (None, 'current'):
            raise SystemExit('score_turn: M_e δ is calibrated for '
                             "znorm='current' only — recalibrate before "
                             'running fatigue configs under %r' % znorm)
        base = base - float(delta) * td.fat
    return base


def evaluate(turns, cfg, znorm=None):
    """One config over all turns → metrics dict (overall + slices)."""
    w = weights(cfg)
    pools = {'all': ([], []), 'val': ([], []), 'normal': ([], []),
             'flagged': ([], []), 'pre_era': ([], []), 'post_era': ([], [])}
    soft_x, soft_y = [], []
    for td in turns:
        s = score_turn(td, cfg, w, znorm=znorm)
        sel, drp = s[td.sel], s[~td.sel]
        keys = ['all'] + (['val'] if td.val else []) + \
            (['flagged'] if td.flagged else ['normal']) + \
            (['post_era'] if td.era_post else ['pre_era'])
        for k in keys:
            pools[k][0].append(sel)
            pools[k][1].append(drp)
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            soft_x.append(s[m])
            soft_y.append(td.soft[m])
    out = {}
    for k, (sel, drp) in pools.items():
        if sel and drp:
            out['auc_' + k] = auc(np.concatenate(sel), np.concatenate(drp))
    if soft_x:
        x, y = np.concatenate(soft_x), np.concatenate(soft_y)
        out['soft_r'] = float(np.corrcoef(x, y)[0, 1])
    return out


def coverage_control(turns):
    """Turns with NO history (achieved window 0) must score EXACTLY K0
    under every moment shape. Hard invariant — harness bug if violated."""
    k0 = configs()[0]
    w0 = weights(k0)
    # empty-history = no op AND no anchor context at j>=1. v6 machine turns
    # keep an anchor with no op, so a turn preceded only by machine turns has
    # anchor-only history — genuine context that op+anchor configs SHOULD use,
    # so it is NOT an empty-history turn (the coverage control caught this).
    no_hist = [td for td in turns
               if all(np.all(np.isnan(td.op[ln][:, 1:]))
                      and np.all(np.isnan(td.anchor[ln][:, 1:]))
                      for ln in GAINS)]
    checked = 0
    for cfg in configs()[1:40:7]:          # a spread of shapes, MG=off only
        if cfg['me'][0] != 'off':
            continue
        w = weights(cfg)
        for td in no_hist[:50]:
            a = score_turn(td, k0, w0)
            b = score_turn(td, cfg, w)
            if not np.allclose(np.nan_to_num(a), np.nan_to_num(b),
                               atol=1e-12):
                raise SystemExit('coverage control FAILED: config %s scores '
                                 'an empty-history turn differently from K0'
                                 % cfg['name'])
            checked += 1
    # M_e coverage variant: at zero-fatigue turns (nothing picked earlier in
    # the session) an M_e config must equal its off-twin exactly; at nonzero-
    # fatigue turns it must differ (the axis is alive, not silently off)
    zero_f = [td for td in turns if td.fat.sum() == 0][:40]
    hot_f = [td for td in turns if td.fat.sum() > 0.5][:40]
    cfg_off = next(c for c in configs() if c['K'] == 2 and c['me'][0] == 'off'
                   and c['comp'] == 'turnsum' and c['agg'] == 'lane'
                   and c['texts'] == 'op' and c['decay'] == ('exp', 0.7))
    cfg_me = dict(cfg_off, me=('fatigue', 0.3),
                  name=cfg_off['name'] + '+me')
    w2 = weights(cfg_off)
    me_checked = me_alive = 0
    for td in zero_f:
        if not np.allclose(np.nan_to_num(score_turn(td, cfg_off, w2)),
                           np.nan_to_num(score_turn(td, cfg_me, w2)),
                           atol=1e-12):
            raise SystemExit('M_e coverage FAILED: zero-fatigue turn scored '
                             'differently under M_e')
        me_checked += 1
    for td in hot_f:
        if not np.allclose(np.nan_to_num(score_turn(td, cfg_off, w2)),
                           np.nan_to_num(score_turn(td, cfg_me, w2))):
            me_alive += 1
    if hot_f and me_alive == 0:
        raise SystemExit('M_e coverage FAILED: axis is silently dead — no '
                         'nonzero-fatigue turn changed score')
    return len(no_hist), checked, me_checked, me_alive, len(hot_f)


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()
    print('loaded %d labeled turns' % len(turns))
    if '--controls' in sys.argv:
        n_empty, checked, me_z, me_alive, me_hot = coverage_control(turns)
        print('coverage control: %d empty-history turns, %d (turn, config) '
              'pairs identical to K0 — PASS' % (n_empty, checked))
        print('M_e coverage: %d zero-fatigue turns ≡ off-twin; axis alive on '
              '%d/%d hot turns — PASS' % (me_z, me_alive, me_hot))
        k0 = evaluate(turns, configs()[0])
        print('K0 baseline metrics: %s'
              % json.dumps({k: round(v, 4) for k, v in k0.items()}))
        return 0
    if '--grid' in sys.argv:
        # H2 FINAL LOOK SIGNED (Tom 2026-07-15): grid stands, M_e pinned 2′.
        n_empty, checked, me_z, me_alive, me_hot = coverage_control(turns)
        print('controls re-verified (coverage %d/%d, M_e %d/%d/%d) — running '
              'the registered grid' % (n_empty, checked, me_z, me_alive,
                                       me_hot))
        results = []
        for i, cfg in enumerate(configs()):
            m = evaluate(turns, cfg)
            m['name'] = cfg['name']
            results.append(m)
            if (i + 1) % 50 == 0:
                print('  %d/%d configs' % (i + 1, len(configs())))
        k0 = next(r for r in results if r['name'] == 'K0')
        # rank verdict metric: June+ holdout AUC delta vs K0
        for r in results:
            r['d_val'] = (r.get('auc_val') or 0) - (k0.get('auc_val') or 0)
        results.sort(key=lambda r: -r['d_val'])
        lines = ['# q1_sweep — rank leg, registered grid (§20.5)', '',
                 'K0 baseline: %s' % json.dumps(
                     {k: round(v, 4) for k, v in k0.items()
                      if k != 'name'}), '',
                 '| config | ΔAUC val (June+) | AUC all | normal | flagged |'
                 ' pre-era | post-era | soft_r |', '|---|---|---|---|---|---|---|---|']
        for r in results[:40]:
            lines.append('| %s | %+.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |'
                         % (r['name'], r['d_val'], r.get('auc_all', 0),
                            r.get('auc_normal', 0), r.get('auc_flagged', 0),
                            r.get('auc_pre_era', 0), r.get('auc_post_era', 0),
                            r.get('soft_r', 0)))
        lines.append('')
        lines.append('- configs evaluated: %d; full table: q1_sweep_full.json'
                     % len(results))
        lines.append('- SHUFFLE CONTROL PENDING on top-3 — no verdict before '
                     'it runs (registered order).')
        (OUT_DIR / 'q1_sweep_full.json').write_text(json.dumps(results))
        REPORT.write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines[:14]))
        print('... full report: %s' % REPORT)
        return 0
    print(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main())
