"""P3.0 — normalization repair mini-verdict (§20.13, pre-registered).

The sparse-lane z inflation (q1_reverse eyeball: enc z=11.4 / pick z=6.8 /
idf z=8.9 beside cosine z≈2) is a substrate bug: pick/enc/idf are mostly-zero
lanes (pool-side j=0: enc 92% / pick 80% / idf 41% zeros) whose zero sea
enters the z statistics, shrinking std until the few activated nodes explode.
Fit before fixing and P3.1 just learns to compensate for the artifact.

Three variants under STATIC gains (production zscore_variant dispatch —
eval measures the exact function production would run):
  current   — plain _zscore (the shipped incumbent)
  support   — stats over the NONZERO finite support; zeros stay neutral 0
  rank      — average-tie fractional ranks, then _zscore of the rank vector
              (bounded ~±1.7 — no lane dominates on sparsity alone)

PRE-DECLARED PICK (written before the run):
  primary   best June+ AUC (auc_val) on the Q1 winner config arm
  gate      gold-24 tier placement not worsened: on BOTH arms (K0, winner),
            gold_plus+gold top-5 and top-25 counts each >= current − 1
            (one-node slack = the ±4pp/24-cue noise band). Silver reported,
            not gated. 'current' is the incumbent — always eligible.
  If no variant beats current on the primary, the mini-verdict is KEEP
  CURRENT and the P3.1 fit substrate stays 'current'.

CONTROLS (run first, hard-fail):
  coverage   empty-history turns score EXACTLY K0 under every variant
  sanity     support ≡ current on a dense lane; rank output bounded;
             support leaves zeros at exactly 0

Winner becomes the P3.1 fit substrate AND a flag-gated production candidate
(K-store flip of recall_laf.z_norm; rollback = flip back).

Run:  ./dev python3 eval/laf/walker/p3_norm.py
Out:  p3_norm.md, p3_norm.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, GOLD_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (LafV1Engine, _unit, _zscore,       # noqa: E402
                                _zscore_support, _zscore_rank)
from q1_sweep import (GAINS, compose, stack_messages, weights,      # noqa: E402
                      configs, evaluate, score_turn, load,
                      gate_provenance)
from q1_tiers import WINNER, TIERS                                  # noqa: E402
from reach_leg import load_cues, cue_fields, rank_rows              # noqa: E402

NORMS = ('current', 'support', 'rank')
GATE_TIERS = ('gold_plus', 'gold')
EYEBALL_CUES = 2
REPORT = WALKER_DIR / 'p3_norm.md'
OUT = WALKER_DIR / 'p3_norm.json'


def variant_sanity():
    """Make each variant's defining property fire once — hard-fail if not."""
    rng = np.random.default_rng(7)
    n = 2000
    dense = rng.normal(0.6, 0.05, n)                  # cosine-like, no zeros
    sparse = np.zeros(n)
    sparse[rng.choice(n, 40, replace=False)] = rng.uniform(0.5, 0.9, 40)
    # support ≡ current on dense
    if not np.allclose(_zscore_support(dense, n), _zscore(dense, n)):
        raise SystemExit('sanity FAIL: support-z differs from current on a '
                         'dense lane')
    # current explodes on sparse; support does not
    z_cur = _zscore(sparse, n)
    z_sup = _zscore_support(sparse, n)
    if z_cur.max() < 4.0:
        raise SystemExit('sanity FAIL: current z did not inflate on the '
                         'sparse fixture (test broken?)')
    if z_sup.max() > 4.0:
        raise SystemExit('sanity FAIL: support-z still inflated on sparse')
    if np.any(z_sup[sparse == 0.0] != 0.0):
        raise SystemExit('sanity FAIL: support-z moved a zero off neutral')
    # rank bounded
    for x in (dense, sparse):
        zr = _zscore_rank(x, n)
        if np.abs(zr).max() > 2.5:
            raise SystemExit('sanity FAIL: rank-norm unbounded (%.2f)'
                             % np.abs(zr).max())
    return ('- support ≡ current on dense lane ✓; sparse fixture: current '
            'z_max %.1f → support z_max %.1f, zeros neutral ✓; rank bounded '
            '(|z|≤%.2f) ✓' % (z_cur.max(), z_sup.max(), 2.5))


def coverage_per_variant(turns):
    """The q1_sweep empty-history invariant, re-asserted per variant."""
    k0 = configs()[0]
    w0 = weights(k0)
    no_hist = [td for td in turns
               if all(np.all(np.isnan(td.op[ln][:, 1:]))
                      and np.all(np.isnan(td.anchor[ln][:, 1:]))
                      for ln in GAINS)][:30]
    cfg = next(c for c in configs() if c['name'] == WINNER)
    w = weights(cfg)
    checked = 0
    for znorm in NORMS:
        for td in no_hist:
            a = score_turn(td, k0, w0, znorm=znorm)
            b = score_turn(td, cfg, w, znorm=znorm)
            if not np.allclose(np.nan_to_num(a), np.nan_to_num(b),
                               atol=1e-12):
                raise SystemExit('coverage FAIL under %s: winner scores an '
                                 'empty-history turn differently from K0'
                                 % znorm)
            checked += 1
    return len(no_hist), checked


def rank_leg(turns):
    """{norm: {arm: metrics}} — June+ AUC is the pick's primary."""
    k0 = configs()[0]
    win = next(c for c in configs() if c['name'] == WINNER)
    out = {}
    for znorm in NORMS:
        out[znorm] = {'k0': evaluate(turns, k0, znorm=znorm),
                      'win': evaluate(turns, win, znorm=znorm)}
    return out


def gold_leg():
    """Tier placements + eyeball per (norm, arm) — cue_fields computed ONCE
    per cue (lane activations are norm-independent RAW values)."""
    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    cues = load_cues()
    gold = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())
    k0 = configs()[0]
    win = next(c for c in configs() if c['name'] == WINNER)
    placements = {zn: {t: {'k0': [], 'win': []} for t in TIERS}
                  for zn in NORMS}
    eyeball = {zn: [] for zn in NORMS}
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
        for ci, cue in enumerate(cues):
            q0 = _unit(embedder.embed_query(cue['text']))
            node_mask, _tm = eng._asof_masks(cue['cutoff'], n)
            op, an = cue_fields(eng, _tm, cue, q0)
            tiers = gold[cue['cue_id']]['tiers']
            golds = {it['node_id'] for t in TIERS for it in tiers.get(t, [])}
            for zn in NORMS:
                for label, c in (('k0', k0), ('win', win)):
                    w = weights(c)
                    mats, ww = {}, None
                    for ln in GAINS:
                        mats[ln], ww = stack_messages(op[ln], an[ln], w, c)
                    s = compose(mats, ww, c, n, mask=node_mask, znorm=zn)
                    order = rank_rows(s, node_mask)
                    rank_of = {eng._master[r][:8]: i + 1
                               for i, r in enumerate(order)}
                    for t in TIERS:
                        for it in tiers.get(t, []):
                            placements[zn][t][label].append(
                                rank_of.get(it['node_id']))
                    if label == 'win' and ci < EYEBALL_CUES:
                        from servers.recall_laf import zscore_variant
                        lane_z = {}
                        for ln in GAINS:
                            v = np.nansum(mats[ln] * ww, axis=1)
                            v[np.all(np.isnan(mats[ln]), axis=1)] = np.nan
                            lane_z[ln] = zscore_variant(v, n, mask=node_mask,
                                                        kind=zn)
                        rows = []
                        for i, r in enumerate(order[:10]):
                            dom = max(GAINS,
                                      key=lambda l: lane_z[l][r] * GAINS[l])
                            rows.append('%2d. [%s] (%s z%.1f) %s%s' % (
                                i + 1, eng._master[r][:8], dom,
                                lane_z[dom][r],
                                (titles.get(eng._master[r])
                                 or eng._master[r])[:80],
                                ' ◀ TIERED' if eng._master[r][:8] in golds
                                else ''))
                        eyeball[zn].append((cue['cue_id'], rows))
    return placements, eyeball


def tier_counts(placements, zn, label):
    """gold_plus+gold aggregate + per-tier rows."""
    agg = {'top1': 0, 'top5': 0, 'top25': 0}
    per = {}
    for t in TIERS:
        rs = placements[zn][t][label]
        known = [r for r in rs if r is not None]
        row = {'n': len(rs),
               'top1': sum(1 for r in known if r <= 1),
               'top5': sum(1 for r in known if r <= 5),
               'top25': sum(1 for r in known if r <= 25),
               'median_rank': float(np.median(known)) if known else None}
        per[t] = row
        if t in GATE_TIERS:
            for k in agg:
                agg[k] += row[k]
    return agg, per


def main():
    walker = open_walker()
    gate_provenance(walker)
    lines = ['# p3_norm — P3.0 normalization repair mini-verdict (§20.13)',
             '']
    lines.append('## 0 · variant sanity — PASS')
    lines.append(variant_sanity())
    turns = load(walker)
    walker.close()
    n_empty, checked = coverage_per_variant(turns)
    lines.append('- coverage invariant: %d empty-history turns × 3 norms, '
                 '%d pairs ≡ K0 ✓' % (n_empty, checked))
    lines.append('')

    print('rank leg (2 arms × 3 norms over %d turns)...' % len(turns))
    rl = rank_leg(turns)
    lines.append('## 1 · rank leg (walker pools, static gains)')
    lines.append('| norm | arm | AUC val (June+) | AUC all | soft_r |')
    lines.append('|---|---|---|---|---|')
    for zn in NORMS:
        for arm in ('k0', 'win'):
            m = rl[zn][arm]
            lines.append('| %s | %s | %.4f | %.4f | %.3f |'
                         % (zn, arm, m.get('auc_val', 0),
                            m.get('auc_all', 0), m.get('soft_r', 0)))
    lines.append('')

    print('gold-24 leg (tiers + eyeball)...')
    placements, eyeball = gold_leg()
    lines.append('## 2 · gold-24 tier placement')
    lines.append('| norm | arm | g+g top-1 | top-5 | top-25 | (gate tiers '
                 '= gold_plus+gold) |')
    lines.append('|---|---|---|---|---|---|')
    aggs = {}
    per_detail = {}
    for zn in NORMS:
        for arm in ('k0', 'win'):
            agg, per = tier_counts(placements, zn, arm)
            aggs[(zn, arm)] = agg
            per_detail['%s/%s' % (zn, arm)] = per
            lines.append('| %s | %s | %d | %d | %d |  |'
                         % (zn, arm, agg['top1'], agg['top5'], agg['top25']))
    lines.append('')

    # pre-declared pick
    gate_ok = {}
    for zn in NORMS:
        ok = all(aggs[(zn, arm)][k] >= aggs[('current', arm)][k] - 1
                 for arm in ('k0', 'win') for k in ('top5', 'top25'))
        gate_ok[zn] = ok
    primary = {zn: rl[zn]['win'].get('auc_val', 0) for zn in NORMS}
    eligible = [zn for zn in NORMS if gate_ok[zn]]
    pick = max(eligible, key=lambda zn: primary[zn])
    if pick != 'current' and primary[pick] <= primary['current']:
        pick = 'current'
    lines.append('## 3 · VERDICT (pre-declared rule)')
    for zn in NORMS:
        lines.append('- %s: June+ AUC (win) %.4f · tier gate %s'
                     % (zn, primary[zn], 'PASS' if gate_ok[zn] else 'FAIL'))
    lines.append('')
    lines.append('**PICK: %s** — %s' % (pick,
                 'normalization repair ships as fit substrate + flag-gated '
                 'production candidate' if pick != 'current' else
                 'no variant beats the incumbent on the pre-declared '
                 'primary; fit substrate stays current'))
    lines.append('')

    lines.append('## 4 · eyeball (winner arm, first %d cues × 3 norms)'
                 % EYEBALL_CUES)
    for zn in NORMS:
        for cid, rows in eyeball[zn]:
            lines.append('### %s · %s' % (zn, cid))
            lines.extend(rows)
            lines.append('')

    OUT.write_text(json.dumps({
        'rank_leg': rl, 'tier_agg': {'%s/%s' % k: v for k, v in aggs.items()},
        'tier_detail': per_detail, 'gate_ok': gate_ok, 'pick': pick}))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
