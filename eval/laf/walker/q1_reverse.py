"""Q1 reverse-engineering — misses, junk, and the eyeball (Tom 2026-07-15).

Per gold cue (winner config, as_of engine path):
  MISSES — every gold/silver node NOT in top-25: its actual rank, per-lane
  composed percentile (which lane comes closest to carrying it), and which
  MOMENT MESSAGE sees it best (max view-cosine per message) — separating:
    lane_buried   some lane ≥ p99 but the blend buries it → selector/gating
                  would rescue; the lane is named
    moment_seen   a j≥1 message's cosine ≥ the j0 cosine + margin → a
                  history-weighted query would rescue (the 'given moment'
                  class Tom asked for)
    near_miss     rank 26–60 → small gain shifts rescue
    unreachable   no lane above p95 anywhere → these lanes cannot carry it;
                  needs a new signal family (graph/encode-side)
  JUNK — top-25 nodes with a suspicious signature, classified:
    pick_echo     pick-z high (≥2) but content percentile low (<p80) —
                  riding past Haiku picks, not content
    stale_hub     high overall but node predates the cue topic by >60d AND
                  content pct < p80 (rough hub smell; listed for the eye)
  EYEBALL — full top-10 with titles for the first N cues, gold marked.

Attribution note: per-lane values under zsum aggregation are the lane's own
message-composed z (turnsum of per-message z) — an attribution view, exact
for ranking each lane alone, approximate as a share of the blended score.

Run:  ./dev python3 eval/laf/walker/q1_reverse.py
Out:  q1_reverse.md (+ stdout summary)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, GOLD_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit, _zscore          # noqa: E402
from q1_sweep import GAINS, compose, stack_messages, weights, configs  # noqa: E402
from reach_leg import load_cues, cue_fields, rank_rows              # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
EYEBALL_CUES = 4
TIERS = ('gold_plus', 'gold', 'silver_plus', 'silver')
REPORT = WALKER_DIR / 'q1_reverse.md'


def lane_attribution(mats, ww, n, node_mask):
    """(lane_z, lane_pct) per lane — THE shared attribution for miss
    classification; p3_eval imports this so the two instruments cannot drift.

    Percentiles are computed WITHIN the eligible universe (ineligible rows
    get -1): ranking over the full matrix inflated every eligible node's
    percentile by the masked fraction, floating the p95/p99 class thresholds
    per cue cutoff (code-review catch 2026-07-16)."""
    lane_z, lane_pct = {}, {}
    idx = np.flatnonzero(node_mask)
    denom = max(len(idx) - 1, 1)
    for ln in GAINS:
        v = np.nansum(mats[ln] * ww, axis=1)
        v[np.all(np.isnan(mats[ln]), axis=1)] = np.nan
        z = _zscore(v, n, mask=node_mask)
        lane_z[ln] = z
        pct = np.full(n, -1.0)
        if len(idx) > 1:
            pct[idx] = z[idx].argsort().argsort() / denom * 100
        lane_pct[ln] = pct
    return lane_z, lane_pct


def classify_miss(rk, best_ln, best_pct, mc):
    """Shared miss-class decision tree — the thresholds ARE the contract.
    `x if x is not None` (not `x or`) so a genuine 0.0 cosine is never
    coerced to -1 (code-review catch 2026-07-16)."""
    j0c = mc['j0-op'] if mc['j0-op'] is not None else -1
    hist_best = max((v for k, v in mc.items()
                     if k != 'j0-op' and v is not None), default=-1)
    if rk is not None and rk <= 60:
        return 'near_miss'
    if best_pct >= 99.0:
        return 'lane_buried:%s' % best_ln
    if hist_best > j0c + 0.05:
        return 'moment_seen'
    if best_pct < 95.0:
        return 'unreachable'
    return 'weak_everywhere'


def main():
    cfg = next(c for c in configs() if c['name'] == WINNER)
    w = weights(cfg)
    cues = load_cues()
    gold = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())

    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    lines = ['# q1_reverse — misses, junk, eyeball (winner config)', '']
    miss_class = Counter()
    junk_class = Counter()
    miss_rows, junk_rows = [], []
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

        def title(short_or_row):
            nid = eng._master[short_or_row] if isinstance(short_or_row, (int,
                np.integer)) else short_or_row
            return (titles.get(nid) or nid)[:80]

        for ci, cue in enumerate(cues):
            q0 = _unit(embedder.embed_query(cue['text']))
            node_mask, trace_mask = eng._asof_masks(cue['cutoff'], n)
            op, an = cue_fields(eng, trace_mask, cue, q0)
            mats, ww = {}, None
            for ln in GAINS:
                mats[ln], ww = stack_messages(op[ln], an[ln], w, cfg)
            s = compose(mats, ww, cfg, n, mask=node_mask)
            order = rank_rows(s, node_mask)
            rank_of = {r: i + 1 for i, r in enumerate(order)}
            # per-lane composed z + percentile (attribution view)
            lane_z, lane_pct = lane_attribution(mats, ww, n, node_mask)
            # message visibility: maxsim per message column
            msg_cos = {'j0-op': op['maxsim'][:, 0], 'j1-op': op['maxsim'][:, 1],
                       'j1-anchor': an['maxsim'][:, 1]}

            tiers = gold[cue['cue_id']]['tiers']
            for t in TIERS:
                for it in tiers.get(t, []):
                    row = eng._resolve(it['node_id'])
                    if row is None:
                        miss_class['not_in_field'] += 1
                        continue
                    rk = rank_of.get(row)
                    if rk is not None and rk <= 25:
                        continue                       # hit — not a miss
                    best_ln = max(GAINS, key=lambda l: lane_pct[l][row])
                    best_pct = lane_pct[best_ln][row]
                    mc = {k: float(v[row]) if np.isfinite(v[row]) else None
                          for k, v in msg_cos.items()}
                    cls = classify_miss(rk, best_ln, best_pct, mc)
                    miss_class[cls.split(':')[0]] += 1
                    miss_rows.append((cue['cue_id'], t, it['node_id'],
                                      rk or 99999, cls, best_ln,
                                      round(best_pct, 1), mc,
                                      title(eng._master[row])))
            # junk in top-25
            content_pct = np.maximum(lane_pct['maxsim'], lane_pct['sit'])
            for r in order[:25]:
                is_tiered = any(eng._master[r][:8] == it['node_id']
                                for t in TIERS for it in tiers.get(t, []))
                if is_tiered:
                    continue
                if lane_z['pick'][r] >= 2.0 and content_pct[r] < 80.0:
                    junk_class['pick_echo'] += 1
                    junk_rows.append((cue['cue_id'], eng._master[r][:8],
                                      int(rank_of[r]), 'pick_echo',
                                      title(int(r))))
            # eyeball: full top-10 with titles
            if ci < EYEBALL_CUES:
                lines.append('## eyeball · %s' % cue['cue_id'])
                lines.append('cue: %s' % cue['text'][:140].replace('\n', ' '))
                golds = {it['node_id'] for t in TIERS
                         for it in tiers.get(t, [])}
                for i, r in enumerate(order[:10]):
                    mark = ' ◀ TIERED' if eng._master[r][:8] in golds else ''
                    dom = max(GAINS, key=lambda l: lane_z[l][r] * GAINS[l])
                    lines.append('%2d. [%s] (%s z%.1f) %s%s'
                                 % (i + 1, eng._master[r][:8], dom,
                                    lane_z[dom][r], title(int(r)), mark))
                lines.append('')

    lines.append('## miss classes (gold+silver not in top-25)')
    for k, v in miss_class.most_common():
        lines.append('- %s: %d' % (k, v))
    lines.append('')
    lines.append('## misses detail (worst-ranked first suppressed; sorted by rank)')
    lines.append('| cue | tier | node | rank | class | best lane (pct) | j0/j1op/j1an cos | title |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for row in sorted(miss_rows, key=lambda x: x[3])[:60]:
        cid, t, nid, rk, cls, bl, bp, mc, ttl = row
        lines.append('| %s | %s | %s | %s | %s | %s (%.0f) | %s/%s/%s | %s |'
                     % (cid, t, nid, rk if rk < 99999 else '—', cls, bl, bp,
                        *('%.2f' % v if v is not None else '—'
                          for v in (mc['j0-op'], mc['j1-op'],
                                    mc['j1-anchor'])), ttl))
    lines.append('')
    lines.append('## junk in top-25 (non-tiered, suspicious signature)')
    for k, v in junk_class.most_common():
        lines.append('- %s: %d' % (k, v))
    for row in junk_rows[:30]:
        lines.append('- %s · rank %d · %s · %s' % (row[0], row[2], row[3],
                                                   row[4]))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-90:]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
