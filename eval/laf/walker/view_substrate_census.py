"""Per-view substrate census — expose what nanmax is aggregating over.

RESEARCH, not an A/B. The decision (keep unweighted nanmax over 6 views vs
weight them vs split a view into its own lane) currently rests on a record
where two of three original justifications no longer hold. Rather than test one
arm, map the substrate the aggregation runs on.

Six readouts, all dense over the engine's full node set, as-of honest:

  A COVERAGE      what fraction of nodes even HAVE each view vector. Sets the
                  magnitude of the field-richness bias (max of n values rises
                  with n).
  B SCALE         per-view cosine mean/std against real queries. This is the
                  per-view-normalisation loss made concrete: if one view runs
                  systematically hot, the max is decided by view identity
                  rather than by relevance.
  C WHO WINS      which view supplies the max — for the GOLD vs for the top-5
                  NON-GOLDS. If golds win on a different view than non-golds
                  do, per-view weighting has real leverage; if the
                  distributions match, it does not.
  D HEADROOM      gold's rank in each view ALONE vs in maxsim. Gives the
                  best-single-view oracle (an honest re-derivation of the
                  §18.12 '51%' claim) and shows whether maxsim tracks the best
                  view or lags it.
  E CONVERGENCE   how many views rank the gold top-25. The multiplicity signal
                  a max discards by construction.
  F RICHNESS BIAS rank vs number-of-views-present, golds vs non-golds — direct
                  quantification of the documented-but-never-measured bias.

Everything is computed in ENGINE row space (self-consistent: alive = finite
maxsim), so no index alignment is needed beyond locating the gold by node id.
Ranks are tie-fair. Read-only.

Run: VECLIB_MAXIMUM_THREADS=3 OMP_NUM_THREADS=3 nice -n 19 \
     ./dev python3 eval/laf/walker/view_substrate_census.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from servers.recall_laf import LafV1Engine, MAXSIM_VIEWS            # noqa: E402
from servers.pipeline_contract import EMBEDDING_GROUPS              # noqa: E402

REPORT = OUT_DIR / 'view_substrate_census.md'
TOPK = 25

WEIGHT_OF = {g['vector_type']: g.get('weight', 0.0)
             for g in EMBEDDING_GROUPS.values() if g.get('vector_type')}


def tie_fair(scores, gi):
    gv = scores[gi]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(scores), scores, -np.inf)
    return int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']

    w = open_walker()
    qvs = {}
    for sess, epoch, seq, q in w.execute(
            'SELECT session_id, epoch, seq, q_vec FROM turns'):
        if q:
            v = np.frombuffer(q, dtype=np.float32).astype(np.float64)
            nrm = np.linalg.norm(v)
            if nrm:
                qvs[(sess, epoch, seq)] = v / nrm
    w.close()

    turns, _enr, _n = D.build_corpus('2026-05-11')
    print('corpus turns: %d' % len(turns))

    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
        n = eng._n
        views = list(MAXSIM_VIEWS)
        mats = {v: eng._mats[v][:n] for v in views}

        # ── A. COVERAGE ──
        cov = {}
        for v in views:
            present = np.isfinite(mats[v]).all(axis=1)
            cov[v] = float(present.mean())
        n_views_present = np.zeros(n, dtype=int)
        for v in views:
            n_views_present += np.isfinite(mats[v]).all(axis=1).astype(int)

        scale = defaultdict(list)
        win_gold, win_non = Counter(), Counter()
        per_view_rank = defaultdict(list)
        maxsim_rank, wmax_rank = [], []
        conv_counts = Counter()
        rich_gold, rich_non = [], []
        n_scored = 0

        for t in turns:
            key = t['key'].split('/')
            q = qvs.get((key[0], int(key[1]), int(key[2])))
            if q is None:
                continue
            gid = master[int(t['gr'])]
            gi = eng._idx.get(gid)
            if gi is None or gi >= n:
                continue
            ts = t.get('turn_dt')
            node_mask = None
            if ts is not None:
                node_mask, _tm = eng._asof_masks(ts.isoformat(), n)

            cos = {}
            for v in views:
                c = mats[v] @ q
                if node_mask is not None:
                    c = np.where(node_mask, c, np.nan)
                cos[v] = c
            with np.errstate(all='ignore'):
                stack = np.stack([cos[v] for v in views])
                mx = np.nanmax(stack, axis=0)
                wstack = np.stack([cos[v] * WEIGHT_OF.get(v, 1.0)
                                   for v in views])
                wmx = np.nanmax(wstack, axis=0)
            alive = np.isfinite(mx)
            if not alive[gi] or alive.sum() < 50:
                continue
            n_scored += 1

            # B scale — per-view distribution over alive nodes
            for v in views:
                cv = cos[v][alive]
                cv = cv[np.isfinite(cv)]
                if cv.size:
                    scale[v].append((float(cv.mean()), float(cv.std())))

            # C who wins the max. nanargmax raises on all-NaN rows (nodes with
            # no view vector as-of this turn), so fill those with -inf first —
            # they are excluded by `alive` anyway and their argmax is unused.
            safe = np.where(np.isfinite(stack), stack, -np.inf)
            argw = np.argmax(safe, axis=0)
            win_gold[views[int(argw[gi])]] += 1
            order = np.argsort(-np.where(alive, mx, -np.inf))
            for j in order[:5]:
                if int(j) != gi:
                    win_non[views[int(argw[int(j)])]] += 1

            # D headroom + maxsim/weighted-maxsim rank
            mr = tie_fair(np.where(alive, mx, np.nan), gi)
            wr = tie_fair(np.where(alive, wmx, np.nan), gi)
            if mr:
                maxsim_rank.append(mr)
            if wr:
                wmax_rank.append(wr)
            best = None
            for v in views:
                r = tie_fair(np.where(alive, cos[v], np.nan), gi)
                if r is not None:
                    per_view_rank[v].append(r)
                    best = r if best is None else min(best, r)
            if best is not None:
                per_view_rank['__best_single__'].append(best)

            # E convergence — how many views put the gold in their top-25
            k = sum(1 for v in views
                    if (tie_fair(np.where(alive, cos[v], np.nan), gi) or 1e9)
                    <= TOPK)
            conv_counts[k] += 1

            # F richness bias
            rich_gold.append(int(n_views_present[gi]))
            for j in order[:5]:
                if int(j) != gi:
                    rich_non.append(int(n_views_present[int(j)]))

    def med(a):
        a = sorted(x for x in a if x is not None)
        return a[len(a) // 2] if a else float('nan')

    def pct_at(a, k):
        a = [x for x in a if x is not None]
        return 100.0 * sum(1 for x in a if x <= k) / max(len(a), 1)

    L = ['# Per-view substrate census — what nanmax aggregates over', '',
         'turns scored: %d · engine nodes: %d · views: %s'
         % (n_scored, n, ', '.join(views)), '',
         '## A. Coverage — which nodes even have the view', '',
         '| view | weight (inert in nanmax) | nodes with vector |',
         '|---|---|---|']
    for v in views:
        L.append('| %s | %.2f | %.1f%% |' % (v, WEIGHT_OF.get(v, 0), 100 * cov[v]))
    L += ['', 'views present per node: %s'
          % ' · '.join('%d views: %d nodes' % (k, int((n_views_present == k).sum()))
                       for k in range(len(views) + 1)
                       if (n_views_present == k).sum()), '',
          '## B. Scale — per-view cosine distribution vs real queries', '',
          '| view | mean cos | std | mean+1σ (what it brings to a max) |',
          '|---|---|---|---|']
    for v in views:
        if not scale[v]:
            continue
        m = float(np.mean([x[0] for x in scale[v]]))
        s = float(np.mean([x[1] for x in scale[v]]))
        L.append('| %s | %.4f | %.4f | %.4f |' % (v, m, s, m + s))
    L += ['', '## C. Which view SUPPLIES the max — gold vs top-5 non-gold', '',
          '| view | wins for GOLD | wins for non-gold | gold share | non share |',
          '|---|---|---|---|---|']
    tg, tn = max(sum(win_gold.values()), 1), max(sum(win_non.values()), 1)
    for v in views:
        L.append('| %s | %d | %d | %.1f%% | %.1f%% |'
                 % (v, win_gold[v], win_non[v],
                    100.0 * win_gold[v] / tg, 100.0 * win_non[v] / tn))
    L += ['', '## D. Headroom — gold rank per view alone vs the aggregate', '',
          '| scorer | median gold rank | gold in top-5 | top-25 |',
          '|---|---|---|---|']
    for v in views + ['__best_single__']:
        a = per_view_rank[v]
        if not a:
            continue
        L.append('| %s | %.0f | %.1f%% | %.1f%% |'
                 % ('BEST SINGLE VIEW (oracle)' if v.startswith('__') else v,
                    med(a), pct_at(a, 5), pct_at(a, TOPK)))
    L.append('| **maxsim (shipped, unweighted)** | %.0f | %.1f%% | %.1f%% |'
             % (med(maxsim_rank), pct_at(maxsim_rank, 5),
                pct_at(maxsim_rank, TOPK)))
    L.append('| weighted max (existing weights) | %.0f | %.1f%% | %.1f%% |'
             % (med(wmax_rank), pct_at(wmax_rank, 5), pct_at(wmax_rank, TOPK)))
    L += ['', '## E. Convergence — how many views rank the gold top-%d' % TOPK,
          '', '| views agreeing | turns | share |', '|---|---|---|']
    tot = max(sum(conv_counts.values()), 1)
    for k in sorted(conv_counts):
        L.append('| %d | %d | %.1f%% |' % (k, conv_counts[k],
                                           100.0 * conv_counts[k] / tot))
    L += ['', '## F. Field-richness bias — views present, gold vs non-gold', '',
          '- GOLD nodes: mean %.2f views present (median %d)'
          % (float(np.mean(rich_gold)), int(np.median(rich_gold))),
          '- top-5 NON-GOLD nodes: mean %.2f views present (median %d)'
          % (float(np.mean(rich_non)), int(np.median(rich_non))),
          '- corpus-wide: mean %.2f views present'
          % float(n_views_present.mean()), '',
          '(non-golds richer than golds ⇒ the max is partly rewarding '
          'field-richness rather than relevance.)']
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
