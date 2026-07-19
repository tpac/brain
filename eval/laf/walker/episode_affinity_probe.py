"""Reverse episode lookup (Tom, 2026-07-18): for the golds the CHAMPION BLEND
missed, reverse-search where each node was PICKED or CREATED and ask — does
its own linked episode have anything to do with THIS prompt?

Mechanism difference vs the current episodic lane (why this isn't the thing
that just washed out): the production lane goes query -> TOP-5 most-similar
past moments -> nodes at those moments. A gold whose own episode is prompt-
relevant but ranks 6th+ among moments gets ZERO — TOP_MOMENTS=5 is a hidden
cutoff. The reverse lane scores EVERY candidate by max cosine(query, its own
episodes) — per-node episodic affinity, no global cutoff.

Also measures Tom's entity hypothesis: rare-token (entity-proxy) overlap
between the query and the gold's episode, for reached vs unreached golds.

Outputs (live corpus, val turns, frozen S_content champion weights):
  1. N blend-missed golds; how many the reverse-affinity lane ranks top-5
  2. of those, how many had their best episode OUTSIDE the query's top-5
     moments (= the cutoff was the bottleneck, not the signal)
  3. entity-overlap distribution gold-vs-competitors
  4. two rendered examples (cue + gold + its best episode text)

Run: ./dev python3 eval/laf/walker/episode_affinity_probe.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit, _zscore_support  # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from definitive_fit import turn_features, FEATURES                  # noqa: E402
from episodic_roles import build_role_map, _stop_of, TOP_MOMENTS     # noqa: E402
from tests.isolated_brain import IsolatedBrain                       # noqa: E402

TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_']+")


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    tmeta, df = {}, Counter()
    for s, e, q, stop, ts, txt, qv in walker.execute(
            "SELECT session_id, epoch, seq, stop, ts, op_text, q_vec "
            "FROM turns"):
        tmeta[(s, e, q)] = (stop, ts, txt or '', _unit(qv) if qv else None)
        toks = set(t.lower() for t in TOK.findall(txt or ''))
        df.update(toks)
    ep_text = {}
    for s, stop, txt in walker.execute(
            "SELECT session_id, stop, op_text FROM turns WHERE stop "
            "IS NOT NULL"):
        if (s, stop) not in ep_text and txt:
            ep_text[(s, stop)] = txt
    walker.close()

    def rare(txt):
        return {t.lower() for t in TOK.findall(txt or '')
                if len(t) >= 4 and df.get(t.lower(), 0) <= 8}

    # frozen champion score (S_content weights from definitive_fit.json)
    W = json.load(open(Path(__file__).parent / 'definitive_fit.json')
                  )['weights']['S_content']
    cols = [FEATURES.index(k) for k in W]
    wv = np.array([W[k] for k in W])

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        role_map = build_role_map(env.brain)

        # node -> [(sess, stop, created)] where picked or created
        inv = defaultdict(list)
        for sess, rec in role_map.items():
            for stop, lst in rec['surf'].items():
                for created, ids in lst:
                    for nid in ids:
                        inv[nid].append((sess, stop, created))
            for run_stop, created, ids in rec['runs']:
                for nid in ids:
                    inv[nid].append((sess, run_stop, created))
        # (sess, stop) -> trace row indices
        ep_rows = defaultdict(list)
        for i, (chain, sess) in enumerate(eng._tr_meta):
            stop = _stop_of(chain or '')
            if stop is not None and sess:
                ep_rows[(sess, stop)].append(i)

        def top5_moments(sims):
            k = min(TOP_MOMENTS * 3, len(sims))
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            out = set()
            for i in top:
                chain = eng._tr_meta[i][0] or ''
                stop = _stop_of(chain)
                sess = eng._tr_meta[i][1]
                if stop is not None and sess:
                    out.add((sess, stop))
                if len(out) >= TOP_MOMENTS:
                    break
            return out

        def affinity(nid, qv, ts, sims):
            """(max cosine of node's own episodes vs query, best (sess,stop))"""
            best, where = 0.0, None
            for sess, stop, created in inv.get(nid, ()):
                if created > ts:
                    continue
                rows = [r for r in ep_rows.get((sess, stop), ())
                        if trace_created[r] <= ts]
                if not rows:
                    continue
                m = float(np.max(sims[rows]))
                if m > best:
                    best, where = m, (sess, stop)
            return best, where

        n_miss = reach_rev = cutoff_bottleneck = 0
        ent_gold, ent_comp = [], []
        examples = []
        for td in turns:
            if not td.val or not np.isfinite(td.soft).any():
                continue
            g = int(np.nanargmax(td.soft))
            if td.soft[g] < hi or td.key not in tmeta:
                continue
            X = turn_features(td)
            score = X[:, cols] @ wv
            if int((score > score[g]).sum()) < 5:
                continue                                  # champion got it
            _stop, ts, cue_txt, qv = tmeta[td.key]
            if qv is None:
                continue
            n_miss += 1
            sims = trace_mat @ qv
            nc = len(td.cands)
            aff = np.zeros(nc)
            wheres = [None] * nc
            for i, nid in enumerate(td.cands):
                aff[i], wheres[i] = affinity(nid, qv, ts, sims)
            z = _zscore_support(aff, nc)
            reached = int((z > z[g]).sum()) < 5 and aff[g] > 0
            reach_rev += int(reached)
            if reached:
                t5 = top5_moments(np.where(trace_created <= ts, sims, -np.inf))
                if wheres[g] is not None and wheres[g] not in t5:
                    cutoff_bottleneck += 1
                if len(examples) < 2 and wheres[g] in ep_text:
                    examples.append((cue_txt[:100], td.cands[g],
                                     float(aff[g]),
                                     ep_text[wheres[g]][:110]))
            qr = rare(cue_txt)
            if wheres[g] in ep_text:
                ent_gold.append(len(qr & rare(ep_text[wheres[g]])))
            for i in np.argsort(-score)[:5]:
                if wheres[int(i)] in ep_text:
                    ent_comp.append(len(qr & rare(ep_text[wheres[int(i)]])))

        print('blend-missed golds (champion rank>=5, val): %d' % n_miss)
        print('reverse-affinity ranks gold top-5: %d (%.0f%%)'
              % (reach_rev, 100 * reach_rev / max(1, n_miss)))
        print('  of those, best episode OUTSIDE query top-5 moments '
              '(cutoff was the bottleneck): %d (%.0f%%)'
              % (cutoff_bottleneck,
                 100 * cutoff_bottleneck / max(1, reach_rev)))
        print('entity(rare-token) overlap w/ query: gold-episode mean %.2f '
              '(n=%d) vs top5-competitor-episode mean %.2f (n=%d)'
              % (np.mean(ent_gold or [0]), len(ent_gold),
                 np.mean(ent_comp or [0]), len(ent_comp)))
        for cue, nid, a, etxt in examples:
            print('\nEX cue: %s' % cue.replace('\n', ' '))
            print('   gold %s aff %.2f · its episode: %s'
                  % (nid, a, etxt.replace('\n', ' ')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
