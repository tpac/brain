"""Does a message's SURFACE SHAPE (length / #sentences / #questions) predict
which lane reaches the gold — i.e. is there a STYLE-INVARIANT routing signal
for the selector? (Tom's thesis, 2026-07-18: short msg -> history wins, long
msg -> current maxsim wins; and it should be a CONTENT feature, not a speaker
proxy, so it transfers to other agents.)

Per val gold turn (gold = argmax soft >= 90th pctile), rank the gold under two
lanes in isolation:
  current = z(maxsim, op j0)          # this message
  hist    = z(maxsim, best j>=1)      # a previous message
ADV = rank_hist - rank_cur  (>0 => current better; <0 => history better)
Then bin ADV / reach by op_len, #sentences, has_question and report the trend.

Run: ./dev python3 eval/laf/walker/length_probe.py [--walker DIR]
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                              # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402


def rank_on(lane, nc, g):
    z = _zscore(lane, nc)
    return int((z > z[g]).sum())


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    meta = {}
    for sess, epoch, seq, op_len, hq, txt in walker.execute(
            "SELECT session_id, epoch, seq, op_len, has_question, op_text "
            "FROM turns"):
        nsent = max(1, sum((txt or '').count(c) for c in '.!?'))
        meta[(sess, epoch, seq)] = (op_len or len(txt or ''), int(hq or 0),
                                    nsent, (txt or '').count('?'))
    walker.close()

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    rows = []          # (op_len, has_q, nsent, nqmark, reach_cur, reach_hist, adv)
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any() or td.key not in meta:
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] < hi:
            continue
        nc = len(td.cands)
        cur = td.op['maxsim'][:, 0]
        hist = np.concatenate([td.op['maxsim'][:, 1:K_MAX + 1],
                               td.anchor['maxsim'][:, 1:K_MAX + 1]], axis=1)
        with np.errstate(all='ignore'):
            hist = np.where(np.all(np.isnan(hist), axis=1), np.nan,
                            np.nanmax(hist, axis=1))
        if not np.isfinite(hist[g]):
            continue                                    # no history to compare
        rc, rh = rank_on(cur, nc, g), rank_on(hist, nc, g)
        ol, hq, ns, nq = meta[td.key]
        rows.append((ol, hq, ns, nq, int(rc < 5), int(rh < 5), rh - rc))
    A = np.array(rows, float)
    ol, hq, ns, nq, rcur, rhist, adv = A.T
    print('val gold turns with history: %d  (soft hi=%.2f)' % (len(A), hi))
    print('overall: current reaches %.0f%% · hist reaches %.0f%% · '
          'ADV mean %+.2f (>0=current better)'
          % (100 * rcur.mean(), 100 * rhist.mean(), adv.mean()))
    print('corr(op_len, ADV)=%+.3f  corr(#sent, ADV)=%+.3f  '
          'corr(#qmark, ADV)=%+.3f'
          % (np.corrcoef(ol, adv)[0, 1], np.corrcoef(ns, adv)[0, 1],
             np.corrcoef(nq, adv)[0, 1]))

    def table(name, key, edges):
        b = np.digitize(key, edges)
        print('\n%-10s   n    cur_reach  hist_reach   ADV(cur-adv)' % name)
        for i in range(len(edges) + 1):
            m = b == i
            if m.sum() < 5:
                continue
            lo = '%g' % edges[i - 1] if i else '-inf'
            hiu = '%g' % edges[i] if i < len(edges) else 'inf'
            print('  [%5s,%5s)  %4d    %4.0f%%      %4.0f%%       %+.2f'
                  % (lo, hiu, int(m.sum()), 100 * rcur[m].mean(),
                     100 * rhist[m].mean(), adv[m].mean()))

    table('op_len', ol, [40, 100, 200, 400])
    table('#sentences', ns, [1, 2, 4, 8])
    print('\nhas_question   n    cur_reach  hist_reach   ADV')
    for v in (0, 1):
        m = hq == v
        if m.sum() >= 5:
            print('  %d            %4d    %4.0f%%      %4.0f%%       %+.2f'
                  % (v, int(m.sum()), 100 * rcur[m].mean(),
                     100 * rhist[m].mean(), adv[m].mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
