"""Is the current-vs-history lane preference a PERSISTENT style, or does it
revert at the MESSAGE level within a conversation? (Tom, 2026-07-18: don't
assume style persists — check for msg-level reversions in the live brain AND
the corpus.) If reversions are real, a style-prior / slow field adaptation
mis-routes them and per-message confidence (cur_maxz) is ESSENTIAL, not a
minor adjustment.

Per val gold turn: ADV = rank_hist - rank_cur (>0 current better), grouped by
session. We decompose var(ADV) into between-session (style) vs within-session
(message-level), measure the sign-flip rate vs the session's own mean, and
test whether cur_maxz predicts the WITHIN-session residual (i.e. catches the
reversions a style-prior cannot).

Run: ./dev python3 eval/laf/walker/revert_probe.py   (WALKER_OUT_DIR to switch)
"""
import sys
from collections import defaultdict
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
    txt = {}
    for sess, epoch, seq, t in walker.execute(
            "SELECT session_id, epoch, seq, op_text FROM turns"):
        txt[(sess, epoch, seq)] = t or ''
    walker.close()

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    per = []                                    # (sess, adv, maxz, rc, rh, key)
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any():
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
            continue
        rc, rh = rank_on(cur, nc, g), rank_on(hist, nc, g)
        maxz = float(np.max(_zscore(cur, nc)))
        per.append((td.key[0], rh - rc, maxz, int(rc < 5), int(rh < 5), td.key))

    sess_turns = defaultdict(list)
    for r in per:
        sess_turns[r[0]].append(r)
    multi = {s: rs for s, rs in sess_turns.items() if len(rs) >= 3}

    adv = np.array([r[1] for r in per], float)
    smean = {s: np.mean([r[1] for r in rs]) for s, rs in sess_turns.items()}
    between = np.var([smean[s] for s in sess_turns for _ in sess_turns[s]])
    within = np.mean([np.var([r[1] for r in rs])
                      for rs in sess_turns.values() if len(rs) >= 2])
    print('val gold turns %d · sessions %d (>=3 turns: %d)  soft hi=%.2f'
          % (len(per), len(sess_turns), len(multi), hi))
    print('var(ADV): total %.1f · between-session(style) %.1f · '
          'within-session(msg) %.1f  -> within/total %.0f%%'
          % (adv.var(), between, within, 100 * within / adv.var()))

    # sign-flip vs session mean (leave-one-out) + reversion counts
    flips = tot = 0
    revert_cur = []      # current-exclusive turn in a history-dominant session
    revert_hist = []     # history-exclusive turn in a current-dominant session
    maxz_rev, maxz_norev = [], []
    for s, rs in multi.items():
        sm = smean[s]
        for r in rs:
            _, a, mz, rc, rh, key = r
            loo = (sm * len(rs) - a) / (len(rs) - 1)
            if loo == 0:
                continue
            tot += 1
            flip = (a < 0) != (loo < 0)
            flips += int(flip)
            (maxz_rev if flip else maxz_norev).append(mz)
            if loo < 0 and rc and not rh:          # session=history, turn=current
                revert_cur.append((a, mz, key))
            if loo > 0 and rh and not rc:           # session=current, turn=history
                revert_hist.append((a, mz, key))

    print('\nmsg-level sign-flip vs session style (LOO): %d/%d = %.0f%%'
          % (flips, tot, 100 * flips / max(1, tot)))
    print('cur_maxz on flip turns %.2f vs non-flip %.2f  (higher on flip = '
          'confidence catches reversions)'
          % (np.mean(maxz_rev or [0]), np.mean(maxz_norev or [0])))
    print('current-exclusive reversions (in history-dominant sessions): %d'
          % len(revert_cur))
    print('history-exclusive reversions (in current-dominant sessions): %d'
          % len(revert_hist))

    def show(label, lst, rev=True):
        print('\n%s:' % label)
        lst = sorted(lst, key=lambda x: -abs(x[0]))[:2]
        for a, mz, key in lst:
            print('  ADV%+.0f maxz%+.2f · %s' % (a, mz, txt.get(key, '')[:95]
                                                 .replace('\n', ' ')))
    show('CURRENT wins mid history-thread (long/specific msg?)', revert_cur)
    show('HISTORY wins mid current-thread (short/deictic msg?)', revert_hist)
    return 0


if __name__ == '__main__':
    sys.exit(main())
