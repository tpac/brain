"""Load-bearing routing: does an information measure (IDF-load) or a deixis
measure route the current-vs-history lane BETTER than raw length? (Tom, 2026-
07-18: length correlates with load-bearing but isn't it; the 'let's open a new
stream with THIS' case has full intent in 6 generic words — the load is in the
referent, so recall must route to history, not be self-centered.)

Features per op message:
  op_len     char length (the proxy we already have)
  idf_load   mean message-corpus IDF of content tokens (info density; low=generic)
  deixis     fraction of tokens that are deictic/anaphoric pointers
Target ADV = rank_hist - rank_cur (>0 current better). Thesis: high idf_load ->
current better (ADV up); high deixis -> history better (ADV down); and |corr|
for load/deixis should beat op_len's.

Run: ./dev python3 eval/laf/walker/load_probe.py    (WALKER_OUT_DIR to switch corpus)
"""
import re
import sys
from collections import Counter
from math import log
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                              # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402

DEIXIS = {'this', 'that', 'these', 'those', 'it', 'they', 'them', 'here',
          'there', 'same', 'one', 'above', 'below', 'previous', 'latter',
          'former', 'said', 'mentioned', 'thing', 'stuff', 'such'}
TOK = re.compile(r"[a-z']+")


def tokens(txt):
    return TOK.findall((txt or '').lower())


def rank_on(lane, nc, g):
    z = _zscore(lane, nc)
    return int((z > z[g]).sum())


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    text = {}
    df = Counter()
    ntxt = 0
    for sess, epoch, seq, op_len, txt in walker.execute(
            "SELECT session_id, epoch, seq, op_len, op_text FROM turns"):
        text[(sess, epoch, seq)] = (op_len or len(txt or ''), txt or '')
        toks = set(tokens(txt))
        if toks:
            ntxt += 1
            df.update(toks)
    walker.close()

    def idf_load(txt):
        tks = [t for t in tokens(txt) if len(t) >= 3 and t not in DEIXIS]
        if not tks:
            return 0.0
        return float(np.mean([log(ntxt / (1 + df.get(t, 0))) for t in tks]))

    def deixis_frac(txt):
        tks = tokens(txt)
        return sum(t in DEIXIS for t in tks) / len(tks) if tks else 0.0

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    rows = []
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any() or td.key not in text:
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
        ol, txt = text[td.key]
        # lane CONFIDENCE (self-calibrating router): how sharply does the
        # current lane peak? flat => no node matches this message => fall back
        zc = np.sort(_zscore(cur, nc))[::-1]
        cur_maxz = float(zc[0])
        cur_gap = float(zc[0] - zc[1]) if len(zc) > 1 else 0.0
        rows.append((ol, idf_load(txt), deixis_frac(txt), cur_maxz, cur_gap,
                     rh - rc))
    A = np.array(rows, float)
    ol, idfl, dx, cmz, cgap, adv = A.T
    print('val gold turns with history: %d  (soft hi=%.2f)' % (len(A), hi))
    print('ADV mean %+.2f (>0=current better)' % adv.mean())
    print('\ncorrelations with ADV (want |x| big):')
    print('  op_len      %+.3f' % np.corrcoef(ol, adv)[0, 1])
    print('  idf_load    %+.3f   (high info -> expect +)' % np.corrcoef(idfl, adv)[0, 1])
    print('  deixis      %+.3f   (high deixis -> expect -)' % np.corrcoef(dx, adv)[0, 1])
    print('  cur_maxz    %+.3f   (sharp current peak -> expect +)'
          % np.corrcoef(cmz, adv)[0, 1])
    print('  cur_gap     %+.3f   (decisive current -> expect +)'
          % np.corrcoef(cgap, adv)[0, 1])

    def table(name, key, edges, cur_reach, hist_reach):
        b = np.digitize(key, edges)
        print('\n%-10s      n    cur_reach  hist_reach   ADV' % name)
        for i in range(len(edges) + 1):
            m = b == i
            if m.sum() < 5:
                continue
            lo = '%.2g' % edges[i - 1] if i else '-inf'
            hiu = '%.2g' % edges[i] if i < len(edges) else 'inf'
            print('  [%6s,%6s) %4d    %4.0f%%      %4.0f%%      %+.2f'
                  % (lo, hiu, int(m.sum()), 100 * cur_reach[m].mean(),
                     100 * hist_reach[m].mean(), adv[m].mean()))

    rcur = (adv < 0).astype(float) * 0            # placeholder, recompute below
    # recompute reach flags for the tables
    rc_flag, rh_flag = [], []
    idx = 0
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any() or td.key not in text:
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
        rc_flag.append(int(rank_on(cur, nc, g) < 5))
        rh_flag.append(int(rank_on(hist, nc, g) < 5))
    rcur = np.array(rc_flag, float)
    rhist = np.array(rh_flag, float)

    qs = np.quantile(idfl, [.25, .5, .75])
    table('idf_load', idfl, list(qs), rcur, rhist)
    table('deixis', dx, [0.001, 0.05, 0.12], rcur, rhist)
    return 0


if __name__ == '__main__':
    sys.exit(main())
