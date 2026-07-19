"""Does a RELATIVE-confidence signal (or a trained combination, or outcome
feedback) route the current-vs-history lane per-message BETTER than the single
best feature cur_maxz? (path "b", 2026-07-18.)

Established (10da74bd / e0ace594 / deaf1fb8): lane preference is 83%/63%
MESSAGE-level, ~1/3 of turns revert against their session; cheap text features
are weak (|corr|<0.17); cur_maxz is the best SINGLE feature (corr +0.115 live /
+0.084 pool60) but weak and doesn't cleanly catch reversions. This probe tests
whether anything cheap beats it.

Target per gold turn (gold = argmax soft, kept only if soft>=90th pctile):
  cur  = z(maxsim, op j0)          # this message's lane
  hist = z(maxsim, best j>=1)      # the moment stack's lane
  ADV  = rank_hist(gold) - rank_cur(gold)   (>0 => current is the better router)

Signals (ALL recall-time — computed from the pool lanes / message text, never
from the gold):
  cur_maxz    max z of current lane           (the incumbent, e0ace594)
  hist_maxz   max z of history lane
  rel_conf    cur_maxz - hist_maxz            (H1: trust whichever peaks sharper)
  cur_gap     top1-top2 z gap, current lane
  hist_gap    top1-top2 z gap, history lane
  gap_rel     cur_gap - hist_gap
  has_question / deixis / idf_load / op_len   (text features, deaf1fb8/e0ace594)

Outputs:
  1. corr(signal, ADV) for every signal, both corpora.
  2. Threshold classifier (fit threshold on TRAIN, eval on VAL): balanced
     accuracy of sign(signal) -> sign(ADV), the honest "does it route" metric
     (ADV is class-imbalanced, so raw accuracy is a trap).
  3. Trained logistic on {all signals} -> sign(ADV), train April-May / eval
     June+, standardized coefficients (which features carry the signal).
  4. Outcome feedback (H3): does the PREVIOUS turn's selected-node lane predict
     THIS turn's ADV sign? (Expected to lag given 83% msg-level reversion.)

Run: ./dev python3 eval/laf/walker/router_signal_probe.py
     WALKER_OUT_DIR=.../0a9baa/walker ./dev python3 .../router_signal_probe.py
"""
import re
import sys
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np

from walker_db import open_walker, WALKER_DIR

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


def hist_lane(td):
    """nanmax maxsim over prior slots op1..8 / anchor1..8 (the moment stack)."""
    h = np.concatenate([td.op['maxsim'][:, 1:K_MAX + 1],
                        td.anchor['maxsim'][:, 1:K_MAX + 1]], axis=1)
    with np.errstate(all='ignore'):
        return np.where(np.all(np.isnan(h), axis=1), np.nan, np.nanmax(h, axis=1))


def peaks(z):
    """top1 maxz and top1-top2 gap of a z-vector."""
    s = np.sort(z)[::-1]
    return float(s[0]), float(s[0] - s[1]) if len(s) > 1 else 0.0


SIGNALS = ['cur_maxz', 'hist_maxz', 'rel_conf', 'cur_gap', 'hist_gap',
           'gap_rel', 'has_question', 'deixis', 'idf_load', 'op_len']


def build_rows(turns, text, df, ntxt):
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

    rows = []       # dict per gold turn
    for td in turns:
        if not np.isfinite(td.soft).any() or td.key not in text:
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] < hi:
            continue
        nc = len(td.cands)
        curl = td.op['maxsim'][:, 0]
        histl = hist_lane(td)
        if not np.isfinite(histl[g]):
            continue                                     # no history to route
        zc, zh = _zscore(curl, nc), _zscore(histl, nc)
        rc = int((zc > zc[g]).sum())
        rh = int((zh > zh[g]).sum())
        cmz, cgap = peaks(zc)
        hmz, hgap = peaks(zh)
        ol, hq, txt = text[td.key]
        rows.append({
            'key': td.key, 'val': td.val, 'adv': rh - rc,
            'cur_maxz': cmz, 'hist_maxz': hmz, 'rel_conf': cmz - hmz,
            'cur_gap': cgap, 'hist_gap': hgap, 'gap_rel': cgap - hgap,
            'has_question': float(hq), 'deixis': deixis_frac(txt),
            'idf_load': idf_load(txt), 'op_len': float(ol),
        })
    return rows, hi


def bal_acc(pred_pos, y_pos):
    """balanced accuracy: mean(TPR, TNR). pred_pos/y_pos boolean arrays."""
    p, y = np.asarray(pred_pos), np.asarray(y_pos)
    tpr = (p & y).sum() / max(1, y.sum())
    tnr = (~p & ~y).sum() / max(1, (~y).sum())
    return 0.5 * (tpr + tnr)


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    text = {}
    df = Counter()
    ntxt = 0
    for sess, epoch, seq, op_len, hq, txt in walker.execute(
            "SELECT session_id, epoch, seq, op_len, has_question, op_text "
            "FROM turns"):
        text[(sess, epoch, seq)] = (op_len or len(txt or ''), int(hq or 0),
                                    txt or '')
        toks = set(tokens(txt))
        if toks:
            ntxt += 1
            df.update(toks)
    walker.close()

    rows, hi = build_rows(turns, text, df, ntxt)
    adv = np.array([r['adv'] for r in rows], float)
    val = np.array([r['val'] for r in rows], bool)
    nz = adv != 0                                    # drop ties for classification
    print('CORPUS %s' % WALKER_DIR)
    print('gold turns w/ history: %d (val %d / train %d) · soft hi=%.2f'
          % (len(rows), val.sum(), (~val).sum(), hi))
    print('ADV mean %+.2f · P(current better) overall %.2f · on val %.2f'
          % (adv.mean(), (adv > 0).mean(), (adv[val] > 0).mean()))

    # ---- 1. correlation of each signal with ADV (val slice, the held-out one)
    print('\n[1] corr(signal, ADV) — val slice (held-out), then all:')
    print('  %-13s   val       all' % 'signal')
    for s in SIGNALS:
        x = np.array([r[s] for r in rows], float)
        cv = np.corrcoef(x[val], adv[val])[0, 1]
        ca = np.corrcoef(x, adv)[0, 1]
        star = ' <-incumbent' if s == 'cur_maxz' else ''
        print('  %-13s  %+.3f    %+.3f%s' % (s, cv, ca, star))

    y = adv > 0                                      # positive = current better

    # ---- 3. trained logistic — GROUP (by session) 5-fold CV, works on both
    # corpora (pool60 has no era split). Standardize inside each fold; predict
    # sign(ADV); balanced-acc pooled over held folds. Grouping by session stops
    # a session's style leaking across the split. ----
    X = np.column_stack([[r[s] for r in rows] for s in SIGNALS]).astype(float)
    sess = np.array([r['key'][0] for r in rows])
    print('\n[2] trained logistic — group(session) 5-fold CV, balanced-acc:')
    subsets = [('ALL 10 signals', SIGNALS),
               ('cur_maxz only', ['cur_maxz']),
               ('rel_conf only', ['rel_conf']),
               ('has_question only', ['has_question']),
               ('cur_maxz+has_question', ['cur_maxz', 'has_question']),
               ('cur_maxz+hist_maxz', ['cur_maxz', 'hist_maxz'])]
    for name, sub in subsets:
        idx = [SIGNALS.index(s) for s in sub]
        ba, coef = group_cv(X[:, idx], y, nz, sess)
        tag = '  <- incumbent' if name == 'cur_maxz only' else ''
        print('  %-24s CV_bal_acc %.3f%s' % (name, ba, tag))
    # full-model coefficients (fit once on all turns, standardized) — direction
    Xs = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    w = fit_logistic_labels(Xs[nz], y[nz])
    print('  full-model standardized coefficients (|w| desc):')
    for i in np.argsort(-np.abs(w)):
        print('    %-13s %+.3f' % (SIGNALS[i], w[i]))

    # ---- 4. outcome feedback: prev turn's selected-node lane -> this ADV ----
    outcome_feedback(turns, rows, hi)
    return 0


def group_cv(X, y, nz, sess, k=5):
    """Group(session) k-fold CV balanced-acc of a logistic sign(ADV) router.
    Standardize inside each fold on train stats; pool held-fold predictions."""
    groups = np.array(sorted(set(sess)))
    rng = np.random.RandomState(0)
    rng.shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    gf = np.array([fold_of[s] for s in sess])
    pred, truth = [], []
    for f in range(k):
        tr = (gf != f) & nz
        te = (gf == f) & nz
        if tr.sum() < 10 or te.sum() < 1:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        w = fit_logistic_labels((X[tr] - mu) / sd, y[tr])
        pred.append(((X[te] - mu) / sd) @ w > 0)
        truth.append(y[te])
    if not pred:
        return float('nan'), None
    return bal_acc(np.concatenate(pred), np.concatenate(truth)), None


def fit_logistic_labels(X, y, lam=1.0, iters=60):
    """Plain L2 logistic to labels y in {0,1} with an intercept, Newton."""
    Xa = np.column_stack([X, np.ones(len(X))])
    k = Xa.shape[1]
    w = np.zeros(k)
    yv = y.astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ w, -35, 35)))
        g = Xa.T @ (yv - p) - lam * w
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + lam * np.eye(k)
        w = w + np.linalg.solve(H, g)
    return w[:-1]                                    # drop intercept for scoring


def outcome_feedback(turns, rows, hi):
    """H3: for each session, order gold turns by ts; does the PREVIOUS gold
    turn's realized lane (sign of its own ADV — which lane actually reached its
    gold) predict THIS turn's ADV sign? Message-level reversion predicts weak."""
    by_key = {r['key']: r for r in rows}
    by_sess = defaultdict(list)
    # need ts order; pull from turns
    ts_of = {td.key: td.ts for td in turns}
    for r in rows:
        by_sess[r['key'][0]].append(r)
    same = tot = 0
    prev_adv, this_adv = [], []
    for sess, rs in by_sess.items():
        rs = sorted(rs, key=lambda r: ts_of.get(r['key'], ''))
        for a, b in zip(rs, rs[1:]):
            if a['adv'] == 0 or b['adv'] == 0:
                continue
            tot += 1
            same += int((a['adv'] > 0) == (b['adv'] > 0))
            prev_adv.append(a['adv'])
            this_adv.append(b['adv'])
    if tot:
        rho = np.corrcoef(np.sign(prev_adv), np.sign(this_adv))[0, 1]
        print('\n[3] outcome feedback (prev gold turn lane -> this): pairs %d'
              % tot)
        print('  P(this lane == prev lane) %.3f  (0.50=no carry)' % (same / tot))
        print('  corr(sign prev ADV, sign this ADV) %+.3f' % rho)


if __name__ == '__main__':
    sys.exit(main())
