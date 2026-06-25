"""LAF parity harness — does log-additive recomposition preserve production rankings?

Piece 1 of the LAF MVP (docs/RECALL-SR-REDESIGN.md §18.6). Runs production recall over
the existing control query set against an IsolatedBrain copy, recomposes each candidate's
score in log-additive form, and reports whether the ranking + essential-gold coverage hold.

Scope note: production already cut to top-`limit` by its own score, so this measures
RE-RANKING AGREEMENT within the surfaced set (the control-safety question for piece 1) —
not whether log-additive would retrieve different candidates. That's the right question
here: "does recomposing the algebra scramble what production already surfaced?"

Run:  ./dev python3 eval/laf/parity_harness.py
"""
import os
import sys
import json
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # project root
sys.path.insert(0, _HERE)                                     # for the scorer

from tests.isolated_brain import IsolatedBrain                # noqa: E402
from log_additive_scorer import (                             # noqa: E402
    extract_features, replay, log_additive, residual_health,
)

CONTROL = os.path.join(os.path.dirname(_HERE), 'oracle_audit', 'control_gold_result.json')


def sid(x):
    return (x or '')[:8]


def topk_overlap(a, b, k):
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / max(len(sa), 1)


def kendall_tau(order_a, order_b):
    """Rank correlation over the common id set. Simple O(n^2)."""
    sb = set(order_b)
    common = [x for x in order_a if x in sb]
    rank_b = {x: i for i, x in enumerate(order_b)}
    seq = [rank_b[x] for x in common]
    n = len(seq)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                conc += 1
            elif seq[i] > seq[j]:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def main():
    with open(CONTROL) as fh:
        queries = json.load(fh)

    feats_by_q = {}
    all_bases = []
    with IsolatedBrain() as env:
        for q in queries:
            out = env.brain.recall(query=q['query'], limit=25)
            results = out.get('results') or out.get('nodes') or []
            feats = [f for f in (extract_features(n) for n in results) if f]
            feats_by_q[q['id']] = (q, feats)
            all_bases.extend(f['base'] for f in feats)

    base_ref = statistics.median(all_bases) if all_bases else 0.5

    # --- validation pass: is the extraction faithful? ---
    repl_err = 0.0
    health = {'ok': 0, 'negative': 0, 'oversized': 0}
    worst = []
    for qid, (q, feats) in feats_by_q.items():
        for f in feats:
            repl_err = max(repl_err, abs(replay(f) - f['production']))
            h = residual_health(f)
            health[h] += 1
            if h != 'ok':
                worst.append((qid, sid(f['id']), round(f['additive_eff'], 3), f['C'], f['M']))

    # --- ranking comparison: log-additive vs production ---
    agg = {k: [] for k in ('top5', 'top25', 'tau',
                           'ess5_p', 'ess5_l', 'ess25_p', 'ess25_l')}
    for qid, (q, feats) in feats_by_q.items():
        if not feats:
            continue
        prod_order = [sid(f['id']) for f in sorted(feats, key=lambda f: -f['production'])]
        la_scored = [(f, log_additive(f, base_ref)[0]) for f in feats]
        la_order = [sid(f['id']) for f, _ in sorted(la_scored, key=lambda t: -t[1])]
        agg['top5'].append(topk_overlap(prod_order, la_order, 5))
        agg['top25'].append(topk_overlap(prod_order, la_order, 25))
        agg['tau'].append(kendall_tau(prod_order, la_order))
        ess = {sid(e) for e in q.get('essential', [])}
        if ess:
            agg['ess5_p'].append(sum(1 for x in prod_order[:5] if x in ess) / len(ess))
            agg['ess5_l'].append(sum(1 for x in la_order[:5] if x in ess) / len(ess))
            agg['ess25_p'].append(sum(1 for x in prod_order[:25] if x in ess) / len(ess))
            agg['ess25_l'].append(sum(1 for x in la_order[:25] if x in ess) / len(ess))

    def m(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    ncand = sum(len(v[1]) for v in feats_by_q.values())
    print("=== LAF parity: production vs log-additive recomposition ===")
    print("control queries: %d   candidates: %d   base_ref(median): %.3f"
          % (len(queries), ncand, base_ref))
    print()
    print("[validation] replay max abs err vs production: %.5f  (tol ~0.001, emb rounding)"
          % repl_err)
    print("[validation] residual health: ok=%d negative=%d oversized=%d"
          % (health['ok'], health['negative'], health['oversized']))
    if worst:
        print("             flagged (qid, id, additive_eff, C, M):")
        for w in worst[:12]:
            print("               %s" % (w,))
    print()
    print("[ranking agreement]  log-additive vs production")
    print("  top-5 overlap   : %.3f" % m(agg['top5']))
    print("  top-25 overlap  : %.3f" % m(agg['top25']))
    print("  Kendall tau     : %.3f" % m(agg['tau']))
    print()
    print("[essential-gold coverage]   production -> log-additive")
    print("  ess@5  : %.3f -> %.3f" % (m(agg['ess5_p']), m(agg['ess5_l'])))
    print("  ess@25 : %.3f -> %.3f" % (m(agg['ess25_p']), m(agg['ess25_l'])))


if __name__ == '__main__':
    main()
