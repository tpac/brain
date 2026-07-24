"""What separates the golds we FIND from the golds we MISS?

Every probe so far asked "which lane/gain/gate scores better". This asks the
node-level question instead: is a miss an ENCODING problem (thin content, no
situation), an AGE problem (the 7d shelf / 40% floor), a CONNECTIVITY problem
(orphan vs hub), a PROVENANCE problem (who encoded it), or purely a CUE
problem (vague message)? The answer decides whether recall work belongs on
the read side at all.

Measured at three depths (@5/@10/@25) because "missed" is threshold-relative:
a gold at rank 12 is missed at @5 and found at @25, and lumping it with a
gold at rank 900 would hide the distinction that matters.

Node properties are read as-of the corpus (present-day content length/degree —
a node's content can have been revised after the turn, so treat content_len
and degree as WEAKLY time-leaky; age uses the turn's own date and is honest).

Read-only. Run:  ./dev python3 eval/laf/walker/laf_gold_anatomy.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from miss_anatomy import rank_in                                     # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
import laf_gate_audit as G                                          # noqa: E402

DEPTHS = (5, 10, 25)
REPORT = OUT_DIR / 'laf_gold_anatomy.md'


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def load_node_props(node_ids):
    b = open_brain_ro()
    props = {}
    q = ','.join('?' * len(node_ids))
    ids = list(node_ids)
    for nid, typ, clen, tlen, created, esrc, locked, acc, conf in b.execute(
            'SELECT id, type, LENGTH(COALESCE(content,\'\')), '
            'LENGTH(COALESCE(title,\'\')), created_at, encoding_source, '
            'locked, access_count, confidence FROM nodes WHERE id IN (%s)' % q,
            ids):
        props[nid] = {'type': typ, 'content_len': clen or 0,
                      'title_len': tlen or 0, 'created': created,
                      'enc_src': esrc or '(none)', 'locked': bool(locked),
                      'access_count': acc or 0, 'confidence': conf}
    # degree (all active relations) + co_accessed degree
    deg, codeg = Counter(), Counter()
    for src, tgt, rel in b.execute(
            'SELECT e.source_id, e.target_id, r.relation FROM edges e '
            'JOIN edge_relations r ON r.edge_id=e.edge_id '
            'WHERE (r.archived IS NULL OR r.archived=0)'):
        for x in (src, tgt):
            if x in props:
                deg[x] += 1
                if rel == 'co_accessed':
                    codeg[x] += 1
    # kv presence
    kv = defaultdict(set)
    for nid, key in b.execute(
            'SELECT node_id, key FROM node_metadata_kv WHERE node_id IN (%s)'
            % q, ids):
        kv[nid].add(key)
    b.close()
    for nid, p in props.items():
        p['degree'] = deg.get(nid, 0)
        p['co_degree'] = codeg.get(nid, 0)
        p['has_situation'] = 'situation' in kv.get(nid, ())
        p['has_question'] = 'question' in kv.get(nid, ())
        p['has_reasoning'] = 'reasoning' in kv.get(nid, ())
        p['has_quote'] = bool({'user_raw_quote', 'anchor_raw_quote'}
                              & kv.get(nid, set()))
    return props


def split_table(L, title, rows, keyfn, found_flag, min_n=8):
    """Generic found-vs-missed contingency table with a lift column."""
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        buckets[k][0] += int(found_flag(r))
        buckets[k][1] += 1
    L += ['### %s' % title, '',
          '| bucket | n | found | rate | vs overall |', '|---|---|---|---|---|']
    overall = (sum(b[0] for b in buckets.values())
               / max(1, sum(b[1] for b in buckets.values())))
    for k, (f, t) in sorted(buckets.items(), key=lambda x: -x[1][1]):
        if t < min_n:
            continue
        L.append('| %s | %d | %d | %.0f%% | %+.0fpp |'
                 % (k, t, f, 100 * f / t, 100 * (f / t - overall)))
    L.append('')
    return L


def main():
    turns, n = A.build()
    P = G.prep(turns)
    tt = [p['t'] for p in P]
    N = len(P)
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']

    # shipped-gain ranks → found/missed at each depth
    ranks = G.ranks_fast(P, dict(A.GAINS), 0.0)
    gold_ids = [master[t['gr']] for t in tt]
    props = load_node_props(set(gold_ids))
    print('node props loaded for %d golds' % len(props))

    rows = []
    for i, t in enumerate(tt):
        nid = gold_ids[i]
        p = dict(props.get(nid, {}))
        turn_dt = None
        # turn date from the bundle ts already used for the corpus cut
        cr = iso(p.get('created'))
        p['rank'] = ranks[i]
        p['stratum'] = t['stratum']
        p['cur_maxz'] = t['cur_maxz']
        p['gold_in_graph'] = t['gold_in_graph']
        p['node_id'] = nid
        # gold's best lane + its z there
        zg = {**t['zl'], 'graph': t['graph_z']}
        best_ln, best_r = None, None
        for ln, z in zg.items():
            r = rank_in(z, t['gr'])
            if r is not None and (best_r is None or r < best_r):
                best_r, best_ln = r, ln
        p['best_lane'] = best_ln
        p['best_lane_rank'] = best_r
        p['created_dt'] = cr
        rows.append(p)

    # age: node created vs turn ts (turn ts from bundles, via A.build cutoff data)
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    for i, t in enumerate(tt):
        bd = bundles.get(t['key'])
        tdt = iso(bd['ts']) if bd else None
        cr = rows[i]['created_dt']
        rows[i]['age_days'] = ((tdt - cr).days
                               if (tdt and cr and tdt >= cr) else None)

    L = ['# Gold anatomy — what separates the golds we find from the ones we '
         'miss', '',
         'n=%d clean valids ≥%s · shipped gains · tie-fair ranks. '
         '"found@k" = gold ranked ≤k in the full field.' % (N, A.CUTOFF), '']

    # headline: rank distribution of the misses
    rk = np.array([r['rank'] for r in rows if r['rank'] is not None])
    L += ['## The miss distribution — how badly do we miss?', '',
          '| band | n | share |', '|---|---|---|']
    bands = [(1, 5), (6, 10), (11, 25), (26, 100), (101, 500), (501, 10 ** 9)]
    for lo, hi in bands:
        c = int(((rk >= lo) & (rk <= hi)).sum())
        L.append('| rank %s | %d | %.0f%% |'
                 % ('%d–%d' % (lo, hi) if hi < 10 ** 9 else '%d+' % lo,
                    c, 100 * c / len(rk)))
    L += ['', '- reach: @5 %.1f%% · @10 %.1f%% · @25 %.1f%% · median miss rank '
          '%.0f (of misses@25: %d)'
          % (100 * np.mean(rk <= 5), 100 * np.mean(rk <= 10),
             100 * np.mean(rk <= 25), np.median(rk[rk > 25]),
             int((rk > 25).sum())), '',
          '**The @5→@25 jump is the cheapest read on how much is "nearly '
          'there" vs "structurally unreachable".**', '']

    # cue-sharpness quartile (needed by the per-depth tables below)
    cmz = np.array([r['cur_maxz'] for r in rows])
    qs = np.percentile(cmz, [25, 50, 75])
    for r in rows:
        r['_q'] = 'Q%d' % (int(np.digitize(r['cur_maxz'], qs)) + 1)

    # per-depth property tables
    for at in DEPTHS:
        found = lambda r, _a=at: (r['rank'] is not None and r['rank'] <= _a)
        L += ['## Found@%d vs missed@%d' % (at, at), '']
        L = split_table(L, 'By node type (@%d)' % at, rows,
                        lambda r: r.get('type'), found)
        L = split_table(L, 'By age at recall time (@%d)' % at, rows,
                        lambda r: (None if r.get('age_days') is None else
                                   ('0–7d' if r['age_days'] <= 7 else
                                    '8–30d' if r['age_days'] <= 30 else
                                    '31–90d' if r['age_days'] <= 90 else '90d+')),
                        found)
        L = split_table(L, 'By connectivity — active edge degree (@%d)' % at,
                        rows,
                        lambda r: (None if r.get('degree') is None else
                                   ('0–2' if r['degree'] <= 2 else
                                    '3–8' if r['degree'] <= 8 else
                                    '9–20' if r['degree'] <= 20 else '21+')),
                        found)
        L = split_table(L, 'By content length (@%d)' % at, rows,
                        lambda r: (None if not r.get('content_len') else
                                   ('<500' if r['content_len'] < 500 else
                                    '500–1500' if r['content_len'] < 1500 else
                                    '1500–3000' if r['content_len'] < 3000
                                    else '3000+')),
                        found)
        L = split_table(L, 'By encoding_source (@%d)' % at, rows,
                        lambda r: r.get('enc_src'), found)
        L = split_table(L, 'By encoding completeness (@%d)' % at, rows,
                        lambda r: ('situation+question' if
                                   (r.get('has_situation') and r.get('has_question'))
                                   else 'situation only' if r.get('has_situation')
                                   else 'question only' if r.get('has_question')
                                   else 'neither'),
                        found)
        L = split_table(L, 'By stratum (@%d)' % at, rows,
                        lambda r: r.get('stratum'), found)
        L = split_table(L, 'By cue sharpness quartile (@%d)' % at, rows,
                        lambda r: r.get('_q'), found)

    # which lane is the gold's BEST, found vs missed
    L += ['## Which lane holds the gold — found@10 vs missed@10', '',
          '| best lane for the gold | found | missed | found rate |',
          '|---|---|---|---|']
    bl = defaultdict(lambda: [0, 0])
    for r in rows:
        f = r['rank'] is not None and r['rank'] <= 10
        bl[r['best_lane']][0 if f else 1] += 1
    for ln, (f, m) in sorted(bl.items(), key=lambda x: -(x[1][0] + x[1][1])):
        L.append('| %s | %d | %d | %.0f%% |'
                 % (ln, f, m, 100 * f / max(1, f + m)))
    L.append('')

    # the structurally-unreachable tail: what are they?
    tail = [r for r in rows if r['rank'] is not None and r['rank'] > 100]
    L += ['## The deep tail (rank >100, n=%d) — the structurally hard golds'
          % len(tail), '',
          '| property | tail median/share | all-golds median/share |',
          '|---|---|---|']
    def med(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return np.median(v) if v else float('nan')
    for k in ('degree', 'co_degree', 'content_len', 'age_days',
              'access_count'):
        L.append('| %s (median) | %.0f | %.0f |'
                 % (k, med(tail, k), med(rows, k)))
    for k in ('has_situation', 'has_question', 'has_quote', 'gold_in_graph'):
        L.append('| %s (share) | %.0f%% | %.0f%% |'
                 % (k, 100 * np.mean([bool(r.get(k)) for r in tail]),
                    100 * np.mean([bool(r.get(k)) for r in rows])))
    L.append('')
    L += ['- deep-tail types: %s' % ', '.join(
        '%s %d' % (k, v) for k, v in
        Counter(r['type'] for r in tail).most_common(6)), '',
        '- deep-tail strata: %s' % ', '.join(
        '%s %d' % (k, v) for k, v in
        Counter(r['stratum'] for r in tail).most_common()), '']

    # multivariate: which property predicts found@10, held out
    L += ['## Multivariate — which property actually predicts found@10?', '',
          'Session-grouped 5-fold logistic (standardized on train). '
          'Coefficients are log-odds per SD; AUC is held-out. This separates '
          '"correlated with" from "carries independent signal".', '']
    feats = ['cur_maxz', 'degree', 'co_degree', 'content_len', 'age_days',
             'access_count', 'has_situation', 'has_question', 'gold_in_graph']
    X, y, sess = [], [], []
    for i, r in enumerate(rows):
        if r['rank'] is None or r.get('age_days') is None:
            continue
        X.append([float(r.get(f) or 0) for f in feats])
        y.append(1.0 if r['rank'] <= 10 else 0.0)
        sess.append(tt[i]['sess'])
    X, y = np.array(X), np.array(y)
    X[:, feats.index('content_len')] = np.log1p(X[:, feats.index('content_len')])
    X[:, feats.index('degree')] = np.log1p(X[:, feats.index('degree')])
    X[:, feats.index('age_days')] = np.log1p(X[:, feats.index('age_days')])
    us = list(dict.fromkeys(sess))
    fo = {s: i % 5 for i, s in enumerate(us)}
    fold = np.array([fo[s] for s in sess])
    from soft_usage import auc
    coefs, aucs = [], []
    for f in range(5):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
        w = np.zeros(Xtr.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-np.clip(Xtr @ w, -35, 35)))
            g = Xtr.T @ (y[tr] - p) - 1.0 * w
            H = (Xtr * (p * (1 - p))[:, None]).T @ Xtr + 1.0 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w += step
            if np.abs(step).max() < 1e-9:
                break
        coefs.append(w[1:])
        s = Xte @ w
        if y[te].any() and not y[te].all():
            aucs.append(auc(s[y[te] == 1], s[y[te] == 0]))
    C = np.array(coefs)
    L += ['| feature | mean coef (log-odds/SD) | sd across folds | stable? |',
          '|---|---|---|---|']
    order = np.argsort(-np.abs(C.mean(0)))
    for j in order:
        m, s = C[:, j].mean(), C[:, j].std()
        L.append('| %s | %+.3f | %.3f | %s |'
                 % (feats[j], m, s,
                    'yes' if abs(m) > 2 * s and abs(m) > 0.05 else 'no'))
    L += ['', '- held-out AUC of node properties alone: **%.3f** '
          '(0.5 = properties carry nothing; the lanes\' own AUC is ~0.81 on '
          'the pool substrate)' % np.mean(aucs), '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
