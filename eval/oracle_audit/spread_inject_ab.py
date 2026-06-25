#!/usr/bin/env python3
"""SPREAD INJECT-PRECISION A/B — measures what reaches the INJECT point (the
top-K Anchor actually sees), not coverage across the whole activated set.

Primary metric = inject-precision: are the gold-essential nodes ranked into the
top-K, with low hub-noise? (Coverage-in-the-590-node-set hides burial; this
doesn't.) Secondary: essential RANK (burial signal) + NOISE@K (off-target slots).

Arms:
  A  spread        — recall top-5 seeds → _graph_expand → rank by render's own
                     key (node_activation, mean_field_activation) → top-K
  B  top25_cosine  — recall top-25 by cosine, no spread → top-K   (baseline to beat)
  (C semantic-anchored convergence-boost — added after the kernel change)

Ranking for arm A replicates format_surface_output_activation's sort_key exactly.
Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/spread_inject_ab.py [K]
"""
import os, sys, json, re
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                            # noqa: E402
from servers.scales.s1.surface import _graph_expand     # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TIMED = {'episode'}


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


def rank_arm_spread(brain, seeds, qv):
    """Run spread once; return TWO inject orders over the same activated set —
    arm A (activation-primary, the historical render sort_key) and arm A2
    (cosine-primary, mirrors BRAIN_SURFACE_RANK_MODE=cosine stopgap)."""
    res = _graph_expand(brain, seeds, query_vec=qv)
    na = res.get('node_activation', {})
    fa = res.get('field_activation', {})

    def mean_fa(nid):
        f = fa.get(nid, {})
        return (sum(f.values()) / len(f)) if f else 0.0

    ranked_act = sorted(na.keys(), key=lambda n: (na[n], mean_fa(n), n), reverse=True)
    ranked_cos = sorted(na.keys(), key=lambda n: (mean_fa(n), na[n], n), reverse=True)
    return [r[:8] for r in ranked_act], [r[:8] for r in ranked_cos], len(na)


def metrics(ranked, ess, helpful, k):
    """inject-precision metrics over a ranked id-prefix list."""
    topk = ranked[:k]
    topk_set = set(topk)
    ess_set, help_set = set(ess), set(helpful)
    covered = ess_set & topk_set
    # rank (1-indexed) of each essential in the FULL ranked list; None = absent
    ranks = {}
    for e in ess_set:
        ranks[e] = (ranked.index(e) + 1) if e in ranked else None
    noise = [n for n in topk if n not in ess_set and n not in help_set]
    return {
        'cov_k': len(covered), 'ess_total': len(ess_set),
        'ranks': ranks, 'noise_k': len(noise),
    }


with IsolatedBrain() as env:
    b = env.brain
    def _z():
        return {'cov': 0, 'tot': 0, 'noise': 0, 'ranks': [], 'miss': 0, 'slots': 0}
    agg = {'A': _z(), 'A2': _z(), 'B': _z()}
    print("K=%d   per-arm shows cov@K and the [essential ranks]  (x=miss)\n" % K)
    print("%-5s %-8s | %-22s | %-22s | %-22s"
          % ('id', 'mode', 'A:spread(act-rank)', 'A2:spread(cos-rank)', 'B:top25-cosine'))
    print("-" * 96)

    for q in QS:
        ess = [e[:8] for e in q.get('gold_essential', [])]
        if not ess:
            continue
        helpful = [h[:8] for h in q.get('gold_helpful', [])]
        cutoff = cutoff_for(q)
        elig = None
        if cutoff:
            elig = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0",
                (cutoff,)).fetchall()}

        qv = np.frombuffer(embedder.embed_query(q['query']), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        filt = {"created_at": {"lte": cutoff}} if cutoff else None
        try:
            out = b.recall(query=q['query'], limit=25, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=25)
        results = out.get('results', []) if isinstance(out, dict) else (out or [])
        cand = [(r.get('id') or r.get('node_id'))[:8] for r in results]
        if elig is not None:
            cand = [c for c in cand if c in elig]

        # Arm B: top-25 cosine order is the inject order
        rankedB = cand
        # Arms A / A2: spread from top-5 seeds (proxy for Haiku picks), ranked
        # by activation (A) and by cosine (A2) over the SAME activated set.
        seeds_full = [(r.get('id') or r.get('node_id')) for r in results][:5]
        rankedA, rankedA2, _ = rank_arm_spread(b, seeds_full, qv)
        if elig is not None:
            rankedA = [c for c in rankedA if c in elig]
            rankedA2 = [c for c in rankedA2 if c in elig]

        mA = metrics(rankedA, ess, helpful, K)
        mA2 = metrics(rankedA2, ess, helpful, K)
        mB = metrics(rankedB, ess, helpful, K)

        def fmt_ranks(m):
            vals = sorted((r if r else 9999) for r in m['ranks'].values())
            return '[' + ','.join(('x' if v == 9999 else str(v)) for v in vals) + ']'

        for tag, m in (('A', mA), ('A2', mA2), ('B', mB)):
            agg[tag]['cov'] += m['cov_k']; agg[tag]['tot'] += m['ess_total']
            agg[tag]['noise'] += m['noise_k']; agg[tag]['slots'] += K
            for r in m['ranks'].values():
                if r is None:
                    agg[tag]['miss'] += 1
                else:
                    agg[tag]['ranks'].append(r)

        def cell(m):
            return "%d/%d %s" % (m['cov_k'], m['ess_total'], fmt_ranks(m))
        print("%-5s %-8s | %-22s | %-22s | %-22s"
              % (q['id'], q['mode'], cell(mA), cell(mA2), cell(mB)))

    print("\n=== AGGREGATE (K=%d) ===" % K)
    print("%-22s %-22s %-22s %-18s" % ('arm', 'ess cov@K', 'mean ess-rank (n)', 'noise/slot'))
    for tag, label in (('A', 'spread(act-rank)'), ('A2', 'spread(cos-rank)'),
                       ('B', 'top25_cosine')):
        a = agg[tag]
        cov = 100.0 * a['cov'] / max(a['tot'], 1)
        mr = (sum(a['ranks']) / len(a['ranks'])) if a['ranks'] else float('nan')
        noise = 100.0 * a['noise'] / max(a['slots'], 1)
        print("%-22s %3d/%3d (%4.0f%%)      mean=%5.1f miss=%2d (n=%3d)   %4.0f%%"
              % (label, a['cov'], a['tot'], cov, mr, a['miss'], len(a['ranks']), noise))
    print("\nREAD: inject-precision = high ess-cov@K + low ess-rank + low noise. If spread (A) has")
    print("worse rank / higher noise than plain top25 (B), it's burying answers, not surfacing them.")
