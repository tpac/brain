#!/usr/bin/env python3
"""POST-SPREAD ESSENTIAL COVERAGE (the decisive test) — does the brain's EXISTING spread activation
(Layer 3, the real pipeline's post-surface step) recover the essential cluster Control's recall top-5
misses? Diagnosis said 60% of missed-essential are 1-2 graph-hops from Control's hits = spread territory.

Per control_corpus question: recall -> top-5 seeds (proxy for Haiku's picks) -> _graph_expand from them.
Measure essential-gold coverage at: top5 | top25 | top5+spread(baseline) | top5+spread(cluster).
If post-spread coverage jumps to ~90%+, the compositional gap is CLOSED by the existing mechanism
-> recall is good enough, the whole arc lands at 'stop'. Also A/Bs the 'cluster' (cluster-completion)
variant vs baseline on exactly the compositional case it was built for.

Time-scoped for episodes. Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/control_postspread_eval.py
"""
import os, sys, json, re
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402
from servers.scales.s1.surface import _graph_expand   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


def spread_ids(brain, seeds, qv, variant):
    if variant:
        os.environ['BRAIN_RECALL_VARIANT'] = variant
    else:
        os.environ.pop('BRAIN_RECALL_VARIANT', None)
    try:
        res = _graph_expand(brain, seeds, query_vec=qv)
        return {k[:8] for k in res.get('node_activation', {}).keys()}
    except Exception as e:
        print("   [spread %s err: %s]" % (variant or 'baseline', e))
        return set()
    finally:
        os.environ.pop('BRAIN_RECALL_VARIANT', None)


with IsolatedBrain() as env:
    b = env.brain
    agg = {'top5': [0, 0], 'top25': [0, 0], 'base': [0, 0], 'cluster': [0, 0]}  # [covered, total]
    rows = []
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        try:
            cutoff = cutoff_for(q)
            elig = None
            if cutoff:
                elig = {x[0][:8] for x in b.conn.execute(
                    "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (cutoff,)).fetchall()}
            qv = np.frombuffer(embedder.embed_query(q['query']), dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) or 1.0)
            filt = {"created_at": {"lte": cutoff}} if cutoff else None
            try:
                out = b.recall(query=q['query'], limit=25, filter=filt)
            except Exception:
                out = b.recall(query=q['query'], limit=25)
            ctrl = [(r.get('id') or r.get('node_id'))[:8]
                    for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
            if elig is not None:
                ctrl = [n for n in ctrl if n in elig]
            top5, top25 = set(ctrl[:5]), set(ctrl[:25])
            base = top5 | spread_ids(b, ctrl[:5], qv, None)
            clus = top5 | spread_ids(b, ctrl[:5], qv, 'cluster')
            if elig is not None:
                base = {n for n in base if n in elig}
                clus = {n for n in clus if n in elig}

            def cov(s):
                return sum(1 for e in ess if e in s)
            c5, c25, cb, cc = cov(top5), cov(top25), cov(base), cov(clus)
            for k, v in (('top5', c5), ('top25', c25), ('base', cb), ('cluster', cc)):
                agg[k][0] += v; agg[k][1] += len(ess)
            rows.append((q['id'], q['mode'], len(ess), c5, c25, cb, cc))
            print("#%-4s %-8s ess=%d  top5=%d top25=%d +base=%d +cluster=%d"
                  % (q['id'], q['mode'], len(ess), c5, c25, cb, cc))
        except Exception as e:
            print("#%-4s ERROR %s" % (q['id'], e))

    print("\n=== ESSENTIAL COVERAGE across the pipeline (covered / total essential) ===")
    for k in ('top5', 'top25', 'base', 'cluster'):
        c, t = agg[k]
        label = {'top5': 'recall top-5', 'top25': 'recall top-25',
                 'base': 'top5 + spread(baseline)', 'cluster': 'top5 + spread(cluster)'}[k]
        print("  %-26s %3d/%3d  (%.0f%%)" % (label, c, t, 100.0 * c / max(t, 1)))
    print("\n  READ: if +spread jumps coverage toward ~90%, the compositional gap is CLOSED by the")
    print("  EXISTING mechanism -> recall is good enough. cluster vs baseline = is the cluster variant worth it.")
    json.dump([{'id': r[0], 'mode': r[1], 'ess': r[2], 'top5': r[3], 'top25': r[4], 'base': r[5], 'cluster': r[6]} for r in rows],
              open(f'{ROOT}/eval/oracle_audit/control_postspread_result.json', 'w'), indent=2)
