#!/usr/bin/env python3
"""RECALL FEATURE TABLE — turn the labeled corpus into a (query,node) feature
matrix for attribution/segment-mining. For arm B (plain recall/cosine, the
current best), dump every candidate's score + rank + node/graph features +
label (essential/helpful/noise). The substrate for the pattern-mining workflow.

Per (query, candidate-node): recall_score, rank, type, degree (edge count),
access_count, age_days, content_len, label. Buried essentials (rank > K) are
included by scanning recall to a deep limit so we can see WHERE they sit.

Daemon-safe (IsolatedBrain). Writes recall_feature_table.json + prints summary.
Usage: ./dev python3 eval/oracle_audit/recall_feature_table.py
"""
import os, sys, json, re
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                            # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
DEEP = 200            # scan depth — find buried essentials
K = 8                 # inject cut
TIMED = {'episode'}
OUT = f'{ROOT}/eval/oracle_audit/recall_feature_table.json'
NOW = 20260616        # ref date (YYYYMMDD as int) for crude age in days


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


# The real per-result score components (from recall output keys) — these ARE
# the attribution: relevance_score is the final rank key; the rest decompose it.
SCORE_FIELDS = ('relevance_score', 'semantic_score', 'keyword_relevance',
                'recency_score', 'embedding_similarity', 'activation',
                'effective_activation', 'stability')


def scores_of(r):
    return {k: (float(r[k]) if isinstance(r, dict) and r.get(k) is not None else None)
            for k in SCORE_FIELDS}


with IsolatedBrain() as env:
    b = env.brain

    # bulk feature lookups (read-only on the isolated copy)
    def features_for(ids):
        if not ids:
            return {}
        qm = ','.join('?' * len(ids))
        out = {}
        rows = b.conn.execute(
            "SELECT id, type, COALESCE(access_count,0), created_at, "
            "COALESCE(LENGTH(content),0) FROM nodes WHERE id IN (%s)" % qm, ids).fetchall()
        for nid, ty, ac, ca, clen in rows:
            yyyymmdd = int((ca or '2026-01-01')[:10].replace('-', '')) if ca else NOW
            out[nid] = {'type': ty, 'access_count': ac,
                        'age_days': max(0, NOW - yyyymmdd), 'content_len': clen}
        # degree: alive edges touching the node (source OR target)
        deg = {nid: 0 for nid in ids}
        drows = b.conn.execute(
            "SELECT id, ("
            " (SELECT COUNT(*) FROM edges WHERE source_id=nodes.id AND COALESCE(archived,0)=0)"
            "+(SELECT COUNT(*) FROM edges WHERE target_id=nodes.id AND COALESCE(archived,0)=0)"
            ") FROM nodes WHERE id IN (%s)" % qm, ids).fetchall()
        for nid, d in drows:
            deg[nid] = d
        for nid in out:
            out[nid]['degree'] = deg.get(nid, 0)
        return out

    rows_out = []
    keys_logged = False
    for q in QS:
        ess = set(e[:8] for e in q.get('gold_essential', []))
        helpful = set(h[:8] for h in q.get('gold_helpful', []))
        if not ess:
            continue
        cutoff = cutoff_for(q)
        filt = {"created_at": {"lte": cutoff}} if cutoff else None
        qv = np.frombuffer(embedder.embed_query(q['query']), dtype=np.float32)
        try:
            out = b.recall(query=q['query'], limit=DEEP, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=DEEP)
        results = out.get('results', []) if isinstance(out, dict) else (out or [])
        if not keys_logged and results:
            print("recall result keys: %s\n" % sorted(results[0].keys()))
            keys_logged = True

        ranked_ids = [(r.get('id') or r.get('node_id')) for r in results]
        full_ids = [r for r in ranked_ids]
        feats = features_for([fid for fid in full_ids])

        for rank, r in enumerate(results, 1):
            nid = (r.get('id') or r.get('node_id'))
            sid = nid[:8]
            label = 'essential' if sid in ess else ('helpful' if sid in helpful else 'noise')
            f = feats.get(nid, {})
            row = {
                'qid': q['id'], 'mode': q['mode'], 'node': sid, 'rank': rank,
                'label': label,
                'type': f.get('type'), 'degree': f.get('degree'),
                'access_count': f.get('access_count'),
                'age_days': f.get('age_days'), 'content_len': f.get('content_len'),
                'in_topK': rank <= K,
            }
            row.update(scores_of(r))
            rows_out.append(row)

    json.dump(rows_out, open(OUT, 'w'), indent=2)
    print("wrote %d (query,node) rows → %s\n" % (len(rows_out), OUT))

    # ── quick aggregate: the three populations we care about ──
    def agg(rows, name):
        if not rows:
            print("  %-26s (none)" % name); return
        def mean(k):
            vs = [r[k] for r in rows if r.get(k) is not None]
            return (sum(vs) / len(vs)) if vs else float('nan')
        types = {}
        for r in rows:
            types[r['type']] = types.get(r['type'], 0) + 1
        top_types = sorted(types.items(), key=lambda kv: -kv[1])[:3]
        print("  %-26s n=%-4d rel=%.3f sem=%.3f kw=%.3f rec=%.3f act=%.3f | deg=%5.1f access=%6.1f age=%4.0f"
              % (name, len(rows), mean('relevance_score'), mean('semantic_score'),
                 mean('keyword_relevance'), mean('recency_score'), mean('effective_activation'),
                 mean('degree'), mean('access_count'), mean('age_days')))
        print("  %-26s   top-types: %s" % ('', ', '.join('%s:%d' % t for t in top_types)))

    ess_top = [r for r in rows_out if r['label'] == 'essential' and r['in_topK']]
    ess_bur = [r for r in rows_out if r['label'] == 'essential' and not r['in_topK']]
    noise_top = [r for r in rows_out if r['label'] == 'noise' and r['in_topK']]
    noise_all = [r for r in rows_out if r['label'] == 'noise']
    print("=== POPULATION COMPARISON (the attribution question) ===")
    agg(ess_top, 'essential IN top-8')
    agg(ess_bur, 'essential BURIED (>8)')
    agg(noise_top, 'noise IN top-8 (the enemy)')
    agg(noise_all, 'noise (all ranks)')
    print("\nREAD: compare 'essential BURIED' vs 'noise IN top-8' — whichever features")
    print("differ most are the levers. High deg/access on top-noise = hub dominance.")
