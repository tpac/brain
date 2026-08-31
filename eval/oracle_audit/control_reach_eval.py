#!/usr/bin/env python3
"""CONTROL REACH EVAL (stage 3+4, deterministic) — does plain recall (Control) MISS relevant nodes
that OTHER search modes recover? For each control_corpus question, build the exhaustive reach
(Control recall + keyword/fts + graph-walk from Control's top-5) and find nodes that are:
  (a) NOT in Control's top-25, and
  (b) relevant (cosine to the query >= REL),
i.e. nodes Control missed but a wider search finds — tagged by which mode recovered them (= which
lane would fix it: fts -> lexical, graph -> traversal).

TIME-THE-BRAIN (Tom): episodic questions are scoped to created_at <= their moment (faithful replay);
evergreen ones (trigger/topic/heavy/remote = current-state or stable facts) use today's brain.

Deterministic (no LLM-judge fragility), daemon-safe (IsolatedBrain), robust (per-question try/except,
incremental save). Usage: ./dev python3 eval/oracle_audit/control_reach_eval.py
"""
import sys, json, re
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED_MODES = {'episode'}     # only episodic questions get time-scoped; rest = today
REL = 0.60                    # cosine-to-query threshold for "this missed node was actually relevant"


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED_MODES) else None


with IsolatedBrain() as env:
    b = env.brain
    model = embedder.stats.get('model_name') or None
    vmap = {}
    for r in b._vec_dal.get_all_vectors(vector_types=['_primary'], model=model):
        e = r['embedding']
        if e:
            v = np.frombuffer(e, dtype=np.float32); n = np.linalg.norm(v)
            if n:
                vmap[r['node_id'][:8]] = v / n

    def neighbors(nid):
        rows = b.conn.execute(
            "SELECT target_id FROM edges WHERE source_id = ? "
            "UNION SELECT source_id FROM edges WHERE target_id = ?",
            (nid, nid)).fetchall()
        return [x[0][:8] for x in rows]

    def relv(n8, qv):
        v = vmap.get(n8)
        return float(np.dot(qv, v)) if v is not None else 0.0

    results = []
    for q in CORPUS:
        try:
            cutoff = cutoff_for(q)
            qv = np.frombuffer(embedder.embed_query(q['query']), dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) or 1.0)
            elig = None
            if cutoff:
                elig = {x[0][:8] for x in b.conn.execute(
                    "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (cutoff,)).fetchall()}

            filt = {"created_at": {"lte": cutoff}} if cutoff else None
            try:
                out = b.recall(query=q['query'], limit=25, filter=filt)
            except Exception:
                out = b.recall(query=q['query'], limit=25)   # fallback if filter shape rejected
            ctrl = [(r.get('id') or r.get('node_id'))[:8]
                    for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
            if elig is not None:
                ctrl = [n for n in ctrl if n in elig]
            cset = set(ctrl)

            fts = {n[:8] for n, _ in b._fts.search_scored(q['query'], 25)}
            graph = set()
            for seed in ctrl[:5]:
                graph.update(neighbors(seed))
            alt = fts | graph
            if elig is not None:
                alt = {n for n in alt if n in elig}

            missed = [(n, round(relv(n, qv), 2)) for n in alt if n not in cset]
            missed = sorted([(n, r) for n, r in missed if r >= REL], key=lambda x: -x[1])

            def via(n):
                m = []
                if n in fts: m.append('fts')
                if n in graph: m.append('graph')
                return '+'.join(m) or '?'

            ctrl_best = max((relv(n, qv) for n in ctrl[:5]), default=0.0)
            rec = {'id': q['id'], 'mode': q['mode'], 'cutoff': cutoff, 'query': q['query'],
                   'ctrl_best_rel': round(ctrl_best, 2), 'n_missed_relevant': len(missed),
                   'missed': [{'n': n, 'rel': r, 'via': via(n)} for n, r in missed[:6]]}
            results.append(rec)
            print("#%-4s %-8s cut=%s ctrl_best=%.2f missed_rel=%d  %s"
                  % (q['id'], q['mode'], (cutoff or 'today')[:10], ctrl_best, len(missed),
                     [(m['n'], m['rel'], m['via']) for m in rec['missed'][:3]]))
        except Exception as e:
            print("#%-4s ERROR %s" % (q['id'], e))
            results.append({'id': q['id'], 'error': str(e)})
        json.dump(results, open(f'{ROOT}/eval/oracle_audit/control_reach_result.json', 'w'), indent=2)

    ok = [r for r in results if 'n_missed_relevant' in r]
    fails = [r for r in ok if r['n_missed_relevant'] > 0]
    from collections import Counter
    viac = Counter()
    for r in fails:
        for m in r['missed']:
            viac[m['via']] += 1
    print("\n=== Control misses >=1 RELEVANT node a wider search finds: %d/%d questions ===" % (len(fails), len(ok)))
    print("recovered-by-mode tally (which lane would fix the miss):", dict(viac))
    print("by question-mode:", dict(Counter(r['mode'] for r in fails)))
