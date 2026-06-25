#!/usr/bin/env python3
"""LAF patchy test — realizable FIND-union (RRF over field-activations) vs the oracle ceiling.

best-field (§18.12) = 32%/51% is an ORACLE (right field per gold, hindsight). This tests
DEPLOYABLE fusions — no hindsight — to see how much of that ceiling a real fusion captures.
Settles the central question (8bcc8c96): is uniform fusion enough, or do we need a selector?

Deployable variants (all on the 73-cue corpus, essential gold):
  rrf_all  : RRF(n) = Σ_field 1/(K+rank_field(n))            — union over ALL fields
  rrf_core : same, semantic fields only (drop sparse voice/meta)
  rrf_max  : score(n) = max_field 1/(K+rank_field(n))         — each node by its OWN best field
             (the realizable shadow of the oracle; oracle = per-gold min-rank with hindsight)

Patchy: pure matmul + rank-fusion, no engine. K=60.
Run: ./dev python3 eval/laf/rrf_union_patch.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                             # noqa: E402
from endo_baseline_recall import load_corpus, make_baseline_ranker  # noqa: E402

K_RRF = 60
TOPN = 500                                               # per-field contribution cap
CORE = {"_primary", "content", "title", "question", "situation", "reasoning"}


def best_rank(golds, ranked_ids):
    pos = {nid: i + 1 for i, nid in enumerate(ranked_ids)}
    rs = [pos[g] for g in golds if g in pos]
    return min(rs) if rs else None


def rank_from_scores(score_dict, golds, descending=True):
    order = sorted(score_dict.items(), key=lambda x: -x[1] if descending else x[1])
    return best_rank(golds, [nid for nid, _ in order])


def main():
    corpus = load_corpus()
    with IsolatedBrain() as env:
        if not embedder.is_ready():
            embedder.load_model()
        conn = env.brain.conn
        vtypes = [r[0] for r in conn.execute(
            "SELECT DISTINCT vector_type FROM node_enrichments").fetchall()]
        vt = {}
        for v in vtypes:
            rows = conn.execute(
                "SELECT n.id, n.created_at, e.embedding FROM node_enrichments e "
                "JOIN nodes n ON n.id=e.node_id WHERE e.vector_type=? AND n.archived=0",
                (v,)).fetchall()
            if rows:
                vt[v] = ([r[0] for r in rows], np.array([r[1] or "" for r in rows]),
                         np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]))
        ranker = make_baseline_ranker(env.brain)

        names = ['pipeline', 'primary', 'bestfield', 'rrf_all', 'rrf_core', 'rrf_max']
        agg = {k: {'h5': 0, 'h25': 0, 'n': 0} for k in names}

        for c in corpus:
            golds, cutoff = c["gold_essential"], c["cutoff"]
            cv = np.frombuffer(embedder.embed_batch([c["query"]], kind="query")[0],
                               dtype=np.float32)
            ranks = {'pipeline': best_rank(golds, ranker(c))}

            field_gold_rank = {}
            rrf_all, rrf_core, rrf_max = {}, {}, {}
            for v in vt:
                ids, created, M = vt[v]
                sc = np.where(created < cutoff, M @ cv, -np.inf)
                order = np.argsort(-sc)
                pos = {ids[order[r]]: r + 1 for r in range(len(order))
                       if np.isfinite(sc[order[r]])}
                rs = [pos[g] for g in golds if g in pos]
                field_gold_rank[v] = min(rs) if rs else None
                for r in range(min(TOPN, len(order))):
                    j = order[r]
                    if not np.isfinite(sc[j]):
                        break
                    nid = ids[j]
                    contrib = 1.0 / (K_RRF + r + 1)
                    rrf_all[nid] = rrf_all.get(nid, 0.0) + contrib
                    rrf_max[nid] = max(rrf_max.get(nid, 0.0), contrib)
                    if v in CORE:
                        rrf_core[nid] = rrf_core.get(nid, 0.0) + contrib

            ranks['primary'] = field_gold_rank.get("_primary")
            valid = [r for r in field_gold_rank.values() if r is not None]
            ranks['bestfield'] = min(valid) if valid else None
            ranks['rrf_all'] = rank_from_scores(rrf_all, golds)
            ranks['rrf_core'] = rank_from_scores(rrf_core, golds)
            ranks['rrf_max'] = rank_from_scores(rrf_max, golds)

            for k in names:
                rk = ranks[k]
                agg[k]['n'] += 1
                if rk and rk <= 5:
                    agg[k]['h5'] += 1
                if rk and rk <= 25:
                    agg[k]['h25'] += 1

        def pct(k, m):
            return 100.0 * agg[k][m] / max(agg[k]['n'], 1)

        print("=== realizable FIND-union (RRF) vs oracle ceiling — n=%d ===" % agg['pipeline']['n'])
        print("  %-10s %7s %8s" % ("ranker", "hit@5", "hit@25"))
        for k in names:
            tag = "  (oracle)" if k == "bestfield" else ("  (today)" if k == "pipeline" else "")
            print("  %-10s %5.0f%% %7.0f%%%s" % (k, pct(k, 'h5'), pct(k, 'h25'), tag))


if __name__ == "__main__":
    main()
