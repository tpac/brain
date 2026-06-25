#!/usr/bin/env python3
"""LAF burial-decomposition — pipeline vs raw-_primary vs best-field, full corpus.

"To see how it feels" before building the fusion engine. Three rankers on the IDENTICAL
73-cue corpus, essential gold, hit@5 / hit@25:
  - pipeline   : today's full brain.recall (the 19% baseline)
  - primary    : raw _primary cosine alone (is the gold cosine-reachable?)
  - bestfield  : ORACLE over all field-vectors (min gold rank across _primary/question/
                 situation/title/...) = the CEILING a fused stack could reach (realizable
                 only with a selector; this is the upper bound, not a deployable number)

Two diagnostics:
  - BURIAL vs RESCUE: does the pipeline bury cosine-reachable golds (primary≤25, pipeline
    missed) or net-rescue cosine-far ones (pipeline≤25, primary missed)? => is "fix the
    fusion" a real lever?
  - FIELD HISTOGRAM: which field reaches @25 where _primary can't => which layers to build.

Run: ./dev python3 eval/laf/burial_decomp.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                             # noqa: E402
from endo_baseline_recall import load_corpus, make_baseline_ranker  # noqa: E402


def best_rank(golds, ranked_ids):
    pos = {nid: i + 1 for i, nid in enumerate(ranked_ids)}
    rs = [pos[g] for g in golds if g in pos]
    return min(rs) if rs else None


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
            if not rows:
                continue
            ids = [r[0] for r in rows]
            vt[v] = (ids, np.array([r[1] or "" for r in rows]),
                     np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]))

        ranker = make_baseline_ranker(env.brain)
        agg = {k: {'h5': 0, 'h25': 0, 'n': 0} for k in ('pipeline', 'primary', 'bestfield')}
        burial, rescue, field_hist = [], [], {}

        for c in corpus:
            golds, cutoff = c["gold_essential"], c["cutoff"]
            cv = np.frombuffer(embedder.embed_batch([c["query"]], kind="query")[0],
                               dtype=np.float32)
            pr = best_rank(golds, ranker(c))               # pipeline rank
            field_ranks = {}
            for v in vt:
                ids, created, M = vt[v]
                sc = np.where(created < cutoff, M @ cv, -np.inf)
                order = np.argsort(-sc)
                pos = {ids[order[r]]: r + 1 for r in range(len(order))
                       if np.isfinite(sc[order[r]])}
                rs = [pos[g] for g in golds if g in pos]
                field_ranks[v] = min(rs) if rs else None
            primr = field_ranks.get("_primary")
            valid = [(v, r) for v, r in field_ranks.items() if r is not None]
            bestv, bestr = min(valid, key=lambda x: x[1]) if valid else (None, None)

            for key, rk in (('pipeline', pr), ('primary', primr), ('bestfield', bestr)):
                agg[key]['n'] += 1
                if rk and rk <= 5:
                    agg[key]['h5'] += 1
                if rk and rk <= 25:
                    agg[key]['h25'] += 1

            p_in = bool(primr and primr <= 25)
            pl_in = bool(pr and pr <= 25)
            if p_in and not pl_in:
                burial.append((c["id"], "prim=%s" % primr, "pipe=%s" % pr))
            if pl_in and not p_in:
                rescue.append((c["id"], "prim=%s" % primr, "pipe=%s" % pr))
            if bestr and bestr <= 25 and not p_in:
                field_hist[bestv] = field_hist.get(bestv, 0) + 1

        def pct(k, m):
            return 100.0 * agg[k][m] / max(agg[k]['n'], 1)

        print("=== ranker comparison (essential gold, full corpus n=%d) ===" % agg['pipeline']['n'])
        for k in ('pipeline', 'primary', 'bestfield'):
            print("  %-10s hit@5 %2.0f%%   hit@25 %2.0f%%" % (k, pct(k, 'h5'), pct(k, 'h25')))
        print("\n[burial vs rescue — raw-primary vs pipeline @25]")
        print("  BURIAL (primary reached ≤25, pipeline missed): %d" % len(burial))
        for b in burial[:8]:
            print("      %s" % (b,))
        print("  RESCUE (pipeline reached ≤25, primary missed): %d" % len(rescue))
        for r in rescue[:8]:
            print("      %s" % (r,))
        print("\n[best-field reaches @25 where _primary can't — which field carried it]")
        for v, n in sorted(field_hist.items(), key=lambda x: -x[1]):
            print("  %-16s %d" % (v, n))


if __name__ == "__main__":
    main()
