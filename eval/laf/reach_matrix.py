#!/usr/bin/env python3
"""LAF reach matrix — REVERSE analysis: per gold, which signal reaches it?

The methodical (bottom-up) step, after the RRF patch drifted top-down. For each gold in the
73-cue corpus, compute reach under EVERY signal, then the COMBINATION MATH that tells us which
layers to keep and how they complement — instead of imposing a fixed fusion and measuring.

Signals (all on the same eligible-by-cutoff field):
  field-vectors : _primary, content, title, question, situation, reasoning, high_meta, *_raw_quote
  FTS           : lexical bm25 rank
  graph-1hop    : gold is a 1-hop neighbor of the _primary top-25 (reach flag)
  episodic      : cue -> top-K episodes (recall_episodes) -> max cosine(episode, node) -> gold rank

Combination math (the output that matters):
  union @5/@25         : best signal per gold = ceiling across ALL signals
  per-signal reach @25 : what each signal reaches alone
  per-signal UNIQUE @25: golds ONLY that signal reaches  => keep (non-redundant) vs drop (redundant)
  winning-signal hist  : which signal carries each gold

Run: ./dev python3 eval/laf/reach_matrix.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                             # noqa: E402
from endo_baseline_recall import load_corpus             # noqa: E402

TOP_EP = 5          # episodes per cue for the episodic bridge


def gold_rank(order_ids, golds):
    pos = {nid: i + 1 for i, nid in enumerate(order_ids)}
    rs = [pos[g] for g in golds if g in pos]
    return min(rs) if rs else None


def main():
    corpus = load_corpus()
    with IsolatedBrain() as env:
        if not embedder.is_ready():
            embedder.load_model()
        b = env.brain
        conn = b.conn
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
        prim_ids, prim_created, prim_M = vt["_primary"]
        try:
            noise_rels = set(b.aspects.by_name("noise").edge_relations)
        except Exception:
            noise_rels = {"co_accessed", "related_to", "related", "community_member",
                          "emergent_bridge", "co_anchored"}
        print("noise relations excluded from typed graph: %d  e.g. %s"
              % (len(noise_rels), sorted(noise_rels)[:8]))

        # signal columns we tally
        SIGNALS = list(vtypes) + ["FTS", "graph1hop", "graph1hop_typed", "episodic"]
        reach5 = {s: 0 for s in SIGNALS}
        reach25 = {s: 0 for s in SIGNALS}
        per_gold_reach25 = []          # list of sets (which signals reached this gold @25)
        winner = {}                    # winning signal histogram (best rank)
        ep_cov = 0

        for c in corpus:
            golds, cutoff = c["gold_essential"], c["cutoff"]
            cv = np.frombuffer(embedder.embed_batch([c["query"]], kind="query")[0],
                               dtype=np.float32)
            ranks = {}                 # signal -> gold rank (or None)

            # field-vector signals
            for v in vt:
                ids, created, M = vt[v]
                sc = np.where(created < cutoff, M @ cv, -np.inf)
                fin = np.argsort(-sc)
                order = [ids[i] for i in fin if np.isfinite(sc[i])]
                ranks[v] = gold_rank(order, golds)

            # FTS (lexical)
            try:
                fts = b._fts.search(c["query"], limit=500)
                fts_ids = list(fts)
                ranks["FTS"] = gold_rank(fts_ids, golds)
            except Exception:
                ranks["FTS"] = None

            # graph-1hop from _primary top-25 — untyped (all edges) vs typed (drop noise relations)
            try:
                sc = np.where(prim_created < cutoff, prim_M @ cv, -np.inf)
                top25 = [prim_ids[i] for i in np.argsort(-sc)[:25]]
                ph = ",".join("?" * len(top25))
                pairs = []
                for row in conn.execute(
                        "SELECT e.target_id, er.relation FROM edges e JOIN edge_relations er "
                        "ON er.edge_id=e.edge_id WHERE e.source_id IN (%s)" % ph, top25):
                    pairs.append((row[0], row[1]))
                for row in conn.execute(
                        "SELECT e.source_id, er.relation FROM edges e JOIN edge_relations er "
                        "ON er.edge_id=e.edge_id WHERE e.target_id IN (%s)" % ph, top25):
                    pairs.append((row[0], row[1]))
                untyped = {nb for nb, _ in pairs}
                typed = {nb for nb, rel in pairs if rel not in noise_rels}
                ranks["graph1hop"] = 1 if any(g in untyped for g in golds) else None
                ranks["graph1hop_typed"] = 1 if any(g in typed for g in golds) else None
            except Exception:
                ranks["graph1hop"] = None
                ranks["graph1hop_typed"] = None

            # episodic bridge: cue -> top-K episodes -> max cosine(episode, node) -> gold rank
            try:
                eps = b.recall_episodes(query=c["query"], limit=TOP_EP,
                                        younger_than="2026-01-01", older_than=cutoff)
                texts = []
                for e in eps:
                    md = e.get("metadata") or {}
                    t = md.get("content") or e.get("summary") or ""
                    if t:
                        texts.append(t[:1500])
                if texts:
                    ep_cov += 1
                    evs = [np.frombuffer(x, dtype=np.float32)
                           for x in embedder.embed_batch(texts, kind="query")]
                    elig = prim_created < cutoff
                    best = np.full(prim_M.shape[0], -np.inf)
                    for ev in evs:
                        best = np.maximum(best, np.where(elig, prim_M @ ev, -np.inf))
                    order = [prim_ids[i] for i in np.argsort(-best) if np.isfinite(best[i])]
                    ranks["episodic"] = gold_rank(order, golds)
                else:
                    ranks["episodic"] = None
            except Exception as e:
                ranks["episodic"] = None

            # tally
            reached25 = set()
            best_sig, best_rk = None, 10 ** 9
            for s in SIGNALS:
                rk = ranks.get(s)
                if rk and rk <= 5:
                    reach5[s] += 1
                if rk and rk <= 25:
                    reach25[s] += 1
                    reached25.add(s)
                if rk and rk < best_rk:
                    best_rk, best_sig = rk, s
            per_gold_reach25.append(reached25)
            if best_sig:
                winner[best_sig] = winner.get(best_sig, 0) + 1

        n = len(corpus)
        union25 = sum(1 for st in per_gold_reach25 if st)

        def pct(x):
            return 100.0 * x / max(n, 1)

        print("=== LAF reach matrix — combination math (n=%d golds-cues) ===" % n)
        print("episodic coverage: %d/%d cues had usable episodes\n" % (ep_cov, n))
        print("UNION reach @25 (gold reached by >=1 signal): %d  (%.0f%%)\n" % (union25, pct(union25)))
        print("%-16s %8s %8s %8s" % ("signal", "reach@5", "reach@25", "UNIQUE@25"))
        for s in SIGNALS:
            uniq = sum(1 for st in per_gold_reach25 if st == {s})
            print("  %-14s %6d   %6d   %7d" % (s, reach5[s], reach25[s], uniq))
        print("\nwinning-signal histogram (best rank per gold):")
        for s, cnt in sorted(winner.items(), key=lambda x: -x[1]):
            print("  %-14s %d" % (s, cnt))


if __name__ == "__main__":
    main()
