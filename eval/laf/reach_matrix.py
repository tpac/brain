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
from collections import defaultdict
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


CUES = ["prompt", "prev_operator", "prev_anchor", "recent_context"]   # query-side seeds
STATE_CUES = os.path.join(os.path.dirname(__file__), "..", "oracle_audit",
                          "endo_corpus", "state_cues.json")


def load_state_cues():
    """{cue_id: state-row} — reconstructed multi-cue state per corpus cue."""
    import json
    return {s["cue_id"]: s for s in json.load(open(STATE_CUES))}


def main():
    corpus = load_corpus()
    state = load_state_cues()
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
        prim_set = set(prim_ids)
        try:
            noise_rels = set(b.aspects.by_name("noise").edge_relations)
        except Exception:
            noise_rels = {"co_accessed", "related_to", "related", "community_member",
                          "emergent_bridge", "co_anchored"}
        print("noise relations excluded from typed graph: %d  e.g. %s"
              % (len(noise_rels), sorted(noise_rels)[:8]))

        # co_accessed-ONLY Hebbian adjacency over the _primary index. co_access is used
        # ONLY as a degree-normalized spread FROM the semantic field (Tom 2026-06-27):
        # persistent Hebbian weight, NO forgetting (forgetting is a future data-derived
        # modulator layer, not baked in). Excluded from every other reach lane.
        #
        # RETIRED SUBSTRATE (2026-08-18, node ab56d25a): co_accessed edges were
        # purged from brain.db — this lane can only measure a live signal on a
        # PRE-RETIREMENT snapshot. On a post-retirement brain it refuses loudly
        # rather than scoring an all-zero lane that reads as "Hebbian spread
        # contributes nothing". A future co-access rebuild derives from
        # surface_selected traces (cache table + its own eval).
        ca_idx = {nid: i for i, nid in enumerate(prim_ids)}
        csrc, cdst, cw = [], [], []
        for s, t, w in conn.execute(
                "SELECT e.source_id, e.target_id, er.weight FROM edge_relations er "
                "JOIN edges e ON e.edge_id=er.edge_id WHERE er.relation='co_accessed' "
                "AND (er.archived IS NULL OR er.archived=0)"):
            if s in ca_idx and t in ca_idx and s != t:
                csrc.append(ca_idx[s]); cdst.append(ca_idx[t]); cw.append(float(w or 0.1))
        if not csrc:
            raise SystemExit(
                "reach_matrix: no co_accessed rows — the family was retired and "
                "purged (2026-08-18). Run this lane against a pre-retirement "
                "brain snapshot, or drop the coaccess lane.")
        csrc = np.asarray(csrc, dtype=np.int64); cdst = np.asarray(cdst, dtype=np.int64)
        cw = np.asarray(cw, dtype=np.float32)
        cdeg = np.zeros(len(prim_ids), dtype=np.float32)
        if csrc.size:
            np.add.at(cdeg, csrc, cw); np.add.at(cdeg, cdst, cw)
        print("co_accessed Hebbian edges over _primary index: %d  (%d nodes have >=1)"
              % (csrc.size, int(np.sum(cdeg > 0))))

        def coaccess_spread(a):
            """One hop of degree-normalized (÷norm) Hebbian spread along co_accessed edges.
            ÷norm = 'spread specific for every node' so a hub can't dominate. NO time decay."""
            nxt = np.zeros_like(a)
            if csrc.size:
                np.add.at(nxt, cdst, a[csrc] * cw)
                np.add.at(nxt, csrc, a[cdst] * cw)
            nz = cdeg > 0
            nxt[nz] = nxt[nz] / cdeg[nz]
            return nxt

        n = len(corpus)
        reach5, reach25, sv25 = defaultdict(int), defaultdict(int), defaultdict(int)
        per_gold_reached = []          # per cue: set(instances reaching ESSENTIAL @25)
        winner = defaultdict(int)
        all_instances = set()
        cue_avail = defaultdict(int)   # how many cues have non-empty text per cue-kind
        ep_cov = defaultdict(int)      # episodic usable-episode coverage per cue-kind

        def embed(t):
            return np.frombuffer(embedder.embed_batch([t], kind="query")[0], dtype=np.float32)

        def graph_neighbors(seed_ids):
            if not seed_ids:
                return set(), set()
            ph = ",".join("?" * len(seed_ids))
            pairs = []
            for q in ("SELECT e.target_id, er.relation FROM edges e JOIN edge_relations er "
                      "ON er.edge_id=e.edge_id WHERE e.source_id IN (%s)" % ph,
                      "SELECT e.source_id, er.relation FROM edges e JOIN edge_relations er "
                      "ON er.edge_id=e.edge_id WHERE e.target_id IN (%s)" % ph):
                for row in conn.execute(q, seed_ids):
                    pairs.append((row[0], row[1]))
            return ({nb for nb, _ in pairs},
                    {nb for nb, rel in pairs if rel not in noise_rels})

        for c in corpus:
            ess, silver, cutoff = c["gold_essential"], c.get("gold_helpful", []), c["cutoff"]
            st = state.get(c["id"], {})
            cue_texts = {"prompt": (c.get("query") or ""),
                         "prev_operator": (st.get("prev_operator") or ""),
                         "prev_anchor": (st.get("prev_anchor") or ""),
                         "recent_context": (st.get("recent_context") or "")}
            in_ctx = [x for x in (st.get("in_context_ids") or []) if x in prim_set]
            cue_vecs = {k: embed(t) for k, t in cue_texts.items() if t.strip()}
            for k, t in cue_texts.items():
                if t.strip():
                    cue_avail[k] += 1
            elig = prim_created < cutoff

            ranks, silver_r = {}, {}    # instance -> ESSENTIAL / SILVER best gold rank

            def record(inst, order):
                ranks[inst] = gold_rank(order, ess)
                silver_r[inst] = gold_rank(order, silver)

            def record_flag(inst, reached_set):
                ranks[inst] = 1 if any(g in reached_set for g in ess) else None
                silver_r[inst] = 1 if any(g in reached_set for g in silver) else None

            # cue × field-vector  (cosine)
            for ck, cv in cue_vecs.items():
                for v in vt:
                    ids, created, M = vt[v]
                    sc = np.where(created < cutoff, M @ cv, -np.inf)
                    order = [ids[i] for i in np.argsort(-sc) if np.isfinite(sc[i])]
                    record("%s×%s" % (ck, v), order)

            # cue × FTS  (lexical)
            for ck, txt in cue_texts.items():
                if not txt.strip():
                    continue
                try:
                    fts_ids = list(b._fts.search(txt, limit=500))
                except Exception as e:
                    print("  [FTS] cue %s/%s raised: %s" % (c["id"], ck, str(e)[:100]))
                    fts_ids = []
                record("%s×FTS" % ck, fts_ids)

            # cue × episodic register-bridge  (cue -> episodes -> max cosine(episode, _primary))
            for ck, txt in cue_texts.items():
                if not txt.strip():
                    continue
                inst = "%s×episodic" % ck
                try:
                    res = b.recall_episodes(query=txt, limit=TOP_EP,
                                            younger_than="2026-01-01", older_than=cutoff)
                    eps = res.get("episodes", []) if isinstance(res, dict) else (res or [])
                    texts = [(e.get("metadata") or {}).get("content") or e.get("summary") or ""
                             for e in eps]
                    texts = [t[:1500] for t in texts if t]
                    if texts:
                        ep_cov[ck] += 1
                        evs = [np.frombuffer(x, dtype=np.float32)
                               for x in embedder.embed_batch(texts, kind="query")]
                        best = np.full(prim_M.shape[0], -np.inf)
                        for ev in evs:
                            best = np.maximum(best, np.where(elig, prim_M @ ev, -np.inf))
                        order = [prim_ids[i] for i in np.argsort(-best) if np.isfinite(best[i])]
                        record(inst, order)
                    else:
                        ranks[inst] = silver_r[inst] = None
                except Exception as e:
                    print("  [episodic] cue %s/%s raised: %s" % (c["id"], ck, str(e)[:120]))
                    ranks[inst] = silver_r[inst] = None

            # semantic-edge 1-hop reach (TYPED — co_accessed EXCLUDED via noise_rels)
            if "prompt" in cue_vecs:
                sc = np.where(prim_created < cutoff, prim_M @ cue_vecs["prompt"], -np.inf)
                top25 = [prim_ids[i] for i in np.argsort(-sc)[:25]]
                _, ty = graph_neighbors(top25)
                record_flag("graph_prompttop25_typed", ty)

                # co_access Hebbian spread FROM the semantic top-25 — the ONLY use of
                # co_accessed: seeded by semantic, follows ONLY co_accessed edges, so it can't
                # reach gold the semantic field didn't seed near (not leaky). Persistent weight,
                # no forgetting. Measured as REACH-complement (does it bring gold the semantic
                # edges miss?), NOT additive-rerank — fusing it into the score is a commensurate
                # z-scaling question deferred to Phase B (same lesson as the LAF settling field).
                seed = np.zeros(len(prim_ids), dtype=np.float32)
                for x in top25:
                    if x in ca_idx:
                        seed[ca_idx[x]] = 1.0
                ca_nbrs = {prim_ids[i] for i in np.nonzero(coaccess_spread(seed) > 0)[0]}
                record_flag("coaccess_from_primtop25", ca_nbrs)
            _, ty = graph_neighbors(in_ctx)
            record_flag("graph_incontext_typed", ty)

            # tally — ESSENTIAL drives reach/union/cover; SILVER is the secondary reach lens
            reached25 = set()
            best_inst, best_rk = None, 10 ** 9
            for inst, rk in ranks.items():
                all_instances.add(inst)
                if rk and rk <= 5:
                    reach5[inst] += 1
                if rk and rk <= 25:
                    reach25[inst] += 1
                    reached25.add(inst)
                if rk and rk < best_rk:
                    best_rk, best_inst = rk, inst
            for inst, rk in silver_r.items():
                if rk and rk <= 25:
                    sv25[inst] += 1
            per_gold_reached.append(reached25)
            if best_inst:
                winner[best_inst] += 1

        def pct(x):
            return 100.0 * x / max(n, 1)

        def union_over(pred):
            return sum(1 for s in per_gold_reached if any(pred(i) for i in s))

        print("\n=== LAF reach matrix — FULL BANK (multi-cue × operators, n=%d) ===" % n)
        print("cue-text availability (of %d):  %s" % (
            n, "  ".join("%s=%d" % (k, cue_avail[k]) for k in CUES)))
        print("episodic usable-episode coverage:  %s" % (
            "  ".join("%s=%d" % (k, ep_cov[k]) for k in CUES)))

        # ceiling ladder — what each cue group adds to the realizable @25 union (ESSENTIAL)
        print("\nUNION reach @25 ceiling ladder (ESSENTIAL gold, %d cues):" % n)
        ladder = [
            ("prompt only", lambda i: i.startswith("prompt×") or i.startswith("graph_prompttop25")),
            ("+ prev_operator", lambda i: i.startswith(("prompt×", "prev_operator×")) or i.startswith("graph_prompttop25")),
            ("+ prev_anchor", lambda i: i.startswith(("prompt×", "prev_operator×", "prev_anchor×")) or i.startswith("graph_prompttop25")),
            ("+ recent_context", lambda i: not i.startswith("graph_incontext")),
            ("+ graph_incontext (FULL)", lambda i: True),
        ]
        for label, pred in ladder:
            u = union_over(pred)
            print("  %-26s %3d  (%.0f%%)" % (label, u, pct(u)))

        # greedy set-cover to the full union = the de-redundant BUILD MENU
        print("\ngreedy set-cover of the ESSENTIAL @25 union (the build menu):")
        covered = [False] * n
        remaining = set(all_instances)
        while True:
            best, gain = None, 0
            for inst in remaining:
                g = sum(1 for i, s in enumerate(per_gold_reached) if not covered[i] and inst in s)
                if g > gain:
                    gain, best = g, inst
            if not best or gain == 0:
                break
            for i, s in enumerate(per_gold_reached):
                if best in s:
                    covered[i] = True
            remaining.discard(best)
            print("  +%-34s covers +%d  (cum %d, %.0f%%)"
                  % (best, gain, sum(covered), pct(sum(covered))))

        # top instances by ESSENTIAL reach@25, with reach@5 and SILVER reach@25 alongside
        print("\ntop instances  (reach@5 / reach@25 ess / reach@25 SILVER):")
        for inst in sorted(all_instances, key=lambda s: -reach25[s])[:24]:
            print("  %-32s  %3d / %3d / %3d" % (inst, reach5[inst], reach25[inst], sv25[inst]))

        print("\nwinning-instance histogram (best ESSENTIAL rank per cue):")
        for inst, cnt in sorted(winner.items(), key=lambda x: -x[1])[:18]:
            print("  %-32s %d" % (inst, cnt))


if __name__ == "__main__":
    main()
