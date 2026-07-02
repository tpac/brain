#!/usr/bin/env python3
"""Does enriching the edge `why` text change the conductance steering / graph reach? (§18.21)

The conductance is cos(cue, edge.why), and edge.why = compose_edge_text = "[relation] description"
(the relation label is ALREADY in). Conductance is partially flat (mean 0.499, 14% >0.6). This probe
re-embeds the edge text four ways and re-runs the relational operator (sparse top-25 seed, 2-hop) to
see whether more structure sharpens it:
  desc_only           — description, relation stripped (isolates the relation's contribution)
  [rel] desc          — the current scheme (baseline)
  [rel]               — bare relation, no description (the floor)
  [stype][rel][ttype]desc — add endpoint node TYPES (currently excluded by compose_edge_text)

Reports per variant: conductance mean/std/frac>0.6 + standalone relational reach@5/@25 (need-collapsed).

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_edge_text_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, unit, relational_reinstatement, EdgeIndex,
)
from gold24_diagnostic import load_cues                              # noqa: E402
from gold24_field_audit import need_hit, ranks                       # noqa: E402


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        ntype = dict(brain.conn.execute("SELECT id, type FROM nodes").fetchall())

        na = brain.aspects.by_name("noise")
        noise = list(na.edge_relations) if na else []
        ph = ",".join("?" * len(noise)) if noise else "''"
        rows = brain.conn.execute(
            "SELECT e.source_id, e.target_id, er.relation, COALESCE(er.description,'') "
            "FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE (er.archived IS NULL OR er.archived = 0) AND er.relation NOT IN (%s)" % ph,
            noise).fetchall()
        src, dst, rel, desc = [], [], [], []
        for s, t, r, d in rows:
            if s in idx and t in idx and s != t:
                src.append(idx[s]); dst.append(idx[t]); rel.append(r or ""); desc.append(d or "")
                # stash endpoint types alongside (parallel lists)
        src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
        st = [ntype.get(master[i], "") for i in src]
        tt = [ntype.get(master[i], "") for i in dst]
        print("edges (non-noise, both endpoints embedded): %d" % len(src))

        VARIANTS = {
            "[rel] desc (current)": ["[%s] %s" % (r, d) if d else "[%s]" % r for r, d in zip(rel, desc)],
            "[stype][rel][ttype]desc": ["[%s] [%s] [%s] %s" % (a, r, b, d)
                                        for a, r, b, d in zip(st, rel, tt, desc)],
        }

        def emat_for(texts):
            blobs = embedder.embed_batch(texts, kind="document")
            dim = next((len(np.frombuffer(b, dtype=np.float32)) for b in blobs if b), 768)
            M = np.zeros((len(texts), dim), dtype=np.float32)
            for i, b in enumerate(blobs):
                uv = unit(b)
                if uv is not None:
                    M[i] = uv
            return M

        print("\n  %-26s %-11s %-9s %-8s %-8s" %
              ("edge-text variant", "cond mean", "cond>0.6", "hit@5", "hit@25"))
        for name, texts in VARIANTS.items():
            emat = emat_for(texts)
            edges = EdgeIndex(src, dst, emat, rel, np.array([""] * len(rel), dtype=object))
            h5 = h25 = nc = 0
            cmean = []; chi = []
            for c in cues:
                qv = query_vec(c["query"])
                if qv is None:
                    continue
                elig = (ca != "") & (ca <= c["cutoff"])
                ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
                seed = np.zeros(N)
                top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:25]
                seed[top] = np.clip(ms[top], 0.0, None)
                cond = np.clip(emat @ qv, 0.0, None)
                cmean.append(float(cond.mean())); chi.append(float(np.mean(cond > 0.6)))
                vec = relational_reinstatement(qv, seed, edges, N, hops=2)
                rk = ranks(vec, elig, master)
                m5 = need_hit(rk, c, 5); m25 = need_hit(rk, c, 25)
                if m5 is not None:
                    h5 += m5; h25 += m25; nc += 1
            print("  %-26s %-11.3f %-9.1f %-8s %-8s" %
                  (name, np.mean(cmean), 100 * np.mean(chi),
                   "%.0f%%" % (100 * h5 / nc), "%.0f%%" % (100 * h25 / nc)))
        print("\n  baseline to beat: [rel] desc (current) — relational sparse-2hop was 6%/24% in the audit.")


if __name__ == "__main__":
    main()
