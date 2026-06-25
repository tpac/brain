#!/usr/bin/env python3
"""LAF baseline + per-case review over the frozen endo gold corpus.

Reuses the canonical endo harness (eval/oracle_audit/endo_baseline_recall.py): the SAME
ranker (raw brain.recall, cutoff-filtered, fatigue-isolated, over-fetched) and the SAME
score_one metrics. Adds a per-case REVIEW dump so Tom + Anchor can (a) check we agree with
the teacher's gold and (b) rate the actual top-5 per pull, 1-by-1.

One pass over the corpus: builds the aggregate baseline AND the review doc together.

Run (daemon need not be locked — IsolatedBrain copies the db):
  ./dev python3 eval/laf/baseline_review.py
Writes: eval/laf/baseline_review.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))           # project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))  # endo harness

from tests.isolated_brain import IsolatedBrain                                      # noqa: E402
from endo_baseline_recall import (                                                  # noqa: E402
    load_corpus, score_one, make_baseline_ranker, report, selfchecks,
)

REVIEW_MD = os.path.join(os.path.dirname(__file__), "baseline_review.md")


def snip(s, n=180):
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def main():
    corpus = load_corpus()
    print("corpus: %d cues" % len(corpus))
    with IsolatedBrain() as env:
        b = env.brain
        selfchecks(b, corpus)
        ranker = make_baseline_ranker(b)

        title_cache = {}

        def node_brief(nid):
            if nid in title_cache:
                return title_cache[nid]
            try:
                n = b.get_node(nid)
                brief = (n.get("title"), n.get("type"), snip(n.get("content"))) if n else (nid, "?", "")
            except Exception:
                brief = (nid, "?", "")
            title_cache[nid] = brief
            return brief

        scored, md, compact = [], ["# LAF baseline review — endo gold corpus (raw brain.recall)\n"], []
        for c in corpus:
            ranked = ranker(c)
            results = getattr(ranker, "_last", [])
            ess, helpful = c["gold_essential"], c.get("gold_helpful", [])
            m = score_one(ranked, ess, helpful)
            m.update(source=c["source"], query_type=c["query_type"], id=c["id"])
            scored.append(m)

            rank = m["best_ess_rank"]
            compact.append("  %-18s %-13s rank=%-4s hit@5=%d  %s"
                           % (c["id"], c["query_type"], rank if rank else "MISS",
                              m["hit5_ess"], snip(c["query"], 60)))

            md.append("\n## %s  (%s / %s) — gold rank: %s  hit@5=%d\n"
                      % (c["id"], c["source"], c["query_type"], rank if rank else "**MISS**", m["hit5_ess"]))
            md.append("**cue:** %s\n" % snip(c["query"], 320))
            md.append("**next move (outcome):** %s\n" % snip(c.get("next_move"), 280))
            md.append("**teacher_why:** %s\n" % snip(c.get("teacher_why"), 280))
            md.append("**gold (essential):**")
            for g in ess:
                t, typ, content = node_brief(g)
                pos = next((i + 1 for i, r in enumerate(results) if r.get("id") == g), None)
                md.append("  - `%s` [%s] %s — rank %s — %s"
                          % (g, typ, t, pos if pos else "NOT IN POOL", content))
            if helpful:
                md.append("**silver (helpful):** " + ", ".join("`%s`" % g for g in helpful))
            md.append("**top-5 surfaced:**")
            for i, r in enumerate(results[:5]):
                gid = r.get("id")
                tag = " ⭐GOLD" if gid in ess else (" ·silver" if gid in helpful else "")
                md.append("  %d. `%s` [%s]%s %s — %s"
                          % (i + 1, gid, r.get("type"), tag, r.get("title"), snip(r.get("content"))))

        report(scored, "BASELINE (raw brain.recall, cosine-on-cue)")
        with open(REVIEW_MD, "w") as fh:
            fh.write("\n".join(md))

    print("\n── per-cue (id / type / gold-rank / hit@5 / cue) ──")
    print("\n".join(compact))
    print("\n[review] full per-case top-5 + gold + teacher_why -> %s" % REVIEW_MD)


if __name__ == "__main__":
    main()
