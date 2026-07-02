#!/usr/bin/env python3
"""Believability gate for the gold24 feeding — run BEFORE trusting any episodic number.

Tom's mandate (2026-06-29): "recheck you're feeding it the right thing and trust its results."
Episodic's history here is a HARNESS bug, not a substrate gap (8fbe480e: "0 coverage in
IsolatedBrain" was a swallowed AttributeError). So this asserts the inputs are real, in the
SAME IsolatedBrain copy the matrix runs against, before episodic is wired as a column.

Three gates:
  1. GOLD COVERAGE — of the essential gold (node,cue) occurrences: how many EXIST in the copy,
     are NOT archived, HAVE an embedding (∈ master), and pre-date the cue's cutoff. This splits
     "unreachable" into real cosine-far reach-failure vs a node that was never embeddable
     (encode/embed gap) — the residual count is only trustworthy after this split.
  2. EPISODIC LIVENESS — does recall_episodes(query, older_than=cutoff) return episodes; does
     trace_links.gather + nodes_for_traces yield candidate nodes; do any candidates = gold.
  3. CUE SANITY — the query text actually fed (operator prompt / anchor stop), spot-checked.

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_verify.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import MAXSIM_GROUPS, build_field_matrices            # noqa: E402
from servers.scales.s1.trace_links import gather, nodes_for_traces   # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402


def main():
    cues = load_cues()
    print("gold24 feeding verification — %d cues" % len(cues))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, _ = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        master_set = set(master)
        created = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())
        try:
            archived = {r[0] for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE archived=1").fetchall()}
        except Exception:
            archived = set()

        # ── GATE 1: gold coverage (per essential occurrence) ──
        occ = exist = not_arch = embedded = eligible = 0
        miss_exist, miss_embed, miss_elig = [], [], []
        for c in cues:
            for nid in c["ess"]:
                occ += 1
                if nid not in created:
                    miss_exist.append((c["id"], nid)); continue
                exist += 1
                if nid in archived:
                    continue
                not_arch += 1
                in_m = nid in master_set
                if in_m:
                    embedded += 1
                else:
                    miss_embed.append((c["id"], nid))
                ca = created.get(nid, "")
                if ca and ca <= c["cutoff"]:
                    eligible += 1
                elif in_m:
                    miss_elig.append((c["id"], nid, ca, c["cutoff"]))
        print("\n── GATE 1: GOLD COVERAGE (essential occurrences) ──")
        print("  occurrences:        %d  (distinct nodes: %d)"
              % (occ, len({n for c in cues for n in c['ess']})))
        print("  exist in copy:      %d  (%d missing — archived-merged or post-copy?)"
              % (exist, len(miss_exist)))
        print("  not archived:       %d" % not_arch)
        print("  embedded (∈master): %d  (%d NOT embedded = encode/embed gap, NOT reach-failure)"
              % (embedded, len(miss_embed)))
        print("  embedded+eligible:  %d  ← the REACH-ABLE universe (denominator for honest reach)"
              % eligible)
        if miss_exist[:5]:
            print("  e.g. missing:", miss_exist[:5])
        if miss_embed[:5]:
            print("  e.g. un-embedded gold:", miss_embed[:5])

        # ── GATE 2: episodic liveness ──
        print("\n── GATE 2: EPISODIC LIVENESS (recall_episodes → gather → nodes_for_traces) ──")
        try:
            te = brain._trace_dal.conn.execute("SELECT COUNT(*) FROM trace_embeddings").fetchone()[0]
        except Exception as e:
            te = "?(%s)" % e
        print("  trace_embeddings rows in copy: %s" % te)
        sample = [c for c in cues if c["source"] == "operator_msg"][:2] + \
                 [c for c in cues if c["source"] == "anchor_turn"][:2]
        for c in sample:
            ep = brain.recall_episodes(query=c["query"][:600], older_than=c["cutoff"],
                                       scale="s0", limit=15)
            episodes = ep.get("episodes", []) if isinstance(ep, dict) else []
            by_sess = defaultdict(list)
            for e in episodes:
                by_sess[e.get("session_id")].append(e)
            cand = {}
            for sess, eps in by_sess.items():
                if not sess:
                    continue
                surf, enc = gather(brain, sess)
                links = nodes_for_traces(surf, enc, eps)
                for tid, link in links.items():
                    for n in (link.get("surfaced", []) + link.get("encoded", [])):
                        cand[n] = cand.get(n, 0) + 1
            gold_hit = sorted(set(cand) & c["ess"])
            print("  [%s|%s] episodes=%d sessions=%d → candidate nodes=%d | gold∈candidates=%d %s"
                  % (c["id"], c["source"], len(episodes), len(by_sess), len(cand),
                     len(gold_hit), gold_hit[:4]))

        # ── GATE 3: cue sanity ──
        print("\n── GATE 3: CUE TEXT (the query actually fed) ──")
        for c in sample[:1] + sample[2:3]:
            t = c["query"].replace("\n", " ")
            print("  [%s|%s] (%d chars) %s" % (c["id"], c["source"], len(c["query"]),
                                               (t[:200] + "…") if len(t) > 200 else t))


if __name__ == "__main__":
    main()
