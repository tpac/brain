#!/usr/bin/env python3
"""Standalone audit of the THREE episodic sub-fields on the lens-independent 24-cue gold.

The episodic counterpart to `gold24_field_audit.py` — but where that audits the cosine/graph/
temporal fields, this audits the proper three-way episodic operator (`episodic_ops.py`):

  • episodic_encoded  (+act)  — nodes learned in similar past moments
  • episodic_picked   (+act)  — nodes Haiku surfaced+selected in similar past moments
  • episodic_dropped  (−inhibit) — the ÷prevalence drop-RATE (repeatedly offered, never picked)

Two jobs:
  1. STANDALONE REACH — each +act sub-field's OWN need-collapsed hit@5 / hit@25 on the 24-cue
     gold (does seeding from similar moments reach the gold at all, by itself?).
  2. INHIBITION VALIDITY — `dropped` must land on NOISE, not gold. Cross-checked two ways:
       (a) gold-avoidance: do dropped-inhibited nodes overlap the cues' GOLD set? (want low)
       (b) anti-gold overlap: for the two cues with hand-judged inhibition cards
           (operator_msg_0094, anchor_turn_0345), do the top dropped-inhibited nodes overlap
           the judges' `noise` lists (GOOD) and AVOID the `clean`/gold lists (GOOD)?

Honesty gate (Tom's mandate): recall_episodes empty ≠ truth. We count cues that returned no
episodes and report them separately so an empty-feed can't masquerade as a zero score.

Run (daemon maintenance-locked — 2nd embedder contends):
  ./dev python3 eval/laf/gold24_episodic_audit.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import MAXSIM_GROUPS, build_field_matrices            # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from episodic_ops import (                                            # noqa: E402
    episodic_roles, episodic_encoded, episodic_picked,
    episodic_dropped, episodic_dropped_detail, DEFAULT_TOP_MOMENTS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "gold24_episodic_audit.md")
# the two hand-judged inhibition anti-gold cards (scratchpad-staged this session)
SCRATCH = os.environ.get(
    "BRAIN_SCRATCH",
    "/private/tmp/claude-503/-Users-tpac-brain--claude-worktrees-gallant-ellis-81dc77/"
    "a27d5563-9539-44c5-954d-e7094bd644ed/scratchpad")
INHIB_CARDS = {
    "operator_msg_0094": os.path.join(SCRATCH, "inhib_operator_msg_0094.json"),
    "anchor_turn_0345": os.path.join(SCRATCH, "inhib_anchor_turn_0345.json"),
}

# moment-window seam config to report on (the default + one ±1-turn variant)
WINDOWS = [("turn", "turn"), (("window", 1), "±1-turn")]


def need_hit(rank_of_node, cue, k):
    """need-collapsed hit@k for one cue given {node_id: rank}. (from gold24_field_audit)"""
    needs = defaultdict(list)
    for nid in cue["ess"]:
        nd = next((n for n, ids in cue["needs"].items() if nid in ids), nid)
        needs[nd].append(nid)
    if not needs:
        return None
    met = sum(1 for nids in needs.values()
              if any((rank_of_node.get(n) or 1e9) <= k for n in nids))
    return met / len(needs)


def ranks(scores, eligible, master):
    """{node_id: rank} by score desc among eligible nodes. (from gold24_field_audit)"""
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-s)
    return {master[i]: r + 1 for r, i in enumerate(order)}


def main():
    cues = load_cues()
    inhib = {}
    for cid, path in INHIB_CARDS.items():
        if os.path.exists(path):
            inhib[cid] = json.load(open(path))
        else:
            print("  ! anti-gold card missing: %s" % path)

    out = []
    def p(s=""):
        print(s); out.append(s)

    p("episodic three-way audit — %d cues, top=%d moments" % (len(cues), DEFAULT_TOP_MOMENTS))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        p("  embedded nodes (master): %d" % N)

        # gold set across all cues (for the dropped gold-avoidance check)
        all_gold = set().union(*[c["gold"] | c["gplus"] for c in cues]) if cues else set()

        for window, wlabel in WINDOWS:
            p("\n================  MOMENT WINDOW = %s  ================" % wlabel)
            # accumulators
            agg = {f: {"h5": 0.0, "h25": 0.0, "nc": 0} for f in ("encoded", "picked")}
            empty_feed = 0           # cues recall_episodes returned nothing for
            n_with_eps = 0
            # dropped validity accumulators
            drop_on_gold_w = 0.0     # inhibition mass landing on gold nodes
            drop_total_w = 0.0       # total inhibition mass (all nodes)
            drop_nodes_total = 0     # nodes with nonzero inhibition
            drop_nodes_gold = 0      # of those, how many are gold

            for c in cues:
                elig = (ca != "") & (ca <= c["cutoff"])
                # ONE roles pull shared across all three operators (cost: 1 episode search)
                records = episodic_roles(brain, c["query"], c["cutoff"], window=window)
                if not records:
                    empty_feed += 1
                    continue
                n_with_eps += 1

                enc = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=records)
                pick = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=records)
                rate, prevalence = episodic_dropped_detail(
                    brain, c["query"], c["cutoff"], idx, N, _records=records)

                for f, vec in (("encoded", enc), ("picked", pick)):
                    rk = ranks(vec, elig, master)
                    m5, m25 = need_hit(rk, c, 5), need_hit(rk, c, 25)
                    if m5 is not None:
                        agg[f]["h5"] += m5; agg[f]["h25"] += m25; agg[f]["nc"] += 1

                # dropped gold-avoidance: inhibition mass on gold vs total
                inhib_vec = np.where(elig, rate, 0.0)
                for i in np.nonzero(inhib_vec > 1e-9)[0]:
                    nid = master[i]; w = float(inhib_vec[i])
                    drop_total_w += w; drop_nodes_total += 1
                    if nid in all_gold:
                        drop_on_gold_w += w; drop_nodes_gold += 1

            p("\n  -- STANDALONE REACH (+act sub-fields, need-collapsed) --")
            p("    %-10s %-7s %-8s %s" % ("field", "hit@5", "hit@25", "(cues scored)"))
            for f in ("encoded", "picked"):
                a = agg[f]; nc = a["nc"] or 1
                p("    %-10s %-7s %-8s %d" % (
                    f, "%.0f%%" % (100 * a["h5"] / nc),
                    "%.0f%%" % (100 * a["h25"] / nc), a["nc"]))
            p("    feed: %d/%d cues returned episodes  |  %d EMPTY (excluded — empty ≠ truth)"
              % (n_with_eps, len(cues), empty_feed))

            p("\n  -- DROPPED inhibition GOLD-AVOIDANCE (want LOW) --")
            gold_mass_pct = 100 * drop_on_gold_w / (drop_total_w or 1)
            gold_node_pct = 100 * drop_nodes_gold / (drop_nodes_total or 1)
            p("    inhibition mass on GOLD nodes:  %.1f%%  (of total inhibition mass)" % gold_mass_pct)
            p("    inhibited nodes that ARE gold:  %d / %d  (%.1f%%)"
              % (drop_nodes_gold, drop_nodes_total, gold_node_pct))
            p("    reading: low = the drop-rate field correctly avoids the gold it should keep.")

            # ---- anti-gold card cross-check (the two hand-judged inhibition cues) ----
            p("\n  -- DROPPED vs ANTI-GOLD CARDS (top-inhibited overlap noise? avoid clean?) --")
            for cid, card in inhib.items():
                c = next((x for x in cues if x["id"] == cid), None)
                if c is None:
                    p("    [%s] not in the 24-cue gold — skipped" % cid); continue
                elig = (ca != "") & (ca <= c["cutoff"])
                records = episodic_roles(brain, c["query"], c["cutoff"], window=window)
                if not records:
                    p("    [%s] recall_episodes EMPTY — no inhibition to check (not a zero!)" % cid)
                    continue
                rate, prevalence = episodic_dropped_detail(
                    brain, c["query"], c["cutoff"], idx, N, _records=records)
                inhib_vec = np.where(elig, rate, 0.0)
                # top-K inhibited nodes by rate (8-char ids to match the card)
                order = np.argsort(-inhib_vec)
                topk = [master[i][:8] for i in order if inhib_vec[i] > 1e-9][:25]
                noise_ids = {x["node_id"][:8] for x in card.get("noise", [])}
                clean_ids = {x[:8] for x in card.get("clean", [])}
                gold_ids = {g[:8] for g in (c["gold"] | c["gplus"])}
                hit_noise = [x for x in topk if x in noise_ids]
                hit_clean = [x for x in topk if x in clean_ids]
                hit_gold = [x for x in topk if x in gold_ids]
                p("    [%s] top-%d inhibited: noise∩=%d/%d (GOOD)  clean∩=%d (want 0)  gold∩=%d (want 0)"
                  % (cid, len(topk), len(hit_noise), len(noise_ids),
                     len(hit_clean), len(hit_gold)))
                if hit_noise:
                    p("        ✓ correctly inhibited noise: %s" % ", ".join(hit_noise[:8]))
                if hit_clean:
                    p("        ✗ WRONGLY inhibited clean: %s" % ", ".join(hit_clean))
                if hit_gold:
                    p("        ✗ WRONGLY inhibited gold:  %s" % ", ".join(hit_gold))
                if not topk:
                    p("        (no nodes cleared the inhibition threshold for this cue)")

    open(OUT_MD, "w").write(
        "# Episodic three-way audit — lens-independent 24-cue gold\n\n"
        "Generated by `eval/laf/gold24_episodic_audit.py`. Standalone reach per +act sub-field "
        "(encoded/picked) + the −inhibit `dropped` field's gold-avoidance and anti-gold-card "
        "cross-check. The moment-window seam is reported for both the `turn` default and a "
        "`±1-turn` variant.\n\n```\n" + "\n".join(out) + "\n```\n")
    p("\n  → written to %s" % os.path.relpath(OUT_MD))


if __name__ == "__main__":
    main()
