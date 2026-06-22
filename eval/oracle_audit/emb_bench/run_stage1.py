#!/usr/bin/env python3
"""Stage 1 orchestrator — fastembed sweep, one subprocess per model.

Subprocess-per-model so each gets a clean peak-RSS measurement and there's no
fastembed singleton/model-cache contention. Collects every results/<key>.json
and prints the head-to-head table (retrieval + discrimination + cost), deltas
vs the nomic_q baseline.

Run: ./dev python3 eval/oracle_audit/emb_bench/run_stage1.py
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PY = sys.executable
WORKER = os.path.join(HERE, "bench.py")

# baseline first, then candidates (ascending size)
KEYS = ["nomic_q", "nomic_full", "bge_base", "gte_large", "arctic_l", "bge_large", "mxbai_large"]

# Interpretation ceilings on THIS (nomic+fts-generated) gold, measured by sibling
# stream 8c28df4e's oracle analysis (perfect query = the actual next-move, nomic-
# embedded): hit@5 capped at 42%, hit@25 at 68%. Reading: a model that CLEARS 42%
# hit@5 is genuinely more discriminative; one that merely matches it may just be
# saturating the gold-bias ceiling. hit@25=68% is the pool-membership ceiling —
# the "cue-far" gold (53% of misses) nomic never surfaced isn't in this gold to be
# found, so the full embedder upside needs a re-minted gold with non-nomic lenses.
ORACLE_HIT5 = 0.42
ORACLE_HIT25 = 0.68


def main():
    keys = sys.argv[1:] or KEYS
    for k in keys:
        print(f"\n──── running {k} ────", flush=True)
        r = subprocess.run([PY, WORKER, k], cwd=HERE)
        if r.returncode != 0:
            print(f"  !! {k} failed (exit {r.returncode})")

    aggs = {}
    for k in KEYS:
        p = os.path.join(RESULTS, f"{k}.json")
        if os.path.exists(p):
            aggs[k] = json.load(open(p))["agg"]
    if not aggs:
        print("no results")
        return

    base = aggs.get("nomic_q")
    print("\n" + "=" * 118)
    print("STAGE 1 — fastembed head-to-head | pure single-vector cosine on `_primary` (title+content) | 73-cue endo gold")
    print("=" * 118)
    hdr = (f"{'model':14s} {'dim':>4s} | {'hit@1':>6s} {'hit@5':>6s} {'hit@25':>6s} {'rec@5':>6s} "
           f"{'nDCG5':>6s} {'MRR':>5s} | {'sprd':>6s} {'goldZ':>6s} {'gMrgn':>7s} | {'ms/q':>5s} {'RAM':>6s} | vs.oracle")
    print(hdr)
    print("-" * 130)
    for k in KEYS:
        a = aggs.get(k)
        if not a:
            continue
        d5 = ""
        if base and k != "nomic_q":
            dd = (a["hit5"] - base["hit5"]) * 100
            d5 = f" ({dd:+.0f})"
        # vs the 42% hit@5 oracle ceiling: ★ clears it (genuinely sharper) /
        # ≈ matches (may be gold-bias-bound) / · below
        oc = "★clears" if a["hit5"] > ORACLE_HIT5 + 0.01 else ("≈ceiling" if a["hit5"] >= ORACLE_HIT5 - 0.03 else "·")
        print(f"{k:14s} {a['dim']:>4d} | {a['hit1']:>5.0%} {a['hit5']:>5.0%}{d5:7s} {a['hit25']:>5.0%} "
              f"{a['recall5']:>5.0%} {a['ndcg5']:>5.2f} {a['mrr']:>5.2f} | "
              f"{a['mean_spread_1_25']:>6.3f} {a['mean_gold_z']:>6.2f} {a['mean_gold_margin_vs_top']:>7.3f} | "
              f"{a['latency_ms']:>5.0f} {a['peak_rss_mb']:>5.0f}M | {oc}")
    print("-" * 130)
    print("sprd = mean(top1-top25) cosine spread (higher = less flat) | goldZ = best-gold cosine z-score in candidate pool")
    print("gMrgn = best-gold minus top-cosine (0 = gold is rank-1; negative = how far the model buries gold)")
    print(f"ORACLE CEILING (sibling 8c28df4e, perfect query nomic-embedded): hit@5={ORACLE_HIT5:.0%}, hit@25={ORACLE_HIT25:.0%}.")
    print("  ★clears hit@5>42% = genuinely more discriminative; ≈ceiling may just be saturating the nomic+fts gold-bias.")
    print("  hit@25 ~ pool membership; the 'cue-far' gold (53% of misses) isn't in THIS gold → full upside needs re-minted gold.")


if __name__ == "__main__":
    main()
