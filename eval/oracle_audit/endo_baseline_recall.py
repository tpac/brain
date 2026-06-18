#!/usr/bin/env python3
"""STAGE 2 — baseline retrieval on the frozen endo gold corpus.

Measures the RAW recall engine (brain.recall top-K, no Haiku) — the unmasked
engine that endo consumes — against teacher gold. Reusable: score_corpus(rank_fn)
takes any ranker so the PPR arm (Step 3) scores on the IDENTICAL corpus + metrics
(apples-to-apples A/B). Per Tom: every metric broken out by source.

Artifact guards (the three §12c traps) are ASSERTED, not assumed:
  - fatigue isolated: fresh session_id per cue; same query twice => same set.
  - cutoff PRE-applied: no returned node may post-date the cue's cutoff.
  - no truncation: over-fetch (limit=120) so the cutoff strip leaves a full pool.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_baseline_recall.py
"""
import json, os, sys, math
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
LIMIT = 120                      # over-fetch so the created_at strip can't truncate the pool
KS = (1, 5, 10, 25)            # hit@25 ~ pool-membership/candidate-gen; hit@5 ~ ranking

def load_corpus():
    return json.load(open(os.path.join(OUT, "endo_gold_corpus.json")))

def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def score_one(ranked_ids, ess, helpful):
    """Metrics for one cue. ess/helpful = gold id lists. ranked_ids = ranker output."""
    pos = {nid: i + 1 for i, nid in enumerate(ranked_ids)}     # 1-based ranks
    ess_ranks = [pos[g] for g in ess if g in pos]
    best = min(ess_ranks) if ess_ranks else None
    top5 = set(ranked_ids[:5])
    m = {
        "best_ess_rank": best,
        "mrr_ess": (1.0 / best) if best else 0.0,
        "recall5_ess": (len(top5 & set(ess)) / len(ess)) if ess else None,
    }
    for k in KS:
        topk = set(ranked_ids[:k])
        m[f"hit{k}_ess"] = 1 if (topk & set(ess)) else 0
        m[f"hit{k}_any"] = 1 if (topk & (set(ess) | set(helpful))) else 0
    rel = {**{g: 2 for g in ess}, **{g: 1 for g in helpful if g not in ess}}
    gains = [rel.get(nid, 0) for nid in ranked_ids[:5]]
    ideal = sorted(rel.values(), reverse=True)[:5]
    m["ndcg5"] = (_dcg(gains) / _dcg(ideal)) if ideal and _dcg(ideal) > 0 else 0.0
    return m

def _agg(rows, label):
    n = len(rows)
    if not n:
        return
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    print(f"  {label:24s} n={n:3d} | "
          f"hit@1 {mean('hit1_ess'):.0%}  hit@5 {mean('hit5_ess'):.0%}  hit@10 {mean('hit10_ess'):.0%}  "
          f"hit@25 {mean('hit25_ess'):.0%} | "
          f"recall@5 {mean('recall5_ess'):.0%} | nDCG@5 {mean('ndcg5'):.2f} | MRR {mean('mrr_ess'):.2f}")

def report(scored, arm):
    print(f"\n================  {arm}  ================")
    print("  (hit@k / recall@5 / nDCG@5 / MRR are on ESSENTIAL gold)")
    _agg(scored, "ALL")
    print("  -- by source --")
    by = defaultdict(list)
    for s in scored:
        by[s["source"]].append(s)
    for src in ("anchor_turn", "operator_msg"):
        _agg(by[src], src)
    print("  -- by query_type --")
    byq = defaultdict(list)
    for s in scored:
        byq[s["query_type"]].append(s)
    for qt, rows in sorted(byq.items(), key=lambda x: -len(x[1])):
        _agg(rows, qt)

def score_corpus(rank_fn, corpus, arm="arm"):
    scored = []
    for c in corpus:
        ranked = rank_fn(c)
        m = score_one(ranked, c["gold_essential"], c.get("gold_helpful", []))
        m.update(source=c["source"], query_type=c["query_type"], id=c["id"])
        scored.append(m)
    report(scored, arm)
    return scored

# ── baseline ranker: raw brain.recall (the unmasked engine endo consumes) ──
def make_baseline_ranker(brain):
    def rank(c, i=[0]):
        i[0] += 1
        res = brain.recall(query=c["query"],
                           filter={"created_at": {"lte": c["cutoff"]}},
                           limit=LIMIT, session_id=f"endo-base-{c['id']}")
        results = res.get("results", []) if isinstance(res, dict) else []
        rank._last = results
        return [r.get("id") for r in results]
    return rank

def selfchecks(brain, corpus):
    print("── artifact self-checks ──")
    c = corpus[0]
    r1 = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                      limit=LIMIT, session_id="chk-A")
    r2 = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                      limit=LIMIT, session_id="chk-B")
    ids1 = [x.get("id") for x in r1.get("results", [])]
    ids2 = [x.get("id") for x in r2.get("results", [])]
    print(f"  fatigue-isolation: two fresh-session recalls of cue[0] -> "
          f"{'IDENTICAL' if ids1 == ids2 else 'DIFFER (!)'} top-set ({len(ids1)} vs {len(ids2)})")
    # cutoff: nothing returned may post-date the cutoff
    viol = sum(1 for x in r1.get("results", []) if (x.get("created_at") or "") >= c["cutoff"])
    print(f"  cutoff-pre-applied: {viol} of {len(ids1)} returned nodes post-date cutoff (want 0)")
    print(f"  no-truncation: cue[0] returned {len(ids1)} nodes (over-fetch limit={LIMIT})")

if __name__ == "__main__":
    corpus = load_corpus()
    print(f"corpus: {len(corpus)} cues")
    with IsolatedBrain() as env:
        selfchecks(env.brain, corpus)
        score_corpus(make_baseline_ranker(env.brain), corpus, arm="BASELINE (raw brain.recall, cosine-on-cue)")
