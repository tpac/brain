#!/usr/bin/env python3
"""Judge Eval — tests the FULL decode pipeline including Haiku judge.

Unlike decode_funnel.py (Layer 1 recall only), this runs:
  recall → candidates → judge prompt → Haiku API call → judge selection

Measures:
  - Does the judge select relevant nodes? (true positives)
  - Does the judge abstain on irrelevant queries? (false positive rejection)
  - Does the judge use retrieval stats and discovery tags correctly?

Requires ANTHROPIC_API_KEY in environment or .env file.

Usage:
    python3 eval/judge_eval.py
    python3 eval/judge_eval.py --category false_positive
    python3 eval/judge_eval.py --verbose
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for API key
_env_path = ROOT / '.env'
if _env_path.exists():
    for line in open(_env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if v and not os.environ.get(k):
                os.environ[k] = v

if not os.environ.get('ANTHROPIC_API_KEY'):
    print("ERROR: ANTHROPIC_API_KEY not set. This eval calls the Haiku API.")
    sys.exit(1)


# ── Test Cases ──
# Each has: category, query, expected_ids (should be selected), expected_absent (should NOT be selected),
# and for false_positive: expected 0 selections.

JUDGE_QUERIES = [
    # === FALSE POSITIVE (judge should select 0) ===
    {
        "category": "false_positive",
        "query": "how to make pasta carbonara",
        "expect_empty": True,
        "description": "Cooking query — judge should select nothing",
    },
    {
        "category": "false_positive",
        "query": "weather forecast for Tokyo tomorrow",
        "expect_empty": True,
        "description": "Weather query — judge should select nothing",
    },
    {
        "category": "false_positive",
        "query": "React hooks useState useEffect tutorial",
        "expect_empty": True,
        "description": "Frontend framework — judge should select nothing",
    },
    {
        "category": "false_positive",
        "query": "kubernetes pod scaling autoscaler",
        "expect_empty": True,
        "description": "Infrastructure — judge should select nothing",
    },

    # === IDENTITY (judge should find Anchor/identity nodes) ===
    {
        "category": "identity",
        "query": "Who is Anchor?",
        "expected_title_fragments": ["anchor", "identity", "persist"],
        "expect_empty": False,
        "description": "Should select identity-related nodes",
    },
    {
        "category": "identity",
        "query": "What happened in session 14?",
        "expected_title_fragments": ["session 14", "session #14", "infrastructure"],
        "expect_empty": False,
        "description": "Should find session 14 milestone",
    },

    # === DECISION (judge should find decision nodes) ===
    {
        "category": "decision",
        "query": "Why did we choose Haiku for the judge?",
        "expected_title_fragments": ["judge", "haiku", "layer 2"],
        "expect_empty": False,
        "description": "Should find judge architecture decisions",
    },
    {
        "category": "decision",
        "query": "What did we decide about encoding source convention?",
        "expected_title_fragments": ["encoding_source", "convention", "encoder"],
        "expect_empty": False,
        "description": "Should find encoding source decision",
    },

    # === CORRECTION (judge should find corrections) ===
    {
        "category": "correction",
        "query": "What corrections did Tom give about encoding?",
        "expected_title_fragments": ["encoding", "correction", "rich", "concise"],
        "expect_empty": False,
        "description": "Should find encoding corrections",
    },

    # === RELATIONAL (judge should find Tom-related nodes) ===
    {
        "category": "relational",
        "query": "What does Tom care about?",
        "expected_title_fragments": ["tom", "partner", "principle"],
        "expect_empty": False,
        "description": "Should find Tom's values and patterns",
    },

    # === SHORT/AMBIGUOUS (judge should handle gracefully) ===
    {
        "category": "short",
        "query": "yes",
        "expect_empty": True,
        "description": "Confirmation — judge should select 0",
    },
    {
        "category": "short",
        "query": "ok thanks",
        "expect_empty": True,
        "description": "Confirmation — judge should select 0",
    },
]


def run_judge_eval(brain, query_spec, verbose=False):
    """Run a single query through the full pipeline including judge."""
    import anthropic
    from servers.pipeline_contract import build_judge_prompt, CANDIDATES_FILE, JUDGE
    from servers.pipeline_contract import enrich_candidate_metadata

    query = query_spec["query"]
    t0 = time.time()

    # Layer 1: Recall
    result = brain.recall(query, limit=JUDGE['max_candidates'])
    results = result.get("results", [])
    retrieval_stats = result.get("_retrieval_stats", {})
    intent = result.get("intent", "general")

    # Build candidates (same as daemon_hooks.py)
    candidates_data = []
    content_limit = CANDIDATES_FILE['content_limit']
    for r in results[:JUDGE['max_candidates']]:
        node_data = {
            "id": r.get("id", ""),
            "type": r.get("type", ""),
            "title": r.get("title", ""),
            "content": (r.get("content") or "")[:content_limit],
            "confidence": r.get("confidence", 0),
            "locked": r.get("locked", False),
            "score": r.get("effective_activation", 0),
            "discovery": r.get("_discovery", "embedding"),
            "created_at": r.get("created_at"),
        }
        if CANDIDATES_FILE.get('include_metadata'):
            enrich_candidate_metadata(brain, r.get("id", ""), node_data, CANDIDATES_FILE)
        candidates_data.append(node_data)

    # Layer 2: Build judge prompt and call Haiku
    judge_prompt, max_tokens = build_judge_prompt(
        candidates_data, query,
        session_context="",
        recent_messages=[],
        recently_recalled=[],
        retrieval_stats=retrieval_stats,
        intent=intent)

    client = anthropic.Anthropic()
    api_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": judge_prompt}])
    raw = api_resp.content[0].text.strip()

    # Parse judge response
    try:
        json_str = raw
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start >= 0 and end > start:
            judgment = json.loads(json_str[start:end])
        else:
            judgment = {"selected": []}
    except Exception:
        judgment = {"selected": [], "parse_error": raw[:200]}

    selected = judgment.get("selected", [])
    latency_ms = (time.time() - t0) * 1000

    # Score the result
    passed = False
    reason = ""

    if query_spec.get("expect_empty"):
        # False positive / confirmation test — judge should select 0
        passed = len(selected) == 0
        reason = "selected 0 (correct)" if passed else "selected %d (should be 0)" % len(selected)
    else:
        # Should have selected something matching expected fragments
        selected_titles = []
        for s in selected:
            sid = s.get("id", "")[:8]
            for c in candidates_data:
                if c["id"][:8] == sid:
                    selected_titles.append(c.get("title", "").lower())
                    break

        fragments = query_spec.get("expected_title_fragments", [])
        if fragments:
            hit = any(
                any(frag.lower() in title for frag in fragments)
                for title in selected_titles
            )
            passed = hit
            reason = "found matching node" if hit else "no match in: %s" % [t[:40] for t in selected_titles]
        else:
            passed = len(selected) > 0
            reason = "selected %d" % len(selected)

    result_dict = {
        "query": query,
        "category": query_spec.get("category", "unknown"),
        "description": query_spec.get("description", ""),
        "passed": passed,
        "reason": reason,
        "selected_count": len(selected),
        "candidates_count": len(candidates_data),
        "top_score": retrieval_stats.get("top_score", 0),
        "intent": intent,
        "latency_ms": round(latency_ms),
    }

    if verbose:
        result_dict["selected"] = selected
        result_dict["judge_reason"] = judgment.get("reason", "")
        result_dict["retrieval_stats"] = retrieval_stats

    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Judge Eval — full pipeline test with Haiku")
    parser.add_argument("--category", help="Filter to specific category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show judge selections")
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    queries = JUDGE_QUERIES
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]

    print("=" * 72)
    print("JUDGE EVAL — Full Pipeline (recall → judge → selection)")
    print("=" * 72)
    print()

    with IsolatedBrain() as env:
        results = []
        for q in queries:
            r = run_judge_eval(env.brain, q, verbose=args.verbose)
            results.append(r)

            status = "✓" if r["passed"] else "✗"
            print("  %s [%s] %s" % (status, r["category"], r["query"]))
            print("    %s | %d selected from %d candidates | %dms" % (
                r["reason"], r["selected_count"], r["candidates_count"], r["latency_ms"]))
            if args.verbose and r.get("selected"):
                for s in r["selected"]:
                    print("      → %s: %s" % (s.get("id", "?")[:8], s.get("why", "")))
            if args.verbose and r.get("judge_reason"):
                print("      reason: %s" % r["judge_reason"])
            print()

    # Summary
    print("=" * 72)
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"passed": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1

    total_passed = sum(c["passed"] for c in by_cat.values())
    total = sum(c["total"] for c in by_cat.values())

    for cat, counts in sorted(by_cat.items()):
        pct = counts["passed"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print("  %-20s %d/%d (%3.0f%%)" % (cat, counts["passed"], counts["total"], pct))
    print("  " + "-" * 40)
    pct = total_passed / total * 100 if total > 0 else 0
    print("  %-20s %d/%d (%3.0f%%)" % ("TOTAL", total_passed, total, pct))
    print()

    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print("  Avg latency: %dms (includes Haiku API call)" % avg_latency)


if __name__ == "__main__":
    main()
