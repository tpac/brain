#!/usr/bin/env python3
"""Surface A/B Eval — compares current vs unified formatter for Haiku selection.

Pipeline A (current): enrich_candidate_metadata() → format_candidate_for_surface() → Haiku
Pipeline B (proposed): get_rich_node() → render_rich_node(HAIKU_FORMAT) → Haiku

Measures per query:
  - Selection overlap (Jaccard similarity of selected IDs)
  - Token usage (prompt char count as proxy)
  - False positive rate (spurious selections)
  - Latency (Haiku response time)

Usage:
    python3 eval/surface_ab_eval.py
    python3 eval/surface_ab_eval.py --verbose
    python3 eval/surface_ab_eval.py --category identity
    python3 eval/surface_ab_eval.py --dry-run  # show prompts without calling Haiku
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env_path = ROOT / '.env'
if _env_path.exists():
    for line in open(_env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if v and not os.environ.get(k):
                os.environ[k] = v


# ── HAIKU_FORMAT config for pipeline B ──
HAIKU_FORMAT = {
    'content_limit': 300,
    'edge_limit': 3,
    'metadata_limit': 120,
    'time_format': 'relative',
}


# ── Test queries (reuse from judge_eval + add more) ──
from eval.judge_eval import JUDGE_QUERIES


def _build_candidates_a(brain, results, config):
    """Pipeline A: current enrichment (enrich_candidate_metadata + flat fields)."""
    from servers.scales.s1.surface_contract import enrich_candidate_metadata

    candidates = []
    content_limit = config.get('content_limit', 1000)
    for r in results:
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
            "revised_at": r.get("revised_at"),
        }
        if config.get('include_metadata'):
            enrich_candidate_metadata(brain, r.get("id", ""), node_data, config)
        candidates.append(node_data)
    return candidates


def _build_candidates_b(brain, results, config):
    """Pipeline B: get_rich_node enrichment (unified shape)."""
    from servers.pipeline_contract import get_rich_node

    candidates = []
    for r in results:
        node_id = r.get("id", "")
        rich = get_rich_node(brain, node_id)
        if not rich:
            # Fallback: use raw recall data
            rich = {
                "id": node_id,
                "type": r.get("type", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "confidence": r.get("confidence", 0),
                "locked": r.get("locked", False),
                "created_at": r.get("created_at"),
                "revised_at": r.get("revised_at"),
            }
        # Attach recall-specific fields (not in DB, only in recall results)
        rich["score"] = r.get("effective_activation", 0)
        rich["discovery"] = r.get("_discovery", "embedding")
        candidates.append(rich)
    return candidates


def _format_candidates_a(candidates):
    """Pipeline A: current format_candidate_for_surface."""
    from servers.scales.s1.surface_contract import format_candidate_for_surface
    text = ""
    for i, c in enumerate(candidates, 1):
        text += format_candidate_for_surface(c, i) + "\n\n"
    return text


def _format_candidates_b(candidates):
    """Pipeline B: render_rich_node with HAIKU_FORMAT wrapper."""
    from servers.contract import render_rich_node

    text = ""
    for i, c in enumerate(candidates, 1):
        # Thin wrapper: index + score/discovery + render_rich_node
        score_parts = []
        score = c.get('score', 0)
        if score:
            display_score = min(score, 1.0)
            score_str = "match:%.2f" % display_score
            if score > 1.0:
                score_str += ",boosted"
            score_parts.append(score_str)
        discovery = c.get('discovery', '')
        if discovery and discovery not in ('embedding', 'embedding_only', 'embedding+keyword'):
            score_parts.append("via:%s" % discovery)

        header = "#%d" % i
        if score_parts:
            header += " (%s)" % ", ".join(score_parts)

        node_text = render_rich_node(c, HAIKU_FORMAT)
        text += header + "\n" + node_text + "\n\n"
    return text


def _build_prompt(candidates_text, query, retrieval_stats=None):
    """Build surface prompt with formatted candidates. Same wrapper for both pipelines."""
    from servers.scales.s1.surface_contract import SURFACE

    cfg = SURFACE

    # Retrieval context
    retrieval_context = ""
    if retrieval_stats:
        rs = retrieval_stats
        top = rs.get('top_score', 0)
        median = rs.get('median_score', 0)
        brain_sz = rs.get('brain_size', 0)
        n_candidates = rs.get('candidates_after_floor', 0)
        retrieval_context = "Retrieval: %d candidates from %d memories. Top: %.2f, median: %.2f." % (
            n_candidates, brain_sz, top, median)
        from servers.brain_constants import RETRIEVAL_LOW_CONFIDENCE
        if top < RETRIEVAL_LOW_CONFIDENCE:
            retrieval_context += (
                "\nNOTE: Top score %.2f is low for %d memories — "
                "brain likely has nothing relevant. Prefer selecting 0." % (top, brain_sz))

    instructions = (
        "You surface relevant memories from a shared AI brain. The brain stores "
        "memories from conversations between an operator (Tom) and an AI assistant "
        "(Anchor). You decide which memories help Anchor respond to Tom's next message.\n\n"
        "Selection rules:\n"
        "- Short confirmations (\"yes\", \"ok\", \"thanks\") → select 0.\n"
        "- Word coincidence without meaning overlap → select 0.\n"
        "- Unsure? Don't select. No context > wrong context.\n\n"
        "Return ONLY JSON:\n"
        "{\"selected\":[{\"id\":\"...\",\"why\":\"one phrase\"}]}\n"
        "If nothing relevant: {\"selected\":[],\"reason\":\"brief reason\"}"
    )

    n_candidates = candidates_text.count("#")  # rough count
    prompt = """%s

Conversation:
Tom: %s

%s
Candidates:

%s""" % (
        instructions,
        query[:300],
        retrieval_context,
        candidates_text,
    )

    return prompt


def _call_haiku(prompt, max_tokens=600):
    """Call Haiku and parse selection response."""
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.time()
    api_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    latency_ms = (time.time() - t0) * 1000
    raw = api_resp.content[0].text.strip()

    # Parse
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

    input_tokens = api_resp.usage.input_tokens
    output_tokens = api_resp.usage.output_tokens

    return {
        "selected": judgment.get("selected", []),
        "reason": judgment.get("reason", ""),
        "latency_ms": round(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw": raw,
    }


def _jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_ab_query(brain, query_spec, verbose=False, dry_run=False):
    """Run a single query through both pipelines and compare."""
    from servers.scales.s1.surface_contract import SURFACE
    from servers.pipeline_contract import CANDIDATE_POOL

    query = query_spec["query"]

    # Recall (shared between A and B)
    result = brain.recall(query, limit=25)
    results = result.get("results", [])
    retrieval_stats = result.get("_retrieval_stats", {})

    if not results:
        return {"query": query, "skipped": True, "reason": "no recall results"}

    # Build candidates both ways
    candidates_a = _build_candidates_a(brain, results[:25], CANDIDATE_POOL)
    candidates_b = _build_candidates_b(brain, results[:25], CANDIDATE_POOL)

    # Format both ways
    text_a = _format_candidates_a(candidates_a)
    text_b = _format_candidates_b(candidates_b)

    # Build prompts (same wrapper)
    prompt_a = _build_prompt(text_a, query, retrieval_stats)
    prompt_b = _build_prompt(text_b, query, retrieval_stats)

    if dry_run:
        return {
            "query": query,
            "category": query_spec.get("category", "?"),
            "prompt_a_chars": len(prompt_a),
            "prompt_b_chars": len(prompt_b),
            "delta_chars": len(prompt_b) - len(prompt_a),
            "delta_pct": "%.1f%%" % ((len(prompt_b) - len(prompt_a)) / max(len(prompt_a), 1) * 100),
            "candidates": len(candidates_a),
            "dry_run": True,
        }

    # Call Haiku for both
    result_a = _call_haiku(prompt_a)
    result_b = _call_haiku(prompt_b)

    # Extract selected IDs
    ids_a = {s.get("id", "")[:8] for s in result_a["selected"]}
    ids_b = {s.get("id", "")[:8] for s in result_b["selected"]}

    # Score
    jaccard = _jaccard(ids_a, ids_b)

    # Check against expected
    expect_empty = query_spec.get("expect_empty", False)
    fragments = query_spec.get("expected_title_fragments", [])

    def _check_pass(selected, candidates):
        if expect_empty:
            return len(selected) == 0
        if fragments:
            selected_titles = []
            for s in selected:
                sid = s.get("id", "")[:8]
                for c in candidates:
                    if str(c["id"])[:8] == sid:
                        selected_titles.append(c.get("title", "").lower())
                        break
            return any(any(f.lower() in t for f in fragments) for t in selected_titles)
        return len(selected) > 0

    pass_a = _check_pass(result_a["selected"], candidates_a)
    pass_b = _check_pass(result_b["selected"], candidates_b)

    out = {
        "query": query,
        "category": query_spec.get("category", "?"),
        "description": query_spec.get("description", ""),
        "jaccard": jaccard,
        "pass_a": pass_a,
        "pass_b": pass_b,
        "count_a": len(ids_a),
        "count_b": len(ids_b),
        "ids_a": sorted(ids_a),
        "ids_b": sorted(ids_b),
        "tokens_a": result_a["input_tokens"],
        "tokens_b": result_b["input_tokens"],
        "token_delta": result_b["input_tokens"] - result_a["input_tokens"],
        "latency_a": result_a["latency_ms"],
        "latency_b": result_b["latency_ms"],
        "candidates": len(candidates_a),
    }

    if verbose:
        out["selected_a"] = result_a["selected"]
        out["selected_b"] = result_b["selected"]
        out["reason_a"] = result_a["reason"]
        out["reason_b"] = result_b["reason"]

    return out


def main():
    parser = argparse.ArgumentParser(description="Surface A/B Eval — compare current vs unified format")
    parser.add_argument("--category", help="Filter to specific category")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt sizes without calling Haiku")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY not set. Use --dry-run to skip API calls.")
        sys.exit(1)

    from tests.isolated_brain import IsolatedBrain

    queries = JUDGE_QUERIES
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]

    print("=" * 80)
    print("SURFACE A/B EVAL — Current (A) vs Unified (B) Format")
    print("  A: enrich_candidate_metadata → format_candidate_for_surface")
    print("  B: get_rich_node → render_rich_node(HAIKU_FORMAT)")
    if args.dry_run:
        print("  MODE: dry run (no Haiku calls)")
    print("=" * 80)
    print()

    with IsolatedBrain() as env:
        results = []
        for q in queries:
            r = run_ab_query(env.brain, q, verbose=args.verbose, dry_run=args.dry_run)
            results.append(r)

            if r.get("skipped"):
                print("  SKIP [%s] %s — %s" % (q.get("category", "?"), q["query"], r["reason"]))
                continue

            if args.dry_run:
                print("  [%s] %s" % (r["category"], r["query"]))
                print("    A: %d chars | B: %d chars | delta: %s (%s)" % (
                    r["prompt_a_chars"], r["prompt_b_chars"],
                    r["delta_chars"], r["delta_pct"]))
                print()
                continue

            # Selection comparison
            match_icon = "=" if r["jaccard"] == 1.0 else ("≈" if r["jaccard"] >= 0.5 else "≠")
            pass_icon_a = "✓" if r["pass_a"] else "✗"
            pass_icon_b = "✓" if r["pass_b"] else "✗"

            print("  [%s] %s" % (r["category"], r["query"]))
            print("    A: %s %d selected, %d tokens, %dms" % (
                pass_icon_a, r["count_a"], r["tokens_a"], r["latency_a"]))
            print("    B: %s %d selected, %d tokens, %dms" % (
                pass_icon_b, r["count_b"], r["tokens_b"], r["latency_b"]))
            print("    %s Jaccard: %.2f | Token Δ: %+d | IDs A: %s B: %s" % (
                match_icon, r["jaccard"], r["token_delta"],
                r["ids_a"], r["ids_b"]))

            if args.verbose:
                if r.get("selected_a"):
                    for s in r["selected_a"]:
                        print("      A → %s: %s" % (s.get("id", "?")[:8], s.get("why", "")))
                if r.get("selected_b"):
                    for s in r["selected_b"]:
                        print("      B → %s: %s" % (s.get("id", "?")[:8], s.get("why", "")))
            print()

    if args.dry_run:
        # Dry run summary
        valid = [r for r in results if not r.get("skipped")]
        if valid:
            avg_delta = sum(r["delta_chars"] for r in valid) / len(valid)
            avg_pct = sum(
                (r["prompt_b_chars"] - r["prompt_a_chars"]) / max(r["prompt_a_chars"], 1) * 100
                for r in valid) / len(valid)
            print("=" * 80)
            print("  %d queries | Avg char delta: %+.0f | Avg pct delta: %+.1f%%" % (
                len(valid), avg_delta, avg_pct))
        return

    # Full summary
    valid = [r for r in results if not r.get("skipped") and not r.get("dry_run")]
    if not valid:
        print("No valid results.")
        return

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Pass rates
    pass_a = sum(1 for r in valid if r["pass_a"])
    pass_b = sum(1 for r in valid if r["pass_b"])
    print("  Pass rate A: %d/%d (%.0f%%)" % (pass_a, len(valid), pass_a / len(valid) * 100))
    print("  Pass rate B: %d/%d (%.0f%%)" % (pass_b, len(valid), pass_b / len(valid) * 100))

    # Jaccard
    avg_jaccard = sum(r["jaccard"] for r in valid) / len(valid)
    exact_match = sum(1 for r in valid if r["jaccard"] == 1.0)
    print("  Avg Jaccard: %.2f | Exact match: %d/%d" % (avg_jaccard, exact_match, len(valid)))

    # Tokens
    avg_tokens_a = sum(r["tokens_a"] for r in valid) / len(valid)
    avg_tokens_b = sum(r["tokens_b"] for r in valid) / len(valid)
    avg_delta = avg_tokens_b - avg_tokens_a
    print("  Avg tokens A: %.0f | B: %.0f | Δ: %+.0f (%+.1f%%)" % (
        avg_tokens_a, avg_tokens_b, avg_delta,
        avg_delta / max(avg_tokens_a, 1) * 100))

    # Latency
    avg_lat_a = sum(r["latency_a"] for r in valid) / len(valid)
    avg_lat_b = sum(r["latency_b"] for r in valid) / len(valid)
    print("  Avg latency A: %dms | B: %dms" % (avg_lat_a, avg_lat_b))

    # Regressions
    regressions = [r for r in valid if r["pass_a"] and not r["pass_b"]]
    improvements = [r for r in valid if not r["pass_a"] and r["pass_b"]]
    if regressions:
        print("\n  ⚠ REGRESSIONS (A passed, B failed):")
        for r in regressions:
            print("    [%s] %s" % (r["category"], r["query"]))
    if improvements:
        print("\n  ✓ IMPROVEMENTS (A failed, B passed):")
        for r in improvements:
            print("    [%s] %s" % (r["category"], r["query"]))

    # Save results
    results_path = ROOT / 'eval' / 'results' / 'surface_ab_latest.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "queries": len(valid),
            "pass_a": pass_a, "pass_b": pass_b,
            "avg_jaccard": round(avg_jaccard, 3),
            "avg_tokens_a": round(avg_tokens_a),
            "avg_tokens_b": round(avg_tokens_b),
            "results": valid,
        }, f, indent=2, default=str)
    print("\n  Results saved: %s" % results_path)


if __name__ == "__main__":
    main()
