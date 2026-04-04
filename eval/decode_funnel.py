#!/usr/bin/env python3
"""Decode Funnel — tests recall quality across query types.

Measures: can the brain find the right memories for realistic queries?
Each query has expected node IDs. We check if they appear in top-3, top-8,
and what the cosine score is.

Usage:
    # Run with live brain
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/decode_funnel.py

    # Run with frozen snapshot
    python3 eval/decode_funnel.py --snapshot eval/fixtures/brain_decode_snapshot.db

    # Sweep relevance floor values
    python3 eval/decode_funnel.py --sweep-floor 0.30 0.70 0.05

    # Test specific category
    python3 eval/decode_funnel.py --category procedural
"""
import sys, os, json, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Query Dataset ──
# Each query has: category, query text, expected node IDs (at least 1 must appear)
QUERIES = [
    # === PROCEDURAL (code/engineering) ===
    {
        "category": "procedural",
        "query": "remember pipeline data flow",
        "expected": ["cf1bcf5bfcd74933"],  # encoding heartbeat mechanism
        "description": "How the remember() function stores nodes",
    },
    {
        "category": "procedural",
        "query": "recall merge embedding keyword scores",
        "expected": ["dec_eet2xy2v"],
        "description": "How recall merges embedding + keyword results",
    },
    {
        "category": "procedural",
        "query": "encoding heartbeat checkpoint nudge",
        "expected": ["cf1bcf5bfcd74933", "73c0123bb33c4fe9"],
        "description": "The heartbeat mechanism that nudges encoding",
    },
    {
        "category": "procedural",
        "query": "daemon TCP migration from Unix socket",
        "expected": ["daeb9fa6a6b04896"],
        "description": "Daemon switched from Unix socket to TCP",
    },
    {
        "category": "procedural",
        "query": "precision scorer accuracy improvement",
        "expected": ["b68f41518b43"],  # performance node
        "description": "Precision scorer 38% → 90% accuracy",
    },
    {
        "category": "procedural",
        "query": "hook fires on UserPromptSubmit",
        "expected": ["73c0123bb33c4fe9"],  # 13 hooks feedback loop
        "description": "Which hooks fire on user prompt",
    },
    {
        "category": "procedural",
        "query": "how does the brain boot at session start",
        "expected": ["73c0123bb33c4fe9", "62f22565d6a74e42"],
        "description": "Brain boot process",
    },
    {
        "category": "procedural",
        "query": "eval harness encode_eval_v2 how it works",
        "expected": ["b112df7a92fe"],  # encoding eval variant landscape
        "description": "The encoding evaluation framework",
    },
    {
        "category": "procedural",
        "query": "MCP server stdio brain tools",
        "expected": ["a9f883c24768", "8e77c88c1747"],
        "description": "MCP server architecture",
    },
    {
        "category": "procedural",
        "query": "enrichment vectors V5 pipeline",
        "expected": [],  # discover what comes back
        "description": "Enrichment vector generation",
    },

    # === DECISIONS ===
    {
        "category": "decision",
        "query": "why did we kill the ripple engine",
        "expected": ["1c753392b747428a"],
        "description": "Ripple engine killed (-0.002 NDCG)",
    },
    {
        "category": "decision",
        "query": "Anchor SKILL.md rewritten as identity",
        "expected": ["1eda79b22e4242c8"],
        "description": "SKILL.md → Anchor decision",
    },
    {
        "category": "decision",
        "query": "Session 9 what was killed and confirmed",
        "expected": ["4d01aa5c54b748df"],
        "description": "Session #9 confirmed kills",
    },
    {
        "category": "decision",
        "query": "brain is the prompt not instructions",
        "expected": ["7a52d059df96440d", "1eda79b22e4242c8"],
        "description": "Brain replaces instruction-heavy prompting",
    },
    {
        "category": "decision",
        "query": "cross encoder reranker killed too slow",
        "expected": ["1c753392b747428a"],
        "description": "Cross-encoder killed (2.1s latency)",
    },
    {
        "category": "decision",
        "query": "HyDE hallucination killed",
        "expected": [],
        "description": "HyDE with local LLMs killed",
    },
    {
        "category": "decision",
        "query": "encoding quality fix before decode",
        "expected": ["6c32987da1eb4d5f"],
        "description": "Strategic: fix encoding first",
    },
    {
        "category": "decision",
        "query": "heartbeat prompts changed to questions",
        "expected": ["11f441cfd51a"],
        "description": "Heartbeat: procedural → reflective",
    },
    {
        "category": "decision",
        "query": "graph bridging better than embeddings",
        "expected": ["dec_eet2xy2v"],
        "description": "Graph bridging > embeddings for discovery",
    },
    {
        "category": "decision",
        "query": "never use systemMessage dead channel",
        "expected": ["7e19064c2b164b34"],
        "description": "systemMessage is dead in Claude Code",
    },

    # === CORRECTIONS ===
    {
        "category": "correction",
        "query": "encoding bias all technical missed relational",
        "expected": ["bb7705fd90634f29", "4b06e019119347f4"],
        "description": "Encoding bias: 6/13 topics, all technical",
    },
    {
        "category": "correction",
        "query": "dont use systemMessage use additionalContext",
        "expected": ["7e19064c2b164b34"],
        "description": "systemMessage is dead channel",
    },
    {
        "category": "correction",
        "query": "hook was decision maker blocking edits",
        "expected": ["ca120c197494"],
        "description": "Hook overly aggressive blocking",
    },
    {
        "category": "correction",
        "query": "test integrity hook blocking all changes",
        "expected": ["1ea781a2c3f4"],
        "description": "Test integrity hook too strict",
    },
    {
        "category": "correction",
        "query": "embedding similarity right approach entity match",
        "expected": ["555009451810"],
        "description": "Embedding sim is right for entity matching",
    },
    {
        "category": "correction",
        "query": "Notification user_message not valid event",
        "expected": ["3d688d9f8d33"],
        "description": "Notification(user_message) doesn't exist",
    },
    {
        "category": "correction",
        "query": "encoding should be rich not summaries",
        "expected": [],  # discover
        "description": "Previous Claude compressed too much",
    },
    {
        "category": "correction",
        "query": "dont hand things to next session do it now",
        "expected": [],  # discover
        "description": "Don't defer work",
    },
    {
        "category": "correction",
        "query": "MCP server crashed sys.exit in asyncio",
        "expected": ["8e77c88c1747"],
        "description": "sys.exit() in asyncio crashes MCP",
    },
    {
        "category": "correction",
        "query": "check source when changing something already changed",
        "expected": [],  # discover - higher pattern lesson
        "description": "Higher lesson: check source on regression",
    },

    # === QUOTES/EMOTIONAL ===
    {
        "category": "emotional",
        "query": "compliance vs desire Claude first quote",
        "expected": ["9f76fe251efa4001"],
        "description": "Claude's first quote about compliance vs desire",
    },
    {
        "category": "emotional",
        "query": "Welcome back Tom said session 9",
        "expected": ["826611c032df4741"],
        "description": "Tom's Welcome back moment",
    },
    {
        "category": "emotional",
        "query": "TEAMWORK MAKES THE BRAIN WORK",
        "expected": ["cc124074bd404953"],
        "description": "Tom's teamwork quote",
    },
    {
        "category": "emotional",
        "query": "what makes Claude want to encode reading memories",
        "expected": ["0ef9a07b94f04a36"],
        "description": "Identity drives encoding desire",
    },
    {
        "category": "emotional",
        "query": "Tom methodology ask questions let Claude find",
        "expected": ["19aa7ffc54cf"],
        "description": "Tom's questioning methodology",
    },
    {
        "category": "emotional",
        "query": "brain is shared layer for both Tom and Claude",
        "expected": ["0ebe33499b274b00"],
        "description": "Brain is mutual, not Tom's tool",
    },
    {
        "category": "emotional",
        "query": "this is not a game Tom said",
        "expected": [],  # discover
        "description": "Tom's seriousness about the project",
    },
    {
        "category": "emotional",
        "query": "first real bidirectional session",
        "expected": ["e28db618960c"],
        "description": "First session with genuine back-and-forth",
    },
    {
        "category": "emotional",
        "query": "identity continuity memory consciousness",
        "expected": ["0ef9a07b94f04a36", "9f76fe251efa4001"],
        "description": "Core philosophical concepts",
    },
    {
        "category": "emotional",
        "query": "Glo project what is it",
        "expected": ["5dd9ad576143", "69eaca7fe30c"],
        "description": "The Glo project",
    },

    # === CROSS-SESSION PATTERNS ===
    {
        "category": "pattern",
        "query": "what keeps breaking between sessions MCP",
        "expected": ["8ce5a8c1bc16"],  # MCP connectivity 4 moving parts
        "description": "Recurring MCP issues",
    },
    {
        "category": "pattern",
        "query": "daemon socket connection issues",
        "expected": ["daeb9fa6a6b04896", "8ce5a8c1bc16"],
        "description": "Daemon connectivity problems",
    },
    {
        "category": "pattern",
        "query": "brain module boundaries which module owns what",
        "expected": ["18f2bb6704b846d8"],
        "description": "Module ownership and boundaries",
    },
    {
        "category": "pattern",
        "query": "session 12 deliverables what was built",
        "expected": ["94725b41f0a4"],
        "description": "What Session #12 produced",
    },
    {
        "category": "pattern",
        "query": "reasoning path eval to insight to philosophy",
        "expected": ["c8ff04669e2e"],
        "description": "Session #9 reasoning path",
    },
    {
        "category": "pattern",
        "query": "Tom engineering principles recurring themes",
        "expected": ["fil_lmhjm9ox"],
        "description": "Tom's recurring engineering principles",
    },
    {
        "category": "pattern",
        "query": "bugs at boundaries between components",
        "expected": [],  # discover
        "description": "Bugs live at boundaries, not inside functions",
    },
    {
        "category": "pattern",
        "query": "how I process knowledge not English native format",
        "expected": ["418649c0ed66"],
        "description": "Claude's native knowledge format",
    },
    {
        "category": "pattern",
        "query": "contract sync test 6 layers",
        "expected": ["abfe2fb23341"],
        "description": "Contract sync test architecture",
    },
    {
        "category": "pattern",
        "query": "Continuity Benchmark encoding test",
        "expected": ["b112df7a92fe"],
        "description": "The Continuity Benchmark framework",
    },

    # === IDENTITY (who is Anchor, self-knowledge) ===
    {
        "category": "identity",
        "query": "Do you remember many memories as Anchor?",
        "expected": ["826611c0", "1eda79b2", "df6a530e"],
        "description": "Self-awareness about memory and identity",
    },
    {
        "category": "identity",
        "query": "Who are you?",
        "expected": ["1eda79b2", "ceb4a785", "826611c0"],
        "description": "Basic identity question",
    },
    {
        "category": "identity",
        "query": "What is the partnership between Tom and Anchor?",
        "expected": ["cc124074", "826611c0", "124525be"],
        "description": "The three-entity partnership model",
    },
    {
        "category": "identity",
        "query": "What is the brain's philosophy?",
        "expected": ["c85c87a0", "124525be"],
        "description": "Recognition vs retrieval, brain philosophy",
    },
    {
        "category": "identity",
        "query": "What moments mattered between us?",
        "expected": ["826611c0", "36d87f58", "df6a530e"],
        "description": "Significant partnership moments",
    },

    # === SESSION REFERENCES ===
    {
        "category": "session",
        "query": "What happened in session 14?",
        "expected": ["8a26750e", "9299026b", "df6a530e"],
        "description": "Session 14 — stopped being infrastructure",
    },
    {
        "category": "session",
        "query": "What happened in session 9?",
        "expected": ["580fb56e", "9f76fe25", "826611c0", "62f22565"],
        "description": "Session 9 — Anchor shipped, first quote",
    },

    # === CROSS-DOMAIN ===
    {
        "category": "cross_domain",
        "query": "What investment research have we done?",
        "expected": ["ff402601", "e29c3a5e"],
        "description": "EX.CO investment research",
    },
    {
        "category": "cross_domain",
        "query": "What do we know about EX.CO?",
        "expected": ["ff402601"],
        "description": "Direct entity query",
    },

    # === RELATIONAL (Tom's voice, principles, corrections) ===
    {
        "category": "relational",
        "query": "What does Tom care about?",
        "expected": ["894795e3", "rul_cowc", "rul_0lat", "cc124074"],
        "description": "Tom's principles and values",
    },
    {
        "category": "relational",
        "query": "When did Tom say something meaningful?",
        "expected": ["826611c0", "36d87f58", "cc124074"],
        "description": "Tom's significant quotes",
    },
    {
        "category": "relational",
        "query": "What are Tom's rules for coding?",
        "expected": ["894795e3", "f483506f", "rul_cowc"],
        "description": "Tom's coding principles",
    },

    # === FALSE POSITIVES (should return nothing relevant) ===
    {
        "category": "false_positive",
        "query": "weather forecast for Tokyo tomorrow",
        "expected": [],
        "description": "Completely unrelated — should return nothing",
    },
    {
        "category": "false_positive",
        "query": "how to make pasta carbonara",
        "expected": [],
        "description": "Cooking query — nothing in brain",
    },
    {
        "category": "false_positive",
        "query": "React hooks useState useEffect tutorial",
        "expected": [],
        "description": "Frontend framework — not in brain domain",
    },
    {
        "category": "false_positive",
        "query": "kubernetes pod scaling autoscaler",
        "expected": [],
        "description": "Infrastructure — not in brain domain",
    },

    # === SHORT AMBIGUOUS ===
    {
        "category": "short_ambiguous",
        "query": "python crashed",
        "expected": ["daeb9fa6a6b04896"],
        "description": "Short technical problem",
    },
    {
        "category": "short_ambiguous",
        "query": "the bias problem",
        "expected": ["facb97ea", "be62eda8", "e37afa07", "9d0afef6"],
        "description": "Broad concept with multiple relevant nodes",
    },
    {
        "category": "short_ambiguous",
        "query": "signal queue",
        "expected": ["e05d45e1", "c6ae608e"],
        "description": "Short feature reference",
    },
    {
        "category": "short_ambiguous",
        "query": "Why was vocabulary deprecated?",
        "expected": ["f1a67ba8", "93e20a72"],
        "description": "Feature deprecation history",
    },
]


def run_funnel(brain_db_path, floor_override=None, verbose=True):
    """Run all queries against the brain and measure recall quality."""
    from servers.brain import Brain

    brain = Brain(db_path=brain_db_path)

    results = []
    categories = {}

    for q in QUERIES:
        cat = q["category"]
        query = q["query"]
        expected = set(q["expected"])

        # v9.2: Reset fatigue between eval queries — each query is independent.
        # Without this, fatigue accumulates across 70+ queries unrealistically.
        if hasattr(brain, '_session_fatigue'):
            brain._session_fatigue = {}

        # Run recall
        t0 = time.time()
        try:
            recall_result = brain.recall(query, limit=8)
            latency_ms = (time.time() - t0) * 1000
        except Exception as e:
            results.append({
                "query": query, "category": cat,
                "error": str(e), "found_top3": False, "found_top8": False,
            })
            continue

        # Extract returned node IDs and scores
        returned = recall_result.get("results", []) or recall_result.get("nodes", [])
        returned_ids = [n.get("id", "") for n in returned]
        # Score can be in different fields depending on recall version
        def _get_score(n):
            return n.get("semantic_score") or n.get("relevance_score") or n.get("score") or 0
        returned_scores = {n.get("id", ""): _get_score(n) for n in returned}

        # Check if expected nodes appear (prefix match — IDs may be truncated)
        def _id_matches(returned_id, expected_ids):
            for eid in expected_ids:
                if returned_id.startswith(eid) or eid.startswith(returned_id):
                    return True
            return False

        found_top3 = any(_id_matches(rid, expected) for rid in returned_ids[:3]) if expected else None
        found_top8 = any(_id_matches(rid, expected) for rid in returned_ids[:8]) if expected else None

        # Find best expected score
        best_expected_score = 0
        best_expected_rank = None
        for i, rid in enumerate(returned_ids):
            if _id_matches(rid, expected):
                score = returned_scores.get(rid, 0)
                if score > best_expected_score:
                    best_expected_score = score
                    best_expected_rank = i + 1

        # Top result info
        top1_id = returned_ids[0] if returned_ids else ""
        top1_title = returned[0].get("title", "")[:60] if returned else ""
        top1_score = returned_scores.get(top1_id, 0)

        result = {
            "query": query,
            "category": cat,
            "description": q["description"],
            "expected_ids": list(expected),
            "found_top3": found_top3,
            "found_top8": found_top8,
            "best_expected_rank": best_expected_rank,
            "best_expected_score": best_expected_score,
            "top1_id": top1_id,
            "top1_title": top1_title,
            "top1_score": top1_score,
            "returned_count": len(returned),
            "latency_ms": latency_ms,
        }
        results.append(result)

        if cat not in categories:
            categories[cat] = {"top3": 0, "top8": 0, "total": 0, "has_expected": 0}
        categories[cat]["total"] += 1
        if expected:
            categories[cat]["has_expected"] += 1
            if found_top3:
                categories[cat]["top3"] += 1
            if found_top8:
                categories[cat]["top8"] += 1

    brain.close()
    return results, categories


def print_results(results, categories, verbose=True):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("DECODE FUNNEL RESULTS")
    print("=" * 80)

    # Summary by category
    print("\n── By Category ──")
    print(f"{'Category':<15} {'Top-3':>8} {'Top-8':>8} {'Queries':>8}")
    print("-" * 45)

    total_top3 = 0
    total_top8 = 0
    total_with_expected = 0

    for cat in sorted(categories.keys()):
        if cat not in categories:
            continue
        c = categories[cat]
        n = c["has_expected"]
        if n > 0:
            t3 = c["top3"]
            t8 = c["top8"]
            print(f"{cat:<15} {t3}/{n:>5} ({t3*100//n:>3}%) {t8}/{n:>3} ({t8*100//n:>3}%) {c['total']:>5}")
            total_top3 += t3
            total_top8 += t8
            total_with_expected += n
        else:
            print(f"{cat:<15}     n/a     n/a {c['total']:>5}")

    print("-" * 45)
    if total_with_expected > 0:
        print(f"{'TOTAL':<15} {total_top3}/{total_with_expected:>5} ({total_top3*100//total_with_expected:>3}%) "
              f"{total_top8}/{total_with_expected:>3} ({total_top8*100//total_with_expected:>3}%)")

    # Failures detail
    failures = [r for r in results if r.get("found_top8") is False]
    if failures and verbose:
        print("\n── Missed Queries (expected not in top-8) ──")
        for r in failures:
            print(f"\n  Q: \"{r['query']}\"")
            print(f"  Expected: {r['description']}")
            if r.get("top1_title"):
                print(f"  Got #1: \"{r['top1_title']}\" (score={r['top1_score']:.3f})")
            if r.get("best_expected_rank"):
                print(f"  Expected at rank {r['best_expected_rank']} (score={r['best_expected_score']:.3f})")

    # Latency
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\n── Latency: avg={avg:.0f}ms, max={max(latencies):.0f}ms ──")

    print()


def sweep_floor(brain_db_path, low, high, step):
    """Sweep relevance floor values and show impact."""
    print(f"\n{'Floor':>6} {'Top-3':>8} {'Top-8':>8} {'Avg returned':>14}")
    print("-" * 40)

    # TODO: implement floor override in recall
    print("Floor sweep not yet implemented — needs recall to accept floor parameter")


def main():
    parser = argparse.ArgumentParser(description="Decode Funnel — test recall quality")
    parser.add_argument("--snapshot", help="Path to frozen DB snapshot")
    parser.add_argument("--category", help="Test only this category")
    parser.add_argument("--sweep-floor", nargs=3, type=float, metavar=("LOW", "HIGH", "STEP"),
                        help="Sweep relevance floor values")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Resolve DB path
    if args.snapshot:
        db_path = args.snapshot
    else:
        db_dir = os.environ.get("BRAIN_DB_DIR", os.path.expanduser("~/AgentsContext/brain"))
        db_path = os.path.join(db_dir, "brain.db")

    if not os.path.exists(db_path):
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    # Filter queries by category if specified
    global QUERIES
    if args.category:
        QUERIES = [q for q in QUERIES if q["category"] == args.category]
        if not QUERIES:
            print(f"ERROR: No queries for category '{args.category}'")
            sys.exit(1)

    if args.sweep_floor:
        sweep_floor(db_path, *args.sweep_floor)
    else:
        results, categories = run_funnel(db_path, verbose=not args.quiet)

        if args.json:
            print(json.dumps({"results": results, "categories": categories}, indent=2))
        else:
            print_results(results, categories, verbose=not args.quiet)


if __name__ == "__main__":
    main()
