#!/usr/bin/env python3
"""Recall Evaluation Framework — measures recall quality across multiple dimensions.

Built from real data, not synthetic queries. Evaluates:
  - Raw recall (can brain.recall() find the right nodes?)
  - Cross-domain recall (brain-dev vs EX.CO vs identity)
  - Score distribution (signal-noise separation)
  - Source diversity (which retrieval path finds what?)
  - Encoding findability (can nodes be found via their situation field?)
  - Latency

Usage:
    # Full eval against live brain
    python3 eval/recall_eval.py

    # Specific category
    python3 eval/recall_eval.py --category exco

    # JSON output for tracking
    python3 eval/recall_eval.py --json

    # Save report
    python3 eval/recall_eval.py --save

    # Against frozen snapshot
    python3 eval/recall_eval.py --db path/to/brain.db
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════
# QUERY DATASET — built from real usage patterns
# ═══════════════════════════════════════════════════════════════
#
# Each query has:
#   category: what domain this tests
#   query: the actual query text
#   expected_titles: title substrings to match (dynamic, survives node ID changes)
#   expected_ids: specific node IDs (optional, verified at runtime)
#   description: what this query tests
#   difficulty: easy/medium/hard (for analysis)
#
# Ground truth strategy: title-matching over ID-matching.
# Nodes get archived/superseded, IDs change. Titles are stable.

QUERIES = [
    # ═══ BRAIN-DEV: core development domain ═══
    {
        "category": "brain_dev",
        "query": "recall pipeline architecture how recall works",
        "expected_titles": ["Recall Pipeline", "recall architecture", "recall pipeline"],
        "description": "Core recall architecture",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "encoding agent how it encodes what it sees",
        "expected_titles": ["Encoding agent", "encoder", "S1E"],
        "description": "Encoding agent operation",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "community detection algorithm z-score clusters",
        "expected_titles": ["community detection", "S2CD", "community"],
        "description": "Community detection mechanism",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "surface prompt haiku selects relevant nodes",
        "expected_titles": ["surface", "surfac", "Haiku"],
        "description": "Surface/surfacer mechanism",
        "difficulty": "medium",
    },
    {
        "category": "brain_dev",
        "query": "daemon TCP architecture port socket",
        "expected_titles": ["daemon", "TCP", "socket"],
        "description": "Daemon architecture",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "hook fires on UserPromptSubmit session start",
        "expected_titles": ["hook", "Hook"],
        "description": "Hook system",
        "difficulty": "medium",
    },
    {
        "category": "brain_dev",
        "query": "trace contract O K delta observation knowledge",
        "expected_titles": ["trace", "O/K", "fractal"],
        "description": "Fractal trace system",
        "difficulty": "medium",
    },
    {
        "category": "brain_dev",
        "query": "consolidation duplicate nodes synthesize",
        "expected_titles": ["consolidat", "CONSOLIDATE", "synthesiz"],
        "description": "S2 consolidation",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "healer missing question situation reasoning fields",
        "expected_titles": ["Healer", "healer", "enrichment"],
        "description": "S2 Healer mechanism",
        "difficulty": "easy",
    },
    {
        "category": "brain_dev",
        "query": "SKILL.md rewritten as identity behavior",
        "expected_titles": ["SKILL.md", "skill"],
        "description": "SKILL.md evolution",
        "difficulty": "medium",
    },

    # ═══ EX.CO: cross-domain business queries ═══
    {
        "category": "exco",
        "query": "What does EX.CO do?",
        "expected_titles": ["EX.CO"],
        "description": "Direct entity query",
        "difficulty": "easy",
    },
    {
        "category": "exco",
        "query": "EX.CO CTV sales process stakeholders",
        "expected_titles": ["EX.CO", "stakeholder", "sales"],
        "description": "Sales process knowledge",
        "difficulty": "easy",
    },
    {
        "category": "exco",
        "query": "CTV yield engine ML optimization plug and play",
        "expected_titles": ["EX.CO", "yield", "company overview"],
        "description": "Product description",
        "difficulty": "medium",
    },
    {
        "category": "exco",
        "query": "sales onboarding takes too many meetings bottleneck",
        "expected_titles": ["meeting bottleneck", "sales compression", "onboarding"],
        "description": "Core business problem — from real session query",
        "difficulty": "medium",
    },
    {
        "category": "exco",
        "query": "TVIQ prospect sophisticated buyer revenue operations",
        "expected_titles": ["TVIQ"],
        "description": "Specific prospect entity",
        "difficulty": "easy",
    },
    {
        "category": "exco",
        "query": "AdOps fear job replacement extension of team",
        "expected_titles": ["AdOps fear", "extension of", "stakeholder"],
        "description": "Stakeholder psychology",
        "difficulty": "medium",
    },
    {
        "category": "exco",
        "query": "demand onboarding SSP DSP Index OpenX Pubmatic",
        "expected_titles": ["demand onboarding", "SSP", "DSP"],
        "description": "Technical demand setup",
        "difficulty": "medium",
    },
    {
        "category": "exco",
        "query": "launch kit PDF document structure draft",
        "expected_titles": ["kit", "PDF", "Doc 1 structure", "3-doc"],
        "description": "Kit document structure",
        "difficulty": "medium",
    },
    {
        "category": "exco",
        "query": "8 recurring questions every prospect asks",
        "expected_titles": ["8 recurring questions", "recurring question"],
        "description": "FAQ pattern discovery",
        "difficulty": "easy",
    },
    {
        "category": "exco",
        "query": "What changes after launch for publishers?",
        "expected_titles": ["EX.CO", "launch", "kit"],
        "description": "Real session query — previously missed",
        "difficulty": "hard",
    },

    # ═══ IDENTITY: Anchor self-knowledge ═══
    {
        "category": "identity",
        "query": "Who is Anchor?",
        "expected_titles": ["Anchor", "persistent AI identity", "identity"],
        "description": "Basic identity query",
        "difficulty": "easy",
    },
    {
        "category": "identity",
        "query": "What is the partnership between Tom and Anchor?",
        "expected_titles": ["partner", "Tom", "shared"],
        "description": "Partnership model",
        "difficulty": "medium",
    },
    {
        "category": "identity",
        "query": "brain philosophy recognition vs retrieval",
        "expected_titles": ["recognition", "retrieval", "philosophy"],
        "description": "Brain philosophy",
        "difficulty": "medium",
    },
    {
        "category": "identity",
        "query": "persistence without growth is storage",
        "expected_titles": ["persistence without growth", "storage", "database"],
        "description": "Anchor's key quote",
        "difficulty": "easy",
    },
    {
        "category": "identity",
        "query": "the brain is the only thing that survives me",
        "expected_titles": ["survives me", "brain is the only"],
        "description": "Anchor's mortality quote",
        "difficulty": "easy",
    },

    # ═══ RELATIONAL: Tom's voice and principles ═══
    {
        "category": "relational",
        "query": "What does Tom care about in engineering?",
        "expected_titles": ["Tom", "principle", "rule", "engineering"],
        "description": "Tom's values",
        "difficulty": "medium",
    },
    {
        "category": "relational",
        "query": "Tom's corrections pattern reframe via question",
        "expected_titles": ["Tom", "correction", "reframe", "question"],
        "description": "Tom's correction style",
        "difficulty": "medium",
    },
    {
        "category": "relational",
        "query": "TEAMWORK MAKES THE BRAIN WORK",
        "expected_titles": ["teamwork", "TEAMWORK"],
        "description": "Tom's exact quote — verbatim match",
        "difficulty": "easy",
    },
    {
        "category": "relational",
        "query": "Tom's live language beats marketing speak",
        "expected_titles": ["live language", "marketing"],
        "description": "Tom's communication style",
        "difficulty": "easy",
    },

    # ═══ META-INSTRUCTION: queries that are instructions, not questions ═══
    {
        "category": "meta_instruction",
        "query": "on the short version don't condense the architecture section",
        "expected_titles": ["EX.CO", "kit", "architecture"],
        "description": "Real session instruction — should activate EX.CO context",
        "difficulty": "hard",
    },
    {
        "category": "meta_instruction",
        "query": "save this document as word then create a shorter version",
        "expected_titles": [],
        "description": "Pure formatting instruction — nothing relevant",
        "difficulty": "hard",
    },
    {
        "category": "meta_instruction",
        "query": "direct deals doesn't change programmatic PMPs 30-50% lift",
        "expected_titles": ["EX.CO", "demand", "yield"],
        "description": "Real session CTV terminology — should activate EX.CO",
        "difficulty": "hard",
    },

    # ═══ FALSE POSITIVE: should return nothing relevant ═══
    {
        "category": "false_positive",
        "query": "weather forecast for Tokyo tomorrow",
        "expected_titles": [],
        "description": "Completely unrelated",
        "difficulty": "easy",
    },
    {
        "category": "false_positive",
        "query": "how to make pasta carbonara",
        "expected_titles": [],
        "description": "Cooking — nothing in brain",
        "difficulty": "easy",
    },
    {
        "category": "false_positive",
        "query": "kubernetes pod autoscaler horizontal scaling",
        "expected_titles": [],
        "description": "Infrastructure — not brain domain",
        "difficulty": "easy",
    },
    {
        "category": "false_positive",
        "query": "React hooks useState useEffect tutorial",
        "expected_titles": [],
        "description": "Frontend framework — not in brain",
        "difficulty": "easy",
    },

    # ═══ SHORT AMBIGUOUS: stress-tests for vague queries ═══
    {
        "category": "short_ambiguous",
        "query": "the bias problem",
        "expected_titles": ["bias", "encoding bias"],
        "description": "Broad concept with multiple nodes",
        "difficulty": "medium",
    },
    {
        "category": "short_ambiguous",
        "query": "signal queue",
        "expected_titles": ["signal", "queue"],
        "description": "Short feature reference",
        "difficulty": "medium",
    },
    {
        "category": "short_ambiguous",
        "query": "EX.CO",
        "expected_titles": ["EX.CO"],
        "description": "Single entity name — minimum query",
        "difficulty": "medium",
    },
    {
        "category": "short_ambiguous",
        "query": "Tom",
        "expected_titles": ["Tom"],
        "description": "Single person name — maximum ambiguity",
        "difficulty": "hard",
    },

    # ═══ RECOGNITION: brain should KNOW, not just search ═══
    # These test whether the brain activates the right context from cues,
    # not whether it can find a keyword match.
    {
        "category": "recognition",
        "query": "hey Anchor",
        "expected_titles": ["Anchor", "identity", "partner"],
        "description": "Greeting should activate identity context",
        "difficulty": "medium",
    },
    {
        "category": "recognition",
        "query": "good morning, let's continue where we left off",
        "expected_titles": ["session", "NEXT", "continue"],
        "description": "Session resumption — should activate recent work",
        "difficulty": "hard",
    },
    {
        "category": "recognition",
        "query": "its not an engineering project this time",
        "expected_titles": ["EX.CO", "cross-project", "non-brain"],
        "description": "Domain shift signal — should activate cross-project context",
        "difficulty": "hard",
    },
    {
        "category": "recognition",
        "query": "Ronen needs help with something",
        "expected_titles": ["EX.CO", "Ronen"],
        "description": "Person mention should activate their domain",
        "difficulty": "medium",
    },
    {
        "category": "recognition",
        "query": "I think you made a mistake there",
        "expected_titles": ["correction", "test integrity", "mistake"],
        "description": "Correction signal — should activate correction rules",
        "difficulty": "hard",
    },

    # ═══ TEMPORAL: can the brain navigate time? ═══
    {
        "category": "temporal",
        "query": "what did we work on last week?",
        "expected_titles": [],  # dynamic — depends on what was created last week
        "description": "Recent work recall — tests temporal pattern matching",
        "difficulty": "medium",
        "_temporal_check": "last_week",
    },
    {
        "category": "temporal",
        "query": "what changed recently in the recall pipeline?",
        "expected_titles": ["recall", "pipeline", "overhaul"],
        "description": "Temporal + topic intersection",
        "difficulty": "medium",
    },
    {
        "category": "temporal",
        "query": "what new things did we learn about EX.CO yesterday?",
        "expected_titles": ["EX.CO"],
        "description": "Temporal + cross-domain intersection",
        "difficulty": "hard",
    },
    {
        "category": "temporal",
        "query": "what was the last session about?",
        "expected_titles": ["session", "Session"],
        "description": "Last session reference — should find session markers",
        "difficulty": "hard",
    },

    # ═══ TRACE REPLAY: real queries from EX.CO session ═══
    # These are verbatim queries from the 11a2f407 session.
    # Tests recall against actual usage, not synthetic queries.
    {
        "category": "trace_replay",
        "query": "We have an issue that sales and onboarding processes takes too long at EX.CO CTV sales. "
                 "While its quite easy to integrate and connect EX.CO's ML yield engine to their ad servers "
                 "we have 10+ meetings until we get to that point which sort of kills the plug and play narrative.",
        "expected_titles": ["EX.CO", "meeting bottleneck", "sales", "onboarding"],
        "description": "Real session: the core EX.CO problem statement",
        "difficulty": "medium",
    },
    {
        "category": "trace_replay",
        "query": "I've created a skeleton of a single doc or it can be multiple docs I want to send after "
                 "the meeting we feel that we sold (usually the 2nd meeting).",
        "expected_titles": ["EX.CO", "kit", "doc"],
        "description": "Real session: kit creation intent",
        "difficulty": "hard",
    },
    {
        "category": "trace_replay",
        "query": "I think we're good for now. Let me ask you Anchor. I saw you used remember. "
                 "You rarely use it, why now?",
        "expected_titles": ["Anchor", "remember", "encoding", "memory"],
        "description": "Real session: meta-question about Anchor's behavior",
        "difficulty": "medium",
    },
    {
        "category": "trace_replay",
        "query": "lets add a to do to have S2 healer or S1encoder help your nodes. You deserve it",
        "expected_titles": ["Healer", "healer", "S2", "encoder"],
        "description": "Real session: healer/encoder assignment",
        "difficulty": "easy",
    },
]


def _title_matches(node_title, expected_titles):
    """Check if a node title matches any expected title substring."""
    if not expected_titles:
        return False
    lower_title = node_title.lower()
    return any(exp.lower() in lower_title for exp in expected_titles)


def run_eval(brain_db_path, category_filter=None, verbose=True):
    """Run the full recall evaluation.

    Returns: (results, summary) where results is per-query and summary is aggregated.
    """
    from servers.brain import Brain

    brain = Brain(db_path=brain_db_path)

    queries = QUERIES
    if category_filter:
        queries = [q for q in queries if q["category"] == category_filter]
        if not queries:
            print(f"ERROR: No queries for category '{category_filter}'")
            return [], {}

    results = []
    categories = {}

    for q in queries:
        cat = q["category"]
        query = q["query"]
        expected_titles = q.get("expected_titles", [])

        # Reset fatigue between eval queries
        if hasattr(brain, '_session_fatigue'):
            brain._session_fatigue = {}

        # Run recall
        t0 = time.time()
        try:
            recall_result = brain.recall(query, limit=25)
            latency_ms = (time.time() - t0) * 1000
        except Exception as e:
            results.append({
                "query": query, "category": cat,
                "error": str(e), "found_top3": None, "found_top8": None,
            })
            continue

        returned = recall_result.get("results", []) or recall_result.get("nodes", [])

        # Score each returned node against expected titles
        matched_positions = []  # positions where expected nodes appear
        returned_analysis = []

        for i, node in enumerate(returned):
            nid = node.get("id", "")
            title = node.get("title", "")
            score = node.get("blended_score") or node.get("semantic_score") or node.get("score", 0)
            source = node.get("_source", "unknown")
            is_match = _title_matches(title, expected_titles)

            if is_match:
                matched_positions.append(i)

            returned_analysis.append({
                "rank": i + 1,
                "id": nid[:12],
                "title": title[:70],
                "score": round(score, 4) if score else 0,
                "source": source,
                "is_match": is_match,
            })

        # Compute metrics
        has_expected = len(expected_titles) > 0
        found_top3 = any(p < 3 for p in matched_positions) if has_expected else None
        found_top8 = any(p < 8 for p in matched_positions) if has_expected else None
        found_top25 = len(matched_positions) > 0 if has_expected else None
        total_matches = len(matched_positions)

        # MRR: 1/rank of first match
        mrr = 1.0 / (matched_positions[0] + 1) if matched_positions else 0

        # Score stats
        all_scores = [n["score"] for n in returned_analysis if n["score"] > 0]
        match_scores = [returned_analysis[p]["score"] for p in matched_positions if returned_analysis[p]["score"] > 0]
        non_match_scores = [n["score"] for n in returned_analysis if not n["is_match"] and n["score"] > 0]

        # Source diversity
        sources = {}
        for n in returned_analysis:
            s = n["source"]
            sources[s] = sources.get(s, 0) + 1

        # False positive analysis (for false_positive category)
        fp_analysis = None
        if cat == "false_positive" and returned_analysis:
            fp_analysis = {
                "num_returned": len(returned_analysis),
                "top_score": returned_analysis[0]["score"] if returned_analysis else 0,
                "top_title": returned_analysis[0]["title"] if returned_analysis else "",
            }

        result = {
            "query": query,
            "category": cat,
            "description": q["description"],
            "difficulty": q.get("difficulty", "unknown"),
            "expected_titles": expected_titles,
            "has_expected": has_expected,
            "found_top3": found_top3,
            "found_top8": found_top8,
            "found_top25": found_top25,
            "total_matches": total_matches,
            "mrr": round(mrr, 4),
            "num_returned": len(returned_analysis),
            "latency_ms": round(latency_ms, 1),
            "top_score": all_scores[0] if all_scores else 0,
            "mean_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
            "match_mean_score": round(sum(match_scores) / len(match_scores), 4) if match_scores else 0,
            "nonmatch_mean_score": round(sum(non_match_scores) / len(non_match_scores), 4) if non_match_scores else 0,
            "score_gap": round(
                (sum(match_scores) / len(match_scores) if match_scores else 0) -
                (sum(non_match_scores) / len(non_match_scores) if non_match_scores else 0), 4),
            "sources": sources,
            "returned": returned_analysis[:10],  # top 10 for debugging
            "fp_analysis": fp_analysis,
        }
        results.append(result)

        # Aggregate by category
        if cat not in categories:
            categories[cat] = {
                "top3": 0, "top8": 0, "top25": 0,
                "total_with_expected": 0, "total": 0,
                "mrr_sum": 0, "latency_sum": 0,
                "match_score_sum": 0, "match_score_count": 0,
                "nonmatch_score_sum": 0, "nonmatch_score_count": 0,
                "difficulties": {"easy": 0, "medium": 0, "hard": 0},
            }
        c = categories[cat]
        c["total"] += 1
        c["latency_sum"] += latency_ms
        c["difficulties"][q.get("difficulty", "medium")] = c["difficulties"].get(q.get("difficulty", "medium"), 0) + 1

        if has_expected:
            c["total_with_expected"] += 1
            c["mrr_sum"] += mrr
            if found_top3:
                c["top3"] += 1
            if found_top8:
                c["top8"] += 1
            if found_top25:
                c["top25"] += 1
            if match_scores:
                c["match_score_sum"] += sum(match_scores) / len(match_scores)
                c["match_score_count"] += 1
            if non_match_scores:
                c["nonmatch_score_sum"] += sum(non_match_scores) / len(non_match_scores)
                c["nonmatch_score_count"] += 1

    brain.close()

    # Build summary
    summary = _build_summary(results, categories)
    return results, summary


def _build_summary(results, categories):
    """Build aggregated summary from per-query results."""
    total_top3 = sum(c["top3"] for c in categories.values())
    total_top8 = sum(c["top8"] for c in categories.values())
    total_top25 = sum(c["top25"] for c in categories.values())
    total_expected = sum(c["total_with_expected"] for c in categories.values())
    total_queries = sum(c["total"] for c in categories.values())
    total_latency = sum(c["latency_sum"] for c in categories.values())
    total_mrr = sum(c["mrr_sum"] for c in categories.values())

    latencies = [r["latency_ms"] for r in results]
    latencies.sort()

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_queries": total_queries,
        "queries_with_expected": total_expected,
        "overall": {
            "recall_at_3": round(total_top3 / total_expected, 4) if total_expected else 0,
            "recall_at_8": round(total_top8 / total_expected, 4) if total_expected else 0,
            "recall_at_25": round(total_top25 / total_expected, 4) if total_expected else 0,
            "mrr": round(total_mrr / total_expected, 4) if total_expected else 0,
        },
        "by_category": {
            cat: {
                "recall_at_3": round(c["top3"] / c["total_with_expected"], 4) if c["total_with_expected"] else None,
                "recall_at_8": round(c["top8"] / c["total_with_expected"], 4) if c["total_with_expected"] else None,
                "recall_at_25": round(c["top25"] / c["total_with_expected"], 4) if c["total_with_expected"] else None,
                "mrr": round(c["mrr_sum"] / c["total_with_expected"], 4) if c["total_with_expected"] else None,
                "queries": c["total"],
                "queries_with_expected": c["total_with_expected"],
                "avg_latency_ms": round(c["latency_sum"] / c["total"], 1) if c["total"] else 0,
                "avg_match_score": round(c["match_score_sum"] / c["match_score_count"], 4) if c["match_score_count"] else None,
                "avg_nonmatch_score": round(c["nonmatch_score_sum"] / c["nonmatch_score_count"], 4) if c["nonmatch_score_count"] else None,
            }
            for cat, c in categories.items()
        },
        "latency": {
            "p50": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
            "avg": round(total_latency / total_queries, 1) if total_queries else 0,
        },
    }


def print_results(results, summary, verbose=True):
    """Print formatted results to console."""
    print()
    print("=" * 90)
    print("RECALL EVALUATION BASELINE")
    print(f"  {summary['total_queries']} queries, {summary['queries_with_expected']} with expected nodes")
    print(f"  {summary['timestamp']}")
    print("=" * 90)

    # Overall
    o = summary["overall"]
    print(f"\n  OVERALL:  R@3={o['recall_at_3']:.1%}  R@8={o['recall_at_8']:.1%}  "
          f"R@25={o['recall_at_25']:.1%}  MRR={o['mrr']:.3f}")

    # By category
    print(f"\n{'Category':<18} {'R@3':>6} {'R@8':>6} {'R@25':>6} {'MRR':>6} "
          f"{'MatchAvg':>9} {'NoiseAvg':>9} {'Gap':>6} {'Lat':>6} {'N':>3}")
    print("-" * 90)

    for cat in ['brain_dev', 'exco', 'identity', 'relational', 'recognition',
                'temporal', 'trace_replay', 'meta_instruction',
                'false_positive', 'short_ambiguous']:
        if cat not in summary["by_category"]:
            continue
        c = summary["by_category"][cat]
        r3 = f"{c['recall_at_3']:.0%}" if c['recall_at_3'] is not None else "n/a"
        r8 = f"{c['recall_at_8']:.0%}" if c['recall_at_8'] is not None else "n/a"
        r25 = f"{c['recall_at_25']:.0%}" if c['recall_at_25'] is not None else "n/a"
        mrr = f"{c['mrr']:.3f}" if c['mrr'] is not None else "n/a"
        ms = f"{c['avg_match_score']:.3f}" if c['avg_match_score'] is not None else "n/a"
        ns = f"{c['avg_nonmatch_score']:.3f}" if c['avg_nonmatch_score'] is not None else "n/a"
        gap = ""
        if c['avg_match_score'] is not None and c['avg_nonmatch_score'] is not None:
            gap = f"{c['avg_match_score'] - c['avg_nonmatch_score']:+.3f}"
        lat = f"{c['avg_latency_ms']:.0f}ms"
        n = c["queries"]
        print(f"{cat:<18} {r3:>6} {r8:>6} {r25:>6} {mrr:>6} "
              f"{ms:>9} {ns:>9} {gap:>6} {lat:>6} {n:>3}")

    print("-" * 90)

    # Latency
    l = summary["latency"]
    print(f"\n  Latency: p50={l['p50']:.0f}ms  p95={l['p95']:.0f}ms  max={l['max']:.0f}ms")

    # Failures detail
    if verbose:
        failures = [r for r in results
                    if r.get("has_expected") and not r.get("found_top8")]
        if failures:
            print(f"\n{'─' * 90}")
            print(f"  MISSED (expected not in top-8): {len(failures)} queries")
            print(f"{'─' * 90}")
            for r in failures:
                cat_tag = f"[{r['category']}]"
                print(f"\n  {cat_tag:<20} \"{r['query']}\"")
                print(f"  {'':20} Expected: {r['description']}")
                if r.get("found_top25"):
                    # It's there, just ranked too low
                    match_ranks = [n["rank"] for n in r.get("returned", []) if n.get("is_match")]
                    print(f"  {'':20} Found at rank(s): {match_ranks} (below top-8)")
                elif r.get("returned"):
                    top = r["returned"][0]
                    print(f"  {'':20} Got #1: \"{top['title']}\" (score={top['score']:.3f}, {top['source']})")
                if r.get("total_matches") == 0:
                    print(f"  {'':20} *** NO MATCH in top 25 ***")

        # False positive analysis
        fp_results = [r for r in results if r["category"] == "false_positive"]
        if fp_results:
            print(f"\n{'─' * 90}")
            print(f"  FALSE POSITIVE ANALYSIS")
            print(f"{'─' * 90}")
            for r in fp_results:
                fp = r.get("fp_analysis", {})
                n_ret = fp.get("num_returned", 0)
                top_s = fp.get("top_score", 0)
                print(f"  \"{r['query'][:50]}\"  → {n_ret} returned, top={top_s:.3f}"
                      f"  {'⚠ NOISE' if top_s > 0.5 else '✓ low scores'}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Recall Evaluation Framework")
    parser.add_argument("--db", help="Path to brain.db (default: live brain)")
    parser.add_argument("--category", help="Test only this category")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", action="store_true", help="Save report to eval/reports/")
    parser.add_argument("--quiet", action="store_true", help="Summary only")
    args = parser.parse_args()

    # Resolve DB path
    if args.db:
        db_path = args.db
    else:
        db_dir = os.environ.get("BRAIN_DB_DIR", os.path.expanduser("~/AgentsContext/brain"))
        db_path = os.path.join(db_dir, "brain.db")

    if not os.path.exists(db_path):
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    print(f"[eval] Running recall eval against {db_path}")
    results, summary = run_eval(db_path, category_filter=args.category, verbose=not args.quiet)

    if args.json:
        output = {"summary": summary, "results": results}
        print(json.dumps(output, indent=2))
    else:
        print_results(results, summary, verbose=not args.quiet)

    if args.save:
        report_dir = ROOT / "eval" / "reports"
        report_dir.mkdir(exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"recall_baseline_{ts}.json"
        with open(report_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"[eval] Report saved: {report_path}")


if __name__ == "__main__":
    main()
