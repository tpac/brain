#!/usr/bin/env python3
"""Encoding Suite — measures encoding quality using real conversations.

Feeds real conversation transcripts to the encoding agent (Sonnet) and
measures what gets encoded against ground truth targets.

Each test uses a FRESH brain — no existing knowledge to confuse results.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 eval/encoding_suite.py

    # Specific category
    python3 eval/encoding_suite.py --category architecture_decisions

    # Specific model
    python3 eval/encoding_suite.py --model claude-sonnet-4-6

    # Parallel workers
    python3 eval/encoding_suite.py --workers 4
"""
import sys
import os
import json
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from eval.scoring import score_encoding_run
from eval.corpus.loader import load_corpus

# ── Fake Brain Tools (capture encoding intent without real brain) ──

ENCODING_TOOLS = [
    {
        "name": "brain_remember",
        "description": "Store a memory node in the brain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Node type: decision, lesson, correction, pattern, context, mechanism, convention, vocabulary"},
                "title": {"type": "string", "description": "Specific, scannable title"},
                "content": {"type": "string", "description": "Rich content with reasoning and context"},
                "keywords": {"type": "string", "description": "Space-separated retrieval keywords"},
                "locked": {"type": "boolean", "description": "Lock to prevent decay"},
            },
            "required": ["type", "title", "content"]
        }
    },
    {
        "name": "brain_remember_lesson",
        "description": "Store a lesson learned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "what_happened": {"type": "string"},
                "root_cause": {"type": "string"},
                "fix": {"type": "string"},
                "preventive_principle": {"type": "string"},
            },
            "required": ["title", "what_happened", "root_cause", "fix", "preventive_principle"]
        }
    },
    {
        "name": "brain_remember_mechanism",
        "description": "Store how something works.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "brain_record_divergence",
        "description": "Record where the AI diverged from reality.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claude_assumed": {"type": "string"},
                "reality": {"type": "string"},
                "underlying_pattern": {"type": "string"},
            },
            "required": ["claude_assumed", "reality", "underlying_pattern"]
        }
    },
    {
        "name": "brain_learn_vocabulary",
        "description": "Map an operator term to its meaning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "maps_to": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["term", "maps_to", "context"]
        }
    },
    {
        "name": "brain_remember_convention",
        "description": "Store a coding convention or pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "pattern": {"type": "string"},
                "anti_pattern": {"type": "string"},
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "brain_connect",
        "description": "Create a link between two brain nodes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "relation": {"type": "string"},
            },
            "required": ["source_id", "target_id"]
        }
    },
]


def _build_encoding_prompt(conversation: dict) -> str:
    """Build the system prompt for the encoding agent."""
    exchanges = conversation["exchanges"]
    conv_text = "\n".join(
        "[%s]: %s" % (ex["role"].upper(), ex["content"][:1000])
        for ex in exchanges
    )
    return conv_text


ENCODING_SYSTEM = """You are the encoding agent for a persistent AI brain.

You've just observed the following conversation between an operator (user) and an AI assistant. Your job is to extract and encode important knowledge using the brain tools.

Encode:
- Decisions and their reasoning (WHY, not just WHAT)
- Lessons from mistakes
- Corrections — when the operator corrected the AI
- Vocabulary — terms with specific meaning
- Mechanisms — how something works
- Conventions — coding/working patterns
- Patterns — recurring preferences or behaviors

Do NOT encode:
- Casual chat ("ok", "yes", "next")
- Things that are obvious from code
- Single-instance observations (wait for patterns)

Quality: Rich content, 100-500 chars. Include reasoning and context.
Volume: 0-5 encodes max. Over-encoding creates noise.

After encoding, respond with a brief summary of what you encoded."""


def run_encoding_test(client, model: str, conversation: dict, verbose: bool = False) -> dict:
    """Run encoding agent on a single conversation.

    Returns dict with encoded nodes and scores.
    """
    conv_text = _build_encoding_prompt(conversation)

    messages = [
        {"role": "user", "content": "Here is the conversation to analyze:\n\n%s\n\nEncode what's important." % conv_text}
    ]

    tool_calls = []
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=ENCODING_SYSTEM,
        messages=messages,
        tools=ENCODING_TOOLS,
    )

    # Tool use loop (max 5 turns)
    for _ in range(5):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls.append({"name": tu.name, "input": tu.input})
            if verbose:
                title = tu.input.get('title', tu.input.get('term', '...'))
                print("    [%s] %s" % (tu.name.replace('brain_', ''), title[:60]))

        # Simulate tool results
        messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content
            ]
        })
        tool_results = [
            {"type": "tool_result", "tool_use_id": tu.id,
             "content": json.dumps({"status": "ok", "id": "node_%d" % i, "message": "Stored: %s" % tu.input.get('title', 'ok')})}
            for i, tu in enumerate(tool_uses)
        ]
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=ENCODING_SYSTEM,
            messages=messages,
            tools=ENCODING_TOOLS,
        )

    # Convert tool calls to node-like dicts for scoring
    encoded_nodes = []
    for tc in tool_calls:
        inp = tc["input"]
        node = {
            "type": inp.get("type", tc["name"].replace("brain_remember_", "").replace("brain_record_", "").replace("brain_learn_", "").replace("brain_", "")),
            "title": inp.get("title", inp.get("term", "")),
            "content": inp.get("content", inp.get("what_happened", inp.get("maps_to", ""))),
            "keywords": inp.get("keywords", ""),
            "connections": 1 if tc["name"] == "brain_connect" else 0,
        }
        encoded_nodes.append(node)

    # Score against ground truth
    ground_truth = conversation.get("ground_truth", {}).get("encode_targets", [])
    scores = score_encoding_run(encoded_nodes, ground_truth)
    scores["conversation_id"] = conversation["id"]
    scores["category"] = conversation["category"]
    scores["tool_calls"] = tool_calls
    scores["encoded_nodes"] = encoded_nodes

    return scores


def run_suite(model: str = "claude-sonnet-4-6",
              category: str = None,
              max_workers: int = 4,
              verbose: bool = True,
              corpus_dir: str = None) -> dict:
    """Run the full encoding suite.

    Args:
        model: Anthropic model to use
        category: Optional category filter
        max_workers: Parallel workers
        verbose: Print progress
        corpus_dir: Override corpus directory
    """
    client = anthropic.Anthropic()
    conversations = load_corpus(category=category, corpus_dir=corpus_dir)

    if not conversations:
        print("No conversations in corpus. Add JSON files to eval/corpus/")
        return {"suite": "encoding", "error": "empty_corpus"}

    if verbose:
        print("Encoding Suite: %d conversations, model=%s, workers=%d" % (
            len(conversations), model, max_workers))

    results = []
    t0 = time.time()

    if max_workers <= 1:
        for conv in conversations:
            if verbose:
                print("  [%s] %s..." % (conv["category"], conv["id"]))
            r = run_encoding_test(client, model, conv, verbose=verbose)
            results.append(r)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for conv in conversations:
                f = pool.submit(run_encoding_test, client, model, conv, verbose=False)
                futures[f] = conv

            for f in as_completed(futures):
                conv = futures[f]
                try:
                    r = f.result()
                    results.append(r)
                    if verbose:
                        print("  [%s] %s: %d nodes, richness=%.0f" % (
                            conv["category"], conv["id"],
                            r.get("volume", 0), r.get("richness", 0)))
                except Exception as e:
                    results.append({
                        "conversation_id": conv["id"],
                        "category": conv["category"],
                        "error": str(e),
                    })
                    if verbose:
                        print("  [ERROR] %s: %s" % (conv["id"], e))

    elapsed = time.time() - t0

    # Aggregate
    valid = [r for r in results if "error" not in r]
    aggregate = {
        "conversations_tested": len(results),
        "conversations_valid": len(valid),
        "elapsed_seconds": elapsed,
    }

    if valid:
        for key in ["coverage", "precision", "type_accuracy", "richness", "volume"]:
            values = [r[key] for r in valid if r.get(key) is not None]
            if values:
                aggregate["avg_%s" % key] = sum(values) / len(values)

        aggregate["volume_ok_pct"] = sum(1 for r in valid if r.get("volume_ok")) / len(valid)

    # Category breakdown
    cats = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r)

    category_scores = {}
    for cat, cat_results in cats.items():
        cat_agg = {}
        for key in ["coverage", "precision", "richness", "volume"]:
            values = [r[key] for r in cat_results if r.get(key) is not None]
            if values:
                cat_agg["avg_%s" % key] = sum(values) / len(values)
        cat_agg["count"] = len(cat_results)
        category_scores[cat] = cat_agg

    return {
        "suite": "encoding",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": aggregate,
        "per_category": category_scores,
        "per_conversation": results,
    }


def print_results(results: dict):
    """Print formatted results."""
    overall = results.get("overall", {})
    print("\n" + "=" * 70)
    print("ENCODING SUITE RESULTS")
    print("=" * 70)

    print("\nOverall (%d conversations, %.0fs):" % (
        overall.get("conversations_tested", 0),
        overall.get("elapsed_seconds", 0)))
    print("  Coverage:      %s" % _fmt_pct(overall.get("avg_coverage")))
    print("  Precision:     %s" % _fmt_pct(overall.get("avg_precision")))
    print("  Type accuracy: %s" % _fmt_pct(overall.get("avg_type_accuracy")))
    print("  Richness:      %.0f/100" % overall.get("avg_richness", 0))
    print("  Avg volume:    %.1f nodes" % overall.get("avg_volume", 0))
    print("  Volume OK:     %s" % _fmt_pct(overall.get("volume_ok_pct")))

    if results.get("per_category"):
        print("\nBy Category:")
        print("  %-20s %8s %8s %8s %5s" % ("Category", "Cover", "Prec", "Rich", "N"))
        for cat, scores in sorted(results["per_category"].items()):
            print("  %-20s %7s %7s %7.0f %5d" % (
                cat[:20],
                _fmt_pct(scores.get("avg_coverage")),
                _fmt_pct(scores.get("avg_precision")),
                scores.get("avg_richness", 0),
                scores.get("count", 0),
            ))

    print()


def _fmt_pct(val):
    return "%.0f%%" % (val * 100) if val is not None else "N/A"


def save_results(results: dict, output_dir: str = None):
    """Save results to JSON."""
    if output_dir is None:
        output_dir = str(ROOT / "eval" / "results")
    os.makedirs(output_dir, exist_ok=True)

    filename = "encoding_%s.json" % time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved: %s" % path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Encoding Suite")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--category", help="Filter to category")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--corpus-dir", help="Override corpus directory")
    args = parser.parse_args()

    results = run_suite(
        model=args.model,
        category=args.category,
        max_workers=args.workers,
        verbose=not args.quiet,
        corpus_dir=args.corpus_dir,
    )
    print_results(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
