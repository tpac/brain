#!/usr/bin/env python3
"""End-to-End Suite — encodes from conversation, then tests recall.

Phase 1: Feed conversation to encoding agent → encode into FRESH brain
Phase 2: Query the brain with ground truth queries → measure recall

This tests the full loop: conversation → encoding → retrieval.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 eval/e2e_suite.py

    # Specific conversation
    python3 eval/e2e_suite.py --id conv_001
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
from eval.scoring import score_encoding_run, score_decoding_query, score_decoding_suite, score_e2e_run
from eval.corpus.loader import load_corpus
from eval.encoding_suite import ENCODING_TOOLS, ENCODING_SYSTEM, _build_encoding_prompt


def _create_fresh_brain(work_dir: str):
    """Create a fresh brain instance with empty DB."""
    from servers.brain import Brain

    db_path = os.path.join(work_dir, "brain.db")
    brain = Brain(db_path=db_path)
    return brain, db_path


def _encode_into_brain(brain, client, model: str, conversation: dict, verbose: bool = False) -> list:
    """Run encoding agent and store nodes in a real brain.

    Unlike encoding_suite which uses fake tools, this stores nodes for real
    so we can test recall afterward.

    Returns list of encoded node dicts.
    """
    conv_text = _build_encoding_prompt(conversation)

    messages = [
        {"role": "user", "content": "Here is the conversation to analyze:\n\n%s\n\nEncode what's important." % conv_text}
    ]

    encoded_nodes = []
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=ENCODING_SYSTEM,
        messages=messages,
        tools=ENCODING_TOOLS,
    )

    for _ in range(5):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            inp = tu.input
            node_id = None

            # Actually store in the real brain
            try:
                if tu.name == "brain_remember":
                    result = brain.remember(
                        type=inp.get("type", "context"),
                        title=inp.get("title", ""),
                        content=inp.get("content", ""),
                        keywords=inp.get("keywords", ""),
                        locked=inp.get("locked", False),
                    )
                    node_id = result.get("id") if result else None
                elif tu.name == "brain_remember_lesson":
                    result = brain.remember_lesson(
                        title=inp.get("title", ""),
                        what_happened=inp.get("what_happened", ""),
                        root_cause=inp.get("root_cause", ""),
                        fix=inp.get("fix", ""),
                        preventive_principle=inp.get("preventive_principle", ""),
                    )
                    node_id = result.get("id") if result else None
                elif tu.name == "brain_remember_mechanism":
                    result = brain.remember_mechanism(
                        title=inp.get("title", ""),
                        content=inp.get("content", ""),
                        steps=inp.get("steps", []),
                    )
                    node_id = result.get("id") if result else None
                elif tu.name == "brain_record_divergence":
                    result = brain.record_divergence(
                        claude_assumed=inp.get("claude_assumed", ""),
                        reality=inp.get("reality", ""),
                        underlying_pattern=inp.get("underlying_pattern", ""),
                    )
                    node_id = "divergence"
                elif tu.name == "brain_learn_vocabulary":
                    result = brain.learn_vocabulary(
                        term=inp.get("term", ""),
                        maps_to=inp.get("maps_to", ""),
                        context=inp.get("context", ""),
                    )
                    node_id = result.get("id") if result else None
                elif tu.name == "brain_remember_convention":
                    result = brain.remember_convention(
                        title=inp.get("title", ""),
                        content=inp.get("content", ""),
                        pattern=inp.get("pattern", ""),
                        anti_pattern=inp.get("anti_pattern", ""),
                    )
                    node_id = result.get("id") if result else None
                elif tu.name == "brain_connect":
                    brain.connect(
                        source_id=inp.get("source_id", ""),
                        target_id=inp.get("target_id", ""),
                        relation=inp.get("relation", "related_to"),
                    )
                    node_id = "connection"
            except Exception as e:
                if verbose:
                    print("    [ERROR] %s: %s" % (tu.name, e))

            encoded_nodes.append({
                "type": inp.get("type", tu.name.replace("brain_", "")),
                "title": inp.get("title", inp.get("term", "")),
                "content": inp.get("content", inp.get("what_happened", "")),
                "id": node_id,
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps({"status": "ok", "id": node_id or "ok"})
            })

            if verbose:
                title = inp.get('title', inp.get('term', '...'))
                print("    [%s] %s → %s" % (tu.name.replace('brain_', ''), title[:50], node_id or 'ok'))

        messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content
            ]
        })
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=ENCODING_SYSTEM,
            messages=messages,
            tools=ENCODING_TOOLS,
        )

    brain.save()
    return encoded_nodes


def run_e2e_test(client, model: str, conversation: dict, verbose: bool = False) -> dict:
    """Run a single end-to-end test: encode then decode.

    Creates a temporary brain, encodes from conversation, then queries.
    """
    work_dir = tempfile.mkdtemp(prefix="brain_e2e_")

    try:
        brain, db_path = _create_fresh_brain(work_dir)

        # Phase 1: Encode
        if verbose:
            print("  Phase 1: Encoding %s..." % conversation["id"])
        encoded_nodes = _encode_into_brain(brain, client, model, conversation, verbose=verbose)

        ground_truth = conversation.get("ground_truth", {})
        encode_targets = ground_truth.get("encode_targets", [])
        encoding_scores = score_encoding_run(encoded_nodes, encode_targets)

        # Phase 2: Decode
        decode_queries = ground_truth.get("decode_queries", [])
        if not decode_queries:
            # Auto-generate queries from encode targets
            decode_queries = [
                {"query": t["topic"], "expected_topics": [t["topic"]]}
                for t in encode_targets
            ]

        if verbose:
            print("  Phase 2: Decoding %d queries..." % len(decode_queries))

        decode_results = []
        for dq in decode_queries:
            query = dq["query"]
            t0 = time.time()
            try:
                recall_result = brain.recall(query, limit=8)
                latency_ms = (time.time() - t0) * 1000
            except Exception as e:
                decode_results.append({"query": query, "error": str(e), "recall_at_3": 0, "recall_at_8": 0, "mrr": 0, "latency_ms": 0})
                continue

            returned = recall_result.get("results", []) or recall_result.get("nodes", [])

            # For E2E, we check if any returned node's title/content matches expected topics
            hit = False
            for node in returned[:8]:
                node_text = (node.get("title", "") + " " + node.get("content", "")).lower()
                for topic in dq.get("expected_topics", []):
                    topic_words = [w for w in topic.lower().split() if len(w) > 2]
                    if topic_words and sum(1 for w in topic_words if w in node_text) >= len(topic_words) * 0.5:
                        hit = True
                        break
                if hit:
                    break

            decode_results.append({
                "query": query,
                "hit_at_8": 1.0 if hit else 0.0,
                "recall_at_8": 1.0 if hit else 0.0,
                "recall_at_3": 1.0 if any(
                    _topic_in_node(returned[i], dq.get("expected_topics", []))
                    for i in range(min(3, len(returned)))
                ) else 0.0,
                "mrr": 0,  # simplified for topic matching
                "latency_ms": latency_ms,
                "returned_count": len(returned),
            })

            if verbose:
                status = "HIT" if hit else "MISS"
                print("    [%s] %s" % (status, query[:50]))

        brain.close()

        decoding_scores = score_decoding_suite(decode_results)
        e2e_scores = score_e2e_run(encoding_scores, decoding_scores)
        e2e_scores["conversation_id"] = conversation["id"]
        e2e_scores["category"] = conversation["category"]
        e2e_scores["encoded_count"] = len(encoded_nodes)
        e2e_scores["queries_count"] = len(decode_queries)

        return e2e_scores

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _topic_in_node(node, expected_topics):
    """Check if a node matches any expected topic."""
    node_text = (node.get("title", "") + " " + node.get("content", "")).lower()
    for topic in expected_topics:
        words = [w for w in topic.lower().split() if len(w) > 2]
        if words and sum(1 for w in words if w in node_text) >= len(words) * 0.5:
            return True
    return False


def run_suite(model: str = "claude-sonnet-4-6",
              category: str = None,
              conv_id: str = None,
              max_workers: int = 4,
              verbose: bool = True,
              corpus_dir: str = None) -> dict:
    """Run the full E2E suite."""
    client = anthropic.Anthropic()
    conversations = load_corpus(category=category, corpus_dir=corpus_dir)

    if conv_id:
        conversations = [c for c in conversations if c["id"] == conv_id]

    if not conversations:
        print("No conversations found.")
        return {"suite": "e2e", "error": "empty_corpus"}

    if verbose:
        print("E2E Suite: %d conversations, model=%s" % (len(conversations), model))

    results = []
    t0 = time.time()

    # E2E tests are heavier — use fewer workers
    workers = min(max_workers, len(conversations))

    if workers <= 1:
        for conv in conversations:
            r = run_e2e_test(client, model, conv, verbose=verbose)
            results.append(r)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for conv in conversations:
                f = pool.submit(run_e2e_test, client, model, conv, verbose=False)
                futures[f] = conv

            for f in as_completed(futures):
                conv = futures[f]
                try:
                    r = f.result()
                    results.append(r)
                    if verbose:
                        print("  [%s] %s: synergy=%.2f, encoded=%d, round_trip=%.0f%%" % (
                            conv["category"], conv["id"],
                            r.get("synergy_score", 0),
                            r.get("encoded_count", 0),
                            r.get("round_trip_accuracy", 0) * 100))
                except Exception as e:
                    results.append({"conversation_id": conv["id"], "error": str(e)})
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
        for key in ["round_trip_accuracy", "synergy_score"]:
            values = [r.get(key, 0) for r in valid]
            aggregate["avg_%s" % key] = sum(values) / len(values) if values else 0

    return {
        "suite": "e2e",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": aggregate,
        "per_conversation": results,
    }


def print_results(results: dict):
    """Print formatted results."""
    overall = results.get("overall", {})
    print("\n" + "=" * 70)
    print("E2E SUITE RESULTS")
    print("=" * 70)

    print("\nOverall (%d conversations, %.0fs):" % (
        overall.get("conversations_tested", 0),
        overall.get("elapsed_seconds", 0)))
    print("  Round-trip accuracy: %.0f%%" % (overall.get("avg_round_trip_accuracy", 0) * 100))
    print("  Synergy score:      %.2f" % overall.get("avg_synergy_score", 0))

    print()


def save_results(results: dict, output_dir: str = None):
    """Save results to JSON."""
    if output_dir is None:
        output_dir = str(ROOT / "eval" / "results")
    os.makedirs(output_dir, exist_ok=True)

    filename = "e2e_%s.json" % time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved: %s" % path)
    return path


def main():
    parser = argparse.ArgumentParser(description="E2E Suite")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--category", help="Filter to category")
    parser.add_argument("--id", help="Run specific conversation")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--corpus-dir", help="Override corpus directory")
    args = parser.parse_args()

    results = run_suite(
        model=args.model,
        category=args.category,
        conv_id=args.id,
        max_workers=args.workers,
        verbose=not args.quiet,
        corpus_dir=args.corpus_dir,
    )
    print_results(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
