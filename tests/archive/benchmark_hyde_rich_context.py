#!/usr/bin/env python3
"""
Benchmark HyDE (Hypothetical Document Embedding) with RICH DOMAIN CONTEXT.

Previous HyDE attempts failed because the LLM got bare queries like "what is glo"
with zero domain context. This version feeds the LLM real context from the brain DB:
- Top 50 node titles (by confidence)
- Distinct project names
- Node type distribution
- Sample keywords

Tests both TinyLlama (1.1B) and Gemma 2B via Ollama.

Control baseline (v1.5 without HyDE):
  NDCG@10: 0.204, MRR: 0.202, Passed: 34/104

Usage:
    python tests/benchmark_hyde_rich_context.py                    # Run both models
    python tests/benchmark_hyde_rich_context.py --model tinyllama  # TinyLlama only
    python tests/benchmark_hyde_rich_context.py --model gemma2:2b  # Gemma only
"""
import sys
import os
import time
import shutil
import json
import sqlite3
import requests
from collections import Counter

os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import copy_brain_for_testing

OLLAMA_URL = "http://localhost:11434/api/generate"

# ═══════════════════════════════════════════════════════════════
# DOMAIN CONTEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_domain_context(db_path: str) -> dict:
    """
    Pull rich domain context from the brain DB to feed the LLM.

    Returns dict with:
      - projects: list of project names
      - top_titles: top 50 node titles by confidence
      - keywords: most common keywords across nodes
      - type_counts: distribution of node types
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Top 50 node titles by confidence (most important knowledge)
    rows = conn.execute(
        "SELECT title, type, confidence FROM nodes "
        "WHERE archived = 0 "
        "ORDER BY confidence DESC, access_count DESC "
        "LIMIT 50"
    ).fetchall()
    top_titles = [(r['title'], r['type']) for r in rows]

    # Distinct project names
    rows = conn.execute(
        "SELECT DISTINCT project FROM nodes "
        "WHERE project IS NOT NULL AND project != '' "
        "AND archived = 0"
    ).fetchall()
    projects = [r['project'] for r in rows]

    # Most common keywords (parse comma-separated keyword fields)
    rows = conn.execute(
        "SELECT keywords FROM nodes "
        "WHERE keywords IS NOT NULL AND keywords != '' "
        "AND archived = 0"
    ).fetchall()
    kw_counter = Counter()
    for r in rows:
        for kw in r['keywords'].split(','):
            kw = kw.strip().lower()
            if kw and len(kw) > 2:
                kw_counter[kw] += 1
    top_keywords = [kw for kw, _ in kw_counter.most_common(40)]

    # Node type distribution
    rows = conn.execute(
        "SELECT type, COUNT(*) as cnt FROM nodes "
        "WHERE archived = 0 GROUP BY type ORDER BY cnt DESC"
    ).fetchall()
    type_counts = {r['type']: r['cnt'] for r in rows}

    # Check if vocabulary table exists
    vocab_terms = []
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if 'vocabulary' in tables:
        rows = conn.execute(
            "SELECT term, meaning FROM vocabulary LIMIT 30"
        ).fetchall()
        vocab_terms = [(r['term'], r['meaning']) for r in rows]

    conn.close()

    return {
        'projects': projects,
        'top_titles': top_titles,
        'top_keywords': top_keywords,
        'type_counts': type_counts,
        'vocab_terms': vocab_terms,
    }


def build_system_prompt(ctx: dict) -> str:
    """Build a rich system prompt from extracted domain context."""

    # Format projects
    project_list = ', '.join(ctx['projects']) if ctx['projects'] else 'various software projects'

    # Format vocabulary if available
    vocab_section = ""
    if ctx['vocab_terms']:
        vocab_lines = [f"  - {term}: {meaning}" for term, meaning in ctx['vocab_terms']]
        vocab_section = "\nKey vocabulary:\n" + "\n".join(vocab_lines) + "\n"

    # Format top titles grouped by type
    titles_by_type = {}
    for title, ntype in ctx['top_titles']:
        titles_by_type.setdefault(ntype, []).append(title)

    title_lines = []
    for ntype in sorted(titles_by_type.keys()):
        titles = titles_by_type[ntype][:8]  # Max 8 per type
        title_lines.append(f"  [{ntype}] {', '.join(titles)}")

    titles_section = "\n".join(title_lines)

    # Format keywords
    keywords_section = ", ".join(ctx['top_keywords'][:30])

    prompt = f"""You are a recall assistant for a personal knowledge base belonging to a software engineer / CEO. The knowledge base contains information about:

Projects: {project_list}

Important topics (sample of stored knowledge):
{titles_section}
{vocab_section}
Frequently used terms: {keywords_section}

The knowledge base stores decisions, lessons learned, architecture notes, bug reports, coding conventions, mental models, and personal reflections.

Given this domain context, write a short paragraph (3-5 sentences) that would be a LIKELY ANSWER stored in this knowledge base for the following query. Use the same terminology and concepts as the knowledge base. Write it as a factual note, not as a response to a question."""

    return prompt


# ═══════════════════════════════════════════════════════════════
# HYDE EXPANSION
# ═══════════════════════════════════════════════════════════════

def hyde_expand(query: str, model: str, system_prompt: str) -> str:
    """
    Use an Ollama model to generate a hypothetical document answering the query.
    The system prompt provides rich domain context.
    """
    user_prompt = f"Query: {query}\n\nHypothetical knowledge base entry:"

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200,
                "top_p": 0.9,
            }
        }, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        generated = result.get("response", "").strip()

        if not generated:
            return query

        # Combine original query + generated text for embedding
        return query + " " + generated

    except requests.exceptions.ConnectionError:
        print(f"[hyde] ERROR: Cannot connect to Ollama at {OLLAMA_URL}", file=sys.stderr)
        print(f"[hyde] Make sure Ollama is running: ollama serve", file=sys.stderr)
        return query
    except Exception as e:
        print(f"[hyde] Error with {model}: {e}", file=sys.stderr)
        return query


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════

def run_benchmark(model: str, brain, system_prompt: str, golden_path: str):
    """
    Run the full HyDE benchmark for a given model.
    Returns (eval_result, hyde_examples, elapsed).
    """
    from servers import embedder
    from tests.eval_runner import GoldenEvaluator

    # Monkey-patch recall to use HyDE-expanded embeddings
    original_recall = brain.recall.__func__

    hyde_cache = {}  # Cache expansions to avoid re-generating for examples

    def hyde_recall(self, query, types=None, limit=20, offset=0,
                    include_archived=False, min_recency=0, project=None,
                    session_id=None):
        """Recall with HyDE query expansion."""
        expanded = hyde_expand(query, model, system_prompt)
        hyde_cache[query] = expanded

        original_embed = embedder.embed
        first_call = [True]

        def hyde_embed(text):
            if first_call[0]:
                first_call[0] = False
                return original_embed(expanded)
            return original_embed(text)

        embedder.embed = hyde_embed
        try:
            result = original_recall(
                self, query, types=types, limit=limit, offset=offset,
                include_archived=include_archived, min_recency=min_recency,
                project=project, session_id=session_id
            )
        finally:
            embedder.embed = original_embed
            first_call[0] = True

        return result

    # Apply patch
    brain.recall = hyde_recall.__get__(brain, type(brain))

    # Run eval
    evaluator = GoldenEvaluator(brain)
    t0 = time.time()
    result = evaluator.run(verbose=False)
    elapsed = time.time() - t0

    # Restore original
    brain.recall = original_recall.__get__(brain, type(brain))

    return result, hyde_cache, elapsed


def print_results(model: str, result: dict, hyde_cache: dict, elapsed: float):
    """Print formatted results for one model."""
    s = result['summary']
    agg = result['aggregate']

    print()
    print("=" * 70)
    print(f"  v1.5 + HyDE with Rich Context ({model})")
    print("=" * 70)
    print()
    print(f"  NDCG@10:     {agg.get('ndcg@10', {}).get('mean', 0):.3f}")
    print(f"  MRR:         {agg.get('mrr', {}).get('mean', 0):.3f}")
    print(f"  hit_rate@10: {agg.get('hit_rate@10', {}).get('mean', 0):.3f}")
    print(f"  precision@5: {agg.get('precision@5', {}).get('mean', 0):.3f}")
    print(f"  recall@10:   {agg.get('recall@10', {}).get('mean', 0):.3f}")
    print(f"  Passed:      {s['passed']}/{s['total']}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/s['total']*1000:.0f}ms/case)")
    print()

    # Comparison with baseline
    baseline_ndcg = 0.204
    baseline_mrr = 0.202
    baseline_passed = 34

    ndcg = agg.get('ndcg@10', {}).get('mean', 0)
    mrr_val = agg.get('mrr', {}).get('mean', 0)

    ndcg_delta = ndcg - baseline_ndcg
    mrr_delta = mrr_val - baseline_mrr
    passed_delta = s['passed'] - baseline_passed

    print(f"  vs. Baseline (no HyDE):")
    print(f"    NDCG@10: {baseline_ndcg:.3f} -> {ndcg:.3f}  ({ndcg_delta:+.3f})")
    print(f"    MRR:     {baseline_mrr:.3f} -> {mrr_val:.3f}  ({mrr_delta:+.3f})")
    print(f"    Passed:  {baseline_passed} -> {s['passed']}  ({passed_delta:+d})")
    print()

    # By category
    print("  --- By Category ---")
    cat_agg = result.get('by_category', {})
    for cat, metrics in sorted(cat_agg.items()):
        ndcg_data = metrics.get('ndcg@10', {})
        mrr_data = metrics.get('mrr', {})
        count = mrr_data.get('count', 0)
        print(f"    {cat:25s}: NDCG={ndcg_data.get('mean', 0):.3f}  MRR={mrr_data.get('mean', 0):.3f}  (n={count})")
    print()

    # Example HyDE expansions
    print("  --- Example HyDE Expansions ---")
    example_queries = [
        "why did we separate the backend from the frontend",
        "how do we make money",
        "what is glo",
        "who runs the company and what does he do",
        "can mixins define __init__",
        "what keeps breaking",
        "how do we prevent fake accounts and bots",
        "what happens when claude runs out of context window",
        "why was brain.py broken into multiple files",
        "claude agrees too easily and makes up what the other person thinks",
    ]

    shown = 0
    for q in example_queries:
        if q in hyde_cache and shown < 10:
            expanded = hyde_cache[q]
            # Show the generated part (after the original query)
            generated_part = expanded[len(q):].strip()
            print(f"\n  Q: {q}")
            print(f"  HyDE: {generated_part[:300]}")
            shown += 1

    # If cache has entries we didn't show, generate fresh for examples
    if shown == 0:
        print("\n  (No cached expansions — generating examples...)")
        for q in example_queries[:5]:
            expanded = hyde_expand(q, model if ':' not in model else model, "")
            generated_part = expanded[len(q):].strip()
            print(f"\n  Q: {q}")
            print(f"  HyDE: {generated_part[:300]}")

    print()


def main():
    args = sys.argv[1:]

    # Parse --model flag
    models_to_run = ["tinyllama", "gemma2:2b"]
    if '--model' in args:
        idx = args.index('--model')
        if idx + 1 < len(args):
            models_to_run = [args[idx + 1]]

    # Source DB
    src_db = os.path.expanduser('~/AgentsContext/brain/brain.db')
    if not os.path.exists(src_db):
        print(f"ERROR: Brain DB not found at {src_db}")
        sys.exit(1)

    # Copy DB to temp location
    tmp_dir, tmp_db = copy_brain_for_testing(src_db)
    os.environ['BRAIN_DB_DIR'] = tmp_dir
    print(f"[setup] Working on temp copy: {tmp_db}")

    # Extract domain context from the copy
    print("[setup] Extracting domain context from brain DB...")
    ctx = extract_domain_context(tmp_db)
    print(f"  Projects: {len(ctx['projects'])}")
    print(f"  Top titles: {len(ctx['top_titles'])}")
    print(f"  Keywords: {len(ctx['top_keywords'])}")
    print(f"  Node types: {len(ctx['type_counts'])}")
    if ctx['vocab_terms']:
        print(f"  Vocabulary: {len(ctx['vocab_terms'])}")

    # Build the rich system prompt
    system_prompt = build_system_prompt(ctx)
    print(f"\n[setup] System prompt length: {len(system_prompt)} chars")
    print("-" * 60)
    print(system_prompt[:500])
    print("..." if len(system_prompt) > 500 else "")
    print("-" * 60)

    # Test Ollama connection
    print("\n[setup] Testing Ollama connection...")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m['name'] for m in resp.json().get('models', [])]
        print(f"  Available models: {', '.join(available)}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        print(f"  And models are pulled: ollama pull tinyllama && ollama pull gemma2:2b")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # Verify requested models are available
    for model in models_to_run:
        # Ollama may list with :latest suffix
        found = any(model in m for m in available)
        if not found:
            print(f"  WARNING: Model '{model}' not found. Pull it: ollama pull {model}")

    # Initialize brain
    from servers.brain import Brain
    brain = Brain(tmp_db)

    # Run benchmarks
    all_results = {}
    for model in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"  Running HyDE benchmark: {model}")
        print(f"{'=' * 70}")

        result, hyde_cache, elapsed = run_benchmark(model, brain, system_prompt, tmp_db)
        all_results[model] = (result, hyde_cache, elapsed)
        print_results(model, result, hyde_cache, elapsed)

    # Summary comparison if both models ran
    if len(all_results) > 1:
        print()
        print("=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"  {'Condition':30s} {'NDCG@10':>10s} {'MRR':>10s} {'Passed':>10s} {'Time':>10s}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Baseline (no HyDE)':30s} {'0.204':>10s} {'0.202':>10s} {'34/104':>10s} {'—':>10s}")

        for model in models_to_run:
            result, _, elapsed = all_results[model]
            s = result['summary']
            agg = result['aggregate']
            ndcg = agg.get('ndcg@10', {}).get('mean', 0)
            mrr_val = agg.get('mrr', {}).get('mean', 0)
            label = f"HyDE + {model}"
            print(f"  {label:30s} {ndcg:>10.3f} {mrr_val:>10.3f} {s['passed']:>4d}/104  {elapsed:>8.1f}s")

        print()

    brain.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.")


if __name__ == '__main__':
    main()
