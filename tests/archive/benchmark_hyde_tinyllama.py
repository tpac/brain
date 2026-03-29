#!/usr/bin/env python3
"""Benchmark HyDE with TinyLlama via Ollama."""
import sys, os, time, shutil, json, requests
os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
sys.path.insert(0, '/Users/tpac/brain')

from tests.brain_test_base import copy_brain_for_testing

# Copy DB — keep existing v1.5 embeddings
tmp_dir, tmp_db = copy_brain_for_testing(os.path.expanduser('~/AgentsContext/brain/brain.db'))
os.environ['BRAIN_DB_DIR'] = tmp_dir

from servers.brain import Brain
from servers import embedder

brain = Brain(tmp_db)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"

def hyde_expand(query):
    """Use TinyLlama to generate a hypothetical document answering the query."""
    prompt = f"""Write a short technical note (2-3 sentences) that would be the answer to this question. Write it as if it's a fact from a knowledge base, not as a response to a question. Be specific and use technical terms.

Question: {query}

Technical note:"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150}
        }, timeout=30)
        result = resp.json()
        generated = result.get("response", "").strip()
        # Combine original query + generated text for embedding
        return query + " " + generated
    except Exception as e:
        print(f"[hyde] Error: {e}", file=sys.stderr)
        return query  # Fallback to original query

# Test that Ollama works
print("Testing Ollama connection...")
test = hyde_expand("test query")
print(f"HyDE test: {test[:100]}...")

# Monkey-patch recall to use HyDE
original_recall = brain.recall.__func__

def hyde_recall(self, query, types=None, limit=20, offset=0,
                include_archived=False, min_recency=0, project=None,
                session_id=None):
    """Recall with HyDE query expansion."""
    # Expand query via HyDE
    expanded = hyde_expand(query)

    # Replace the query embedding step inside recall
    # We need to intercept at the embedding level, not the recall level
    # Simplest approach: temporarily patch embedder.embed to embed the expanded text
    original_embed = embedder.embed

    first_call = [True]  # Use list for closure mutability
    def hyde_embed(text):
        if first_call[0]:
            first_call[0] = False
            result = original_embed(expanded)  # Embed the HyDE-expanded text
            return result
        return original_embed(text)

    embedder.embed = hyde_embed
    try:
        result = original_recall(self, query, types=types, limit=limit, offset=offset,
                                include_archived=include_archived, min_recency=min_recency,
                                project=project, session_id=session_id)
    finally:
        embedder.embed = original_embed
        first_call[0] = True

    return result

brain.recall = hyde_recall.__get__(brain, type(brain))

# Run eval
from tests.eval_runner import GoldenEvaluator

print("\n" + "="*60)
print("=== v1.5 + HyDE (TinyLlama 1.1B) ===")
print("="*60)

evaluator = GoldenEvaluator(brain)
t0 = time.time()
result = evaluator.run(verbose=False)
elapsed = time.time() - t0

s = result['summary']
agg = result['aggregate']

print(f"NDCG@10: {agg['ndcg@10']['mean']:.3f}")
print(f"MRR: {agg['mrr']['mean']:.3f}")
print(f"hit_rate@10: {agg['hit_rate@10']['mean']:.3f}")
print(f"Passed: {s['passed']}/{s['total']}")
print(f"Time: {elapsed:.1f}s ({elapsed/s['total']*1000:.0f}ms/case)")

print(f"\nBy category:")
for cat, metrics in sorted(result.get('by_category', {}).items()):
    mrr_data = metrics.get('mrr', {})
    ndcg_data = metrics.get('ndcg@10', {})
    count = mrr_data.get('count', 0)
    print(f"  {cat:25s}: NDCG={ndcg_data.get('mean', 0):.3f} MRR={mrr_data.get('mean', 0):.3f} (n={count})")

# Also print some example HyDE expansions for interesting queries
print("\n--- Example HyDE expansions ---")
example_queries = [
    "why did we separate the backend from the frontend",
    "how do we make money",
    "what is glo",
    "who runs the company and what does he do",
    "can mixins define __init__",
]
for q in example_queries:
    expanded = hyde_expand(q)
    print(f"\nQ: {q}")
    print(f"HyDE: {expanded[:200]}")

brain.close()
shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nDone.")
