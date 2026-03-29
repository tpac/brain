#!/usr/bin/env python3
"""Benchmark v1.5 + cross-encoder reranker."""
import sys, os, time, shutil
os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
sys.path.insert(0, '/Users/tpac/brain')

from tests.brain_test_base import copy_brain_for_testing

# Copy DB — keep existing v1.5 embeddings untouched
tmp_dir, tmp_db = copy_brain_for_testing(os.path.expanduser('~/AgentsContext/brain/brain.db'))
os.environ['BRAIN_DB_DIR'] = tmp_dir
print(f'Working on temp copy: {tmp_db}')

from servers.brain import Brain
brain = Brain(tmp_db)

# Verify embeddings exist
count = brain.conn.execute('SELECT COUNT(*) FROM node_embeddings').fetchone()[0]
print(f'Existing v1.5 embeddings: {count}')

from sentence_transformers import CrossEncoder
from tests.eval_runner import GoldenEvaluator

# Monkey-patch recall to add reranking AFTER the normal pipeline
original_recall = brain.recall.__func__

def make_reranked_recall(reranker_model_name):
    """Create a patched recall function that adds cross-encoder reranking."""
    print(f'Loading reranker: {reranker_model_name}...')
    reranker = CrossEncoder(reranker_model_name, device='cpu')
    print(f'Reranker loaded.')

    def reranked_recall(self, query, types=None, limit=20, offset=0,
                        include_archived=False, min_recency=0, project=None,
                        session_id=None):
        # Run normal recall with larger limit to get more candidates
        result = original_recall(self, query, types=types, limit=50, offset=offset,
                                include_archived=include_archived, min_recency=min_recency,
                                project=project, session_id=session_id)

        results = result.get('results', [])
        if not results or len(results) <= 1:
            return result

        # Build pairs for cross-encoder
        pairs = []
        for node in results:
            text = (node.get('title', '') + ' ' + node.get('content', ''))[:512]
            pairs.append((query, text))

        # Score with cross-encoder
        scores = reranker.predict(pairs)

        # Apply reranker scores and re-sort
        for i, node in enumerate(results):
            node['_reranker_score'] = float(scores[i])
            node['effective_activation'] = float(scores[i])

        results.sort(key=lambda x: -x.get('_reranker_score', 0))
        result['results'] = results[:limit]
        return result

    return reranked_recall

# Test three reranker models
models = [
    ('cross-encoder/ms-marco-MiniLM-L-6-v2', 'MiniLM-22M'),
    ('BAAI/bge-reranker-v2-m3', 'bge-v2-m3-278M'),
    ('Alibaba-NLP/gte-reranker-modernbert-base', 'gte-modernbert-149M'),
]

for model_name, label in models:
    print(f'\n{"="*60}')
    print(f'=== v1.5 + {label} ===')
    print(f'{"="*60}')

    try:
        patched = make_reranked_recall(model_name)
        brain.recall = patched.__get__(brain, type(brain))

        evaluator = GoldenEvaluator(brain)
        t0 = time.time()
        result = evaluator.run(verbose=False)
        elapsed = time.time() - t0

        s = result['summary']
        agg = result['aggregate']

        print(f'NDCG@10: {agg["ndcg@10"]["mean"]:.3f}')
        print(f'MRR: {agg["mrr"]["mean"]:.3f}')
        print(f'hit_rate@10: {agg["hit_rate@10"]["mean"]:.3f}')
        print(f'Passed: {s["passed"]}/{s["total"]}')
        print(f'Time: {elapsed:.1f}s ({elapsed/s["total"]*1000:.0f}ms/case)')

        print(f'\nBy category:')
        for cat, metrics in sorted(result.get('by_category', {}).items()):
            ndcg_val = metrics["ndcg@10"]["mean"] if isinstance(metrics["ndcg@10"], dict) else metrics["ndcg@10"]
            mrr_val = metrics["mrr"]["mean"] if isinstance(metrics["mrr"], dict) else metrics["mrr"]
            cnt = metrics["mrr"]["count"] if isinstance(metrics["mrr"], dict) else metrics.get("count", "?")
            print(f'  {cat:25s}: NDCG={ndcg_val:.3f} MRR={mrr_val:.3f} (n={cnt})')

    except Exception as e:
        print(f'ERROR with {label}: {e}')
        import traceback
        traceback.print_exc()

    # Restore original for next iteration
    brain.recall = original_recall.__get__(brain, type(brain))

brain.close()
shutil.rmtree(tmp_dir, ignore_errors=True)
print('\nDone. Temp cleaned up.')
