#!/usr/bin/env python3
"""Compare retrieval quality: baseline (cosine+Haiku) vs trained models (B/C/C2).

Generates test queries from S1R traces (real recall events with known ground truth),
then measures how well each approach surfaces the right nodes.

Usage:
    # Generate test queries from traces (run once)
    python3 eval/compare_models.py --generate-queries --output eval/test_queries.json

    # Run baseline (current system) against test queries
    python3 eval/compare_models.py --eval-baseline --queries eval/test_queries.json

    # Run trained model against test queries (after downloading adapter from Colab)
    python3 eval/compare_models.py --eval-model brain-adapter-B --queries eval/test_queries.json

    # Compare all results
    python3 eval/compare_models.py --compare eval/results_*.json
"""

import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def generate_test_queries(brain, output_path, n=100):
    """Extract test queries from S1R traces with ground truth.

    Each query has: the original query string, and which nodes the judge
    actually selected (ground truth for retrieval quality).
    """
    # Read S1R traces — O events (recall candidates) and K events (judge selections)
    traces = brain._trace_dal.conn.execute('''
        SELECT chain_id, event_type, ref_type, ref_id, summary, metadata
        FROM trace_events
        WHERE scale = 's1' AND event_type IN ('O', 'K')
        AND ref_type IN ('recall', 'surface_selected')
        ORDER BY created_at DESC
        LIMIT 2000
    ''').fetchall()

    # Group by chain_id
    chains = defaultdict(dict)
    for chain_id, event_type, ref_type, ref_id, summary, metadata in traces:
        if event_type == 'O' and ref_type == 'recall':
            meta = json.loads(metadata) if metadata else {}
            chains[chain_id]['query'] = meta.get('query', '')
            candidates = []
            for c in meta.get('candidates', []):
                if isinstance(c, str) and '|' in c:
                    parts = c.split('|')
                    if len(parts) >= 4:
                        candidates.append({
                            'id': parts[0],
                            'title': '|'.join(parts[1:-2]),
                            'score': float(parts[-2]) if parts[-2] else 0,
                            'type': parts[-1],
                        })
            chains[chain_id]['candidates'] = candidates

        elif event_type == 'K' and ref_type == 'surface_selected':
            try:
                selected_ids = json.loads(ref_id) if ref_id else []
                if isinstance(selected_ids, list):
                    chains[chain_id]['selected'] = selected_ids
            except (json.JSONDecodeError, TypeError):
                pass

    # Build test queries: must have both query and selected
    queries = []
    seen_queries = set()
    for chain_id, data in chains.items():
        query = data.get('query', '').strip()
        selected = data.get('selected', [])
        candidates = data.get('candidates', [])

        if not query or not selected or query in seen_queries:
            continue
        seen_queries.add(query)

        # Get titles for selected nodes
        selected_titles = {}
        for sid in selected:
            row = brain.conn.execute(
                'SELECT title, type FROM nodes WHERE id = ?', (sid,)).fetchone()
            if row:
                selected_titles[sid] = {'title': row[0], 'type': row[1]}

        queries.append({
            'query': query,
            'selected_ids': selected,
            'selected_titles': selected_titles,
            'candidate_count': len(candidates),
            'chain_id': chain_id,
        })

        if len(queries) >= n:
            break

    # Save
    with open(output_path, 'w') as f:
        json.dump({'queries': queries, 'count': len(queries)}, f, indent=2)

    print('Generated %d test queries → %s' % (len(queries), output_path))
    return queries


def eval_baseline(brain, queries):
    """Run current recall system (cosine + Haiku judge) on test queries."""
    results = []
    for i, q in enumerate(queries):
        t0 = time.time()
        recall_result = brain.recall(query=q['query'], limit=25, source='eval')
        latency = (time.time() - t0) * 1000

        recalled = recall_result.get('results', recall_result) if isinstance(recall_result, dict) else recall_result
        recalled_ids = [r.get('id', '') for r in recalled]

        # How many of the ground truth nodes were in the recall results?
        ground_truth = set(q['selected_ids'])
        recalled_set = set(recalled_ids)
        hits = ground_truth & recalled_set

        # Positions of ground truth nodes in recall ranking
        positions = {}
        for gt_id in ground_truth:
            if gt_id in recalled_ids:
                positions[gt_id] = recalled_ids.index(gt_id) + 1
            else:
                positions[gt_id] = -1  # not found

        results.append({
            'query': q['query'],
            'ground_truth': list(ground_truth),
            'hits': len(hits),
            'total_gt': len(ground_truth),
            'precision': len(hits) / len(ground_truth) if ground_truth else 0,
            'positions': positions,
            'recalled_count': len(recalled_ids),
            'latency_ms': round(latency, 1),
        })

        if (i + 1) % 20 == 0:
            avg_prec = sum(r['precision'] for r in results) / len(results)
            print('  %d/%d queries, avg precision: %.2f' % (i + 1, len(queries), avg_prec))

    return results


def eval_trained_model(adapter_path, queries):
    """Run trained Gemma model on test queries.

    The model receives the query and should output relevant node IDs/titles.
    We compare its output against ground truth.
    """
    try:
        from unsloth import FastModel
    except ImportError:
        print('ERROR: unsloth not installed. Run: pip install unsloth')
        return []

    print('Loading model from %s...' % adapter_path)
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)

    results = []
    for i, q in enumerate(queries):
        t0 = time.time()
        messages = [{"role": "user", "content":
            "Based on your knowledge of the brain graph, what nodes are most "
            "relevant to this query? List the node IDs and titles.\n\n"
            "Query: %s" % q['query']}]

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:],
                                     skip_special_tokens=True)
        latency = (time.time() - t0) * 1000

        # Extract node IDs from response (look for id:XXXXXXXX patterns)
        import re
        found_ids = re.findall(r'id:([a-z0-9]{6,12})', response)

        # Match against ground truth
        ground_truth = set(q['selected_ids'])
        # Try prefix matching (model might output short IDs)
        hits = set()
        for fid in found_ids:
            for gt_id in ground_truth:
                if gt_id.startswith(fid) or fid.startswith(gt_id[:8]):
                    hits.add(gt_id)

        results.append({
            'query': q['query'],
            'ground_truth': list(ground_truth),
            'model_response': response[:500],
            'found_ids': found_ids,
            'hits': len(hits),
            'total_gt': len(ground_truth),
            'precision': len(hits) / len(ground_truth) if ground_truth else 0,
            'latency_ms': round(latency, 1),
        })

        if (i + 1) % 10 == 0:
            avg_prec = sum(r['precision'] for r in results) / len(results)
            print('  %d/%d queries, avg precision: %.2f' % (i + 1, len(queries), avg_prec))

    return results


def score_results(results, label):
    """Compute aggregate scores for a result set."""
    if not results:
        return {}

    precisions = [r['precision'] for r in results]
    latencies = [r['latency_ms'] for r in results]
    total_hits = sum(r['hits'] for r in results)
    total_gt = sum(r['total_gt'] for r in results)

    return {
        'label': label,
        'queries': len(results),
        'avg_precision': round(sum(precisions) / len(precisions), 3),
        'median_precision': round(sorted(precisions)[len(precisions) // 2], 3),
        'total_hits': total_hits,
        'total_ground_truth': total_gt,
        'hit_rate': round(total_hits / total_gt, 3) if total_gt else 0,
        'avg_latency_ms': round(sum(latencies) / len(latencies), 1),
        'p95_latency_ms': round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
    }


def print_comparison(scores_list):
    """Pretty-print comparison table."""
    print('\n' + '=' * 80)
    print('MODEL COMPARISON')
    print('=' * 80)

    # Header
    labels = [s['label'] for s in scores_list]
    print('\n%-25s' % 'Metric', end='')
    for label in labels:
        print('%-20s' % label, end='')
    print()
    print('-' * (25 + 20 * len(labels)))

    # Rows
    metrics = [
        ('Avg Precision', 'avg_precision'),
        ('Median Precision', 'median_precision'),
        ('Hit Rate', 'hit_rate'),
        ('Total Hits / GT', None),
        ('Avg Latency (ms)', 'avg_latency_ms'),
        ('P95 Latency (ms)', 'p95_latency_ms'),
    ]

    for label, key in metrics:
        print('%-25s' % label, end='')
        for s in scores_list:
            if key:
                val = s.get(key, '?')
                if isinstance(val, float):
                    print('%-20s' % ('%.3f' % val), end='')
                else:
                    print('%-20s' % str(val), end='')
            else:
                print('%-20s' % ('%d / %d' % (s['total_hits'], s['total_ground_truth'])), end='')
        print()

    # Winner
    best = max(scores_list, key=lambda s: s.get('avg_precision', 0))
    print('\n🏆 Best avg precision: %s (%.3f)' % (best['label'], best['avg_precision']))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-queries', action='store_true',
                        help='Generate test queries from S1R traces')
    parser.add_argument('--eval-baseline', action='store_true',
                        help='Run baseline (cosine+judge) on queries')
    parser.add_argument('--eval-model', help='Run trained model on queries (path to adapter)')
    parser.add_argument('--queries', help='Path to test queries JSON')
    parser.add_argument('--output', help='Output path for queries or results')
    parser.add_argument('--compare', nargs='+', help='Compare result JSON files')
    parser.add_argument('--limit', type=int, default=50, help='Max queries to eval')
    args = parser.parse_args()

    if args.compare:
        scores = []
        for path in args.compare:
            with open(path) as f:
                data = json.load(f)
            scores.append(score_results(data['results'], data.get('label', path)))
        print_comparison(scores)
        return

    os.environ.setdefault('BRAIN_DB_DIR',
                          os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain'))

    if args.generate_queries:
        from servers.brain import Brain
        brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))
        output = args.output or 'eval/test_queries.json'
        generate_test_queries(brain, output, n=args.limit)
        return

    if not args.queries:
        print('ERROR: --queries required for eval')
        return

    with open(args.queries) as f:
        queries = json.load(f)['queries'][:args.limit]
    print('Loaded %d test queries' % len(queries))

    if args.eval_baseline:
        from servers.brain import Brain
        brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))
        print('\nRunning baseline eval (cosine recall)...')
        results = eval_baseline(brain, queries)
        scores = score_results(results, 'Baseline (cosine)')

        output = args.output or 'eval/results_baseline.json'
        with open(output, 'w') as f:
            json.dump({'label': 'Baseline (cosine)', 'results': results, 'scores': scores}, f, indent=2)
        print('\nBaseline scores:')
        print('  Avg precision: %.3f' % scores['avg_precision'])
        print('  Hit rate: %.3f' % scores['hit_rate'])
        print('  Avg latency: %.1f ms' % scores['avg_latency_ms'])
        print('Saved → %s' % output)

    elif args.eval_model:
        print('\nRunning trained model eval: %s' % args.eval_model)
        results = eval_trained_model(args.eval_model, queries)
        label = 'Model %s' % os.path.basename(args.eval_model)
        scores = score_results(results, label)

        output = args.output or 'eval/results_%s.json' % os.path.basename(args.eval_model)
        with open(output, 'w') as f:
            json.dump({'label': label, 'results': results, 'scores': scores}, f, indent=2)
        print('\n%s scores:' % label)
        print('  Avg precision: %.3f' % scores['avg_precision'])
        print('  Hit rate: %.3f' % scores['hit_rate'])
        print('  Avg latency: %.1f ms' % scores['avg_latency_ms'])
        print('Saved → %s' % output)


if __name__ == '__main__':
    main()
