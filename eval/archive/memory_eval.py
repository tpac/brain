#!/usr/bin/env python3
"""Memory Eval: Community-First vs Current Recall.

Runs both recall algorithms on hand-crafted memory scenarios and compares:
- Community activation accuracy
- Critical node hit rate
- Behavioral principle activation
- Hub noise rate
- Per-mode and per-scenario breakdown

Usage:
    python3 eval/memory_eval.py                          # Run full eval
    python3 eval/memory_eval.py --mode pre-tool          # Run specific mode
    python3 eval/memory_eval.py --scenario a1_edit_recall # Run single scenario
    python3 eval/memory_eval.py --quick                  # Top-line numbers only
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.phase1_redesign import phase1_recall, baseline_recall
from eval.memory_scenarios import SCENARIOS, get_scenarios_by_mode, get_all_modes
from eval.memory_kpis import score_all_kpis, print_kpi_report, KPI_DEFINITIONS


# ── Scoring ──

def score_scenario(scenario, result, algorithm_name):
    """Score a single scenario's result against ground truth.

    Returns dict with per-metric scores and details.
    """
    result_ids = [r['id'] for r in result.get('results', [])]
    result_set = set(result_ids)
    top8_ids = set(result_ids[:8])

    scores = {}

    # 1. Community Activation (Jaccard similarity)
    if algorithm_name == 'recognition' and scenario.get('expected_communities'):
        activated = {c['id'] for c in result.get('activated_communities', [])}
        expected = set(scenario['expected_communities'])
        if expected:
            intersection = activated & expected
            union = activated | expected
            scores['community_activation'] = len(intersection) / len(union) if union else 0
            scores['_community_detail'] = {
                'expected': list(expected),
                'activated': list(activated),
                'hits': list(intersection),
            }
        else:
            scores['community_activation'] = 1.0  # No communities expected = pass
    else:
        scores['community_activation'] = None  # Not applicable for baseline

    # 2. Critical Node Hits (in top 25)
    expected_nodes = set(scenario.get('expected_nodes', []))
    if expected_nodes:
        hits = expected_nodes & result_set
        scores['node_hit_rate'] = len(hits) / len(expected_nodes)
        scores['_node_detail'] = {
            'expected': list(expected_nodes),
            'found': list(hits),
            'missed': list(expected_nodes - result_set),
            'positions': {nid: result_ids.index(nid) + 1
                         for nid in hits if nid in result_ids},
        }
    else:
        scores['node_hit_rate'] = None  # No nodes expected

    # 3. Critical Node Hits in top 8 (stricter)
    if expected_nodes:
        hits_top8 = expected_nodes & top8_ids
        scores['node_hit_rate_top8'] = len(hits_top8) / len(expected_nodes)
    else:
        scores['node_hit_rate_top8'] = None

    # 4. Behavioral Principle Activation
    expected_principles = set(scenario.get('expected_principles', []))
    if expected_principles:
        hits = expected_principles & result_set
        scores['principle_activation'] = len(hits) / len(expected_principles)
        scores['_principle_detail'] = {
            'expected': list(expected_principles),
            'found': list(hits),
            'missed': list(expected_principles - result_set),
        }
    else:
        scores['principle_activation'] = None

    # 5. Noise Rate (anti-expected hubs in top 8)
    anti_expected = set(scenario.get('anti_expected', []))
    if anti_expected:
        noise = anti_expected & top8_ids
        # Noise rate: 0 = no noise (good), 1 = all anti-expected present (bad)
        scores['noise_rate'] = len(noise) / min(len(anti_expected), 8)
        scores['_noise_detail'] = {
            'anti_expected': list(anti_expected),
            'found_in_top8': list(noise),
        }
    else:
        scores['noise_rate'] = 0

    # 6. Type diversity (how many unique types in top 8)
    top8_types = set(r['type'] for r in result.get('results', [])[:8])
    scores['type_diversity'] = len(top8_types) / 8 if result.get('results') else 0

    # 7. Recency check (for scenarios that care)
    if scenario.get('recency_matters'):
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        recent_count = 0
        for r in result.get('results', [])[:8]:
            # Check if results are actually from this week
            # This is a rough check — results don't always have created_at
            recent_count += 1  # TODO: check actual dates when available
        scores['recency_score'] = recent_count / 8

    # 8. Composite score
    weights = {
        'node_hit_rate': 0.30,
        'principle_activation': 0.25,
        'noise_rate': -0.20,  # Negative: noise hurts
        'community_activation': 0.15,
        'type_diversity': 0.10,
    }

    composite = 0.0
    weight_sum = 0.0
    for metric, weight in weights.items():
        val = scores.get(metric)
        if val is not None:
            if weight < 0:
                composite += weight * val  # noise_rate penalizes
            else:
                composite += weight * val
            weight_sum += abs(weight)

    scores['composite'] = composite / weight_sum if weight_sum > 0 else 0

    return scores


def run_eval(brain, scenarios, algorithms=None):
    """Run eval on all scenarios with both algorithms.

    Returns: {algorithm_name: {scenario_id: {scores, result, latency_ms}}}
    """
    if algorithms is None:
        algorithms = {
            'baseline': baseline_recall,
            'recognition': phase1_recall,
        }

    all_results = {}

    for algo_name, algo_fn in algorithms.items():
        all_results[algo_name] = {}

        for scenario in scenarios:
            # Reset fatigue before each call — prevents cross-scenario contamination
            if hasattr(brain, '_session_fatigue'):
                brain._session_fatigue = {}
            if hasattr(brain, '_fatigue_ctx') and brain._fatigue_ctx:
                brain._fatigue_ctx.fatigue = {}

            t0 = time.time()
            result = algo_fn(brain, scenario['query'], limit=25)
            latency = (time.time() - t0) * 1000

            scores = score_scenario(scenario, result, algo_name)

            all_results[algo_name][scenario['id']] = {
                'scores': scores,
                'latency_ms': round(latency, 1),
                'result_count': len(result.get('results', [])),
                'top5': [(r['id'], r['title'][:50], r['score'])
                         for r in result.get('results', [])[:5]],
                'activated_communities': [
                    (c['id'], c['title'][:40], c['score'])
                    for c in result.get('activated_communities', [])
                ],
            }

    return all_results


def aggregate_results(all_results, scenarios):
    """Compute aggregate metrics per algorithm."""
    aggregates = {}

    for algo_name, algo_data in all_results.items():
        metrics = {
            'node_hit_rate': [],
            'node_hit_rate_top8': [],
            'principle_activation': [],
            'noise_rate': [],
            'community_activation': [],
            'type_diversity': [],
            'composite': [],
            'latency_ms': [],
        }

        for sid, data in algo_data.items():
            scores = data['scores']
            for metric in metrics:
                if metric == 'latency_ms':
                    metrics[metric].append(data['latency_ms'])
                else:
                    val = scores.get(metric)
                    if val is not None:
                        metrics[metric].append(val)

        agg = {}
        for metric, values in metrics.items():
            if values:
                agg[metric] = {
                    'mean': round(sum(values) / len(values), 3),
                    'count': len(values),
                }
                if metric == 'latency_ms':
                    agg[metric]['p95'] = round(sorted(values)[int(len(values) * 0.95)], 1)

        aggregates[algo_name] = agg

    return aggregates


def aggregate_by_mode(all_results, scenarios):
    """Compute per-mode metrics."""
    mode_data = {}

    for scenario in scenarios:
        mode = scenario['mode']
        sid = scenario['id']
        mode_data.setdefault(mode, {'baseline': [], 'recognition': []})

        for algo in ['baseline', 'recognition']:
            if algo in all_results and sid in all_results[algo]:
                scores = all_results[algo][sid]['scores']
                mode_data[mode][algo].append(scores.get('composite', 0))

    result = {}
    for mode, algos in mode_data.items():
        result[mode] = {}
        for algo, composites in algos.items():
            if composites:
                result[mode][algo] = round(sum(composites) / len(composites), 3)
    return result


def win_loss_analysis(all_results, scenarios):
    """Per-scenario: which algorithm won?"""
    wins = {'baseline': 0, 'recognition': 0, 'tie': 0}
    details = []

    for scenario in scenarios:
        sid = scenario['id']
        b_composite = all_results.get('baseline', {}).get(sid, {}).get('scores', {}).get('composite', 0)
        c_composite = all_results.get('recognition', {}).get(sid, {}).get('scores', {}).get('composite', 0)

        diff = c_composite - b_composite
        if abs(diff) < 0.01:
            winner = 'tie'
            wins['tie'] += 1
        elif diff > 0:
            winner = 'recognition'
            wins['recognition'] += 1
        else:
            winner = 'baseline'
            wins['baseline'] += 1

        details.append({
            'id': sid,
            'name': scenario['name'],
            'mode': scenario['mode'],
            'baseline': round(b_composite, 3),
            'recognition': round(c_composite, 3),
            'delta': round(diff, 3),
            'winner': winner,
        })

    return wins, details


# ── Output ──

def print_results(all_results, scenarios, quick=False):
    """Print formatted results."""
    aggregates = aggregate_results(all_results, scenarios)
    mode_agg = aggregate_by_mode(all_results, scenarios)
    wins, win_details = win_loss_analysis(all_results, scenarios)

    print("\n" + "=" * 80)
    print("MEMORY EVAL: Community-First vs Baseline Recall")
    print("=" * 80)

    # Aggregate comparison
    print("\n── Aggregate Metrics ──\n")
    print(f"{'Metric':<25} {'Baseline':>10} {'Community':>10} {'Delta':>10}")
    print("-" * 55)

    for metric in ['node_hit_rate', 'node_hit_rate_top8', 'principle_activation',
                    'noise_rate', 'type_diversity', 'composite']:
        b_val = aggregates.get('baseline', {}).get(metric, {}).get('mean', '-')
        c_val = aggregates.get('recognition', {}).get(metric, {}).get('mean', '-')
        if isinstance(b_val, float) and isinstance(c_val, float):
            delta = c_val - b_val
            # For noise_rate, negative delta is good
            indicator = ''
            if metric == 'noise_rate':
                indicator = ' *' if delta < 0 else ''
            else:
                indicator = ' *' if delta > 0 else ''
            print(f"{metric:<25} {b_val:>10.3f} {c_val:>10.3f} {delta:>+10.3f}{indicator}")
        else:
            print(f"{metric:<25} {str(b_val):>10} {str(c_val):>10}")

    # Latency
    b_lat = aggregates.get('baseline', {}).get('latency_ms', {}).get('mean', '-')
    c_lat = aggregates.get('recognition', {}).get('latency_ms', {}).get('mean', '-')
    if isinstance(b_lat, float) and isinstance(c_lat, float):
        print(f"{'latency_ms':<25} {b_lat:>10.1f} {c_lat:>10.1f} {c_lat-b_lat:>+10.1f}")

    # Win/Loss
    print(f"\n── Win/Loss ──\n")
    print(f"  Community wins: {wins['recognition']}/{len(scenarios)}")
    print(f"  Baseline wins:  {wins['baseline']}/{len(scenarios)}")
    print(f"  Ties:           {wins['tie']}/{len(scenarios)}")

    # Per-mode breakdown
    print(f"\n── Per-Mode Composite ──\n")
    print(f"{'Mode':<20} {'Baseline':>10} {'Community':>10} {'Delta':>10}")
    print("-" * 50)
    for mode in sorted(mode_agg.keys()):
        b = mode_agg[mode].get('baseline', 0)
        c = mode_agg[mode].get('recognition', 0)
        print(f"{mode:<20} {b:>10.3f} {c:>10.3f} {c-b:>+10.3f}")

    if quick:
        return

    # Per-scenario details
    print(f"\n── Per-Scenario ──\n")
    print(f"{'ID':<20} {'Mode':<14} {'B.comp':>7} {'C.comp':>7} {'Delta':>7} {'Win':>10}")
    print("-" * 70)

    for d in sorted(win_details, key=lambda x: x['delta'], reverse=True):
        print(f"{d['id']:<20} {d['mode']:<14} {d['baseline']:>7.3f} "
              f"{d['recognition']:>7.3f} {d['delta']:>+7.3f} {d['winner']:>10}")

    # Detailed scenario analysis (top 5 per algorithm)
    print(f"\n── Detailed Top-5 per Scenario ──\n")
    for scenario in scenarios:
        sid = scenario['id']
        print(f"\n  {sid}: {scenario['name']}")
        print(f"  Query: \"{scenario['query'][:70]}\"")

        # Community activations
        c_data = all_results.get('recognition', {}).get(sid, {})
        if c_data.get('activated_communities'):
            print(f"  Communities:")
            for cid, title, score in c_data['activated_communities']:
                expected = cid in set(scenario.get('expected_communities', []))
                marker = ' HIT' if expected else ''
                print(f"    [{score:.3f}] {title}{marker}")

        for algo in ['baseline', 'recognition']:
            data = all_results.get(algo, {}).get(sid, {})
            label = 'B' if algo == 'baseline' else 'C'
            print(f"  {label} top5 (composite={data.get('scores', {}).get('composite', 0):.3f}):")
            for nid, title, score in data.get('top5', []):
                expected = nid in set(scenario.get('expected_nodes', []) +
                                       scenario.get('expected_principles', []))
                anti = nid in set(scenario.get('anti_expected', []))
                marker = ' HIT' if expected else (' NOISE' if anti else '')
                print(f"    [{score:.3f}] {title}{marker}")


def save_results(all_results, scenarios, output_dir=None):
    """Save results to timestamped JSON."""
    if output_dir is None:
        output_dir = os.path.join(ROOT, 'eval', 'results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    path = os.path.join(output_dir, f'memory_eval_{timestamp}.json')

    aggregates = aggregate_results(all_results, scenarios)
    mode_agg = aggregate_by_mode(all_results, scenarios)
    wins, win_details = win_loss_analysis(all_results, scenarios)

    output = {
        'timestamp': timestamp,
        'scenario_count': len(scenarios),
        'aggregates': aggregates,
        'mode_breakdown': mode_agg,
        'wins': wins,
        'win_details': win_details,
        'per_scenario': {
            algo: {
                sid: {
                    'scores': data['scores'],
                    'latency_ms': data['latency_ms'],
                    'result_count': data['result_count'],
                }
                for sid, data in algo_data.items()
            }
            for algo, algo_data in all_results.items()
        },
    }

    # Clean non-serializable items
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (set,)):
            return list(obj)
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    with open(path, 'w') as f:
        json.dump(_clean(output), f, indent=2, default=str)

    print(f"\nResults saved: {path}")
    return path


# ── Main ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory Eval: Community-First vs Baseline')
    parser.add_argument('--mode', help='Run only scenarios for this mode')
    parser.add_argument('--scenario', help='Run single scenario by ID')
    parser.add_argument('--quick', action='store_true', help='Top-line numbers only')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results')
    parser.add_argument('--kpis', action='store_true', help='Run KPI-focused report (baseline only)')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    # Select scenarios
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s['id'] == args.scenario]
        if not scenarios:
            print(f"Unknown scenario: {args.scenario}")
            sys.exit(1)
    elif args.mode:
        scenarios = get_scenarios_by_mode(args.mode)
        if not scenarios:
            print(f"Unknown mode: {args.mode}. Available: {get_all_modes()}")
            sys.exit(1)
    else:
        scenarios = SCENARIOS

    print(f"Running {len(scenarios)} scenarios on isolated brain copy...")

    with IsolatedBrain(cleanup=True) as env:
        print(f"Brain: {env.node_count()} nodes")

        if args.kpis:
            # KPI-focused report: score each scenario against KPIs
            # Run BOTH baseline and phase1v2 for comparison
            from eval.phase1_redesign import phase1_recall as p1r

            for algo_label, algo_fn in [('BASELINE', None), ('PHASE1V2', p1r)]:
                all_kpis = {}
                for scenario in scenarios:
                    if hasattr(env.brain, '_session_fatigue'):
                        env.brain._session_fatigue = {}
                    if hasattr(env.brain, '_fatigue_ctx') and env.brain._fatigue_ctx:
                        env.brain._fatigue_ctx.fatigue = {}

                    if algo_fn is None:
                        result = env.brain.recall(query=scenario['query'], limit=25, source='eval')
                        recalled = result.get('results', [])
                        normalized = [{
                            'id': r.get('id', ''), 'title': r.get('title', ''),
                            'type': r.get('type', ''), 'score': r.get('effective_activation', 0),
                        } for r in recalled]
                    else:
                        result = algo_fn(env.brain, scenario['query'], limit=25)
                        normalized = result.get('results', [])

                    all_kpis[scenario['id']] = score_all_kpis(scenario, normalized)

                print(f"\n{'#'*80}")
                print(f"# {algo_label}")
                print(f"{'#'*80}")
                print_kpi_report(all_kpis, scenarios)
        else:
            all_results = run_eval(env.brain, scenarios)
            print_results(all_results, scenarios, quick=args.quick)

        if not args.no_save and not args.kpis:
            save_results(all_results, scenarios)


if __name__ == '__main__':
    main()
