#!/usr/bin/env python3
"""
brain — Evaluation Runner

Runs golden dataset test cases against a live brain and reports metrics.
Supports both golden eval (Approach 1) and snapshot regression (Approach 2).

Usage:
    python eval_runner.py [brain.db path]          # Run full eval
    python eval_runner.py --baseline [brain.db]     # Capture new baseline
    python eval_runner.py --regression [brain.db]   # Compare against baseline
    python eval_runner.py --report [brain.db]       # Full report with details
"""

import sqlite3
import json
import os
import sys
import time
import math
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent to path for brain import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.metrics import (
    compute_all_metrics, aggregate_metrics,
    precision_at_k, recall_at_k, mrr, ndcg_at_k,
    hit_rate_at_k, average_precision
)


# ═══════════════════════════════════════════════════════════════
# GOLDEN DATASET EVALUATOR
# ═══════════════════════════════════════════════════════════════

class GoldenEvaluator:
    """
    Run golden dataset test cases against a brain and compute retrieval metrics.
    """

    def __init__(self, brain, golden_path: str = None):
        """
        Args:
            brain: Brain instance (already connected)
            golden_path: Path to golden_dataset.json
        """
        self.brain = brain
        self.golden_path = golden_path or os.path.join(
            os.path.dirname(__file__), 'golden_dataset.json'
        )
        self.test_cases = self._load_golden()

    def _load_golden(self) -> List[Dict]:
        with open(self.golden_path) as f:
            return json.load(f)

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all golden test cases and compute metrics.

        Returns:
            Dict with:
                - case_results: per-test-case metrics
                - aggregate: aggregated metrics across all cases
                - by_category: aggregated by category
                - summary: pass/fail counts and overall health
                - timing: total and per-case timing
        """
        t0 = time.time()
        case_results = []
        failures = []
        crashes = []

        for tc in self.test_cases:
            tc_result = self._run_single(tc, verbose)
            case_results.append(tc_result)

            if tc_result.get('crashed'):
                crashes.append(tc['id'])
            elif not tc_result.get('passed'):
                failures.append(tc['id'])

        total_time = time.time() - t0

        # Aggregate metrics (only for cases with expected results)
        scorable = [r['metrics'] for r in case_results
                    if r.get('metrics') and r['metrics'].get('mrr') is not None]
        agg = aggregate_metrics(scorable) if scorable else {}

        # Aggregate by category
        by_category = {}
        for r in case_results:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = []
            if r.get('metrics') and r['metrics'].get('mrr') is not None:
                by_category[cat].append(r['metrics'])

        category_agg = {}
        for cat, metrics_list in by_category.items():
            if metrics_list:
                category_agg[cat] = aggregate_metrics(metrics_list)

        total = len(case_results)
        passed = total - len(failures) - len(crashes)

        return {
            'case_results': case_results,
            'aggregate': agg,
            'by_category': category_agg,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': len(failures),
                'crashed': len(crashes),
                'pass_rate': passed / total if total > 0 else 0,
                'failures': failures,
                'crashes': crashes,
            },
            'timing': {
                'total_seconds': total_time,
                'avg_per_case_ms': (total_time / total * 1000) if total > 0 else 0,
            },
        }

    def _run_single(self, tc: Dict, verbose: bool = False) -> Dict[str, Any]:
        """Run a single test case."""
        tc_id = tc['id']
        query = tc['query']
        category = tc['category']
        expected = tc.get('expected_relevant', {})
        min_hit_rate = tc.get('min_hit_rate_at_10', 0.0)
        type_filter = tc.get('type_filter')
        expect_no_crash = tc.get('expect_no_crash', False)

        t0 = time.time()

        try:
            # Run recall
            if type_filter:
                result = self.brain.recall(query, types=type_filter, limit=20)
            else:
                result = self.brain.recall(query, limit=20)

            elapsed_ms = (time.time() - t0) * 1000
            retrieved_ids = [r['id'] for r in result.get('results', [])]
            retrieved_scores = {r['id']: r.get('effective_activation', 0)
                               for r in result.get('results', [])}

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            if expect_no_crash:
                return {
                    'id': tc_id,
                    'category': category,
                    'query': query,
                    'crashed': True,
                    'error': str(e),
                    'passed': False,
                    'elapsed_ms': elapsed_ms,
                    'metrics': {},
                }
            else:
                return {
                    'id': tc_id,
                    'category': category,
                    'query': query,
                    'crashed': True,
                    'error': str(e),
                    'passed': False,
                    'elapsed_ms': elapsed_ms,
                    'metrics': {},
                }

        # Compute metrics
        metrics = {}
        if expected:
            # Convert expected_relevant to graded relevance dict
            relevance = {}
            for nid, grade in expected.items():
                relevance[nid] = grade

            metrics = compute_all_metrics(retrieved_ids, relevance, k_values=[5, 10, 20])
        else:
            # Edge case: just check it didn't crash and returned something
            metrics = {'result_count': len(retrieved_ids)}

        # Check pass/fail
        passed = True
        fail_reasons = []

        if expect_no_crash:
            # Just not crashing is a pass
            passed = True
        elif expected:
            # Check hit rate threshold
            actual_hit_rate = hit_rate_at_k(retrieved_ids, set(expected.keys()), 10)
            if actual_hit_rate < min_hit_rate:
                passed = False
                fail_reasons.append(
                    f'hit_rate@10={actual_hit_rate:.2f} < min={min_hit_rate:.2f}'
                )

        # Type filter validation
        if type_filter and result.get('results'):
            wrong_types = [r for r in result['results'] if r.get('type') not in type_filter]
            if wrong_types:
                passed = False
                fail_reasons.append(f'{len(wrong_types)} results with wrong type')

        if verbose and not passed:
            print(f'  FAIL [{tc_id}] {query[:50]}')
            for reason in fail_reasons:
                print(f'    - {reason}')

        return {
            'id': tc_id,
            'category': category,
            'query': query,
            'description': tc.get('description', ''),
            'passed': passed,
            'crashed': False,
            'fail_reasons': fail_reasons,
            'elapsed_ms': elapsed_ms,
            'metrics': metrics,
            'retrieved_count': len(retrieved_ids),
            'retrieved_ids': retrieved_ids[:10],  # Keep top 10 for debugging
            'retrieved_scores': {k: v for k, v in list(retrieved_scores.items())[:10]},
            'intent': result.get('intent', 'unknown'),
        }


# ═══════════════════════════════════════════════════════════════
# SNAPSHOT REGRESSION SYSTEM
# ═══════════════════════════════════════════════════════════════

class SnapshotRegression:
    """
    Capture and compare recall baselines for regression detection.

    Workflow:
      1. capture_baseline() — save current recall results as baseline
      2. check_regression() — compare current results against baseline
    """

    BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baselines')

    def __init__(self, brain):
        self.brain = brain
        os.makedirs(self.BASELINE_DIR, exist_ok=True)

    def _get_query_suite(self) -> List[Dict[str, Any]]:
        """
        Load golden dataset queries as the regression query suite.
        Only use queries with expected_relevant (scorable).
        """
        golden_path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
        with open(golden_path) as f:
            cases = json.load(f)

        return [c for c in cases if c.get('expected_relevant')]

    def capture_baseline(self, label: str = 'current') -> Dict[str, Any]:
        """
        Capture current recall results as a baseline snapshot.

        Args:
            label: Label for this baseline (e.g., 'v3.0.0', 'pre-refactor')

        Returns:
            Baseline data dict (also saved to disk)
        """
        query_suite = self._get_query_suite()
        snapshot = {
            'label': label,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'query_count': len(query_suite),
            'results': {},
        }

        for tc in query_suite:
            query = tc['query']
            tc_id = tc['id']
            type_filter = tc.get('type_filter')

            try:
                if type_filter:
                    result = self.brain.recall(query, types=type_filter, limit=20)
                else:
                    result = self.brain.recall(query, limit=20)

                retrieved = result.get('results', [])
                snapshot['results'][tc_id] = {
                    'query': query,
                    'node_ids': [r['id'] for r in retrieved],
                    'scores': [r.get('effective_activation', 0) for r in retrieved],
                    'types': [r.get('type', '') for r in retrieved],
                    'intent': result.get('intent', 'unknown'),
                }

            except Exception as e:
                snapshot['results'][tc_id] = {
                    'query': query,
                    'error': str(e),
                    'node_ids': [],
                    'scores': [],
                }

        # Save to disk
        filename = f'baseline_{label}_{int(time.time())}.json'
        filepath = os.path.join(self.BASELINE_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

        # Also save as 'latest'
        latest_path = os.path.join(self.BASELINE_DIR, 'baseline_latest.json')
        with open(latest_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f'[regression] Baseline captured: {len(query_suite)} queries → {filepath}')
        return snapshot

    def check_regression(self, baseline_path: str = None,
                        score_drop_threshold: float = 0.05,
                        rank_shift_threshold: int = 3) -> Dict[str, Any]:
        """
        Compare current recall against a baseline. Detect regressions.

        Args:
            baseline_path: Path to baseline JSON (default: latest)
            score_drop_threshold: Flag if mean score drops by this fraction (0.05 = 5%)
            rank_shift_threshold: Flag if a relevant node drops by this many rank positions

        Returns:
            Dict with regression report
        """
        if not baseline_path:
            baseline_path = os.path.join(self.BASELINE_DIR, 'baseline_latest.json')

        if not os.path.exists(baseline_path):
            return {
                'status': 'no_baseline',
                'message': 'No baseline found. Run --baseline first.',
            }

        with open(baseline_path) as f:
            baseline = json.load(f)

        query_suite = self._get_query_suite()
        regressions = []
        improvements = []
        stable = []

        for tc in query_suite:
            tc_id = tc['id']
            query = tc['query']
            type_filter = tc.get('type_filter')

            baseline_data = baseline.get('results', {}).get(tc_id)
            if not baseline_data or baseline_data.get('error'):
                continue

            # Run current recall
            try:
                if type_filter:
                    result = self.brain.recall(query, types=type_filter, limit=20)
                else:
                    result = self.brain.recall(query, limit=20)

                current_ids = [r['id'] for r in result.get('results', [])]
                current_scores = [r.get('effective_activation', 0) for r in result.get('results', [])]
            except Exception:
                regressions.append({
                    'query_id': tc_id,
                    'query': query,
                    'type': 'crash',
                    'detail': 'Current recall crashed',
                })
                continue

            baseline_ids = baseline_data.get('node_ids', [])
            baseline_scores = baseline_data.get('scores', [])

            # Check 1: Score distribution shift
            if baseline_scores and current_scores:
                baseline_mean = sum(baseline_scores[:10]) / min(len(baseline_scores), 10)
                current_mean = sum(current_scores[:10]) / min(len(current_scores), 10)

                if baseline_mean > 0:
                    pct_change = (current_mean - baseline_mean) / baseline_mean

                    if pct_change < -score_drop_threshold:
                        regressions.append({
                            'query_id': tc_id,
                            'query': query,
                            'type': 'score_drop',
                            'baseline_mean': round(baseline_mean, 4),
                            'current_mean': round(current_mean, 4),
                            'pct_change': round(pct_change * 100, 2),
                        })
                    elif pct_change > score_drop_threshold:
                        improvements.append({
                            'query_id': tc_id,
                            'query': query,
                            'type': 'score_improvement',
                            'baseline_mean': round(baseline_mean, 4),
                            'current_mean': round(current_mean, 4),
                            'pct_change': round(pct_change * 100, 2),
                        })
                    else:
                        stable.append(tc_id)

            # Check 2: Rank shifts for expected relevant nodes
            expected = tc.get('expected_relevant', {})
            for node_id in expected:
                baseline_rank = _find_rank(baseline_ids, node_id)
                current_rank = _find_rank(current_ids, node_id)

                if baseline_rank is not None and current_rank is not None:
                    shift = current_rank - baseline_rank
                    if shift > rank_shift_threshold:
                        regressions.append({
                            'query_id': tc_id,
                            'query': query,
                            'type': 'rank_drop',
                            'node_id': node_id[:8],
                            'baseline_rank': baseline_rank + 1,
                            'current_rank': current_rank + 1,
                            'shift': shift,
                        })
                elif baseline_rank is not None and current_rank is None:
                    regressions.append({
                        'query_id': tc_id,
                        'query': query,
                        'type': 'node_lost',
                        'node_id': node_id[:8],
                        'baseline_rank': baseline_rank + 1,
                        'detail': 'Node was in results but now missing',
                    })

            # Check 3: Result set overlap (Jaccard similarity)
            baseline_set = set(baseline_ids[:10])
            current_set = set(current_ids[:10])
            if baseline_set:
                jaccard = len(baseline_set & current_set) / len(baseline_set | current_set)
                if jaccard < 0.5:
                    regressions.append({
                        'query_id': tc_id,
                        'query': query,
                        'type': 'result_set_shift',
                        'jaccard_similarity': round(jaccard, 3),
                        'detail': f'Top-10 results changed significantly (Jaccard={jaccard:.3f})',
                    })

        has_regressions = len(regressions) > 0

        return {
            'status': 'regression_detected' if has_regressions else 'stable',
            'baseline_label': baseline.get('label', 'unknown'),
            'baseline_timestamp': baseline.get('timestamp', 'unknown'),
            'queries_checked': len(query_suite),
            'regressions': regressions,
            'improvements': improvements,
            'stable_count': len(stable),
            'regression_count': len(regressions),
            'improvement_count': len(improvements),
        }


def _find_rank(id_list: List[str], target_id: str) -> Optional[int]:
    """Find 0-based rank of target_id in list, or None if not found."""
    try:
        return id_list.index(target_id)
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════
# TERMINAL REPORTER
# ═══════════════════════════════════════════════════════════════

def print_golden_report(result: Dict[str, Any]):
    """Print a human-readable golden eval report to terminal."""
    summary = result['summary']
    timing = result['timing']

    print()
    print('=' * 70)
    print('  brain GOLDEN DATASET EVALUATION REPORT')
    print('=' * 70)
    print()

    # Summary
    total = summary['total']
    passed = summary['passed']
    failed = summary['failed']
    crashed = summary['crashed']
    rate = summary['pass_rate']

    status = 'PASS' if rate >= 0.8 else 'WARN' if rate >= 0.6 else 'FAIL'
    print(f'  Status:  {status}')
    print(f'  Total:   {total} test cases')
    print(f'  Passed:  {passed} ({rate:.0%})')
    if failed > 0:
        print(f'  Failed:  {failed}')
    if crashed > 0:
        print(f'  Crashed: {crashed}')
    print(f'  Time:    {timing["total_seconds"]:.2f}s ({timing["avg_per_case_ms"]:.0f}ms/case)')
    print()

    # Aggregate metrics
    agg = result.get('aggregate', {})
    if agg:
        print('  ─── Aggregate Metrics ───')
        for metric in ['mrr', 'ndcg@10', 'precision@5', 'recall@10', 'hit_rate@10']:
            if metric in agg:
                m = agg[metric]
                print(f'    {metric:>16s}: {m["mean"]:.3f} (min={m["min"]:.3f} max={m["max"]:.3f} std={m["std"]:.3f})')
        print()

    # By category
    cat_agg = result.get('by_category', {})
    if cat_agg:
        print('  ─── By Category ───')
        for cat, metrics in sorted(cat_agg.items()):
            mrr_data = metrics.get('mrr', {})
            ndcg_data = metrics.get('ndcg@10', {})
            count = mrr_data.get('count', 0)
            mrr_mean = mrr_data.get('mean', 0)
            ndcg_mean = ndcg_data.get('mean', 0)
            print(f'    {cat:>20s}: MRR={mrr_mean:.3f}  NDCG@10={ndcg_mean:.3f}  (n={count})')
        print()

    # Failures
    if summary['failures']:
        print('  ─── Failures ───')
        for r in result['case_results']:
            if not r['passed'] and not r['crashed']:
                reasons = ', '.join(r.get('fail_reasons', ['unknown']))
                print(f'    FAIL [{r["id"]}] {r["query"][:40]}')
                print(f'         {reasons}')
        print()

    if summary['crashes']:
        print('  ─── Crashes ───')
        for r in result['case_results']:
            if r.get('crashed'):
                print(f'    CRASH [{r["id"]}] {r.get("error", "unknown")}')
        print()

    print('=' * 70)
    print()


def print_regression_report(result: Dict[str, Any]):
    """Print human-readable regression report."""
    print()
    print('=' * 70)
    print('  brain REGRESSION CHECK REPORT')
    print('=' * 70)
    print()

    status = result['status']
    if status == 'no_baseline':
        print(f'  {result["message"]}')
        print()
        return

    status_label = 'REGRESSION DETECTED' if status == 'regression_detected' else 'STABLE'
    print(f'  Status:     {status_label}')
    print(f'  Baseline:   {result["baseline_label"]} ({result["baseline_timestamp"]})')
    print(f'  Queries:    {result["queries_checked"]}')
    print(f'  Stable:     {result["stable_count"]}')
    print(f'  Regressions: {result["regression_count"]}')
    print(f'  Improvements: {result["improvement_count"]}')
    print()

    if result['regressions']:
        print('  ─── Regressions ───')
        for reg in result['regressions']:
            rtype = reg['type']
            query = reg.get('query', '')[:40]
            if rtype == 'score_drop':
                print(f'    SCORE DROP [{reg["query_id"]}] {query}')
                print(f'      baseline={reg["baseline_mean"]:.4f} → current={reg["current_mean"]:.4f} ({reg["pct_change"]:+.1f}%)')
            elif rtype == 'rank_drop':
                print(f'    RANK DROP [{reg["query_id"]}] node={reg["node_id"]}')
                print(f'      rank {reg["baseline_rank"]} → {reg["current_rank"]} (shifted {reg["shift"]})')
            elif rtype == 'node_lost':
                print(f'    NODE LOST [{reg["query_id"]}] node={reg["node_id"]}')
                print(f'      was at rank {reg["baseline_rank"]}, now missing from results')
            elif rtype == 'result_set_shift':
                print(f'    SET SHIFT [{reg["query_id"]}] {query}')
                print(f'      Jaccard similarity: {reg["jaccard_similarity"]:.3f}')
            elif rtype == 'crash':
                print(f'    CRASH [{reg["query_id"]}] {reg.get("detail", "")}')
        print()

    if result['improvements']:
        print('  ─── Improvements ───')
        for imp in result['improvements']:
            query = imp.get('query', '')[:40]
            print(f'    IMPROVED [{imp["query_id"]}] {query}')
            print(f'      baseline={imp["baseline_mean"]:.4f} → current={imp["current_mean"]:.4f} ({imp["pct_change"]:+.1f}%)')
        print()

    print('=' * 70)
    print()


# ═══════════════════════════════════════════════════════════════
# JSON REPORT SAVER
# ═══════════════════════════════════════════════════════════════

def save_json_report(result: Dict[str, Any], output_path: str):
    """Save evaluation results as JSON for programmatic use."""
    # Make a serializable copy (strip large data)
    report = copy.deepcopy(result)

    # Trim per-case data for readability
    for cr in report.get('case_results', []):
        # Keep only top 5 retrieved IDs
        cr['retrieved_ids'] = cr.get('retrieved_ids', [])[:5]
        cr.pop('retrieved_scores', None)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[eval] JSON report saved → {output_path}')


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def find_brain_db(args: List[str]) -> str:
    """Find brain.db path from args or standard locations."""
    # Check for explicit path in args
    for arg in args:
        if arg.endswith('.db') and os.path.exists(arg):
            return arg

    # Standard search paths
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'brain.db'),
        os.path.expanduser('~/Documents/Claude/AgentsContext/brain/brain.db'),
        os.environ.get('BRAIN_DB_DIR', ''),
    ]

    for p in search_paths:
        if p and os.path.exists(p):
            return p

    return None


def main():
    args = sys.argv[1:]

    # Find brain.db
    db_path = find_brain_db(args)
    if not db_path:
        print('ERROR: brain.db not found. Pass path as argument.')
        sys.exit(1)

    print(f'[eval] Using brain.db: {db_path}')

    # Import and initialize brain
    from servers.brain import Brain
    brain = Brain(db_path)

    if '--baseline' in args:
        label = 'manual'
        for i, arg in enumerate(args):
            if arg == '--label' and i + 1 < len(args):
                label = args[i + 1]
        reg = SnapshotRegression(brain)
        reg.capture_baseline(label)

    elif '--regression' in args:
        reg = SnapshotRegression(brain)
        result = reg.check_regression()
        print_regression_report(result)

        # Save JSON too
        json_path = os.path.join(os.path.dirname(__file__), 'results', 'regression_report.json')
        save_json_report(result, json_path)

        if result['status'] == 'regression_detected':
            sys.exit(1)

    else:
        # Full golden eval (default)
        evaluator = GoldenEvaluator(brain)
        verbose = '--verbose' in args or '-v' in args
        result = evaluator.run(verbose=verbose)

        # Print terminal report
        print_golden_report(result)

        # Save JSON report
        json_path = os.path.join(os.path.dirname(__file__), 'results', 'golden_eval.json')
        save_json_report(result, json_path)

        # Also capture baseline if this is first run
        baseline_path = os.path.join(SnapshotRegression.BASELINE_DIR, 'baseline_latest.json')
        if not os.path.exists(baseline_path):
            print('[eval] No baseline exists — capturing initial baseline...')
            reg = SnapshotRegression(brain)
            reg.capture_baseline('initial')

        # Exit with error code if too many failures
        if result['summary']['pass_rate'] < 0.6:
            sys.exit(1)

    brain.close()


if __name__ == '__main__':
    main()
