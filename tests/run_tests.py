#!/usr/bin/env python3
"""
brain — Unified Test Runner

Single entry point for all testing:
    python tests/run_tests.py                        # Full eval + regression
    python tests/run_tests.py --unit                  # Unit + recall + hook tests (133 tests)
    python tests/run_tests.py --unit --encode         # Run tests + encode findings to brain
    python tests/run_tests.py --unit --trends         # Show historical test trends
    python tests/run_tests.py --golden               # Golden dataset only
    python tests/run_tests.py --regression            # Regression check only
    python tests/run_tests.py --baseline [label]      # Capture new baseline
    python tests/run_tests.py --generate              # Regenerate golden dataset
    python tests/run_tests.py --metrics-test          # Self-test metrics module
    python tests/run_tests.py --verbose               # Verbose output
    python tests/run_tests.py [brain.db path]         # Use specific brain.db
"""

import sys
import os
import time

# Ensure parent is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_metrics_self_test():
    """Validate the metrics module itself with known inputs."""
    from tests.metrics import (
        precision_at_k, recall_at_k, mrr, ndcg_at_k,
        hit_rate_at_k, average_precision, compute_all_metrics
    )

    print('[self-test] Running metrics module self-test...')
    errors = []

    # Test 1: Perfect ranking
    retrieved = ['a', 'b', 'c']
    relevance = {'a': 2, 'b': 1, 'c': 1}
    assert abs(precision_at_k(retrieved, relevance, 3) - 1.0) < 1e-6, 'P@3 perfect'
    assert abs(recall_at_k(retrieved, relevance, 3) - 1.0) < 1e-6, 'R@3 perfect'
    assert abs(mrr(retrieved, relevance) - 1.0) < 1e-6, 'MRR perfect'
    assert abs(ndcg_at_k(retrieved, relevance, 3) - 1.0) < 1e-6, 'NDCG@3 perfect'

    # Test 2: No relevant results
    retrieved = ['x', 'y', 'z']
    relevance = {'a': 2, 'b': 1}
    assert abs(precision_at_k(retrieved, relevance, 3) - 0.0) < 1e-6, 'P@3 zero'
    assert abs(recall_at_k(retrieved, relevance, 3) - 0.0) < 1e-6, 'R@3 zero'
    assert abs(mrr(retrieved, relevance) - 0.0) < 1e-6, 'MRR zero'
    assert abs(hit_rate_at_k(retrieved, relevance, 3) - 0.0) < 1e-6, 'Hit@3 zero'

    # Test 3: Relevant at position 2
    retrieved = ['x', 'a', 'y']
    relevance = {'a': 2, 'b': 1}
    assert abs(mrr(retrieved, relevance) - 0.5) < 1e-6, 'MRR at pos 2'
    assert abs(precision_at_k(retrieved, relevance, 2) - 0.5) < 1e-6, 'P@2 half'
    assert abs(hit_rate_at_k(retrieved, relevance, 2) - 1.0) < 1e-6, 'Hit@2'

    # Test 4: NDCG with imperfect ranking
    retrieved = ['b', 'a']  # b=1, a=2 (should be a, b for ideal)
    relevance = {'a': 2, 'b': 1}
    ndcg = ndcg_at_k(retrieved, relevance, 2)
    assert 0 < ndcg < 1.0, f'NDCG imperfect ranking: {ndcg}'

    # Test 5: Empty inputs
    assert abs(precision_at_k([], {'a': 1}, 5) - 0.0) < 1e-6, 'Empty retrieved'
    assert abs(recall_at_k(['a'], {}, 5) - 1.0) < 1e-6, 'Empty relevant (vacuous)'
    assert abs(mrr([], {'a': 1}) - 0.0) < 1e-6, 'MRR empty'

    # Test 6: Average precision
    retrieved = ['a', 'x', 'b', 'y', 'c']
    relevant = {'a', 'b', 'c'}
    ap = average_precision(retrieved, relevant)
    expected_ap = (1/1 + 2/3 + 3/5) / 3  # P@1=1, P@3=2/3, P@5=3/5
    assert abs(ap - expected_ap) < 1e-6, f'AP: {ap} != {expected_ap}'

    # Test 7: compute_all_metrics integration
    metrics = compute_all_metrics(['a', 'b', 'c'], {'a': 2, 'b': 1}, k_values=[5, 10])
    assert 'mrr' in metrics
    assert 'ndcg@5' in metrics
    assert 'precision@10' in metrics

    print('[self-test] All metrics tests PASSED')
    return True


def find_brain_db(args):
    """Find brain.db from args or standard locations."""
    for arg in args:
        if arg.endswith('.db') and os.path.exists(arg):
            return arg

    brain_db_dir = os.environ.get('BRAIN_DB_DIR', '')
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'brain.db'),
        os.path.expanduser('~/AgentsContext/brain/brain.db'),
        os.path.expanduser('~/Documents/Claude/AgentsContext/brain/brain.db'),
        os.path.join(brain_db_dir, 'brain.db') if brain_db_dir else '',
    ]

    for p in search_paths:
        if p and os.path.exists(p):
            return p

    return None


def main():
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith('-')]
    positionals = [a for a in args if not a.startswith('-')]

    verbose = '--verbose' in flags or '-v' in flags

    # Unit test mode (test_core + test_recall_quality + test_hooks)
    if '--unit' in flags:
        import unittest
        print()
        print('=' * 70)
        print('  brain UNIT TEST SUITE')
        print('=' * 70)
        print()

        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromName('tests.test_core'))
        suite.addTests(loader.loadTestsFromName('tests.test_recall_quality'))
        suite.addTests(loader.loadTestsFromName('tests.test_hooks'))

        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        # Post-run reporting
        from tests.brain_test_base import get_run_summary

        summary = get_run_summary()
        if summary:
            print()
            print(f'[test] {summary["total"]} tests: '
                  f'{summary["passed"]} pass, {summary["failed"]} fail, '
                  f'{summary["errors"]} error ({summary["total_duration_ms"]/1000:.1f}s)')

        # Encode to brain if --encode flag
        if '--encode' in flags:
            print()
            print('[test] Encoding findings to brain...')
            db_path = find_brain_db(positionals)
            if db_path:
                from servers.brain import Brain
                brain = Brain(db_path)
                from tests.test_encoder import encode_run_to_brain
                enc_result = encode_run_to_brain(brain, verbose=verbose)
                brain.close()
                print(f'[test] Encoded {enc_result["encoded"]} findings')
            else:
                print('[test] WARNING: brain.db not found, skipping encoding')

        # Show trends if --trends flag
        if '--trends' in flags:
            from tests.test_encoder import print_trends_report
            print_trends_report()

        sys.exit(0 if result.wasSuccessful() else 1)

    # Show trends only
    if '--trends' in flags:
        from tests.test_encoder import print_trends_report
        print_trends_report()
        sys.exit(0)

    # Self-test mode
    if '--metrics-test' in flags:
        success = run_metrics_self_test()
        sys.exit(0 if success else 1)

    # Generate golden dataset
    if '--generate' in flags:
        from tests.generate_golden import generate_golden_dataset
        db_path = find_brain_db(positionals)
        if not db_path:
            print('ERROR: brain.db not found')
            sys.exit(1)
        output = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
        generate_golden_dataset(db_path, output)
        sys.exit(0)

    # Find brain.db
    db_path = find_brain_db(positionals)
    if not db_path:
        print('ERROR: brain.db not found. Pass path as argument.')
        print('Usage: python tests/run_tests.py [brain.db path]')
        sys.exit(1)

    print(f'[test] Brain: {db_path}')

    # Import brain
    from servers.brain import Brain
    brain = Brain(db_path)

    exit_code = 0
    t0 = time.time()

    from tests.eval_runner import (
        GoldenEvaluator, SnapshotRegression,
        print_golden_report, print_regression_report,
        save_json_report
    )

    # Baseline capture mode
    if '--baseline' in flags:
        label = positionals[0] if positionals and not positionals[0].endswith('.db') else 'manual'
        reg = SnapshotRegression(brain)
        reg.capture_baseline(label)
        brain.close()
        sys.exit(0)

    # Regression only
    if '--regression' in flags:
        reg = SnapshotRegression(brain)
        result = reg.check_regression()
        print_regression_report(result)

        json_path = os.path.join(os.path.dirname(__file__), 'results', 'regression_report.json')
        save_json_report(result, json_path)

        brain.close()
        sys.exit(1 if result['status'] == 'regression_detected' else 0)

    # Golden only
    if '--golden' in flags:
        evaluator = GoldenEvaluator(brain)
        result = evaluator.run(verbose=verbose)
        print_golden_report(result)

        json_path = os.path.join(os.path.dirname(__file__), 'results', 'golden_eval.json')
        save_json_report(result, json_path)

        brain.close()
        sys.exit(1 if result['summary']['pass_rate'] < 0.6 else 0)

    # Default: run everything
    print()
    print('=' * 70)
    print('  brain FULL TEST SUITE')
    print('=' * 70)

    # 1. Metrics self-test
    print()
    run_metrics_self_test()

    # 2. Golden evaluation
    print()
    print('[test] Running golden dataset evaluation...')
    evaluator = GoldenEvaluator(brain)
    golden_result = evaluator.run(verbose=verbose)
    print_golden_report(golden_result)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    json_path = os.path.join(os.path.dirname(__file__), 'results', 'golden_eval.json')
    save_json_report(golden_result, json_path)

    if golden_result['summary']['pass_rate'] < 0.6:
        exit_code = 1

    # 3. Regression check
    print()
    baseline_path = os.path.join(SnapshotRegression.BASELINE_DIR, 'baseline_latest.json')
    if os.path.exists(baseline_path):
        print('[test] Running regression check...')
        reg = SnapshotRegression(brain)
        reg_result = reg.check_regression()
        print_regression_report(reg_result)

        json_path = os.path.join(os.path.dirname(__file__), 'results', 'regression_report.json')
        save_json_report(reg_result, json_path)

        if reg_result['status'] == 'regression_detected':
            exit_code = 1
    else:
        print('[test] No baseline found — capturing initial baseline...')
        reg = SnapshotRegression(brain)
        reg.capture_baseline('initial')

    # Summary
    total_time = time.time() - t0
    print()
    print(f'[test] Full suite completed in {total_time:.2f}s')
    print(f'[test] Exit code: {exit_code}')

    brain.close()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
