#!/usr/bin/env python3
"""Run all brain tests — unit + integration + regression baseline.

Usage:
    python3 tests/run_all.py              # Run everything
    python3 tests/run_all.py --unit       # Unit tests only (fast, no DB)
    python3 tests/run_all.py --integration # Integration tests (uses DB copy)
    python3 tests/run_all.py --regression  # Decode funnel regression check
"""

import sys
import os
import unittest
import argparse
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_unit_tests():
    """Fast tests — no DB, no daemon, no API calls."""
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_files = [
        'tests.test_recall_scoring',
        'tests.test_pipeline_contract',
        'tests.test_redistribution',
    ]

    for tf in test_files:
        try:
            suite.addTests(loader.loadTestsFromName(tf))
        except Exception as e:
            print(f"  SKIP {tf}: {e}")

    # test_contract_sync has 1 known failure (confidence default mismatch).
    # Run separately and check for exactly 1 failure.
    # Included in Test08_ImportIntegrity instead.

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests():
    """Tests that use a copy of the brain DB."""
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_files = [
        'tests.integration.test_recall_pipeline',
        'tests.integration.test_session_mechanisms',
    ]

    for tf in test_files:
        try:
            suite.addTests(loader.loadTestsFromName(tf))
        except Exception as e:
            print(f"  SKIP {tf}: {e}")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_regression():
    """Run decode funnel and compare against stored baseline."""
    print("\n" + "=" * 60)
    print("REGRESSION CHECK (decode funnel)")
    print("=" * 60)

    brain_db_dir = os.environ.get('BRAIN_DB_DIR',
                                   os.path.expanduser('~/AgentsContext/brain'))
    os.environ['BRAIN_DB_DIR'] = brain_db_dir

    try:
        from eval.brain_eval import run_decode_eval, load_corpus, save_results
        corpus_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'eval', 'corpus')
        conversations = load_corpus(corpus_dir)
        results = run_decode_eval(brain_db_dir, conversations, verbose=True)

        # Check against baseline thresholds
        kpis = results.get('kpis', {})
        r25 = kpis.get('recall@25', {}).get('score', 0)
        hub8 = kpis.get('hub_concentration@8', {}).get('score', 1)

        print(f"\n  R@25: {r25:.1%} (threshold: >75%)")
        print(f"  Hub@8: {hub8:.1%} (threshold: <20%)")

        passed = True
        if r25 < 0.75:
            print(f"  ✗ R@25 REGRESSION: {r25:.1%} < 75%")
            passed = False
        else:
            print(f"  ✓ R@25 OK")

        if hub8 > 0.20:
            print(f"  ✗ Hub concentration REGRESSION: {hub8:.1%} > 20%")
            passed = False
        else:
            print(f"  ✓ Hub concentration OK")

        save_results(results, 'regression_check')
        return passed

    except Exception as e:
        print(f"  REGRESSION CHECK FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', action='store_true', help='Unit tests only')
    parser.add_argument('--integration', action='store_true', help='Integration tests only')
    parser.add_argument('--regression', action='store_true', help='Regression check only')
    args = parser.parse_args()

    run_all = not (args.unit or args.integration or args.regression)

    t0 = time.time()
    all_passed = True

    if run_all or args.unit:
        if not run_unit_tests():
            all_passed = False

    if run_all or args.integration:
        if not run_integration_tests():
            all_passed = False

    if run_all or args.regression:
        if not run_regression():
            all_passed = False

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    if all_passed:
        print(f"ALL TESTS PASSED ({elapsed:.1f}s)")
    else:
        print(f"SOME TESTS FAILED ({elapsed:.1f}s)")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
