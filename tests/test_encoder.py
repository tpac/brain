"""
brain — Test Result Encoder

After a test run completes, this module analyzes the results and encodes
findings into the brain as first-class knowledge:

  - Failures → lesson nodes (what broke, where, error type)
  - Regressions → tension nodes (was passing, now failing)
  - Flaky tests → pattern nodes (intermittent failures)
  - Slow tests → constraint nodes (performance concerns)
  - Clean runs → validation of prior fixes

The brain can then query its own test health:
    brain.recall_with_embeddings("test failures consciousness signals")
    brain.logs_conn.execute("SELECT * FROM debug_log WHERE event_type='test_result'")

Usage:
    from tests.test_encoder import encode_run_to_brain
    encode_run_to_brain(brain)  # Call after test run completes
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def encode_run_to_brain(brain, verbose=False):
    """Encode the current test run's results into the brain.

    Reads from brain_test_base.get_run_summary() and historical data,
    then creates/updates brain nodes based on the findings.

    Returns dict with encoding stats.
    """
    from tests.brain_test_base import (
        get_run_summary, get_historical_trends, get_flaky_tests, get_slowest_tests
    )

    summary = get_run_summary()
    if not summary:
        if verbose:
            print('[encoder] No test results to encode')
        return {'encoded': 0}

    encoded = 0
    nodes_created = []

    # ── 1. Encode failures as lessons ────────────────────────────────
    for failure in summary['failures']:
        cls = failure['test_class']
        method = failure['test_method']
        error = failure.get('error', 'Unknown error')
        error_type = failure.get('error_type', 'Unknown')

        # Truncate error for title, keep full in content
        error_summary = error.split('\n')[-1][:100] if error else 'Unknown'

        result = brain.remember(
            type='lesson',
            title=f'Test failure: {cls}.{method} — {error_type}',
            content=f'Test {cls}.{method} failed with {error_type}.\n\n'
                    f'Error: {error_summary}\n\n'
                    f'Full traceback (last 500 chars): {error[-500:] if error else "N/A"}\n\n'
                    f'This was detected during test run {summary["run_id"]} '
                    f'on {datetime.now().strftime("%Y-%m-%d")}.',
            keywords=f'test failure {cls} {method} {error_type} regression bug',
            confidence=0.8,
            emotion=0.6,
            emotion_label='concern')
        nodes_created.append(result.get('id'))
        encoded += 1

        if verbose:
            print(f'  [encoder] Failure → lesson: {cls}.{method}')

    # ── 2. Encode failure clusters as patterns ───────────────────────
    for cls, count in summary.get('failure_classes', {}).items():
        if count >= 2:
            result = brain.create_pattern(
                title=f'Pattern: {count} test failures in {cls}',
                content=f'{count} tests failed in {cls} during run {summary["run_id"]}. '
                        f'This suggests a systemic issue in the module being tested, '
                        f'not isolated test bugs. Investigate the underlying code.')
            nodes_created.append(result.get('id'))
            encoded += 1
            if verbose:
                print(f'  [encoder] Cluster → pattern: {cls} ({count} failures)')

    # ── 3. Detect regressions (was passing in prior runs, now failing) ─
    trends = get_historical_trends(limit=5)
    if len(trends) >= 2:
        current = trends[0] if trends[0]['run_id'] == summary['run_id'] else None
        previous = trends[1] if len(trends) > 1 else None

        if current and previous:
            if current['failed'] > previous['failed']:
                regression_count = current['failed'] - previous['failed']
                # Find which tests regressed by comparing with prior passing
                result = brain.create_tension(
                    title=f'Regression: {regression_count} new test failure(s) since last run',
                    content=f'Test run {current["run_id"]}: {current["failed"]} failures '
                            f'(was {previous["failed"]} in {previous["run_id"]}). '
                            f'Total tests: {current["total"]}. '
                            f'Pass rate dropped from '
                            f'{round(previous["passed"]/max(1,previous["total"])*100)}% to '
                            f'{round(current["passed"]/max(1,current["total"])*100)}%.',
                    node_a_id=nodes_created[0] if nodes_created else None,
                    node_b_id=nodes_created[-1] if len(nodes_created) > 1 else nodes_created[0] if nodes_created else None,
                ) if nodes_created and len(nodes_created) >= 2 else None
                if result:
                    encoded += 1
                    if verbose:
                        print(f'  [encoder] Regression → tension: {regression_count} new failures')

    # ── 4. Detect flaky tests ────────────────────────────────────────
    flaky = get_flaky_tests(min_runs=3)
    for f in flaky[:3]:  # Top 3 flakiest
        result = brain.create_pattern(
            title=f'Flaky test: {f["test"]} — {f["flake_rate"]*100:.0f}% failure rate',
            content=f'Test {f["test"]} has failed {f["failures"]} out of {f["runs"]} runs '
                    f'({f["flake_rate"]*100:.0f}% failure rate). This test needs investigation: '
                    f'either the test is non-deterministic or it depends on timing/order.')
        encoded += 1
        if verbose:
            print(f'  [encoder] Flaky → pattern: {f["test"]}')

    # ── 5. Detect slow tests ─────────────────────────────────────────
    slow = get_slowest_tests(limit=5)
    slow_threshold_ms = 2000  # 2 seconds
    for s in slow:
        if s['avg_ms'] > slow_threshold_ms:
            brain.remember(
                type='constraint',
                title=f'Slow test: {s["test"]} — avg {s["avg_ms"]:.0f}ms',
                content=f'Test {s["test"]} averages {s["avg_ms"]:.0f}ms over {s["runs"]} runs. '
                        f'Tests should complete in under {slow_threshold_ms}ms. '
                        f'Consider: is the embedder loading per test? Can setup be shared?',
                keywords=f'test slow performance {s["test"]} constraint',
                confidence=0.7)
            encoded += 1
            if verbose:
                print(f'  [encoder] Slow → constraint: {s["test"]}')

    # ── 6. Record clean run as validation ────────────────────────────
    if summary['failed'] == 0 and summary['errors'] == 0:
        brain.remember(
            type='context',
            title=f'Clean test run: {summary["total"]} tests passed '
                  f'({summary["total_duration_ms"]/1000:.1f}s)',
            content=f'All {summary["total"]} tests passed on '
                    f'{datetime.now().strftime("%Y-%m-%d %H:%M")}. '
                    f'Run ID: {summary["run_id"]}. '
                    f'Average test duration: {summary["avg_duration_ms"]:.0f}ms.',
            keywords='test clean pass all green validation confidence')
        encoded += 1
        if verbose:
            print(f'  [encoder] Clean run → validation: {summary["total"]} tests')

    brain.save()

    result = {
        'encoded': encoded,
        'nodes_created': [n for n in nodes_created if n],
        'run_summary': {
            'total': summary['total'],
            'passed': summary['passed'],
            'failed': summary['failed'],
            'errors': summary['errors'],
            'pass_rate': round(summary['pass_rate'] * 100, 1),
        },
    }

    if verbose:
        print(f'[encoder] Encoded {encoded} findings from {summary["total"]} tests '
              f'({summary["passed"]} pass, {summary["failed"]} fail, {summary["errors"]} error)')

    return result


def print_trends_report():
    """Print a human-readable trends report from historical test data."""
    from tests.brain_test_base import get_historical_trends, get_flaky_tests, get_slowest_tests

    trends = get_historical_trends(limit=10)
    flaky = get_flaky_tests(min_runs=3)
    slow = get_slowest_tests(limit=5)

    print()
    print('=' * 60)
    print('  TEST TRENDS REPORT')
    print('=' * 60)

    if trends:
        print()
        print('  Recent runs:')
        for t in trends:
            status = 'PASS' if t['failed'] == 0 and t['errors'] == 0 else 'FAIL'
            print(f'    [{status}] {t["run_time"]}: '
                  f'{t["passed"]}/{t["total"]} pass, '
                  f'{t["failed"]} fail, {t["errors"]} error')
    else:
        print('\n  No historical data yet.')

    if flaky:
        print()
        print('  Flaky tests:')
        for f in flaky:
            print(f'    {f["test"]}: {f["flake_rate"]*100:.0f}% failure rate '
                  f'({f["failures"]}/{f["runs"]} runs)')

    if slow:
        print()
        print('  Slowest tests (avg):')
        for s in slow:
            print(f'    {s["test"]}: {s["avg_ms"]:.0f}ms ({s["runs"]} runs)')

    print()
    print('=' * 60)
