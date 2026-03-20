"""
brain — Shared Test Base with Brain-Aware Logging

Every test result is logged to brain_logs.db (debug_log table) so the brain
can analyze trends, detect flaky tests, and spot regression patterns.

Usage:
    from tests.brain_test_base import BrainTestBase, HookTestBase

All test classes should inherit from BrainTestBase (unit tests) or
HookTestBase (integration tests) instead of defining their own.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

# Ensure parent is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.schema import ensure_logs_schema


# ── Shared logs DB for cross-test-run persistence ────────────────────
# Test results accumulate in this DB across runs. The brain can query it
# to detect trends: "TestConsciousness fails 80% of the time", etc.

def _get_test_logs_db_path():
    """Resolve the test logs DB path. Uses real brain_logs.db if available."""
    # Try real brain location first
    brain_db_dir = os.environ.get('BRAIN_DB_DIR', '')
    if brain_db_dir:
        return os.path.join(brain_db_dir, 'brain_logs.db')
    home_path = os.path.expanduser('~/AgentsContext/brain/brain_logs.db')
    if os.path.exists(os.path.dirname(home_path)):
        return home_path
    # Fallback: local to test directory
    return os.path.join(os.path.dirname(__file__), 'results', 'test_logs.db')


def _get_test_logs_conn():
    """Get a connection to the shared test logs DB."""
    db_path = _get_test_logs_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    # Ensure the debug_log table exists
    conn.execute("""CREATE TABLE IF NOT EXISTS debug_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        event_type TEXT NOT NULL,
        source TEXT,
        metadata TEXT,
        created_at TEXT
    )""")
    conn.commit()
    return conn


# Module-level connection — shared across all tests in a run
_logs_conn = None
_run_id = None
_run_start = None
_test_results = []  # Collected for post-run encoding


def _ensure_logs():
    global _logs_conn, _run_id, _run_start
    if _logs_conn is None:
        _logs_conn = _get_test_logs_conn()
        _run_id = 'test_run_%d' % int(time.time())
        _run_start = time.time()
    return _logs_conn


def log_test_result(test_class, test_method, status, duration_ms,
                    error_msg=None, error_type=None):
    """Log a single test result to the shared debug_log."""
    conn = _ensure_logs()
    metadata = {
        'test_class': test_class,
        'test_method': test_method,
        'status': status,
        'duration_ms': round(duration_ms, 1),
    }
    if error_msg:
        metadata['error'] = error_msg[:500]
    if error_type:
        metadata['error_type'] = error_type

    conn.execute(
        "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (_run_id, 'test_result', f'test:{test_class}.{test_method}',
         json.dumps(metadata), ))
    conn.commit()

    # Also accumulate for post-run encoding
    _test_results.append(metadata)


def get_run_summary():
    """Get summary of the current test run for encoding."""
    if not _test_results:
        return None
    passed = sum(1 for r in _test_results if r['status'] == 'pass')
    failed = sum(1 for r in _test_results if r['status'] == 'fail')
    errors = sum(1 for r in _test_results if r['status'] == 'error')
    total_ms = sum(r['duration_ms'] for r in _test_results)

    failures = [r for r in _test_results if r['status'] in ('fail', 'error')]
    # Group failures by test class
    failure_classes = {}
    for f in failures:
        cls = f['test_class']
        failure_classes.setdefault(cls, []).append(f)

    return {
        'run_id': _run_id,
        'total': len(_test_results),
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': passed / len(_test_results) if _test_results else 0,
        'total_duration_ms': round(total_ms, 1),
        'avg_duration_ms': round(total_ms / len(_test_results), 1),
        'failures': failures,
        'failure_classes': {k: len(v) for k, v in failure_classes.items()},
        'results': _test_results,
    }


def get_historical_trends(limit=10):
    """Query recent test runs from debug_log for trend detection."""
    conn = _ensure_logs()
    try:
        rows = conn.execute("""
            SELECT session_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN json_extract(metadata, '$.status') = 'pass' THEN 1 ELSE 0 END) as passed,
                   SUM(CASE WHEN json_extract(metadata, '$.status') = 'fail' THEN 1 ELSE 0 END) as failed,
                   SUM(CASE WHEN json_extract(metadata, '$.status') = 'error' THEN 1 ELSE 0 END) as errors,
                   MIN(created_at) as run_time
            FROM debug_log
            WHERE event_type = 'test_result'
            GROUP BY session_id
            ORDER BY run_time DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {'run_id': r[0], 'total': r[1], 'passed': r[2],
             'failed': r[3], 'errors': r[4], 'run_time': r[5]}
            for r in rows
        ]
    except Exception:
        return []


def get_flaky_tests(min_runs=3):
    """Detect tests that sometimes pass and sometimes fail."""
    conn = _ensure_logs()
    try:
        rows = conn.execute("""
            SELECT source,
                   COUNT(*) as runs,
                   SUM(CASE WHEN json_extract(metadata, '$.status') = 'pass' THEN 1 ELSE 0 END) as passes,
                   SUM(CASE WHEN json_extract(metadata, '$.status') != 'pass' THEN 1 ELSE 0 END) as failures
            FROM debug_log
            WHERE event_type = 'test_result'
            GROUP BY source
            HAVING runs >= ? AND passes > 0 AND failures > 0
            ORDER BY failures DESC
        """, (min_runs,)).fetchall()
        return [
            {'test': r[0], 'runs': r[1], 'passes': r[2], 'failures': r[3],
             'flake_rate': round(r[3] / r[1], 2)}
            for r in rows
        ]
    except Exception:
        return []


def get_slowest_tests(limit=10):
    """Find consistently slowest tests across recent runs."""
    conn = _ensure_logs()
    try:
        rows = conn.execute("""
            SELECT source,
                   AVG(json_extract(metadata, '$.duration_ms')) as avg_ms,
                   COUNT(*) as runs
            FROM debug_log
            WHERE event_type = 'test_result'
            GROUP BY source
            HAVING runs >= 2
            ORDER BY avg_ms DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {'test': r[0], 'avg_ms': round(r[1], 1), 'runs': r[2]}
            for r in rows
        ]
    except Exception:
        return []


# ── BrainTestBase — for unit tests ──────────────────────────────────

class BrainTestBase(unittest.TestCase):
    """Base class that creates a fresh brain per test and logs results.

    Every test result (pass/fail/error + duration) is logged to brain_logs.db
    so the brain can analyze testing trends over time.
    """

    def setUp(self):
        self._test_start = time.time()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()

    def tearDown(self):
        duration_ms = (time.time() - self._test_start) * 1000

        # Determine test outcome
        result = self._outcome.result if hasattr(self, '_outcome') else None
        status = 'pass'
        error_msg = None
        error_type = None

        if result:
            # Check for failures and errors for THIS specific test
            test_id = str(self)
            for failed_test, traceback in result.failures:
                if str(failed_test) == test_id:
                    status = 'fail'
                    error_msg = traceback
                    error_type = 'AssertionError'
                    break
            for errored_test, traceback in result.errors:
                if str(errored_test) == test_id:
                    status = 'error'
                    error_msg = traceback
                    # Extract error type from traceback
                    if traceback:
                        lines = traceback.strip().split('\n')
                        if lines:
                            error_type = lines[-1].split(':')[0]
                    break

        log_test_result(
            test_class=self.__class__.__name__,
            test_method=self._testMethodName,
            status=status,
            duration_ms=duration_ms,
            error_msg=error_msg,
            error_type=error_type,
        )

        # Original cleanup
        if self.brain is not None:
            self.brain.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ── HookTestBase — for integration tests ────────────────────────────

class HookTestBase(unittest.TestCase):
    """Base class for hook integration tests with logging.

    Creates a temp brain.db with realistic test data, configures env vars,
    provides run_hook(), and logs results to brain_logs.db.
    """

    def setUp(self):
        self._test_start = time.time()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()
        # Add realistic test data
        self.brain.remember(
            type='rule',
            title='Authentication must use Clerk with magic links',
            content='Clerk handles auth flow. Magic links for login, no passwords. '
                    'Webhook syncs user data to our DB. Free tier covers MVP needs.',
            keywords='auth clerk login passwordless magic-link webhook',
            locked=True)
        self.brain.remember(
            type='decision',
            title='Supply Adapter pattern for ad delivery abstraction layer',
            content='Clean abstraction layer between Glo and ad delivery. '
                    'Interface: createCampaign, updateCampaign, pauseCampaign. '
                    'V1: GAM adapter only. Swap without touching business logic.',
            keywords='adapter pattern supply abstraction gam swappable interface',
            locked=True)
        self.brain.save()

        # Resolve paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.hooks_dir = os.path.join(project_root, 'hooks', 'scripts')
        self.servers_dir = os.path.join(project_root, 'servers')

        # Build env
        self.env = os.environ.copy()
        self.env['BRAIN_DB_DIR'] = self.tmp
        self.env['BRAIN_SERVER_DIR'] = os.path.join(self.servers_dir, 'brain.py')
        self.env['BRAIN_HOOKS_DIR'] = self.hooks_dir

    def tearDown(self):
        duration_ms = (time.time() - self._test_start) * 1000

        result = self._outcome.result if hasattr(self, '_outcome') else None
        status = 'pass'
        error_msg = None
        error_type = None

        if result:
            test_id = str(self)
            for failed_test, traceback in result.failures:
                if str(failed_test) == test_id:
                    status = 'fail'
                    error_msg = traceback
                    error_type = 'AssertionError'
                    break
            for errored_test, traceback in result.errors:
                if str(errored_test) == test_id:
                    status = 'error'
                    error_msg = traceback
                    if traceback:
                        lines = traceback.strip().split('\n')
                        if lines:
                            error_type = lines[-1].split(':')[0]
                    break

        log_test_result(
            test_class=self.__class__.__name__,
            test_method=self._testMethodName,
            status=status,
            duration_ms=duration_ms,
            error_msg=error_msg,
            error_type=error_type,
        )

        # Kill any daemon that tests may have started
        socket_path = os.path.join(self.tmp, 'brain_daemon.sock')
        if os.path.exists(socket_path):
            try:
                subprocess.run(['pkill', '-f', f'daemon.*{self.tmp}'],
                             capture_output=True, timeout=3)
            except Exception:
                pass

        if self.brain is not None:
            self.brain.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_hook(self, script_name, stdin_data=None, timeout=15, extra_env=None):
        """Run a hook script and return (exit_code, stdout, stderr)."""
        script_path = os.path.join(self.hooks_dir, script_name)
        if not os.path.exists(script_path):
            self.skipTest(f'Hook script not found: {script_path}')

        env = self.env.copy()
        if extra_env:
            env.update(extra_env)

        proc = subprocess.run(
            ['bash', script_path],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env)
        return proc.returncode, proc.stdout, proc.stderr
