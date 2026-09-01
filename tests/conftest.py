"""Pytest fixtures + dev guard rails.

Runs once per pytest invocation. Refuses to proceed if the wrong Python
interpreter is active — catches the "tests pass here but the daemon
runs a different Python" class of bug that bit us on 2026-04-19.

Expected interpreter: the bundled CPython 3.11 at
`<repo>/venv/bin/python`. That's the Python the daemon uses, the hooks
use via brain-env.sh, and the one not blocked by macOS SIP so debuggers
(py-spy, lldb) can attach.

To bypass (rare — e.g. quick smoke test on a machine without the venv):
    BRAIN_ALLOW_ANY_PYTHON=1 pytest ...

To run with the right Python without thinking about it:
    ./dev pytest ...
"""
import os
import sys
import tempfile

import pytest


# Environment isolation. os.environ is process-global and pytest runs the whole
# suite in one process, so a test that sets a variable and doesn't put it back
# hands that value to every test after it. The bite that motivated this: four
# classes set $BRAIN_DB_DIR in setUpClass and rmtree'd the directory in
# tearDownClass, so the rest of the suite inherited a pointer to a directory
# that no longer existed — and resolve_db_dir() trusts that variable
# unconditionally (D-13's top rung, on the premise that the hook wrappers
# validated it).
#
# Two fixtures because test setup nests two deep, and they nest the same way:
# the class-scoped one wraps unittest's setUpClass/tearDownClass, the
# function-scoped one wraps each test. So a variable set in setUpClass survives
# that class's tests and dies with the class; one set inside a test dies with
# the test. Broader-scoped fixtures are unaffected — pytest orders a test's
# fixture closure by scope, so a module-scoped IsolatedBrain is entered before
# either guard snapshots and its env is inside the snapshot, not stripped by it.
#
# This is the chokepoint, not a fifth convention: callers keep using whatever
# they already use — monkeypatch, patch.dict, explicit save/restore — and none
# of them can leak past its own scope even when the restore is missing or a
# tearDown raises before reaching it.
#
# pytest owns PYTEST_CURRENT_TEST and rewrites it per test; leave it alone.
_ENV_NOT_OURS = frozenset({'PYTEST_CURRENT_TEST'})


def _restore_env(saved):
    for key in [k for k in os.environ if k not in saved and k not in _ENV_NOT_OURS]:
        del os.environ[key]
    for key, value in saved.items():
        if key not in _ENV_NOT_OURS and os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture(autouse=True, scope='class')
def _env_isolation_class():
    saved = dict(os.environ)
    yield
    _restore_env(saved)


@pytest.fixture(autouse=True)
def _env_isolation_function():
    saved = dict(os.environ)
    yield
    _restore_env(saved)


# Live-brain aspects file guard. aspects_json_path() resolves from env at
# CALL time and falls back to the OPERATOR'S LIVE $BRAIN_DB_DIR/
# aspects_v1.json — and AspectRegistry (constructed by every Brain.__init__)
# runs reconcile_working_copy, which WRITES that file (seed heal). Any test
# that builds a raw Brain(db_path=tmp) without pinning ASPECTS_JSON_PATH
# therefore heals the live taxonomy from the repo seed under test
# (observed 2026-07-28: a worktree suite run stamped unmerged schema fields
# into the live file via test_prompt_sync / test_daemon).
#
# Pinned at import — before collection — rather than from a fixture, because
# setUpClass runs inside the class fixture and a function-scoped guard is too
# late to cover the raw Brain(db_path=tmp) that a setUpClass builds. Fills only
# when ABSENT, so deliberate overrides (BrainTestBase, IsolatedBrain,
# run_aspect_cycles_on_clone) still win; _env_isolation restores this baseline
# after each of them, which is what makes the pin self-healing.
if not os.environ.get('ASPECTS_JSON_PATH'):
    os.environ['ASPECTS_JSON_PATH'] = os.path.join(
        tempfile.mkdtemp(prefix='pytest_aspects_guard_'), 'aspects_v1.json')


# Hermetic key: brain.llm_available gates surface/encode/S2/warms on a
# resolved sk-* key (keyless first-run onboarding). Without pinning, real-
# Brain tests would behave differently on machines with vs without a key
# in ~/.config/brain/env — passing on the dev box, gating (and failing) on
# a keyless one. A fake sk- prefix keeps the suite deterministic; unit
# tests mock actual LLM calls, so the fake never reaches the API. Also
# stops load_env() from pulling the developer's REAL key into test runs.
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-test-hermetic-not-a-key')

_BYPASS_ENV = 'BRAIN_ALLOW_ANY_PYTHON'


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _expected_python():
    return os.path.join(_repo_root(), 'venv', 'bin', 'python')


def pytest_configure(config):
    """Fail fast if the wrong Python is running the tests."""
    if os.environ.get(_BYPASS_ENV):
        return

    expected = _expected_python()
    actual = os.path.realpath(sys.executable)
    expected_real = os.path.realpath(expected) if os.path.exists(expected) else expected

    # If the bundled venv doesn't exist yet (fresh clone), just warn.
    if not os.path.exists(expected):
        sys.stderr.write(
            '\n[conftest] WARN: bundled venv not found at %s — '
            'running under %s (%s).\n'
            '  Bootstrap via hooks/scripts/ensure-runtime.sh or set BRAIN_ALLOW_ANY_PYTHON=1.\n\n'
            % (expected, actual, '.'.join(map(str, sys.version_info[:3]))))
        return

    if actual != expected_real:
        ver = '.'.join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            '\n'
            '========================================================\n'
            ' WRONG PYTHON — tests refuse to run\n'
            '========================================================\n'
            '  expected: %s  (Python 3.11, not SIP-protected)\n'
            '  got:      %s  (Python %s)\n'
            '\n'
            '  Run tests via:    ./dev pytest ...\n'
            '  Or source:        source hooks/scripts/brain-env.sh\n'
            '  Or bypass once:   %s=1 pytest ...\n'
            '========================================================\n'
            % (expected_real, actual, ver, _BYPASS_ENV))

    # Sanity: reject SIP-protected system Pythons even by chance.
    if '/Xcode.app/' in actual or actual.startswith('/Applications/'):
        raise RuntimeError(
            '[conftest] Tests running under SIP-protected Python (%s). '
            'Debuggers cannot attach. Use ./dev pytest ...' % actual)
