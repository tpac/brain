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


# Live-brain aspects file guard. aspects_json_path() resolves from env at
# CALL time and falls back to the OPERATOR'S LIVE $BRAIN_DB_DIR/
# aspects_v1.json — and AspectRegistry (constructed by every Brain.__init__)
# runs reconcile_working_copy, which WRITES that file (seed heal). Any test
# that builds a raw Brain(db_path=tmp) without pinning ASPECTS_JSON_PATH
# therefore heals the live taxonomy from the repo seed under test
# (observed 2026-07-28: a worktree suite run stamped unmerged schema fields
# into the live file via test_prompt_sync / test_daemon).
#
# Function-scoped + self-healing on purpose: it fills the var only when
# ABSENT, so deliberate overrides (BrainTestBase, IsolatedBrain,
# run_aspect_cycles_on_clone) always win, and a test that pops the var
# doesn't strip protection from the tests after it.
_ASPECTS_GUARD_PATH = os.path.join(
    tempfile.mkdtemp(prefix='pytest_aspects_guard_'), 'aspects_v1.json')


@pytest.fixture(autouse=True)
def _aspects_live_file_guard():
    if not os.environ.get('ASPECTS_JSON_PATH'):
        os.environ['ASPECTS_JSON_PATH'] = _ASPECTS_GUARD_PATH
    yield


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
