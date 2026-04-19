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
