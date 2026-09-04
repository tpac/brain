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
import atexit
import os
import re
import shutil
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
# the test.
#
# This is the chokepoint, not a fifth convention: callers keep using whatever
# they already use — monkeypatch, patch.dict, explicit save/restore — and none
# of them can leak past its own scope even when the restore is missing or a
# tearDown raises before reaching it.
#
# LIMIT, and it is a real one. A guard can only restore what its own scope
# opened, so anything that outlives the scope but is first created INSIDE it
# gets clobbered. Two shapes to keep out of the suite:
#   - a module/session-scoped fixture that writes env and is first requested by
#     a LATER test of a class rather than its first (the class guard snapshotted
#     before the fixture existed, so its restore deletes env the fixture still
#     owns). IsolatedBrain as a module-scoped fixture is exactly this shape, and
#     every file using it is safe only because its class requests it from the
#     FIRST test — verified by watching BRAIN_DB_DIR across test_fetch_tools.py.
#     Nothing enforces that: it is a real, unguarded limit, because the only
#     check available is textual and a grep for this shape matches its own
#     description.
#   - a module whose IMPORT writes env, first imported inside a test: the write
#     is reverted and sys.modules caching means it never happens again. That is
#     why daemon_config is imported below rather than left to whichever test
#     file happens to pull it in first.
#
# REPAIRING SILENTLY IS ITS OWN DRIFT. A guard that quietly absorbs an
# unrestored write means the next sloppy setUpClass is invisible forever, so
# every repair is recorded and the session FAILS on any of them. The suite sits
# at zero — the four sites that motivated this turned out to be writing
# $BRAIN_DB_DIR that nothing read, and were deleted rather than annotated — so
# a non-empty report means someone just introduced a leak, named with the test
# and the keys. Set BRAIN_TEST_ENV_LEAKS_OK=1 to downgrade it to a report while
# mid-refactor.
#
# pytest owns PYTEST_CURRENT_TEST and rewrites it per test; leave it alone.
_ENV_NOT_OURS = frozenset({'PYTEST_CURRENT_TEST'})
_ENV_LEAKS = []
# The one file whose job is to leak: its probes exist to prove the guard puts
# things back, so reporting them would make the report permanently non-empty
# and train everyone to ignore it.
_ENV_LEAK_EXEMPT = 'test_env_isolation.py'


_SECRET_KEY = re.compile(r'KEY|TOKEN|SECRET|PASSWORD', re.I)


def _shown(key, value):
    """A leaked value as the report may print it. A credential is shown as
    its fingerprint — the report lands in terminals and CI logs, where a raw
    key must never appear (the daemon's own rule, key_fingerprint)."""
    if value is None:
        return 'unset'
    if _SECRET_KEY.search(key):
        from servers.scales.dispatch import key_fingerprint
        return 'fp:%s' % key_fingerprint(value)
    return repr(value)


def _restore_env(saved, where=''):
    # Re-add before delete: a restore that dies half-done must not leave a
    # deliberately-popped baseline (the ASPECTS_JSON_PATH pin) missing, and
    # pop-with-default cannot raise on a key another thread already removed.
    leaked = []
    for key, value in saved.items():
        if key not in _ENV_NOT_OURS and os.environ.get(key) != value:
            leaked.append('%s: %s -> %s' % (key, _shown(key, value),
                                            _shown(key, os.environ.get(key))))
            os.environ[key] = value
    for key in [k for k in os.environ if k not in saved and k not in _ENV_NOT_OURS]:
        leaked.append('%s: unset -> %s' % (key, _shown(key, os.environ.get(key))))
        os.environ.pop(key, None)
    if leaked and where and _ENV_LEAK_EXEMPT not in where:
        _ENV_LEAKS.append((where, leaked))


@pytest.fixture(autouse=True, scope='class')
def _env_isolation_class(request):
    saved = dict(os.environ)
    yield
    _restore_env(saved, 'setUpClass of %s' % request.node.nodeid)


@pytest.fixture(autouse=True)
def _env_isolation_function(request):
    saved = dict(os.environ)
    yield
    _restore_env(saved, request.node.nodeid)


def pytest_terminal_summary(terminalreporter):
    if not _ENV_LEAKS:
        return
    terminalreporter.section('environment leaks', red=True)
    for where, leaked in _ENV_LEAKS:
        terminalreporter.write_line('%s' % where)
        for change in leaked:
            terminalreporter.write_line('    %s' % change)
    terminalreporter.write_line(
        '%d scope(s) mutated os.environ without restoring it. The guard put it '
        'back, so nothing downstream broke — but restore it at the source '
        '(monkeypatch, patch.dict, or setUp/tearDown). '
        'BRAIN_TEST_ENV_LEAKS_OK=1 downgrades this to a report.'
        % len(_ENV_LEAKS))


def pytest_sessionfinish(session, exitstatus):
    if _ENV_LEAKS and not os.environ.get('BRAIN_TEST_ENV_LEAKS_OK'):
        session.exitstatus = 1


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
# The pin is also ANNOUNCED, in BRAIN_TEST_ASPECTS_DEFAULT_PIN. IsolatedBrain
# honors a pre-set ASPECTS_JSON_PATH as "the caller owns the aspects location"
# and otherwise pins its own seeded copy — a safety default it cannot
# distinguish from a deliberate override would silently disable that, leaving
# every IsolatedBrain in the process sharing this one file. Publishing the
# value lets it tell the two apart; absent (no conftest, e.g. an eval script),
# it falls back to the is-None test.
if not os.environ.get('ASPECTS_JSON_PATH'):
    _guard_dir = tempfile.mkdtemp(prefix='pytest_aspects_guard_')
    # Swept at exit: one dir per pytest invocation adds up silently — the
    # unconditional version of this line had left 1131 of them behind.
    atexit.register(shutil.rmtree, _guard_dir, True)
    os.environ['ASPECTS_JSON_PATH'] = os.path.join(_guard_dir, 'aspects_v1.json')
    os.environ['BRAIN_TEST_ASPECTS_DEFAULT_PIN'] = os.environ['ASPECTS_JSON_PATH']


# Hermetic config home + key. brain.llm_available gates surface/encode/S2/warms
# on a resolved sk-* key (keyless first-run onboarding); without pinning,
# real-Brain tests would behave differently on machines with vs without a key.
# A fake sk- prefix keeps the suite deterministic; unit tests mock LLM calls,
# so the fake never reaches the API.
#
# The pin covers the FILE as well as the variable: resolve_api_key() reads
# ${XDG_CONFIG_HOME:-~/.config}/brain/env on every llm_available check and
# writes the file's key into os.environ (the file wins — that is what makes
# key replacement work without a restart), so a real env file in reach swaps
# the fake for a real credential on the first test that touches it, and the
# leak guard above then reports what it caught. So the config home is a
# per-run dir holding the fake key and nothing secret — set unconditionally
# (at import no test has run; one that wants its own config home sets it
# later) and BEFORE servers.daemon_config is imported below, which resolves
# the daemon port from the same file at import. The brain-location POINTER
# (BRAIN_DB_DIR, from the knob or resolved.env) is carried over: IsolatedBrain
# finds the production brain through it, and it holds no secret.
_HERMETIC_KEY = 'sk-test-hermetic-not-a-key'
_real_cfg = os.path.join(os.environ.get('XDG_CONFIG_HOME')
                         or os.path.expanduser('~/.config'), 'brain')
_cfg_home = tempfile.mkdtemp(prefix='pytest_xdg_config_')
atexit.register(shutil.rmtree, _cfg_home, True)
os.makedirs(os.path.join(_cfg_home, 'brain'))
_pointer = ''
for _name in ('env', 'resolved.env'):
    try:
        with open(os.path.join(_real_cfg, _name)) as _f:
            _lines = [l for l in _f if l.startswith('BRAIN_DB_DIR=')]
    except OSError:
        continue
    if _lines:
        _pointer = _lines[-1]
        break
with open(os.path.join(_cfg_home, 'brain', 'env'), 'w') as _f:
    _f.write('ANTHROPIC_API_KEY=%s\n%s' % (_HERMETIC_KEY, _pointer))
os.environ['XDG_CONFIG_HOME'] = _cfg_home
os.environ['ANTHROPIC_API_KEY'] = _HERMETIC_KEY


# CPU-only pin. daemon_config is the one module under servers/ that writes
# os.environ at IMPORT (DAEMON_CPU_ENV — ORT_DISABLE_ALL_ACCELERATORS is the
# load-bearing SIGABRT guard on Apple Silicon), and most test files reach it
# through a lazy in-function import. A process-wide invariant established
# inside a test is one _env_isolation restore away from being deleted for good,
# because sys.modules caching means the module body never runs twice — so
# import it here, before anything can snapshot, and the invariant is part of
# every baseline instead of a side effect of which files got collected.
import servers.daemon_config  # noqa: E402,F401



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
