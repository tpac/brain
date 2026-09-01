"""Contract: conftest's env guard makes an unrestored env write impossible.

os.environ is process-global and pytest runs the suite in one process, so a
test that sets a variable and doesn't put it back hands that value to every
test after it. That is not hypothetical here — four classes set $BRAIN_DB_DIR
in setUpClass and rmtree'd the directory in tearDownClass, leaving the rest of
the suite pointed at a directory that no longer existed, and resolve_db_dir()
trusts that variable unconditionally (D-13's top rung).

conftest closes it with a snapshot/restore pair — one class-scoped (wraps
setUpClass/tearDownClass), one function-scoped (wraps each test). These tests
assert the guard's BEHAVIOUR, so deleting either fixture from conftest fails
them: a gate that goes green when the mechanism it watches disappears is not a
gate (id:89ba43e8).

Ordering: each leak source is followed by the item that checks it escaped
nothing, and pytest runs items in collection (file) order — requirements-test
pins pytest + pytest-timeout only, no randomizer.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.daemon_config import resolve_db_dir  # noqa: E402

_CLASS_PROBE = 'BRAIN_TEST_ENV_PROBE_CLASS'
_FUNC_PROBE = 'BRAIN_TEST_ENV_PROBE_FUNC'


class ClassScopeLeakSource(unittest.TestCase):
    """The exact shape of the bug: setUpClass writes, tearDownClass forgets."""

    @classmethod
    def setUpClass(cls):
        os.environ[_CLASS_PROBE] = 'written-by-setUpClass'

    def test_class_scope_write_is_visible_within_the_class(self):
        self.assertEqual(os.environ.get(_CLASS_PROBE), 'written-by-setUpClass')


def test_class_scope_write_did_not_escape_the_class():
    """Fails if the class-scoped guard is removed — setUpClass ran before this."""
    assert _CLASS_PROBE not in os.environ, (
        'setUpClass wrote %s=%r and it survived tearDownClass — the '
        'class-scoped guard in conftest is missing or disarmed'
        % (_CLASS_PROBE, os.environ.get(_CLASS_PROBE)))


def test_function_scope_leak_source():
    os.environ[_FUNC_PROBE] = 'written-by-a-test-body'


def test_function_scope_write_did_not_escape_the_test():
    """Fails if the function-scoped guard is removed."""
    assert _FUNC_PROBE not in os.environ, (
        'a test body wrote %s=%r and it survived into the next test — the '
        'function-scoped guard in conftest is missing or disarmed'
        % (_FUNC_PROBE, os.environ.get(_FUNC_PROBE)))


def test_deletion_leak_source():
    """The other direction: a test that POPS a var the suite relies on."""
    os.environ.pop('ASPECTS_JSON_PATH', None)


def test_deleted_var_was_restored():
    """Fails if the guard only strips additions and never re-adds removals."""
    assert os.environ.get('ASPECTS_JSON_PATH'), (
        'the previous test popped ASPECTS_JSON_PATH and it was not put back — '
        'the guard restores additions but not deletions')


def test_aspects_pin_never_points_at_the_live_taxonomy():
    """conftest pins ASPECTS_JSON_PATH at import, before collection, because
    setUpClass builds Brains before any fixture could pin it — and every
    Brain.__init__ reconciles (WRITES) the resolved aspects file."""
    pinned = os.environ.get('ASPECTS_JSON_PATH')
    assert pinned, 'ASPECTS_JSON_PATH is unpinned — a raw Brain() in a ' \
                   'setUpClass would heal the operator\'s live taxonomy'
    live_dir = resolve_db_dir(trust_env=False)
    assert os.path.dirname(os.path.abspath(pinned)) != os.path.abspath(live_dir), (
        'ASPECTS_JSON_PATH points into the live brain dir (%s)' % live_dir)
