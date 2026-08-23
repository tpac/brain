"""IsolatedBrain's own guarantees — the fixture the rest of the suite trusts.

It backs ~2800 tests plus every eval that runs against a production copy, so
what it silently tolerates becomes what those tests silently assume.

Run: ./dev pytest tests/test_isolated_brain.py -v
"""
import os
import shutil
import tempfile

import pytest

from tests.isolated_brain import IsolatedBrain, _default_production_dir


def test_missing_logs_db_is_refused_not_tolerated():
    """A production dir with brain.db but no brain_logs.db must RAISE.

    Brain() seeds a fresh logs schema when the file is absent, so the copy
    comes up looking like a clean install rather than a failure: zero override
    pointers, every interaction reporting source='default', no traces, no
    error rows. That is the same reading a successful override collapse
    produces — so a collapse verification, or any A/B that clears an override
    and re-measures, would confirm itself against a brain that never held the
    state under test. Measured before this guard existed: 14 freshly seeded
    interaction rows, active_version None for all of them, and
    get_interaction_stamp('s1e') == {'source': 'default', 'version': 0}.
    """
    src = _default_production_dir()
    if not src or not os.path.exists(os.path.join(src, 'brain.db')):
        pytest.skip('no production brain.db to build the fixture from')

    with tempfile.TemporaryDirectory() as fake:
        shutil.copy2(os.path.join(src, 'brain.db'),
                     os.path.join(fake, 'brain.db'))
        assert not os.path.exists(os.path.join(fake, 'brain_logs.db'))

        with pytest.raises(RuntimeError, match='no brain_logs.db'):
            with IsolatedBrain(production_dir=fake, cleanup=True,
                               load_env=False):
                pass


def test_both_databases_are_copied_when_present():
    """The normal path still works, and the logs copy is real — the guard must
    not have been satisfied by an empty file."""
    with IsolatedBrain(load_env=False) as env:
        assert os.path.exists(env.brain_db)
        assert os.path.exists(env.logs_db)
        # Copied, not seeded: a production copy carries interaction rows AND
        # the pointers that a freshly seeded schema would lack.
        pointers = env.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM interaction_active').fetchone()[0]
        assert pointers > 0, (
            'logs DB has no active pointers — that is what a fresh seed looks '
            'like, so the copy did not carry production state')
