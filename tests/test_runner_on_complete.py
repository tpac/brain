"""run_in_background's on_complete callback contract.

The S1 Encoder counts its runs toward the S2 activity gate ON COMPLETION (not
dispatch) and only when it actually wrote material. The wrapper fires
on_complete(write_actions) after a successful run_fn — in the background
thread, against the CALLER's brain (closure), since the run itself uses a
throwaway read_brain. These tests pin that contract without a real DB or
flakiness: the runner releases `lock` in its finally, so re-acquiring it
deterministically waits for the thread to finish.
"""

import os
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales import runner


class _StubReadBrain:
    """Stands in for the throwaway read_brain the runner builds per thread."""
    def __init__(self, *a, **k):
        pass

    def close(self):
        pass

    def _log_error(self, *a, **k):
        pass


def _drive(run_fn, on_complete):
    """Run run_in_background to completion (sync via the lock) and return."""
    lock = threading.Lock()
    lock.acquire()  # caller holds it; the runner releases in finally
    with patch('servers.brain.Brain', _StubReadBrain), \
         patch.object(runner, 'make_scale_dispatch', return_value=lambda *a, **k: {}):
        runner.run_in_background(
            name='test', brain_db_path='/tmp/none.db', session_id='s',
            counter=1, lock=lock, run_fn=run_fn, on_complete=on_complete)
        # Block until the thread's finally releases the lock → run complete.
        assert lock.acquire(timeout=5.0), "background thread never finished"


def test_on_complete_receives_write_actions():
    got = []
    _drive(lambda b, d, c, s: {'write_actions': 5, 'actions': 7},
           on_complete=got.append)
    assert got == [5], "on_complete must receive the run's write_actions"


def test_on_complete_zero_when_no_writes():
    got = []
    _drive(lambda b, d, c, s: {'actions': 2},  # no write_actions key → 0
           on_complete=got.append)
    assert got == [0], "a run that wrote nothing reports write_actions=0"


def test_on_complete_not_called_when_run_raises():
    got = []

    def boom(b, d, c, s):
        raise RuntimeError("encoder died")

    _drive(boom, on_complete=got.append)
    assert got == [], "on_complete must NOT fire when run_fn raised"


def test_callback_error_does_not_break_thread():
    # A raising callback must be swallowed (logged) — the lock still releases.
    def bad(_write_actions):
        raise ValueError("callback boom")

    # _drive asserts the lock is reacquired, i.e. the thread finished cleanly
    # despite the callback raising.
    _drive(lambda b, d, c, s: {'write_actions': 1}, on_complete=bad)
