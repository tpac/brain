"""recall(mark_accessed=False) must not touch the brain's own access record.

Why this exists: the flag was added so a read-only observer (the dashboard's
recall probe) can run the REAL recall pipeline without its looking becoming
part of what the brain believes it recalled — access_count / last_accessed are
what the graph renders as recall heat and what LAF scores against.

The first implementation guarded only `_recall_impl`'s marking loop and missed
that `_keyword_recall` ALSO marks, and that it runs on every recall as STEP 4
with `limit*3`. The flag was therefore false for up to 3× the requested nodes,
under a UI badge reading "read-only". These tests pin both paths so the
guarantee can't silently regress — a guarantee with no test is how it broke.

Assertions COUNT CALLS to `_mark_accessed`, which is exactly what the flag
governs. The two tempting observation points are both races: `_mark_accessed`
does no DB I/O (it enqueues into recall_write_queue), and a background worker
drains that queue every EMBED_DRAIN_INTERVAL seconds — so asserting on the
`nodes` table OR on `queue_depth()` passes or fails depending on whether the
worker happened to fire mid-test. Measured: the queue-depth version failed its
positive case for exactly that reason. A flaky test on this contract is worse
than none.
"""

import pytest

from tests.isolated_brain import IsolatedBrain

QUERY = 'dashboard journals encoder residue'


@pytest.fixture(scope='module')
def brain():
    with IsolatedBrain() as env:
        if env.brain is None:
            pytest.skip('no production brain to copy')
        yield env.brain


@pytest.fixture
def marks(brain, monkeypatch):
    """Record every node id `_mark_accessed` is called with, and suppress the
    real enqueue so tests never feed the live write queue."""
    seen = []
    monkeypatch.setattr(type(brain), '_mark_accessed',
                        lambda self, node_id, session_id, ctx=None: seen.append(node_id))
    return seen


def test_read_only_recall_marks_nothing(brain, marks):
    """The guarantee: mark_accessed=False marks zero nodes, on every path."""
    result = brain.recall(query=QUERY, limit=8, session_id='probe-session',
                          source='dashboard', mark_accessed=False)
    # Guard against a vacuous pass — if recall found nothing there is nothing
    # it could have marked, and the assertion below would be meaningless.
    assert result.get('results'), 'recall returned no results — test proves nothing'
    assert marks == [], (
        'read-only recall marked %d node(s): %r' % (len(marks), marks[:5]))


def test_default_recall_still_marks(brain, marks):
    """The default path must keep marking — the exception must not become the
    rule. Without this, disabling marking everywhere would pass the suite."""
    result = brain.recall(query=QUERY, limit=8, session_id='real-session',
                          source='mcp', mark_accessed=True)
    assert result.get('results'), 'recall returned no results — test proves nothing'
    assert marks, 'default recall marked nothing — access tracking is broken'


def test_keyword_lane_honors_the_flag(brain, marks):
    """The actual regression.

    `_keyword_recall` is not just the keyword-only fallback — `_recall_impl`
    calls it as STEP 4 on EVERY recall with limit*3, and it marks its own page.
    Guarding only the outer loop left the flag false for the keyword lane, so
    this calls the lane directly with the flag off.
    """
    out = brain._keyword_recall(QUERY, None, 24, 0, False, 0, 'probe-session',
                                _skip_log=True, mark_accessed=False)
    assert out.get('results'), 'keyword lane returned nothing — test proves nothing'
    assert marks == [], (
        'keyword lane marked %d node(s) despite mark_accessed=False' % len(marks))


def test_keyword_lane_marks_by_default(brain, marks):
    """Counterpart to the above: the lane's normal behavior is preserved."""
    out = brain._keyword_recall(QUERY, None, 24, 0, False, 0, 'real-session',
                                _skip_log=True)
    assert out.get('results'), 'keyword lane returned nothing — test proves nothing'
    assert marks, 'keyword lane stopped marking on its default path'
