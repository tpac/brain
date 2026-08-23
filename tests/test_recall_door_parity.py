"""Guard: every recall door hands back the same canonical shape.

Recall has more than one door — MCP by-query, MCP by-id, MCP batch, and the
in-brain hook path — and for months they silently disagreed. `recall_node`
read the bare DB row, so `recall({node_id})` returned a node with NO
corrections, NO situation and NO connections: a superseded claim handed over
with its correction marker stripped off, while the by-query door on the same
node returned all three. `recall_batch` had the same hole.

The fix was one door — `Brain.canonicalize_results` — that every shape routes
through. These tests hold that line, because the failure mode is invisible
from any single door: each one looks fine on its own, and only a comparison
shows the drift.

Two ways the drift comes back, one test each:

1. A NEW door is added that skips the canonical pull. Caught by
   `test_recall_producing_handlers_are_known` — a tripwire on the set of
   dispatch handlers that produce recall results. It fails when someone adds
   one, which is the moment to add it to the parity test below.

2. `get_node` GROWS an attachment that CANONICAL_ATTACHMENT_KEYS doesn't
   list. Then the doors diverge again on the new field, silently, because the
   overlay only copies what the list names. Caught by
   `test_canonical_keys_cover_what_get_node_attaches`.
"""

import ast
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servers import dispatch_read
from servers.contract import CANONICAL_ATTACHMENT_KEYS
from servers.dispatch_common import CALLER_SESSION_KEY
from tests.brain_test_base import BrainTestBase

# Dispatch handlers that turn a brain.recall*/recall_node call into results
# handed to a caller. Each must appear in the parity test below. Adding a door
# without adding it there is the regression this list exists to make loud.
KNOWN_RECALL_DOORS = {'_handle_recall', '_handle_recall_batch'}

SESSION = 'sess-parity'


def _handlers_producing_recall_results():
    """Dispatch-read functions whose body calls brain.recall / recall_node."""
    tree = ast.parse(inspect.getsource(dispatch_read))
    found = set()
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)]:
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
            attr = getattr(call.func, 'attr', None)
            if attr in ('recall', 'recall_node', 'recall_batch'):
                found.add(fn.name)
    return found


def test_recall_producing_handlers_are_known():
    """A new recall door must be added to the parity test, deliberately."""
    found = _handlers_producing_recall_results()
    assert found == KNOWN_RECALL_DOORS, (
        'recall-producing dispatch handlers changed: %s. Every door must '
        'route through Brain.canonicalize_results and get its own '
        '*_door_matches_the_canonical_pull case in TestRecallDoorParity '
        'below — add it there, then update KNOWN_RECALL_DOORS.'
        % sorted(found ^ KNOWN_RECALL_DOORS))


class TestRecallDoorParity(BrainTestBase):
    """Behavioural parity, per door, against the canonical pull."""

    def setUp(self):
        super().setUp()
        # A node worth comparing: it carries a situation, an incoming
        # correction and an edge, so all the canonical attachments are
        # non-empty and a dropped one is visible.
        self.claim = self.brain.remember_rich(
            type='finding',
            title='Zarquon telemetry drifted during the eclipse window',
            content='The zarquon counter disagreed with the eclipse log.',
            situation='When reconciling zarquon telemetry against eclipse logs',
        )['id']
        self.fix = self.brain.remember_rich(
            type='correction',
            title='Zarquon drift was the clock, not the counter',
            content='The counter was right; the clock had skewed.',
        )['id']
        self.brain.connect_typed(
            self.fix, self.claim, relation='corrects',
            description='the skew explains the whole disagreement')
        self.canonical = self.brain.get_node(self.claim)
        # Compare against what get_node ACTUALLY attaches, derived live —
        # deliberately not against CANONICAL_ATTACHMENT_KEYS. Deriving the
        # comparison from the same constant the overlay reads would give both
        # the same blind spot: drop a key from the tuple and the doors would
        # stop copying that field AND the test would stop checking it.
        naked = self.brain._nodes.get_naked_node(self.claim)
        self.attached = set(self.canonical) - set(naked)

    def _assert_matches_canonical(self, node, door):
        for key in sorted(self.attached):
            self.assertEqual(
                node.get(key), self.canonical.get(key),
                '%s door disagrees with get_node on %r — the recall doors '
                'have drifted apart' % (door, key))

    def test_by_id_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall(
            self.brain, {'node_id': self.claim,
                         CALLER_SESSION_KEY: SESSION}, None)
        self._assert_matches_canonical(out['result']['results'][0], 'by-id')

    def test_by_query_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall(
            self.brain, {'query': 'zarquon telemetry eclipse window',
                         CALLER_SESSION_KEY: SESSION}, None)
        node = self._find(out['result']['results'], 'by-query')
        self._assert_matches_canonical(node, 'by-query')

    def test_batch_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall_batch(
            self.brain, {'queries': ['zarquon telemetry eclipse window'],
                         CALLER_SESSION_KEY: SESSION}, None)
        node = self._find(out['result'][0]['results'], 'batch')
        self._assert_matches_canonical(node, 'batch')

    def _find(self, results, door):
        for r in results:
            if r.get('id') == self.claim:
                return r
        self.fail('%s door returned no result for the seeded node (%d '
                  'results) — recall itself is broken, or the seed no longer '
                  'matches the query' % (door, len(results)))

    def test_canonical_keys_cover_what_get_node_attaches(self):
        """The overlay list must not fall behind get_node's assembly.

        get_node attaches on top of the bare DB row; anything it adds that
        CANONICAL_ATTACHMENT_KEYS doesn't name would be copied by no door,
        reintroducing the divergence on a new field.
        """
        missing = self.attached - set(CANONICAL_ATTACHMENT_KEYS)
        assert not missing, (
            'get_node now attaches %s, which CANONICAL_ATTACHMENT_KEYS does '
            'not list — canonicalize_results will not copy it and the recall '
            'doors will disagree on it. Add it to the tuple in contract.py.'
            % sorted(missing))
