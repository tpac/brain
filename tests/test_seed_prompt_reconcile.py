"""Contract tests for the shipped-prompt reconcile (the fleet gap fix).

Seeding alone froze an install at first boot: `_register` no-ops once a name
exists, so prompt improvements only ever reached brand-new brains. Reconcile
advances an install that is still running the shipped default, and must NEVER
touch one that made its own deployment decision.

The two guards are the whole safety argument, so they get the most tests:
  • active_version == the version we recorded putting there
  • max_version == active_version  (a parked dormant candidate means a human
    chose the current active on purpose — `trace_recording` is exactly this)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers import interaction_seed as IS
from servers.schema import read_schema_version


class TestSeedPromptReconcile(BrainTestBase):
    needs_embedder = False

    # ── helpers ──────────────────────────────────────────────────────

    def _reset_stream(self):
        """Clear the stream version so reconcile re-runs, as after a bump."""
        self.brain.logs_conn.execute(
            "DELETE FROM logs_meta WHERE key = ?",
            (IS.SEED_PROMPTS_VERSION_KEY,))
        self.brain.logs_conn.commit()

    def _state(self, name):
        info = {i['name']: i for i in self.brain.list_interactions()}[name]
        return info['active_version'], info['max_version']

    def _set_template(self, name, version, text):
        self.brain.logs_conn.execute(
            "UPDATE interactions SET template = ? WHERE name = ? AND version = ?",
            (text, name, version))
        self.brain.logs_conn.commit()

    def _shipped(self, name):
        return IS._shipped_prompts()[name][0]

    # ── the fix ──────────────────────────────────────────────────────

    def test_frozen_install_advances_to_shipped(self):
        """An install still on its install-day prompt gets moved forward."""
        self._set_template('s1e', 1, 'STALE PROMPT FROM INSTALL DAY')
        self._reset_stream()

        IS.reconcile_seeded_prompts(self.brain)

        active, _ = self._state('s1e')
        self.assertEqual(active, 2, 's1e should have advanced to a new version')
        self.assertEqual(self.brain.get_interaction('s1e')['template'],
                         self._shipped('s1e'),
                         'active template should now be the shipped one')

    def test_reconcile_is_idempotent(self):
        """Re-running writes nothing once the install is current."""
        self._set_template('s1e', 1, 'STALE')
        self._reset_stream()
        IS.reconcile_seeded_prompts(self.brain)
        first = self._state('s1e')

        self._reset_stream()
        IS.reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._state('s1e'), first,
                         'second reconcile must not churn versions')

    def test_version_gate_blocks_rerun_without_bump(self):
        """Without clearing/bumping the stream version, nothing runs."""
        self._set_template('s1e', 1, 'STALE')
        # stream version is already stamped from Brain.__init__
        IS.reconcile_seeded_prompts(self.brain)
        self.assertEqual(self._state('s1e'), (1, 1),
                         'gate should stop reconcile until the version bumps')

    # ── guard 1: locally evolved ─────────────────────────────────────

    def test_locally_evolved_prompt_is_never_overwritten(self):
        """A human-activated version wins forever."""
        self.brain.register_interaction('s2_healer', template='MY LOCAL PROMPT',
                                        created_by='anchor')
        self.brain.set_interaction_active('s2_healer', 2, set_by='operator')
        self._reset_stream()

        IS.reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._state('s2_healer'), (2, 2))
        self.assertEqual(self.brain.get_interaction('s2_healer')['template'],
                         'MY LOCAL PROMPT',
                         'local prompt must survive reconcile')

    # ── guard 2: dormant candidate (the trace_recording shape) ───────

    def test_parked_dormant_candidate_blocks_advance(self):
        """max > active means a human chose the active version deliberately.

        Without this guard reconcile would publish over that choice — the
        exact registration/activation conflation interaction_active exists
        to prevent.
        """
        self._set_template('s2_aspects', 1, 'STALE')
        self.brain.register_interaction('s2_aspects',
                                        template='DORMANT AWAITING EVAL',
                                        created_by='anchor')  # v2, not active
        self._reset_stream()

        IS.reconcile_seeded_prompts(self.brain)

        active, _ = self._state('s2_aspects')
        self.assertEqual(active, 1, 'must stay on the deliberately-active v1')
        self.assertEqual(
            self.brain.get_interaction('s2_aspects', 2)['template'],
            'DORMANT AWAITING EVAL', 'dormant candidate must be untouched')

    # ── failure containment ──────────────────────────────────────────

    def test_reconcile_failure_does_not_block_boot(self):
        """A broken reconcile must not take the brain down with it."""
        original = IS._reconcile_pristine_prompts
        IS._reconcile_pristine_prompts = lambda _b: (_ for _ in ()).throw(
            RuntimeError('boom'))
        try:
            self._reset_stream()
            IS.reconcile_seeded_prompts(self.brain)  # must not raise
        finally:
            IS._reconcile_pristine_prompts = original

        self.assertEqual(
            read_schema_version(self.brain.logs_conn, 'logs_meta',
                                IS.SEED_PROMPTS_VERSION_KEY),
            0, 'a failed run must stay unstamped so the next open retries')


if __name__ == '__main__':
    import unittest
    unittest.main()
