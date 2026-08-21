"""Shipped-prompt reconciliation — the fleet path for prompt improvements.

Seeding is create-only, so before this mechanism an install captured the prompts
of its install date and froze there permanently. Reconcile advances a prompt only
while the install still runs the shipped default.

The two tests that would have caught attempt 1's real gaps:
  - `test_two_consecutive_bumps_both_land` — the actual fleet path (an install
    opening code that has moved twice). Attempt 1 covered a single bump only.
  - `test_shipped_content_fingerprint` — a forgotten SEED_PROMPTS_VERSION bump
    rebuilds the original freeze in total silence. Nothing else can fail on it.
"""

import hashlib
import inspect
import json
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase

from servers import interaction_seed
from servers.dal_logs import (BACKSTOP_PROVENANCE, RECONCILE_PROVENANCE,
                              SYSTEM_PROVENANCE)
from servers.interaction_seed import (
    PRISTINE_ACTIVATIONS,
    SEED_PROMPTS_VERSION,
    SEED_PROMPTS_VERSION_KEY,
    reconcile_seeded_prompts,
    shipped_prompts,
)
from servers.schema import read_schema_version, stamp_schema_version

NAME = 's1_scout_facts'  # a template-carrying prompt, cheap to drive


class ReconcileTestBase(BrainTestBase):
    """A brain whose prompts are seeded but otherwise untouched — i.e. a
    freshly-installed fleet member from the pre-override-model era.

    Registration never activates anymore, so the legacy state reconcile
    exists for (every seeded v1 auto-activated as AUTO_V1) is constructed
    explicitly here. Installs created after the override model carry no
    pointers and reconcile correctly leaves them alone — the code default
    already flows."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        from servers.dal_logs import AUTO_V1_PROVENANCE
        for row in self.brain.list_interactions():
            if not row.get('active_version'):
                self.brain._interaction_dal.set_active(
                    row['name'], 1, set_by=AUTO_V1_PROVENANCE)

    def _shipped(self, name=NAME):
        return shipped_prompts()[name]

    def _active(self, name=NAME):
        return self.brain.get_interaction(name) or {}

    def _pointer(self, name=NAME):
        row = self.brain.logs_conn.execute(
            'SELECT version, set_by FROM interaction_active WHERE name = ?',
            (name,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def _version_count(self, name=NAME):
        return len(self.brain.list_interaction_versions(name))

    def _reset_stream(self):
        """Un-stamp the seed-prompts counter so the next call reconciles."""
        self.brain.logs_conn.execute(
            "DELETE FROM logs_meta WHERE key = ?", (SEED_PROMPTS_VERSION_KEY,))
        self.brain.logs_conn.commit()

    def _drift(self, name=NAME, template=None, params=None):
        """Rewrite the ACTIVE version's content so it no longer matches shipped.

        Simulates the real situation from the other side: the install is frozen
        at old content while the repo has moved on. Editing the row (rather than
        the module constants) keeps the shipped side authentic.
        """
        active_v, _ = self._pointer(name)
        sets, vals = [], []
        if template is not None:
            sets.append('template = ?')
            vals.append(template)
        if params is not None:
            sets.append('parameters = ?')
            vals.append(json.dumps(params))
        vals += [name, active_v]
        self.brain.logs_conn.execute(
            'UPDATE interactions SET %s WHERE name = ? AND version = ?'
            % ', '.join(sets), vals)
        self.brain.logs_conn.commit()


class PristineAdvanceTest(ReconcileTestBase):

    def test_frozen_install_advances_template_and_params(self):
        template, config = self._shipped()
        self._drift(template='STALE PROMPT', params={'model': 'old-model-id'})
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        active = self._active()
        self.assertEqual(active['template'], template)
        self.assertEqual(json.loads(active['parameters']), config)
        _, set_by = self._pointer()
        self.assertEqual(set_by, RECONCILE_PROVENANCE)

    def test_params_only_change_advances(self):
        """Config-only drift is the motivating case: a frozen install keeps a
        dated model ID the API will retire. A template-only comparison skips it.
        """
        template, config = self._shipped()
        stale = dict(config)
        stale['model'] = 'claude-3-dinosaur-20240101'
        self._drift(params=stale)
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(json.loads(self._active()['parameters']), config)

    def test_already_current_writes_nothing(self):
        before_versions = self._version_count()
        before_pointer = self._pointer()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._version_count(), before_versions)
        self.assertEqual(self._pointer(), before_pointer)

    def test_version_gate_blocks_a_second_run(self):
        self._drift(template='STALE')
        self._reset_stream()
        reconcile_seeded_prompts(self.brain)
        advanced_to = self._pointer()

        # Same version, run again: the gate must stop it dead.
        self._drift(template='STALE AGAIN')
        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._pointer(), advanced_to)
        self.assertEqual(self._active()['template'], 'STALE AGAIN')

    def test_two_consecutive_bumps_both_land(self):
        """The real fleet path: content moves, bump, content moves, bump."""
        self._drift(template='STALE ONE')
        self._reset_stream()
        reconcile_seeded_prompts(self.brain)
        first = self._pointer()[0]

        # Next release: shipped content changed again and the counter bumped.
        self._drift(template='STALE TWO')
        orig = interaction_seed.SEED_PROMPTS_VERSION
        try:
            interaction_seed.SEED_PROMPTS_VERSION = orig + 1
            reconcile_seeded_prompts(self.brain)
        finally:
            interaction_seed.SEED_PROMPTS_VERSION = orig

        second = self._pointer()[0]
        self.assertGreater(second, first)
        self.assertEqual(self._active()['template'], self._shipped()[0])
        self.assertEqual(
            read_schema_version(self.brain.logs_conn, 'logs_meta',
                                SEED_PROMPTS_VERSION_KEY), orig + 1)

    def test_residue_from_the_reverted_first_attempt_still_reconciles(self):
        """Version 1 is burned: the reverted attempt (dfc74ee) stamped it on
        real installs before being removed. An install carrying that row must
        still receive this generation, or the mechanism arrives dead on exactly
        the brains it was built for."""
        self.assertGreaterEqual(
            SEED_PROMPTS_VERSION, 2,
            'SEED_PROMPTS_VERSION 1 was consumed by the reverted attempt')
        self._drift(template='STALE')
        self._reset_stream()
        stamp_schema_version(self.brain.logs_conn, 'logs_meta',
                             SEED_PROMPTS_VERSION_KEY, 1)  # attempt-1 residue
        self.brain.logs_conn.commit()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._active()['template'], self._shipped()[0])

    def test_stream_is_stamped_so_it_runs_once_per_bump(self):
        self._reset_stream()
        reconcile_seeded_prompts(self.brain)
        self.assertEqual(
            read_schema_version(self.brain.logs_conn, 'logs_meta',
                                SEED_PROMPTS_VERSION_KEY),
            SEED_PROMPTS_VERSION)


class HandsOffTest(ReconcileTestBase):
    """Everything reconcile must refuse to touch."""

    def test_human_activation_is_never_published_over(self):
        self.brain.register_interaction(NAME, template='TOM WROTE THIS',
                                        parameters='{}', created_by='anchor')
        versions = self.brain.list_interaction_versions(NAME)
        self.brain.set_interaction_active(NAME, versions[-1]['version'],
                                          set_by='anchor')
        before = self._pointer()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._pointer(), before)
        self.assertEqual(self._active()['template'], 'TOM WROTE THIS')

    def test_dormant_human_candidate_freezes_the_name(self):
        """`trace_recording` sits at active=1 with a dormant v2 exactly like
        this. A registered-but-inactive version is a deployment decision in
        progress — often an eval gate — and must never be jumped over."""
        self._drift(template='STALE')
        self.brain.register_interaction(NAME, template='EVAL CANDIDATE',
                                        parameters='{}', created_by='anchor')
        before = self._pointer()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._pointer(), before)
        self.assertEqual(self._active()['template'], 'STALE')

    def test_backstop_pointer_with_a_human_version_is_not_pristine(self):
        """The pointer backstop points at MAX(version), which on a pre-split
        install can be a version a human registered by hand. Reserved, but not
        the shipped default whenever more than one version exists."""
        self.assertIn(BACKSTOP_PROVENANCE, SYSTEM_PROVENANCE)
        self.assertNotIn(BACKSTOP_PROVENANCE, PRISTINE_ACTIVATIONS)

        # A second version on record — a human registered something here.
        self.brain.register_interaction(NAME, template='HUMAN VERSION',
                                        parameters='{}', created_by='anchor')
        versions = self.brain.list_interaction_versions(NAME)
        self.brain.set_interaction_active(NAME, versions[-1]['version'],
                                          set_by='anchor')
        self.brain.logs_conn.execute(
            'UPDATE interaction_active SET set_by = ? WHERE name = ?',
            (BACKSTOP_PROVENANCE, NAME))
        self.brain.logs_conn.commit()
        before = self._pointer()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._pointer(), before)
        self.assertEqual(self._active()['template'], 'HUMAN VERSION')

    def test_backstop_pointer_with_a_single_version_is_pristine(self):
        """The oldest installs: pre-`interaction_active`, never hand-edited.

        With exactly one version on record nothing but the seed ever wrote for
        this name, so the backstop's MAX(version) pointer IS the shipped
        default. These installs are frozen longest and are the reason the
        mechanism exists; excluding them would miss the target population.
        """
        self._drift(template='FROZEN SINCE INSTALL DAY')
        self.assertEqual(self._version_count(), 1)
        self.brain.logs_conn.execute(
            'UPDATE interaction_active SET set_by = ? WHERE name = ?',
            (BACKSTOP_PROVENANCE, NAME))
        self.brain.logs_conn.commit()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._active()['template'], self._shipped()[0])
        self.assertEqual(self._pointer()[1], RECONCILE_PROVENANCE)

    def test_pristine_set_is_a_subset_of_the_reserved_vocabulary(self):
        """Every pristine value must also be refused at the MCP door — else a
        caller could mint one and have its own write read as untouched."""
        self.assertTrue(set(PRISTINE_ACTIVATIONS) <= set(SYSTEM_PROVENANCE),
                        '%s not all reserved' % (PRISTINE_ACTIVATIONS,))

    def test_only_scouts_production_actually_runs_are_shipped(self):
        """`quote` and `temporal` are registered but never mustered.

        Production runs the lived arm (BRAIN_S1E_LIVED_SEQUENCE=1) and encode.py
        excludes those two there, so advancing their prompts would push content
        for machinery that cannot fire. Asserted against the exclusion list in
        the code rather than a hardcoded pair, so re-enabling a scout makes this
        test demand it be shipped again.
        """
        import re
        from servers.scales.s1 import encode
        src = inspect.getsource(encode)
        m = re.search(r'exclude_scouts=\(\((.*?)\) if lived', src)
        self.assertIsNotNone(m, 'lived-arm scout exclusions not found in encode')
        excluded = set(re.findall(r"'([a-z_]+)'", m.group(1)))
        self.assertTrue(excluded, 'expected at least one excluded scout')

        shipped = shipped_prompts()
        for scout in excluded:
            self.assertNotIn('s1_scout_%s' % scout, shipped,
                             'scout %r never runs on the production arm' % scout)
        # And the live scout IS shipped — the exclusion must not swallow it.
        self.assertIn('s1_scout_facts', shipped)
        self.assertNotIn('facts', excluded)

    def test_config_only_interactions_are_out_of_scope(self):
        # Names verified to exist in seed_interactions — asserting a name that
        # was never an interaction ('signal' vs 'signal_assembler') would pass
        # no matter what shipped_prompts() contained.
        # `surface` left this list in generation 3: its template + layout
        # config are live reads, so it ships like the encoder prompts.
        # boot/signal_assembler left in the override migration (Step 2): their
        # configs were reader-less and the seed no longer registers them.
        seeded = {i['name'] for i in self.brain.list_interactions()}
        for name in ('trace_recording', 'scopes', 's2_community'):
            self.assertIn(name, seeded, '%s should be a seeded interaction' % name)
            self.assertNotIn(name, shipped_prompts())
        self.assertIn('surface', shipped_prompts(),
                      'surface ships template + layout config since gen 3')


class CrashResidueTest(ReconcileTestBase):

    def test_dangling_reconcile_version_is_adopted_not_duplicated(self):
        """A reconcile that registered and died before flipping the pointer.

        Re-adopting the version it left keeps retries from stacking a duplicate
        on every boot.
        """
        template, config = self._shipped()
        self._drift(template='STALE')
        residue = self.brain.register_interaction(
            NAME, template=template, parameters=json.dumps(config),
            created_by=RECONCILE_PROVENANCE)
        count_before = self._version_count()
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        self.assertEqual(self._pointer(),
                         (residue['version'], RECONCILE_PROVENANCE))
        self.assertEqual(self._version_count(), count_before,
                         'adoption must not register another version')

    def test_stale_reconcile_residue_is_superseded_not_adopted(self):
        """Residue from an OLDER shipped version must not be activated."""
        template, config = self._shipped()
        self._drift(template='STALE')
        residue = self.brain.register_interaction(
            NAME, template='PREVIOUS SHIPPED TEXT', parameters='{}',
            created_by=RECONCILE_PROVENANCE)
        self._reset_stream()

        reconcile_seeded_prompts(self.brain)

        active_v, set_by = self._pointer()
        self.assertGreater(active_v, residue['version'])
        self.assertEqual(set_by, RECONCILE_PROVENANCE)
        self.assertEqual(self._active()['template'], template)

    def test_failure_leaves_the_stream_unstamped_for_retry(self):
        self._drift(template='STALE')
        self._reset_stream()
        original = interaction_seed._reconcile_pristine_prompts

        def boom(_brain):
            raise RuntimeError('simulated mid-reconcile failure')

        try:
            interaction_seed._reconcile_pristine_prompts = boom
            reconcile_seeded_prompts(self.brain)  # must not raise
        finally:
            interaction_seed._reconcile_pristine_prompts = original

        self.assertEqual(
            read_schema_version(self.brain.logs_conn, 'logs_meta',
                                SEED_PROMPTS_VERSION_KEY), 0)
        # The retry succeeds.
        reconcile_seeded_prompts(self.brain)
        self.assertEqual(self._active()['template'], self._shipped()[0])


class CallSiteTest(ReconcileTestBase):
    """Requirement 8: reconcile runs from the daemon, never from `Brain()`."""

    def test_brain_init_does_not_reconcile(self):
        """Eval corpora, IsolatedBrain copies, tests, and the boot_brain
        fallback all construct a Brain directly. Reconciling there would mutate
        frozen corpora and race two processes on UNIQUE(name, version)."""
        self._drift(template='FROZEN CORPUS CONTENT')
        self._reset_stream()

        from servers.brain import Brain
        reopened = Brain(self.brain.db_path)
        try:
            row = reopened.get_interaction(NAME) or {}
            self.assertEqual(row.get('template'), 'FROZEN CORPUS CONTENT')
            self.assertEqual(
                read_schema_version(reopened.logs_conn, 'logs_meta',
                                    SEED_PROMPTS_VERSION_KEY), 0,
                'constructing a Brain must not run the reconcile')
        finally:
            reopened.close() if hasattr(reopened, 'close') else None

    def test_daemon_load_brain_calls_reconcile(self):
        """The wiring itself — without it the mechanism ships dead."""
        import inspect
        from servers import daemon_server
        src = inspect.getsource(daemon_server.BrainDaemon._load_brain)
        self.assertIn('reconcile_seeded_prompts', src)


class ReservedProvenanceTest(ReconcileTestBase):
    """The MCP door may not mint the values reconcile reads."""

    def test_set_active_refuses_reserved_set_by(self):
        from servers.dispatch_observability import _handle_set_interaction_active
        for reserved in SYSTEM_PROVENANCE:
            result = _handle_set_interaction_active(
                self.brain, {'name': NAME, 'version': 1, 'set_by': reserved},
                None)
            self.assertFalse(result['ok'], reserved)
            self.assertIn('reserved', result['error'])

    def test_register_refuses_reserved_created_by(self):
        from servers.dispatch_observability import _handle_register_interaction
        for reserved in SYSTEM_PROVENANCE:
            result = _handle_register_interaction(
                self.brain,
                {'name': NAME, 'template': 'x', 'created_by': reserved}, None)
            self.assertFalse(result['ok'], reserved)
            self.assertIn('reserved', result['error'])

    def test_normal_provenance_still_works(self):
        from servers.dispatch_observability import _handle_register_interaction
        result = _handle_register_interaction(
            self.brain, {'name': NAME, 'template': 'x', 'created_by': 'anchor'},
            None)
        self.assertTrue(result['ok'])


class ShippedContentFingerprintTest(unittest.TestCase):
    """A forgotten bump silently rebuilds the freeze — nothing else fails on it.

    This asserts a RELATIONSHIP, not a snapshot: changed content requires a
    version above every version already shipped. A snapshot pin cannot express
    that — pinning `(version, hash)` as one value lets you repair a failure by
    pasting the new hash next to the OLD version, which goes green while no
    install ever receives the change. The guard would then be talking you out of
    the bump it exists to enforce.

    HISTORY is append-only. Bumping is a deployment decision, so recording it is
    a deliberate edit in a reviewable diff — never a regenerated line.
    """

    # SEED_PROMPTS_VERSION -> fingerprint of the content THAT version shipped.
    # APPEND a row when you bump. Never edit an existing row: an old row is the
    # record of what a released generation contained, and installs stamped at
    # that version are relying on it.
    HISTORY = {
        2: '42b2eb3fceb655caed9b75a014fb845ed04d9546f27b0cd504f38545cc957a23',
        3: '1c9eb4451651c2b5d8af205c0df1e3ee8309203a1f856e24da41672eba8336cf',
        4: '7378aa7d24900cbdbf0ba5ae96adffbe9cf6250235fd156fa6a7ed064b5dff0f',
        5: '265385cc56e1167b36d1a78176a5294d5ad8ad0846278536434ac43e0f37344a',
        6: '64df5f53cd40f7ef085fe63a7a2665ae9db939be02fdab761d52332d81740c05',
        7: '182761f6e8d734c501f35408795acfc298822aeb44d97cfb58c7333bf1c0418f',
        8: '20cc74ce796388f201956c58c5f8913e107a24fd21cc45fe72f9e63284d578e8',
        9: 'b6ac54c0e9c3895ffaf9e0529b928b1568d004f0d4dcdb77ee2b7281496b2747',
    }

    @staticmethod
    def _fingerprint():
        h = hashlib.sha256()
        for name, (template, config) in sorted(shipped_prompts().items()):
            h.update(name.encode())
            h.update(template.encode())
            h.update(json.dumps(config, sort_keys=True).encode())
        return h.hexdigest()

    def test_shipped_content_matches_its_recorded_generation(self):
        actual = self._fingerprint()
        recorded = self.HISTORY.get(SEED_PROMPTS_VERSION)

        if recorded == actual:
            return  # this generation is recorded, and unchanged since

        highest = max(self.HISTORY)
        if recorded is not None:
            self.fail(
                '\nShipped prompt content changed under SEED_PROMPTS_VERSION '
                '%d, which is already a released generation.\n'
                'Do NOT update the HISTORY row — installs stamped at %d relied '
                'on it.\n'
                'BUMP SEED_PROMPTS_VERSION to %d and append:\n'
                '    %d: %r,\n'
                % (SEED_PROMPTS_VERSION, SEED_PROMPTS_VERSION, highest + 1,
                   highest + 1, actual))

        self.assertGreater(
            SEED_PROMPTS_VERSION, highest,
            '\nSEED_PROMPTS_VERSION %d is not above the highest shipped '
            'generation (%d).\nA new generation must move the counter forward — '
            'forward-only logic means installs never revisit a version.\n'
            % (SEED_PROMPTS_VERSION, highest))

        self.fail(
            '\nSEED_PROMPTS_VERSION %d is a new generation with no recorded '
            'fingerprint.\nAppend it to HISTORY:\n    %d: %r,\n'
            % (SEED_PROMPTS_VERSION, SEED_PROMPTS_VERSION, actual))


if __name__ == '__main__':
    unittest.main()
