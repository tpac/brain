"""Contract tests for the seed-prompt sync discipline.

Invariant: every prompt listed in `sync_prompts.SEED_PROMPTS` lives in a
sibling .py file and must export the constant that list names, matching what
`interaction_seed.py` expects to register. The roster is DERIVED from
SEED_PROMPTS, so adding a prompt there covers it here automatically.

If someone registers a new version of a prompt via register_interaction
and forgets to run `./dev python3 -m servers.tools.sync_prompts`, these
tests won't catch the drift (that would require talking to a live DB).
What they DO catch:
  - A seed file with no SYSTEM_PROMPT constant (import error).
  - An empty SYSTEM_PROMPT (seed produces a broken fresh brain).
  - The sync tool's round-trip escape bugs (render → parse → compare).
  - A freshly seeded brain not registering every prompt in SEED_PROMPTS.

Run: ./dev python3 -m pytest tests/test_prompt_sync.py -v
"""
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.tools.sync_prompts import SEED_PROMPTS as _SYNC_SEED_PROMPTS


# The seed files under contract — DERIVED from sync_prompts.SEED_PROMPTS, never
# hand-listed. A hand-maintained copy drifted: it omitted `s2_aspects`, so that
# prompt had zero seed-shape coverage while CLAUDE.md claimed this test enforced
# "fresh brains must seed every prompt in SEED_PROMPTS". Deriving makes the claim
# true and makes a new prompt covered the moment it is registered for sync.
#
# The constant name comes from the tuple too — a seed file that exports something
# other than SYSTEM_PROMPT should fail on its own name, not on a misleading
# "missing SYSTEM_PROMPT".
SEED_FILES = []
for _name, _rel_path, _constant in _SYNC_SEED_PROMPTS:
    assert _rel_path.endswith('.py'), (
        'SEED_PROMPTS path is not a module: %r' % _rel_path)
    SEED_FILES.append(
        (_name, _rel_path[:-len('.py')].replace('/', '.'), _constant))


class TestSeedFileShape:
    """Each seed .py file must have a non-empty SYSTEM_PROMPT string."""

    @pytest.mark.parametrize('name,module,constant', SEED_FILES)
    def test_exports_system_prompt(self, name, module, constant):
        import importlib
        m = importlib.import_module(module)
        assert hasattr(m, constant), f'{module} missing {constant} constant'
        prompt = getattr(m, constant)
        assert isinstance(prompt, str), f'{module}.{constant} must be str, got {type(prompt)}'
        assert len(prompt) > 100, (
            f'{module}.{constant} is suspiciously short ({len(prompt)} chars). '
            f'Seed files should contain real encoder prompts, not placeholders. '
            f'If you just registered a new version, run: '
            f'./dev python3 -m servers.tools.sync_prompts')

    @pytest.mark.parametrize('name,module,constant', SEED_FILES)
    def test_seed_role_in_docstring(self, name, module, constant):
        """Docstring must flag the file's seed-only role so editors don't
        try to change prompt behavior by touching the file directly."""
        import importlib
        m = importlib.import_module(module)
        doc = (m.__doc__ or '').lower()
        # Any of these phrases is fine — we just want the intent stated.
        assert ('seed' in doc or 'authoritative' in doc), (
            f'{module} docstring should state this is a seed-only file '
            f'and that the DB is authoritative. Re-run sync_prompts to regenerate.')


class TestSeedConfigCarriesWireSchema:
    """The facts scout's Structured Outputs schema must reach the seed config.

    `scouts/base.py` reads `params['output_schema']` off the interaction, not
    off the contract module — so the schema is inert unless the seed dict
    carries it. Lives here rather than in test_scout_contract.py because the
    subject is what a fresh brain gets seeded, which is this file's concern.
    """

    def test_facts_config_carries_the_contract_schema(self):
        from servers.scales.s1.scouts.contract import (
            FACTS_OUTPUT_SCHEMA, SCOUT_FACTS_INTERACTION_DEFAULT)
        # Identity, not equality: the default must ship the ACTIVE-tracking
        # constant itself. A copy could drift from it silently, and the
        # by-reference embed is what makes an edit to the constant a
        # deployment — see the invariant note at FACTS_OUTPUT_SCHEMA.
        assert SCOUT_FACTS_INTERACTION_DEFAULT['output_schema'] is FACTS_OUTPUT_SCHEMA

    def test_only_the_mustered_scout_ships_a_schema(self):
        """quote/temporal are excluded from the production arm
        (`exclude_scouts=('quote', 'temporal')`), so neither should acquire a
        wire schema without that exclusion changing first."""
        from servers.scales.s1.scouts import contract as scouts_contract
        for name in ('SCOUT_QUOTE_INTERACTION_DEFAULT',
                      'SCOUT_TEMPORAL_INTERACTION_DEFAULT'):
            assert 'output_schema' not in getattr(scouts_contract, name), name


class TestFreshBrainSeeding:
    """seed_interactions() must register every prompt in SEED_PROMPTS on a
    fresh brain. Regression guard against reintroducing the 'consolidation
    not seeded' bug that bit us on 2026-04-19."""

    def test_every_seed_prompt_is_seeded(self, tmp_path):
        from servers.brain import Brain
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            seeded = {i['name'] for i in brain._interaction_dal.list_all()}
            for name, _module, _const in SEED_FILES:
                assert name in seeded, (
                    f'Fresh brain missing {name!r}. seed_interactions() '
                    f'didn\'t register it. See servers/interaction_seed.py.')
                # Also assert the SEEDED ROW's template is non-empty — the
                # resolver would serve the code default even for a blank row,
                # so reading get_interaction_prompt here would be vacuous.
                row = brain._interaction_dal.get_version(name, 1) or {}
                tmpl = row.get('template')
                assert tmpl and len(tmpl) > 100, (
                    f'{name} was seeded with empty/tiny template ({len(tmpl or "")} chars).')
        finally:
            brain.close()

    # test_seed_is_idempotent — REMOVED (redundant). The seed-idempotency invariant
    # is owned by test_interactions_runtime.py::TestInteractionSeeding::test_seed_is_idempotent,
    # which seeds twice from a cleared table and asserts total_versions==1 across ALL
    # interactions (a stronger, cleaner check than this fresh-brain max_version diff,
    # and one that covers the encoder prompts too). CLAUDE.md's enumerated
    # test_prompt_sync contract deliberately omits idempotency — this file owns
    # SYSTEM_PROMPT shape, fresh-brain seeding, active-version mirroring, and the
    # no-clobber guarantee.

    def test_sync_grabs_active_not_latest_version(self, tmp_path):
        """`sync_prompts._fetch_active` must mirror the ACTIVE version, not
        the highest registered one.

        Regression guard: if someone changes the fetch back to ORDER BY
        version DESC, dormant candidates (e.g. an eval-gated v22 registered
        but not yet activated) would leak into the seed file and fresh
        brains would skip the eval gate. This test locks the active-version
        semantics.
        """
        from servers.brain import Brain
        from servers.tools.sync_prompts import _fetch_active
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            # v1 was seeded dormant by Brain.__init__ (registration never
            # activates). With NO pointer, _fetch_active must return None —
            # "no override deployed, seed file is authoritative" — and NEVER
            # fall back to a registered version. This is the no-pointer half
            # of the dormant-leak guard: the old MAX(version) fallback here
            # would have mirrored an un-eval'd dormant candidate into the
            # seed .py, which interaction_defaults imports as the CODE
            # DEFAULT — a fleet-wide eval-gate bypass.
            assert _fetch_active(brain.logs_conn, 's1e') is None

            # Deploy v1 so there is an ACTIVE version to mirror.
            brain._interaction_dal.set_active('s1e', 1, set_by='test')
            initial_v1 = brain.get_interaction_prompt('s1e')
            assert initial_v1

            # Register v2 as DORMANT — do NOT activate.
            dormant_v2 = 'DORMANT v2 — must not leak into seed.\n' * 20
            result = brain._interaction_dal.register(
                name='s1e', template=dormant_v2,
                parameters='{}', created_by='test')
            assert result['version'] == 2

            # Fetch via sync's helper. Must return v1 (active), NOT v2 (latest).
            fetched = _fetch_active(brain.logs_conn, 's1e')
            assert fetched is not None
            assert fetched['version'] == 1, (
                f'_fetch_active returned v{fetched["version"]} — must return '
                f'the ACTIVE version (v1), not the highest registered (v2). '
                f'Dormant candidates leaking into the seed bypasses eval gates.')
            assert fetched['template'] == initial_v1

            # Now activate v2. Fetch must follow.
            brain._interaction_dal.set_active('s1e', 2, set_by='test')
            fetched_after = _fetch_active(brain.logs_conn, 's1e')
            assert fetched_after['version'] == 2
            assert fetched_after['template'] == dormant_v2
        finally:
            brain.close()

    def test_seed_doesnt_override_externally_registered_version(self, tmp_path):
        """After an external register + set_active bumps a prompt to v2,
        a subsequent seed call must leave v2 as the active version — not
        clobber with v1 content from the .py file. This is the core
        guarantee that lets S3 evolve prompts without fighting the seed.

        Note: under the active-version model (2026-05-10), register alone
        doesn't change runtime — must call set_active. This test verifies
        the FULL externally-evolved path survives seed re-runs.
        """
        from servers.brain import Brain
        from servers.interaction_seed import seed_interactions
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            # v1 is seeded by Brain.__init__ and auto-activated.
            v1_prompt = brain.get_interaction_prompt('s1e')
            assert v1_prompt

            # Simulate an external update (operator or S3): register + activate.
            custom_v2 = 'CUSTOM v2 — do not overwrite me with the .py seed.\n' * 20
            result = brain._interaction_dal.register(
                name='s1e', template=custom_v2,
                parameters='{}', created_by='test')
            brain._interaction_dal.set_active('s1e', result['version'], set_by='test')

            # Re-run seed. Must NOT revert active back to v1 or bump it.
            seed_interactions(brain)
            active = brain.get_interaction_prompt('s1e')
            assert active == custom_v2, (
                'Seed clobbered an externally-activated v2. The guard in '
                "seed_interactions() must skip any interaction already in the DB.")
        finally:
            brain.close()


class TestSyncComparison:
    """The DB→.py comparison itself: what counts as drift, and what a repair
    is allowed to touch.

    The comparison has two independent halves with different repairs, and the
    distinction is load-bearing:
      · BODY drift  → the seed would boot a fresh brain on the wrong prompt.
                      Repair is a full regenerate.
      · HEADER drift → the body is right but the `Last sync:` line lies about
                      which version it mirrors (what a hand-edited-then-
                      registered seed produces). Repair MUST be a one-line
                      patch: the rest of a seed docstring can be hand-written
                      (that a prompt is a dormant fallback, where the real code
                      lives) and a regenerate silently deletes it.
    """

    def _interaction(self, template, version):
        return {'template': template, 'version': version,
                'parameters': '{}', 'created_by': 'test',
                'created_at': '2026-01-01T00:00:00.000000+00:00'}

    def _write(self, path, text):
        with open(path, 'w') as f:
            f.write(text)

    def test_header_repair_preserves_hand_written_docs(self):
        """A stale version line is patched in place; hand-written prose stays."""
        from servers.tools.sync_prompts import (
            _patch_header_version, _read_header_version, _render_py)
        inter = self._interaction('BODY', 7)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'seed.py')
            hand_written = (
                '"""Seed for interaction `x` — DB is authoritative at runtime.\n'
                '\n'
                'HAND-WRITTEN: this prompt is a dormant fallback; the real work\n'
                'lives in scouts/temporal.py.\n'
                '\n'
                'Last sync: DB v3 (2020-01-01T00:00:00, by someone).\n'
                '"""\n'
                '\n'
                'SYSTEM_PROMPT = """BODY"""\n')
            self._write(path, hand_written)
            assert _read_header_version(path) == 3

            assert _patch_header_version(path, inter) is True
            after = open(path).read()
            assert 'HAND-WRITTEN: this prompt is a dormant fallback' in after, (
                'header repair regenerated the file and ate hand-written docs')
            assert 'lives in scouts/temporal.py' in after
            assert _read_header_version(path) == 7
            assert 'SYSTEM_PROMPT = """BODY"""' in after
            # and it is NOT the generic rendered template
            assert after != _render_py('x', 'SYSTEM_PROMPT', inter)

    def test_header_inserted_when_absent(self):
        """A docstring with no provenance line gets one, keeping its content."""
        from servers.tools.sync_prompts import (
            _patch_header_version, _read_header_version)
        inter = self._interaction('BODY', 2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'seed.py')
            self._write(path,
                        '"""Older seed with NO provenance line.\n'
                        '\n'
                        'KEEP ME: debugging pointer.\n'
                        '"""\n'
                        '\n'
                        'SYSTEM_PROMPT = """BODY"""\n')
            assert _read_header_version(path) is None
            assert _patch_header_version(path, inter) is True
            after = open(path).read()
            assert 'KEEP ME: debugging pointer.' in after
            assert _read_header_version(path) == 2

    def test_header_repair_is_idempotent(self):
        from servers.tools.sync_prompts import _patch_header_version
        inter = self._interaction('BODY', 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'seed.py')
            self._write(path,
                        '"""Doc.\n\nLast sync: DB v1 (x, by y).\n"""\n'
                        '\nSYSTEM_PROMPT = """BODY"""\n')
            assert _patch_header_version(path, inter) is True
            once = open(path).read()
            assert _patch_header_version(path, inter) is True
            assert open(path).read() == once

    def test_non_utf8_seed_does_not_crash_the_reader(self):
        """A mojibake seed reports as drift rather than aborting the run."""
        from servers.tools.sync_prompts import (
            _patch_header_version, _read_header_version)
        inter = self._interaction('BODY', 1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'seed.py')
            with open(path, 'wb') as f:
                f.write(b'"""Doc \xff\xfe bad bytes.\n\nLast sync: DB v1 (x, by y).\n"""\n')
            assert _read_header_version(path) is None      # drift, not a crash
            assert _patch_header_version(path, inter) is False   # falls back

    def test_render_py_is_byte_stable(self):
        """Same DB row renders identically — the comparison can't false-positive."""
        from servers.tools.sync_prompts import _render_py
        inter = self._interaction('BODY with \\n escapes and "quotes"', 9)
        a = _render_py('x', 'SYSTEM_PROMPT', inter)
        b = _render_py('x', 'SYSTEM_PROMPT', inter)
        assert a == b
        assert 'Last sync: DB v9' in a
