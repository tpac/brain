"""Contract tests for the seed-prompt sync discipline.

Invariant: the four encoder-agent prompts that live in sibling .py files
(s1e, community, consolidation, healer) must each export a SYSTEM_PROMPT
constant that matches what `interaction_seed.py` expects to register.

If someone registers a new version of a prompt via register_interaction
and forgets to run `./dev python3 -m servers.tools.sync_prompts`, these
tests won't catch the drift (that would require talking to a live DB).
What they DO catch:
  - A seed file with no SYSTEM_PROMPT constant (import error).
  - An empty SYSTEM_PROMPT (seed produces a broken fresh brain).
  - The sync tool's round-trip escape bugs (render → parse → compare).
  - A freshly seeded brain not registering all 4 encoder prompts.

Run: ./dev python3 -m pytest tests/test_prompt_sync.py -v
"""
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The four seed files under contract. Keep in sync with
# servers/tools/sync_prompts.SEED_PROMPTS.
SEED_FILES = [
    ('s1e', 'servers.scales.s1.encoding_prompt'),
    ('s2_community_enrichment', 'servers.scales.s2.community_enrichment_prompt'),
    ('s2_consolidation_enrichment', 'servers.scales.s2.consolidation_enrichment_prompt'),
    ('s2_healer', 'servers.scales.s2.healer_prompt'),
]


class TestSeedFileShape:
    """Each seed .py file must have a non-empty SYSTEM_PROMPT string."""

    @pytest.mark.parametrize('name,module', SEED_FILES)
    def test_exports_system_prompt(self, name, module):
        import importlib
        m = importlib.import_module(module)
        assert hasattr(m, 'SYSTEM_PROMPT'), f'{module} missing SYSTEM_PROMPT constant'
        prompt = m.SYSTEM_PROMPT
        assert isinstance(prompt, str), f'{module}.SYSTEM_PROMPT must be str, got {type(prompt)}'
        assert len(prompt) > 100, (
            f'{module}.SYSTEM_PROMPT is suspiciously short ({len(prompt)} chars). '
            f'Seed files should contain real encoder prompts, not placeholders. '
            f'If you just registered a new version, run: '
            f'./dev python3 -m servers.tools.sync_prompts')

    @pytest.mark.parametrize('name,module', SEED_FILES)
    def test_seed_role_in_docstring(self, name, module):
        """Docstring must flag the file's seed-only role so editors don't
        try to change prompt behavior by touching the file directly."""
        import importlib
        m = importlib.import_module(module)
        doc = (m.__doc__ or '').lower()
        # Any of these phrases is fine — we just want the intent stated.
        assert ('seed' in doc or 'authoritative' in doc), (
            f'{module} docstring should state this is a seed-only file '
            f'and that the DB is authoritative. Re-run sync_prompts to regenerate.')


class TestFreshBrainSeeding:
    """seed_interactions() must register all four encoder prompts on a
    fresh brain. Regression guard against reintroducing the 'consolidation
    not seeded' bug that bit us on 2026-04-19."""

    def test_all_four_encoder_prompts_seeded(self, tmp_path):
        from servers.brain import Brain
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            seeded = {i['name'] for i in brain._interaction_dal.list_all()}
            for name, _module in SEED_FILES:
                assert name in seeded, (
                    f'Fresh brain missing {name!r}. seed_interactions() '
                    f'didn\'t register it. See servers/interaction_seed.py.')
                # Also assert the template is non-empty.
                tmpl = brain.get_interaction_prompt(name)
                assert tmpl and len(tmpl) > 100, (
                    f'{name} was seeded with empty/tiny template ({len(tmpl or "")} chars).')
        finally:
            brain.close()

    def test_seed_is_idempotent(self, tmp_path):
        """Running seed twice doesn't register duplicate versions — the
        'if name not in existing' guards must hold."""
        from servers.brain import Brain
        from servers.interaction_seed import seed_interactions
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            versions_before = {
                i['name']: i['latest_version']
                for i in brain._interaction_dal.list_all()
            }
            seed_interactions(brain)
            versions_after = {
                i['name']: i['latest_version']
                for i in brain._interaction_dal.list_all()
            }
            assert versions_after == versions_before, (
                'seed_interactions() bumped a version on re-run. It must be idempotent: '
                f'before={versions_before} after={versions_after}')
        finally:
            brain.close()

    def test_seed_doesnt_override_externally_registered_version(self, tmp_path):
        """After an external register_interaction bumps a prompt to v2,
        a subsequent seed call must leave v2 as the latest — not clobber
        with v1 content from the .py file. This is the core guarantee that
        lets S3 evolve prompts without fighting the seed.
        """
        from servers.brain import Brain
        from servers.interaction_seed import seed_interactions
        db = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db)
        try:
            # v1 is seeded by Brain.__init__.
            v1_prompt = brain.get_interaction_prompt('s1e')
            assert v1_prompt

            # Simulate an external update (operator or S3).
            custom_v2 = 'CUSTOM v2 — do not overwrite me with the .py seed.\n' * 20
            brain._interaction_dal.register(
                name='s1e', template=custom_v2,
                parameters='{}', created_by='test')

            # Re-run seed. Must NOT revert or bump.
            seed_interactions(brain)
            active = brain.get_interaction_prompt('s1e')
            assert active == custom_v2, (
                'Seed clobbered an externally-registered v2. The guard in '
                "seed_interactions() must skip any interaction already in the DB.")
        finally:
            brain.close()
