"""Tests for interactions as living boundaries.

Interactions hold prompt text (for LLM boundaries) and config JSON
(for all boundaries). Code reads from the interactions table at runtime.
Higher scales write new versions to evolve behavior.

Run: python3 -m pytest tests/test_interactions_runtime.py -v
"""
import json
import pytest
from tests.isolated_brain import IsolatedBrain


# ═══════════════════════════════════════════════════════
# Seeding
# ═══════════════════════════════════════════════════════

class TestInteractionSeeding:
    """Verify seed_interactions populates all 6 boundaries."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._interaction_dal
            # Clear existing interactions for clean test
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.commit()
            yield

    def test_seed_creates_all_interactions(self):
        """Seed registers the core S1/S2 interactions.

        Asserts a core subset is present rather than exact equality —
        the seed list grows as new boundaries get learnable prompts
        (scouts were added 2026-04-23, healer earlier). A subset check
        keeps this from breaking on every legitimate addition.
        """
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        all_interactions = self.dal.list_all()
        names = {i['name'] for i in all_interactions}
        core_required = {'surface', 's1e', 's2_community',
                         's2_community_enrichment',
                         's2_consolidation_enrichment',
                         's2_healer'}
        missing = core_required - names
        assert not missing, "Seed missing core interactions: %s" % missing

    def test_seed_is_idempotent(self):
        """Running seed twice doesn't create duplicates.

        Idempotency = re-seeding adds NOTHING: same names, same version
        counts. (Not "exactly one version" — trace_recording deliberately
        seeds two: v1 normal active, v2 debug dormant, per
        docs/TRACE-MODES-DESIGN.md "modes as config versions".)
        """
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        before = {i['name']: i['total_versions'] for i in self.dal.list_all()}
        seed_interactions(self.brain)
        after = {i['name']: i['total_versions'] for i in self.dal.list_all()}
        assert after == before

    def test_surface_has_prompt_and_config(self):
        """Fresh brains seed a real surface prompt whose layout the runtime
        renderer actually implements, and whose template teaches the same
        grammar that layout renders. Template + layout flip atomically —
        that's why layout rides in the interaction config — so a seed that
        pairs an XML template with a legacy layout (or names a layout
        build_surface_prompt doesn't implement) must fail here.

        Deliberately does NOT pin config keys the runtime never reads
        (max_candidates etc. lived here until 2026-07-15; prompt-size
        limits come from surface_contract.SURFACE, not this config)."""
        from servers.interaction_seed import seed_interactions
        from servers.scales.s1.surface_contract import build_surface_prompt
        seed_interactions(self.brain)
        surface = self.dal.get_active('surface')
        assert surface is not None
        template = surface['template']
        assert len(template) > 100  # real prompt, not placeholder

        config = json.loads(surface['parameters'])
        layout = config.get('layout', 'legacy')

        # Render one candidate through the seeded layout — an unknown
        # layout value silently falls back to legacy rendering and fails
        # the grammar checks below (behavior-based, no layout whitelist).
        cand = {'id': 'a' * 32, 'title': 'Seeded check', 'type': 'fact',
                'content': 'body', 'score': 0.9,
                'created_at': '2026-07-01T00:00:00+00:00'}
        prompt, _ = build_surface_prompt([cand], 'a message', layout=layout)
        if layout == 'xml_v13':
            assert '<candidate id="aaaaaaaa"' in prompt, \
                'seeded layout did not reach the XML renderer'
            assert '<candidate' in template, \
                'xml_v13 layout paired with a template that never ' \
                'teaches the <candidate> grammar'
        else:
            assert '<candidate' not in prompt
            assert '<candidate' not in template, \
                'XML-speaking template paired with legacy layout — ' \
                'template and layout must flip together'

    def test_encoding_agent_has_prompt_and_config(self):
        """The S1 encoder interaction has a real prompt + config.

        Renamed from `encoding_agent` to `s1e` when scale-name conventions
        landed; runtime reads 's1e' (see scales/s1/encode.py).
        """
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        enc = self.dal.get_active('s1e')
        assert enc is not None
        assert len(enc['template']) > 100  # real prompt
        config = json.loads(enc['parameters'])
        assert 'max_messages' in config
        assert 'max_rounds' in config

    def test_code_boundaries_have_empty_template(self):
        """voice_surface, boot, pre_edit, signal_assembler have no LLM prompt."""
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        for name in ('voice_surface', 'boot', 'pre_edit', 'signal_assembler'):
            interaction = self.dal.get_active(name)
            assert interaction['template'] == '', \
                "%s should have empty template, got %d chars" % (name, len(interaction['template']))

    def test_all_have_config(self):
        """Every boundary has a config dict, even code-only ones."""
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        for interaction in self.dal.list_all():
            active = self.dal.get_active(interaction['name'])
            config = json.loads(active['parameters'])
            assert isinstance(config, dict), "%s config is not a dict" % interaction['name']
            assert len(config) > 0, "%s config is empty" % interaction['name']


# ═══════════════════════════════════════════════════════
# Brain methods
# ═══════════════════════════════════════════════════════

class TestBrainInteractionMethods:
    """Verify Brain.get_interaction_config() and get_interaction_prompt()."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.commit()
            yield

    def test_get_config_returns_dict(self):
        self.brain._interaction_dal.register(
            'test_boundary', template='', parameters=json.dumps({'x': 10, 'y': 20}))
        config = self.brain.get_interaction_config('test_boundary')
        assert config == {'x': 10, 'y': 20}

    def test_get_config_missing_returns_empty(self):
        config = self.brain.get_interaction_config('nonexistent')
        assert config == {}

    def test_get_prompt_returns_text(self):
        self.brain._interaction_dal.register(
            'test_llm', template='You are a test judge. Select wisely.',
            parameters=json.dumps({}))
        prompt = self.brain.get_interaction_prompt('test_llm')
        assert prompt == 'You are a test judge. Select wisely.'

    def test_get_prompt_missing_returns_empty(self):
        prompt = self.brain.get_interaction_prompt('nonexistent')
        assert prompt == ''

    def test_register_v2_does_not_auto_activate(self):
        """After registering v2, runtime still reads v1 until set_active is called.

        This locks the active-version semantic (2026-05-10): registration
        creates a version row but does NOT change the runtime pointer.
        """
        self.brain._interaction_dal.register(
            'evolving', template='', parameters=json.dumps({'threshold': 0.5}))
        # v1 should be auto-active
        config = self.brain.get_interaction_config('evolving')
        assert config['threshold'] == 0.5

        # Register v2 — must NOT change what runtime reads
        self.brain._interaction_dal.register(
            'evolving', template='', parameters=json.dumps({'threshold': 0.8}),
            created_by='sleep:s3')
        config = self.brain.get_interaction_config('evolving')
        assert config['threshold'] == 0.5, \
            "v2 register should NOT auto-activate; runtime should still see v1"

        # Now explicitly activate v2 — runtime flips
        self.brain._interaction_dal.set_active('evolving', 2, set_by='test')
        config = self.brain.get_interaction_config('evolving')
        assert config['threshold'] == 0.8

        # Rollback to v1 by re-activating
        self.brain._interaction_dal.set_active('evolving', 1, set_by='test')
        config = self.brain.get_interaction_config('evolving')
        assert config['threshold'] == 0.5

    def test_set_active_rejects_unknown_version(self):
        """set_active raises when target version isn't registered."""
        self.brain._interaction_dal.register(
            'one_version', template='', parameters=json.dumps({'k': 1}))
        with pytest.raises(ValueError):
            self.brain._interaction_dal.set_active('one_version', 99, set_by='test')
        with pytest.raises(ValueError):
            self.brain._interaction_dal.set_active('never_registered', 1, set_by='test')


# ═══════════════════════════════════════════════════════
# Versioning and comparison
# ═══════════════════════════════════════════════════════

class TestInteractionVersioning:
    """Verify version tracking and lineage."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._interaction_dal
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.commit()
            yield

    def test_auto_increment_version(self):
        r1 = self.dal.register('test', template='v1', parameters='{}')
        r2 = self.dal.register('test', template='v2', parameters='{}')
        assert r1['version'] == 1
        assert r2['version'] == 2

    def test_old_versions_preserved(self):
        self.dal.register('test', template='v1 prompt', parameters='{}')
        self.dal.register('test', template='v2 prompt', parameters='{}')
        v1 = self.dal.get_version('test', 1)
        v2 = self.dal.get_version('test', 2)
        assert v1['template'] == 'v1 prompt'
        assert v2['template'] == 'v2 prompt'

    def test_created_by_tracked(self):
        self.dal.register('test', template='', parameters='{}', created_by='anchor')
        self.dal.register('test', template='', parameters='{}', created_by='sleep:s3')
        v1 = self.dal.get_version('test', 1)
        v2 = self.dal.get_version('test', 2)
        assert v1['created_by'] == 'anchor'
        assert v2['created_by'] == 'sleep:s3'

    def test_parent_version_linked(self):
        self.dal.register('test', template='', parameters='{}')
        self.dal.register('test', template='', parameters='{}')
        v2 = self.dal.get_version('test', 2)
        assert v2.get('parent_version') == 1


# ═══════════════════════════════════════════════════════
# Trace linkage
# ═══════════════════════════════════════════════════════

class TestInteractionTraceLinkage:
    """Verify traces reference interaction versions."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._interaction_dal
            self.trace_dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_trace_includes_interaction_id(self):
        """Trace events reference which interaction version produced them."""
        result = self.dal.register('judge', template='test', parameters='{}')
        interaction_id = result['id']

        self.trace_dal.append(
            chain_id='s1r-test-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='test output',
            interaction_id=interaction_id, session_id='test')

        chain = self.trace_dal.get_chain('s1r-test-1')
        assert len(chain) == 1
        # interaction_id should be queryable
        row = self.brain.logs_conn.execute(
            'SELECT interaction_id FROM trace_events WHERE chain_id = ?',
            ('s1r-test-1',)).fetchone()
        assert row[0] == interaction_id

    def test_compare_versions_via_traces(self):
        """Two interaction versions produce traces that can be compared."""
        r1 = self.dal.register('judge', template='v1', parameters='{}')
        r2 = self.dal.register('judge', template='v2', parameters='{}')

        # v1 produced a recall
        self.trace_dal.append(
            chain_id='s1r-v1-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='v1 output',
            interaction_id=r1['id'], session_id='test')

        # v2 produced a recall
        self.trace_dal.append(
            chain_id='s1r-v2-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='v2 output',
            interaction_id=r2['id'], session_id='test')

        # Query traces by interaction version
        v1_traces = self.brain.logs_conn.execute(
            'SELECT summary FROM trace_events WHERE interaction_id = ?',
            (r1['id'],)).fetchall()
        v2_traces = self.brain.logs_conn.execute(
            'SELECT summary FROM trace_events WHERE interaction_id = ?',
            (r2['id'],)).fetchall()

        assert len(v1_traces) == 1
        assert len(v2_traces) == 1
        assert v1_traces[0][0] == 'v1 output'
        assert v2_traces[0][0] == 'v2 output'


# ═══════════════════════════════════════════════════════
# Fallback behavior
# ═══════════════════════════════════════════════════════

class TestInteractionFallback:
    """Verify system works when interactions table is empty."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.commit()
            yield

    def test_config_fallback_empty_dict(self):
        """When no interactions exist, get_interaction_config returns {}."""
        assert self.brain.get_interaction_config('judge') == {}

    def test_prompt_fallback_empty_string(self):
        """When no interactions exist, get_interaction_prompt returns ''."""
        assert self.brain.get_interaction_prompt('judge') == ''
