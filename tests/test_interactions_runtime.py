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
            # Clear interactions AND pointers — a faithful fresh brain has
            # neither (seeding registers dormant rows; nothing activates).
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
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
        # Pin the multi-version seed's FIRST-seed shape too: a bug that
        # registers debug twice in one run would otherwise pass before==after.
        assert before['trace_recording'] == 2
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
        # Seeded rows are dormant (registration never activates) — read the
        # seeded content itself.
        surface = self.dal.get_version('surface', 1)
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
        """The S1 encoder interaction has a real prompt, and seeds the one
        config key the runtime actually reads.

        Renamed from `encoding_agent` to `s1e` when scale-name conventions
        landed; runtime reads 's1e' (see scales/s1/encode.py).

        Pins `effort` because `encode.py` reads it off this config and maps it
        to the API's output_config — drop it from the seed and fresh brains
        silently lose the encoder's effort setting.

        Deliberately does NOT pin config keys the runtime never reads
        (`max_messages`, `max_rounds` and friends lived here until the seed was
        aligned to the production-ACTIVE config; the encoder reads them from
        `encode_contract.ENCODING_AGENT`, not this config). Same reasoning as
        test_surface_has_prompt_and_config.
        """
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        enc = self.dal.get_version('s1e', 1)
        assert enc is not None
        assert len(enc['template']) > 100  # real prompt
        config = json.loads(enc['parameters'])
        assert 'effort' in config

    def test_retired_boundaries_are_not_seeded(self):
        """voice_surface, boot, pre_edit, signal_assembler carry no reader and
        no config default — the seed must not reintroduce their rows."""
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        seeded = {i['name'] for i in self.dal.list_all()}
        for name in ('voice_surface', 'boot', 'pre_edit', 'signal_assembler'):
            assert name not in seeded, \
                "%s is retired (zero readers) and must not be seeded" % name

    def test_all_have_config(self):
        """Every boundary has a config dict, even code-only ones."""
        from servers.interaction_seed import seed_interactions
        seed_interactions(self.brain)
        for interaction in self.dal.list_all():
            seeded = self.dal.get_version(interaction['name'], 1)
            config = json.loads(seeded['parameters'])
            assert isinstance(config, dict), "%s config is not a dict" % interaction['name']
            assert len(config) > 0, "%s config is empty" % interaction['name']


# ═══════════════════════════════════════════════════════
# Brain methods
# ═══════════════════════════════════════════════════════

class TestBrainInteractionMethods:
    """Verify the resolver accessors: get_interaction_config() /
    get_interaction_prompt() overlay the active DB override onto the code
    default from INTERACTION_DEFAULTS."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def test_get_config_overlays_partial_override(self):
        """A one-key override changes that key and ONLY that key — every
        unmentioned key still tracks the code default (decision e183d22c:
        overlay, not whole-value replace)."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        default = INTERACTION_DEFAULTS['s1e'][1]
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'sentinel-x'}))
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'sentinel-x'
        for key, value in default.items():
            if key != 'effort':
                assert config[key] == value

    def test_get_config_unknown_name_raises(self):
        with pytest.raises(KeyError):
            self.brain.get_interaction_config('nonexistent')

    def test_get_prompt_returns_override_text(self):
        self.brain._interaction_dal.register(
            'surface', template='You are a test judge. Select wisely.',
            parameters=json.dumps({}))
        self.brain._interaction_dal.set_active('surface', 1, set_by='test')
        prompt = self.brain.get_interaction_prompt('surface')
        assert prompt == 'You are a test judge. Select wisely.'

    def test_get_prompt_unknown_name_raises(self):
        with pytest.raises(KeyError):
            self.brain.get_interaction_prompt('nonexistent')

    def test_register_never_auto_activates(self):
        """Registration NEVER changes the runtime pointer — not v1, not v2.
        A write is not a deployment decision; the name runs on its code
        default until set_active deploys an override.
        """
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'v1-effort'}))
        # v1 is dormant — runtime still reads the code default
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == INTERACTION_DEFAULTS['s1e'][1]['effort'], \
            "v1 register must NOT auto-activate; runtime runs the code default"

        # Deploy v1 explicitly — runtime flips to the override
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort'

        # Register v2 — must NOT change what runtime reads
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'v2-effort'}),
            created_by='sleep:s3')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort', \
            "v2 register should NOT auto-activate; runtime should still see v1"

        # Now explicitly activate v2 — runtime flips
        self.brain._interaction_dal.set_active('s1e', 2, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v2-effort'

        # Rollback to v1 by re-activating
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort'

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
# Fallback behavior + resolver guards
# ═══════════════════════════════════════════════════════

class TestInteractionFallback:
    """No DB row is the normal state after the override collapse: the
    resolver returns the code default, silently."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def test_config_fallback_returns_code_default(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        assert (self.brain.get_interaction_config('surface')
                == INTERACTION_DEFAULTS['surface'][1])

    def test_prompt_fallback_returns_code_default(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        assert (self.brain.get_interaction_prompt('surface')
                == INTERACTION_DEFAULTS['surface'][0])

    def test_stamp_fallback_is_default_source(self):
        """No row → source 'default', version 0, id None — and the
        fingerprint hashes the RESOLVED default, not ''/{}."""
        from servers.interaction_defaults import (
            INTERACTION_DEFAULTS, interaction_fingerprint)
        stamp = self.brain.get_interaction_stamp('surface')
        template, config = INTERACTION_DEFAULTS['surface']
        assert stamp == {
            'fingerprint': interaction_fingerprint('surface', template, config),
            'source': 'default', 'version': 0, 'id': None}

    def test_unknown_name_raises(self):
        """A typo'd or unregistered name must raise, not run on an empty
        prompt (guard 2 — safe because tests/test_interaction_defaults.py
        keeps the registry complete)."""
        with pytest.raises(KeyError):
            self.brain.get_interaction_config('judge')
        with pytest.raises(KeyError):
            self.brain.get_interaction_prompt('judge')
        with pytest.raises(KeyError):
            self.brain.get_interaction_stamp('judge')


class TestResolverGuards:
    """The dangerous rows: unparseable JSON and invalid values must degrade
    LOUDLY to the code default — a typo'd override silently reverting a
    boundary is the failure this resolver exists to prevent."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def _deploy(self, name, version=1):
        self.brain._interaction_dal.set_active(name, version, set_by='test')

    def _resolve_errors(self):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source = "
            "'interaction_resolve'").fetchone()[0]

    def test_unparseable_json_falls_back_loudly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters='{not valid json')
        self._deploy('s1e')
        config = self.brain.get_interaction_config('s1e')
        assert config == INTERACTION_DEFAULTS['s1e'][1]
        assert self._resolve_errors() >= 1

    def test_non_object_json_falls_back_loudly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps([1, 2, 3]))
        self._deploy('s1e')
        config = self.brain.get_interaction_config('s1e')
        assert config == INTERACTION_DEFAULTS['s1e'][1]
        assert self._resolve_errors() >= 1

    def test_validator_violation_falls_back_loudly(self):
        """A scopes override with an invalid mode (written past the
        register_interaction door, e.g. by an older version) must not
        become the running policy."""
        from servers.scopes import SCOPES_CONFIG_V1
        self.brain._interaction_dal.register(
            'scopes', template='',
            parameters=json.dumps({'project': {'mode': 'bogus'}}))
        self._deploy('scopes')
        config = self.brain.get_interaction_config('scopes')
        assert config == SCOPES_CONFIG_V1
        assert self._resolve_errors() >= 1

    def test_register_door_refuses_validator_violations(self):
        """The write door REFUSES what the read seam degrades — the generic
        INTERACTION_VALIDATORS path that replaced the scopes special-case."""
        with pytest.raises(ValueError):
            self.brain.register_interaction(
                'scopes',
                parameters=json.dumps({'project': {'mode': 'bogus'}}))
        with pytest.raises(ValueError):
            self.brain.register_interaction('scopes', parameters='{not json')

    def test_stamp_override_fingerprints_resolved_value(self):
        """source='override' when a row shadows the default, and the
        fingerprint hashes the OVERLAID effective config — not the raw
        override fragment."""
        from servers.interaction_defaults import (
            INTERACTION_DEFAULTS, interaction_fingerprint)
        row = self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'high'}))
        self._deploy('s1e')
        stamp = self.brain.get_interaction_stamp('s1e')
        template, default = INTERACTION_DEFAULTS['s1e']
        assert stamp['source'] == 'override'
        assert stamp['version'] == row['version']
        assert stamp['id'] == row['id']
        assert stamp['fingerprint'] == interaction_fingerprint(
            's1e', template, {**default, 'effort': 'high'})

    def test_stamp_vacuous_row_is_default_source(self):
        """A row that contributes nothing (empty template, empty config)
        does not shadow the default — it stamps 'default' like no row."""
        self.brain._interaction_dal.register(
            's1e', template='', parameters='{}')
        self._deploy('s1e')
        stamp = self.brain.get_interaction_stamp('s1e')
        assert stamp['source'] == 'default'
        assert stamp['version'] == 0
        assert stamp['id'] is None


# ═══════════════════════════════════════════════════════
# Pointer delete = clear (Step 5: nothing resurrects MAX)
# ═══════════════════════════════════════════════════════

class TestPointerDeleteIsClear:
    """'No pointer' means 'no override deployed' — nothing may resurrect
    MAX(version). The motivating landmine: trace_recording sits at active=1
    with a dormant v2 = DEBUG (all payload kinds on); the old get_active
    fallback turned a pointer delete into silent full-payload capture, and
    the old ensure_logs_schema backstop re-pinned MAX on the next boot."""

    def test_pointer_delete_reverts_to_code_default_and_survives_reopen(self):
        from servers.brain import Brain
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        with IsolatedBrain() as env:
            brain = env.brain
            # Establish our own deployed state (don't lean on whatever the
            # production copy's pointer happens to be), then register a
            # dormant version ABOVE it — the exact shape where a MAX(version)
            # fallback returns something nobody deployed.
            brain._interaction_dal.set_active('surface', 1, set_by='test')
            brain.register_interaction(
                'surface', template='SENTINEL vNext — never deployed',
                parameters='{}')
            assert brain.get_interaction_stamp('surface')['source'] == 'override'

            brain.logs_conn.execute(
                "DELETE FROM interaction_active WHERE name = 'surface'")
            brain.logs_conn.commit()

            # DAL: no pointer -> None, never the dormant MAX version.
            assert brain.get_interaction('surface') is None
            # Resolver: falls through to the code default.
            assert (brain.get_interaction_prompt('surface')
                    == INTERACTION_DEFAULTS['surface'][0])
            assert brain.get_interaction_stamp('surface')['source'] == 'default'

            # A boot (ensure_logs_schema + seed_interactions) must not
            # re-create the pointer.
            brain.save()
            brain.close()
            env.brain = Brain(env.brain_db)
            assert env.brain.get_interaction('surface') is None
            count = env.brain.logs_conn.execute(
                "SELECT COUNT(*) FROM interaction_active "
                "WHERE name = 'surface'").fetchone()[0]
            assert count == 0


# ═══════════════════════════════════════════════════════
# The clear verb (Step 6: override is a two-way door)
# ═══════════════════════════════════════════════════════

class TestClearInteractionOverride:
    """clear_interaction_override is the inverse of set_interaction_active:
    without it an override is a one-way door and the model degrades back
    into the per-name freeze. Clearing must revert to the code default
    IMMEDIATELY — the TTL caches are what would make a clear run late."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def _deploy(self, name, template='', config=None):
        self.brain._interaction_dal.register(
            name, template=template, parameters=json.dumps(config or {}))
        self.brain._interaction_dal.set_active(name, 1, set_by='test')

    def test_clear_reverts_to_code_default_and_reports_distinctly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self._deploy('s1e', config={'effort': 'sentinel-clear'})
        assert (self.brain.get_interaction_config('s1e')['effort']
                == 'sentinel-clear')

        result = self.brain.clear_interaction_override('s1e')
        assert result == {'name': 's1e', 'cleared': True}
        assert (self.brain.get_interaction_config('s1e')
                == INTERACTION_DEFAULTS['s1e'][1])
        assert self.brain.get_interaction_stamp('s1e')['source'] == 'default'
        # Version rows survive the clear — re-activation stays possible.
        assert self.brain.get_interaction('s1e', version=1) is not None

        # No pointer existed — reported distinctly from "cleared".
        assert self.brain.clear_interaction_override('s1e') == {
            'name': 's1e', 'cleared': False}

    def test_clear_reaches_trace_recording_cache_immediately(self):
        """The landmine workflow: deploy a debug-ish override, clear it —
        the payload recorder must see the revert NOW, not a TTL later."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self._deploy('trace_recording', config={'round_payload': True})
        warmed = self.brain._trace_recording_config()
        assert warmed['round_payload'] is True

        self.brain.clear_interaction_override('trace_recording')
        assert (self.brain._trace_recording_config()
                == INTERACTION_DEFAULTS['trace_recording'][1])

    def test_set_active_and_clear_reach_recall_laf_cache_immediately(self):
        """recall_laf's engine TTL cache had NO invalidation hook — a flip
        or clear ran up to CONFIG_TTL_S late. Both verbs must drop it."""
        from servers.recall_laf import get_engine
        engine = get_engine(self.brain)
        engine.config(self.brain)
        assert engine._cfg is not None

        # Through the Brain verbs — cache invalidation is their policy,
        # not the DAL's.
        self.brain._interaction_dal.register(
            'recall_laf', template='', parameters='{}')
        self.brain.set_interaction_active('recall_laf', 1, set_by='test')
        assert engine._cfg is None, 'set_active must drop the laf cache'

        engine.config(self.brain)
        assert engine._cfg is not None
        self.brain.clear_interaction_override('recall_laf')
        assert engine._cfg is None, 'clear must drop the laf cache'
        # Behavioral half: the very next config() read resolves the current
        # value (the code default) — no TTL wait.
        assert (engine.config(self.brain)
                == self.brain.get_interaction_config('recall_laf'))

    def test_clear_refuses_unknown_names(self):
        """A typo'd clear must never report 'already on the default' while
        the real override keeps running — nothing deleted + no code default
        means refusal, not a benign no-op."""
        with pytest.raises(KeyError):
            self.brain.clear_interaction_override('trace_recoding')  # typo

    def test_clear_purges_the_recall_result_cache(self):
        """The recall RESULT cache keys on the query alone — a laf flip or
        clear must purge it or an identical query re-asked within the TTL
        returns the pre-flip result."""
        self.brain._recall_cache_put(('probe',), {'results': ['stale']})
        assert self.brain._recall_cache_get(('probe',)) is not None
        self.brain.clear_interaction_override('recall_laf')
        assert self.brain._recall_cache_get(('probe',)) is None
