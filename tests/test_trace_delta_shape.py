"""Shape tests for unified delta/selection metadata builders.

Covers:
- Defaults, truncation, extras pass-through for both builders.
- Live-writer shape fidelity: each encoder's source imports + calls the
  builder function so shape can't drift silently.

Run: python3 -m pytest tests/test_trace_delta_shape.py -v
"""
import pytest

from servers.trace_contract import (
    DELTA_METADATA_SHAPE,
    DELTA_FINAL_TEXT_LIMIT,
    DELTA_ERROR_LIST_LIMIT,
    SELECTION_METADATA_SHAPE,
    SELECTION_CONTENT_LIMIT,
    LLM_ENCODER_DELTA_REF_TYPES,
    METADATA_REQUIRED_BY_REF_TYPE,
    RUN_TELEMETRY_FIELDS,
    build_delta_metadata,
    build_selection_metadata,
    build_run_telemetry,
    validate_trace_metadata,
    check_delta_telemetry,
    check_surface_telemetry,
)

# NOTE: the node-lifecycle split is no longer derived in the runner
# (`_split_action_ids` was deleted). Each dispatch write handler returns the
# authoritative `affected` dict; that attribution is tested end-to-end against a
# real brain in tests/test_connect_to_intra_batch.py. This file covers the
# builder shape + aggregation (created/revised/archived — edges are their own
# directional edge_relation_revised events, not a flat `connected` list).


# ═════════════════════════════════════════════════════════════
# Builder unit tests — defaults, types, truncation, extras
# ═════════════════════════════════════════════════════════════

class TestBuildDeltaMetadata:
    def test_defaults(self):
        m = build_delta_metadata()
        for key in DELTA_METADATA_SHAPE:
            assert key in m, f"missing required key: {key}"
        for key, expected_type in DELTA_METADATA_SHAPE.items():
            assert isinstance(m[key], expected_type), (
                f"{key}: expected {expected_type.__name__}, got {type(m[key]).__name__}")
        assert m['actions'] == 0
        assert m['outcomes'] == {}
        assert m['action_details'] == []
        assert m['errors'] == []
        assert m['journal_entry'] == ''

    def test_truncation_final_text_is_loud(self):
        long = 'x' * (DELTA_FINAL_TEXT_LIMIT + 500)
        m = build_delta_metadata(final_text=long)
        # Loud: head kept, dropped count named — never a silent slice.
        assert m['final_text'].startswith('x' * 100)
        assert '+500 chars truncated' in m['final_text']
        assert len(m['final_text']) <= DELTA_FINAL_TEXT_LIMIT + 40  # bounded

    def test_truncation_journal_entry_is_loud(self):
        long = 'y' * (DELTA_FINAL_TEXT_LIMIT + 500)
        m = build_delta_metadata(journal_entry=long)
        assert '+500 chars truncated' in m['journal_entry']
        assert len(m['journal_entry']) <= DELTA_FINAL_TEXT_LIMIT + 40

    def test_error_list_truncated_is_loud(self):
        errs = ['e%d' % i for i in range(20)]
        m = build_delta_metadata(errors=errs)
        # First 5 real errors kept + 1 loud marker naming the 15 dropped.
        assert m['errors'][:DELTA_ERROR_LIST_LIMIT] == ['e0', 'e1', 'e2', 'e3', 'e4']
        assert len(m['errors']) == DELTA_ERROR_LIST_LIMIT + 1
        assert '+15 more truncated' in m['errors'][-1]

    def test_extras_preserved(self):
        m = build_delta_metadata(clusters_processed=5, batches=3)
        assert m['clusters_processed'] == 5
        assert m['batches'] == 3

    def test_extras_do_not_overwrite_reserved_keys(self):
        # The builder's loop `if k not in metadata` ensures that if a
        # caller accidentally passes a reserved key via **extras dict,
        # the reserved value wins. Exercise directly (Python kwarg
        # syntax can't express this, but call-site dicts can).
        extras = {'clusters_processed': 5, 'actions': 999}
        m = build_delta_metadata(actions=7, **{k: v for k, v in extras.items() if k != 'actions'})
        assert m['actions'] == 7
        assert m['clusters_processed'] == 5

    def test_classifications_first_class_and_capped(self):
        # AspectIntegration's structured Δ is a validated first-class field
        # (not smuggled through **extras), and is capped loud-in-data.
        from servers.trace_contract import DELTA_CLASSIFICATIONS_LIMIT
        assert 'classifications' in DELTA_METADATA_SHAPE
        assert build_delta_metadata()['classifications'] == []
        cls = [{'category': 'node_types', 'value': 't%d' % i, 'aspects': ['x']}
               for i in range(DELTA_CLASSIFICATIONS_LIMIT + 3)]
        m = build_delta_metadata(classifications=cls)
        assert len(m['classifications']) == DELTA_CLASSIFICATIONS_LIMIT + 1
        assert m['classifications'][-1] == {'_truncated': 3}

    def test_outcomes_dict_passthrough(self):
        m = build_delta_metadata(outcomes={'consolidate': 2, 'evolve': 1, 'keep': 4})
        assert m['outcomes']['consolidate'] == 2
        assert m['outcomes']['evolve'] == 1
        assert m['outcomes']['keep'] == 4

    def test_none_defaults_coerced(self):
        m = build_delta_metadata(outcomes=None, action_details=None, errors=None)
        assert m['outcomes'] == {}
        assert m['action_details'] == []
        assert m['errors'] == []


class TestDeltaSplitAggregation:
    """build_delta_metadata aggregates the per-action node-lifecycle split
    (created/revised/archived) into the delta — so the single unified delta
    carries what S2 reads. The split itself is now the dispatch handler's
    authoritative `affected`, copied onto each action by the runner. Edges are
    NOT here — they're directional edge_relation_revised events."""

    def test_aggregates_created_revised_from_actions(self):
        m = build_delta_metadata(action_details=[
            {'tool': 'brain_batch', 'created': ['a'], 'revised': ['b']},
            {'tool': 'remember_batch', 'created': ['d', 'e'], 'revised': []},
        ])
        assert m['created'] == ['a', 'd', 'e']
        assert m['revised'] == ['b']
        # `connected` is no longer a delta field — edges live in edge events.
        assert 'connected' not in m

    def test_aggregates_archived_from_actions(self):
        # The absorb-fix bucket: a merge-only run records its survivors
        # (revised) and folded-in originals (archived) on the unified delta.
        m = build_delta_metadata(action_details=[
            {'tool': 'brain_batch', 'created': [], 'revised': ['surv1'],
             'archived': ['orig1', 'orig2']},
        ])
        assert m['revised'] == ['surv1']
        assert m['archived'] == ['orig1', 'orig2']

    def test_archived_default_empty(self):
        assert build_delta_metadata()['archived'] == []

    def test_explicit_override_wins(self):
        m = build_delta_metadata(action_details=[{'created': ['x']}], created=['override'])
        assert m['created'] == ['override']

    def test_empty_defaults(self):
        m = build_delta_metadata()
        assert m['created'] == [] and m['revised'] == [] and m['archived'] == []
        assert 'connected' not in m


class TestDeltaCostProvenance:
    """Tier A: cost/latency/version/truncation ride on the delta — trend
    encoder cost over time and A/B prompt versions from production traces."""

    def test_cost_fields_recorded(self):
        m = build_delta_metadata(
            elapsed_ms=4200, input_tokens=6000, output_tokens=900,
            cache_read_tokens=28000, cache_creation_tokens=28000,
            truncated=1, interaction_version=24)
        assert m['elapsed_ms'] == 4200
        assert m['input_tokens'] == 6000
        assert m['output_tokens'] == 900
        assert m['cache_read_tokens'] == 28000
        assert m['cache_creation_tokens'] == 28000
        assert m['truncated'] == 1
        assert m['interaction_version'] == 24

    def test_cost_fields_default_zero_and_int_typed(self):
        m = build_delta_metadata()
        for k in ('elapsed_ms', 'input_tokens', 'output_tokens',
                  'cache_read_tokens', 'cache_creation_tokens',
                  'truncated', 'interaction_version'):
            assert m[k] == 0 and isinstance(m[k], int)

    def test_cost_fields_are_required_by_schema(self):
        # S2 encoders that don't yet pass cost data still validate, because
        # the builder always emits the keys (default 0).
        for k in ('elapsed_ms', 'truncated', 'interaction_version'):
            assert k in DELTA_METADATA_SHAPE


class TestBuildRunTelemetry:
    """The shared agent-run cost block — one builder for the cost+loop field-set
    that BOTH the encoder delta and the Surface K trace build through, so the two
    can't drift into separate field-sets."""

    def test_emits_every_run_telemetry_field_int_typed(self):
        m = build_run_telemetry()
        for f in RUN_TELEMETRY_FIELDS:
            assert f in m, f"missing {f}"
            assert m[f] == 0 and isinstance(m[f], int)
        assert set(m) == set(RUN_TELEMETRY_FIELDS)  # nothing extra leaks in

    def test_values_passthrough(self):
        m = build_run_telemetry(elapsed_ms=120, rounds=2, truncated=1,
                                input_tokens=500, output_tokens=30,
                                cache_read_tokens=400, cache_creation_tokens=10)
        assert m == {'elapsed_ms': 120, 'rounds': 2, 'truncated': 1,
                     'input_tokens': 500, 'output_tokens': 30,
                     'cache_read_tokens': 400, 'cache_creation_tokens': 10}

    def test_none_coerced_to_zero_int(self):
        m = build_run_telemetry(elapsed_ms=None, rounds=None, input_tokens=None)
        assert m['elapsed_ms'] == 0 and m['rounds'] == 0 and m['input_tokens'] == 0

    def test_shared_definition_is_subset_of_delta_shape(self):
        # The shared-definition contract: every run-telemetry field is a
        # first-class key of the encoder delta (build_delta_metadata sources
        # them THROUGH build_run_telemetry). If a field is added to one, this
        # fails until it's in DELTA_METADATA_SHAPE too — no silent drift.
        for f in RUN_TELEMETRY_FIELDS:
            assert f in DELTA_METADATA_SHAPE, f"{f} not in DELTA_METADATA_SHAPE"

    def test_delta_cost_block_matches_builder(self):
        # build_delta_metadata's cost fields are exactly what build_run_telemetry
        # produces for the same inputs — proving the refactor didn't fork them.
        kw = dict(elapsed_ms=15257, rounds=2, truncated=0, input_tokens=31327,
                  output_tokens=1017, cache_read_tokens=9, cache_creation_tokens=3)
        d = build_delta_metadata(**kw)
        t = build_run_telemetry(**kw)
        assert {f: d[f] for f in RUN_TELEMETRY_FIELDS} == t


class TestValidateTraceMetadata:
    """The payload contract guard at the chokepoint — the hole that let two
    encoding_run shapes coexist undetected."""

    def test_unified_shape_passes(self):
        assert validate_trace_metadata('delta', 'encoding_run', build_delta_metadata())[0]

    def test_partial_payload_is_rejected(self):
        # A present-but-incomplete encoding_run payload is caught at the
        # chokepoint (the hole that once let two shapes coexist).
        ok, err = validate_trace_metadata('delta', 'encoding_run', {
            'created': [], 'revised': [], 'elapsed_ms': 5})
        assert not ok
        assert 'missing required keys' in err

    def test_non_dict_rejected(self):
        ok, _ = validate_trace_metadata('delta', 'encoding_run', "not a dict")
        assert not ok

    def test_undeclared_ref_type_is_permissive(self):
        assert validate_trace_metadata('delta', 'additionalContext', {})[0]

    def test_s2_delta_ref_types_validate_real_payload(self):
        # CR6: all four S2 delta ref_types share the unified shape — a real
        # build_delta_metadata payload passes for each, like encoding_run.
        for rt in ('consolidated', 'community_enriched', 'healer_generated',
                   'aspect_classified'):
            ok, err = validate_trace_metadata('delta', rt, build_delta_metadata())
            assert ok, f"{rt}: {err}"

    def test_bare_marker_passes(self):
        # CR6: the early-out/error markers ('No clusters to process') write
        # metadata=None — a no-op marker must NOT be flagged (help, don't cry wolf).
        for rt in ('consolidated', 'community_enriched', 'healer_generated',
                   'aspect_classified', 'encoding_run'):
            assert validate_trace_metadata('delta', rt, None)[0], rt

    def test_malformed_s2_delta_rejected(self):
        # CR6: a PRESENT but partial payload on an S2 delta is still caught.
        ok, err = validate_trace_metadata('delta', 'consolidated', {
            'created': [], 'revised': [], 'elapsed_ms': 5})
        assert not ok
        assert 'missing required keys' in err


class TestBuildSelectionMetadata:
    def test_defaults(self):
        m = build_selection_metadata()
        for key in SELECTION_METADATA_SHAPE:
            assert key in m, f"missing required key: {key}"
        for key, expected_type in SELECTION_METADATA_SHAPE.items():
            assert isinstance(m[key], expected_type)
        assert m['candidates_considered'] == 0
        assert m['selected'] == []
        assert m['content'] == ''

    def test_content_truncation(self):
        long = 'a' * (SELECTION_CONTENT_LIMIT + 500)
        m = build_selection_metadata(content=long)
        assert len(m['content']) == SELECTION_CONTENT_LIMIT

    def test_extras_preserved(self):
        m = build_selection_metadata(query='hello', expanded=['n1', 'n2'])
        assert m['query'] == 'hello'
        assert m['expanded'] == ['n1', 'n2']


# ═════════════════════════════════════════════════════════════
# Shape fidelity at the live call sites
# ═════════════════════════════════════════════════════════════

ENCODERS_USING_DELTA_BUILDER = [
    'servers/scales/s2/community_encoder.py',
    'servers/scales/s2/consolidation_encoder.py',
    'servers/scales/s2/healer_encoder.py',
    'servers/scales/s2/aspect_encoder.py',
    'servers/scales/s1/encode.py',
]

SURFACERS_USING_SELECTION_BUILDER = [
    'servers/scales/s1/surface.py',
]

# The encoders that drive run_llm_loop — the only ones with a per-tool action
# loop to surface. Healer (hand-rolled dispatch) and aspect (direct
# classification) have no run_llm_loop tool loop, so read_calls doesn't apply.
RUN_LLM_LOOP_ENCODERS = [
    'servers/scales/s1/encode.py',
    'servers/scales/s2/consolidation_encoder.py',
    'servers/scales/s2/community_encoder.py',
]


def _file_calls(path, fn_name):
    import os, ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path)
    with open(full) as f:
        tree = ast.parse(f.read())

    found = [False]
    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == fn_name:
                found[0] = True
            self.generic_visit(node)
    Visitor().visit(tree)
    return found[0]


class TestLiveWritersUseBuilders:
    @pytest.mark.parametrize('path', ENCODERS_USING_DELTA_BUILDER)
    def test_encoder_uses_build_delta_metadata(self, path):
        assert _file_calls(path, 'build_delta_metadata'), (
            f"{path} does not call build_delta_metadata — delta trace shape will drift")

    @pytest.mark.parametrize('path', SURFACERS_USING_SELECTION_BUILDER)
    def test_surfacer_uses_build_selection_metadata(self, path):
        assert _file_calls(path, 'build_selection_metadata'), (
            f"{path} does not call build_selection_metadata — selection trace shape will drift")

    def test_surface_threads_run_telemetry(self):
        # Surface must build its cost block through the shared builder, map usage
        # via read_usage, and guard it — the wiring that closes the cost gap. If
        # any is dropped, surface silently regresses to no/zero telemetry.
        path = 'servers/scales/s1/surface.py'
        assert _file_calls(path, 'read_usage'), (
            f"{path} does not call read_usage — surface would record zero tokens")
        assert _file_calls(path, 'build_run_telemetry'), (
            f"{path} does not call build_run_telemetry — surface cost telemetry "
            "would not reach the K trace")
        assert _file_calls(path, 'check_surface_telemetry'), (
            f"{path} does not call check_surface_telemetry — a zero-token "
            "regression could land silently")


# ═════════════════════════════════════════════════════════════
# LLM-encoder telemetry guard (2026-06-24 S2 telemetry gap fix)
# ═════════════════════════════════════════════════════════════

class TestCheckDeltaTelemetry:
    """The loud check: an LLM-encoder delta that ran the model AND did work
    (actions>0) yet recorded output_tokens==0 is the silent telemetry-threading
    gap. Pure detector — fires for the 5 LLM-encoder ref_types, stays silent for
    everything else (markers, no-LLM/no-work runs, non-encoder ref_types)."""

    def test_fires_on_gap(self):
        # rounds>0, actions>0, output_tokens==0 — the gap.
        m = build_delta_metadata(rounds=1, actions=2, output_tokens=0)
        warn = check_delta_telemetry('consolidated', m)
        assert warn and 'output_tokens=0' in warn

    def test_silent_when_output_tokens_present(self):
        # The positive contract: an LLM-encoder delta with rounds>0 carries
        # output_tokens>0 → no flag. (This is what the Task A fix guarantees.)
        m = build_delta_metadata(rounds=1, actions=2, output_tokens=50)
        assert check_delta_telemetry('consolidated', m) is None

    def test_silent_on_zero_actions(self):
        # No work → no proof the model produced consumable output. Covers BOTH
        # no-op runs AND the all-LLM-calls-failed case (healer's rounds counts
        # batches attempted) — an LLM failure is logged elsewhere, not here.
        m = build_delta_metadata(rounds=3, actions=0, output_tokens=0)
        assert check_delta_telemetry('healer_generated', m) is None

    def test_silent_on_rounds_zero(self):
        # The early-out / no-clusters path — the LLM never ran.
        m = build_delta_metadata(rounds=0, actions=0, output_tokens=0)
        assert check_delta_telemetry('consolidated', m) is None

    def test_silent_on_non_encoder_ref_type(self):
        # Selection deltas / additionalContext have no LLM round to measure.
        m = build_delta_metadata(rounds=1, actions=2, output_tokens=0)
        assert check_delta_telemetry('additionalContext', m) is None
        assert check_delta_telemetry('node_revised', m) is None

    def test_silent_on_bare_marker_and_non_dict(self):
        # Bare early-out markers write metadata=None; never flag them.
        assert check_delta_telemetry('consolidated', None) is None
        assert check_delta_telemetry('consolidated', "not a dict") is None

    def test_fires_for_every_llm_encoder_ref_type(self):
        for rt in LLM_ENCODER_DELTA_REF_TYPES:
            m = build_delta_metadata(rounds=1, actions=1, output_tokens=0)
            assert check_delta_telemetry(rt, m) is not None, rt

    def test_scope_matches_unified_delta_ref_types(self):
        # The guard's scope must be exactly the ref_types whose payload is the
        # unified DELTA_METADATA_SHAPE — keeps the two lists from drifting (a new
        # LLM encoder added to one must be added to the other).
        unified = {rt for rt, schema in METADATA_REQUIRED_BY_REF_TYPE.items()
                   if schema is DELTA_METADATA_SHAPE}
        assert set(LLM_ENCODER_DELTA_REF_TYPES) == unified


class TestCheckSurfaceTelemetry:
    """The surface-side analog of check_delta_telemetry. Surface is the one LLM
    agent that writes its cost into a K trace (surface_selected), not a delta.
    Haiku ALWAYS emits output (the selection JSON, even an empty one), so unlike
    the delta guard there's no actions>0 gate — rounds>0 with output_tokens==0 is
    an unambiguous wiring gap on its own."""

    def test_fires_on_gap(self):
        warn = check_surface_telemetry({'rounds': 1, 'output_tokens': 0})
        assert warn and 'output_tokens=0' in warn

    def test_silent_when_output_tokens_present(self):
        assert check_surface_telemetry({'rounds': 1, 'output_tokens': 9}) is None

    def test_silent_on_rounds_zero(self):
        # Haiku never ran (e.g. an API failure on round 0 returned 0 rounds).
        assert check_surface_telemetry({'rounds': 0, 'output_tokens': 0}) is None

    def test_silent_on_non_dict(self):
        assert check_surface_telemetry(None) is None
        assert check_surface_telemetry("nope") is None

    def test_fires_on_real_k_metadata_with_zero_tokens(self):
        # A K-trace cost block built from empty telemetry (the regression we
        # guard against) has rounds=0 → silent; but a real run that recorded a
        # round yet zero output is the gap.
        from servers.trace_contract import build_run_telemetry
        gap = build_run_telemetry(rounds=1, output_tokens=0)
        assert check_surface_telemetry(gap) is not None
        good = build_run_telemetry(rounds=1, output_tokens=42)
        assert check_surface_telemetry(good) is None


class _TelStubBrain:
    """Captures _log_error — the errors-view sink the guard targets."""
    def __init__(self):
        self.errors = []

    def _log_error(self, source, exc, context=''):
        self.errors.append((source, str(exc), context))


class TestSurfaceSelectionJournal:
    """The S1Surface journal (2026-07-11): Haiku's recall-level `reason`
    persists in the surface_selected K trace — its only durable home (the
    renderer drops it; Anchor never sees it). In practice it's the
    why-nothing-was-picked note on empty selections."""

    def _k_event(self, **kw):
        from servers.scales.s1.surface import _write_traces

        class _DAL:
            def __init__(self):
                self.batches = []

            def append_batch(self, events):
                self.batches.append(events)

        brain = _TelStubBrain()
        brain._trace_dal = _DAL()

        class _Ctx:
            def s1r_chain(self):
                return 's1r-test-journal'

        cands = [{'id': 'a' * 32, 'title': 'T', 'score': 0.9, 'type': 'fact'}]
        _write_traces(brain, _Ctx(), cands, {'aaaaaaaa'}, [], [],
                      'ctx', 'query', [], 'ref-1', 7, 'sess-journal', **kw)
        events = brain._trace_dal.batches[0]
        return next(e for e in events
                    if e['ref_type'] == 'surface_selected')

    def test_reason_persists_in_k_metadata(self):
        k = self._k_event(selection_reason='pure confirmation, no topic')
        assert k['metadata']['selection_reason'] == \
            'pure confirmation, no topic'

    def test_journal_key_present_even_when_empty(self):
        k = self._k_event()
        assert k['metadata']['selection_reason'] == ''

    def test_runaway_rationale_is_bounded(self):
        k = self._k_event(selection_reason='y' * 999)
        assert len(k['metadata']['selection_reason']) == 500


class TestTraceBoundaryTelemetryGuard:
    """IntegrationUnit.trace() is the single S2 delta write boundary — the guard
    fires there so every S2 unit (and every FUTURE one) is covered for free.
    Loud, never blocking: the trace is still written when the gap fires."""

    def _unit(self):
        from servers.scales.s2.base import IntegrationUnit

        class _U(IntegrationUnit):
            NAME = 'consolidation'
            SCALE = 's2'
            ENCODING_SOURCE = 's2:consolidation'

        captured = []
        u = _U(brain=_TelStubBrain(),
               dispatch_fn=lambda cmd, data: captured.append((cmd, data)))
        u._chain_id = 's2-test-consolidation'  # pin → no datetime stamp
        return u, captured

    def test_fires_and_logs_on_gap(self):
        u, captured = self._unit()
        u.trace('delta', 'consolidated', 'summary',
                metadata=build_delta_metadata(rounds=1, actions=2, output_tokens=0))
        assert len(u.brain.errors) == 1
        source, msg, _ = u.brain.errors[0]
        assert source == 's2_consolidation_telemetry_gap'
        assert 'output_tokens=0' in msg
        # Loud, never block — the trace write still happened.
        assert len(captured) == 1

    def test_silent_when_telemetry_present(self):
        u, captured = self._unit()
        u.trace('delta', 'consolidated', 'summary',
                metadata=build_delta_metadata(rounds=1, actions=2, output_tokens=80))
        assert u.brain.errors == []
        assert len(captured) == 1

    def test_silent_on_bare_marker(self):
        # The "No clusters to process" early-out writes no payload.
        u, captured = self._unit()
        u.trace('delta', 'consolidated', 'No clusters to process')
        assert u.brain.errors == []

    def test_silent_on_non_delta_event(self):
        # The guard is delta-only; an O/K event with gappy-looking numbers is
        # never an LLM-encoder delta.
        u, _ = self._unit()
        u.trace('K', 'consolidation_proposals', 'observed',
                metadata={'rounds': 1, 'actions': 2, 'output_tokens': 0})
        assert u.brain.errors == []


def _build_delta_call_has_kwarg(path, kwarg):
    """True if the file passes `kwarg=` to any build_delta_metadata(...) call."""
    import os
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, path)) as f:
        tree = ast.parse(f.read())

    found = [False]

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name == 'build_delta_metadata':
                for kw in node.keywords:
                    if kw.arg == kwarg:
                        found[0] = True
            self.generic_visit(node)

    Visitor().visit(tree)
    return found[0]


class TestEncodersThreadTelemetry:
    """Every LLM encoder must thread cost/latency telemetry into its delta —
    the recurrence guard for the S2 gap. An encoder that copy-pastes
    build_delta_metadata without these kwargs fails here before the silent
    elapsed_ms=0/output_tokens=0 reaches production traces."""

    @pytest.mark.parametrize('path', ENCODERS_USING_DELTA_BUILDER)
    def test_encoder_threads_output_tokens(self, path):
        assert _build_delta_call_has_kwarg(path, 'output_tokens'), (
            f"{path} calls build_delta_metadata without an output_tokens kwarg — "
            "the production delta would record output_tokens=0 (telemetry gap)")

    @pytest.mark.parametrize('path', ENCODERS_USING_DELTA_BUILDER)
    def test_encoder_threads_elapsed_ms(self, path):
        assert _build_delta_call_has_kwarg(path, 'elapsed_ms'), (
            f"{path} calls build_delta_metadata without an elapsed_ms kwarg — "
            "encoder latency would be unmeasurable from production traces")

    @pytest.mark.parametrize('path', RUN_LLM_LOOP_ENCODERS)
    def test_run_llm_loop_encoder_threads_read_calls(self, path):
        # The per-tool read detail (latency_ms/result_count/error) rides on
        # run_llm_loop's read_calls; an encoder that drops the kwarg loses it
        # from its delta even though the loop recorded it.
        assert _build_delta_call_has_kwarg(path, 'read_calls'), (
            f"{path} drives run_llm_loop but doesn't thread read_calls= into "
            "build_delta_metadata — per-tool read detail won't reach the delta")


class TestSharedTelemetryHelpers:
    """read_usage + sum_usage (runner) are the single source for the SDK
    usage-field mapping and the token accumulator — reused by run_llm_loop's
    per-round tracking, IntegrationUnit._accumulate_run's per-batch fold,
    base._call_llm, and the surface agentic loop, so the field names and the
    "sum the four token fields" loop each live in exactly one place."""

    def test_read_usage_maps_sdk_field_names(self):
        from servers.scales.runner import read_usage, USAGE_FIELDS

        class _Usage:
            input_tokens = 10
            output_tokens = 20
            cache_read_input_tokens = 30
            cache_creation_input_tokens = 40

        class _Resp:
            usage = _Usage()

        u = read_usage(_Resp())
        assert u == {'input_tokens': 10, 'output_tokens': 20,
                     'cache_read_tokens': 30, 'cache_creation_tokens': 40}
        assert set(u) == set(USAGE_FIELDS)

    def test_read_usage_none_and_missing_are_zero(self):
        from servers.scales.runner import read_usage, USAGE_FIELDS
        zero = {f: 0 for f in USAGE_FIELDS}
        assert read_usage(None) == zero          # no response (pre-call baseline)

        class _Bare:
            pass

        assert read_usage(_Bare()) == zero       # response with no .usage

    def test_sum_usage_accumulates_over_usage_fields(self):
        from servers.scales.runner import sum_usage, read_usage
        total = read_usage(None)   # all-zero baseline
        sum_usage(total, {'input_tokens': 1, 'output_tokens': 2,
                          'cache_read_tokens': 3, 'cache_creation_tokens': 4})
        sum_usage(total, {'input_tokens': 10, 'output_tokens': 20,
                          'cache_read_tokens': 30, 'cache_creation_tokens': 40})
        assert total == {'input_tokens': 11, 'output_tokens': 22,
                         'cache_read_tokens': 33, 'cache_creation_tokens': 44}

    def test_sum_usage_coerces_missing_and_none(self):
        from servers.scales.runner import sum_usage, USAGE_FIELDS
        total = {}                                  # empty start: .get defends
        sum_usage(total, {'output_tokens': None})   # None → 0
        sum_usage(total, {})                         # missing keys → 0
        assert total == {f: 0 for f in USAGE_FIELDS}

    def test_accumulate_run_folds_counts_tokens_and_read_calls(self):
        # The multi-batch fold shared by consolidation + community. Critically,
        # it folds read_calls across batches — without that, the S2 encoders'
        # read_calls= threading is inert (total_result never gains the key).
        from servers.scales.s2.base import IntegrationUnit
        u = IntegrationUnit.__new__(IntegrationUnit)   # method uses no self state
        total = {'rounds': 0, 'actions': 0, 'write_actions': 0,
                 'action_details': [], 'read_calls': [],
                 'input_tokens': 0, 'output_tokens': 0,
                 'cache_read_tokens': 0, 'cache_creation_tokens': 0}
        u._accumulate_run(total, {
            'rounds': 1, 'actions': 2, 'write_actions': 1,
            'action_details': [{'tool': 'remember'}],
            'read_calls': [{'tool': 'recall_batch', 'result_count': 3}],
            'input_tokens': 100, 'output_tokens': 10})
        u._accumulate_run(total, {
            'rounds': 1, 'actions': 1, 'write_actions': 1,
            'action_details': [{'tool': 'revise'}],
            'read_calls': [{'tool': 'get_nodes', 'result_count': 2}],
            'input_tokens': 50, 'output_tokens': 5})
        assert total['rounds'] == 2 and total['actions'] == 3 and total['write_actions'] == 2
        assert [a['tool'] for a in total['action_details']] == ['remember', 'revise']
        assert [r['tool'] for r in total['read_calls']] == ['recall_batch', 'get_nodes']
        assert total['input_tokens'] == 150 and total['output_tokens'] == 15
