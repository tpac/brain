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
    build_delta_metadata,
    build_selection_metadata,
    validate_trace_metadata,
)
from servers.scales.runner import _split_action_ids


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


class TestSplitActionIds:
    """The op-attributed split that replaced the legacy, brain_batch-blind
    runner delta. created/revised feed S2; this is the regression guard for
    the exact bug — brain_batch revises/connects must not vanish."""

    def test_brain_batch_routes_by_op_not_tool(self):
        # The legacy writer had no brain_batch branch, so these all vanished.
        result = {'results': [
            {'op': 'remember', 'ok': True, 'result': {'id': 'c0de0001'}},
            {'op': 'revise',   'ok': True, 'result': {'id': 'c0de0002'}},
            {'op': 'connect',  'ok': True, 'result': {'id': 'c0de0003'}},
            {'op': 'absorb',   'ok': True, 'survivor_id': 'c0de0004'},
            {'op': 'archive',  'ok': True, 'node_id': 'c0de0005'},
        ]}
        s = _split_action_ids('brain_batch', result)
        assert s['created'] == ['c0de0001']
        assert s['revised'] == ['c0de0002']
        assert s['connected'] == ['c0de0003']
        assert s['absorbed'] == ['c0de0004']
        assert s['archived'] == ['c0de0005']

    def test_brain_batch_skips_failed_ops(self):
        result = {'results': [
            {'op': 'remember', 'ok': True,  'result': {'id': 'aaaa0001'}},
            {'op': 'revise',   'ok': False, 'error': 'boom'},
        ]}
        s = _split_action_ids('brain_batch', result)
        assert s['created'] == ['aaaa0001']
        assert s['revised'] == []

    def test_homogeneous_tools_route_by_name(self):
        rb = {'results': [{'id': 'n1'}, {'result': {'id': 'n2'}}]}
        assert _split_action_ids('remember_batch', rb)['created'] == ['n1', 'n2']
        assert _split_action_ids('revise_batch', rb)['revised'] == ['n1', 'n2']
        assert _split_action_ids('connect_batch', rb)['connected'] == ['n1', 'n2']

    def test_single_tool_top_level_id(self):
        assert _split_action_ids('remember', {'id': 'solo1'})['created'] == ['solo1']

    def test_non_dict_result_is_safe(self):
        s = _split_action_ids('brain_batch', None)
        assert s == {'created': [], 'revised': [], 'connected': [],
                     'absorbed': [], 'archived': []}


class TestDeltaSplitAggregation:
    """build_delta_metadata aggregates the per-action split into the delta —
    so the single unified delta carries what S2 reads."""

    def test_aggregates_created_revised_connected_from_actions(self):
        m = build_delta_metadata(action_details=[
            {'tool': 'brain_batch', 'created': ['a'], 'revised': ['b'], 'connected': ['c']},
            {'tool': 'remember_batch', 'created': ['d', 'e'], 'revised': [], 'connected': []},
        ])
        assert m['created'] == ['a', 'd', 'e']
        assert m['revised'] == ['b']
        assert m['connected'] == ['c']

    def test_explicit_override_wins(self):
        m = build_delta_metadata(action_details=[{'created': ['x']}], created=['override'])
        assert m['created'] == ['override']

    def test_empty_defaults(self):
        m = build_delta_metadata()
        assert m['created'] == [] and m['revised'] == [] and m['connected'] == []


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


class TestValidateTraceMetadata:
    """The payload contract guard at the chokepoint — the hole that let two
    encoding_run shapes coexist undetected."""

    def test_unified_shape_passes(self):
        assert validate_trace_metadata('delta', 'encoding_run', build_delta_metadata())[0]

    def test_legacy_runner_shape_is_rejected(self):
        # The exact shape the deleted runner writer produced.
        ok, err = validate_trace_metadata('delta', 'encoding_run', {
            'created': [], 'revised': [], 'connected': [], 'elapsed_ms': 5})
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
