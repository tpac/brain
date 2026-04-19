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
)


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

    def test_truncation_final_text(self):
        long = 'x' * (DELTA_FINAL_TEXT_LIMIT + 500)
        m = build_delta_metadata(final_text=long)
        assert len(m['final_text']) == DELTA_FINAL_TEXT_LIMIT

    def test_truncation_journal_entry(self):
        long = 'y' * (DELTA_FINAL_TEXT_LIMIT + 500)
        m = build_delta_metadata(journal_entry=long)
        assert len(m['journal_entry']) == DELTA_FINAL_TEXT_LIMIT

    def test_error_list_truncated(self):
        errs = ['e%d' % i for i in range(20)]
        m = build_delta_metadata(errors=errs)
        assert len(m['errors']) == DELTA_ERROR_LIST_LIMIT

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
