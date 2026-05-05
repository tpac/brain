"""Contract sync test for the trace system.

Scans all trace write sites in the codebase and verifies they only use
scale/event_type/ref_type combinations that exist in the trace contract.

Run after EVERY change to trace writers or trace_contract.py.

Run: python3 -m pytest tests/test_trace_contract_sync.py -v
"""
import ast
import os
import re
import pytest

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files that contain trace writes (production code only, not tests)
TRACE_WRITER_FILES = [
    'servers/daemon_hooks.py',
    'servers/brain_remember.py',   # archive_node writes a delta trace
    'servers/scales/s1/encode.py',
    'servers/scales/s1/surface.py',
    'servers/scales/s2/community_decoder.py',
    'servers/scales/s2/community_encoder.py',
    'servers/scales/s2/consolidation_decoder.py',
    'servers/scales/s2/consolidation_encoder.py',
    'servers/scales/s2/healer_decoder.py',
    'servers/scales/s2/healer_encoder.py',
    # edge_families.py removed 2026-05-04 — disabled in coordinator + file
    # deleted as part of unified-aspects Step 12 cleanup. Step 13 will add
    # servers/scales/s2/aspect_integration.py to this list.
    'servers/scales/s2/reclassify.py',
    'hooks/scripts/post_tool_trace.py',
]


def _extract_trace_writes_from_file(filepath):
    """Extract (scale, event_type, ref_type) triples from trace write calls in a file.

    Looks for patterns:
    - brain._trace_dal.append(..., scale='X', event_type='Y', ref_type='Z')
    - dispatch_fn('trace_append', {... 'scale': 'X', 'event_type': 'Y', 'ref_type': 'Z'})
    - _daemon_tcp_send('trace_append', {... 'scale': 'X', ...})
    - "cmd": "trace_append", "args": {... "scale": "X", ...}
    """
    full_path = os.path.join(ROOT, filepath)
    with open(full_path) as f:
        content = f.read()

    triples = []

    # Pattern 1: brain._trace_dal.append(...) or self.dal.append(...)
    # Extract keyword args: scale='...', event_type='...', ref_type='...'
    for match in re.finditer(
            r'\.append\([^)]*?scale=[\'"](\w+)[\'"][^)]*?event_type=[\'"](\w+)[\'"]'
            r'(?:[^)]*?ref_type=[\'"](\w+)[\'"])?',
            content, re.DOTALL):
        scale, event_type, ref_type = match.group(1), match.group(2), match.group(3) or ''
        triples.append((scale, event_type, ref_type, filepath, match.start()))

    # Pattern 2: dispatch_fn('trace_append', {...}) or _daemon_tcp_send('trace_append', {...})
    # These use dict literals with string keys
    for match in re.finditer(
            r"(?:dispatch_fn|_daemon_tcp_send)\s*\(\s*['\"]trace_append['\"]"
            r"\s*,\s*\{([^}]+)\}",
            content, re.DOTALL):
        block = match.group(1)
        scale = _extract_dict_value(block, 'scale')
        event_type = _extract_dict_value(block, 'event_type')
        ref_type = _extract_dict_value(block, 'ref_type')
        if scale and event_type:
            triples.append((scale, event_type, ref_type or '', filepath, match.start()))

    # Pattern 3: JSON dict in post_tool_trace.py ("cmd": "trace_append")
    for match in re.finditer(
            r'"cmd"\s*:\s*"trace_append"[^}]*?"scale"\s*:\s*"(\w+)"'
            r'[^}]*?"event_type"\s*:\s*"(\w+)"'
            r'(?:[^}]*?"ref_type"\s*:\s*"(\w+)")?',
            content, re.DOTALL):
        scale, event_type, ref_type = match.group(1), match.group(2), match.group(3) or ''
        triples.append((scale, event_type, ref_type, filepath, match.start()))

    return triples


def _extract_dict_value(block, key):
    """Extract a string value from a dict-like text block."""
    m = re.search(r"['\"]%s['\"]\s*:\s*['\"](\w+)['\"]" % key, block)
    return m.group(1) if m else ''


def _extract_chain_patterns(filepath):
    """Extract chain_id format patterns from a file."""
    full_path = os.path.join(ROOT, filepath)
    with open(full_path) as f:
        content = f.read()

    patterns = []
    # Match: chain_id = "s0-..." or chain_id="s1r-..." etc
    for match in re.finditer(
            r'chain_id\s*[=:]\s*["\']?(s[0-4][re]?)-',
            content):
        patterns.append((match.group(1), filepath, match.start()))

    # Match: "chain_id": "s0-..." in JSON dict
    for match in re.finditer(
            r'["\']chain_id["\']\s*:\s*["\'](s[0-4][re]?)-',
            content):
        patterns.append((match.group(1), filepath, match.start()))

    return patterns


# ═══════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════

class TestTraceContractSync:
    """Verify all trace writers in the codebase match the contract."""

    def setup_method(self):
        from servers.trace_contract import validate_trace_event, SCALES, CHAIN_PREFIXES
        self.validate = validate_trace_event
        self.SCALES = SCALES
        self.CHAIN_PREFIXES = CHAIN_PREFIXES

    def test_all_writer_files_exist(self):
        """All listed trace writer files exist."""
        for f in TRACE_WRITER_FILES:
            full = os.path.join(ROOT, f)
            assert os.path.exists(full), "Trace writer file missing: %s" % f

    def test_all_trace_writes_use_valid_scales(self):
        """Every trace write uses a scale defined in the contract."""
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, pos in triples:
                assert scale in self.SCALES, \
                    "Invalid scale '%s' in %s (event_type=%s, ref_type=%s)" % (
                        scale, fpath, event_type, ref_type)

    def test_all_trace_writes_use_valid_event_types(self):
        """Every trace write uses a valid event_type."""
        from servers.trace_contract import EVENT_TYPES
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, pos in triples:
                assert event_type in EVENT_TYPES, \
                    "Invalid event_type '%s' in %s (scale=%s, ref_type=%s)" % (
                        event_type, fpath, scale, ref_type)

    def test_all_trace_writes_use_valid_ref_types(self):
        """Every (scale, event_type, ref_type) triple passes contract validation."""
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, pos in triples:
                ok, error = self.validate(scale, event_type, ref_type)
                assert ok, "Contract violation in %s: %s (scale=%s, event_type=%s, ref_type=%s)" % (
                    fpath, error, scale, event_type, ref_type)

    def test_all_chain_ids_follow_conventions(self):
        """Chain ID prefixes match CHAIN_PREFIXES patterns."""
        valid_prefixes = set()
        for key, pattern in self.CHAIN_PREFIXES.items():
            # Extract the prefix before the first {
            prefix = pattern.split('{')[0]
            valid_prefixes.add(prefix)

        for filepath in TRACE_WRITER_FILES:
            patterns = _extract_chain_patterns(filepath)
            for prefix, fpath, pos in patterns:
                matched = any(prefix + '-' == vp or prefix == vp.rstrip('-')
                              for vp in valid_prefixes)
                if not matched:
                    # Check if prefix- matches any valid prefix pattern
                    matched = any((prefix + '-').startswith(vp) for vp in valid_prefixes)
                assert matched, \
                    "Chain prefix '%s' in %s doesn't match any CHAIN_PREFIXES: %s" % (
                        prefix, fpath, valid_prefixes)

    def test_contract_covers_s0_and_s1(self):
        """S0 and S1 have REF_TYPES defined for all event types in use."""
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1'):
            # At minimum, K and delta should have ref_types defined
            assert (scale, 'K') in REF_TYPES, \
                "Missing REF_TYPES for (%s, K)" % scale
            assert (scale, 'delta') in REF_TYPES, \
                "Missing REF_TYPES for (%s, delta)" % scale

    def test_no_trace_writes_outside_known_files(self):
        """No unexpected files contain trace writes (catch rogue writers)."""
        import glob
        all_py = glob.glob(os.path.join(ROOT, 'servers', '*.py'))
        all_py += glob.glob(os.path.join(ROOT, 'hooks', 'scripts', '*.py'))

        known_writers = set(os.path.join(ROOT, f) for f in TRACE_WRITER_FILES)
        # Also allow dal.py (the DAL itself) and daemon_dispatch.py (the router)
        known_writers.add(os.path.join(ROOT, 'servers', 'dal.py'))
        known_writers.add(os.path.join(ROOT, 'servers', 'daemon_dispatch.py'))

        for py_file in all_py:
            if py_file in known_writers:
                continue
            with open(py_file) as f:
                content = f.read()
            # Check for direct trace writes (not imports or comments)
            if 'trace_dal.append(' in content or "trace_append'" in content:
                # Ignore if it's just a reference in a comment or string
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue
                    if 'trace_dal.append(' in stripped or "trace_append'" in stripped:
                        assert False, \
                            "Unexpected trace write in %s line %d: %s\n" \
                            "Add to TRACE_WRITER_FILES in test_trace_contract_sync.py" % (
                                os.path.basename(py_file), i + 1, stripped[:80])


# ═══════════════════════════════════════════════════════
# Revise contract (Stage 1A — added 2026-05-04)
# ═══════════════════════════════════════════════════════

class TestReviseContract:
    """Verify the node_revised event contract.

    Stage 1A: every revise() call emits a (delta, node_revised) trace event
    carrying field-level deltas + reason. Replaces the old _sys_revision_history
    KV blob as the canonical revision history substrate.
    """

    def test_node_revised_registered_for_all_revise_origins(self):
        """node_revised must be a valid ref_type for s0/s1/s2 delta events.

        Revisions originate at every scale: S0 (operator via MCP), S1 (encoder),
        S2 (healer, consolidation, future aspect_integration).
        """
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1', 's2'):
            ref_types = REF_TYPES[(scale, 'delta')]
            assert 'node_revised' in ref_types, (
                "node_revised missing from (%s, delta) ref_types: %s" % (
                    scale, ref_types))

    def test_validate_accepts_node_revised(self):
        """validate_trace_event accepts (scale, 'delta', 'node_revised') for s0/s1/s2."""
        from servers.trace_contract import validate_trace_event
        for scale in ('s0', 's1', 's2'):
            ok, err = validate_trace_event(scale, 'delta', 'node_revised')
            assert ok, "Validation failed for (%s, delta, node_revised): %s" % (
                scale, err)

    def test_revise_metadata_shape_exists(self):
        """REVISE_METADATA_SHAPE is exported with the expected keys."""
        from servers.trace_contract import REVISE_METADATA_SHAPE
        expected_keys = {'node_id', 'reason', 'encoding_source',
                         'deltas', 'warnings'}
        assert set(REVISE_METADATA_SHAPE.keys()) == expected_keys, (
            "REVISE_METADATA_SHAPE keys: %s, expected: %s" % (
                set(REVISE_METADATA_SHAPE.keys()), expected_keys))

    def test_build_revise_metadata_full_call(self):
        """build_revise_metadata preserves all passed values."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(
            node_id='test123',
            reason='unit test',
            encoding_source='test:phase_a',
            deltas=[{'field': 'situation', 'old': 'before', 'new': 'after'}],
            warnings=['immutable field skipped: id'],
        )
        assert meta['node_id'] == 'test123'
        assert meta['reason'] == 'unit test'
        assert meta['encoding_source'] == 'test:phase_a'
        assert meta['deltas'] == [
            {'field': 'situation', 'old': 'before', 'new': 'after'}]
        assert meta['warnings'] == ['immutable field skipped: id']

    def test_build_revise_metadata_defaults(self):
        """build_revise_metadata fills sensible defaults for omitted args."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(node_id='test', reason='r')
        assert meta['node_id'] == 'test'
        assert meta['reason'] == 'r'
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_build_revise_metadata_handles_none(self):
        """build_revise_metadata coerces None values to safe defaults."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(
            node_id='test', reason=None,
            encoding_source=None, deltas=None, warnings=None)
        assert meta['reason'] == ''
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []


# ═══════════════════════════════════════════════════════
# Edge revise contract (Stage 1B — added 2026-05-04)
# ═══════════════════════════════════════════════════════

class TestEdgeReviseContract:
    """Verify the edge_relation_revised event contract.

    Stage 1B: every connect upsert (create or field update) and every
    archive that targets an edge_relation emits one trace event with
    field-level deltas, mirroring node revise semantics. Single ref_type
    covers create-via-upsert AND update-via-upsert AND polymorphic archive.

    ref_id encoding: f"{edge_id}:{relation}".
    """

    def test_edge_relation_revised_registered_for_all_revise_origins(self):
        """edge_relation_revised must be a valid ref_type for s0/s1/s2 delta."""
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1', 's2'):
            ref_types = REF_TYPES[(scale, 'delta')]
            assert 'edge_relation_revised' in ref_types, (
                "edge_relation_revised missing from (%s, delta) ref_types: %s" % (
                    scale, ref_types))

    def test_validate_accepts_edge_relation_revised(self):
        """validate_trace_event accepts (scale, 'delta', 'edge_relation_revised')."""
        from servers.trace_contract import validate_trace_event
        for scale in ('s0', 's1', 's2'):
            ok, err = validate_trace_event(scale, 'delta', 'edge_relation_revised')
            assert ok, "Validation failed for (%s, delta, edge_relation_revised): %s" % (
                scale, err)

    def test_edge_revise_metadata_shape_exists(self):
        """EDGE_REVISE_METADATA_SHAPE is exported with the expected keys."""
        from servers.trace_contract import EDGE_REVISE_METADATA_SHAPE
        expected_keys = {'edge_id', 'relation', 'reason',
                         'encoding_source', 'deltas', 'warnings'}
        assert set(EDGE_REVISE_METADATA_SHAPE.keys()) == expected_keys, (
            "EDGE_REVISE_METADATA_SHAPE keys: %s, expected: %s" % (
                set(EDGE_REVISE_METADATA_SHAPE.keys()), expected_keys))

    def test_build_edge_revise_metadata_full_call(self):
        """build_edge_revise_metadata preserves all passed values."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_a3f12c',
            relation='extends',
            reason='upsert via connect',
            encoding_source='encoder:sonnet',
            deltas=[{'field': 'description', 'old': 'old', 'new': 'new'}],
            warnings=[],
        )
        assert meta['edge_id'] == 'edg_a3f12c'
        assert meta['relation'] == 'extends'
        assert meta['reason'] == 'upsert via connect'
        assert meta['encoding_source'] == 'encoder:sonnet'
        assert meta['deltas'] == [
            {'field': 'description', 'old': 'old', 'new': 'new'}]
        assert meta['warnings'] == []

    def test_build_edge_revise_metadata_defaults(self):
        """build_edge_revise_metadata fills sensible defaults for omitted args."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_a3f12c', relation='extends', reason='r')
        assert meta['edge_id'] == 'edg_a3f12c'
        assert meta['relation'] == 'extends'
        assert meta['reason'] == 'r'
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_build_edge_revise_metadata_handles_none(self):
        """build_edge_revise_metadata coerces None values to safe defaults."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_x', relation='r', reason=None,
            encoding_source=None, deltas=None, warnings=None)
        assert meta['reason'] == ''
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_create_via_upsert_uses_same_ref_type(self):
        """Create-via-upsert uses edge_relation_revised with empty `old` in deltas.

        Single ref_type for create + update is the design call (Stage 1B Option A —
        mirror node pattern). Empty `old` in a delta semantically means 'created'.
        """
        from servers.trace_contract import build_edge_revise_metadata, validate_trace_event
        # Validates the contract supports it
        ok, _ = validate_trace_event('s2', 'delta', 'edge_relation_revised')
        assert ok
        # Builder accepts create-style deltas (old=None)
        meta = build_edge_revise_metadata(
            edge_id='edg_new', relation='extends', reason='created',
            encoding_source='encoder:sonnet',
            deltas=[
                {'field': 'description', 'old': None, 'new': 'new desc'},
                {'field': 'weight', 'old': None, 'new': 0.7},
            ])
        assert len(meta['deltas']) == 2
        assert all(d['old'] is None for d in meta['deltas'])
