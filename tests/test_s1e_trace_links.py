"""Piece 2 of the S1E code-half rebuild — the trace↔node link capability.

Verifies servers/scales/s1/trace_links.py: the PURE composer joins S1R surface
selections and S1E encode runs to target traces by stop_counter (the structural
chain-suffix key, never timestamps). Driven entirely by a raw synthetic trace
dataset — plain dicts, no brain, no DB — which is exactly the robustness the
architecture is for (eval replay / sequential historical data drive it the same).

See docs/S1-SCRIBE-REDESIGN.md §10.3.2.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.trace_links import (  # noqa: E402
    nodes_for_traces, gather, session_node_ids, _stop_of, _surface_ids,
    _candidate_outcomes,
)

SHORT = 'abcd1234'  # session_short in every chain id


def _turn(stop, tid=None):
    return {'id': tid or ('u%d' % stop), 'chain_id': 's0-%s-%d' % (SHORT, stop)}


def _surface(stop, ids, tid=None):
    import json
    return {'id': tid or ('sr%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
            'ref_id': json.dumps(ids)}


def _encode(stop, created, revised, tid=None):
    return {'id': tid or ('e%d' % stop), 'chain_id': 's1e-%s-%d' % (SHORT, stop),
            'metadata': {'created': created, 'revised': revised}}


# ── the stop parser (the join key) ──

def test_stop_of_parses_trailing_counter():
    assert _stop_of('s0-abcd1234-5') == 5
    assert _stop_of('s1r-abcd1234-12') == 12
    assert _stop_of('s1e-abcd1234-0') == 0
    assert _stop_of('') is None
    assert _stop_of(None) is None
    assert _stop_of('s0-abcd1234-notanint') is None


def test_stop_of_round_trips_session_context_chains():
    # Pin the build↔parse coupling: the stop SessionContext stamps into a chain id
    # must be the stop _stop_of recovers. Guards a silent join break if the
    # chain-id format ever changes on one side only.
    from servers.session_context import SessionContext
    ctx = SessionContext(session_id='abcd1234deadbeef', stop_counter=7)
    assert _stop_of(ctx.s0_chain()) == 7
    assert _stop_of(ctx.s1r_chain()) == 7
    assert _stop_of(ctx.s1e_chain()) == 7


def test_surface_ids_resilient():
    assert _surface_ids('["a","b"]') == ['a', 'b']
    assert _surface_ids('') == []
    assert _surface_ids('not json') == []      # malformed → [], no raise
    assert _surface_ids('{"a":1}') == []        # not a list → []


# ── surfaced: 1:1 with the turn, same stop ──

def test_surfaced_joins_by_stop():
    links = nodes_for_traces(
        surface_traces=[_surface(5, ['nodeA', 'nodeB']),
                        _surface(6, ['nodeC'])],
        encode_traces=[],
        target_traces=[_turn(5), _turn(6)])
    assert links['u5']['surfaced'] == ['nodeA', 'nodeB']
    assert links['u6']['surfaced'] == ['nodeC']
    # no encode runs → nothing encoded, no owning run
    assert links['u5']['encoded'] == [] and links['u5']['encoded_by'] is None


# ── encoded: per-run, the owning run closes the turn-range at/after the turn ──

def test_encoded_owning_run_is_first_run_at_or_after_turn():
    # runs at stop 5 and 10. turn 3 and 5 belong to run@5; turn 7 to run@10.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(5, ['c5'], ['r5'], tid='run5'),
                       _encode(10, ['c10'], [], tid='run10')],
        target_traces=[_turn(3), _turn(5), _turn(7)])
    assert links['u3']['encoded_by'] == 'run5'
    assert links['u5']['encoded_by'] == 'run5'
    assert links['u7']['encoded_by'] == 'run10'
    # the run's COMPLETE node-set (created + revised), deduped & order-preserved
    assert links['u3']['encoded'] == ['c5', 'r5']
    assert links['u7']['encoded'] == ['c10']


def test_unencoded_tail_has_no_owning_run():
    # a turn past the last run = the emergent unencoded boundary.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(5, ['c5'], [])],
        target_traces=[_turn(8)])
    assert links['u8']['encoded'] == []
    assert links['u8']['encoded_by'] is None


def test_created_and_revised_merged_and_deduped():
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(5, ['x', 'y'], ['y', 'z'])],  # y in both
        target_traces=[_turn(5)])
    assert links['u5']['encoded'] == ['x', 'y', 'z']


def test_full_link_shape_surfaced_and_encoded_together():
    links = nodes_for_traces(
        surface_traces=[_surface(5, ['surfA', 'surfB'])],
        encode_traces=[_encode(5, ['encC'], ['encD'], tid='run5')],
        target_traces=[_turn(5)])
    assert links['u5'] == {
        'surfaced': ['surfA', 'surfB'],
        'encoded': ['encC', 'encD'],
        'encoded_by': 'run5',
        'authored': [], 'recalled': [], 'endo': [],  # no touched stream here
        'dropped': [],                               # no recall stream here
    }


# ── resilience: a raw dataset with odd rows must not crash the join ──

def test_malformed_rows_are_survivable():
    links = nodes_for_traces(
        surface_traces=[{'id': 'bad', 'chain_id': 's1r-x-5', 'ref_id': 'garbage'},
                        {'id': 'nochain'},  # no chain_id → skipped
                        _surface(5, ['ok'])],
        encode_traces=[{'id': 'badrun'}],   # no chain_id → skipped
        target_traces=[_turn(5), {'id': 'orphan'}])  # orphan has no chain_id
    assert links['u5']['surfaced'] == ['ok']           # the good row survived
    assert links['orphan'] == {'surfaced': [], 'encoded': [], 'encoded_by': None,
                               'authored': [], 'recalled': [], 'endo': [],
                               'dropped': []}


def test_runs_out_of_order_still_pick_earliest_owning_run():
    # encode_traces fed newest-first; the composer sorts by stop internally.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(10, ['c10'], [], tid='run10'),
                       _encode(5, ['c5'], [], tid='run5')],
        target_traces=[_turn(6)])
    assert links['u6']['encoded_by'] == 'run10'   # first run with stop >= 6


# ── the live adapter pulls both streams via the query_traces door ──

def _touched(stop, created=None, revised=None, recalled=None, endo=None, tid=None):
    return {'id': tid or ('at%d' % stop), 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'metadata': {'created': created or [], 'revised': revised or [],
                         'archived': [], 'recalled': recalled or [], 'endo': endo or []}}


class _StubBrain:
    def __init__(self):
        self.calls = []

    def query_traces(self, **kw):
        self.calls.append(kw)
        if kw.get('ref_type') == 'surface_selected':
            return {'events': [_surface(5, ['s'])]}
        if kw.get('ref_type') == 'encoding_run':
            return {'events': [_encode(5, ['e'], [])]}
        if kw.get('ref_type') == 'anchor_touched':
            return {'events': [_touched(5, created=['w'], recalled=['r'])]}
        return {'events': []}


def test_gather_pulls_all_three_streams_via_door():
    brain = _StubBrain()
    st = gather(brain, 'sess-xyz')          # default streams: surface/encode/touched
    surf, enc, touched = st['surface'], st['encode'], st['touched']
    assert len(surf) == 1 and len(enc) == 1 and len(touched) == 1
    # all three pulls scoped to the session, window disabled (hours=None);
    # surface/encode at s1, anchor_touched at s0.
    by_rt = {c['ref_type']: c for c in brain.calls}
    assert set(by_rt) == {'surface_selected', 'encoding_run', 'anchor_touched'}
    assert by_rt['surface_selected']['scale'] == 's1'
    assert by_rt['anchor_touched']['scale'] == 's0'
    for c in brain.calls:
        assert c['session_id'] == 'sess-xyz' and c['hours'] is None
    # end-to-end through the pure core
    links = nodes_for_traces(surf, enc, [_turn(5)], touched_traces=touched)
    assert links['u5']['surfaced'] == ['s'] and links['u5']['encoded'] == ['e']
    assert links['u5']['authored'] == ['w'] and links['u5']['recalled'] == ['r']


# ── the touched feed: anchor_touched joins 1:1 with the turn by stop ──

def test_touched_authored_and_recalled_join_by_stop():
    links = nodes_for_traces(
        surface_traces=[], encode_traces=[],
        target_traces=[_turn(5), _turn(6)],
        touched_traces=[_touched(5, created=['nA'], revised=['nB'], recalled=['nC']),
                        _touched(6, created=['nD'])])
    assert links['u5']['authored'] == ['nA', 'nB']   # created ∪ revised
    assert links['u5']['recalled'] == ['nC']
    assert links['u6']['authored'] == ['nD'] and links['u6']['recalled'] == []


def test_touched_archived_not_in_authored():
    # archived is recorded in the delta but a gone node must not surface as authored.
    t = {'id': 'at5', 'chain_id': 's0-%s-5' % SHORT,
         'metadata': {'created': ['live'], 'revised': [], 'archived': ['dead'],
                      'recalled': [], 'endo': []}}
    links = nodes_for_traces([], [], [_turn(5)], touched_traces=[t])
    assert links['u5']['authored'] == ['live']
    assert 'dead' not in links['u5']['authored']


def test_touched_absent_yields_empty_relations():
    # No touched stream → authored/recalled/endo are empty, never missing.
    links = nodes_for_traces([_surface(5, ['s'])], [], [_turn(5)])
    assert links['u5']['authored'] == [] and links['u5']['endo'] == []


def test_delta_ids_shared_parser_symmetry():
    # The SAME parser reads the S1 encode delta and the S0 touched delta — both
    # carry created/revised. Proven by feeding one meta dict through both lenses.
    from servers.scales.s1.trace_links import _delta_ids
    meta = {'created': ['a', 'b'], 'revised': ['b', 'c'], 'recalled': ['d']}
    assert _delta_ids(meta, 'created', 'revised') == ['a', 'b', 'c']  # encode lens
    assert _delta_ids(meta, 'recalled') == ['d']                       # touched-only lens
    assert _delta_ids(None, 'created') == []


def test_session_node_ids_unions_encode_and_touched():
    # The catalog's session-level view: union across all encode + touched traces,
    # no turn keying. encoded = created∪revised over runs; authored/recalled over
    # touched. surfaced is NOT here (the catalog has it from judge outputs).
    ids = session_node_ids(
        encode_traces=[_encode(5, ['e1'], ['e2']), _encode(10, ['e3'], [])],
        touched_traces=[_touched(5, created=['a1'], revised=['a2'], recalled=['r1']),
                        _touched(6, created=['a3'], recalled=['r1'])])  # r1 dup
    assert ids['encoded'] == {'e1', 'e2', 'e3'}
    assert ids['authored'] == {'a1', 'a2', 'a3'}
    assert ids['recalled'] == {'r1'}                  # deduped across turns
    assert 'surfaced' not in ids and 'endo' not in ids


def test_session_node_ids_empty_streams():
    ids = session_node_ids([], [])
    assert ids == {'encoded': set(), 'authored': set(), 'recalled': set()}


def test_anchor_touched_metadata_builder_shape_and_dedup():
    # The contract builder: all five keys present, order-preserving dedup, and it
    # mirrors the encode delta's created/revised/archived key names (the symmetry).
    from servers.trace_contract import (build_anchor_touched_metadata,
                                        ANCHOR_TOUCHED_KEYS, DELTA_METADATA_SHAPE)
    m = build_anchor_touched_metadata(created=['a', 'a', 'b'], recalled=['c'])
    assert set(m) == set(ANCHOR_TOUCHED_KEYS)
    assert m['created'] == ['a', 'b'] and m['recalled'] == ['c']
    assert m['revised'] == [] and m['archived'] == [] and m['endo'] == []
    # the shared keys are exactly the encode delta's id-list field names
    assert {'created', 'revised', 'archived'} <= set(DELTA_METADATA_SHAPE)


# ── the dropped role: recall_traces + _candidate_outcomes (episodic view) ──

def _outcomes(stop, outcomes, tid=None):
    """A Δ additionalContext candidate-pool trace (authoritative verdict map)."""
    return {'id': tid or ('ac%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
            'metadata': {'outcomes_per_candidate': outcomes}}


def _recall_pool(stop, cand_short_ids, tid=None):
    """An O recall candidate-pool trace (raw pool, no verdict)."""
    cands = ['%s|some title|0.70|finding' % c for c in cand_short_ids]
    return {'id': tid or ('rc%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
            'metadata': {'candidates': cands}}


def test_candidate_outcomes_reads_verdict_map():
    cands, dropped = _candidate_outcomes(
        {'outcomes_per_candidate': {'a': 'selected', 'b': 'dropped', 'c': 'dropped'}})
    assert set(cands) == {'a', 'b', 'c'}
    assert set(dropped) == {'b', 'c'}


def test_candidate_outcomes_falls_back_to_mirror_lists():
    cands, dropped = _candidate_outcomes({'selected': ['a'], 'dropped': ['b', 'c']})
    assert set(cands) == {'a', 'b', 'c'} and set(dropped) == {'b', 'c'}


def test_candidate_outcomes_raw_pool_has_no_verdict():
    cands, dropped = _candidate_outcomes(
        {'candidates': ['a|t|0.7|x', 'b|t|0.6|y']})
    assert cands == ['a', 'b'] and dropped is None   # caller derives dropped


def test_candidate_outcomes_malformed_is_empty():
    assert _candidate_outcomes(None) == ([], None)
    assert _candidate_outcomes({}) == ([], None)
    assert _candidate_outcomes({'candidates': 'not a list'}) == ([], None)


def test_dropped_authoritative_verdict_split():
    # surfaced from K surface_selected; dropped from the Δ verdict map; encoded
    # from the owning encode run at the same stop — ONE join, all roles.
    links = nodes_for_traces(
        [_surface(5, ['p1'])],
        [_encode(5, ['enc_full_1'], ['enc_full_2'])],
        [_turn(5)],
        recall_traces=[_outcomes(5, {'p1': 'selected', 'd1': 'dropped',
                                     'd2': 'dropped'})])
    r = links['u5']
    assert r['surfaced'] == ['p1']
    assert set(r['dropped']) == {'d1', 'd2'}
    assert set(r['encoded']) == {'enc_full_1', 'enc_full_2'}


def test_dropped_raw_pool_derives_pool_minus_surfaced():
    # No verdict map — only the raw O pool. dropped = candidates − surfaced.
    links = nodes_for_traces(
        [_surface(7, ['p1'])], [], [_turn(7)],
        recall_traces=[_recall_pool(7, ['p1', 'd1', 'd2', 'd3'])])
    r = links['u7']
    assert r['surfaced'] == ['p1']
    assert set(r['dropped']) == {'d1', 'd2', 'd3'}
    assert r['encoded'] == []


def test_surfaced_never_appears_in_dropped():
    # Even if a stale pool lists a surfaced id as a candidate, surfaced wins.
    links = nodes_for_traces(
        [_surface(3, ['p1', 'p2'])], [], [_turn(3)],
        recall_traces=[_recall_pool(3, ['p1', 'p2', 'd1'])])
    r = links['u3']
    assert set(r['surfaced']) == {'p1', 'p2'}
    assert r['dropped'] == ['d1']


def test_dropped_empty_without_recall_stream():
    # recall_traces omitted → dropped is empty, never missing (mirrors touched).
    links = nodes_for_traces([_surface(5, ['s'])], [], [_turn(5)])
    assert links['u5']['dropped'] == []


def test_dropped_missing_stop_yields_empty():
    links = nodes_for_traces([], [], [{'id': 'x', 'chain_id': 'no-stop-here'}],
                             recall_traces=[])
    assert links['x']['surfaced'] == [] and links['x']['dropped'] == []
    assert links['x']['encoded'] == []


class _RolesStubBrain:
    """Stub serving the three doors the episodic gather pulls."""
    def __init__(self):
        self.calls = []

    def query_traces(self, **kw):
        self.calls.append(kw)
        rt = kw.get('ref_type')
        if rt == 'additionalContext':
            return {'events': [_outcomes(5, {'p': 'selected', 'd': 'dropped'})]}
        if rt == 'surface_selected':
            return {'events': [_surface(5, ['p'])]}
        if rt == 'encoding_run':
            return {'events': [_encode(5, ['e'], [])]}
        return {'events': []}


def test_gather_recall_streams_and_dropped_compose():
    brain = _RolesStubBrain()
    st = gather(brain, 'sess-xyz', streams=('surface', 'encode', 'recall'))
    assert len(st['surface']) == 1 and len(st['encode']) == 1 and len(st['recall']) == 1
    rts = {c['ref_type'] for c in brain.calls}
    assert rts == {'additionalContext', 'surface_selected', 'encoding_run'}
    for c in brain.calls:
        assert c['scale'] == 's1' and c['session_id'] == 'sess-xyz' and c['hours'] is None
    links = nodes_for_traces(st['surface'], st['encode'], [_turn(5)],
                             recall_traces=st['recall'])
    r = links['u5']
    assert r['surfaced'] == ['p'] and r['encoded'] == ['e'] and r['dropped'] == ['d']


def test_gather_unknown_stream_is_loud():
    import pytest
    with pytest.raises(KeyError):
        gather(_RolesStubBrain(), 'sess-xyz', streams=('surface', 'bogus'))


# ── render integration: <provenance> in the lived-sequence timeline (encode.py) ──

from servers.scales.s1.encode import _render_lived_sequence_timeline  # noqa: E402


def _msg(ref_type, tid, stop, content, ts):
    return {'id': tid, 'ref_type': ref_type, 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'summary': content[:60], 'created_at': ts, 'metadata': {'content': content}}


class _ProvBrain:
    """A brain stub that serves BOTH doors the lived timeline needs: episodes
    (recall_episodes) for the turns, and query_traces for gather()."""
    def __init__(self, episodes, surface_traces, encode_traces):
        self._episodes = episodes
        self._surface = surface_traces
        self._encode = encode_traces

    def recall_episodes(self, **kw):
        rts = kw.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts] if rts else self._episodes
        return {'episodes': eps, 'ranked_by': 'time', 'truncated': False}

    def query_traces(self, **kw):
        if kw.get('ref_type') == 'surface_selected':
            return {'events': self._surface}
        if kw.get('ref_type') == 'encoding_run':
            return {'events': self._encode}
        return {'events': []}


# Four turns (stops 5–8). A prior run@7 covers turns 5,6,7; turn 8 is the
# unencoded tail. Surface fired (only) on turn 5.
_EPS = [
    _msg('user_message', 'u5', 5, 'turn five user', '2026-06-29T00:00:01'),
    _msg('assistant_message', 'a5', 5, 'turn five reply', '2026-06-29T00:00:02'),
    _msg('user_message', 'u6', 6, 'turn six user', '2026-06-29T00:00:03'),
    _msg('assistant_message', 'a6', 6, 'turn six reply', '2026-06-29T00:00:04'),
    _msg('user_message', 'u7', 7, 'turn seven user', '2026-06-29T00:00:05'),
    _msg('assistant_message', 'a7', 7, 'turn seven reply', '2026-06-29T00:00:06'),
    _msg('user_message', 'u8', 8, 'turn eight user', '2026-06-29T00:00:07'),
    _msg('assistant_message', 'a8', 8, 'turn eight reply', '2026-06-29T00:00:08'),
]
_SURF = [_surface(5, ['nodeAAAA1111', 'nodeBBBB2222'])]
_ENC = [_encode(7, ['nodeCCCC3333'], [], tid='run7')]
_FOUR_MSGS = [{'role': 'user'} for _ in range(4)]  # window = 4 user turns


def test_provenance_renders_surfaced_and_encoded_at_frontier():
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _FOUR_MSGS)
    t5, t6, t7, t8 = (out.split('<turn n="%d">' % n)[1].split('</turn>')[0]
                      for n in range(1, 5))
    # turn 5: surfaced (8-char refs) + covered-not-frontier ✓
    assert 'surfaced: id:nodeAAAA id:nodeBBBB' in t5
    assert 'encoded(S1S): ✓' in t5
    # turn 6: covered, ✓, no surfaced (surface only fired turn 5)
    assert 'encoded(S1S): ✓' in t6 and 'surfaced:' not in t6
    # turn 7: frontier — the run's full encoded id-list shows here, once
    assert 'encoded(S1S): id:nodeCCCC' in t7 and '✓' not in t7


def test_provenance_unencoded_tail_has_no_encoded_marker():
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _FOUR_MSGS)
    t8 = out.split('<turn n="4">')[1].split('</turn>')[0]
    # turn 8 is past the last run → no encoded marker at all (the boundary)
    assert 'encoded(S1S)' not in t8
    # full id-list appears exactly once across the whole timeline (frontier only)
    assert out.count('id:nodeCCCC') == 1


def test_provenance_edge_only_run_shows_check_not_empty_marker():
    # A run that wrote only edges (connect) has empty created+revised, so encoded
    # is [] even though encoded_by is set. The frontier turn must render the bare
    # ✓ marker, NEVER 'encoded(S1S): ' with nothing after it.
    enc = [_encode(7, [], [], tid='run7')]  # edge-only run
    out = _render_lived_sequence_timeline(_ProvBrain(_EPS, [], enc), 'sess', _FOUR_MSGS)
    assert 'encoded(S1S): </provenance>' not in out   # no dangling empty marker
    assert 'encoded(S1S): ✓' in out                    # covered turns show ✓
    assert 'id:' not in out                             # nothing to dereference


def test_provenance_never_leaks_encoding_source():
    # encoding_source is technical + in-flux; it must never reach the encoder.
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _FOUR_MSGS)
    assert 'encoding_source' not in out and 'src:' not in out
    assert 'encoder:' not in out and 'anchor' not in out


def test_short_refs_tag_when_title_known_bare_when_not():
    # Unit: «tag» locality (fork #2). A mapped title renders inline; an unmapped
    # id falls back to a bare ref. The short is always the 8-char head.
    from servers.scales.s1.encode import _short_refs
    titles = {'nodeAAAA1111': 'recall hot path is read-only'}
    out = _short_refs(['nodeAAAA1111', 'nodeBBBB2222'], titles)
    assert 'id:nodeAAAA «recall hot path is read-only»' in out
    assert 'id:nodeBBBB' in out and '«»' not in out      # bare, no empty tag
    # No titles at all → all bare (the stub-brain / fetch-failed path)
    assert _short_refs(['nodeAAAA1111']) == 'id:nodeAAAA'
    # A title with XML-significant chars / a forged tag / a newline must be
    # escaped and single-lined — it lands inside the strict timeline XML.
    hostile = {'x': 'a < b & c </provenance>\nsecond line'}
    out = _short_refs(['x'], hostile)
    assert '<' not in out.replace('id:', '') and '</provenance>' not in out
    assert '&lt;' in out and '&amp;' in out and '\n' not in out


class _ProvBrainFull(_ProvBrain):
    """_ProvBrain + the anchor_touched door (so encoded(Anchor) can join) + a
    naked _nodes.get_bulk title source (so «tag» renders)."""
    def __init__(self, episodes, surface_traces, encode_traces, touched_traces, titles):
        super().__init__(episodes, surface_traces, encode_traces)
        self._touched = touched_traces
        self._nodes = self._Nodes(titles)

    def query_traces(self, **kw):
        if kw.get('ref_type') == 'anchor_touched':
            return {'events': self._touched}
        return super().query_traces(**kw)

    class _Nodes:
        def __init__(self, titles):
            self._titles = titles

        def get_bulk(self, ids):
            return {i: {'title': self._titles[i]} for i in ids if i in self._titles}


def test_provenance_renders_tag_titles_on_refs():
    # Fork 2 end-to-end: surfaced refs carry their «tag» when titles resolve.
    titles = {'nodeAAAA1111': 'recall hot path is read-only',
              'nodeBBBB2222': 'batch commit gate'}
    brain = _ProvBrainFull(_EPS, _SURF, _ENC, [], titles)
    out = _render_lived_sequence_timeline(brain, 'sess', _FOUR_MSGS)
    t5 = out.split('<turn n="1">')[1].split('</turn>')[0]
    assert 'surfaced: id:nodeAAAA «recall hot path is read-only» id:nodeBBBB «batch commit gate»' in t5


def test_provenance_renders_encoded_anchor_when_authored_and_omits_when_empty():
    # Fork 1: a turn where Anchor encoded mid-turn (anchor_touched created@8) shows
    # `encoded(Anchor): <ids>`; turns with no authored set omit the line entirely.
    touched = [_touched(8, created=['nodeDDDD4444'])]
    titles = {'nodeDDDD4444': 'mid-turn insight'}
    brain = _ProvBrainFull(_EPS, _SURF, _ENC, touched, titles)
    out = _render_lived_sequence_timeline(brain, 'sess', _FOUR_MSGS)
    t5, t8 = (out.split('<turn n="%d">' % n)[1].split('</turn>')[0] for n in (1, 4))
    assert 'encoded(Anchor): id:nodeDDDD «mid-turn insight»' in t8   # filled at turn 8
    assert 'encoded(Anchor)' not in t5                               # omitted elsewhere
    assert out.count('encoded(Anchor)') == 1                          # exactly the one turn


def test_timeline_degrades_to_piece1_when_provenance_unavailable():
    # A brain with no query_traces (the piece-1 stub shape) → guarded fallback:
    # the timeline still renders, just without any <provenance> line.
    class _NoDoor:
        def __init__(self, eps):
            self._eps = eps

        def recall_episodes(self, **kw):
            rts = kw.get('ref_type') or []
            return {'episodes': [e for e in self._eps if e.get('ref_type') in rts]}

    out = _render_lived_sequence_timeline(_NoDoor(_EPS), 'sess', _FOUR_MSGS)
    assert '<turn n="1">' in out and '<provenance>' not in out
