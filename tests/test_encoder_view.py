"""Encoder view policy (servers/scales/s1/encoder_view.py) — the flag-gated
feeding decisions: catalog aging, action filtering/stubbing, the provenance
verb split, and the encoded-turn trim reversal.

Two invariants under test, per policy surface:
  • Flag OFF (default) → byte-identical to the pre-policy render — the A/B
    control arm. Every render test in test_s1e_trace_links.py already runs
    with the default; here the OFF arm is asserted explicitly against the
    legacy markers.
  • Flag ON → the filter marks itself in place (a stubbed <actions> says
    trimmed, an aged entry carries [aged] + the expand hint) — absence must
    never read as "nothing happened".
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.encoder_view import (  # noqa: E402
    view_policy_enabled, aging_cutoff, catalog_view, action_hidden,
    actions_stub_line, AGED_TAG, AGED_CONTENT_CHARS, CATALOG_FULL_ROUNDS,
    ENCODED_TURN_MESSAGE_CAP, HIDDEN_ACTION_TOOLS, PROVENANCE_SPLIT,
)
from servers.scales.s1.encode import _render_lived_sequence_timeline  # noqa: E402
from servers.scales.s1.encode_contract import build_node_catalog  # noqa: E402

SHORT = 'abcd1234'


# ── the flag ──

def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv('BRAIN_S1E_VIEW_POLICY', raising=False)
    assert view_policy_enabled() is False
    monkeypatch.setenv('BRAIN_S1E_VIEW_POLICY', '1')
    assert view_policy_enabled() is True
    monkeypatch.setenv('BRAIN_S1E_VIEW_POLICY', '0')
    assert view_policy_enabled() is False


# ── pure policy: aging tiers ──

def test_aging_cutoff_is_nth_newest_run_stop():
    assert aging_cutoff([5, 10, 15]) == 10        # 2nd-newest of 3
    assert aging_cutoff([5, 10]) == 5             # exactly N runs
    assert aging_cutoff([5]) is None              # fewer than N → no aging
    assert aging_cutoff([]) is None
    assert aging_cutoff(None) is None
    assert aging_cutoff([10, 5, 10]) == 5         # unsorted + dup input


def test_catalog_view_orders_oldest_to_newest_unknown_last():
    ids = {'old1', 'new1', 'surf'}
    stops = {'old1': 3, 'new1': 15}
    ordered, aged = catalog_view(ids, stops, run_stops=[10, 15])
    assert ordered == ['old1', 'new1', 'surf']    # unknown stop sorts last
    assert aged == {'old1'}                       # 3 < cutoff(10)


def test_catalog_view_protected_ids_never_age():
    stops = {'old1': 3, 'old2': 4}
    ordered, aged = catalog_view({'old1', 'old2'}, stops, run_stops=[10, 15],
                                 protected={'old2'})   # surfaced this window
    assert aged == {'old1'}
    assert 'old2' not in aged


def test_catalog_view_no_aging_before_enough_runs():
    ordered, aged = catalog_view({'a', 'b'}, {'a': 1, 'b': 2}, run_stops=[5])
    assert aged == set()
    assert ordered == ['a', 'b']                  # still ordered by stop


def test_catalog_view_boundary_stop_stays_full():
    # An id last touched exactly AT the cutoff belongs to the newest N rounds.
    ordered, aged = catalog_view({'edge'}, {'edge': 10}, run_stops=[10, 15])
    assert aged == set()


# ── pure policy: action visibility ──

def test_action_hidden_covers_node_ops_and_lookups():
    assert action_hidden('mcp__plugin_brain_brain__remember_batch')
    assert action_hidden('mcp__plugin_brain_brain__get_nodes')
    assert action_hidden('mcp__brain__revise')          # user-scope registration
    # id:27db2472: batch + all lookup tools move to provenance too
    assert action_hidden('mcp__plugin_brain_brain__brain_batch')
    assert action_hidden('mcp__plugin_brain_brain__recall')
    assert action_hidden('mcp__plugin_brain_brain__recall_batch')
    assert action_hidden('mcp__plugin_brain_brain__find_node_by_title')
    assert action_hidden('mcp__plugin_brain_brain__filter_nodes')
    assert action_hidden('mcp__plugin_brain_brain__enrich')
    # edge ops stay visible — "connect has no provenance home, by contract"
    assert not action_hidden('mcp__plugin_brain_brain__connect')
    assert not action_hidden('mcp__plugin_brain_brain__disconnect')
    assert not action_hidden('mcp__plugin_brain_brain__revise_edge')
    # non-brain tools stay visible, whatever their basename
    assert not action_hidden('Bash')
    assert not action_hidden('Edit')
    assert not action_hidden('mcp__slack__get_nodes')
    # unknown defaults to visible
    assert not action_hidden(None)
    assert not action_hidden('')


def test_actions_stub_marks_itself():
    line = actions_stub_line(7)
    assert 'trimmed' in line and '7' in line


# ── timeline render, policy ON vs OFF (the encode.py wiring) ──
# Fixtures mirror test_s1e_trace_links: turns at stops 5-8, run@7 covers 5,6.

def _msg(ref_type, tid, stop, content, ts, tool=None):
    md = {'content': content} if tool is None else {'tool': tool}
    return {'id': tid, 'ref_type': ref_type,
            'chain_id': 's0-%s-%d' % (SHORT, stop),
            'summary': content[:120], 'created_at': ts, 'metadata': md}


def _encode_trace(stop, created, revised, tid=None):
    return {'id': tid or ('e%d' % stop),
            'chain_id': 's1e-%s-%d' % (SHORT, stop),
            'metadata': {'created': created, 'revised': revised}}


def _touched_trace(stop, **keys):
    md = {k: [] for k in ('created', 'revised', 'archived', 'recalled', 'endo')}
    md.update(keys)
    return {'id': 'at%d' % stop, 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'metadata': md}


class _Brain:
    """Serves the doors the lived timeline pulls: recall_episodes,
    query_traces (gather), and naked title lookups."""
    def __init__(self, episodes, surface=(), encode=(), touched=(), titles=None):
        self._episodes = list(episodes)
        self._surface, self._encode, self._touched = \
            list(surface), list(encode), list(touched)
        self._nodes = self._Nodes(titles or {})

    class _Nodes:
        def __init__(self, titles):
            self._titles = titles

        def get_bulk(self, ids):
            return {i: {'title': self._titles[i]}
                    for i in ids if i in self._titles}

    def recall_episodes(self, **kw):
        rts = kw.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts]
        return {'episodes': eps}

    def query_traces(self, **kw):
        rt = kw.get('ref_type')
        events = {'surface_selected': self._surface,
                  'encoding_run': self._encode,
                  'anchor_touched': self._touched}.get(rt, [])
        return {'events': events}


def _two_turn_eps(covered_body='covered turn text', tail_body='tail turn text'):
    return [
        _msg('user_message', 'u5', 5, covered_body, '2026-08-01T00:00:01'),
        _msg('assistant_message', 'a5', 5, 'reply five', '2026-08-01T00:00:02'),
        _msg('tool_result', 't5a', 5, 'Bash: ls -la', '2026-08-01T00:00:03',
             tool='Bash'),
        _msg('user_message', 'u6', 6, tail_body, '2026-08-01T00:00:04'),
        _msg('assistant_message', 'a6', 6, 'reply six', '2026-08-01T00:00:05'),
        _msg('tool_result', 't6a', 6, 'Edit: foo.py', '2026-08-01T00:00:06',
             tool='Edit'),
        _msg('tool_result', 't6b', 6,
             'mcp__plugin_brain_brain__remember_batch: {"nodes": [...]}',
             '2026-08-01T00:00:07', tool='mcp__plugin_brain_brain__remember_batch'),
    ]


_TWO_MSGS = [{'role': 'user'} for _ in range(2)]


def _render(view_policy, eps=None, **brain_kw):
    brain_kw.setdefault('encode', [_encode_trace(6, ['nodeCCCC3333'], [],
                                                 tid='run6')])
    brain = _Brain(eps if eps is not None else _two_turn_eps(), **brain_kw)
    return _render_lived_sequence_timeline(brain, 'sess', _TWO_MSGS,
                                           view_policy=view_policy)


def test_policy_off_is_legacy_render():
    # Control arm: encoded turn trimmed, actions listed verbatim, no markers.
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    trim = ENCODING_AGENT['encoded_turn_trim']
    long_body = 'x' * (trim + 500)
    out = _render(False, eps=_two_turn_eps(covered_body='covered ' + long_body))
    t5 = out.split('<turn n="1"')[1].split('</turn>')[0]
    assert '…' in t5.split('<other')[1].split('</other>')[0]   # trim applied
    assert 'trimmed —' not in out                              # no stub
    assert AGED_TAG not in out
    assert 'remember_batch' in out                             # node op visible
    assert 'Bash: ls -la' in out


def test_policy_on_encoded_turn_full_text_and_actions_stub():
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    trim = ENCODING_AGENT['encoded_turn_trim']
    long_body = 'x' * (trim + 500)
    out = _render(True, eps=_two_turn_eps(covered_body='covered ' + long_body))
    t5 = out.split('<turn n="1"')[1].split('</turn>')[0]
    t6 = out.split('<turn n="2"')[1].split('</turn>')[0]
    # covered turn keeps FULL text (the trim reversal) …
    assert ENCODED_TURN_MESSAGE_CAP is None
    assert t5.split('<other')[1].split('</other>')[0].count('x') == trim + 500
    # … and its actions collapse to the self-marking stub (element stays)
    assert '<actions>trimmed — 1 action(s)' in t5
    assert 'Bash: ls -la' not in t5
    # the unencoded tail keeps its actions
    assert 'Edit: foo.py' in t6


def test_policy_on_drops_node_op_lines_on_unencoded_turns():
    out = _render(True)
    t6 = out.split('<turn n="2"')[1].split('</turn>')[0]
    assert 'remember_batch' not in t6          # provenance owns it now
    assert 'Edit: foo.py' in t6                # world-changing lines stay


def test_policy_on_all_actions_hidden_drops_element():
    # A tail turn whose only action is a hidden node-op: nothing to show —
    # unencoded turns carry no coverage claim, so the element may drop.
    eps = [e for e in _two_turn_eps() if e['id'] != 't6a']
    out = _render(True, eps=eps)
    t6 = out.split('<turn n="2"')[1].split('</turn>')[0]
    assert '<actions>' not in t6


def test_policy_on_provenance_verb_split():
    touched = [_touched_trace(6, created=['nodeAAAA1111'],
                              revised=['nodeBBBB2222'],
                              recalled=['nodeDDDD4444'],
                              looked_up=['nodeFFFF6666', 'nodeDDDD4444'],
                              archived=['nodeEEEE5555'])]
    titles = {'nodeAAAA1111': 'fresh insight', 'nodeDDDD4444': 'looked up',
              'nodeFFFF6666': 'search hit'}
    on = _render(True, touched=touched, titles=titles)
    t6 = on.split('<turn n="2"')[1].split('</turn>')[0]
    assert 'created(me): id:nodeAAAA «fresh insight»' in t6
    assert 'revised(me): id:nodeBBBB' in t6
    # recalled(me) merges by-id reads with search results, deduped
    assert ('recalled(me): id:nodeDDDD «looked up» id:nodeFFFF «search hit»'
            in t6)
    assert t6.count('id:nodeDDDD') == 1
    assert 'archived(me): id:nodeEEEE' in t6
    assert 'encoded(Anchor)' not in on          # the merged label retires
    # control arm: merged label, no verbs, no looked_up render
    off = _render(False, touched=touched, titles=titles)
    assert 'encoded(Anchor): id:nodeAAAA' in off
    assert 'created(me)' not in off and 'recalled(me)' not in off
    assert 'nodeFFFF' not in off


# ── catalog aging (build_node_catalog wiring) ──

class _CatalogBrain:
    """Stub for build_node_catalog: node bodies + the community-filter conn."""
    def __init__(self, nodes):
        self._map = nodes

    def execute(self, sql, params=None):
        return []                                # no community nodes

    conn = property(lambda self: self)

    def get_node(self, ids):
        return {i: dict(self._map[i]) for i in ids if i in self._map}


def _node(nid, content_chars=1200, edges=2):
    return {
        'id': nid, 'type': 'finding', 'title': 'title of %s' % nid,
        'content': ('claim first. ' + 'body ' * 400)[:content_chars],
        'situation': 'when testing catalog aging',
        'created_at': '2026-08-01T00:00:00',
        '_metadata': {'reasoning': 'long reasoning ' * 20},
        '_corrections': [{'id': 'corr1234', 'title': 'the correction',
                          'direction': 'corrected_by', 'relation': 'corrects',
                          'content': 'corrector body ' * 30}],
        'connections': [{'id': 'tgt%d' % i, 'title': 'target %d' % i,
                         'relation': 'grounds', 'direction': 'outgoing',
                         'description': 'edge description ' * 5,
                         'created_at': '2026-08-01T00:00:00'}
                        for i in range(edges)],
    }


def _catalog(view_policy, extra_ids):
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    return build_node_catalog([], brain, extra_ids=extra_ids,
                              view_policy=view_policy)


_EXTRA = {'encoded': {'oldnode1', 'newnode1'}, 'authored': set(),
          'recalled': set(), 'stops': {'oldnode1': 5, 'newnode1': 15},
          'run_stops': [5, 10, 15]}


def test_catalog_aging_trims_old_rounds_keeps_new_full():
    text, ids = _catalog(True, _EXTRA)
    assert ids == {'oldnode1', 'newnode1'}
    old_entry = text.split('title of oldnode1')[1].split('[encoded]')[0]
    new_entry = text.split('title of newnode1')[1]
    # aged: no edges, no reasoning, lean correction, content head only
    assert 'Edges:' not in old_entry
    assert 'Reasoning' not in old_entry
    assert 'corrector body' not in old_entry           # heavy body gone …
    assert '⚠ Updated by: "the correction"' in old_entry   # … marker stays
    assert 'Situation: when testing catalog aging' in old_entry
    content_line = [l for l in old_entry.split('\n') if 'Content:' in l][0]
    assert len(content_line) < AGED_CONTENT_CHARS + 30
    # fresh round keeps the full render
    assert 'Edges:' in new_entry
    assert 'corrector body' in new_entry
    # aged entries are marked and the header explains the tag once
    assert ('%s [encoded]' % AGED_TAG) in text
    assert 'get_nodes expands any id' in text
    assert str(CATALOG_FULL_ROUNDS) in text.split('\n')[1]


def test_catalog_aging_orders_oldest_first():
    text, _ = _catalog(True, _EXTRA)
    assert text.index('title of oldnode1') < text.index('title of newnode1')


def test_catalog_policy_off_renders_full_everywhere():
    text, ids = _catalog(False, _EXTRA)   # stops/run_stops present but inert
    assert AGED_TAG not in text
    assert 'get_nodes expands' not in text
    assert text.count('Edges:') == 2      # both nodes full depth


def test_catalog_surfaced_ids_protected_from_aging():
    # oldnode1 is old by stop but surfaced for the CURRENT window → full body.
    judge = ['picked id:oldnode1 for this turn']
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    text, _ = build_node_catalog(judge, brain, extra_ids=_EXTRA,
                                 view_policy=True)
    assert AGED_TAG not in text           # nothing left to age
    assert text.count('Edges:') == 2


def test_hidden_tools_and_split_stay_in_sync_with_substrate():
    # The split keys must exist in the link contract; the hidden set must never
    # cover the edge tools ("connect has no provenance home, by contract"), and
    # every hidden lookup tool must have an accumulator route (else the drop
    # loses information instead of moving it).
    from servers.scales.s1.trace_links import nodes_for_traces
    from servers.daemon_server import BrainDaemon
    link = nodes_for_traces([], [], [{'id': 'x', 'chain_id': 's0-a-1'}])['x']
    for _label, keys in PROVENANCE_SPLIT:
        for key in keys:
            assert key in link
    assert not {'connect', 'disconnect', 'revise_edge'} & HIDDEN_ACTION_TOOLS
    write_ops = {'remember', 'remember_batch', 'revise', 'revise_batch',
                 'brain_batch'}                      # provenance via `affected`
    for tool in HIDDEN_ACTION_TOOLS - write_ops:     # lookups need _LOOKUP_KEY
        assert tool in BrainDaemon._LOOKUP_KEY, tool
