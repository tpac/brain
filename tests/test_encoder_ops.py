"""eval/encoder_ops — the one reader of encoder op dumps, swap-aware, shared by
the A/B harness and the shape scorer. Every text it returns is the NEW text a
revise proposes; the removed `old` must never satisfy a fact check."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.encoder_ops import (edge_entries, kind, new_text, ops_of,  # noqa: E402
                              revise_surfaces, revise_text, swaps_of)


def test_ops_of_flattens_every_batch_shape():
    assert ops_of({'args': {'operations': [{'op': 'remember', 'title': 'a'}, 'junk']}}) \
        == [{'op': 'remember', 'title': 'a'}]
    # brain_batch operations that arrived JSON-encoded, unwrapped as dispatch does
    assert ops_of({'args': {'operations': '[{"op": "archive", "node_id": "abcdef01"}]'}}) \
        == [{'op': 'archive', 'node_id': 'abcdef01'}]
    assert ops_of({'args': {'revisions': [{'node_id': 'abcdef01', 'reason': 'r'}]}}) \
        == [{'node_id': 'abcdef01', 'reason': 'r'}]
    assert ops_of({'args': {'connections': [{'source_id': 'a' * 8, 'target_id': 'b' * 8}]}}) \
        [0]['source_id'] == 'a' * 8
    assert ops_of({'args': {'node_id': 'abcdef01', 'reason': 'single'}}) \
        == [{'node_id': 'abcdef01', 'reason': 'single'}]
    assert ops_of({'args': {}}) == []


def test_ops_of_folds_batch_level_connect_to_into_each_node():
    w = {'args': {'nodes': [{'title': 'n1', 'connect_to': [
                                {'target': 'aaaaaaaa', 'relation': 'grounds', 'why': 'w'}]},
                            {'title': 'n2'}],
                  'connect_to': [{'target': 'bbbbbbbb', 'relation': 'during', 'why': 'shared'}]}}
    ops = ops_of(w)
    assert [len(o['connect_to']) for o in ops] == [2, 1]
    assert ops[1]['connect_to'][0]['target'] == 'bbbbbbbb'
    assert w['args']['nodes'][1].get('connect_to') is None   # copies — the recorded write is untouched


def test_kind_declared_or_inferred():
    assert kind({'op': 'absorb'}) == 'absorb'
    assert kind({'node_id': 'abcdef01'}) == 'revise'
    assert kind({'source_id': 'a' * 8, 'target_id': 'b' * 8}) == 'connect'
    assert kind({'title': 't', 'content': 'c'}) == 'remember'


def test_new_text_is_the_proposed_text_only():
    assert new_text('plain') == 'plain'
    assert new_text({'old': 'stale 9.6.0', 'new': 'fresh 9.7.2'}) == 'fresh 9.7.2'
    assert new_text([{'old': 'a', 'new': 'x'}, {'old': 'b', 'new': 'y'}]) == 'x y'
    assert new_text(['1a2b3c4d', '5e6f7a8b']) == '1a2b3c4d 5e6f7a8b'
    assert new_text(None) == ''


def test_swaps_of_normalizes_the_alias_onto_content():
    op = {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r',
          'title': {'old': 'week 6', 'new': 'completed'},
          'content_edits': [{'old': 'a', 'new': 'b'}],
          'situation': 'a bare value is not a swap',
          'connect_to': [{'target': 'aaaaaaaa', 'relation': 'x', 'why': 'y'}]}
    s = swaps_of(op)
    assert set(s) == {'title', 'content'}
    assert s['content'] == [{'old': 'a', 'new': 'b'}]
    assert s['title'] == [{'old': 'week 6', 'new': 'completed'}]


def test_edge_entries_read_target_or_title_and_swap_new_text():
    rem = {'op': 'remember', 'title': 'new node', 'connect_to': [
        {'title': 'aaaaaaaa', 'relation': 'grounds', 'why': 'why-a'},
        {'target': 'bbbbbbbb', 'relations': [{'relation': 'after', 'why': 'w1'},
                                             {'relation': 'during', 'why': 'w2'}]}]}
    e = edge_entries(rem)
    assert [x['target'] for x in e] == ['aaaaaaaa', 'bbbbbbbb', 'bbbbbbbb']
    assert [x['relation'] for x in e] == ['grounds', 'after', 'during']
    assert all(x['source'] is None and x['via'] == 'connect_to' for x in e)

    rev = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r', 'connect_to': [
        {'target': '15bbfd64', 'relation': {'old': 'gaps_in', 'new': 'blocks'},
         'why': {'old': 'both manifests still say 9.6.0',
                 'new': 'manifests moved to 9.7.2, still short of 0.9.0'}}]}
    (x,) = edge_entries(rev)
    assert x == {'via': 'connect_to', 'source': 'd827d22f', 'target': '15bbfd64',
                 'relation': 'blocks',
                 'why': 'manifests moved to 9.7.2, still short of 0.9.0'}

    con = {'op': 'connect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
           'relation': 'similar_to', 'description': 'desc'}
    (y,) = edge_entries(con)
    assert y == {'via': 'connect', 'source': 'a' * 8, 'target': 'b' * 8,
                 'relation': 'similar_to', 'why': 'desc'}
    # a why-less edge is still an edge (empty why), never dropped
    (z,) = edge_entries({'op': 'remember', 'connect_to': [{'target': 'c' * 8, 'relation': 'r'}]})
    assert z['why'] == ''


def test_revise_surfaces_and_text_carry_new_values_only():
    op = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
          'title': {'old': 'version 9.6.0', 'new': 'version 9.7.2'},
          'content': [{'old': 'say 9.6.0', 'new': 'say 9.7.2 (was 9.6.0)'}],
          'situation': 'when the manifest version is asked'}
    assert revise_surfaces(op) == {'title': 'version 9.7.2',
                                   'situation': 'when the manifest version is asked',
                                   'content': 'say 9.7.2 (was 9.6.0)'}
    t = revise_text(op)
    assert 'version 9.6.0' not in t and 'version 9.7.2' in t
    alias = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
             'content_edits': [{'old': 'a', 'new': 'b'}]}
    assert revise_surfaces(alias) == {'content': 'b'}


def test_score_arm_counts_ops_by_kind():
    """The 2026-09-03 defect: connects and disconnects counted as creates,
    archives as revises (30 of 77 dumps carried a wrong `creates`)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
    from eval.encoder_prompt_ab import score_arm
    log = {'reads': [], 'rounds': 1, 'final_text': '',
           'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0},
           'writes': [{'tool': 'brain_batch', 'args': {'operations': [
               {'op': 'remember', 'title': 'n', 'content': 'c'},
               {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r', 'title': 't'},
               {'op': 'connect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
                'relation': 'similar_to', 'description': 'd'},
               {'op': 'archive', 'node_id': 'abcdef02', 'reason': 'gone'},
               {'op': 'disconnect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
                'relation': 'x'}]}}]}
    s = score_arm(log, set(), {})
    assert (s['creates'], s['revises'], s['connects'], s['archives']) == (1, 1, 1, 2)
