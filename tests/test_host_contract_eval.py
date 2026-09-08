"""The paired-code gate rejects input drift, including the observed clock trap."""
import json

import pytest

from tests.eval_optional import require_eval
require_eval()

from eval.host_contract_eval import compare_inputs


def _pair(tmp_path):
    report = dict(snapshot_hashes={'brain.db': 'frozen'}, session='session',
                  render_now='2026-09-08T17:25:48+00:00', flags={}, messages=[],
                  catalog_ids=[], interaction_stamp={}, interaction_config={},
                  eval_execution={'model': 'same-model', 'effort': 'API default'},
                  stamped_edits=1,
                  calls=[{'is_tail': True,
                          'episodes': [{'metadata': {'kind': 'edit', 'kind_status': 'ok'}}],
                          'parsed': [{'protected': True, 'label': 'edit: src/code.py'}]}])
    before, after = tmp_path / 'before', tmp_path / 'after'
    for p, action in ((before, 'rolled up'), (after, 'edit: src/code.py')):
        p.mkdir()
        report['calls'][0]['lines'] = [action]
        (p / 'input.json').write_text(json.dumps(report))
        (p / 'system.txt').write_text('same system')
        (p / 'tools.json').write_text('[]')
        (p / 'prompt.txt').write_text('catalog age 5m\n<actions>' + action + '</actions>')
    return before, after, report


def test_pair_allows_only_action_changes(tmp_path):
    before, after, _ = _pair(tmp_path)
    assert compare_inputs(before, after)['paired'] is True
    prompt = (after / 'prompt.txt').read_text()
    (after / 'prompt.txt').write_text(prompt.replace('age 5m', 'age 12m'))
    with pytest.raises(ValueError, match='outside actions'):
        compare_inputs(before, after)


@pytest.mark.parametrize('field', ['snapshot_hashes', 'render_now', 'messages',
                                   'interaction_config', 'eval_execution'])
def test_pair_refuses_changed_substrate(tmp_path, field):
    before, after, report = _pair(tmp_path)
    report[field] = 'changed'
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=field):
        compare_inputs(before, after)


def test_pair_refuses_unprotected_stamped_edits(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['parsed'][0]['protected'] = False
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not all protected'):
        compare_inputs(before, after)


def test_pair_refuses_changed_budget_input(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['is_tail'] = False
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='different action windows'):
        compare_inputs(before, after)


def test_pair_accepts_legacy_metadata_absence(tmp_path):
    before, after, _ = _pair(tmp_path)
    for path in (before, after):
        report = json.loads((path / 'input.json').read_text())
        report['calls'][0]['episodes'].append({'metadata': None})
        report['calls'][0]['parsed'].append({'protected': False})
        (path / 'input.json').write_text(json.dumps(report))
    assert compare_inputs(before, after)['protected_edits'] == 1


def test_pair_refuses_protected_edits_lost_by_condenser(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['lines'] = ['all actions rolled up']
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not all retained'):
        compare_inputs(before, after)


def test_pair_refuses_action_blocks_lost_by_prompt_renderer(tmp_path):
    before, after, _ = _pair(tmp_path)
    (after / 'prompt.txt').write_text('catalog age 5m\n<actions>ALL ACTIONS LOST</actions>')
    with pytest.raises(ValueError, match='omits a measured action block'):
        compare_inputs(before, after)


def test_eval_settings_apply_to_initial_and_followup_requests(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from eval import s1_encode_eval as runner

    monkeypatch.setattr(runner, 'EVAL_MODEL', 'fixture-model')
    monkeypatch.setattr(runner, 'EVAL_MAX_TOKENS', 128)
    monkeypatch.setattr(runner, 'EVAL_TOOL_ROUNDS', 2)
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        content=[SimpleNamespace(type='tool_use', id='call', name='remember_batch',
                                 input={'nodes': []})])
    client = Mock()
    client.messages.create.return_value = response
    result = runner.run_encoding(client, 'system', 'content', [], object())
    assert client.messages.create.call_count == 3
    assert len(result['actions']) == 2
    for call in client.messages.create.call_args_list:
        assert call.kwargs['model'] == 'fixture-model'
        assert call.kwargs['max_tokens'] == 128
        assert 'output_config' not in call.kwargs
