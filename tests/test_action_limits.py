"""The total action budget survives protected floods, Unicode and view controls."""
import re
import xml.etree.ElementTree as ET

import pytest

from servers.action_policy import ActionPolicy
from servers.scales.s1.encoder_actions import (
    ActionBlock, ActionLine, prepare_action_block, render_action_blocks)


def _ep(cue, kind='shell', tool='Bash', **metadata):
    return {'summary': cue, 'metadata': {'tool': tool, 'kind': kind, 'kind_status': 'ok',
            'vocab_version': 1, 'impl_identity': 'test', **metadata}}


def _prepare(episodes, policy=ActionPolicy(), **kwargs):
    return prepare_action_block(episodes, is_tail=True, encoded=False,
                                view_policy=True, policy=policy, **kwargs)


def test_thin_rolls_import_scaffolding_without_claiming_read_only():
    episodes = [_ep("Bash: ./dev python3 - <<'PY'\nimport json\n" +
                    'target = "files/f%d.json"\nwrite_unknown()' % i) for i in range(30)]
    block = _prepare(episodes)
    assert sum(line.count for line in block.lines) == 30
    assert all('import json' not in line.text for line in block.lines)
    assert any('script' in line.text and 'files/' in line.text for line in block.lines)
    assert not any('read-only' in line.text for line in block.lines)


def test_edit_groups_are_distinct_and_do_not_cross_other_operations():
    a = _ep('Edit: /repo/a.py\nold: x\nnew: y', 'edit', 'Edit')
    b = _ep('Edit: /repo/a.py\nold: y\nnew: z', 'edit', 'Edit')
    block = _prepare([a, b, _ep('Bash: pytest test_a.py'), a,
                      _ep('Bash: pytest test_b.py'), _ep('Bash: deploy app')])
    assert sum(line.count for line in block.lines) == 6
    assert '2 edit actions' in block.lines[0].text
    assert '×' not in block.lines[0].text
    assert block.lines[1].text == 'Bash: pytest test_a.py'
    assert 'Edit: /repo/a.py' in block.lines[2].text


def test_repeated_inspection_keeps_edit_groups_separate():
    read = _ep('Bash: cat a.py')
    block = _prepare([read, _ep('Edit: a.py\nold: x', 'edit', 'Edit'),
                      read, _ep('Edit: a.py\nold: y', 'edit', 'Edit'),
                      _ep('Bash: pytest'), _ep('Bash: deploy')])
    assert not any('edit actions' in line.text for line in block.lines)
    assert sum(line.count for line in block.lines) == 6


@pytest.mark.parametrize('ending', ['\r', '\r\n', '\n'])
def test_raw_view_cannot_bypass_line_limit(ending):
    policy = ActionPolicy(max_lines=4)
    block = prepare_action_block([_ep('Read: ' + ending.join(['x'] * 70), 'read', 'Read')],
                                 is_tail=True, encoded=False, view_policy=False, policy=policy)
    rendered, notice = render_action_blocks([block], policy)
    assert rendered == ['']
    assert '1 actions omitted' in notice


def test_raw_view_keeps_diagnostics_and_edit_priority():
    block = prepare_action_block([_ep('Edit: a.py', 'edit', 'Edit'),
                                  _ep('Bash: script', capture_filter_incomplete=True),
                                  _ep('Future: something', kind_status='unknown', kind='')],
                                 is_tail=True, encoded=False, view_policy=False, policy=ActionPolicy())
    assert [line.priority for line in block.lines] == [2, 3, 3]
    assert 'capture filter incomplete' in block.lines[1].text
    assert 'tool kind unknown' in block.lines[2].text


@pytest.mark.parametrize('view_policy,encoded', [(False, False), (True, False), (True, True)])
def test_git_exclusion_precedes_every_view_branch(view_policy, encoded):
    eps = [_ep('Bash: git status && rg useful .'), _ep('Bash: pytest')]
    block = prepare_action_block(eps, is_tail=True, encoded=encoded,
                                 view_policy=view_policy, policy=ActionPolicy())
    assert sum(line.count for line in block.lines) == 1
    assert 'git' not in str(block)
    assert 'useful' not in str(block)


def test_captured_git_fact_survives_a_short_historical_summary():
    block = _prepare([_ep('Bash: echo prefix ...', git_invocation=True)])
    assert not block.lines


def test_diagnostics_are_not_dropped_as_git_or_boilerplate():
    block = _prepare([_ep('Bash: git status', kind_status='malformed'),
                      _ep('Bash: git status', capture_filter_incomplete=True)])
    assert sum(line.count for line in block.lines) == 2
    assert all(line.priority == 3 for line in block.lines)


@pytest.mark.parametrize('policy', [ActionPolicy(), ActionPolicy('full', False, 4, 512),
                                  ActionPolicy('balanced', True, 200, 24000)])
def test_hard_window_cap_counts_escaped_bytes_and_priority_overflow(policy):
    blocks = [ActionBlock([ActionLine('Edit %d/%d: &<🧠>' % (ti, ai) + '界' * 90,
                                      count=ai + 1, priority=2 if ai % 2 else 3)
                           for ai in range(40)]) for ti in range(30)]
    rendered, notice = render_action_blocks(blocks, policy)
    payload = ''.join(rendered) + notice
    assert len(payload.encode('utf-8')) <= policy.max_bytes
    root = ET.fromstring('<timeline>' + payload + '</timeline>')
    detail_lines = [ln.strip() for action in root.findall('actions')
                    for ln in (action.text or '').splitlines() if ln.strip()]
    assert len(detail_lines) + 1 <= policy.max_lines
    assert root.find('action_limit') is not None
    selected = [line for block in blocks for line in block.lines if line.text in detail_lines]
    omitted = int(re.match(r'(\d+) actions omitted', root.find('action_limit').text)[1])
    assert sum(line.count for line in selected) + omitted == sum(
        line.count for block in blocks for line in block.lines)
    assert 'priority actions' in notice


def test_newest_priority_records_win_but_render_chronologically():
    blocks = [ActionBlock([ActionLine('edit-%d' % n, priority=2)]) for n in range(20)]
    rendered, notice = render_action_blocks(blocks, ActionPolicy(max_lines=4))
    assert [i for i, xml in enumerate(rendered) if xml] == [17, 18, 19]
    assert '17 actions omitted across 17 turns' in notice


def test_huge_legacy_line_and_many_inline_stubs_cannot_bypass_cap():
    blocks = [ActionBlock([ActionLine('&\n' * 10000, priority=3)])]
    blocks += [ActionBlock([ActionLine('trimmed — already read', count=300)], inline=True)
               for _ in range(1000)]
    policy = ActionPolicy(max_lines=4, max_bytes=512)
    rendered, notice = render_action_blocks(blocks, policy)
    assert not rendered[0]
    assert len((''.join(rendered) + notice).encode()) <= 512
    assert sum(bool(x) for x in rendered) <= 3
    assert 'priority actions' in notice


def test_small_payload_is_unchanged_and_has_no_omission_notice():
    block = ActionBlock([ActionLine('Bash: pytest & check')])
    rendered, notice = render_action_blocks([block], ActionPolicy())
    assert rendered == ['  <actions>\n    Bash: pytest &amp; check\n  </actions>\n']
    assert notice == ''


@pytest.mark.parametrize('view_policy', [False, True])
def test_production_timeline_enforces_one_budget_and_preserves_messages(monkeypatch, view_policy):
    from tests.test_s1e_lived_sequence import _StubBrain
    from servers.scales.s1.encode import _render_lived_sequence_timeline

    episodes, messages = [], []
    for i in range(20):
        uid = 'u%d' % i
        messages.append({'role': 'user', 'content': uid, 'trace_id': uid, 'id': uid})
        episodes.extend([
            {'id': uid, 'ref_type': 'user_message', 'summary': 'message-' + uid,
             'created_at': '2026-09-08T00:%02d:00' % i},
            {'id': 'a%d' % i, 'ref_type': 'tool_result',
             'created_at': '2026-09-08T00:%02d:01' % i,
             **_ep('Edit: /repo/important-%d.py' % i, 'edit', 'Edit')},
        ])
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_LINES', '4')
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_BYTES', '512')
    rendered = _render_lived_sequence_timeline(_StubBrain(episodes), 'session', messages,
                                              view_policy=view_policy)
    root = ET.fromstring('<timeline>' + rendered + '</timeline>')
    assert len(root.findall('turn/actions')) == 3
    assert root.find('action_limit').text.startswith('17 actions omitted across 17 turns')
    assert [turn.find('other').text for turn in root.findall('turn')] == [
        'message-u%d' % i for i in range(20)]
    assert [turn.find('actions').text.strip() for turn in root.findall('turn')
            if turn.find('actions') is not None] == [
        'Edit: /repo/important-%d.py' % i for i in (17, 18, 19)]
