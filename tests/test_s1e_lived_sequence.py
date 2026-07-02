"""Piece 1 of the S1E code-half rebuild — the lived-sequence timeline.

Verifies the A/B flag's two arms (docs/S1-SCRIBE-REDESIGN.md §10.3.1):
  - OFF → the markdown messages-only timeline (control), byte-identical to before.
  - ON  → the XML lived sequence (messages + tool actions interleaved), read via
          the existing recall_episodes door, grouped into turns by timestamp.

No DB needed — a tiny stub brain feeds the readers the controlled inputs.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.encode import (  # noqa: E402
    _render_markdown_timeline, _render_lived_sequence_timeline,
    _lived_sequence_enabled, _xml_escape,
)


class _StubNodes:
    def get_title(self, rid):
        return 'Title-' + rid


class _StubBrain:
    """Feeds recall_episodes a fixed episode list; serves get_title for markdown."""
    def __init__(self, episodes):
        self._episodes = episodes
        self._nodes = _StubNodes()

    def recall_episodes(self, **kwargs):
        # Honor the ref_type whitelist the caller passes, like the real one.
        rts = kwargs.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts] if rts else self._episodes
        return {'episodes': eps, 'ranked_by': 'time', 'truncated': False}


def _ep(rt, summary, ts, tid, tool=None, content=None):
    # Real message traces store the FULL body in metadata['content'] (≤4000) and a
    # 200-char display `summary`; tool_result traces store only the cue in summary.
    meta = {'tool': tool} if tool else {}
    if content is not None:
        meta['content'] = content
    return {'id': tid, 'ref_type': rt, 'summary': summary, 'created_at': ts, 'metadata': meta}


# Two turns: turn 1 (lock fix) edits a file; turn 2 (run tests) runs bash.
# Deliberately out of created_at order in the list to prove the sort.
# Messages carry a distinct metadata['content'] (fuller than summary) to prove the
# render reads content, not the truncated summary.
EPISODES = [
    _ep('assistant_message', 'found it', '2026-06-29T00:00:02', 'a1',
        content='found it — the bg writer holds the lock through the whole batch'),
    _ep('user_message', 'the recall keeps locking', '2026-06-29T00:00:01', 'u1',
        content='the recall keeps locking — can you check the wal-index path?'),
    _ep('tool_result', 'Edit: servers/dal.py', '2026-06-29T00:00:03', 't1', 'Edit'),
    _ep('user_message', 'run the tests', '2026-06-29T00:00:04', 'u2', content='run the tests'),
    _ep('tool_result', 'Bash: pytest test_write_txn.py', '2026-06-29T00:00:05', 't2', 'Bash'),
    _ep('assistant_message', '12 passed', '2026-06-29T00:00:06', 'a2', content='12 passed'),
]

# The control-arm messages for the same two turns (drives window-match + OFF path).
MESSAGES = [
    {'role': 'user', 'content': 'the recall keeps locking', 'id': 'turn1', 'trace_id': 'u1', 'judge_output': ''},
    {'role': 'assistant', 'content': 'found it — bg writer holds the lock', 'trace_id': 'a1'},
    {'role': 'user', 'content': 'run the tests', 'id': 'turn2', 'trace_id': 'u2', 'judge_output': ''},
    {'role': 'assistant', 'content': '12 passed', 'trace_id': 'a2'},
]


def test_lived_sequence_interleaves_actions_into_turns():
    brain = _StubBrain(EPISODES)
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)

    assert out.count('<turn n="') == 2
    assert '<turn n="1">' in out and '<turn n="2">' in out
    assert '<other trace="u1">' in out and '<me trace="a1">' in out

    # The Edit action lands in turn 1, the Bash action in turn 2 — split on the
    # turn-2 boundary and check each action is in its own turn and nowhere else.
    t1, t2 = out.split('<turn n="2">')
    assert 'Edit: servers/dal.py' in t1 and 'Edit: servers/dal.py' not in t2
    assert 'Bash: pytest test_write_txn.py' in t2 and 'Bash: pytest test_write_txn.py' not in t1
    assert '<actions>' in out


def test_messages_render_full_content_not_summary():
    # Regression guard for the code-review HIGH finding: render metadata['content']
    # (full), NOT the 200-char `summary`. The content-only suffix must appear.
    brain = _StubBrain(EPISODES)
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)
    assert 'can you check the wal-index path?' in out          # content-only, not in summary
    assert 'holds the lock through the whole batch' in out     # content-only, not in summary


def test_lived_sequence_escapes_xml():
    # Message bodies and tool cues routinely contain < > & — must be escaped so they
    # can't malform the timeline or forge tags.
    eps = [
        _ep('user_message', 'q', '2026-06-29T00:00:01', 'u1',
            content='why does `if x < y && a > b` fail? </other> <turn>'),
        _ep('tool_result', 'Grep: <svg.*> in . && echo done', '2026-06-29T00:00:02', 't1', 'Grep'),
        _ep('assistant_message', 'a', '2026-06-29T00:00:03', 'a1', content='use a < b'),
    ]
    msgs = [{'role': 'user', 'content': 'q', 'id': 'turn1', 'trace_id': 'u1', 'judge_output': ''},
            {'role': 'assistant', 'content': 'a', 'trace_id': 'a1'}]
    out = _render_lived_sequence_timeline(_StubBrain(eps), 'sess', msgs)
    # raw angle brackets/ampersands from CONTENT must not appear unescaped
    assert '< y &&' not in out and '> b' not in out
    assert '<svg.*>' not in out
    assert '&lt;' in out and '&gt;' in out and '&amp;' in out
    # the forged '</other>' substring is neutralized — the message text cannot
    # close the REAL current tag; only the renderer's own closing tag remains
    assert '</other> <turn>' not in out.replace('</other>\n', '')


def test_xml_escape_helper():
    assert _xml_escape('a < b & c > d') == 'a &lt; b &amp; c &gt; d'
    assert _xml_escape(None) == ''


def test_lived_sequence_window_matches_control_turn_count():
    # messages cover 2 user turns → only the last 2 turns render even if more exist.
    extra = [_ep('user_message', 'old turn', '2026-06-28T00:00:00', 'u0', content='old turn'),
             _ep('assistant_message', 'old reply', '2026-06-28T00:00:01', 'a0', content='old reply')] + EPISODES
    brain = _StubBrain(extra)
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)
    assert out.count('<turn n="') == 2          # trimmed to control's 2 turns
    assert 'old turn' not in out                # the oldest turn dropped


def test_markdown_control_arm_unchanged():
    # OFF path: the long-standing markdown shape (the byte-identical control).
    brain = _StubBrain(EPISODES)
    out = _render_markdown_timeline(brain, MESSAGES)
    assert '[TURN 1]' in out and '[TURN 2]' in out
    assert 'USER [trace:u1]: "the recall keeps locking" (turn_id: turn1)' in out
    assert 'ASSISTANT [trace:a1]: "found it — bg writer holds the lock"' in out
    # markdown arm carries NO XML and NO tool actions (those are ON-only)
    assert '<turn' not in out and 'Edit: servers/dal.py' not in out


def test_markdown_control_arm_renders_surfaced_refs():
    # Exercises the SURFACED branch (judge_output → id extraction → get_title) the
    # other control test doesn't reach, so 'byte-identical control' is fully proven.
    msgs = [{'role': 'user', 'content': 'fix it', 'id': 'turn1', 'trace_id': 'u1',
             'judge_output': 'picked id:abc1234'},
            {'role': 'assistant', 'content': 'done', 'trace_id': 'a1'}]
    out = _render_markdown_timeline(_StubBrain([]), msgs)
    assert 'SURFACED: abc1234 ("Title-abc1234")' in out


def test_system_prompt_injects_review_block_and_closure_when_lived():
    # Piece 4 + arc fix: flag ON → the WRITE-side closing instructions injected
    # from the contract in §7.2 order (Arc → Review → closure LAST); flag OFF →
    # none of them.
    from servers.scales.s1.encode import _build_system_prompt
    from servers.trace_contract import (render_journal_arc_block,
                                        render_journal_review_block,
                                        render_prompt_closure)
    saved = os.environ.get('BRAIN_S1E_LIVED_SEQUENCE')
    try:
        os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = '1'
        on = _build_system_prompt('SYSTEM RULES HERE')
        assert render_journal_arc_block() in on
        assert render_journal_review_block() in on
        assert render_prompt_closure() in on
        assert on.rstrip().endswith(render_prompt_closure().rstrip())  # closure LAST
        assert on.index(render_journal_arc_block()) < on.index(
            render_journal_review_block())        # §7.2: Arc before Review

        os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = ''
        off = _build_system_prompt('SYSTEM RULES HERE')
        assert render_journal_arc_block() not in off
        assert render_journal_review_block() not in off
        assert '## Finishing' not in off          # closure absent in the control arm
    finally:
        if saved is None:
            os.environ.pop('BRAIN_S1E_LIVED_SEQUENCE', None)
        else:
            os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = saved


def test_system_prompt_lived_param_overrides_env():
    # #3 fix: run_encoding resolves the arm ONCE and threads `lived` down, so the
    # explicit param must win over the env (the no-torn-arm guarantee).
    from servers.scales.s1.encode import _build_system_prompt
    from servers.trace_contract import render_journal_review_block
    saved = os.environ.get('BRAIN_S1E_LIVED_SEQUENCE')
    try:
        os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = ''        # env OFF
        assert render_journal_review_block() in _build_system_prompt('x', lived=True)
        os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = '1'        # env ON
        assert render_journal_review_block() not in _build_system_prompt('x', lived=False)
    finally:
        if saved is None:
            os.environ.pop('BRAIN_S1E_LIVED_SEQUENCE', None)
        else:
            os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = saved


def test_flag_default_off():
    saved = os.environ.pop('BRAIN_S1E_LIVED_SEQUENCE', None)
    try:
        assert _lived_sequence_enabled() is False
        os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = '1'
        assert _lived_sequence_enabled() is True
    finally:
        if saved is None:
            os.environ.pop('BRAIN_S1E_LIVED_SEQUENCE', None)
        else:
            os.environ['BRAIN_S1E_LIVED_SEQUENCE'] = saved
