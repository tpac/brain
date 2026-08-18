"""Actions condenser (servers/scales/s1/encoder_actions.py) — parse →
condense → render over typed records.

Pins the module's three invariants:
  • SELF-MARKING — every omission names itself ('×N', the '… N more' rollup)
  • COUNT-CONSERVING — rendered ×N counts + rollup count = policy-visible total
  • TOTAL PARSE — unknown tools degrade to a generic record, never an error

Fixtures mirror the three production timelines the design was built from
(review-sweep flood, build session, journal-check mix).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.encoder_actions import (  # noqa: E402
    condense_actions, parse_action)
from servers.scales.s1.encoder_view import (  # noqa: E402
    ACTIONS_BUDGET, ACTIONS_BUDGET_TAIL, ACTIONS_KEEP_LAST)

PLUGIN_TOOL = 'mcp__plugin_brain_brain__%s'


def _ep(summary, tool='Bash'):
    return {'summary': summary, 'metadata': {'tool': tool}}


# ── parse: tool families ──

def test_plain_bash_single_line():
    a = parse_action(_ep('Bash: git status --short'))
    assert a.tool == 'Bash' and a.sub == 'git'
    assert a.label == 'Bash: git status --short'


def test_bash_shell_prefix_skipped_for_subcommand():
    a = parse_action(_ep('Bash: cd /Users/x/brain && git merge main --no-edit'))
    assert a.sub == 'git'


def test_heredoc_harvests_intent_comment():
    a = parse_action(_ep(
        "Bash: python3 - <<'EOF'\n"
        '# dal_graph.py: delete decay_edges + DEFAULT_EXCLUDED_RELATIONS\n'
        "p = 'servers/dal_graph.py'\nsrc = open(p).read()"))
    assert 'delete decay_edges + DEFAULT_EXCLUDED_RELATIONS' in a.label
    assert '\n' not in a.label


def test_python_dash_c_comment_on_later_line():
    a = parse_action(_ep(
        'Bash: ./dev python3 -c "\n'
        'from servers.daemon_client import send_command\n'
        "# the daemon's own view: is the policy on?\n"
        "r = send_command('ping', {})"))
    assert "the daemon's own view" in a.label
    assert a.sub == 'dev'


def test_multiline_without_comment_marks_trim():
    a = parse_action(_ep('Edit: /Users/x/brain/servers/foo.py\n'
                         '  old: class Test03\n  new: class Test03', tool='Edit'))
    assert a.label.endswith(' …')          # the old/new heads are trimmed, marked
    assert 'old:' not in a.label


def test_long_paths_squeeze_to_last_three_segments():
    a = parse_action(_ep(
        'Read: /Users/tpac/brain/.claude/worktrees/retire-edge-families-719e30/'
        'servers/scales/s2/community.py', tool='Read'))
    assert a.label == 'Read: …/scales/s2/community.py'


def test_mcp_tool_name_shortens():
    a = parse_action(_ep(PLUGIN_TOOL % 'query_traces' + ': {"scale": "s1"}',
                         tool=PLUGIN_TOOL % 'query_traces'))
    assert a.tool == 'query_traces'


def test_unknown_tool_is_total():
    a = parse_action(_ep('SomeFutureTool: whatever args', tool='SomeFutureTool'))
    assert a is not None and a.label.startswith('SomeFutureTool:')


def test_existing_policy_drop_and_stub_respected():
    dropped = parse_action(_ep(PLUGIN_TOOL % 'remember' + ': {...}',
                               tool=PLUGIN_TOOL % 'remember'))
    assert dropped is None                 # provenance owns node-ops
    stubbed = parse_action(_ep(PLUGIN_TOOL % 'recall' + ': {"query": "x"}',
                               tool=PLUGIN_TOOL % 'recall'))
    assert stubbed.label.endswith('→ results in provenance')


# ── condense: dedup + budget ──

def test_exact_repeats_fold_to_count():
    lines = condense_actions([_ep('Bash: git show a73e622 --stat')] * 3)
    assert lines == ['Bash: git show a73e622 --stat ×3']


def test_under_budget_renders_verbatim_no_rollup():
    eps = [_ep('Bash: step %d' % i) for i in range(ACTIONS_BUDGET)]
    lines = condense_actions(eps)
    assert len(lines) == ACTIONS_BUDGET
    assert not any(ln.startswith('…') for ln in lines)


def test_over_budget_rolls_middle_and_keeps_last():
    n = 60
    eps = [_ep('Bash: grep sweep %d servers/dal_graph.py' % i) for i in range(n - 1)]
    eps.append(_ep('Bash: git commit -m done'))
    lines = condense_actions(eps)
    assert len(lines) == ACTIONS_BUDGET + 1          # budget lines + rollup
    assert lines[-1] == 'Bash: git commit -m done'   # outcome survives verbatim
    rollup = [ln for ln in lines if ln.startswith('…')]
    assert len(rollup) == 1 and 'targets:' in rollup[0]
    assert 'servers/dal_graph.py' in rollup[0]


def test_count_conservation_over_mixed_flood():
    # 105-action review-sweep shape: greps, git shows, reads, edits, dups.
    eps = ([_ep('Bash: grep -rn "SYM_%d" servers/' % i) for i in range(70)]
           + [_ep('Bash: git show a73e622 --stat')] * 3
           + [_ep('Read: /Users/x/brain/servers/f%d.py' % i, tool='Read')
              for i in range(20)]
           + [_ep('Edit: /Users/x/brain/servers/g%d.py\n  old: a\n  new: b' % i,
                  tool='Edit') for i in range(12)])
    total = 70 + 3 + 20 + 12
    lines = condense_actions(eps)
    rendered = 0
    for ln in lines:
        if ln.startswith('…'):
            rendered += int(re.match(r'… (\d+) more', ln).group(1))
        else:
            m = re.search(r' ×(\d+)$', ln)
            rendered += int(m.group(1)) if m else 1
    assert rendered == total


def test_tail_turn_gets_larger_budget():
    eps = [_ep('Bash: step %d' % i) for i in range(ACTIONS_BUDGET_TAIL)]
    assert len(condense_actions(eps, is_tail=True)) == ACTIONS_BUDGET_TAIL
    assert len(condense_actions(eps, is_tail=False)) == ACTIONS_BUDGET + 1


def test_rollup_reports_tool_mix_with_bash_subcommands():
    eps = ([_ep('Bash: grep -n "x" f%d' % i) for i in range(30)]
           + [_ep('Bash: sed -n "1,5p" f%d' % i) for i in range(10)]
           + [_ep('Read: /a/b/c/d%d.py' % i, tool='Read') for i in range(8)])
    lines = condense_actions(eps)
    rollup = next(ln for ln in lines if ln.startswith('…'))
    assert 'Bash ×' in rollup and 'grep ×' in rollup and 'Read ×' in rollup


def test_keep_last_constant_is_honored():
    eps = [_ep('Bash: a%d' % i) for i in range(50)]
    lines = condense_actions(eps)
    tail_labels = ['Bash: a%d' % i for i in range(50 - ACTIONS_KEEP_LAST, 50)]
    assert lines[-ACTIONS_KEEP_LAST:] == tail_labels
