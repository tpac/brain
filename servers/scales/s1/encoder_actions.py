"""Actions condenser — the <actions> half of the encoder view policy.

Turns a turn's raw tool_result episodes into the lines the encoder reads.
Structure is parse → condense → render over typed records — heuristics never
operate on rendered strings, so a new tool or a new condensing pass can't
conflate with an existing one:

  parse    one place that knows tool shapes; any unknown tool degrades to a
           generic one-line record (total by construction, never an error)
  condense an ORDERED registry of passes, each doing one thing:
             P_policy  drop/stub per encoder_view.action_mode (existing policy)
             P0        exact-label dedup → first occurrence ×N (lossless)
             P2        per-turn budget; the middle rolls up into one
                       accounting line (tool mix + target union)
           P1 (similarity coalescing of near-identical sweeps) is a reserved
           slot — it wants more production fixtures before its grouping rule
           is frozen; P2 already collapses the floods it would target.
  render   records → plain text lines; the caller XML-escapes and indents

Invariants every pass must keep (tests/test_encoder_actions.py pins them):
- SELF-MARKING: every omission names itself in place ('×3', '… N more: …').
  Absence must never read as "nothing happened".
- COUNT-CONSERVING: rendered ×N counts + the rollup count sum to the true
  total of policy-visible actions — the render is auditable from itself.
- FILTER AT RENDER, NEVER AT CAPTURE: traces keep recording everything.

Policy constants (budgets, caps) live in encoder_view.py with the rest of
the view policy's vocabulary; this module is the machinery.
"""
import re
from collections import Counter

from servers.scales.s1.encoder_view import (
    ACTIONS_BUDGET, ACTIONS_BUDGET_TAIL, ACTIONS_KEEP_LAST, ACTION_LABEL_CAP,
    ROLLUP_TARGET_CAP, action_mode, action_stub)

# File-ish tokens (a directory part + basename with a short extension) — the
# rollup's target vocabulary. Deliberately extension-gated: bare dirs and
# flag soup stay out.
_TARGET_RE = re.compile(r'(?:[\w.@~-]+/)+[\w.@-]+\.[A-Za-z0-9]{1,5}\b')

# Absolute paths ≥4 segments squeeze to '…/<last three>' — the worktree and
# session-dir prefixes that repeat on every line carry no per-line signal.
_LONG_PATH_RE = re.compile(r'(?:/[\w.@~+-]+){4,}')

# Leading shell noise before the command that names the action's intent.
_SHELL_PREFIX_RE = re.compile(r'^(?:cd\s+\S+\s*(?:&&|;)\s*|sleep\s+\d+\s*(?:&&|;)\s*)+')


class _Action:
    __slots__ = ('tool', 'sub', 'label', 'targets', 'count')

    def __init__(self, tool, sub, label, targets):
        self.tool, self.sub, self.label = tool, sub, label
        self.targets, self.count = targets, 1


def _squeeze_paths(s):
    return _LONG_PATH_RE.sub(
        lambda m: '…/' + '/'.join(m.group(0).split('/')[-3:]), s)


def _one_line(s):
    return ' '.join(str(s or '').split())


def _cap(s):
    return s if len(s) <= ACTION_LABEL_CAP else s[:ACTION_LABEL_CAP] + '…'


def _short_tool(head):
    """'mcp__plugin_brain_brain__query_traces' → 'query_traces'."""
    return head.split('__')[-1] if head.startswith('mcp__') else head


def _bash_sub(args):
    """The command word that names a Bash action's intent — first token past
    any leading `cd … &&` / `sleep N;` noise, './dev' stripped to 'dev'."""
    args = _SHELL_PREFIX_RE.sub('', args.strip())
    tok = args.split()[0] if args.split() else ''
    return tok.lstrip('./') or ''


def _label(tool, first, rest_lines):
    """One line per action. Multi-line bodies (heredocs, `python3 -c`
    scripts, Edit old/new blocks) keep their first line and harvest the
    script's leading '#' intent comment when one exists — the comment is the
    information; naive first-line truncation would keep `python3 - <<'EOF'`
    and lose it. ' …' marks every multi-line trim."""
    comment = ''
    for ln in rest_lines[:8]:
        stripped = ln.strip()
        if stripped.startswith('#'):
            comment = stripped.lstrip('# ').strip()
            break
    if comment:
        return '%s · %s' % (first, comment)
    return first + (' …' if rest_lines else '')


def parse_action(episode):
    """One tool_result episode → an _Action, or None when existing policy
    drops the line (node-ops provenance already shows). Total: an unseen
    tool shape falls through to the generic one-line record."""
    summary = str(episode.get('summary') or '')
    md = episode.get('metadata')
    raw_tool = md.get('tool') if isinstance(md, dict) else None

    mode = action_mode(raw_tool)
    if mode == 'drop':
        return None
    if mode == 'stub':
        stub = action_stub(summary)
        return _Action(stub.split(':', 1)[0], '', stub, ())

    lines = summary.split('\n')
    first = _one_line(lines[0])
    head, _, args = first.partition(': ')
    tool = _short_tool(head) if head else 'tool'
    sub = _bash_sub(args) if tool == 'Bash' else ''
    label = _cap(_squeeze_paths(_label(tool, first, lines[1:])))
    targets = tuple(_squeeze_paths(t) for t in _TARGET_RE.findall(summary)[:6])
    return _Action(tool, sub, label, targets)


def _dedup(actions):
    """P0 — exact-label repeats fold into their first occurrence ×N.
    Order of first occurrences is preserved; lossless by construction."""
    kept, by_label = [], {}
    for a in actions:
        prior = by_label.get(a.label)
        if prior is not None:
            prior.count += a.count
        else:
            by_label[a.label] = a
            kept.append(a)
    return kept


def _rollup_line(mid):
    """The accounting line for the rolled-up middle: total, tool mix (Bash
    broken down by command word), and the union of file targets."""
    total = sum(a.count for a in mid)
    tools = Counter()
    subs = Counter()
    targets, seen = [], set()
    for a in mid:
        tools[a.tool] += a.count
        if a.tool == 'Bash' and a.sub:
            subs[a.sub] += a.count
        for t in a.targets:
            if t not in seen:
                seen.add(t)
                targets.append(t)
    parts = []
    for tool, cnt in tools.most_common():
        part = '%s ×%d' % (tool, cnt)
        if tool == 'Bash' and subs:
            part += ' (%s)' % ', '.join(
                '%s ×%d' % (s, c) for s, c in subs.most_common(4))
        parts.append(part)
    line = '… %d more action(s): %s' % (total, ', '.join(parts))
    if targets:
        shown = targets[:ROLLUP_TARGET_CAP]
        more = len(targets) - len(shown)
        line += ' — targets: %s' % ', '.join(shown)
        if more > 0:
            line += ', +%d more' % more
    return line


def _render(a):
    return a.label + (' ×%d' % a.count if a.count > 1 else '')


def condense_actions(episodes, is_tail=False):
    """Episodes of one turn → the lines its <actions> element renders.
    `is_tail`: the newest turn — the encoder's actual working material —
    gets the larger budget; older unencoded turns the smaller."""
    actions = [a for a in (parse_action(e) for e in episodes) if a]
    actions = _dedup(actions)

    budget = ACTIONS_BUDGET_TAIL if is_tail else ACTIONS_BUDGET
    # Soft edge: a rollup of one or two actions costs more than it saves —
    # only condense when the middle is worth a line.
    if len(actions) <= budget + 2:
        return [_render(a) for a in actions]

    head_n = budget - ACTIONS_KEEP_LAST
    head, mid, tail = (actions[:head_n],
                       actions[head_n:-ACTIONS_KEEP_LAST],
                       actions[-ACTIONS_KEEP_LAST:])
    return ([_render(a) for a in head]
            + [_rollup_line(mid)]
            + [_render(a) for a in tail])
