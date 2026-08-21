"""Actions condenser — the <actions> half of the encoder view policy.

Turns a turn's raw tool_result episodes into the lines the encoder reads.
Three steps — parse, condense, render — over records, never over rendered
strings, so a new tool or a new condensing heuristic can't conflate with an
existing one. Adding a heuristic = adding one pure function to the condense
chain in condense_actions (similarity-coalescing of near-identical sweeps is
deliberately deferred until more production fixtures exist; the budget
already collapses the floods it would target).

  parse    one place that knows tool shapes; any unknown tool degrades to a
           generic one-line record (total by construction, never an error)
  condense drop/stub per encoder_view.action_mode (existing policy), then
           exact-repeat dedup, then the per-turn budget. Three protections
           bound what may be condensed away:
             - the last ACTIONS_KEEP_LAST actions render verbatim in place
               (the turn's outcome) and are never folded INTO earlier lines
             - write actions (Edit/Write/git write-verbs/intent scripts)
               never roll up — they are rare and they are the story
             - dedup identity is the RAW summary, never the rendered label,
               so two different actions can never fold into a false '×N'
  render   records → plain text lines; the caller XML-escapes and indents

Invariants (tests/test_encoder_actions.py pins them):
- SELF-MARKING: every omission names itself in place — '×N' (a true repeat
  of the same recorded action), the '(N more actions, not shown: …)'
  accounting line, ' …' on every trimmed multi-line body, '+k more' on every
  capped breakdown, '/…/' on every shortened path.
- COUNT-CONSERVING: rendered ×N counts + the accounting line's count sum to
  the true total of policy-visible actions.
- TOTAL PARSE: any input degrades to a generic record, never raises.
- FILTER AT RENDER, NEVER AT CAPTURE: traces keep recording everything.

Policy constants (budgets, caps, protected-tool sets) live in
encoder_view.py with the rest of the view policy's vocabulary; this module
is the machinery. The s1e prompt's <actions> gloss documents this render's
notation — a render change here that adds notation must ride a prompt
version (the stale-gloss rule).
"""
import re
from collections import Counter

from servers.scales.s1.encoder_view import (
    ACTIONS_BUDGET, ACTIONS_BUDGET_TAIL, ACTIONS_KEEP_LAST, ACTION_LABEL_CAP,
    ACTIONS_BUDGET_SOFT_EDGE, COMMENT_SCAN_DEPTH, PATH_KEEP_SEGMENTS,
    ROLLUP_SUBS_CAP, ROLLUP_TOOLS_CAP, ROLLUP_TARGET_CAP,
    WRITE_ACTION_TOOLS, GIT_WRITE_VERBS, action_mode, action_stub)

# File-ish tokens (optional leading '/', a directory part, basename with a
# short extension) — the rollup's target vocabulary. Extension-gated: bare
# dirs and flag soup stay out. URL innards are rejected at the call site.
_TARGET_RE = re.compile(r'/?(?:[\w.@~-]+/)+[\w.@-]+\.[A-Za-z0-9]{1,5}\b')

# Absolute paths ≥4 segments squeeze to '/…/<last PATH_KEEP_SEGMENTS>' — the
# worktree/session prefixes that repeat on every line carry no per-line
# signal. The lookbehind keeps the match off URL innards ('https://host/a/b')
# and protocol-relative forms: a real path start is never preceded by a word
# char, ':' or another '/'.
_LONG_PATH_RE = re.compile(r'(?<![\w:/])(?:/[\w.@~+-]+){4,}')

# Leading shell noise before the command word that names the intent.
# cd targets may be quoted (paths with spaces).
_SHELL_PREFIX_RE = re.compile(
    r'^(?:cd\s+(?:"[^"]*"|\'[^\']*\'|\S+)\s*(?:&&|;)\s*'
    r'|sleep\s+\d+\s*(?:&&|;)\s*)+')

# Wrapper words the Bash verb extractor walks past to find the real command
# ('./dev python3 -m pytest …' counts as pytest, not dev).
_VERB_WRAPPERS = frozenset({'dev', 'python', 'python3', 'uv', 'env', 'nice',
                            'time', 'timeout', 'caffeinate'})

# Heredoc / inline-script openers: the first line is scaffolding; the intent
# lives in the body (a '#' comment, or failing that the first code line).
_SCRIPT_OPENER_RE = re.compile(r'''(?:<<-?\s*['"]?\w+['"]?\s*$|-c\s+["']\s*$)''')


class _Action:
    __slots__ = ('raw', 'tool', 'sub', 'label', 'targets', 'protected', 'count')

    def __init__(self, raw, tool, sub, label, targets, protected):
        self.raw, self.tool, self.sub = raw, tool, sub
        self.label, self.targets = label, targets
        self.protected, self.count = protected, 1


def _squeeze_paths(s):
    return _LONG_PATH_RE.sub(
        lambda m: '/…/' + '/'.join(m.group(0).split('/')[-PATH_KEEP_SEGMENTS:]), s)


def _one_line(s):
    return ' '.join(str(s or '').split())


def _cap(s):
    return s if len(s) <= ACTION_LABEL_CAP else s[:ACTION_LABEL_CAP] + '…'


def _short_tool(head):
    """'mcp__<server>__query_traces' → 'query_traces'."""
    return head.split('__')[-1] if head.startswith('mcp__') else head


def _bash_verb(args):
    """The command word that names a Bash action's intent: walk past shell
    prefixes (quoted-cd, sleep) and wrapper words ('./dev python3 -m pytest'
    → 'pytest'); path-shaped commands reduce to their basename. Bounded —
    inspects at most the first 6 tokens; falls back to the first token."""
    args = _SHELL_PREFIX_RE.sub('', args.strip())
    tokens = args.split()
    if not tokens:
        return ''
    fallback = tokens[0].split('/')[-1] or tokens[0]
    take_next = False
    for tok in tokens[:6]:
        if take_next:
            return tok.split('/')[-1]
        base = tok.split('/')[-1]
        if tok == '-m':
            take_next = True            # `python -m pytest` → pytest
            continue
        if tok.startswith('-') or tok.startswith('(') or '=' in tok:
            continue
        if base in _VERB_WRAPPERS:
            continue
        return base or fallback
    return fallback


def _extract_targets(summary):
    """All file-ish tokens in the summary (deduped downstream by the union).
    URL innards rejected: a match preceded by '//' or ':' is protocol
    territory, not a filesystem path."""
    out = []
    for m in _TARGET_RE.finditer(summary):
        s = m.start()
        if summary[max(0, s - 2):s] in ('//', ':/') or \
                summary[max(0, s - 1):s] == ':':
            continue
        out.append(_squeeze_paths(m.group(0)))
    return tuple(out)


def _commit_subject(lines):
    """The commit subject from a `git commit` action — the densest
    what-did-this-turn-do string in the stream. Heredoc form: the first
    non-empty body line; inline form: the head of the -m string."""
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip().strip('"\'')
        if s and not s.startswith(('$(', 'EOF', '<<')):
            return s
    m = re.search(r'-m\s+["\']([^"\']+)', lines[0])
    return m.group(1).strip() if m else ''


def _script_intent(lines):
    """A script body's stated intent: its first '#' comment (skipping
    shebangs and editor cookies), else its first non-empty code line — the
    evidence line ('from servers.db_backup import backup_before_destructive'
    tells the reader what was verified)."""
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip()
        if s.startswith('#') and not s.startswith('#!') and '-*-' not in s:
            return s.lstrip('# ').strip()
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip()
        if s and not s.startswith('#'):    # rejected comments don't fall back
            return s
    return ''


def _label(tool, first, lines):
    """One line per action. Every multi-line trim is marked with ' …'.
    Intent harvest (the '·' segment) fires only when it can be attributed
    honestly: a commit subject for `git commit` actions; a script body's
    comment/first line only when the FIRST command opens the script (a
    compound whose heredoc comes later would mis-attribute the comment to
    the leading command)."""
    intent = ''
    if len(lines) > 1:
        if tool == 'Bash' and 'git commit' in first:
            intent = _commit_subject(lines)
        elif _SCRIPT_OPENER_RE.search(first):
            intent = _script_intent(lines)
    label = ('%s · %s' % (first, _one_line(intent))) if intent else first
    if len(lines) > 1:
        label += ' …'
    return label


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
        return _Action(summary, stub.split(':', 1)[0], '', stub, (), False)

    lines = summary.split('\n')
    first = _one_line(lines[0])
    head, sep, args = first.partition(': ')
    tool = _short_tool(head) if sep else 'tool'
    sub = _bash_verb(args) if tool == 'Bash' else ''
    label = _cap(_squeeze_paths(_label(tool, first, lines)))
    if not label.strip():
        label = '%s (no cue)' % (tool or 'tool')
    # Writes never roll up: explicit write tools, git write-verbs, and
    # scripts whose intent was harvested (they are this repo's file editors).
    protected = (tool in WRITE_ACTION_TOOLS
                 or (tool == 'Bash' and sub in GIT_WRITE_VERBS)
                 or ' · ' in label)
    return _Action(summary, tool, sub, label, _extract_targets(summary),
                   protected)


def _dedup(actions):
    """Exact-repeat dedup keyed on the RAW summary — never the rendered
    label (a rendered label is lossy: squeezed paths, trimmed bodies, the
    180-char cap; folding on it would claim two different actions were the
    same). '×N' therefore always means the identical recorded action."""
    kept, by_raw = [], {}
    for a in actions:
        prior = by_raw.get(a.raw)
        if prior is not None:
            prior.count += a.count
        else:
            by_raw[a.raw] = a
            kept.append(a)
    return kept


def _rollup_line(mid):
    """The accounting line for the unrendered middle. Every internal cap
    marks itself ('+k more') — this line's entire job is auditability."""
    total = sum(a.count for a in mid)
    tools, subs = Counter(), Counter()
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
    top_tools = tools.most_common(ROLLUP_TOOLS_CAP)
    for tool, cnt in top_tools:
        part = '%s ×%d' % (tool, cnt)
        if tool == 'Bash' and subs:
            top_subs = subs.most_common(ROLLUP_SUBS_CAP)
            sub_txt = ', '.join('%s ×%d' % (s, c) for s, c in top_subs)
            if len(subs) > len(top_subs):
                sub_txt += ', +%d more' % (len(subs) - len(top_subs))
            part += ' (%s)' % sub_txt
        parts.append(part)
    if len(tools) > len(top_tools):
        parts.append('+%d more tools' % (len(tools) - len(top_tools)))
    line = '(%d more actions, not shown: %s' % (total, ', '.join(parts))
    if targets:
        shown = targets[:ROLLUP_TARGET_CAP]
        line += ' — touched: %s' % ', '.join(shown)
        if len(targets) > len(shown):
            line += ', +%d more' % (len(targets) - len(shown))
    return line + ')'


def _render(a):
    return a.label + (' ×%d' % a.count if a.count > 1 else '')


def condense_actions(episodes, is_tail=False):
    """Episodes of one turn → the lines its <actions> element renders.
    `is_tail`: the newest turn — the encoder's actual working material —
    gets the larger budget; older unencoded turns the smaller."""
    actions = [a for a in (parse_action(e) for e in episodes) if a]

    # The closing actions are the turn's outcome: split them off BEFORE
    # dedup so a final action that repeats an earlier one can never be
    # folded forward out of its outcome slot.
    closing = actions[-ACTIONS_KEEP_LAST:] if len(actions) > ACTIONS_KEEP_LAST \
        else actions
    body = actions[:-len(closing)] if closing is not actions else []
    body = _dedup(body)

    budget = ACTIONS_BUDGET_TAIL if is_tail else ACTIONS_BUDGET
    # Soft edge: an accounting line for one or two actions costs more than
    # it saves — only condense when the middle is worth a line.
    if len(body) + len(closing) <= budget + ACTIONS_BUDGET_SOFT_EDGE:
        return [_render(a) for a in body] + [_render(a) for a in closing]

    # Writes render regardless of budget; the budget's head slots go to the
    # leading regular actions; everything else rolls into the accounting
    # line. Rendered body lines keep their original relative order.
    head_slots = max(0, budget - len(closing))
    kept, mid, regular_kept = [], [], 0
    for a in body:
        if a.protected:
            kept.append(a)
        elif regular_kept < head_slots:
            kept.append(a)
            regular_kept += 1
        else:
            mid.append(a)
    out = [_render(a) for a in kept]
    if mid:
        out.append(_rollup_line(mid))
    return out + [_render(a) for a in closing]
