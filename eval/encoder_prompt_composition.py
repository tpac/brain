"""Where an S1E prompt's characters actually go — catalog vs timeline vs actions.

Sibling to the other three encoder_prompt_* tools, and the odd one out: those
read the TEMPLATE (probe interviews it, diff compares probe runs, encoding_prompt_eval
A/Bs two versions by behaviour). This one reads the ASSEMBLED PAYLOAD — the exact
bytes a live run sent — and reports the split. Use it to decide what to cut before
touching anything, and to measure the cut afterwards.

INPUT is the payload recorder's own output: every S1E run writes its prompt under
`$BRAIN_DB_DIR/payloads/<date>/<chain>/000-prompt.md`, and the run's
`encoding_prompt` O trace carries that pointer as its ref_id:

    query_traces(ref_type='encoding_prompt', hours=96, limit=5)  →  ev['ref_id']

Counts only — no prompt content is printed, so output is safe to paste into a
report or hand to another install's operator.

WHAT IT FOUND (2026-08-16, three live payloads, 84K / 277K / 105K chars):
  catalog        63-67% of every prompt      <- the weight, and it grows per run
  actions        8.6-27.7%
  per catalog node   4,010-5,044 chars rendered, of which content is only ~22%
So trimming a catalog node's CONTENT saves little; dropping its edges and heavy
corrections is the lever. And on the one run with encoded turns, 89.8% of the
actions block sat on turns a previous run had already read.

Usage:
    ./dev python3 eval/encoder_prompt_composition.py <payload.md> [more.md ...]
"""
import re
import sys

# Read-only shell verbs: an action line whose every segment is one of these
# changed nothing, so it is an observation, not an outcome. `sed -i` writes and
# is deliberately excluded below.
NOISE = {'grep', 'ls', 'cat', 'head', 'tail', 'wc', 'find', 'awk', 'jq',
         'sort', 'uniq', 'diff', 'echo', 'cd', 'file', 'which', 'sed', 'du',
         'ps', 'pwd', 'stat', 'basename', 'dirname', 'tree', 'env'}
READ_TOOLS = ('Read:', 'Grep:', 'Glob:')


def bash_is_readonly(cmd):
    """True when EVERY segment is a noise verb — `cd /x && ./dev pytest` is not
    read-only, and a first-token matcher would call it `cd` and drop it (19% of
    Bash volume hides behind a leading cd)."""
    for seg in re.split(r'&&|\|\||;|\|', cmd):
        seg = seg.strip()
        if not seg:
            continue
        verb = seg.split()[0] if seg.split() else ''
        if verb == 'sed' and ' -i' in seg:
            return False
        if verb not in NOISE:
            return False
    return True


def classify(line):
    s = line.strip()
    if s.startswith('mcp__plugin_brain_brain__'):
        return 'brain'          # represented in <provenance> as well
    if s.startswith('mcp__'):
        return 'other-mcp'
    if s.startswith(READ_TOOLS):
        return 'readonly'
    if s.startswith('Bash: '):
        return 'readonly' if bash_is_readonly(s[6:]) else 'keep'
    return 'keep'


def analyse(path):
    text = open(path).read()
    first_turn = text.find('<turn ')
    # Everything before the first turn is the catalog block (plus a 3-line
    # preamble and the scout legend — both negligible against a 55-176K catalog).
    catalog = first_turn if first_turn > 0 else 0

    turns = re.findall(r'<turn n="(\d+)"([^>]*)>(.*?)</turn>', text, re.S)
    s = {'turns': len(turns), 'encoded': 0, 'actions_chars': 0,
         'actions_on_encoded': 0, 'lines': 0,
         'brain': 0, 'other-mcp': 0, 'readonly': 0, 'keep': 0,
         'readonly_live': 0, 'brain_live': 0}
    for _n, attrs, body in turns:
        is_enc = 'encoded="true"' in attrs
        s['encoded'] += is_enc
        m = re.search(r'<actions>(.*?)</actions>', body, re.S)
        if not m:
            continue
        s['actions_chars'] += len(m.group(0))
        if is_enc:
            s['actions_on_encoded'] += len(m.group(0))
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            kind = classify(line)
            s['lines'] += 1
            s[kind] += len(line) + 1
            # "live" = on a turn no previous run has read; dropping the encoded
            # turns' actions already covers the rest, so don't double-count.
            if not is_enc and kind == 'readonly':
                s['readonly_live'] += len(line) + 1
            if not is_enc and kind == 'brain':
                s['brain_live'] += len(line) + 1
    return len(text), catalog, s


def pct(part, whole):
    return '%5.1f%%' % (100.0 * part / whole) if whole else '    - '


def main(paths):
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 1
    for path in paths:
        total, catalog, s = analyse(path)
        saved = s['actions_on_encoded'] + s['readonly_live'] + s['brain_live']
        print('\n%s' % path)
        print('  total %d chars | catalog %d (%s) | actions %d (%s)'
              % (total, catalog, pct(catalog, total),
                 s['actions_chars'], pct(s['actions_chars'], total)))
        print('  turns %d (encoded=true %d) | action lines %d'
              % (s['turns'], s['encoded'], s['lines']))
        print('  action chars: brain %d | other-mcp %d | readonly %d | keep %d'
              % (s['brain'], s['other-mcp'], s['readonly'], s['keep']))
        print('  if dropped: encoded-turn actions %d (%s of actions) | '
              'readonly %d | brain-ops %d'
              % (s['actions_on_encoded'],
                 pct(s['actions_on_encoded'], s['actions_chars']),
                 s['readonly_live'], s['brain_live']))
        print('  TOTAL %d = %s of the actions block, %s of the prompt'
              % (saved, pct(saved, s['actions_chars']), pct(saved, total)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
