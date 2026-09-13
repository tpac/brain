"""Static checks over the authored V3.6 carriers (adapted from V3.5) — no model calls.
Run: ./dev python3 static_checks.py [template_full.md gist_full.md | template_layer.md gist_layer.md]

What is checked, and the law each check serves (docs/S1E-CHECKLIST.md):
  1. every ```json fence parses (T7 one sign system)
  2. every op name is a BATCH_OP_SPECS op; every connect_to item key is a
     schema property; every why has the tool's 30-char floor (T8, C1, E18)
  3. every swap `old` in a worked op appears in the template BEFORE the op,
     i.e. in a depicted catalog/result the encoder could copy from (E12)
  4. every revise node_id and id-form target appears before the op (A5)
  5. no agent-name literal; `{trace-…}` / `{id-of-…}` placeholders only in
     the identity section (A5, D-12)
  6. sizes against V3.2 and production; hedge census (A8/A9 spread)
  7. A10 census: fields per worked revise op, JSON and pseudo-JSON alike
  8. E15: new edge whys inside the 120–180 guidance band are reported
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from servers.contract import BATCH_OP_SPECS, CONNECT_TO_ITEM_SCHEMA  # noqa: E402

PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_4_2026-09-12'
PROD = ROOT / 'eval/fixtures/s1e_production_comparison_2026-09-11/effective_s1e.json'
HEDGE = re.compile(r"does not establish|do not establish|not establish|does not (prove|measure|validate|define|describe|settle|justify)|could also|may be |might |without (establishing|inventing|proving)|is established|not evidence")
CT_KEYS = set(CONNECT_TO_ITEM_SCHEMA['properties']) | {'old', 'new'}
failures, notes = [], []


def fail(msg):
    failures.append(msg)


def fences(text, lang):
    out = []
    for m in re.finditer(r'```' + lang + r'\n(.*?)\n```', text, re.S):
        out.append((m.start(), m.group(1)))
    return out


def walk_ops(obj, pos, sink):
    if isinstance(obj, dict):
        if 'op' in obj:
            sink.append((pos, obj))
        for v in obj.values():
            walk_ops(v, pos, sink)
    elif isinstance(obj, list):
        for v in obj:
            walk_ops(v, pos, sink)


def swaps_in(value):
    if isinstance(value, dict) and set(value) == {'old', 'new'}:
        return [value]
    if isinstance(value, list):
        return [s for s in value if isinstance(s, dict) and set(s) == {'old', 'new'}]
    return []


def match_brace(text, start):
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def main():
    tpl_name = sys.argv[1] if len(sys.argv) > 1 else 'template_full.md'
    gist_name = sys.argv[2] if len(sys.argv) > 2 else 'gist_full.md'
    t = (HERE / tpl_name).read_text()
    g = (HERE / gist_name).read_text()
    s = (HERE / 'strategy.md').read_text()
    old_t = (PARENT / 'template.md').read_text()
    prod = json.load(open(PROD))['template']
    ops_json = []
    # 1 + 2 + 3 + 4 on real JSON fences
    for pos, body in fences(t, 'json'):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            fail(f'json fence at {pos} does not parse: {e}')
            continue
        walk_ops(obj, pos, ops_json)
    for pos, op in ops_json:
        name = op.get('op')
        if name not in BATCH_OP_SPECS:
            fail(f'op {name!r} at {pos} not in BATCH_OP_SPECS')
        before = t[:pos]
        if name == 'revise':
            nid = op.get('node_id', '')
            if not re.fullmatch(r'[0-9a-f]{8}', nid):
                fail(f'revise at {pos}: node_id {nid!r} not 8-hex')
            elif nid not in before:
                fail(f'revise at {pos}: node_id {nid} not depicted before the op (A5/E12)')
            for field, value in op.items():
                for sw in swaps_in(value):
                    if before.count(sw['old']) < 1:
                        fail(f'revise {nid} field {field}: swap old {sw["old"][:60]!r} not visible before the op (E12)')
        for item in op.get('connect_to', []) or []:
            bad = set(item) - CT_KEYS - {'relations'}
            if bad:
                fail(f'connect_to item keys {bad} at {pos} not in CONNECT_TO_ITEM_SCHEMA')
            tgt = item.get('target') or item.get('title') or ''
            if re.fullmatch(r'[0-9a-f]{8}', tgt) and tgt not in before:
                fail(f'connect_to target id {tgt} at {pos} not depicted before the op (A5)')
            why = item.get('why')
            if isinstance(why, str) and len(why) < 30:
                fail(f'why under 30 chars at {pos}: {why!r}')
            if isinstance(why, dict) and before.count(why.get('old', '\x00')) < 1:
                fail(f'why swap old not depicted before op at {pos}')
    # pseudo-JSON revise blocks (ladder, sweep, canonical): census + old visibility
    revise_census = []
    for m in re.finditer(r'\{\s*"?op"?:\s*"revise",\s*"?node_id"?:\s*"([0-9a-f]{8})"', t):
        end = match_brace(t, m.start())
        span = t[m.start():end + 1]
        nid = m.group(1)
        keys = re.findall(r'(?:^|[\s{,])"?([a-z_]+)"?\s*:', span)
        keys = [k for k in dict.fromkeys(keys) if k not in ('op', 'node_id', 'reason', 'old', 'new', 'target', 'relation', 'why', 'title_', )]
        # `title` inside connect_to items is an alias key; count only top-level-ish occurrences
        revise_census.append((nid, keys, m.start()))
        for old in re.findall(r'\bold:\s*"((?:[^"\\]|\\.)*)"', span):
            if t[:m.start()].count(old) < 1 and t.count(old) < 2:
                fail(f'pseudo-JSON revise {nid}: swap old {old[:60]!r} never depicted (E12)')
    # 5 placeholders and agent name
    if 'Anchor' in t:
        fail('agent-name literal "Anchor" present (D-12)')
    ident = t.index('## Identity-bearing examples')
    for ph in re.finditer(r'\{(trace|id-of)-[a-z-]+\}', t):
        if ph.start() < ident and 'Examples ground ids in their own excerpts' not in t[ph.start()-400:ph.start()+200]:
            fail(f'placeholder {ph.group()} outside identity section at {ph.start()}')
    # 6 sizes and hedges
    notes.append(f'{tpl_name}: template {len(old_t)} -> {len(t)} ({len(t)-len(old_t):+d}); {gist_name}: gist {len((PARENT/"gist.md").read_text())} -> {len(g)}; strategy -> {len(s)}; production template {len(prod)}')
    notes.append(f'hedge clauses: V3.3 {len(HEDGE.findall(old_t))} -> V3.4 {len(HEDGE.findall(t))} (production {len(HEDGE.findall(prod))})')
    # 7 A10 census
    notes.append('A10 worked revise ops (node: fields touched):')
    for nid, keys, pos in revise_census:
        notes.append(f'   {nid}: {", ".join(keys)}')
    # 8 E15 — whys in the new second window
    if '### A thin window' in t:
        sw = t[t.index('### A thin window'):t.index('### Detail and meaning')]
        for m in re.finditer(r'"why":\s*"((?:[^"\\]|\\.)*)"', sw):
            n = len(m.group(1))
            notes.append(f'   thin-window why {n} chars {"(in 120-180 band)" if 120 <= n <= 180 else "(outside band)"}: {m.group(1)[:70]}…')
    for line in notes:
        print(line)
    if failures:
        print('\nFAILURES:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('\nSTATIC CHECKS PASSED')


if __name__ == '__main__':
    main()
