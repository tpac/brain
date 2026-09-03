"""The prompt, the gist, and the tool layer teach ONE vocabulary — the contract's.

Drift guardrail for changes to the batch-op contract (docs/REVISE-SHAPE-SPEC.md
§5). The tool layer outranks the prompt by position (E10: the field summary and
tool descriptions are injected last), so a name that lives in the contract but
is silent on a teaching surface — or retired from the contract but still
taught — makes the prompt lose without anyone noticing. Each assertion here
reads the contract as the source and checks the surfaces against it.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.contract import (BATCH_OP_SPECS, CONNECT_TO_ITEM_SCHEMA,  # noqa: E402
                              RETIRED_OP_FIELDS, generate_field_summary,
                              get_writable_fields)
from servers.scales.s1.encode_contract import ENCODER_GIST  # noqa: E402
from servers.scales.s1.encoding_prompt import SYSTEM_PROMPT  # noqa: E402
from servers import brain_mcp  # noqa: E402

TOOLS = {t['name']: t for t in brain_mcp.TOOLS}
ENCODER_TOOL_NAMES = {'remember_batch', 'revise_batch', 'brain_batch',
                      'connect_batch', 'recall_batch', 'get_nodes'}
# Relation verbs the gist names on purpose — relations are open vocabulary,
# so they are allowlisted here rather than derived. Extend when the gist does.
GIST_RELATIONS = {'resolves', 'partially_resolves'}


def _fenced(text):
    out, cur, inside = [], [], False
    for line in text.split('\n'):
        if line.startswith('```'):
            if inside:
                out.append('\n'.join(cur))
                cur = []
            inside = not inside
            continue
        if inside:
            cur.append(line)
    return out


def _connect_to_arrays(block):
    """Text of every `connect_to: [ … ]` array in a fenced example."""
    arrays = []
    for m in re.finditer(r'connect_to:\s*\[', block):
        depth, i = 0, m.end() - 1
        while i < len(block):
            if block[i] == '[':
                depth += 1
            elif block[i] == ']':
                depth -= 1
                if depth == 0:
                    arrays.append(block[m.end():i])
                    break
            i += 1
    return arrays


def _teaching_surfaces():
    """(name, text) for every surface that teaches the revise vocabulary."""
    revise = TOOLS['revise']
    revise_batch = TOOLS['revise_batch']
    return [
        ('s1e prompt', SYSTEM_PROMPT),
        ('gist', ENCODER_GIST),
        ('field summary', generate_field_summary()),
        ('MCP revise', revise['description'] + ' '
         + ' '.join(str(p) for p in revise['inputSchema']['properties'].values())),
        ('MCP revise_batch', revise_batch['description'] + ' '
         + str(revise_batch['inputSchema']['properties'].get('revisions', ''))),
        ('brain_batch revise spec', BATCH_OP_SPECS['revise']['description']),
    ]


def test_every_example_op_is_a_batch_op():
    ops = set()
    for block in _fenced(SYSTEM_PROMPT):
        ops.update(re.findall(r'\bop:\s*"([a-z_]+)"', block))
    assert ops, 'no {op: "…"} examples found in the prompt — the regex or the prompt moved'
    unknown = ops - set(BATCH_OP_SPECS)
    assert not unknown, 'prompt examples use ops the contract has no spec for: %s' % sorted(unknown)


def test_every_connect_to_key_in_examples_is_in_the_item_schema():
    allowed = set(CONNECT_TO_ITEM_SCHEMA['properties'])
    allowed |= set(CONNECT_TO_ITEM_SCHEMA['properties']['relations']['items']['properties'])
    seen, unknown = set(), set()
    for block in _fenced(SYSTEM_PROMPT):
        for arr in _connect_to_arrays(block):
            for key in re.findall(r'[{,]\s*([a-z_]+)\s*:', arr):
                seen.add(key)
                if key not in allowed:
                    unknown.add(key)
    assert seen, 'no connect_to arrays found in the prompt examples'
    assert not unknown, 'connect_to items in the prompt use keys the schema lacks: %s' % sorted(unknown)


def test_every_revise_spec_field_is_taught_on_every_surface():
    """A field the contract offers on revise but a surface never names is the
    E11 defect (present, not taught) — and the position asymmetry means the
    silent surface wins."""
    fields = [f for f in BATCH_OP_SPECS['revise']['properties'] if f not in ('node_id', 'reason')]
    assert fields, 'revise spec has no teachable fields — did the contract move?'
    silent = [(name, f) for f in fields for name, text in _teaching_surfaces()
              if f not in text]
    assert not silent, 'revise fields the contract offers but a surface never names: %s' % silent


def test_retired_fields_are_taught_nowhere():
    stale = [(name, f) for f in RETIRED_OP_FIELDS for name, text in _teaching_surfaces()
             if re.search(r'\b%s\b' % re.escape(f), text)]
    assert not stale, 'retired op fields still taught: %s' % stale


def test_gist_uses_only_contract_vocabulary():
    known = set(BATCH_OP_SPECS) | ENCODER_TOOL_NAMES | set(get_writable_fields())
    known |= {prop for spec in BATCH_OP_SPECS.values() for prop in spec['properties']}
    known |= set(CONNECT_TO_ITEM_SCHEMA['properties']) | {'old', 'new'} | GIST_RELATIONS
    known |= {'open'}  # the type the gist names (types are open vocabulary; this one is load-bearing)
    tokens = {t for t in re.findall(r'`([a-z_]+)`', ENCODER_GIST)}
    assert tokens, 'gist has no backticked identifiers — did the gist move?'
    unknown = tokens - known
    assert not unknown, 'gist names identifiers the contract does not have: %s' % sorted(unknown)


def test_encoder_tool_names_in_prompt_exist():
    names = set(re.findall(r'`([a-z_]+_batch|get_nodes)`', SYSTEM_PROMPT))
    assert names, 'prompt names no batch tools'
    unknown = names - set(TOOLS)
    assert not unknown, 'prompt names tools that do not exist: %s' % sorted(unknown)
    not_encoder = names - ENCODER_TOOL_NAMES
    assert not not_encoder, 'prompt names tools the encoder does not hold: %s' % sorted(not_encoder)


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('  PASS  %s' % name)
            except AssertionError as e:
                fails += 1
                print('  FAIL  %s — %s' % (name, e))
    print('\n%s' % ('ALL PASS' if not fails else '%d FAILED' % fails))
    sys.exit(1 if fails else 0)
