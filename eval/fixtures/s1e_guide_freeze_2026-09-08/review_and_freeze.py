"""Offline authoring review, composition checks and freeze creation.

No LLM client, IsolatedBrain or database is instantiated. Socket connections
are disabled while runtime field/schema renderers are imported and called.
Run with ./dev; default checks without sealing; --seal creates the immutable
review set once. After sealing, the default verifies the existing set too.
"""
import argparse
import ast
from collections import Counter
import copy
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from author_arms import FILES, HERE, PARENTS, PROMPTS, ROOT, generate, sha, with_cues
from frozen_arms import FROZEN, checked_path, digest, load_arm, render_system

sys.path.insert(0, str(ROOT))
GIST_FILE = PROMPTS / 's1e_gist_guide_v2_2026-09-08.md'
GIST_SHA = '54cf9c77132ce3eac680f9703f71a612293b26117c6886048ba9f001634b3063'
SUFFIX_SHA = '4ab165a7a7568af8a2a834058a206dbd17df4e326bc924a13f128ea650d74a13'
TOOLS_SHA = '0481113df84d107888b657679c3ded2296852b26266192dca6802e458dc8c93f'
RUNTIME_FILES = ['servers/scales/s1/encode.py', 'servers/scales/s1/encode_contract.py',
                 'servers/scales/s1/encoding_prompt.py', 'servers/scales/runner.py',
                 'servers/scales/journal.py', 'servers/trace_contract.py',
                 'servers/contract.py', 'servers/brain_mcp.py']


def offline(*args, **kwargs):
    raise RuntimeError('Network disabled during prompt authoring review')


def reject(fn, label):
    try:
        fn()
    except (ValueError, AssertionError):
        return label
    raise AssertionError('Guard accepted invalid input: '+label)


def fences(text):
    blocks = []; opened = None; lines = []
    for line in text.splitlines():
        if line.startswith('```'):
            if opened is None:
                opened = line[3:].strip(); lines = []
            else:
                blocks.append((opened, '\n'.join(lines))); opened = None
        elif opened is not None:
            lines.append(line)
    assert opened is None, 'Unclosed fence'
    return blocks


def json_calls(text):
    return [json.loads(body) for lang, body in fences(text) if lang == 'json']


def parse_teaching_array(code, parameter='operations'):
    """Parse the trusted example's unquoted keys without evaluating code."""
    array = code.split(parameter+':', 1)[1].strip()
    assert array.endswith(')')
    tree = ast.parse(array[:-1].strip(), mode='eval')
    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant({'true': True, 'false': False,
                                                   'null': None}.get(node.id, node.id)), node)
    return ast.literal_eval(Names().visit(tree))


def sweep_ops(text):
    tail = text.split('The sweep, in the next reply', 1)[1]
    return parse_teaching_array(fences(tail)[0][1])


def ladder_ops(text):
    tail = text.split('revise_batch(', 1)[1]
    body = tail.split('```', 1)[0]
    body = '\n'.join(line for line in body.splitlines() if not line.lstrip().startswith('//'))
    return [dict(op='revise', **op) for op in parse_teaching_array(body, 'revisions')]


def catalog(text):
    """Only the explicitly labeled before-state excerpt, not a live catalog."""
    out = {}; node = None
    keys = {'Content': 'content', 'Situation': 'situation', 'Question': 'question',
            'Reasoning': 'reasoning', 'Event Time': 'event_time'}
    for line in text.splitlines():
        match = re.match(r'^\[([^\]]+)\] "(.+)" \(id:([0-9a-f]{8})\)', line)
        if match:
            kind, title, node_id = match.groups()
            node = {'type': kind, 'title': title}; out[node_id] = node
            continue
        if node is not None:
            for prefix, key in keys.items():
                if line.strip().startswith(prefix+':'):
                    node[key] = line.strip()[len(prefix)+1:].strip()
    return out


def apply_revise(node, op, apply_swaps):
    before = copy.deepcopy(node)
    for field, value in op.items():
        if field in {'op', 'node_id', 'reason', 'connect_to'}:
            continue
        if isinstance(value, dict) or (isinstance(value, list) and value and isinstance(value[0], dict)):
            value, error = apply_swaps(node.get(field, ''), value, field)
            assert error is None, error
        node[field] = value
    return before


def inspect_examples(text, validate_field, apply_swaps):
    calls = json_calls(text)
    assert len(calls) == 7
    assert calls[0] == {'node_ids': ['61de80a2'], 'rich': True}
    all_ops = [o for call in calls for o in call.get('operations', [])]
    revised_ops = [op for op in all_ops if op['op'] == 'revise'] + sweep_ops(text)[:-1] + ladder_ops(text)
    for op in all_ops + sweep_ops(text) + ladder_ops(text):
        assert op['op'] in {'remember', 'revise', 'connect'}
        if op['op'] == 'remember':
            assert all(op.get(k) for k in ['type', 'title', 'content', 'situation', 'reasoning'])
        if op['op'] == 'revise':
            assert re.fullmatch('[0-9a-f]{8}', op['node_id']) and op['reason']
        for key, value in op.items():
            if key in {'op', 'reason', 'node_id', 'source_id', 'target_id', 'connect_to'}:
                continue
            ok, error = validate_field(key, value, revising=op['op'] == 'revise')
            assert ok, (key, error)
        for edge in op.get('connect_to', []):
            assert edge['target'] and edge['relation'] and edge['why']
            if op['op'] == 'revise':
                assert re.fullmatch('[0-9a-f]{8}', edge['target'])
        if 'source_refs' in op:
            assert all(re.fullmatch('[0-9a-f]{8}', value) for value in op['source_refs'])
    episode = text.split('### One encoding episode —', 1)[1].split('### A later window —', 1)[0]
    state = catalog(fences(episode)[0][1])
    assert set(state) == {'a6b0139d', '82c41f0b'}
    for op in calls[1]['operations']:
        if op['op'] == 'revise':
            apply_revise(state[op['node_id']], op, apply_swaps)
    assert '17:00–19:00' in state['a6b0139d']['content']
    assert 'simply browse' in state['a6b0139d']['content']
    assert 'not confirmed' not in state['a6b0139d']['reasoning']
    assert state['82c41f0b']['type'] == 'open'
    assert 'unknown' in state['82c41f0b']['content']
    current_thought = next(o for o in calls[1]['operations'] if 'thought' in o)
    later = calls[2]['operations']
    assert len(later) == 2 and later[0]['op'] == 'remember'
    assert set(later[1]) == {'op', 'node_id', 'reason', 'thought'}
    prior = copy.deepcopy(current_thought)
    apply_revise(prior, later[1], apply_swaps)
    assert prior['content'] == current_thought['content']
    assert prior['reasoning'] == current_thought['reasoning']
    assert prior['thought'] != current_thought['thought']
    assert 'tentative' in prior['thought']
    sweep = sweep_ops(text)
    rollout = next(o for o in sweep if o.get('node_id') == 'a45c88f1')
    old = {'content': 'Approved order: auth-rewrite lands first, then api-gateway, then cli. Auth is the dependency the gateway builds on.'}
    apply_revise(old, {'op': 'revise', 'content': rollout['content']}, apply_swaps)
    assert 'Auth is the dependency' not in old['content']
    assert 'no longer a prerequisite' in old['content']
    verdict = next(o for o in sweep if o.get('node_id') == 'b8e05f92')
    assert verdict['type'] == 'finding'
    architecture = next(o for o in ladder_ops(text) if o['node_id'] == '4a9f21c7')
    old_architecture = {'title': 'Surfacer architecture — hook subprocess',
                        'content': 'Surfacer runs as a hook subprocess (2s timeout). Recall calls it per turn; results ride additionalContext...'}
    apply_revise(old_architecture, architecture, apply_swaps)
    assert old_architecture['title'] == 'Surfacer architecture — daemon hook_recall()'
    assert 'Recall calls it per turn; results ride additionalContext...' in old_architecture['content']
    assert 'hook subprocess (2s timeout)' not in old_architecture['content']
    assert 'Situation: When deciding whether to merge the current auth-rewrite branch' in text
    assert 'Done — my branch is deleted (commits recoverable by hash), workspace clean.' in text
    for call in calls:
        siblings = {o['title'] for o in call.get('operations', []) if o['op'] == 'remember'}
        for op in call.get('operations', []):
            for edge in op.get('connect_to', []):
                if not re.fullmatch('[0-9a-f]{8}', edge['target']):
                    assert op['op'] == 'remember' and edge['target'] in siblings
    return {'json_call_blocks': len(calls), 'json_operations': len(all_ops),
            'sweep_operations': len(sweep),
            'revision_ladder_operations': len(ladder_ops(text)),
            'revise_field_counts': dict(sorted(Counter(field for op in revised_ops for field in op
                if field not in {'op', 'node_id', 'reason'}).items())),
            'revise_operations': len(revised_ops),
            'checks': ['contract field values and swap shapes', 'same-batch title targets',
                       'central before-state swaps and retained facts',
                       'thought-only update retains content/reasoning',
                       'sweep repairs both order and dependency claim',
                       'moot merge question leaves open type',
                       'architecture title and content change together']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seal', action='store_true')
    args = ap.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).startswith('9a1727f')
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'claude/sweet-lichterman-ba9854'
    subprocess.run(['git', 'diff', '--exit-code'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--exit-code'], cwd=ROOT, check=True)
    outputs, audit, parents = generate()
    for name, text in outputs.items():
        assert (PROMPTS / FILES[name]).read_text() == text, name+' generation drift'
    assert sha(GIST_FILE.read_text()) == GIST_SHA
    socket.socket.connect = offline
    socket.create_connection = offline
    from servers.scales.s1.encode import _build_system_prompt, _get_tool_schemas
    from servers.scales.s1.encode_contract import ENCODING_AGENT, S1E_INTERACTION_DEFAULT
    from servers.trace_contract import render_prompt_closure
    from servers.contract import validate_field, apply_swaps
    tools = _get_tool_schemas()
    assert digest(json.dumps(tools, sort_keys=True)) == TOOLS_SHA
    assert ENCODING_AGENT['max_tokens'] == 12288 and ENCODING_AGENT['max_rounds'] == 5
    assert S1E_INTERACTION_DEFAULT == {'model': 'claude-sonnet-4-6', 'effort': 'medium'}
    closure = render_prompt_closure()
    reports = {}
    for key in ['v2_revised', 'v3']:
        reports[key] = inspect_examples(outputs[key], validate_field, apply_swaps)
        titled, _ = with_cues(outputs[key])
        assert titled == outputs[key+'_titles']
        stripped = re.sub(r'\n\*Reading cue: [^\n]+\*\n', '', titled)
        assert stripped == outputs[key]
        assert fences(titled) == fences(outputs[key])
    assert json_calls(outputs['v2_revised']) == json_calls(outputs['v3'])
    assert sweep_ops(outputs['v2_revised']) == sweep_ops(outputs['v3'])
    assert ladder_ops(outputs['v2_revised']) == ladder_ops(outputs['v3'])
    for key in ['v2_revised', 'v3']:
        for token in ['Oren', 'Kasia', 'locker B17', 'North Quay', 'seed-library', 'green umbrella']:
            assert token not in outputs[key], 'Diagnostic material leaked: '+token
        assert outputs[key].split('## What I Receive')[0] == parents['v2'].split('## What I Receive')[0]
    settings = {'model': 'claude-sonnet-4-6', 'effort': 'medium', 'max_tokens': 12288,
                'max_rounds': 5, 'lists_preamble': True, 'lived': True}
    arms = {'v2_frozen': (PARENTS['v2'][0], parents['v2'], None)}
    for key in ['v2_revised', 'v3', 'v2_revised_titles', 'v3_titles']:
        arms[key] = (FILES[key], outputs[key], outputs['strategy'] if key.endswith('_titles') else None)
    systems = {}
    for key, (_, template, strategy) in arms.items():
        systems[key] = render_system(template, strategy, _build_system_prompt, closure, SUFFIX_SHA)
        assert systems[key].endswith(closure)
        if strategy:
            assert systems[key].count('## Working strategy') == 1
            assert systems[key].endswith(strategy.rstrip()+'\n\n'+closure)
            assert systems[key].replace(strategy.rstrip()+'\n\n', '') == _build_system_prompt(prompt_instructions=template, lived=True)
    negative = [
        reject(lambda: render_system('x', None, lambda **kw: 'wrong', closure, SUFFIX_SHA), 'template substitution'),
        reject(lambda: render_system(parents['v2'], None, _build_system_prompt, closure, '0'*64), 'suffix drift'),
        reject(lambda: render_system(parents['v2'], None, _build_system_prompt, 'wrong closure', SUFFIX_SHA), 'closure placement'),
        reject(lambda: inspect_examples(outputs['v3'].replace('"old": "booking pending"', '"old": "absent booking value"'), validate_field, apply_swaps), 'unmatched example swap'),
        reject(lambda: inspect_examples(outputs['v3'].replace('"target": "October 17 arrival plan agreed — street route, ramp answer, entrance sign"', '"target": "unknown sibling"'), validate_field, apply_swaps), 'unresolved sibling target'),
        reject(lambda: checked_path(str(GIST_FILE.relative_to(ROOT)), '0'*64), 'frozen input hash mismatch'),
        reject(lambda: checked_path('../outside-review-worktree', '0'*64), 'path outside worktree'),
    ]
    report = {'status': 'passed', 'model_calls': 0, 'database_instances': 0,
              'network': 'socket connections disabled', 'arm_reviews': reports,
              'negative_checks': negative, 'cues_preserve_parent_and_examples': True,
              'shared_json_calls_sweep_and_ladder_equal_between_revised_bases': True,
              'limitations': 'Contract field checks, not a full JSON Schema validator or database dispatch. No behavioral claim.'}
    if args.seal:
        if FROZEN.exists():
            raise SystemExit('Freeze directory already exists; refusing overwrite')
        staging_context = TemporaryDirectory(prefix='.freeze-stage-', dir=HERE)
        stage = Path(staging_context.name)
        files = {}
        def track(path):
            destination = FROZEN / path.relative_to(stage) if path.is_relative_to(stage) else path
            files[str(destination.relative_to(ROOT))] = digest(path.read_bytes())
        for filename, _ in PARENTS.values():
            track(PROMPTS / filename)
        for filename in FILES.values():
            track(PROMPTS / filename)
        track(GIST_FILE)
        for name in ['author_arms.py', 'later_window.md', 'authoring_audit.json', 'semantic_review.md',
                     'frozen_arms.py', 'review_and_freeze.py', 'v2_revised.diff', 'v3.diff',
                     'v2_revised_titles.diff', 'v3_titles.diff']:
            track(HERE / name)
        for name in RUNTIME_FILES:
            track(ROOT / name)
        tools_path = stage / 'tools.json'
        tools_path.write_text(json.dumps(tools, indent=2)+'\n'); track(tools_path)
        review_path = stage / 'offline_review.json'
        review_path.write_text(json.dumps(report, indent=2)+'\n'); track(review_path)
        adapter_hash = digest((HERE / 'frozen_arms.py').read_bytes())
        manifest = {'format_version': 1, 'status': 'frozen_for_review',
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'gist': str(GIST_FILE.relative_to(ROOT)), 'tools': str((FROZEN / 'tools.json').relative_to(ROOT)),
                    'shared_suffix_sha256': SUFFIX_SHA, 'assembly_adapter_sha256': adapter_hash,
                    'settings': settings, 'arms': {}, 'files': files,
                    'eval_status': 'not run; authoring review only',
                    'cache_rule': 'Include arm_sha256, not only template fingerprint; no corpus runner is wired by this freeze.'}
        for key, (filename, template, strategy) in arms.items():
            system_path = stage / (key+'.system.md')
            system_path.write_text(systems[key]); track(system_path)
            identity = {'system_sha256': digest(systems[key]), 'gist_sha256': GIST_SHA,
                        'tools_sha256': TOOLS_SHA, 'settings': settings,
                        'assembly_adapter_sha256': adapter_hash}
            manifest['arms'][key] = {'template': str((PROMPTS / filename).relative_to(ROOT)),
                                    'template_chars': len(template),
                                    'strategy': str((PROMPTS / FILES['strategy']).relative_to(ROOT)) if strategy else None,
                                    'strategy_chars': len(strategy or ''), 'system': str((FROZEN / system_path.name).relative_to(ROOT)),
                                    'system_chars': len(systems[key]),
                                    'arm_sha256': digest(json.dumps(identity, sort_keys=True))}
        encoded = json.dumps(manifest, indent=2)+'\n'
        (stage / 'manifest.json').write_text(encoded)
        (stage / 'manifest.sha256').write_text(digest(encoded)+'\n')
        for relative, expected in files.items():
            path = ROOT / relative
            path = stage / path.relative_to(FROZEN) if path.is_relative_to(FROZEN) else path
            assert digest(path.read_bytes()) == expected, relative
        assert digest((stage / 'manifest.json').read_bytes()) == (stage / 'manifest.sha256').read_text().strip()
        # Only a complete, verified directory acquires the frozen name.
        stage.rename(FROZEN)
        staging_context.cleanup()
    if FROZEN.exists():
        for key in arms:
            loaded = load_arm(key)
            assert loaded['system_prompt'] == systems[key]
        assert json.loads((FROZEN / 'offline_review.json').read_text()) == report
    print(json.dumps(report, indent=2))
    for key in arms:
        print(key, 'system chars', len(systems[key]))


if __name__ == '__main__':
    main()
