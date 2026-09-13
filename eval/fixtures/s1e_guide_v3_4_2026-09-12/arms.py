"""Freeze/load the V3.4 candidate and its attribution arm without a live brain.

Parent: frozen V3.3 (template, gist, closure, generated tail, settings). V3.4
revises template and strategy (author.py). Tools are a recorded choice at
freeze time, because the V3.x eval arms have been carrying the 2026-09-08
generic description candidate while the branch's live encoder schemas
(brain_mcp.TOOLS via encode._get_tool_schemas — the deploy path) carry the
contract's own field descriptions; shapes are identical, descriptions differ.

  v3_4_titles      — V3.4 prompt + the chosen tools
  v3_3_live_tools  — V3.3 prompt + live tools (attributes prompt vs tools)

Freeze once, after author + static checks: `./dev python3 arms.py --freeze --tools live|parent`.
"""
import argparse, difflib, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_3_2026-09-11'
CHALLENGE = ROOT / 'docs/challenges/semantic-fidelity.md'
sys.path.insert(0, str(ROOT))


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def parent_arm():
    spec = importlib.util.spec_from_file_location('_v33_frozen_arms', PARENT / 'arms.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_arm('v3_3_titles')


def live_tools():
    from servers.scales.s1 import encode
    return encode._get_tool_schemas()


def build(tools_choice):
    parent = parent_arm()
    before = (PARENT / 'template.md').read_text()
    old_strategy = (PARENT / 'strategy.md').read_text().rstrip()
    parts = {name: (HERE / name).read_text() for name in ('template.md', 'gist.md', 'strategy.md', 'closure.md')}
    if not parent['system_prompt'].startswith(before):
        raise ValueError('Parent template no longer matches its assembled system')
    suffix = parent['system_prompt'][len(before):]
    if suffix.count(old_strategy) != 1:
        raise ValueError('Parent strategy anchor is not unique')
    if parts['closure.md'] != (PARENT / 'closure.md').read_text() or parts['gist.md'] != (PARENT / 'gist.md').read_text():
        raise ValueError('V3.4 keeps the parent gist and finishing contract')
    suffix = suffix.replace(old_strategy, parts['strategy.md'].rstrip())
    tools = live_tools() if tools_choice == 'live' else parent['tools']
    def strip(o):
        if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k != 'description'}
        if isinstance(o, list): return [strip(x) for x in o]
        return o
    if json.dumps(strip(tools), sort_keys=True) != json.dumps(strip(parent['tools']), sort_keys=True):
        raise ValueError('Tool shapes differ from the parent — descriptions only may change')
    v34 = {key: parent[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    v34.update(system_prompt=parts['template.md'] + suffix, tools=tools)
    v34.update(arm_id='v3_4_titles', arm_sha256=digest({k: v34[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}))
    attrib = {key: parent[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    attrib.update(tools=live_tools())
    attrib.update(arm_id='v3_3_live_tools', arm_sha256=digest({k: attrib[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}))
    if not v34['system_prompt'].endswith(parts['closure.md'].rstrip()):
        raise ValueError('Finishing contract is no longer last')
    if v34['settings']['model'] != 'claude-sonnet-4-6':
        raise ValueError('This comparison remains on Sonnet 4.6')
    return parent, v34, attrib, parts


def freeze(tools_choice):
    if (HERE / 'manifest.json').exists():
        raise FileExistsError('V3.4 is already frozen')
    parent, v34, attrib, parts = build(tools_choice)
    for name in parts:
        (HERE / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT / name).read_text().splitlines(True), parts[name].splitlines(True), fromfile='v3.3/' + name, tofile='v3.4/' + name)))
    (HERE / 'v3_4_titles.json').write_text(json.dumps(v34, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'v3_3_live_tools.json').write_text(json.dumps(attrib, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'tools_live.json').write_text(json.dumps(live_tools(), indent=2, ensure_ascii=False) + '\n')
    (HERE / 'CHALLENGES.md').write_text(CHALLENGE.read_text())
    files = [Path(__file__), HERE / 'author.py', HERE / 'author_log.json', HERE / 'static_checks.py', HERE / 'transfer_split.json',
             HERE / 'v3_4_titles.json', HERE / 'v3_3_live_tools.json', HERE / 'tools_live.json', HERE / 'CHALLENGES.md']
    files += [HERE / name for name in parts] + [HERE / (name + '.diff') for name in parts]
    manifest = {'status': 'frozen_after_author_and_static_checks_before_model_calls',
                'parent_arm': parent['arm_sha256'], 'candidate_arm': v34['arm_sha256'], 'attribution_arm': attrib['arm_sha256'],
                'tools_choice': tools_choice, 'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
                'scope': 'eval-only prompt candidate; tools = ' + ('live branch encoder schemas (deploy path)' if tools_choice == 'live' else 'the parent arm\'s 2026-09-08 candidate descriptions'),
                'development_sources_already_seen': ['creative_design', 'eace081b', 'b6019101', '2133c1b5', '80ec1f4f', '2ebe6c92', 'conv_005_emotions'],
                'transfer_split_recorded_before_authoring': True, 'generalization_claim': 'none until the fresh transfer cell reports'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'v3_4_titles': v34['arm_sha256'], 'v3_3_live_tools': attrib['arm_sha256'], 'tools': tools_choice,
                      'system_chars': len(v34['system_prompt']), 'tools_chars': len(json.dumps(v34['tools'], ensure_ascii=False))}, indent=2))


def load_arm(name):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen V3.4 artifact changed: ' + relative)
    parent = parent_arm()
    if parent['arm_sha256'] != manifest['parent_arm']:
        raise ValueError('Parent arm changed')
    if name == 'v3_3_titles':
        return parent
    if name not in ('v3_4_titles', 'v3_3_live_tools'):
        raise ValueError('Unknown arm: ' + name)
    arm = json.loads((HERE / (name + '.json')).read_text())
    identity = {key: arm[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    expected = manifest['candidate_arm'] if name == 'v3_4_titles' else manifest['attribution_arm']
    if digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != expected:
        raise ValueError('Arm identity mismatch: ' + name)
    return arm


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--build-check', action='store_true'); ap.add_argument('--tools', choices=('live', 'parent'), default='live')
    a = ap.parse_args()
    if a.freeze: freeze(a.tools)
    elif a.build_check:
        parent, v34, attrib, parts = build(a.tools)
        print(json.dumps({'parent': parent['arm_sha256'][:12], 'v3_4': v34['arm_sha256'][:12], 'v3_3_live_tools': attrib['arm_sha256'][:12], 'tools': a.tools,
                          'system_chars': len(v34['system_prompt']), 'parent_system_chars': len(parent['system_prompt']), 'tools_chars': len(json.dumps(v34['tools'], ensure_ascii=False)), 'parent_tools_chars': len(json.dumps(parent['tools'], ensure_ascii=False))}, indent=2))
    else:
        for name in ('v3_3_titles', 'v3_3_live_tools', 'v3_4_titles'): print(name, load_arm(name)['arm_sha256'])
