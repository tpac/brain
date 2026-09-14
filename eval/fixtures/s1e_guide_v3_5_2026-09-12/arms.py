"""Freeze/load the V3.5 candidate without a live brain.

Parent: frozen V3.4 (template, gist, strategy, closure, generated tail, settings, LIVE tools).
V3.5 revises template and gist (author.py); strategy and closure are the parent's. The tools
are the parent's live schemas — the same descriptions the deploy path carries — so the only
axis between v3_4_titles and v3_5_titles is the prompt (template + gist).

  v3_5_titles — V3.5 template + V3.5 gist + parent strategy/closure + parent (live) tools

Freeze once, after author + static checks: `./dev python3 arms.py --freeze`.
"""
import argparse, difflib, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_4_2026-09-12'
CHALLENGE = ROOT / 'docs/challenges/semantic-fidelity.md'
sys.path.insert(0, str(ROOT))


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def parent_arm():
    spec = importlib.util.spec_from_file_location('_v34_frozen_arms', PARENT / 'arms.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_arm('v3_4_titles')


def build():
    parent = parent_arm()
    before = (PARENT / 'template.md').read_text()
    old_strategy = (PARENT / 'strategy.md').read_text().rstrip()
    parts = {name: (HERE / name).read_text() for name in ('template.md', 'gist.md', 'strategy.md', 'closure.md')}
    if not parent['system_prompt'].startswith(before):
        raise ValueError('Parent template no longer matches its assembled system')
    suffix = parent['system_prompt'][len(before):]
    if suffix.count(old_strategy) != 1:
        raise ValueError('Parent strategy anchor is not unique')
    if parts['closure.md'] != (PARENT / 'closure.md').read_text():
        raise ValueError('V3.5 keeps the parent finishing contract')
    if parent['gist'] != (PARENT / 'gist.md').read_text():
        raise ValueError('Parent gist no longer matches its frozen part')
    suffix = suffix.replace(old_strategy, parts['strategy.md'].rstrip())
    if parent['tools'] != json.loads((PARENT / 'tools_live.json').read_text()):
        raise ValueError('Parent tools are not the recorded live schemas')
    v35 = {key: parent[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    v35.update(system_prompt=parts['template.md'] + suffix, gist=parts['gist.md'])
    v35.update(arm_id='v3_5_titles', arm_sha256=digest({k: v35[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}))
    if not v35['system_prompt'].endswith(parts['closure.md'].rstrip()):
        raise ValueError('Finishing contract is no longer last')
    if v35['settings']['model'] != 'claude-sonnet-4-6':
        raise ValueError('This comparison remains on Sonnet 4.6')
    return parent, v35, parts


def freeze():
    if (HERE / 'manifest.json').exists():
        raise FileExistsError('V3.5 is already frozen')
    parent, v35, parts = build()
    for name in parts:
        (HERE / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT / name).read_text().splitlines(True), parts[name].splitlines(True), fromfile='v3.4/' + name, tofile='v3.5/' + name)))
    (HERE / 'v3_5_titles.json').write_text(json.dumps(v35, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'CHALLENGES.md').write_text(CHALLENGE.read_text())
    files = [Path(__file__), HERE / 'author.py', HERE / 'author_log.json', HERE / 'static_checks.py', HERE / 'transfer_split.json',
             HERE / 'v3_5_titles.json', HERE / 'CHALLENGES.md', HERE / 'READ-AUDIT-V3-4.md', HERE / 'ANALYSIS.md']
    files += [HERE / name for name in parts] + [HERE / (name + '.diff') for name in parts]
    manifest = {'status': 'frozen_after_author_and_static_checks_before_model_calls',
                'parent_arm': parent['arm_sha256'], 'candidate_arm': v35['arm_sha256'], 'tools_choice': 'live (parent tools_live.json)',
                'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
                'scope': 'eval-only prompt candidate; template + gist changed; strategy, closure and tools are the frozen V3.4 parts',
                'development_sources_already_seen': ['creative_design', 'eace081b', 'b6019101', '2133c1b5', '80ec1f4f', '2ebe6c92', 'conv_005_emotions',
                                                     '69fee5aa', 'f685340e', '2b8f3739', 'eac54adc', 'gpt4_1d4ab0c9', 'conv_002_debugging'],
                'transfer_split_recorded_before_authoring': True, 'generalization_claim': 'none until the fresh transfer cell reports'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'v3_5_titles': v35['arm_sha256'], 'parent_v3_4': parent['arm_sha256'], 'system_chars': len(v35['system_prompt']),
                      'gist_chars': len(v35['gist']), 'tools_chars': len(json.dumps(v35['tools'], ensure_ascii=False))}, indent=2))


def load_arm(name):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen V3.5 artifact changed: ' + relative)
    parent = parent_arm()
    if parent['arm_sha256'] != manifest['parent_arm']:
        raise ValueError('Parent arm changed')
    if name == 'v3_4_titles':
        return parent
    if name != 'v3_5_titles':
        raise ValueError('Unknown arm: ' + name)
    arm = json.loads((HERE / 'v3_5_titles.json').read_text())
    identity = {key: arm[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    if digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != manifest['candidate_arm']:
        raise ValueError('Arm identity mismatch: ' + name)
    return arm


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--build-check', action='store_true')
    a = ap.parse_args()
    if a.freeze: freeze()
    elif a.build_check:
        parent, v35, parts = build()
        print(json.dumps({'parent': parent['arm_sha256'][:12], 'v3_5': v35['arm_sha256'][:12], 'system_chars': len(v35['system_prompt']), 'parent_system_chars': len(parent['system_prompt']),
                          'gist_chars': len(v35['gist']), 'parent_gist_chars': len(parent['gist'])}, indent=2))
    else:
        for name in ('v3_4_titles', 'v3_5_titles'): print(name, load_arm(name)['arm_sha256'])
