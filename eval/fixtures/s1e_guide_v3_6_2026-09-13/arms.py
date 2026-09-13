"""Freeze/load the V3.6 arms — assembled the way the runtime assembles them.

Every V3.6 arm's system prompt is `encode._build_system_prompt(template, lived=True)`
on this branch: the candidate template, then the contract field summary (with the
event_time reword and the encoder_summary hiding), then the arc, review and closure
blocks the runtime injects. What the cell measures is byte for byte what a deploy
of the template would run. The baseline is the frozen V3.4 arm exactly as it was
measured (its own tail). Tools are the branch's live encoder schemas.

  v3_4_titles        — frozen V3.4 (parent fixture), the baseline
  v3_6_layer         — template_layer.md + gist_layer.md: the text pass, runtime tail
  v3_6_full          — template_full.md + gist_full.md: + fact-first `new`, thin window, position
  v3_6_full_v34tail  — v3_6_full with the runtime closure replaced by V3.4's strategy + closure
                       (the with/without strategy test, strategy_subsample corpora only)

Freeze once, after author + static checks: `./dev python3 arms.py --freeze`.
`--build-check` prints sizes, the tool-schema comparison against V3.4's frozen tools,
and the assembly diff of the layer arm against the frozen V3.4 system (the deploy check).
"""
import argparse, difflib, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_4_2026-09-12'
sys.path.insert(0, str(ROOT))

# (template, gist, v34_tail). Paths are fixture-relative; 'PARENT:' reads V3.4's part.
SPECS = {
    # V3.4's own template and gist assembled on today's runtime and tools — what
    # deploying V3.4 unchanged would run. Separates the runtime/tool/contract
    # delta from the text pass in the ladder frozen V3.4 → this → layer → full.
    'v3_4_live': ('PARENT:template.md', 'PARENT:gist.md', False),
    'v3_6_layer': ('template_layer.md', 'gist_layer.md', False),
    'v3_6_full': ('template_full.md', 'gist_full.md', False),
    'v3_6_full_v34tail': ('template_full.md', 'gist_full.md', True),
}


def part(name):
    return (PARENT / name[len('PARENT:'):]).read_text() if name.startswith('PARENT:') else (HERE / name).read_text()
PARTS = ('template_layer.md', 'gist_layer.md', 'template_full.md', 'gist_full.md', 'strategy.md', 'closure.md')


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def parent_arm():
    """The frozen V3.4 arm from its own record. The fixture chain's loaders pin
    runtime files (encode.py, contract.py) that the reconcile with main changed
    on purpose, so the chain refuses; the arm's identity — the digest of its
    system prompt, gist, tools and settings — is checked against the record and
    against V3.4's manifest instead. The baseline re-runs on the reconciled
    runtime, as every arm in this cell does."""
    arm = json.loads((PARENT / 'v3_4_titles.json').read_text())
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    identity = {k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}
    if digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != manifest['candidate_arm']:
        raise ValueError('Frozen V3.4 arm identity mismatch')
    return arm


def runtime_system(template):
    from servers.scales.s1 import encode
    return encode._build_system_prompt(template, lived=True)


def with_v34_tail(system):
    from servers.trace_contract import render_prompt_closure
    closure = render_prompt_closure()
    if system.count(closure) != 1:
        raise ValueError('runtime closure not found exactly once in the assembled system')
    tail = (HERE / 'strategy.md').read_text().rstrip() + '\n\n' + (HERE / 'closure.md').read_text().rstrip()
    return system.replace(closure, tail)


def strip_descriptions(o):
    if isinstance(o, dict): return {k: strip_descriptions(v) for k, v in o.items() if k != 'description'}
    if isinstance(o, list): return [strip_descriptions(x) for x in o]
    return o


def build():
    from servers.scales.s1 import encode
    parent = parent_arm()
    tools = encode._get_tool_schemas()
    if json.dumps(strip_descriptions(tools), sort_keys=True) != json.dumps(strip_descriptions(parent['tools']), sort_keys=True):
        raise ValueError('Tool shapes differ from the frozen V3.4 tools — descriptions only may change')
    if parent['settings']['model'] != 'claude-sonnet-4-6':
        raise ValueError('This comparison remains on Sonnet 4.6')
    arms = {}
    for name, (tpl, gist, tail) in SPECS.items():
        system = runtime_system(part(tpl))
        if tail:
            system = with_v34_tail(system)
        arm = {'system_prompt': system, 'gist': part(gist), 'tools': tools, 'settings': parent['settings']}
        arm.update(arm_id=name, arm_sha256=digest({k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}))
        arms[name] = arm
    return parent, arms, tools


def build_check():
    parent, arms, tools = build()
    out = {'parent_v3_4': parent['arm_sha256'][:12], 'parent_system_chars': len(parent['system_prompt']), 'parent_gist_chars': len(parent['gist'])}
    for name, arm in arms.items():
        out[name] = {'sha': arm['arm_sha256'][:12], 'system_chars': len(arm['system_prompt']), 'gist_chars': len(arm['gist'])}
    tool_desc_diffs = [t['name'] for t, p in zip(tools, parent['tools']) if json.dumps(t, sort_keys=True) != json.dumps(p, sort_keys=True)]
    out['tool_description_changes_vs_v34'] = tool_desc_diffs
    print(json.dumps(out, indent=2))
    print('\n--- assembly diff: frozen V3.4 system -> v3_6_layer system (the deploy check) ---')
    d = list(difflib.unified_diff(parent['system_prompt'].splitlines(True), arms['v3_6_layer']['system_prompt'].splitlines(True),
                                  fromfile='V3.4 frozen', tofile='v3_6_layer (runtime assembly)', n=0))
    print(''.join(d) if len(d) < 200 else ''.join(d[:200]) + f'\n... {len(d) - 200} more lines')


def freeze():
    if (HERE / 'manifest.json').exists():
        raise FileExistsError('V3.6 is already frozen')
    parent, arms, tools = build()
    for name, arm in arms.items():
        (HERE / (name + '.json')).write_text(json.dumps(arm, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'tools_live.json').write_text(json.dumps(tools, indent=2, ensure_ascii=False) + '\n')
    for name in ('template_layer.md', 'template_full.md'):
        (HERE / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT / 'template.md').read_text().splitlines(True), (HERE / name).read_text().splitlines(True), fromfile='v3.4/template.md', tofile='v3.6/' + name)))
    for name in ('gist_layer.md', 'gist_full.md'):
        (HERE / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT / 'gist.md').read_text().splitlines(True), (HERE / name).read_text().splitlines(True), fromfile='v3.4/gist.md', tofile='v3.6/' + name)))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    files = [Path(__file__), HERE / 'author.py', HERE / 'author_log.json', HERE / 'static_checks.py', HERE / 'transfer_split.json',
             HERE / 'tools_live.json', ROOT / 'servers/contract.py', ROOT / 'servers/trace_contract.py', ROOT / 'servers/scales/s1/encode.py']
    files += [HERE / p for p in PARTS] + [HERE / (n + '.json') for n in arms] + [HERE / (n + '.diff') for n in ('template_layer.md', 'template_full.md', 'gist_layer.md', 'gist_full.md')]
    manifest = {'status': 'frozen_after_author_and_static_checks_before_model_calls', 'branch_commit': commit,
                'parent_arm': parent['arm_sha256'], 'arms': {n: a['arm_sha256'] for n, a in arms.items()},
                'assembly': 'encode._build_system_prompt(template, lived=True) on branch_commit; v3_6_full_v34tail swaps the runtime closure for V3.4 strategy + closure',
                'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
                'transfer_split_recorded_before_authoring': True, 'generalization_claim': 'none until the fresh cell reports'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({n: a['arm_sha256'][:12] for n, a in arms.items()} | {'branch_commit': commit[:12]}, indent=2))


def load_arm(name):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen V3.6 artifact changed: ' + relative)
    parent = parent_arm()
    if parent['arm_sha256'] != manifest['parent_arm']:
        raise ValueError('Parent arm changed')
    if name == 'v3_4_titles':
        return parent
    if name not in SPECS:
        raise ValueError('Unknown arm: ' + name)
    arm = json.loads((HERE / (name + '.json')).read_text())
    identity = {k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}
    if digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != manifest['arms'][name]:
        raise ValueError('Arm identity mismatch: ' + name)
    return arm


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--build-check', action='store_true')
    a = ap.parse_args()
    if a.freeze: freeze()
    elif a.build_check: build_check()
    else:
        for name in ('v3_4_titles', *SPECS): print(name, load_arm(name)['arm_sha256'])
