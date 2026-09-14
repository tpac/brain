"""Freeze/load the V3.7 arms — the production baseline frozen, three carrier arms assembled the way the runtime assembles them.

V3.7 is one carrier per arm on top of the deployed V3.6 full (Tom, 2026-09-13: arms are
separate carriers, never bundles). Each carrier arm's system prompt is
`encode._build_system_prompt(template, lived=True)` on this branch — the candidate
template, the contract field summary, the arc, review and closure blocks the runtime
injects — so what the cell measures is byte for byte what a merge would run. The baseline
is the V3.6 cell's frozen `v3_6_full` arm, which is production: `deploy_defaults.py --arm
v3_6_full --check` proves the code default assembles to it byte for byte, and `build()`
re-proves it here before any carrier is frozen.

  v3_4_live    — the V3.6 cell's frozen v3_4_live (V3.4's template and gist on the branch runtime);
                 continuity with the V3.6 cell's measurements
  v3_6_full    — the V3.6 cell's frozen v3_6_full == production; the baseline every carrier is read against
  v3_7_advice  — template_advice.md + gist_full.md: carrier 1, the advice-node scope guard inside the thin window (example)
  v3_7_quote   — template_quote.md + gist_full.md: carrier 2, a quote before-state in the Priya revise, replaced whole (example)
  v3_7_event   — template_event.md + gist_full.md: carrier 3, a plan comes due and its event_time moves with its state (example)

Freeze once, after author + static checks: `./dev python3 arms.py --freeze`.
`--build-check` prints sizes, the tool-schema comparison against the parent's frozen tools,
the baseline re-assembly check, and the assembly diff of each carrier arm against v3_6_full.
"""
import argparse, difflib, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_6_2026-09-13'
sys.path.insert(0, str(ROOT))

# Frozen in the parent fixture; loaded from their own records, identity-checked against the parent manifest.
FROZEN = ('v3_4_live', 'v3_6_full')
# (template, gist) — fixture-relative parts assembled on this branch's runtime.
SPECS = {
    'v3_7_advice': ('template_advice.md', 'gist_full.md'),
    'v3_7_quote': ('template_quote.md', 'gist_full.md'),
    'v3_7_event': ('template_event.md', 'gist_full.md'),
}
ARMS = FROZEN + tuple(SPECS)
PARTS = ('template_advice.md', 'template_quote.md', 'template_event.md', 'gist_full.md')


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def parent_arm(name):
    """A V3.6 frozen arm from its own record, identity-checked against the V3.6 manifest.
    The V3.6 loader pins runtime files this branch may have moved on purpose; the arm's
    identity — the digest of its system prompt, gist, tools and settings — is what is
    checked, as V3.6 did for its own parent."""
    arm = json.loads((PARENT / (name + '.json')).read_text())
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    identity = {k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}
    if digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != manifest['arms'][name]:
        raise ValueError('Frozen V3.6 arm identity mismatch: ' + name)
    return arm


def runtime_system(template):
    from servers.scales.s1 import encode
    return encode._build_system_prompt(template, lived=True)


def strip_descriptions(o):
    if isinstance(o, dict): return {k: strip_descriptions(v) for k, v in o.items() if k != 'description'}
    if isinstance(o, list): return [strip_descriptions(x) for x in o]
    return o


def build():
    from servers.scales.s1 import encode
    baseline = parent_arm('v3_6_full')
    tools = encode._get_tool_schemas()
    if json.dumps(strip_descriptions(tools), sort_keys=True) != json.dumps(strip_descriptions(baseline['tools']), sort_keys=True):
        raise ValueError('Tool shapes differ from the frozen V3.6 tools — descriptions only may change')
    if baseline['settings']['model'] != 'claude-sonnet-4-6':
        raise ValueError('This comparison remains on Sonnet 4.6')
    # The baseline must be what today's runtime assembles from the parent's template — the deploy check, re-proved here.
    if runtime_system((PARENT / 'template_full.md').read_text()) != baseline['system_prompt']:
        raise ValueError('Today\'s runtime does not assemble the parent template_full.md to the frozen v3_6_full system — the baseline is not production')
    if (HERE / 'gist_full.md').read_text() != baseline['gist']:
        raise ValueError('gist_full.md differs from the frozen v3_6_full gist — the carriers do not touch the gist')
    arms = {}
    for name, (tpl, gist) in SPECS.items():
        arm = {'system_prompt': runtime_system((HERE / tpl).read_text()), 'gist': (HERE / gist).read_text(), 'tools': tools, 'settings': baseline['settings']}
        arm.update(arm_id=name, arm_sha256=digest({k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}))
        arms[name] = arm
    return baseline, arms, tools


def build_check():
    baseline, arms, tools = build()
    out = {'baseline_v3_6_full': baseline['arm_sha256'][:12], 'baseline_system_chars': len(baseline['system_prompt']), 'baseline_gist_chars': len(baseline['gist']),
           'baseline_reassembles_on_this_runtime': True}
    for name, arm in arms.items():
        out[name] = {'sha': arm['arm_sha256'][:12], 'system_chars': len(arm['system_prompt']), 'delta_chars': len(arm['system_prompt']) - len(baseline['system_prompt'])}
    out['tool_description_changes_vs_v36'] = [t['name'] for t, p in zip(tools, baseline['tools']) if json.dumps(t, sort_keys=True) != json.dumps(p, sort_keys=True)]
    print(json.dumps(out, indent=2))
    for name, arm in arms.items():
        print(f'\n--- assembly diff: v3_6_full system -> {name} system (the deploy check) ---')
        d = list(difflib.unified_diff(baseline['system_prompt'].splitlines(True), arm['system_prompt'].splitlines(True),
                                      fromfile='v3_6_full (frozen == production)', tofile=name + ' (runtime assembly)', n=0))
        print(''.join(d) if len(d) < 200 else ''.join(d[:200]) + f'\n... {len(d) - 200} more lines')


def freeze():
    if (HERE / 'manifest.json').exists():
        raise FileExistsError('V3.7 is already frozen')
    baseline, arms, tools = build()
    for name, arm in arms.items():
        (HERE / (name + '.json')).write_text(json.dumps(arm, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'tools_live.json').write_text(json.dumps(tools, indent=2, ensure_ascii=False) + '\n')
    for tpl, _ in SPECS.values():
        (HERE / (tpl + '.diff')).write_text(''.join(difflib.unified_diff(
            (PARENT / 'template_full.md').read_text().splitlines(True), (HERE / tpl).read_text().splitlines(True), fromfile='v3.6/template_full.md', tofile='v3.7/' + tpl)))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    files = [Path(__file__), HERE / 'author.py', HERE / 'author_log.json', HERE / 'static_checks.py', HERE / 'transfer_split.json', HERE / 'ANALYSIS.md',
             HERE / 'tools_live.json', ROOT / 'servers/contract.py', ROOT / 'servers/trace_contract.py', ROOT / 'servers/scales/s1/encode.py',
             PARENT / 'v3_4_live.json', PARENT / 'v3_6_full.json', PARENT / 'manifest.json']
    files += [HERE / p for p in PARTS] + [HERE / (n + '.json') for n in arms] + [HERE / (tpl + '.diff') for tpl, _ in SPECS.values()]
    manifest = {'status': 'frozen_after_author_and_static_checks_before_model_calls', 'branch_commit': commit,
                'baseline_arm': baseline['arm_sha256'], 'frozen_from_parent': {n: parent_arm(n)['arm_sha256'] for n in FROZEN},
                'arms': {n: a['arm_sha256'] for n, a in arms.items()},
                'assembly': 'encode._build_system_prompt(template, lived=True) on branch_commit; gist_full.md unchanged from v3_6_full',
                'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
                'transfer_split_recorded_before_authoring': True, 'generalization_claim': 'none until the fresh cell reports'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({n: a['arm_sha256'][:12] for n, a in arms.items()} | {'branch_commit': commit[:12]}, indent=2))


def load_arm(name):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen V3.7 artifact changed: ' + relative)
    if name in FROZEN:
        arm = parent_arm(name)
        if arm['arm_sha256'] != manifest['frozen_from_parent'][name]:
            raise ValueError('Parent arm changed: ' + name)
        return arm
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
        for name in ARMS: print(name, load_arm(name)['arm_sha256'])
