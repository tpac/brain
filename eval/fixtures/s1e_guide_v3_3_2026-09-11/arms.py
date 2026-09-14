"""Freeze/load the V3.3 candidate without a live brain.

The parent (frozen V3.2) owns tools, model settings and the generated system
tail. Only template, gist and final strategy are revised; closure is unchanged.
Freeze after author, static checks and independent review; never regenerate
an existing freeze.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_2_2026-09-10'
REVIEW = ROOT / 'docs/S1E-V3-3-AUTHORING-REVIEW-2026-09-11.md'
CHALLENGE = ROOT / 'docs/challenges/semantic-fidelity.md'


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def parent_arm():
    spec = importlib.util.spec_from_file_location('_v32_frozen_arms', PARENT / 'arms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_arm('v3_2_titles')


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
        raise ValueError('V3.3 must retain the existing finishing contract')
    suffix = suffix.replace(old_strategy, parts['strategy.md'].rstrip())
    candidate = {key: parent[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    candidate.update(system_prompt=parts['template.md'] + suffix, gist=parts['gist.md'])
    candidate.update(arm_id='v3_3_titles', arm_sha256=digest(candidate))
    if not candidate['system_prompt'].endswith(parts['closure.md'].rstrip()):
        raise ValueError('Finishing contract is no longer last')
    if candidate['settings']['model'] != 'claude-sonnet-4-6':
        raise ValueError('This comparison remains on Sonnet 4.6')
    return parent, candidate, parts


def freeze():
    if (HERE / 'manifest.json').exists():
        raise FileExistsError('V3.3 is already frozen')
    parent, candidate, parts = build()
    for name in parts:
        before, after = (PARENT / name).read_text(), parts[name]
        (HERE / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile='v3.2/' + name, tofile='v3.3/' + name)))
    arm_path = HERE / 'v3_3_titles.json'
    arm_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + '\n')
    challenge_snapshot = HERE / 'CHALLENGES.md'
    challenge_snapshot.write_text(CHALLENGE.read_text())
    files = [Path(__file__), HERE / 'author.py', HERE / 'second_window.md', HERE / 'static_checks.py',
             HERE / 'transfer_split.json', arm_path, REVIEW, challenge_snapshot]
    files += [HERE / name for name in parts]
    files += [HERE / (name + '.diff') for name in parts]
    manifest = {
        'status': 'frozen_after_author_static_checks_cold_reader_and_independent_review_before_model_calls',
        'parent_arm': parent['arm_sha256'], 'candidate_arm': candidate['arm_sha256'],
        'files': {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in files},
        'scope': 'eval-only prompt candidate; no runtime, tools, catalog rendering or model-setting changes',
        'development_sources_already_seen': ['creative_design', 'longmem_unseen (gpt4_f49edff3)'],
        'transfer_split_recorded_before_authoring': True,
        'generalization_claim': 'none until the transfer cell reports',
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'candidate_arm': candidate['arm_sha256'], 'system_chars': len(candidate['system_prompt']),
                      'gist_chars': len(candidate['gist']), 'tools_chars': len(json.dumps(candidate['tools'], ensure_ascii=False))}, indent=2))


def load_arm(name):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen V3.3 artifact changed: ' + relative)
    parent = parent_arm()
    if parent['arm_sha256'] != manifest['parent_arm']:
        raise ValueError('Parent arm changed')
    if name == 'v3_2_titles':
        return parent
    if name != 'v3_3_titles':
        raise ValueError('Unknown comparison arm: ' + name)
    candidate = json.loads((HERE / 'v3_3_titles.json').read_text())
    identity = {key: candidate[key] for key in ('system_prompt', 'gist', 'tools', 'settings')}
    if digest(identity) != candidate['arm_sha256'] or candidate['arm_sha256'] != manifest['candidate_arm']:
        raise ValueError('Candidate identity mismatch')
    return candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--build-check', action='store_true', help='assemble without freezing; print sizes')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.build_check:
        parent, candidate, parts = build()
        print(json.dumps({'parent': parent['arm_sha256'][:12], 'candidate': candidate['arm_sha256'][:12],
                          'system_chars': len(candidate['system_prompt']), 'gist_chars': len(candidate['gist'])}, indent=2))
    else:
        for name in ('v3_2_titles', 'v3_3_titles'):
            print(name, load_arm(name)['arm_sha256'])
