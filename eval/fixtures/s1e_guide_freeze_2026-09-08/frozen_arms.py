"""Load reviewed, content-addressed prompt arms without a live brain.

Evaluation callers must consume load_arm(), not send the cue template alone:
the strategy belongs after the generated references. This module does not run
an eval, register a prompt or patch production assembly. Future corpus runners
must key their cache by the returned arm_sha256 in addition to normal inputs.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = HERE / 'frozen'


def digest(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def checked_path(relative, expected):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Frozen path escapes review worktree')
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f'Frozen input changed: {relative}')
    return data


def render_system(template, strategy, build_system, closure, suffix_sha256):
    system = build_system(prompt_instructions=template, lived=True)
    if not system.startswith(template):
        raise ValueError('Runtime did not preserve the supplied template')
    suffix = system[len(template):]
    if digest(suffix) != suffix_sha256:
        raise ValueError('Shared system suffix changed')
    if not system.endswith(closure):
        raise ValueError('Shared Finishing contract is not last')
    if strategy:
        if '## Working strategy' in system:
            raise ValueError('Strategy already present')
        system = system[:-len(closure)].rstrip()+'\n\n'+strategy.rstrip()+'\n\n'+closure
    return system


def manifest():
    data = (FROZEN / 'manifest.json').read_bytes()
    expected = (FROZEN / 'manifest.sha256').read_text().strip()
    if digest(data) != expected:
        raise ValueError('Freeze manifest changed')
    value = json.loads(data)
    if value['format_version'] != 1 or value['status'] != 'frozen_for_review':
        raise ValueError('Unknown freeze format or status')
    for relative, expected in value['files'].items():
        checked_path(relative, expected)
    return value


def load_arm(arm_id):
    """Return exact reviewed inputs; reject any changed file in this set."""
    data = manifest()
    arm = data['arms'][arm_id]
    system = checked_path(arm['system'], data['files'][arm['system']]).decode()
    gist = checked_path(data['gist'], data['files'][data['gist']]).decode()
    tools = json.loads(checked_path(data['tools'], data['files'][data['tools']]))
    identity = {'system_sha256': digest(system), 'gist_sha256': digest(gist),
                'tools_sha256': digest(json.dumps(tools, sort_keys=True)),
                'settings': data['settings'], 'assembly_adapter_sha256': data['assembly_adapter_sha256']}
    if digest(json.dumps(identity, sort_keys=True)) != arm['arm_sha256']:
        raise ValueError('Arm identity does not match assembled inputs')
    return {'arm_id': arm_id, 'arm_sha256': arm['arm_sha256'],
            'system_prompt': system, 'gist': gist, 'tools': tools,
            'settings': data['settings'],
            'cache_rule': 'Include arm_sha256 in the corpus cache identity; raw template hash is insufficient.'}
