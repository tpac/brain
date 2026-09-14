"""Freeze the deployed S1E prompt/tool package without opening a brain DB."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path('/Users/tpac/brain')
EXPECTED_FINGERPRINT = '8c33e3f67dc54c06'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    from servers.daemon_config import _code_fingerprint
    if _code_fingerprint() != EXPECTED_FINGERPRINT:
        raise RuntimeError('Source code differs from the running daemon fingerprint')
    effective = json.loads((HERE / 'effective_s1e.json').read_text())
    from servers.scales.s1.encoding_prompt import SYSTEM_PROMPT
    from servers.scales.s1.encode import _build_system_prompt, _get_tool_schemas
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    if effective['template'] != SYSTEM_PROMPT or effective['stamp']['source'] != 'default':
        raise RuntimeError('Effective deployed prompt does not match source default')
    tree = ast.parse((SOURCE / 'servers/scales/s1/encode.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_user_content')
    preambles = [ast.literal_eval(n.value) for n in ast.walk(fn)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'preamble' for t in n.targets)]
    lived = [p for p in preambles if p.startswith("I'm encoding")]
    if len(lived) != 1:
        raise RuntimeError('Production lived preamble is not unique')
    arm = {'system_prompt': _build_system_prompt(effective['template'], lived=True),
           'gist': '', 'user_preamble': lived[0], 'tools': _get_tool_schemas(),
           'settings': {**effective['config'], 'max_tokens': ENCODING_AGENT['max_tokens'],
                        'max_rounds': ENCODING_AGENT.get('max_rounds', 5),
                        'lived_sequence': True, 'lists_preamble': False}}
    arm.update(arm_id='production_deployed', arm_sha256=digest(arm))
    source_files = {str(Path(m.__file__).resolve().relative_to(SOURCE))
                    for m in sys.modules.values() if getattr(m, '__file__', None)
                    and Path(m.__file__).resolve().is_relative_to(SOURCE)
                    and str(m.__file__).endswith('.py')}
    source_files |= {'servers/scales/s1/encode.py', 'hooks/scripts/brain-env.sh'}
    if _code_fingerprint() != EXPECTED_FINGERPRINT:
        raise RuntimeError('Production source changed during export')
    save('production_deployed.json', arm)
    save('export_metadata.json', {
        'daemon_source_dir': str(SOURCE), 'daemon_code_fingerprint': EXPECTED_FINGERPRINT,
        'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'main_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'main'], text=True).strip(),
        'effective_stamp': effective['stamp'], 'template_chars': len(effective['template']),
        'system_chars': len(arm['system_prompt']), 'tools_chars': len(json.dumps(arm['tools'], ensure_ascii=False)),
        'source_hashes': {p: digest((SOURCE / p).read_bytes()) for p in sorted(source_files)},
        'scope': 'Deployed template, generated field/Arc/Review/closure instructions, native tools, model/limits, and preamble. Shared prior eval catalog/storage/journal replay; not a full live-daemon A/B.',
    })
    print(json.dumps({'arm_sha256': arm['arm_sha256'], 'settings': arm['settings'],
                      'system_chars': len(arm['system_prompt']), 'tools_chars': len(json.dumps(arm['tools'], ensure_ascii=False)),
                      'template_matches_effective': True, 'code_matches_running_daemon': True}, indent=2))


if __name__ == '__main__':
    main()
