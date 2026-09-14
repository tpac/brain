"""Put a frozen arm's template and gist into the code defaults, and prove the runtime assembles the measured prompt.

The deploy of a candidate is the edit of two literals — `SYSTEM_PROMPT` in
servers/scales/s1/encoding_prompt.py (interaction `s1e`) and in encoding_gist_prompt.py
(`s1e_gist`) — followed by the merge. What must be true afterwards: the runtime's
`encode._build_system_prompt(SYSTEM_PROMPT, lived=True)` equals the frozen arm's
`system_prompt` byte for byte, and the gist module equals the arm's `gist`. `--check`
asserts exactly that (fails loudly with a unified diff); `--write` performs the edit and
then runs the check. Reads the arm through the fixture's own loader, so the arm's identity
against the frozen manifest is verified on the way.

    ./dev python3 deploy_defaults.py --arm v3_6_full --check
    ./dev python3 deploy_defaults.py --arm v3_6_full --write
"""
import argparse, difflib, importlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
TEMPLATE_MODULE = ROOT / 'servers/scales/s1/encoding_prompt.py'
GIST_MODULE = ROOT / 'servers/scales/s1/encoding_gist_prompt.py'
OPEN = 'SYSTEM_PROMPT = """'
CLOSE = '"""'


def load_arm(name):
    """The frozen arm from its own record, identity-checked against the manifest.

    Not the fixture loader: that one pins the runtime files (trace_contract.py,
    encode.py, contract.py) the arm was built with and refuses once main moves
    them — correct for re-running the cell, wrong for this question, which is
    exactly whether TODAY's runtime still assembles the measured prompt."""
    spec = importlib.util.spec_from_file_location('_arms', HERE / 'arms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = json.loads((HERE / (name + '.json')).read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    identity = {k: arm[k] for k in ('system_prompt', 'gist', 'tools', 'settings')}
    if module.digest(identity) != arm['arm_sha256'] or arm['arm_sha256'] != manifest['arms'][name]:
        raise ValueError('Arm identity mismatch: ' + name)
    return arm


def replace_literal(path, body):
    """Replace the body of the module's SYSTEM_PROMPT triple-quoted literal with `body`."""
    if CLOSE in body:
        raise ValueError('candidate text contains a triple quote; the literal form cannot carry it')
    text = path.read_text()
    start = text.index(OPEN) + len(OPEN)
    end = text.index(CLOSE, start)
    path.write_text(text[:start] + body + text[end:])


def check(arm):
    for module_name in ('servers.scales.s1.encoding_prompt', 'servers.scales.s1.encoding_gist_prompt', 'servers.scales.s1.encode'):
        sys.modules.pop(module_name, None)
    from servers.scales.s1 import encoding_prompt, encoding_gist_prompt, encode
    assembled = encode._build_system_prompt(encoding_prompt.SYSTEM_PROMPT, lived=True)
    problems = []
    if assembled != arm['system_prompt']:
        diff = difflib.unified_diff(arm['system_prompt'].splitlines(True), assembled.splitlines(True), 'frozen arm system_prompt', 'runtime assembly of the code default', n=1)
        problems.append('SYSTEM PROMPT DIFFERS:\n' + ''.join(list(diff)[:80]))
    if encoding_gist_prompt.SYSTEM_PROMPT != arm['gist']:
        diff = difflib.unified_diff(arm['gist'].splitlines(True), encoding_gist_prompt.SYSTEM_PROMPT.splitlines(True), 'frozen arm gist', 'code default gist', n=1)
        problems.append('GIST DIFFERS:\n' + ''.join(list(diff)[:80]))
    if problems:
        raise SystemExit('\n\n'.join(problems))
    print('OK: runtime assembly of the code default == %s system_prompt (%d chars); gist == arm gist (%d chars)'
          % (arm['arm_id'], len(assembled), len(arm['gist'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    arm = load_arm(a.arm)
    if a.write:
        # the arm carries the ASSEMBLED system prompt; the template is the fixture part the arm was built from
        spec_name = {'v3_6_full': ('template_full.md', 'gist_full.md'), 'v3_6_layer': ('template_layer.md', 'gist_layer.md'),
                     'v3_6_full_v34tail': None, 'v3_4_live': None, 'v3_4_titles': None}.get(a.arm)
        if not spec_name:
            raise SystemExit('%s is not deployable as two code defaults (it carries V3.4 parts or a swapped closure); '
                             'for v3_4_live use the parent fixture\'s template.md / gist.md by hand' % a.arm)
        replace_literal(TEMPLATE_MODULE, (HERE / spec_name[0]).read_text())
        replace_literal(GIST_MODULE, (HERE / spec_name[1]).read_text())
        print('written:', TEMPLATE_MODULE.relative_to(ROOT), GIST_MODULE.relative_to(ROOT))
    if a.write or a.check:
        check(arm)
    else:
        ap.error('--write or --check')
