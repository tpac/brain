"""Nine deployed-package encodes beside nine saved V3.2 encodes.

Uses the existing isolated sequence runner. Production keeps its own system,
tools, preamble and absence of a gist. The factual replay and mutation engine
remain common so the saved V3.2 outputs can be reused. No live DB writes.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'eval/results/s1e_production_comparison_2026-09-11'
PRIOR = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'
spec = importlib.util.spec_from_file_location('_v32_saved_cell', ROOT / 'eval/fixtures/s1e_guide_v3_2_2026-09-10/corpus_cell.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
digest = baseline.digest
cell = baseline.cell
ARM = 'production_deployed'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def load_arm(name):
    if name != ARM:
        raise ValueError('Only the production arm may run in this comparison')
    arm = json.loads((HERE / 'production_deployed.json').read_text())
    identity = {k: v for k, v in arm.items() if k not in ('arm_id', 'arm_sha256')}
    if digest(identity) != arm['arm_sha256']:
        raise ValueError('Frozen production package changed')
    return arm


def prepare():
    baseline.check_pin()
    prod, candidate = load_arm(ARM), baseline.load_arm('v3_2_titles')
    for key in ('model', 'effort', 'max_tokens', 'max_rounds'):
        if prod['settings'][key] != candidate['settings'][key]:
            raise ValueError('Model or limits differ: ' + key)
    if not json.loads((PRIOR / 'completion.json').read_text())['status'] == 'complete':
        raise ValueError('Saved candidate is incomplete')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PRIOR / 'seed_baseline', OUT / 'seed_baseline')
    shutil.copy2(PRIOR / 'creative_design.json', OUT / 'creative_design.json')
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.json'))
    files += list((OUT / 'seed_baseline').iterdir()) + [OUT / 'creative_design.json']
    files += [Path(cell.__file__), ROOT / 'servers/scales/runner.py', ROOT / 'servers/contract.py',
              ROOT / 'servers/scales/s1/encode_contract.py', ROOT / 'eval/longmem/replay.py',
              ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py']
    for repeat in range(1, 4):
        files += [p for p in (PRIOR / 'v3_2_titles' / f'repeat{repeat}' / 'creative_design').rglob('*') if p.is_file()]
    save(OUT / 'manifest.json', {'status': 'frozen_before_model_calls', 'production_arm': prod['arm_sha256'],
        'candidate_arm': candidate['arm_sha256'], 'new_encodes': 9, 'reused_candidate_encodes': 9,
        'source': 'conv_004_art_design_extended; same 15 pairs, 3 sequential windows, 3 repetitions',
        'scope': 'Current deployed prompt/tool/preamble package in shared isolated replay. Not full daemon scheduling, provenance or retrieval comparison.',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}})
    print('FROZEN: nine new production encodes; nine saved V3.2 encodes', flush=True)


def check_pin():
    pin = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in pin['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Comparison input changed: ' + relative)
    if load_arm(ARM)['arm_sha256'] != pin['production_arm']:
        raise ValueError('Production identity changed')
    return pin


def configure():
    cell.OUT, cell.load_arm, cell.check_pin = OUT, load_arm, check_pin
    cell.PREAMBLE = load_arm(ARM)['user_preamble']


def sections(text):
    return {name: re.search(r'<' + name + r'(?: [^>]*)?>[\s\S]*?</' + name + r'>', text).group()
            for name in ('continuity', 'node_catalog', 'timeline')}


def preflight():
    check_pin()
    configure()
    cell.run_sequence(ARM, 0, 'creative_design', True)
    new = sections((OUT / ARM / 'preflight/creative_design/window1/user.txt').read_text())
    for repeat in range(1, 4):
        old = sections((PRIOR / 'v3_2_titles' / f'repeat{repeat}' / 'creative_design/window1/user.txt').read_text())
        if new != old:
            raise ValueError('Different initial factual sections')
    save(OUT / 'preflight.json', {'status': 'passed', 'model_calls': 0,
        'same_initial_factual_sections_sha256': digest(new), 'saved_repeats_matched': 3,
        'native_tools_from_verified_production_export': True, 'production_has_no_gist': True,
        'separate_production_preamble_retained': True})
    print('PREFLIGHT PASSED: exact factual sections, frozen native package, zero model calls', flush=True)


def launch():
    check_pin()
    if not (OUT / 'preflight.json').exists():
        raise ValueError('Preflight required')
    save(OUT / 'launch.json', {'status': 'started', 'new_encodes': 9, 'repeats': 3,
        'authorization': 'Tom asked to compare the latest runs against current deployed code; existing Sonnet isolated-eval workflow already approved.'})
    live = []
    try:
        for repeat in range(1, 4):
            cwd = OUT / 'process_dirs' / f'repeat{repeat}'
            cwd.mkdir(parents=True, exist_ok=False)
            log = (OUT / f'repeat{repeat}.log').open('x')
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--repeat', str(repeat)],
                cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            live.append((process, log))
            print('LAUNCHED production repetition', repeat, flush=True)
        for process, log in live:
            if process.wait() != 0:
                raise RuntimeError('Production repetition failed; inspect logs')
            log.close()
    finally:
        for process, log in live:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in live:
            process.wait()
            log.close()
    save(OUT / 'completion.json', {'status': 'complete', 'new_encodes': 9, 'reused_candidate_encodes': 9})
    print('COMPLETE: nine production encodes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.repeat:
        configure()
        cell.run_sequence(ARM, args.repeat, 'creative_design')
    elif args.run:
        launch()
    else:
        print(check_pin()['status'])
