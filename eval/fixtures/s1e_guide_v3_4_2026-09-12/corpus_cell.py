"""V3.4 sanity on the development source: nine new encodes beside twenty-seven saved.

Reuses the production comparison's seed and creative_design fixture so the nine
saved production encodes and nine saved V3.2 encodes stay comparable. Three
independent repetitions in separate processes, three sequential windows each.
No live DB writes; final isolated brains are kept for inspection.
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
OUT = ROOT / 'eval/results/s1e_v34_sanity_2026-09-12'
V33 = ROOT / 'eval/results/s1e_v33_sanity_2026-09-11'
PRIOR = ROOT / 'eval/results/s1e_production_comparison_2026-09-11'
V32 = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'
sys.path.insert(0, str(HERE))
import sequence  # noqa: E402
spec = importlib.util.spec_from_file_location('_v34_arms', HERE / 'arms.py')
arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arms)
ARM = 'v3_4_titles'
digest, save = sequence.digest, sequence.save


def prepare():
    candidate = arms.load_arm(ARM)
    prior = json.loads((PRIOR / 'manifest.json').read_text())
    if json.loads((PRIOR / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Saved production comparison is incomplete')
    if json.loads((V32 / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Saved V3.2 sanity is incomplete')
    v33 = arms.load_arm('v3_3_titles')
    if candidate['settings'] != v33['settings']:
        raise ValueError('Candidate changed model settings')
    if json.loads((V33 / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Saved V3.3 sanity is incomplete')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PRIOR / 'seed_baseline', OUT / 'seed_baseline')
    shutil.copy2(PRIOR / 'creative_design.json', OUT / 'creative_design.json')
    files = [Path(__file__), HERE / 'sequence.py', HERE / 'manifest.json', ROOT / 'servers/scales/runner.py',
             ROOT / 'servers/contract.py', ROOT / 'servers/scales/s1/encode_contract.py', ROOT / 'eval/longmem/replay.py',
             ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py',
             ROOT / 'eval/s1e_guide_v2_sequence_probe.py', OUT / 'creative_design.json']
    files += list((OUT / 'seed_baseline').iterdir())
    for repeat in range(1, 4):
        files += [p for p in (PRIOR / 'production_deployed' / f'repeat{repeat}' / 'creative_design').rglob('*') if p.is_file()]
        files += [p for p in (V32 / 'v3_2_titles' / f'repeat{repeat}' / 'creative_design').rglob('*') if p.is_file()]
        files += [p for p in (V33 / 'v3_3_titles' / f'repeat{repeat}' / 'creative_design').rglob('*') if p.is_file() and p.suffix in ('.json', '.txt')]
    save(OUT / 'manifest.json', {'status': 'pinned_before_new_model_calls',
        'arms': {ARM: candidate['arm_sha256'], 'v3_3_titles': v33['arm_sha256'], 'production_deployed': prior['production_arm']},
        'tools_choice': json.loads((HERE / 'manifest.json').read_text())['tools_choice'],
        'baselines': {'production_deployed': str((PRIOR / 'production_deployed').relative_to(ROOT)),
                      'v3_2_titles': str((V32 / 'v3_2_titles').relative_to(ROOT)), 'v3_3_titles': str((V33 / 'v3_3_titles').relative_to(ROOT))},
        'new_encodes': 9, 'reused_encodes': 27, 'repeats': 3,
        'corpus': 'existing creative_design, three sequential windows of five pairs',
        'scope': 'inspected development source; regression check, no transfer claim',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}})
    print('PREPARED: nine V3.4 encodes beside twenty-seven saved', flush=True)


def check_pin():
    pin = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in pin['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Sanity input changed: ' + relative)
    if arms.load_arm(ARM)['arm_sha256'] != pin['arms'][ARM]:
        raise ValueError('Candidate arm changed')
    return pin


def sections(text):
    return {name: re.search(r'<' + name + r'(?: [^>]*)?>[\s\S]*?</' + name + r'>', text).group()
            for name in ('continuity', 'node_catalog', 'timeline')}


def preflight():
    check_pin()
    sequence.run_sequence(arms.load_arm(ARM), ARM, 0, 'creative_design', OUT, OUT / 'seed_baseline', dry_run=True)
    new = sections((OUT / ARM / 'preflight/creative_design/window1/user.txt').read_text())
    for repeat in range(1, 4):
        for base in (PRIOR / 'production_deployed', V32 / 'v3_2_titles', V33 / 'v3_3_titles'):
            old = sections((base / f'repeat{repeat}' / 'creative_design/window1/user.txt').read_text())
            if new != old:
                raise ValueError('Different initial factual sections versus ' + str(base))
    save(OUT / 'preflight.json', {'status': 'passed', 'model_calls': 0,
        'same_initial_factual_sections_sha256': digest(new), 'saved_repeats_matched': 9})
    print('PREFLIGHT PASSED: exact factual sections against both saved baselines, zero model calls', flush=True)


def launch():
    check_pin()
    if json.loads((OUT / 'preflight.json').read_text())['status'] != 'passed':
        raise ValueError('Preflight required')
    save(OUT / 'launch.json', {'status': 'started', 'new_encodes': 9, 'repeats': 3,
        'authorization': 'Tom, 2026-09-12: one more weave; sanity regression on the development source beside the saved arms.'})
    live = []
    try:
        for repeat in range(1, 4):
            cwd = OUT / 'process_dirs' / f'repeat{repeat}'
            cwd.mkdir(parents=True, exist_ok=False)
            log = (OUT / f'repeat{repeat}.log').open('x')
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--repeat', str(repeat)],
                cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            live.append((process, log))
            print('LAUNCHED repetition', repeat, flush=True)
        for process, log in live:
            if process.wait() != 0:
                raise RuntimeError('Candidate repetition failed; inspect its log')
            log.close()
    finally:
        for process, log in live:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in live:
            process.wait(); log.close()
    save(OUT / 'completion.json', {'status': 'complete', 'new_encodes': 9, 'reused_baseline_encodes': 27})
    print('COMPLETE: nine V3.4 encodes; twenty-seven saved baseline encodes retained', flush=True)


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
        check_pin()
        sequence.run_sequence(arms.load_arm(ARM), ARM, args.repeat, 'creative_design', OUT, OUT / 'seed_baseline')
    elif args.run:
        launch()
    else:
        print(check_pin()['status'])
