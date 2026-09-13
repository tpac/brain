"""Small V3.2 sanity using the existing sequential corpus runner.

Reuse pinned V3.1 results; three V3.2 repetitions cost nine new encodes.
All writes occur in independent IsolatedBrain copies of the closed seed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'
PREVIOUS = ROOT / 'eval/results/s1e_v31_cross_corpus_2026-09-08'
spec = importlib.util.spec_from_file_location('_v32_arms', HERE / 'arms.py')
arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arms)
load_arm, digest = arms.load_arm, arms.digest

spec = importlib.util.spec_from_file_location('_v31_corpus', arms.PARENT / 'corpus_cell.py')
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)
previous_check = cell.check_pin


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def prepare():
    previous_check()
    old, new = load_arm('v3_1_titles'), load_arm('v3_2_titles')
    if old['settings'] != new['settings'] or old['tools'] != new['tools']:
        raise ValueError('Comparison changed model settings or tools')
    OUT.mkdir(parents=True, exist_ok=False)
    seed = OUT / 'seed_baseline'
    seed.mkdir()
    for name in ('brain.db', 'brain_logs.db', 'aspects_v1.json'):
        shutil.copy2(PREVIOUS / 'seed_baseline' / name, seed / name)
    shutil.copy2(PREVIOUS / 'creative_design.json', OUT / 'creative_design.json')
    files = [Path(__file__), HERE / 'manifest.json', arms.PARENT / 'corpus_cell.py',
             ROOT / 'servers/scales/runner.py', ROOT / 'servers/scales/s1/encode_contract.py',
             ROOT / 'servers/contract.py', ROOT / 'eval/longmem/replay.py',
             ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py',
             ROOT / 'eval/s1e_guide_v2_sequence_probe.py', OUT / 'creative_design.json']
    files += list(seed.iterdir())
    for repeat in range(1, 4):
        source = PREVIOUS / 'v3_1_titles' / ('repeat' + str(repeat)) / 'creative_design'
        recorded = json.loads((source / 'arm.json').read_text())
        if recorded['arm_sha256'] != old['arm_sha256']:
            raise ValueError('Saved baseline arm differs')
        files += [p for p in source.rglob('*') if p.is_file()]
    save(OUT / 'manifest.json', {
        'status': 'pinned_before_new_model_calls',
        'arms': {name: load_arm(name)['arm_sha256'] for name in ('v3_1_titles', 'v3_2_titles')},
        'baseline': str((PREVIOUS / 'v3_1_titles').relative_to(ROOT)),
        'new_encodes': 9, 'reused_encodes': 9, 'repeats': 3,
        'corpus': 'existing creative_design, three sequential windows of five pairs',
        'scope': 'inspected development source; no transfer claim or benchmark score',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
    })


def check_pin():
    pin = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in pin['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Sanity input changed: ' + relative)
    for name, expected in pin['arms'].items():
        if load_arm(name)['arm_sha256'] != expected:
            raise ValueError('Sanity arm changed: ' + name)
    return pin


def configure():
    cell.OUT = OUT
    cell.load_arm = load_arm
    cell.check_pin = check_pin


def preflight():
    configure()
    for name in ('v3_1_titles', 'v3_2_titles'):
        cell.run_sequence(name, 0, 'creative_design', True)
    expected = None
    for name in ('v3_1_titles', 'v3_2_titles'):
        path = OUT / name / 'preflight/creative_design/window1/user.txt'
        body = path.read_text().replace(load_arm(name)['gist'].rstrip(), '<GIST>')
        if expected is not None and body != expected:
            raise ValueError('Arms do not receive identical initial factual input')
        expected = body
    for repeat in range(1, 4):
        source = PREVIOUS / 'v3_1_titles' / ('repeat' + str(repeat)) / 'creative_design/window1/user.txt'
        body = source.read_text().replace(load_arm('v3_1_titles')['gist'].rstrip(), '<GIST>')
        if body != expected:
            raise ValueError('Saved baseline initial request does not match')
    save(OUT / 'preflight.json', {'status': 'passed', 'model_calls': 0,
        'matching_initial_body_sha256': digest(expected), 'saved_baseline_repetitions_matched': 3,
        'parent_freeze_verified': True, 'settings_tools_and_closed_seed_matched': True})


def launch():
    check_pin()
    if json.loads((OUT / 'preflight.json').read_text())['status'] != 'passed':
        raise ValueError('Preflight required')
    live = []
    try:
        for repeat in range(1, 4):
            cwd = OUT / 'process_dirs' / ('repeat' + str(repeat))
            cwd.mkdir(parents=True, exist_ok=False)
            log = (OUT / ('repeat' + str(repeat) + '.log')).open('x')
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--repeat', str(repeat)],
                                       cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            live.append((process, log))
            print('LAUNCHED repetition', repeat, flush=True)
        for process, log in live:
            if process.wait() != 0:
                raise RuntimeError('Candidate repetition failed; inspect its log')
            log.close()
    finally:
        import os
        import signal
        for process, log in live:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in live:
            process.wait()
            log.close()
    save(OUT / 'completion.json', {'status': 'complete', 'new_encodes': 9, 'reused_baseline_encodes': 9})
    print('COMPLETE: nine candidate encodes; nine saved baseline encodes retained', flush=True)


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
        cell.run_sequence('v3_2_titles', args.repeat, 'creative_design')
    elif args.run:
        launch()
    else:
        parser.error('Select --prepare, --preflight, --repeat, or --run')
