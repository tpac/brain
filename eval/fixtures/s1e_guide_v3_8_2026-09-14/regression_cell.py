"""The shipping rung(s) on the six V3.6 fresh corpora (development data now), against the saved V3.6 brains.

  V37_REGRESSION_ARMS=v3_7_advice           one or more V3.7 arms, comma-separated; no default —
                                            the rung is named after the carriers cell is read

Sources and seed come from the saved V3.6 refine cell (eval/results/s1e_v36_refine_2026-09-13):
the same fixtures, the same preseeded traces, the same clock; no new source is read. Every
(arm, repeat, corpus) sequence runs in its own isolated copy and the final copy is kept for the
downstream test. The saved brains of v3_4_live and v3_6_full (that cell) are the comparison;
analyze.py's `regression_v36` cell gathers them.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'eval/results/s1e_v37_regression_v36_2026-09-14'
SAVED = ROOT / 'eval/results/s1e_v36_refine_2026-09-13'
SAVED_ARM = 'v3_6_full'
sys.path.insert(0, str(HERE))
import sequence  # noqa: E402
digest, save = sequence.digest, sequence.save
ARMS = tuple(a for a in os.environ.get('V37_REGRESSION_ARMS', '').split(',') if a)
MAX_LIVE = 6


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arm(name):
    return _module(HERE / 'arms.py', '_v37_arms').load_arm(name)


def prepare():
    """Copy the saved cell's seed (source traces preseeded) and its fixtures; no new source is read."""
    if not ARMS:
        raise ValueError('V37_REGRESSION_ARMS names the rung(s) to run')
    for arm in ARMS:
        load_arm(arm)
    if json.loads((SAVED / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Saved cell is incomplete: ' + str(SAVED))
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SAVED / 'seed_baseline', OUT / 'seed_baseline')
    names = sorted(json.loads((SAVED / 'manifest.json').read_text())['corpora'])
    for name in names:
        shutil.copy2(SAVED / (name + '.json'), OUT / (name + '.json'))
    fixtures = {name: json.loads((OUT / (name + '.json')).read_text()) for name in names}
    files = [Path(__file__), HERE / 'sequence.py', HERE / 'manifest.json', HERE / 'arms.py', ROOT / 'servers/scales/runner.py',
             ROOT / 'servers/contract.py', ROOT / 'servers/scales/s1/encode_contract.py', ROOT / 'eval/longmem/replay.py',
             ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py', ROOT / 'eval/s1e_guide_v2_sequence_probe.py']
    files += [OUT / (name + '.json') for name in names] + [p for p in (OUT / 'seed_baseline').iterdir() if p.is_file()]
    total = 3 * len(ARMS) * sum(len(f['windows']) for f in fixtures.values())
    save(OUT / 'manifest.json', {'status': 'pinned_before_model_calls',
        'arms': {a: load_arm(a)['arm_sha256'] for a in ARMS},
        'baselines': str(SAVED.relative_to(ROOT)),
        'corpora': {n: {'source_id': f['source_id'], 'kind': f['source_kind'], 'turn_counts': [len(w['turns']) for w in f['windows']]} for n, f in fixtures.items()},
        'repeats': 3, 'arms_count': len(ARMS), 'total_encodes': total,
        'parallelism': f'up to {MAX_LIVE} (arm, repeat) processes; windows and corpora sequential inside each',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
        'limits': ['regression on inspected development corpora; no transfer claim',
                   'compared against the saved v3_4_live and v3_6_full brains of ' + str(SAVED.relative_to(ROOT))]})
    print('PREPARED:', len(fixtures), 'development corpora;', total, 'encodes', flush=True)


def check_pin():
    pin = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in pin['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Regression input changed: ' + relative)
    for arm, expected in pin['arms'].items():
        if load_arm(arm)['arm_sha256'] != expected:
            raise ValueError('Regression arm changed: ' + arm)
    if tuple(pin['arms']) != ARMS:
        raise ValueError('V37_REGRESSION_ARMS differs from the pinned arms: ' + str(list(pin['arms'])))
    return pin


def corpora():
    return sorted(json.loads((OUT / 'manifest.json').read_text())['corpora'])


def sections(text):
    return {name: re.search(r'<' + name + r'(?: [^>]*)?>[\s\S]*?</' + name + r'>', text).group()
            for name in ('continuity', 'node_catalog', 'timeline')}


def preflight():
    """Identical factual sections across this cell's arms (the gate); equality with the saved
    cell's preflight is recorded, not gated — the runtime render may have moved since V3.4."""
    check_pin()
    same_as_saved = {}
    for corpus in corpora():
        seen = None
        for arm in ARMS:
            sequence.run_sequence(load_arm(arm), arm, 0, corpus, OUT, OUT / 'seed_baseline', dry_run=True)
            now = sections((OUT / arm / 'preflight' / corpus / 'window1/user.txt').read_text())
            if seen is not None and now != seen:
                raise ValueError('Arms do not receive identical factual sections on ' + corpus)
            seen = now
        saved = sections((SAVED / SAVED_ARM / 'preflight' / corpus / 'window1/user.txt').read_text())
        same_as_saved[corpus] = {name: seen[name] == saved[name] for name in seen}
    save(OUT / 'preflight.json', {'status': 'passed', 'model_calls': 0, 'corpora': corpora(), 'arms': list(ARMS),
                                  'factual_sections_identical_to_saved_cell': same_as_saved})
    print('PREFLIGHT PASSED: identical factual sections for every arm on every corpus; zero model calls', flush=True)
    print('identical to the saved V3.6 cell:', json.dumps(same_as_saved), flush=True)


def run_arm(arm, repeat):
    check_pin()
    frozen = load_arm(arm)
    for corpus in corpora():
        sequence.run_sequence(frozen, arm, repeat, corpus, OUT, OUT / 'seed_baseline')


def launch():
    check_pin()
    if json.loads((OUT / 'preflight.json').read_text())['status'] != 'passed':
        raise ValueError('Preflight required')
    save(OUT / 'launch.json', {'status': 'started', 'arms': list(ARMS), 'repeats': 3,
        'authorization': 'Tom, 2026-09-14: the V3.7 round; the regression check of the shipping rung on the V3.6 corpora (development data now) against the saved v3_6_full brains.'})
    jobs = [(arm, repeat) for repeat in range(1, 4) for arm in ARMS]
    live, completed = [], []
    try:
        while jobs or live:
            while jobs and len(live) < MAX_LIVE:
                arm, repeat = jobs.pop(0)
                log = (OUT / f'{arm}_repeat{repeat}.log').open('x')
                cwd = OUT / 'process_dirs' / arm / f'repeat{repeat}'
                cwd.mkdir(parents=True, exist_ok=False)
                process = subprocess.Popen([sys.executable, str(Path(__file__)), '--arm', arm, '--repeat', str(repeat)],
                    cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, V37_REGRESSION_ARMS=','.join(ARMS)))
                live.append((process, log, arm, repeat))
                print('LAUNCHED', arm, repeat, flush=True)
            for job in live[:]:
                process, log, arm, repeat = job
                if process.poll() is not None:
                    log.close(); live.remove(job)
                    if process.returncode:
                        raise RuntimeError(f'Repetition failed: {arm} {repeat}; see log')
                    completed.append([arm, repeat]); print('REPETITION COMPLETE', arm, repeat, flush=True)
            if live:
                time.sleep(2)
    finally:
        for process, log, _, _ in live:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log, _, _ in live:
            process.wait(); log.close()
    save(OUT / 'completion.json', {'status': 'complete', 'repetitions': completed})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--arm', choices=ARMS or None)
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.arm and args.repeat:
        run_arm(args.arm, args.repeat)
    elif args.run:
        launch()
    else:
        print(check_pin()['status'])
