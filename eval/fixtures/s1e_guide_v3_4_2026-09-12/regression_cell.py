"""V3.4 alone on the six V3.3 transfer corpora (development data now), against the saved brains.

Sources come from transfer_split.json (selected by id before authoring). Each
LongMemEval item becomes one fixture with one window per haystack session on
the session's own date; the synthetic emotions conversation gets three short
windows on a fixed clock. A fresh eval brain is the common closed seed; every
(arm, repeat, corpus) sequence runs in its own isolated copy, and the final
copy is kept for the downstream retrieved-subset test.
"""
import argparse
from datetime import datetime
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
OUT = ROOT / 'eval/results/s1e_v34_regression_2026-09-12'
V33T = ROOT / 'eval/results/s1e_v33_transfer_2026-09-11'
sys.path.insert(0, str(HERE))
import sequence  # noqa: E402
digest, save, pairs = sequence.digest, sequence.save, sequence.pairs
ARMS = ('v3_4_titles',)
MAX_LIVE = 6


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arm(name):
    if name == 'production_deployed':
        return _module(ROOT / 'eval/fixtures/s1e_production_comparison_2026-09-11/corpus_cell.py', '_prod_cell').load_arm(name)
    return _module(HERE / 'arms.py', '_v34_arms').load_arm(name)


def longmem_fixture(item, kind):
    windows = []
    for date, session in zip(item['haystack_dates'], item['haystack_sessions']):
        now = datetime.strptime(date, '%Y/%m/%d (%a) %H:%M').strftime('%Y-%m-%d %H:%M UTC')
        windows.append({'now': now, 'turns': pairs(session)})
    return {'source_id': item['question_id'], 'source_kind': 'LongMemEval oracle ' + item['question_type'] + ' (' + kind + ')',
            'clock': 'Original source dates; timezone unspecified in source, rendered as UTC consistently.',
            'counterpart': None, 'windows': windows,
            'prior_gold': {k: item[k] for k in ('question', 'answer', 'question_date', 'question_type', 'answer_session_ids')}}


def prepare():
    """Copy the V3.3 transfer cell's seed (source traces preseeded) and its six fixtures; no new source is read."""
    for arm in ARMS:
        load_arm(arm)
    if json.loads((V33T / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Saved V3.3 transfer cell is incomplete')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(V33T / 'seed_baseline', OUT / 'seed_baseline')
    names = sorted(json.loads((V33T / 'manifest.json').read_text())['corpora'])
    for name in names:
        shutil.copy2(V33T / (name + '.json'), OUT / (name + '.json'))
    fixtures = {name: json.loads((OUT / (name + '.json')).read_text()) for name in names}
    files = [Path(__file__), HERE / 'sequence.py', HERE / 'manifest.json', HERE / 'arms.py', ROOT / 'servers/scales/runner.py',
             ROOT / 'servers/contract.py', ROOT / 'servers/scales/s1/encode_contract.py', ROOT / 'eval/longmem/replay.py',
             ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py', ROOT / 'eval/s1e_guide_v2_sequence_probe.py']
    files += [OUT / (name + '.json') for name in names] + [p for p in (OUT / 'seed_baseline').iterdir() if p.is_file()]
    save(OUT / 'manifest.json', {'status': 'pinned_before_model_calls',
        'arms': {a: load_arm(a)['arm_sha256'] for a in ARMS},
        'baselines': str(V33T.relative_to(ROOT)),
        'corpora': {n: {'source_id': f['source_id'], 'kind': f['source_kind'], 'turn_counts': [len(w['turns']) for w in f['windows']]} for n, f in fixtures.items()},
        'substitutions': [], 'repeats': 3, 'arms_count': len(ARMS),
        'total_encodes': 3 * len(ARMS) * sum(len(f['windows']) for f in fixtures.values()),
        'parallelism': f'up to {MAX_LIVE} (arm, repeat) processes; windows and corpora sequential inside each',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
        'limits': ['regression on inspected development corpora; no transfer claim', 'compared against the saved production / V3.2 / V3.3 brains of the V3.3 transfer cell']})
    print('PREPARED:', len(fixtures), 'development corpora;', 3 * len(ARMS) * sum(len(f['windows']) for f in fixtures.values()), 'encodes', flush=True)


def check_pin():
    pin = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in pin['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Transfer input changed: ' + relative)
    for arm, expected in pin['arms'].items():
        if load_arm(arm)['arm_sha256'] != expected:
            raise ValueError('Transfer arm changed: ' + arm)
    return pin


def corpora():
    return sorted(json.loads((OUT / 'manifest.json').read_text())['corpora'])


def sections(text):
    return {name: re.search(r'<' + name + r'(?: [^>]*)?>[\s\S]*?</' + name + r'>', text).group()
            for name in ('continuity', 'node_catalog', 'timeline')}


def preflight():
    check_pin()
    for corpus in corpora():
        for arm in ARMS:
            sequence.run_sequence(load_arm(arm), arm, 0, corpus, OUT, OUT / 'seed_baseline', dry_run=True)
            now = sections((OUT / arm / 'preflight' / corpus / 'window1/user.txt').read_text())
            saved = sections((V33T / 'v3_3_titles' / 'preflight' / corpus / 'window1/user.txt').read_text())
            if now != saved:
                raise ValueError('Factual sections differ from the saved V3.3 transfer preflight on ' + corpus)
    save(OUT / 'preflight.json', {'status': 'passed', 'model_calls': 0, 'corpora': corpora(), 'arms': list(ARMS)})
    print('PREFLIGHT PASSED: identical factual sections for every arm on every corpus; zero model calls', flush=True)


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
        'authorization': 'Tom, 2026-09-12: one more weave; this set is the regression check on the inspected corpora against the saved brains.'})
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
                    cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
    parser.add_argument('--arm', choices=ARMS)
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
