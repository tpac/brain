"""Finish a V3.6 cell whose launcher died: run every (arm, repeat, corpus) sequence not yet complete.

The launcher in transfer_cell.py owns the schedule; when the process supervising it is
killed (a harness timeout, a closed terminal) the children it already started run on in
their own sessions, but the jobs it had not yet started never begin and completion.json is
never written. This driver reads the same pin and the same arm records, finds the sequences
without a final_brain.json, removes a corpus folder left half-written by a dead child, and
runs the missing sequences with the same parallelism. Nothing about an arm, a corpus or a
prompt changes; only the schedule is re-derived from what is on disk.

    ./dev python3 eval/fixtures/s1e_guide_v3_6_2026-09-13/resume_cell.py --plan
    ./dev python3 eval/fixtures/s1e_guide_v3_6_2026-09-13/resume_cell.py --run
    V36_REGRESSION_ARMS=v3_4_live ./dev python3 .../resume_cell.py --driver regression_cell.py --run

`--driver` names the cell module (transfer_cell.py by default, or regression_cell.py — whose
ARMS come from V36_REGRESSION_ARMS, so pass the same environment).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = sys.argv[sys.argv.index('--driver') + 1] if '--driver' in sys.argv else 'transfer_cell.py'
cell = _module(HERE / DRIVER, '_v36_cell')
OUT, ARMS, MAX_LIVE = cell.OUT, cell.ARMS, cell.MAX_LIVE


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def plan():
    """(arm, repeat, corpus) sequences still to run, and the half-written folders to clear first."""
    cell.check_pin()
    missing, clear = [], []
    for arm in ARMS:
        for repeat in (1, 2, 3):
            for corpus in cell.corpora_for(arm):
                folder = OUT / arm / f'repeat{repeat}' / corpus
                if (folder / 'final_brain.json').exists():
                    continue
                if folder.exists():
                    env = folder / 'environment.json'
                    pid = json.loads(env.read_text())['pid'] if env.exists() else None
                    if pid and alive(pid):
                        continue  # a surviving child is still writing this one
                    clear.append(folder)
                missing.append((arm, repeat, corpus))
    return missing, clear


def run_one(arm, repeat, corpus):
    cell.check_pin()
    cell.sequence.run_sequence(cell.load_arm(arm), arm, repeat, corpus, OUT, OUT / 'seed_baseline')


def launch(missing, clear):
    for folder in clear:
        shutil.rmtree(folder)
    live, completed = [], []
    log = (OUT / 'resume.log').open('a')
    try:
        while missing or live:
            while missing and len(live) < MAX_LIVE:
                arm, repeat, corpus = missing.pop(0)
                process = subprocess.Popen([sys.executable, str(Path(__file__)), '--driver', DRIVER, '--one', arm, str(repeat), corpus],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                live.append((process, arm, repeat, corpus))
                print('RESUMED', arm, repeat, corpus, flush=True)
            for job in live[:]:
                process, arm, repeat, corpus = job
                if process.poll() is not None:
                    live.remove(job)
                    if process.returncode:
                        raise RuntimeError(f'Sequence failed: {arm} {repeat} {corpus}; see resume.log')
                    completed.append([arm, repeat, corpus]); print('SEQUENCE COMPLETE', arm, repeat, corpus, flush=True)
            if live:
                time.sleep(2)
    finally:
        for process, *_ in live:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, *_ in live:
            process.wait()
        log.close()
    return completed


def finish(resumed):
    """Write completion.json once every sequence on disk is complete."""
    still, _ = plan()
    if still:
        raise RuntimeError(f'{len(still)} sequences still incomplete: {still[:5]}')
    repetitions = [[arm, repeat] for arm in ARMS for repeat in (1, 2, 3)]
    cell.save(OUT / 'completion.json', {'status': 'complete', 'repetitions': repetitions,
              'resumed_sequences': resumed, 'note': 'launcher died before scheduling every job; the missing sequences ran through resume_cell.py'})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--one', nargs=3, metavar=('ARM', 'REPEAT', 'CORPUS'))
    parser.add_argument('--driver', default='transfer_cell.py')
    args = parser.parse_args()
    if args.one:
        run_one(args.one[0], int(args.one[1]), args.one[2])
    elif args.plan or args.run:
        missing, clear = plan()
        print(json.dumps({'missing': missing, 'clear': [str(p.relative_to(OUT)) for p in clear]}, indent=1))
        if args.run and (OUT / 'completion.json').exists():
            raise FileExistsError('completion.json already present')
        if args.run:
            finish(launch(missing, clear))
    else:
        parser.error('--plan, --run or --one')
