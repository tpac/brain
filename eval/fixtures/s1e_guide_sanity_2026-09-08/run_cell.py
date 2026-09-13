"""Two independent arm processes; three repetitions of three sequential encodes.

--pin records the reviewed harness/fixture/baseline, after both offline
preflights. Default launches the approved model cell. No outputs overwritten.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
OUT = WT / 'eval/results/s1e_guide_sanity_2026-09-08'
RUNNER = WT / 'eval/s1e_guide_v2_sequence_probe.py'
FIXTURE = HERE / 'contrasts_three_windows.json'
ARMS = ['v2_frozen', 'v3_titles']
SID = 'ca930f27-three-window-sanity'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_pin():
    pin = json.loads((HERE / 'cell_manifest.json').read_text())
    for path, expected in pin['files'].items():
        assert digest(WT / path) == expected, 'Changed cell input: '+path
    for path, expected in pin['baseline_files'].items():
        assert digest(Path(path)) == expected, 'Changed baseline: '+path
    sys.path.insert(0, str(WT / 'eval/fixtures/s1e_guide_freeze_2026-09-08'))
    from frozen_arms import load_arm
    for arm in ARMS:
        assert load_arm(arm)['arm_sha256'] == pin['arm_sha256'][arm]
    return pin


def run_arm(arm):
    pin = check_pin()
    for repeat in range(1, 4):
        check_pin()
        log = OUT / f'{arm}_repeat{repeat}.log'
        if log.exists() or (OUT / arm / f'repeat{repeat}').exists():
            raise SystemExit('Refusing to overwrite '+str(log))
        cwd = OUT / 'process_dirs' / arm / f'repeat{repeat}'
        cwd.mkdir(parents=True, exist_ok=False)
        args = [sys.executable, str(RUNNER), '--frozen-arm', arm,
                '--run-id', f'repeat{repeat}', '--fixture', str(FIXTURE),
                '--out-dir', str(OUT), '--session-id', SID]
        print('START', arm, f'repeat{repeat}', flush=True)
        with log.open('x') as stream:
            result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        print('FINISH', arm, f'repeat{repeat}', 'exit', result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        for wn in range(1, 4):
            path = OUT / arm / f'repeat{repeat}' / f'window{wn}/result.json'
            assert path.exists(), 'Missing encode: '+str(path)
    print('ARM COMPLETE', arm, flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pin', action='store_true')
    group.add_argument('--run-arm', choices=ARMS)
    args = parser.parse_args()
    assert os.environ.get('BRAIN_S1E_LISTS_PREAMBLE') == '1'
    if args.pin:
        sys.path.insert(0, str(WT / 'eval/fixtures/s1e_guide_freeze_2026-09-08'))
        from frozen_arms import load_arm
        fixture = json.loads(FIXTURE.read_text())
        counts = [len(w['turns']) for w in fixture['windows']]
        assert len(counts) == 3 and all(0 < n <= 5 for n in counts) and sum(counts) <= 15
        preflights = [json.loads((OUT / arm / 'preflight/window1/preflight.json').read_text()) for arm in ARMS]
        assert all(p['status'] == 'passed' and p['model_calls'] == 0 for p in preflights)
        assert len({p['user_sha256'] for p in preflights}) == 1
        paths = [FIXTURE, RUNNER, Path(__file__), OUT / 'baseline.json',
                 OUT / 'background_catalog.txt', OUT / 'comparison_contract.json',
                 OUT / 'first_user_sha256.txt']
        baseline = Path(json.loads((OUT / 'baseline.json').read_text())['baseline_dir'])
        baseline_files = [baseline / name for name in ('brain.db','brain_logs.db')]
        if (baseline / 'aspects_v1.json').exists():
            baseline_files.append(baseline / 'aspects_v1.json')
        pin = {'status':'pinned_before_model_calls', 'arms':ARMS, 'repetitions':3,
               'encodes_per_repetition':3, 'turn_counts':counts,
               'total_encodes':18, 'parallelism':'two arm processes, sequential repeats/windows',
               'session_id':SID,
               'arm_sha256':{arm:load_arm(arm)['arm_sha256'] for arm in ARMS},
               'files':{str(p.relative_to(WT)):digest(p) for p in paths},
               'baseline_files':{str(p):digest(p) for p in baseline_files},
               'interpretation':'one synthetic regression sequence, not independent corpus items; no S1R/S2',
               'source_trace_limit':'source traces for all windows are preseeded for stable ids; only current-window text is shown and the encoder toolset has no episode-search tool'}
        with (HERE / 'cell_manifest.json').open('x') as stream:
            json.dump(pin, stream, indent=2); stream.write('\n')
        print(json.dumps(pin, indent=2))
        return
    check_pin()
    if args.run_arm:
        run_arm(args.run_arm)
        return
    children = []
    try:
        for arm in ARMS:
            children.append(subprocess.Popen([sys.executable, str(Path(__file__)), '--run-arm', arm], cwd=OUT, start_new_session=True))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise SystemExit('An arm failed; stopping the other arm before further spending')
            time.sleep(0.25)
        if any(child.returncode for child in children):
            raise SystemExit('Arm processes failed: '+str([child.returncode for child in children]))
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            if child.poll() is None:
                child.wait()
    print('SANITY CELL COMPLETE: 18 encodes', flush=True)


if __name__ == '__main__':
    main()
