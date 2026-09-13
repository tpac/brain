"""Detached launcher/resumer for a pinned cell: starts every (arm, repeat) job whose log does not
exist yet, keeps at most MAX_LIVE jobs alive, waits for all of them, then writes completion.json.
Safe to run beside jobs already started by the cell's own --run launcher (they are detected by
their logs and processes and simply waited for). Usage:
  nohup setsid ./dev python3 resume_launch.py <cell_script.py> <out_dir> [max_live] > <out>/launcher_resume.log 2>&1 &
"""
import json, os, subprocess, sys, time
from pathlib import Path

cell = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); MAX_LIVE = int(sys.argv[3]) if len(sys.argv) > 3 else 6
manifest = json.loads((OUT / 'manifest.json').read_text())
arms = list(manifest['arms']); corpora = sorted(manifest['corpora']); repeats = (1, 2, 3)
def last_window(corpus): return len(manifest['corpora'][corpus]['turn_counts'])
def done(arm, rep): return all((OUT / arm / f'repeat{rep}' / c / f'window{last_window(c)}' / 'result.json').exists() for c in corpora)
def running(arm, rep):
    out = subprocess.run(['pgrep', '-f', '--', f'{cell.name} --arm {arm} --repeat {rep}$'], capture_output=True, text=True).stdout.split()
    return [p for p in out if p != str(os.getpid())]
live = {}
while True:
    states = {}
    for arm in arms:
        for rep in repeats:
            if done(arm, rep): states[(arm, rep)] = 'done'
            elif running(arm, rep) or (arm, rep) in live and live[(arm, rep)].poll() is None: states[(arm, rep)] = 'running'
            elif (OUT / f'{arm}_repeat{rep}.log').exists(): states[(arm, rep)] = 'orphan-log'  # started earlier, not running, not done
            else: states[(arm, rep)] = 'pending'
    n_live = sum(1 for s in states.values() if s == 'running')
    for (arm, rep), s in states.items():
        if s == 'pending' and n_live < MAX_LIVE:
            log = (OUT / f'{arm}_repeat{rep}.log').open('x'); cwd = OUT / 'process_dirs' / arm / f'repeat{rep}'; cwd.mkdir(parents=True, exist_ok=False)
            live[(arm, rep)] = subprocess.Popen([sys.executable, str(cell), '--arm', arm, '--repeat', str(rep)], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            n_live += 1; print('LAUNCHED', arm, rep, flush=True)
    if all(s == 'done' for s in states.values()):
        break
    bad = [k for k, s in states.items() if s == 'orphan-log']
    if bad and not any(s in ('running', 'pending') for s in states.values()):
        print('STALLED — jobs with a log but no process and no final result:', bad, flush=True); sys.exit(1)
    time.sleep(15)
completed = [[arm, rep] for arm in arms for rep in repeats]
(OUT / 'completion.json').write_text(json.dumps({'status': 'complete', 'repetitions': completed, 'launcher': 'resume_launch.py'}, indent=1))
print('COMPLETE', flush=True)
