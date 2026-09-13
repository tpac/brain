"""Three arms on fresh material: production, frozen V3.4 (parent), frozen V3.5 (candidate); all on the live tools.

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
OUT = ROOT / 'eval/results/s1e_v35_transfer_2026-09-12'
sys.path.insert(0, str(HERE))
import sequence  # noqa: E402
digest, save, pairs = sequence.digest, sequence.save, sequence.pairs
ARMS = ('production_deployed', 'v3_4_titles', 'v3_5_titles')
MAX_LIVE = 6


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arm(name):
    if name == 'production_deployed':
        return _module(ROOT / 'eval/fixtures/s1e_production_comparison_2026-09-11/corpus_cell.py', '_prod_cell').load_arm(name)
    return _module(HERE / 'arms.py', '_v35_arms').load_arm(name)


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
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    from servers.brain_traces import _s0_trace
    split = json.loads((HERE / 'transfer_split.json').read_text())
    for arm in ARMS:
        load_arm(arm)
    oracle = {it['question_id']: it for it in json.loads((ROOT / 'eval/longmem/data/longmemeval_oracle.json').read_text())}
    fixtures, substitutions = {}, []
    for qtype, ids in split['fresh_primary'].items():
        for qid in ids:
            fixtures['lm_' + qid] = longmem_fixture(oracle[qid], qtype)
    (synthetic_name, synthetic_note), = split['fresh_synthetic'].items()
    design = json.loads((ROOT / f'eval/corpus/{synthetic_name}.json').read_text())
    turns = pairs(design['exchanges'])
    cuts = [(0, 2), (2, 4), (4, len(turns))]
    fixtures[synthetic_name] = {'source_id': design['id'], 'source_kind': 'existing repository synthetic conversation, never used in an encoder eval before: ' + synthetic_note,
        'clock': 'No source dates; fixed synthetic date April 2 2026, one hour between encoding cuts.',
        'counterpart': design.get('counterpart'), 'windows': [{'now': f'2026-04-02 {12+i:02}:00 UTC', 'turns': turns[a:b]} for i, (a, b) in enumerate(cuts)],
        'prior_gold': design.get('ground_truth')}
    OUT.mkdir(parents=True, exist_ok=False)
    seed = OUT / 'seed_baseline'
    seed.mkdir()
    shutil.copy2(ROOT / 'servers/scales/s2/aspects_v1.json', seed / 'aspects_v1.json')
    os.environ['ASPECTS_JSON_PATH'] = str(seed / 'aspects_v1.json')
    brain = create_fresh_eval_brain(str(seed), wipe=False)
    for name, fixture in fixtures.items():
        fixture['session_id'] = digest('s1e-v35:' + name)[:8] + '-transfer'
        ctx = brain.get_or_create_session(fixture['session_id'])
        for window in fixture['windows']:
            for turn in window['turns']:
                ctx.stop_counter += 1
                for role, ref_type, event in [('other', 'user_message', 'K'), ('me', 'assistant_message', 'delta')]:
                    turn[role + '_trace'] = _s0_trace(brain, ctx, event, ref_type, turn[role][:200], content=turn[role])
        save(OUT / (name + '.json'), fixture)
    brain.save(); brain.close()
    files = [Path(__file__), HERE / 'sequence.py', HERE / 'manifest.json', HERE / 'transfer_split.json', HERE / 'arms.py',
             ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py',
             ROOT / 'eval/s1e_guide_v2_sequence_probe.py', ROOT / 'servers/scales/runner.py',
             ROOT / 'servers/contract.py', ROOT / 'servers/scales/s1/encode_contract.py', ROOT / 'eval/longmem/replay.py']
    files += [OUT / (name + '.json') for name in fixtures] + [p for p in seed.iterdir() if p.is_file()]
    save(OUT / 'manifest.json', {'status': 'pinned_before_model_calls',
        'arms': {a: load_arm(a)['arm_sha256'] for a in ARMS},
        'corpora': {n: {'source_id': f['source_id'], 'kind': f['source_kind'], 'turn_counts': [len(w['turns']) for w in f['windows']]} for n, f in fixtures.items()},
        'substitutions': substitutions, 'repeats': 3, 'arms_count': len(ARMS),
        'total_encodes': 3 * len(ARMS) * sum(len(f['windows']) for f in fixtures.values()),
        'parallelism': f'up to {MAX_LIVE} (arm, repeat) processes; windows and corpora sequential inside each',
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
        'limits': ['encode-only replay; downstream retrieved-subset test runs separately on the kept final brains',
                   'no S1R/S2 during encoding', 'all source traces preseeded; only the current window shown',
                   'items selected by id before the candidate was authored; no content or answer read during authoring',
                   'the twelve V3.3/V3.4 corpora are development data now and run separately as regression sets for V3.5']})
    print('PREPARED:', len(fixtures), 'corpora;', 3 * len(ARMS) * sum(len(f['windows']) for f in fixtures.values()), 'encodes', flush=True)


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
        seen = None
        for arm in ARMS:
            sequence.run_sequence(load_arm(arm), arm, 0, corpus, OUT, OUT / 'seed_baseline', dry_run=True)
            now = sections((OUT / arm / 'preflight' / corpus / 'window1/user.txt').read_text())
            if seen is not None and now != seen:
                raise ValueError('Arms do not receive identical factual sections on ' + corpus)
            seen = now
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
        'authorization': 'Tom, 2026-09-12: spend on the weave and implement the take; analysis hardened before the run; fresh corpora chosen by id before authoring; Sonnet 4.6, three repeats.'})
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
