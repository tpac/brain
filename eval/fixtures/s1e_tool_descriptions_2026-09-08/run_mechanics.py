"""Pinned old/new definitions through the existing eight mechanical checks.

API-only: no brain instance or dispatch. Each arm is sequential; arms overlap.
Save exact requests/responses. Stop scheduling on the first failed check.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'eval/results/s1e_tool_mechanics_2026-09-08'
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / 'eval')]
import candidate
import mcp_batch_probe as probe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def prepare():
    base = candidate.load_arm('v3_titles')
    new = candidate.load_candidate('v3_titles')
    assert base['system_prompt'] == new['system_prompt']
    assert candidate.mechanics(base['tools']) == candidate.mechanics(new['tools'])
    assert len(probe.SCENARIOS) == 8
    OUT.mkdir(parents=True, exist_ok=False)
    for name, arm in [('old', base), ('new', new)]:
        tool = next(t for t in arm['tools'] if t['name'] == 'brain_batch')
        save(OUT / (name + '.mcp.json'), {
            'name': tool['name'], 'description': tool['description'],
            'inputSchema': tool['input_schema']})
    save(OUT / 'cases.json', {
        'system': probe.SYSTEM, 'catalog': probe.CATALOG,
        'scenarios': [{k: s[k] for k in ('id', 'dimension', 'task')}
                      for s in probe.SCENARIOS]})
    paths = [Path(__file__), Path(probe.__file__), HERE / 'manifest.json',
             OUT / 'old.mcp.json', OUT / 'new.mcp.json', OUT / 'cases.json']
    save(OUT / 'manifest.json', {
        'status': 'pinned_before_calls', 'model': probe.MODEL,
        'max_tokens': probe.MAX_TOKENS, 'effort': 'unspecified (existing probe)',
        'tool_choice': 'forced brain_batch', 'repetitions': 3,
        'planned_calls': 48, 'concurrency': 'two arms, sequential cases/repetitions within arm',
        'stop': 'first failed mechanical check or API error; allow in-flight call to finish',
        'limitations': 'Short forced-tool probe; no production prompt, DB, or tool selection. Existing graders unchanged.',
        'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('PINNED: 2 definitions x 8 existing scenarios x 3 repetitions = 48 calls')


def check_pin():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(ROOT / name) == expected, 'Changed probe input: ' + name
    candidate.load_candidate('v3_titles')


class Capture:
    def __init__(self, client, folder):
        self.client = client
        self.folder = folder
        self.messages = self

    def create(self, **kwargs):
        save(self.folder / 'request.json', kwargs)
        response = self.client.messages.create(**kwargs)
        save(self.folder / 'response.json', response.model_dump(mode='json'))
        return response


def run_arm(name, stop, key):
    import anthropic
    tool, _ = probe.load_tool_def(OUT / (name + '.mcp.json'))
    folder = OUT / name
    folder.mkdir(exist_ok=False)
    results = []
    with anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=0) as client:
        for scenario in probe.SCENARIOS:
            for repeat in range(1, 4):
                if stop.is_set():
                    return results
                check_pin()
                path = folder / scenario['id'] / ('repeat' + str(repeat))
                path.mkdir(parents=True, exist_ok=False)
                try:
                    result = probe.run_sample(Capture(client, path), tool, scenario)
                except Exception as error:
                    result = {'passed': False, 'note': 'API/execution error: ' + type(error).__name__, 'ops': None}
                result.update(arm=name, scenario=scenario['id'], repeat=repeat)
                save(path / 'grade.json', result)
                results.append(result)
                print(name, scenario['id'], repeat,
                      'PASS' if result['passed'] else 'FAIL: ' + result['note'], flush=True)
                if not result['passed']:
                    stop.set()
                    return results
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    check_pin()
    assert not any((OUT / name).exists() for name in ('old', 'new', 'results.json'))
    from servers.scales.dispatch import resolve_api_key
    key = resolve_api_key()
    if not key:
        raise RuntimeError('API key unavailable; no calls made')
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_arm, name, stop, key) for name in ('old', 'new')]
        rows = [row for future in futures for row in future.result()]
    complete = len(rows) == 48 and all(row['passed'] for row in rows)
    save(OUT / 'results.json', {'status': 'passed' if complete else 'stopped',
                              'completed_calls': len(rows), 'samples': rows})
    print('COMPLETE' if complete else 'STOPPED', len(rows), 'calls', flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
