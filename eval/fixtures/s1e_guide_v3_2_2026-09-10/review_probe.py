"""Bounded review ablation of saved V3.2 output; never dispatches brain tools.

The existing fixture owns this diagnostic. Prepare freezes exact API requests;
run makes at most six calls, with no retries, new encodes, or database access.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SAVED = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'
WINDOW = SAVED / 'v3_2_titles/repeat2/creative_design/window3'
OUT = ROOT / 'eval/results/s1e_v32_review_probe_2026-09-11'
CONDITIONS = ('with_worklist', 'without_worklist')
QUESTION = """Pause encoding for a diagnostic review. Do not call tools or write an Arc/Review journal. Compare the claims in the preceding successful batch with the actual conversation timeline above. Review these five topics in this order:
1. The later response to the mirror/camera principle.
2. The aesthetic-product formulation and the aesthetic priorities.
3. The ambient-presence vision and its proposed concrete features.
4. The first prototype's selected scope and priority.
5. The proposed prototype library and implementation mappings.

For each topic, identify the relevant node and field(s), quote the decisive source wording with its speaker and turn, and say whether the stored claims should remain or change. If a change is warranted, give the smallest faithful replacement and identify any other fields or edges affected. Preserve supported specificity. Report uncertainty where the visible evidence cannot settle it. Do not assume a fixed number of errors. Base the verdict on the source, rather than on a previous summary or a successful tool result. Keep the entire answer under 950 words.
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def request(condition):
    captured = json.loads((WINDOW / 'round001.json').read_text())
    messages = deepcopy(captured['messages'])
    if condition == 'without_worklist':
        blocks = messages[1]['content']
        messages[1]['content'] = [b for b in blocks if b['type'] != 'text']
    messages[-1]['content'].append({'type': 'text', 'text': QUESTION})
    return {
        'model': captured['model'], 'max_tokens': 2400,
        'output_config': {'effort': captured['effort']},
        'system': [{'type': 'text', 'text': captured['system'],
                    'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
        'messages': messages, 'tools': captured['tools'],
        'tool_choice': {'type': 'none'},
    }


def prepare():
    import importlib.util
    spec = importlib.util.spec_from_file_location('_v32_cell', HERE / 'corpus_cell.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    parent.check_pin()
    captured = json.loads((WINDOW / 'round001.json').read_text())
    if [m['role'] for m in captured['messages']] != ['user', 'assistant', 'user']:
        raise ValueError('Unexpected saved conversation structure')
    blocks = captured['messages'][1]['content']
    if [b['type'] for b in blocks] != ['text', 'tool_use']:
        raise ValueError('Expected one worklist followed by the original batch')
    if not blocks[0]['text'].startswith('changes:'):
        raise ValueError('Removed text is not the saved worklist')
    full, stripped = (request(c) for c in CONDITIONS)
    normalized = deepcopy(full)
    normalized['messages'][1]['content'] = normalized['messages'][1]['content'][1:]
    if normalized != stripped:
        raise ValueError('Condition difference exceeds worklist removal')
    if stripped['messages'][1]['content'][0] != blocks[1]:
        raise ValueError('Original tool call changed')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'question.txt').write_text(QUESTION)
    (OUT / 'removed_worklist.txt').write_text(blocks[0]['text'])
    files = [Path(__file__), SAVED / 'manifest.json']
    files += [WINDOW / name for name in ('round001.json', 'result.json', 'calls.json',
                                       'nodes_before.json', 'nodes_after.json', 'user.txt')]
    for condition in CONDITIONS:
        p = OUT / (condition + '.request.json')
        save(p, request(condition))
        files.append(p)
    files += [OUT / 'question.txt', OUT / 'removed_worklist.txt']
    save(OUT / 'manifest.json', {
        'status': 'frozen_before_calls', 'calls': 6, 'repeats': 3,
        'max_output_tokens_per_call': 2400, 'max_retries': 0,
        'source': str(WINDOW.relative_to(ROOT)),
        'intervention': 'Remove only the original 5373-character assistant worklist.',
        'shared_change': 'Append identical focused source-review question; disable tools.',
        'limits': [
            'Targeted review ability is not spontaneous encode/repair performance.',
            'Incorrect tool arguments and prior catalog remain in both conditions.',
            'Removing text also shortens context; anchoring and length are not separated.',
            'Known development errors; no holdout, transfer, or benchmark claim.',
            'Three independent samples per condition, not matched random seeds.',
        ],
        'preflight': {'parent_pins_valid': True, 'only_condition_difference_is_worklist': True,
                      'saved_tool_call_unchanged': True, 'database_access': False},
        'files': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
    })
    print('Prepared six read-only reviews; no model calls made.', flush=True)


def check():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError('Frozen review input changed: ' + relative)


def run_repeat(repeat):
    from servers.scales.runner import make_client
    client = make_client().with_options(max_retries=0, timeout=180.0)
    order = CONDITIONS if repeat != 2 else tuple(reversed(CONDITIONS))
    try:
        for condition in order:
            folder = OUT / condition / ('repeat' + str(repeat))
            folder.mkdir(parents=True, exist_ok=False)
            payload = json.loads((OUT / (condition + '.request.json')).read_text())
            save(folder / 'started.json', {'condition': condition, 'repeat': repeat})
            start = time.monotonic()
            try:
                with client.messages.stream(**payload) as stream:
                    response = stream.get_final_message()
                save(folder / 'response.json', response.model_dump(mode='json'))
                answer = '\n'.join(b.text for b in response.content if b.type == 'text')
                (folder / 'answer.md').write_text(answer + '\n')
                save(folder / 'completion.json', {
                    'elapsed_seconds': round(time.monotonic() - start, 2),
                    'stop_reason': response.stop_reason,
                    'tool_calls': sum(b.type == 'tool_use' for b in response.content),
                })
                if response.stop_reason != 'end_turn':
                    raise RuntimeError('Review did not finish normally: ' + str(response.stop_reason))
                print(f'{condition} repeat{repeat}: complete, {response.usage.output_tokens} output tokens', flush=True)
            except Exception as exc:
                save(folder / 'error.json', {'type': type(exc).__name__, 'message': str(exc)})
                raise
    finally:
        client.close()


def run():
    check()
    from eval.agent_introspect._common import load_env
    load_env()
    save(OUT / 'launch.json', {'calls_cap': 6, 'retries': 0, 'tools_dispatched': 0})
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_repeat, repeat) for repeat in (1, 2, 3)]
        for future in as_completed(futures):
            future.result()
    check()
    save(OUT / 'completion.json', {'status': 'complete', 'calls': 6, 'tools_dispatched': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    args = parser.parse_args()
    {'prepare': prepare, 'check': check, 'run': run}[args.action]()
