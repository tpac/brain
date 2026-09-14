"""Fixture-local sequential runner for the V3.3 cells.

The same replay as the frozen V3.1 cell's run_sequence — catalog built from the
brain's own prior writes, real local dispatch, real journal harvest and render
on the conversation clock, exact request capture — with two additions that the
frozen file cannot take without breaking the pins of completed cells: a
per-arm preamble (the production package carries its own) and retention of the
final isolated databases for the downstream retrieved-subset test.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval'), str(ROOT / 'tests'),
                str(ROOT / 'eval/fixtures/s1e_guide_context_v2_2026-09-08')]
from context_support import ReplayJournal  # noqa: E402
from s1e_guide_v2_sequence_probe import result_ids, PREAMBLE  # noqa: E402


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write('\n')


def pairs(messages):
    if len(messages) % 2:
        raise ValueError('Unpaired source dialogue')
    out = []
    for i in range(0, len(messages), 2):
        a, b = messages[i:i + 2]
        if a['role'] != 'user' or b['role'] != 'assistant':
            raise ValueError('Source roles not alternating')
        out.append({'other': a['content'], 'me': b['content']})
    return out


def render_turn(turn_number, turn, counterpart):
    who = (' speaker="' + escape(counterpart) + '"') if counterpart else ''
    return (f'<turn n="{turn_number}" age="just now" encoded="false">\n'
            f'<other{who} trace="{turn["other_trace"]}">{escape(turn["other"])}</other>\n'
            f'<me trace="{turn["me_trace"]}">{escape(turn["me"])}</me>\n</turn>')


def body_for(continuity, catalog, gist, now, rendered):
    return ('<continuity>\n' + escape(continuity) + '\n</continuity>\n\n<node_catalog>\n' + catalog
            + '\n</node_catalog>\n\n' + gist.rstrip() + '\n\n<timeline now="' + now + '">\n'
            + '\n\n'.join(rendered) + '\n</timeline>')


def run_sequence(frozen, arm_name, repeat, corpus, out, seed_dir, dry_run=False, keep_db=True):
    from isolated_brain import IsolatedBrain
    from encoder_prompt_ab import _Metered
    from encoder_ops import ops_of, kind
    from eval.longmem.replay import _make_local_dispatch
    from eval.longmem.connect_ab import WRITE_TOOLS
    from servers.scales.s1.encode import _journal
    from servers.scales.s1.encode_contract import build_node_catalog
    from servers.scales.runner import run_llm_loop
    preamble = frozen.get('user_preamble') or PREAMBLE
    fixture = json.loads((Path(out) / (corpus + '.json')).read_text())
    folder = Path(out) / arm_name / ('preflight' if dry_run else 'repeat' + str(repeat)) / corpus
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'arm.json', frozen)
    ids = set(); continuity = ''; turn_number = 0; sid = fixture['session_id']
    env = IsolatedBrain(production_dir=str(seed_dir), cleanup=not keep_db, load_env=True)
    with env:
        brain = env.brain; dispatch_real = _make_local_dispatch(brain)
        journal = ReplayJournal(_journal(brain, sid))
        save(folder / 'environment.json', {'pid': os.getpid(), 'db_dir': env.db_dir, 'arm_sha256': frozen['arm_sha256'],
                                            'source_id': fixture['source_id'], 'preamble': preamble})
        for wn, window in enumerate(fixture['windows'], 1):
            journal.set_window(window['now']); d = folder / ('window' + str(wn)); d.mkdir()
            catalog, _ = build_node_catalog([], brain, extra_ids={'encoded': ids})
            rendered = []
            for turn in window['turns']:
                turn_number += 1
                rendered.append(render_turn(turn_number, turn, fixture['counterpart']))
            body = body_for(continuity, catalog, frozen['gist'], window['now'], rendered)
            (d / 'user.txt').write_text(preamble + '\n\n' + body)
            save(d / 'nodes_before.json', brain.get_node(sorted(ids)) if ids else {})
            if dry_run:
                save(d / 'preflight.json', {'status': 'passed', 'model_calls': 0, 'source_id': fixture['source_id'],
                    'system_chars': len(frozen['system_prompt']), 'body_chars': len(body),
                    'user_sha256': digest(preamble + '\n\n' + body),
                    'turn_counts': [len(w['turns']) for w in fixture['windows']]})
                return
            calls = []; usage = {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}

            def dispatch(cmd, args=None):
                name = str(cmd).split('__')[-1]; args = args or {}
                result = dispatch_real(cmd, args)
                calls.append({'call': {'tool': name, 'args': args}, 'result': result})
                if name in WRITE_TOOLS:
                    ids.update(result_ids(result))
                    ids.update(str(op['node_id']) for op in ops_of({'tool': name, 'args': args}) if kind(op) == 'revise' and op.get('node_id'))
                return result

            def capture(round_number, payload):
                assert payload['system'] == frozen['system_prompt']
                assert payload['tools'] == [t['name'] for t in frozen['tools']]
                save(d / f'round{round_number:03}.json', payload)

            print('START', arm_name, repeat, corpus, wn, flush=True)
            cfg = frozen['settings']
            result = run_llm_loop(client=_Metered(brain._ensure_anthropic_client(), usage), model=cfg['model'],
                effort=cfg['effort'], max_tokens=cfg['max_tokens'], max_rounds=cfg['max_rounds'],
                system_prompt=frozen['system_prompt'], user_content=body, user_preamble=preamble,
                tools=frozen['tools'], dispatch_fn=dispatch, record_round_fn=capture)
            save(d / 'calls.json', calls); save(d / 'result.json', {'result': result, 'usage': usage})
            if result.get('error'):
                raise RuntimeError('Encoder loop failed: ' + str(result['error']))
            journal.harvest(result.get('final_text') or '', f's1e-{sid[:8]}-{wn}')
            continuity = journal.continuity()
            arc = brain.session_context_for(sid)
            if arc:
                continuity += '\nSession arc: ' + arc + '\n'
            (d / 'next_continuity.txt').write_text(continuity)
            save(d / 'nodes_after.json', brain.get_node(sorted(ids)) if ids else {})
            brain.save()
            print('DONE', arm_name, repeat, corpus, wn, 'rounds', result.get('rounds'), 'nodes', len(ids), flush=True)
    if keep_db and not dry_run:
        # the brain is closed by __exit__; copy the whole isolated directory, then remove the temp copy
        kept = folder / 'final_brain'
        shutil.copytree(env.db_dir, kept)
        shutil.rmtree(env.db_dir, ignore_errors=True)
        save(folder / 'final_brain.json', {'dir': str(kept.relative_to(ROOT)), 'node_ids': sorted(ids)})
