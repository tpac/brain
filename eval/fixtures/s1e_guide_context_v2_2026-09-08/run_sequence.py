"""Sequential-window diagnostic. Real branch runner/dispatch/journal; isolated writes.

--prepare pins one baseline from IsolatedBrain; --arm uses sequential
copies of that baseline. Concurrent arms require separate OS processes.
Catalog background is frozen historical input, not a semantic retrieval test.
Inspect persisted nodes and per-op results against the fixture's prior criteria.
No automatic semantic score, synthetic success, corrective nudge, or interview.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from context_support import ReplayJournal

WT = Path(__file__).resolve().parents[3]
for p in (WT, WT / 'tests', WT / 'eval'):
    sys.path.insert(0, str(p))
OUT = WT / 'eval/results/s1e_guide_context_v2_2026-09-08'
FIXTURE = Path(__file__).resolve().parent / 'contrasts_three_windows.json'
CAP = Path('/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/payloads_patched_guide/2026-08-31/s1e-5076cdc2-17/000-prompt.md')
SID = 'e42ad630-context-v2-sequence'
PREAMBLE = "I'm encoding what I've just observed. I read everything below; my first reply opens with my four lists — changes, targets, fetch, new — and ends in the tool call they call for.\n"
SOURCE_BASELINE = None
RUN_ID = None
FROZEN_ARM = None
DRY_RUN = False
ARM_FILES = {
    'v1': 's1e_guide_v1_2026-09-08.md',
    'v2': 's1e_guide_v2_2026-09-08.md',
    'shapes': 's1e_guide_v2_compact_shapes_2026-09-08.md',
    'episode': 's1e_guide_v2_compact_episode_2026-09-08.md',
}
EXPECTED = {
    'v2': '4637785918cab686ec9abd29096def0691fb9aaf443b63d893736c4883551edb',
    'shapes': 'ac1417f95e85de0f8af15542bccfe327dbc9122aa6c9a81de7b6334a664c7448',
    'episode': '8f2e7d9b19df96cfc4b1b62dfe12fbe19b0c094e573d4375dfe3a717130df1ee',
}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.model_dump() if hasattr(x, 'model_dump') else str(x)))


def prepare():
    from isolated_brain import IsolatedBrain
    from servers.brain_traces import _s0_trace
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'baseline.json'
    if manifest.exists():
        raise SystemExit('Baseline already pinned; refusing to replace it')
    fixture = json.loads(FIXTURE.read_text())
    background = re.search(r'<node_catalog>\n(.*?)\n</node_catalog>', CAP.read_text(), re.S).group(1)
    (OUT / 'background_catalog.txt').write_text(background)
    with IsolatedBrain(production_dir=SOURCE_BASELINE, cleanup=False, load_env=False) as env:
        brain = env.brain
        ctx = brain.get_or_create_session(SID)
        seeds = {}
        for row in fixture['seed_nodes']:
            args = {k: v for k, v in row.items() if k != 'key'}
            r = brain.remember(**args)
            if not r.get('id'):
                raise RuntimeError('seed remember failed: %r' % r)
            seeds[row['key']] = r['id']
        for w in fixture['windows']:
            for t in w['turns']:
                ctx.stop_counter += 1
                for role in ('other', 'me'):
                    ref_type = 'user_message' if role == 'other' else 'assistant_message'
                    t[role + '_trace'] = _s0_trace(brain, ctx, 'K' if role == 'other' else 'delta', ref_type,
                                                   t[role][:200], content=t[role])
                if t.get('actions'):
                    t['actions_trace'] = _s0_trace(brain, ctx, 'delta', 'tool_result', t['actions'], content=t['actions'])
        brain.save()
        pinned = {'baseline_dir': env.db_dir, 'seed_ids': seeds,
                  'session_id': SID,
                  'source_baseline': SOURCE_BASELINE,
                  'fixture_sha256': sha(FIXTURE.read_text()),
                  'background_sha256': sha(background), 'fixture': fixture}
    save(manifest, pinned)
    print('PREPARED', manifest, 'baseline closed; model not called', flush=True)


def result_ids(value):
    """Successful operation ids only; related-node suggestions are not writes."""
    found = set()
    if isinstance(value, list):
        for item in value:
            found |= result_ids(item)
    elif isinstance(value, dict):
        if value.get('ok') is False or value.get('error'):
            return found
        if isinstance(value.get('id'), str):
            found.add(value['id'])
        for key in ('result', 'results'):
            if key in value:
                found |= result_ids(value[key])
    return found


def run(arm):
    from isolated_brain import IsolatedBrain
    from encoder_prompt_ab import _Metered
    from encoder_ops import ops_of, kind
    from eval.longmem.replay import _make_local_dispatch
    from eval.longmem.connect_ab import WRITE_TOOLS
    from servers.scales.s1.encode import _build_system_prompt, _get_tool_schemas, _journal
    from servers.scales.s1.encode_contract import ENCODING_AGENT, build_node_catalog
    from servers.scales.runner import run_llm_loop
    pinned = json.loads((OUT / 'baseline.json').read_text())
    assert sha(FIXTURE.read_text()) == pinned['fixture_sha256']
    assert SID == pinned.get('session_id', 'd15c0a72-v2-fresh-sequence')
    folder = OUT / arm
    if RUN_ID is not None:
        folder = folder / RUN_ID
    folder.mkdir(parents=True, exist_ok=False)
    frozen = None
    if FROZEN_ARM:
        sys.path.insert(0, str(WT / 'eval/fixtures/s1e_guide_freeze_2026-09-08'))
        from frozen_arms import load_arm, manifest
        frozen = load_arm(FROZEN_ARM)
        info = manifest()['arms'][FROZEN_ARM]
        tpl = (WT / info['template']).read_text()
        gist_raw = frozen['gist']
    else:
        tpl = (WT / 'eval/candidate_prompts' / ARM_FILES[arm]).read_text()
        gist_arm = 'v1' if arm == 'v1' else 'v2'
        gist_raw = (WT / f'eval/candidate_prompts/s1e_gist_guide_{gist_arm}_2026-09-08.md').read_text()
        if gist_arm == 'v2':
            assert sha(gist_raw) == '54cf9c77132ce3eac680f9703f71a612293b26117c6886048ba9f001634b3063'
        if arm in EXPECTED:
            assert sha(tpl) == EXPECTED[arm]
    background = (OUT / 'background_catalog.txt').read_text()
    assert sha(background) == pinned['background_sha256']
    with IsolatedBrain(production_dir=pinned['baseline_dir'], cleanup=True, load_env=True) as env:
        brain = env.brain
        assert Path(env.db_dir) != Path(pinned['baseline_dir'])
        cfg = brain.get_interaction_config('s1e') or {}
        journal = ReplayJournal(_journal(brain, SID))
        system = frozen['system_prompt'] if frozen else _build_system_prompt(prompt_instructions=tpl, lived=True, journal=journal)
        assert system.startswith(tpl)
        tools = _get_tool_schemas()
        if frozen:
            assert tools == frozen['tools']
            for key in ('model', 'effort'):
                assert cfg.get(key) == frozen['settings'][key], key+' config drift'
            for key in ('max_tokens', 'max_rounds'):
                assert ENCODING_AGENT[key] == frozen['settings'][key], key+' limit drift'
            assert os.environ.get('BRAIN_S1E_LISTS_PREAMBLE') == '1'
        fixed = {
            'gist_sha256':sha(gist_raw),
            'suffix_sha256':sha(_build_system_prompt(prompt_instructions=tpl, lived=True, journal=journal)[len(tpl):]),
            'tools_sha256':sha(json.dumps(tools,sort_keys=True)),
            'model':cfg.get('model') or 'claude-sonnet-4-6',
            'effort':cfg.get('effort'),
            'max_tokens':ENCODING_AGENT['max_tokens'],
            'max_rounds':ENCODING_AGENT['max_rounds'],
        }
        if RUN_ID is not None:
            contract_path = OUT / 'comparison_contract.json'
            if contract_path.exists():
                assert json.loads(contract_path.read_text()) == fixed, 'Comparison composition drift'
            else:
                save(contract_path, fixed)
        save(folder / 'tools.json', tools)
        (folder / 'system.txt').write_text(system)
        save(folder / 'manifest.json', {
            'arm':arm, 'run_id':RUN_ID, 'session_id':SID,
            'arm_sha256': frozen['arm_sha256'] if frozen else None,
            'runner_sha256': sha(Path(__file__).read_text()),
            'dry_run': DRY_RUN,
            'template_sha256':sha(tpl), 'gist_sha256':sha(gist_raw),
            'fixture_sha256':pinned['fixture_sha256'], 'background_sha256':sha(background),
            'baseline_seed_ids':pinned['seed_ids'], 'system_chars':len(system),
            'model':cfg.get('model') or 'claude-sonnet-4-6', 'effort':cfg.get('effort'),
            'dispatch':'real, isolated copy only', 'continuity':'real JournalBinding lifecycle; displayed first_seen dates mapped to the producing fixture window',
            'limits':'constructed context and covered catalog; no S1R or S2, no benchmark score; same shown background in each arm'})
        ids = set(pinned['seed_ids'].values())
        real = _make_local_dispatch(brain)
        continuity = pinned['fixture']['initial_residue']
        turn_number = 0
        for wn, window in enumerate(pinned['fixture']['windows'], 1):
            journal.set_window(window['now'])
            d = folder / f'window{wn}'
            d.mkdir()
            cat, _ = build_node_catalog([], brain, extra_ids={'encoded':set(ids)})
            turns = []
            for turn in window['turns']:
                turn_number += 1
                parts = [f'<turn n="{turn_number}" age="just now" encoded="false">']
                parts.append(f'  <other speaker="{escape(pinned["fixture"]["other_speaker"])}" trace="{turn["other_trace"]}">{escape(turn["other"])}</other>')
                if turn.get('actions'):
                    parts.append(f'  <actions>{escape(turn["actions"])}</actions>')
                parts.append(f'  <me trace="{turn["me_trace"]}">{escape(turn["me"])}</me>')
                parts.append('</turn>')
                turns.append('\n'.join(parts))
            body = ('<continuity>\n' + escape(continuity) + '\n</continuity>\n\n<node_catalog>\n'
                    + background + '\n\n' + cat + '\n</node_catalog>\n\n' + gist_raw.rstrip('\n')
                    + '\n\n<timeline now="' + window['now'] + '">\n' + '\n\n'.join(turns) + '\n</timeline>')
            (d / 'user.txt').write_text(PREAMBLE + '\n\n' + body)
            if RUN_ID is not None and wn == 1:
                first_path = OUT / 'first_user_sha256.txt'
                user_hash = sha(PREAMBLE + '\n\n' + body)
                if first_path.exists():
                    assert first_path.read_text() == user_hash, 'First-window input differs across arms'
                else:
                    first_path.write_text(user_hash)
            save(d / 'nodes_before.json', brain.get_node(sorted(ids)))
            if DRY_RUN:
                save(d / 'preflight.json', {'status':'passed', 'model_calls':0,
                     'system_sha256':sha(system), 'user_sha256':sha(PREAMBLE+'\n\n'+body),
                     'windows':len(pinned['fixture']['windows']),
                     'turn_counts':[len(w['turns']) for w in pinned['fixture']['windows']],
                     'arm_sha256':frozen['arm_sha256'] if frozen else None})
                print('PREFLIGHT PASSED',arm,'system',len(system),'body',len(body),flush=True)
                break
            writes, reads, call_results = [], [], []
            usage = {'input':0, 'output':0, 'cache_read':0, 'cache_write':0}
            def dispatch(cmd, args=None):
                name = str(cmd).split('__')[-1]
                args = args or {}
                record = {'tool':name, 'args':args}
                (writes if name in WRITE_TOOLS else reads).append(record)
                result = real(cmd, args)
                call_results.append({'call':record, 'result':result})
                if name in WRITE_TOOLS:
                    ids.update(result_ids(result))
                    ids.update(str(o['node_id']) for o in ops_of(record) if kind(o)=='revise' and o.get('node_id'))
                save(d / 'calls.json', call_results)
                return result
            def capture(r, payload):
                assert payload['system'] == system
                assert payload['model'] == fixed['model']
                assert payload['effort'] == fixed['effort']
                assert payload['tools'] == [tool['name'] for tool in tools]
                save(d / f'round{r:03}.json', payload)
            print('START',arm,'window',wn,'system',len(system),'body',len(body),flush=True)
            result = run_llm_loop(client=_Metered(brain._ensure_anthropic_client(),usage),
                                  model=cfg.get('model') or 'claude-sonnet-4-6', effort=cfg.get('effort') or None,
                                  max_tokens=ENCODING_AGENT['max_tokens'], max_rounds=ENCODING_AGENT['max_rounds'],
                                  system_prompt=system, user_content=body, user_preamble=PREAMBLE,
                                  tools=tools, dispatch_fn=dispatch, record_round_fn=capture)
            save(d / 'result.json', {'result':result,'usage':usage,'writes':writes,'reads':reads})
            if result.get('error'):
                raise RuntimeError('Encoder failed: '+str(result['error']))
            journal.harvest(result.get('final_text') or '', f's1e-{SID[:8]}-{wn}')
            continuity = journal.continuity()
            if not isinstance(continuity,str):
                raise TypeError('journal continuity changed shape: %r' % type(continuity))
            arc = brain.session_context_for(SID)
            if arc:
                continuity += '\nSession arc: ' + arc + '\n'
            save(d / 'result.json', {'result':result,'usage':usage,'writes':writes,'reads':reads})
            save(d / 'nodes_after.json', brain.get_node(sorted(ids)))
            (d / 'next_continuity.txt').write_text(continuity)
            brain.save()
            print('DONE',arm,'window',wn,'rounds',result.get('rounds'),'write calls',len(writes),'read calls',len(reads),flush=True)
    print('SEQUENCE DONE',arm,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--arm',choices=list(ARM_FILES))
    group.add_argument('--frozen-arm',choices=['v2_frozen','v2_revised','v3','v2_revised_titles','v3_titles'])
    parser.add_argument('--fixture',type=Path,default=FIXTURE)
    parser.add_argument('--out-dir',type=Path,default=OUT)
    parser.add_argument('--session-id',default=SID)
    parser.add_argument('--run-id')
    parser.add_argument('--source-baseline')
    parser.add_argument('--dry-run',action='store_true',help='render first input against an isolated copy, no model call')
    args=parser.parse_args()
    FIXTURE = args.fixture.resolve()
    OUT = args.out_dir.resolve()
    SID = args.session_id
    RUN_ID = args.run_id
    SOURCE_BASELINE = args.source_baseline
    FROZEN_ARM = args.frozen_arm
    DRY_RUN = args.dry_run
    if RUN_ID and (Path(RUN_ID).name != RUN_ID or RUN_ID in ('.', '..')):
        parser.error('--run-id must be one directory name')
    if args.prepare:
        prepare()
    else:
        run(args.frozen_arm or args.arm)
