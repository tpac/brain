"""Two existing corpora, three arms, three independent sequential repetitions.

Fresh isolated substrate, real writes and journal carry, exact prompt captures.
No S1R/S2 or answerer: this cell measures encoded node quality, not end-to-end
LongMemEval accuracy. Repetitions may overlap only in distinct OS processes.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from xml.sax.saxutils import escape

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'eval/results/s1e_v31_cross_corpus_2026-09-08'
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'eval'),str(ROOT/'tests'),
             str(ROOT/'eval/fixtures/s1e_guide_context_v2_2026-09-08')]
from arms import load_arm, digest
from context_support import ReplayJournal
from s1e_guide_v2_sequence_probe import result_ids, PREAMBLE

ARMS=('v3_titles','v3_titles_new_tools','v3_1_titles')
CORPORA=('creative_design','longmem_unseen')
EXCLUDED={'54026fce','fca762bc','2311e44b','bc149d6b','71017276','gpt4_b0863698',
          'cc5ded98','59524333','09ba9854_abs','edced276_abs'}


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,default=str); f.write('\n')


def pairs(messages):
    if len(messages)%2: raise ValueError('Unpaired source dialogue')
    out=[]
    for i in range(0,len(messages),2):
        a,b=messages[i:i+2]
        if a['role']!='user' or b['role']!='assistant': raise ValueError('Source roles not alternating')
        out.append({'other':a['content'],'me':b['content']})
    return out


def prepare():
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    from servers.brain_traces import _s0_trace
    from servers.scales.s1.encode_contract import build_node_catalog
    for arm in ARMS: load_arm(arm)
    OUT.mkdir(parents=True,exist_ok=False)
    src_design=ROOT/'eval/corpus/conv_004_art_design_extended.json'
    src_longmem=ROOT/'eval/longmem/data/longmemeval_oracle.json'
    design=json.loads(src_design.read_text())
    turns=pairs(design['exchanges'])
    assert len(turns)==15
    eligible=[]
    for item in json.loads(src_longmem.read_text()):
        if item['question_id'] in EXCLUDED: continue
        sessions=item['haystack_sessions']
        if len(sessions)!=3: continue
        counts=[sum(t['role']=='user' for t in ss) for ss in sessions]
        if min(counts)>0 and max(counts)<=5 and sum(counts)<=15:
            eligible.append(item)
    eligible.sort(key=lambda x:digest('s1e-v31-holdout:'+x['question_id']))
    item=eligible[0]
    fixtures={
      'creative_design':{'source_id':design['id'],'source_kind':'existing repository synthetic conversation',
          'clock':'No source dates; fixed synthetic date March 25 2026, one hour between encoding cuts.',
          'counterpart':'Tom','windows':[{'now':f'2026-03-25 {12+i:02}:00 UTC','turns':turns[i*5:(i+1)*5]} for i in range(3)],
          'prior_gold':design['ground_truth']},
      'longmem_unseen':{'source_id':item['question_id'],'source_kind':'existing LongMemEval oracle item outside the prior ten-item slice',
          'clock':'Original source dates; timezone unspecified in source, rendered as UTC consistently.',
          'counterpart':None,'windows':[{'now':datetime.strptime(date,'%Y/%m/%d (%a) %H:%M').strftime('%Y-%m-%d %H:%M UTC'),
              'turns':pairs(session)} for date,session in zip(item['haystack_dates'],item['haystack_sessions'])],
          'prior_gold':{k:item[k] for k in ('question','answer','question_date','question_type')}}}
    seed=OUT/'seed_baseline'
    seed.mkdir()
    shutil.copy2(ROOT/'servers/scales/s2/aspects_v1.json',seed/'aspects_v1.json')
    os.environ['ASPECTS_JSON_PATH']=str(seed/'aspects_v1.json')
    brain=create_fresh_eval_brain(str(seed),wipe=False)
    for name,fixture in fixtures.items():
        fixture['session_id']=digest('s1e-v31:'+name)[:8]+'-cross-corpus'
        ctx=brain.get_or_create_session(fixture['session_id'])
        for window in fixture['windows']:
            for turn in window['turns']:
                ctx.stop_counter+=1
                for role,ref_type,event in [('other','user_message','K'),('me','assistant_message','delta')]:
                    turn[role+'_trace']=_s0_trace(brain,ctx,event,ref_type,turn[role][:200],content=turn[role])
        save(OUT/(name+'.json'),fixture)
    brain.save(); brain.close()
    files=[Path(__file__),HERE/'arms.py',HERE/'manifest.json',
           ROOT/'eval/fixtures/s1e_guide_context_v2_2026-09-08/context_support.py',
           ROOT/'eval/s1e_guide_v2_sequence_probe.py',src_design,src_longmem,
           OUT/'creative_design.json',OUT/'longmem_unseen.json',
           seed/'brain.db',seed/'brain_logs.db',seed/'aspects_v1.json']
    save(OUT/'manifest.json',{'status':'pinned_before_model_calls','arms':{a:load_arm(a)['arm_sha256'] for a in ARMS},
        'corpora':{n:{'source_id':f['source_id'],'turn_counts':[len(w['turns']) for w in f['windows']]} for n,f in fixtures.items()},
        'selection':{'longmem_eligible':len(eligible),'rule':'3 complete source sessions, at most5 user/assistant pairs each, at most15 total, outside previous10; hash sort s1e-v31-holdout:qid',
                     'creative':'existing 15-turn extended design conversation; no dialogue edits'},
        'repeats':3,'total_encodes':54,'parallelism':'up to6 independent repetition processes; windows sequential',
        'files':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in files},
        'limits':['encode-only; no S1R/S2/answerer','small samples of two corpora, not full benchmark scores',
                  'all source traces preseeded for stable refs, only current window shown, no episode-search tool',
                  'all prior created/touched nodes visible in next window catalog; not a missing-catalog reach test',
                  'source-selection and prompt freeze precede outputs; no prompt changes from this cell']})
    print('PREPARED: 54 encodes, corpus ids',[(k,v['source_id']) for k,v in fixtures.items()],flush=True)


def check_pin():
    pin=json.loads((OUT/'manifest.json').read_text())
    for path,expected in pin['files'].items():
        assert digest((ROOT/path).read_bytes())==expected,'Changed cell input: '+path
    for arm,expected in pin['arms'].items(): assert load_arm(arm)['arm_sha256']==expected
    return pin


def run_sequence(arm,repeat,corpus,dry_run=False):
    from isolated_brain import IsolatedBrain
    from encoder_prompt_ab import _Metered
    from encoder_ops import ops_of,kind
    from eval.longmem.replay import _make_local_dispatch
    from eval.longmem.connect_ab import WRITE_TOOLS
    from servers.scales.s1.encode import _journal
    from servers.scales.s1.encode_contract import build_node_catalog
    from servers.scales.runner import run_llm_loop
    check_pin(); frozen=load_arm(arm)
    fixture=json.loads((OUT/(corpus+'.json')).read_text())
    folder=OUT/arm/('preflight' if dry_run else 'repeat'+str(repeat))/corpus
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'arm.json',frozen)
    ids=set(); continuity=''; turn_number=0; sid=fixture['session_id']
    with IsolatedBrain(production_dir=str(OUT/'seed_baseline'),cleanup=True,load_env=True) as env:
        brain=env.brain; dispatch_real=_make_local_dispatch(brain)
        journal=ReplayJournal(_journal(brain,sid))
        save(folder/'environment.json',{'pid':os.getpid(),'db_dir':env.db_dir,'arm_sha256':frozen['arm_sha256'],'source_id':fixture['source_id']})
        for wn,window in enumerate(fixture['windows'],1):
            journal.set_window(window['now']); d=folder/('window'+str(wn)); d.mkdir()
            catalog,_=build_node_catalog([],brain,extra_ids={'encoded':ids})
            rendered=[]
            for turn in window['turns']:
                turn_number+=1
                who=(' speaker="'+escape(fixture['counterpart'])+'"') if fixture['counterpart'] else ''
                rendered.append(f'<turn n="{turn_number}" age="just now" encoded="false">\n'
                    f'<other{who} trace="{turn["other_trace"]}">{escape(turn["other"])}</other>\n'
                    f'<me trace="{turn["me_trace"]}">{escape(turn["me"])}</me>\n</turn>')
            body='<continuity>\n'+escape(continuity)+'\n</continuity>\n\n<node_catalog>\n'+catalog+'\n</node_catalog>\n\n'+frozen['gist'].rstrip()+'\n\n<timeline now="'+window['now']+'">\n'+'\n\n'.join(rendered)+'\n</timeline>'
            (d/'user.txt').write_text(PREAMBLE+'\n\n'+body)
            save(d/'nodes_before.json',brain.get_node(sorted(ids)) if ids else {})
            if dry_run:
                save(d/'preflight.json',{'status':'passed','model_calls':0,'source_id':fixture['source_id'],
                    'system_chars':len(frozen['system_prompt']),'body_chars':len(body),'user_sha256':digest(PREAMBLE+'\n\n'+body),
                    'guide_removed_body_sha256':digest(body.replace(frozen['gist'].rstrip(),'<GIST>')),
                    'turn_counts':[len(w['turns']) for w in fixture['windows']]})
                return
            calls=[]; usage={'input':0,'output':0,'cache_read':0,'cache_write':0}
            def dispatch(cmd,args=None):
                name=str(cmd).split('__')[-1]; args=args or {}
                result=dispatch_real(cmd,args)
                calls.append({'call':{'tool':name,'args':args},'result':result})
                if name in WRITE_TOOLS:
                    ids.update(result_ids(result))
                    ids.update(str(op['node_id']) for op in ops_of({'tool':name,'args':args}) if kind(op)=='revise' and op.get('node_id'))
                return result
            def capture(round_number,payload):
                assert payload['system']==frozen['system_prompt']
                assert payload['tools']==[t['name'] for t in frozen['tools']]
                save(d/f'round{round_number:03}.json',payload)
            print('START',arm,repeat,corpus,wn,flush=True)
            cfg=frozen['settings']
            result=run_llm_loop(client=_Metered(brain._ensure_anthropic_client(),usage),model=cfg['model'],
                effort=cfg['effort'],max_tokens=cfg['max_tokens'],max_rounds=cfg['max_rounds'],
                system_prompt=frozen['system_prompt'],user_content=body,user_preamble=PREAMBLE,
                tools=frozen['tools'],dispatch_fn=dispatch,record_round_fn=capture)
            save(d/'calls.json',calls); save(d/'result.json',{'result':result,'usage':usage})
            if result.get('error'): raise RuntimeError('Encoder loop failed: '+str(result['error']))
            journal.harvest(result.get('final_text') or '',f's1e-{sid[:8]}-{wn}')
            continuity=journal.continuity()
            arc=brain.session_context_for(sid)
            if arc: continuity+='\nSession arc: '+arc+'\n'
            (d/'next_continuity.txt').write_text(continuity)
            save(d/'nodes_after.json',brain.get_node(sorted(ids)) if ids else {})
            brain.save()
            print('DONE',arm,repeat,corpus,wn,'rounds',result.get('rounds'),'nodes',len(ids),flush=True)


def launch():
    check_pin()
    for corpus in CORPORA:
        pre=[json.loads((OUT/a/'preflight'/corpus/'window1/preflight.json').read_text()) for a in ARMS]
        assert len({p['guide_removed_body_sha256'] for p in pre})==1,'Different factual inputs'
    jobs=[(a,r) for r in range(1,4) for a in ARMS]
    live=[]; completed=[]
    try:
        while jobs or live:
            while jobs and len(live)<6:
                arm,repeat=jobs.pop(0)
                log=OUT/f'{arm}_repeat{repeat}.log'; stream=log.open('x')
                cwd=OUT/'process_dirs'/arm/('repeat'+str(repeat)); cwd.mkdir(parents=True,exist_ok=False)
                process=subprocess.Popen([sys.executable,str(Path(__file__)),'--arm',arm,'--repeat',str(repeat)],
                    cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                live.append((process,stream,arm,repeat)); print('LAUNCHED',arm,repeat,flush=True)
            for job in live[:]:
                process,stream,arm,repeat=job
                if process.poll() is not None:
                    stream.close(); live.remove(job)
                    if process.returncode: raise RuntimeError(f'Repetition failed: {arm} {repeat}; see log')
                    completed.append([arm,repeat]); print('REPETITION COMPLETE',arm,repeat,flush=True)
            if live: time.sleep(0.5)
    finally:
        import signal
        for process,stream,_,_ in live:
            if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
        for process,stream,_,_ in live: process.wait(); stream.close()
    save(OUT/'completion.json',{'status':'complete','repetitions':completed,'encodes':54})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=ARMS); p.add_argument('--repeat',type=int); args=p.parse_args()
    if args.prepare: prepare()
    elif args.preflight:
        for arm in ARMS:
            for corpus in CORPORA: run_sequence(arm,0,corpus,True)
        print('ALL SIX PREFLIGHTS COMPLETE',flush=True)
    elif args.arm:
        for corpus in CORPORA: run_sequence(args.arm,args.repeat,corpus)
    else: launch()
