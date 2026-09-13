import sys, json, hashlib, os
from pathlib import Path
WT=Path('/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242')
for p in (WT, WT/'tests', WT/'eval'): sys.path.insert(0,str(p))
os.chdir(WT)
from servers.scales import runner
OUT=WT/'eval/results/s1e_guide_v2_2026-09-08/gold_bb5b1ef4'
TPL=(WT/'eval/candidate_prompts/s1e_guide_v2_2026-09-08.md').read_text()
GIST=(WT/'eval/candidate_prompts/s1e_gist_guide_v2_2026-09-08.md').read_text().rstrip('\n')
assert hashlib.sha256(TPL.encode()).hexdigest()=='4637785918cab686ec9abd29096def0691fb9aaf443b63d893736c4883551edb'
original=runner.run_llm_loop
counter=0
def captured(**kw):
 global counter
 counter+=1
 assert kw['system_prompt'].startswith(TPL)
 assert GIST in kw['user_content']
 assert 'four lists — changes, targets, fetch, new' in kw['user_content'].split('<continuity>')[0]
 d=OUT/f'capture_run{counter}'; d.mkdir(exist_ok=False)
 (d/'tools.json').write_text(json.dumps(kw['tools'],indent=2))
 (d/'manifest.json').write_text(json.dumps({'template_sha256':hashlib.sha256(TPL.encode()).hexdigest(),'gist_sha256':hashlib.sha256(GIST.encode()).hexdigest(),'model':kw['model'],'effort':kw.get('effort'),'system_chars':len(kw['system_prompt']),'user_chars':len(kw['user_content']),'lists_first_verified':True,'writes':'intercepted with synthetic success, same historical gold harness','reads':'isolated copy; live drift can void item'},indent=2))
 def record(r,payload):
  (d/f'round{r:03}.json').write_text(json.dumps(payload,indent=2,default=lambda o:o.model_dump() if hasattr(o,'model_dump') else str(o)))
 kw['record_round_fn']=record
 print('COMPOSITION VERIFIED',counter,flush=True)
 return original(**kw)
runner.run_llm_loop=captured
from encoder_prompt_ab import main
sys.argv=['encoder_prompt_ab.py','/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/payloads_patched_guide/2026-08-31/s1e-5076cdc2-17/000-prompt.md','--arms','F','--s1e-template','eval/candidate_prompts/s1e_guide_v2_2026-09-08.md','--gist-file','eval/candidate_prompts/s1e_gist_guide_v2_2026-09-08.md','--behavior','--repeat','2','--gold','eval/ground_truth/s1e_fieldcov_bb5b1ef4.json','--dump-ops',str(OUT)]
main()
