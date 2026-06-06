#!/usr/bin/env python3
"""RRF stage-1 fusion A/B — baseline (current blend) vs BRAIN_RRF_FUSION=1, on an
ISOLATED brain copy, over the oracle-audit corpus. See docs/RECALL-HYBRID-FUSION-DESIGN.md.
Never touches the live brain (IsolatedBrain copies to a temp dir).
Usage: ./dev python3 eval/oracle_audit/rrf_ab.py
"""
import os, json, sys
ROOT = '/Users/tpac/brain/.claude/worktrees/frosty-feistel-90c7a9'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
# Known EX.CO / ad-tech node ids (from the oracle + sweeps) — the "should surface" set.
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}
EXCO_SRCS = {'EXCO-recall', 'B1-20'}  # swamp-tests; everything else is a brain-dev control

def ids_of(result):
    res = result.get('results', []) if isinstance(result, dict) else (result or [])
    out = []
    for r in res:
        out.append(r.get('id') or r.get('node_id'))
    return out

def run_arm(brain, flag):
    os.environ['BRAIN_RRF_FUSION'] = flag
    out = {}
    for it in CORPUS:
        if hasattr(brain, '_recall_cache'):
            brain._recall_cache.clear()  # bust the 10s recall cache so the flag actually takes effect
        out[it['rank']] = ids_of(brain.recall(query=it['prompt'], limit=25))
    return out

def exco_rank(ids):
    for i, nid in enumerate(ids, 1):
        if nid in KNOWN_EXCO:
            return i
    return None

with IsolatedBrain() as env:
    base = run_arm(env.brain, '0')
    rrf  = run_arm(env.brain, '1')

print("\n=== RRF stage-1 A/B (isolated copy) ===")
print(f"{'#':>2} {'src':<12} {'kind':<5} {'exco@base':>9} {'exco@rrf':>8} {'top5Δ':>6}  n_base/n_rrf")
for it in CORPUS:
    k, src = it['rank'], it['src']
    kind = 'EXCO' if src in EXCO_SRCS else 'ctrl'
    br, rr = exco_rank(base[k]), exco_rank(rrf[k])
    overlap = len(set(base[k][:5]) & set(rrf[k][:5]))
    print(f"{k:>2} {src:<12} {kind:<5} {str(br or '-'):>9} {str(rr or '-'):>8} {overlap:>4}/5  {len(base[k])}/{len(rrf[k])}")

print("\n=== EX.CO query detail (top-6 ids each arm) ===")
for it in CORPUS:
    if it['src'] not in EXCO_SRCS:
        continue
    k = it['rank']
    print(f"[{k}] {it['prompt'][:60]}")
    print(f"    base: {base[k][:6]}")
    print(f"    rrf : {rrf[k][:6]}")

n_exco = sum(1 for it in CORPUS if it['src'] in EXCO_SRCS)
base_hit = sum(1 for it in CORPUS if it['src'] in EXCO_SRCS and exco_rank(base[it['rank']]))
rrf_hit  = sum(1 for it in CORPUS if it['src'] in EXCO_SRCS and exco_rank(rrf[it['rank']]))
ctrl_overlap = [len(set(base[it['rank']][:5]) & set(rrf[it['rank']][:5])) for it in CORPUS if it['src'] not in EXCO_SRCS]
print(f"\nSUMMARY: EX.CO surfaced — base {base_hit}/{n_exco}  →  rrf {rrf_hit}/{n_exco}")
print(f"         control top-5 overlap (base vs rrf): {ctrl_overlap}  (5=no change)")
