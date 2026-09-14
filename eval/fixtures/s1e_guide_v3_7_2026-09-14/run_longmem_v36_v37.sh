#!/bin/zsh
# Longmem, V3.6 vs V3.7 — the frozen-corpus longmem benchmark, coupled old vs new (Tom, 2026-09-14:
# "run just for fun 3.6 vs 3.7 on some longmem benchmark"). Same ten items as gate 4 and Longmem 9
# (2 per axis), so the numbers sit beside lm9 (prod 22/30 → branch 26/30 → guide 20/30). Both arms
# run the same gist (production's, enabled); only the s1e template differs:
#   v36 = eval/fixtures/s1e_guide_v3_6_2026-09-13/template_full.md (the V3.6 production template)
#   v37 = eval/fixtures/s1e_guide_v3_7_2026-09-14/template_advice.md (== the code default after deploy_defaults --write)
# Both arms are passed as FILES so each corpus is content-addressed by the file's hash — a default `--s1e active`
# addresses by the literal word and can silently reuse an old corpus (the 2026-08-24 cache hazard); --force too.
# The two builds run in parallel (separate eval brains); then a variance-3 sweep per arm with
# --force-preflight (zero-encode items abort a sweep otherwise — id:a0224457), then compare_arms.
set -u
cd /Users/tpac/brain/.claude/worktrees/s1e-v3-7-eval-63d861 || exit 1
QIDS=54026fce,fca762bc,2311e44b,bc149d6b,71017276,gpt4_b0863698,cc5ded98,59524333,09ba9854_abs,edced276_abs
V36=eval/fixtures/s1e_guide_v3_6_2026-09-13/template_full.md
V37=eval/fixtures/s1e_guide_v3_7_2026-09-14/template_advice.md
LOGDIR=eval/results/longmem_v36_v37_2026-09-14
mkdir -p $LOGDIR

echo "=== LONGMEM v36 vs v37 START  $(date -u) ==="
git log --oneline -1

./dev python3 eval/longmem/build_corpus.py --qids $QIDS --s1e $V36 --force --label lm37_v36 > $LOGDIR/build_v36.log 2>&1 &
P36=$!
./dev python3 eval/longmem/build_corpus.py --qids $QIDS --s1e $V37 --force --label lm37_v37 > $LOGDIR/build_v37.log 2>&1 &
P37=$!
wait $P36; echo "exit=$? (build v36)"
wait $P37; echo "exit=$? (build v37)"
H36=$(grep -o 'config hash = [0-9a-f]*' $LOGDIR/build_v36.log | head -1 | awk '{print $4}')
H37=$(grep -o 'config hash = [0-9a-f]*' $LOGDIR/build_v37.log | head -1 | awk '{print $4}')
echo "H36=$H36 H37=$H37"

for arm in v36 v37; do
  H=$([ $arm = v36 ] && echo $H36 || echo $H37)
  echo "=== sweep $arm corpus=$H variance=3  $(date -u) ==="
  ./dev python3 eval/longmem/sweep.py --corpus $H --variance 3 --force-preflight --label lm37_${arm}_sweep > $LOGDIR/sweep_$arm.log 2>&1
  echo "exit=$? (sweep $arm)"
done

echo "=== compare  $(date -u) ==="
./dev python3 eval/longmem/compare_arms.py lm37_v36_sweep lm37_v37_sweep --labels v36,v37 --out-dir eval/longmem/reports/ab_compare_lm37 > $LOGDIR/compare.log 2>&1
echo "exit=$? (compare)"
echo "=== LONGMEM DONE  $(date -u) ==="
