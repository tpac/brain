#!/bin/zsh
# Longmem, V3.6 vs a V3.8 candidate — the same frozen-corpus benchmark as run_longmem_v36_v37.sh (Tom's brief
# item 2, 2026-09-14: "rerun the same longmem corpus with the same per-layer analysis"). Same ten items as gate 4,
# Longmem 9 and the V3.6/V3.7 pair (2 per axis), so the numbers sit beside prod 22 → branch 26 → guide 20 → V3.6 20
# → V3.7 22 of 30. Both arms are passed as FILES so each corpus is content-addressed by the file's hash; --force
# too (the 2026-08-24 cache hazard). Builds run in parallel (separate eval brains); then a variance-3 sweep per arm
# with --force-preflight (zero-encode items abort a sweep otherwise — id:a0224457), then compare_arms.
#
# Usage:  run_longmem_v36_v38.sh <arm-name> <template.md> [gist.md]
#   arm-name     e.g. gloss | example | walk — used in labels (lm38_<arm>)
#   template.md  the candidate s1e template (template_gloss.md, template_example.md, or the V3.6 template for a gist arm)
#   gist.md      optional: a candidate gist, applied with --interaction-template s1e_gist=<file> (the walk arm)
# Cost: a pair ≈ $15 and ~1 h wall clock (two parallel builds ≈ 45 min, two sweeps, compare).
set -u
cd "$(dirname "$0")/../../.." || exit 1
ARM=${1:?arm name}; TPL=${2:?candidate template}; GIST=${3:-}
QIDS=54026fce,fca762bc,2311e44b,bc149d6b,71017276,gpt4_b0863698,cc5ded98,59524333,09ba9854_abs,edced276_abs
V36=eval/fixtures/s1e_guide_v3_6_2026-09-13/template_full.md
LOGDIR=eval/results/longmem_v36_v38_${ARM}_$(date -u +%Y-%m-%d)
mkdir -p $LOGDIR
GIST_ARGS=()
[[ -n $GIST ]] && GIST_ARGS=(--interaction-template "s1e_gist=$GIST")

echo "=== LONGMEM v36 vs v38_$ARM START  $(date -u) ==="
git log --oneline -1
echo "candidate template=$TPL gist=${GIST:-<production gist>}"

./dev python3 eval/longmem/build_corpus.py --qids $QIDS --s1e $V36 --force --label lm38_v36 > $LOGDIR/build_v36.log 2>&1 &
P36=$!
./dev python3 eval/longmem/build_corpus.py --qids $QIDS --s1e $TPL "${GIST_ARGS[@]}" --force --label lm38_$ARM > $LOGDIR/build_$ARM.log 2>&1 &
P38=$!
wait $P36; echo "exit=$? (build v36)"
wait $P38; echo "exit=$? (build $ARM)"
H36=$(grep -o 'config hash = [0-9a-f]*' $LOGDIR/build_v36.log | head -1 | awk '{print $4}')
H38=$(grep -o 'config hash = [0-9a-f]*' $LOGDIR/build_$ARM.log | head -1 | awk '{print $4}')
echo "H36=$H36 H38=$H38"

for arm in v36 $ARM; do
  H=$([ $arm = v36 ] && echo $H36 || echo $H38)
  echo "=== sweep $arm corpus=$H variance=3  $(date -u) ==="
  ./dev python3 eval/longmem/sweep.py --corpus $H --variance 3 --force-preflight --label lm38_${arm}_sweep > $LOGDIR/sweep_$arm.log 2>&1
  echo "exit=$? (sweep $arm)"
done

echo "=== compare  $(date -u) ==="
./dev python3 eval/longmem/compare_arms.py lm38_v36_sweep lm38_${ARM}_sweep --labels v36,v38_$ARM --out-dir eval/longmem/reports/ab_compare_lm38_$ARM > $LOGDIR/compare.log 2>&1
echo "exit=$? (compare)"
echo "=== LONGMEM DONE  $(date -u) ==="
