#!/bin/zsh
# V3.8 replay probes — the three captured gym windows (item 59524333, both longmem corpora) replayed under
# V3.6 and each draft carrier, three repeats each, replies dumped for probe_score.py. No brain, no isolation:
# one Sonnet call per repeat at the captured size (~$0.12). Arms:
#   v36      the V3.6 system assembled on this checkout (== the b0175a captures' system, checked)
#   gloss    template_gloss.md assembled            (position)
#   example  template_example.md assembled          (example)
#   walk     V3.6 system + gist_walk.md substituted whole into the user content (procedure)
# Windows: r2 = run 2 (stop 10) on the V3.6 brain — turn 8's 6 pm is UNCOVERED here (the misfile window);
#          r3 = run 3 (stop 11) on the V3.6 brain — the open + half-revised node in the catalog;
#          r3b = run 3 on the V3.7 brain — a clean 7 pm node with a "stable" thought (the anchoring window).
# Usage: probe_run.sh <results-dir> [repeats]
set -u
cd "$(dirname "$0")/../../.." || exit 1
F=eval/fixtures/s1e_guide_v3_8_2026-09-14
O=${1:?results dir}; R=${2:-3}
B=/Users/tpac/AgentsContext/eval-corpus
typeset -A WIN
WIN=(r2 $B/b0175a/59524333/payloads/2026-09-14/s1e-ingest-5-10/000-round_payload.json
     r3 $B/b0175a/59524333/payloads/2026-09-14/s1e-ingest-5-11/000-round_payload.json
     r3b $B/ef2443/59524333/payloads/2026-09-14/s1e-ingest-5-11/000-round_payload.json)
mkdir -p $O
run_arm() {
  local arm=$1; shift
  for w in r2 r3 r3b; do
    ./dev python3 $F/replay_payload.py ${WIN[$w]} --repeats $R --dump $O/dump/${arm}_$w "$@" > $O/probe_${arm}_$w.txt 2>&1
    echo "done $arm $w exit=$?"
  done
}
run_arm v36 --system $O/system_v36.txt &
run_arm gloss --system $O/system_gloss.txt &
run_arm example --system $O/system_example.txt &
run_arm walk --system $O/system_v36.txt --sub-file $O/gist_old.txt $O/gist_new.txt &
wait
echo "ALL DONE $(date -u)"
