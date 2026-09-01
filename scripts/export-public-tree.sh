#!/bin/bash
# 5.1 — materialize the PUBLIC tree: exactly what the clean repo will hold.
# DEV TOOL: lives in scripts/, never ships (scripts/ is outside the manifest,
# and this file names the denylist it enforces).
#
#   Usage: export-public-tree.sh [out-dir]          (default: dist/public-tree)
#          export-public-tree.sh --scrub-only DIR    (gate B alone, for tests)
#          export-public-tree.sh --denylist-only DIR (gate A alone, for tests)
#
# Contents = the package manifest (build-plugin.sh --list — the ONE owner of
# "what ships", LICENSES/ included) + additive extras (README, CONTRIBUTING, tests/)
# − the denylist. Three hard-fail gates run on the RESULT, not the intent:
#   A. denylist — private artifacts must not exist in the output
#   B. scrub — personal-information patterns must not appear anywhere in the
#      output, except an explicit per-file attribution allowlist
#   C. version — plugin.json and marketplace.json must agree (a drift breaks
#      `/plugin update`, which compares them)
# The gates exist so 5.2–5.7 are enforced instead of remembered: a leak fails
# the export; it cannot ship quietly.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
say() { printf '%s\n' "[export-public] $*"; }
fail() { printf '%s\n' "[export-public] GATE FAILED: $*" >&2; exit 1; }

# ── Gate A: private artifacts that must never reach the public tree.
# Paths, relative to the tree root. Directories mean the whole subtree.
DENYLIST=(
  docs/DISTRIBUTION-READINESS.md   # names personal-data findings + internal paths
  CLAUDE.md                        # dev guide naming internal streams
  eval                             # a published harness is a claim (D-8)
  docs/archive                     # tracked session logs
  tests/archive                    # dev residue with personal paths
  tests/results                    # dev residue with personal paths
  tests/conversations              # REAL session logs as fixtures — personal
                                   # data; consuming tests need graceful-skip
  conversations                    # belt-and-braces (not tracked today)
  archives                         # belt-and-braces (not tracked today)

  # Gold corpora — real session content, incl. a real birthday. Ruled
  # 2026-08-31: exclude rather than sanitize, because a sanitized corpus that
  # still passes proves less than an honest gap. Consumers graceful-skip.
  tests/golden_dataset_v2.json
  tests/golden_dataset.json
  tests/golden_canary.json
  tests/corpus

  # The deploy gate itself. It only runs in the dev repo — but the PUBLIC repo
  # IS a git checkout with plugin.json tracked, so it would NOT skip there: it
  # would run and fail, because TestPublicTreeExport shells out to
  # scripts/export-public-tree.sh and scripts/ is outside the manifest. It also
  # carries this gate's own fixtures (deliberate `/Users/tpac` + `Tom Pachys`
  # strings that prove gate B catches them), so it can never pass gate B and
  # must not be allowlisted either — allowlisting the co-located leak on line
  # 487 would defeat the very check that line exists to prove.
  tests/test_deploy_contract.py

  # Dev harnesses, not tests — zero importers among tests/test_*.py, and each
  # hardcodes the author's machine as a default source. Same class as
  # tests/archive and tests/results above.
  tests/run_tests.py
  tests/eval_runner.py
  tests/generate_golden.py
  tests/relearning.py
  tests/benchmark_canary.py
  tests/benchmark_multivec_encoding.py
  tests/benchmark_real_conversations.py
  tests/bench_vector_cache.py
  tests/bench_precision_corpus.py
  tests/bench_precision_lifecycle.py
  tests/benchmark_full_baseline_214.py
)

# ── Gate B: personal-information patterns (grep -E, case-insensitive) and the
# attribution allowlist — the ONLY file:pattern pairs permitted to match.
SCRUB_PATTERNS='/Users/tpac|\btom\b|Pachys|playbuzz|AgentsContext'
ALLOWLIST=(
  # deliberate author attribution
  "LICENSE:Tom Pachys"
  "README.md:Tom Pachys"
  ".claude-plugin/plugin.json:Tom Pachys"
  ".claude-plugin/marketplace.json:Tom Pachys"
  # the legacy adoption rung (~/AgentsContext/brain) is shipped behavior:
  # existing brains at the pre-plugin default must stay reachable
  "hooks/scripts/resolve-brain-db.sh:AgentsContext"
  "hooks/scripts/boot-brain.sh:AgentsContext"
  "servers/daemon_config.py:AgentsContext"
  "dashboard/db.py:AgentsContext"
  # ...and the tests that exercise that rung. Same rationale as the four
  # resolver files above: the string names shipped behavior, not the author.
  "tests/test_db_resolution.py:AgentsContext"
  "tests/test_daemon_recovery.py:AgentsContext"
  # The seed pack's three origin-story tributes are deliberate CONTENT, not
  # identity assertions (D-5 amendment, ratified 2026-08-30) — and the test
  # that holds their boundary must name them to check them.
  "servers/seed_pack.py:Tom"
  "tests/test_seed_pack.py:Tom"
  # A D-12 guardrail that must name what it forbids: the test asserts a
  # surface prompt says "Operator:" and never a hardcoded operator name.
  # Renaming its literal to a name nobody uses would make it assert nothing.
  "tests/test_pipeline_contract.py:Tom"
)

_denylist_gate() {
  local root="$1" bad=0
  for p in "${DENYLIST[@]}"; do
    if [ -e "$root/$p" ]; then
      printf '%s\n' "  denylisted path present: $p" >&2
      bad=1
    fi
  done
  [ "$bad" -eq 0 ] || fail "denylist (gate A)"
  say "gate A (denylist): clean"
}

_scrub_gate() {
  local root="$1"
  # Every match, then filter through the allowlist in python: EXACT-string
  # file match (a regex here let `Xclaude-plugin` inherit `.claude-plugin`'s
  # allowance), and instead of dropping the whole line, STRIP the allowed
  # pattern's occurrences and re-test what's left — an attribution line that
  # also carries a leak ("Tom Pachys, /Users/tpac/…") must still fail.
  local hits
  hits="$(cd "$root" && grep -rIinE "$SCRUB_PATTERNS" . --exclude-dir=.git 2>/dev/null | sed 's|^\./||' || true)"
  local allow=""
  for pair in "${ALLOWLIST[@]}"; do allow+="${pair%%:*}"$'\t'"${pair#*:}"$'\n'; done
  local remaining
  remaining="$(printf '%s' "$hits" | ALLOW="$allow" SCRUB="$SCRUB_PATTERNS" python3 -c '
import os, re, sys
scrub = re.compile(os.environ["SCRUB"], re.IGNORECASE)
allow = {}
for row in os.environ["ALLOW"].splitlines():
    if row.strip():
        f, pat = row.split("\t", 1)
        allow.setdefault(f, []).append(pat)
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    fname, _, rest = line.partition(":")
    content = rest.partition(":")[2]
    for pat in allow.get(fname, []):
        content = content.replace(pat, "")
    if scrub.search(content):
        print(line)
')"
  if [ -n "$remaining" ]; then
    printf '%s\n' "$remaining" | head -50 >&2
    local n; n="$(printf '%s\n' "$remaining" | wc -l | tr -d ' ')"
    fail "scrub (gate B) — $n personal-information hit(s) above (first 50 shown). This list IS the 5.3 worklist."
  fi
  say "gate B (scrub): clean"
}

# Test doors: run one gate against an arbitrary tree, no export.
case "${1:-}" in
  --scrub-only)    _scrub_gate "${2:?dir required}";    exit 0 ;;
  --denylist-only) _denylist_gate "${2:?dir required}"; exit 0 ;;
esac

OUT="${1:-$REPO/dist/public-tree}"
cd "$REPO"

# ── Gate C first — cheapest, and nothing should be built on a drifted pair.
_pv="$(python3 -c 'import json;print(json.load(open(".claude-plugin/plugin.json"))["version"])')"
_mv="$(python3 -c 'import json;print(json.load(open(".claude-plugin/marketplace.json"))["plugins"][0]["version"])')"
[ "$_pv" = "$_mv" ] || fail "version (gate C) — plugin.json=$_pv marketplace.json=$_mv must agree (/plugin update compares them)"
say "gate C (version): $_pv"

# ── Materialize: manifest + extras, paths preserved. The DENYLIST is the one
# owner of "never ship": the copy filter derives from it (no second regex to
# drift), and gate A still verifies the RESULT.
_denied() {
  for p in "${DENYLIST[@]}"; do
    case "$1" in "$p"|"$p"/*) return 0 ;; esac
  done
  return 1
}
# Clobber guard: only wipe something that is empty, absent, or a previous
# export of ours — never an arbitrary directory handed in by mistake.
if [ -e "$OUT" ] && [ ! -e "$OUT/.claude-plugin" ] \
   && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  fail "$OUT exists and does not look like a previous export — refusing to remove it"
fi
rm -rf "$OUT"
mkdir -p "$OUT"
_copy() { _denied "$1" && return 0
          mkdir -p "$OUT/$(dirname "$1")"; cp "$1" "$OUT/$1"; _count=$((_count+1)); }
_count=0
while IFS= read -r f; do _copy "$f"; done < <(./build-plugin.sh --list)
# extras: the public-repo face + the test suite (D-8: runtime + tests ship).
# LICENSES/ is NOT listed here — it ships in the package manifest above, which
# stays the one owner of "what ships". Copying it twice only inflated $_count.
for f in README.md CONTRIBUTING.md; do _copy "$f"; done
while IFS= read -r f; do _copy "$f"; done < <(git ls-files tests)
say "materialized $_count files -> $OUT"

_denylist_gate "$OUT"
_scrub_gate "$OUT"
say "public tree is clean: $OUT"
