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
#      `/plugin update`, which compares them); with EXPECT_VERSION=X in the
#      environment they must also both READ X — the release command passes
#      the version it is releasing, so agreement on the wrong value fails too
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
  # must not be allowlisted either — allowlisting the co-located leak in
  # test_scrub_allowlist_cannot_mask_a_colocated_leak would defeat the very
  # check that fixture exists to prove.
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
# The author's handle is matched BARE (`tpac`), not just as a path — the
# `/Users/tpac` form missed "never left tpac's laptop" in a comment. NO trailing
# `\b` on it: `\btpac\b` would be NARROWER than the path form it replaced and
# would miss a sibling checkout like `/Users/tpac_old`. Both the employer's
# current name (`ex.co`) and its former one (`playbuzz`) are here: scrubbing
# only the old name is how 20 hits of the current one survived.
SCRUB_PATTERNS='\btpac|\btom\b|Pachys|playbuzz|\bex\.co\b|\bexco\b|AgentsContext'
ALLOWLIST=(
  # deliberate author attribution
  "LICENSE:Tom Pachys"
  "README.md:Tom Pachys"
  ".claude-plugin/plugin.json:Tom Pachys"
  ".claude-plugin/marketplace.json:Tom Pachys"
  # the publish target itself — the GitHub org the plugin is installed from
  "README.md:tpac/entity"
  ".claude-plugin/plugin.json:github.com/tpac"
  ".claude-plugin/marketplace.json:github.com/tpac"
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
  # The strip is BOUNDARY-AWARE: an allowed pattern only cancels a match when
  # it is not the prefix of a longer word. Without that, allowing
  # `github.com/tpac` would silently also allow `github.com/tpachys`, and
  # allowing `Tom` would allow `Tommy` — the allowlist would grant more than
  # it names.
  #
  # THREE checks, because `grep -rIin` alone sees only the text it can read:
  #   content — the lines themselves (allowlist applies)
  #   PATH    — the file NAMES. grep prints only matching CONTENT, so
  #             `tests/fixtures/tom_pachys_session.json` full of anodyne JSON
  #             passed clean. Naming a fixture after the session it came from is
  #             exactly what this repo does — `tests/conversations` is
  #             denylisted for that reason.
  #   BINARY  — files `grep -I` SKIPS. A skipped file is unexamined, and gate B
  #             must never report clean over something it could not read. There
  #             are zero binaries in the export today, which is why this is free
  #             to assert now rather than after one arrives.
  local hits
  hits="$(cd "$root" && grep -rIinE "$SCRUB_PATTERNS" . --exclude-dir=.git 2>/dev/null | sed 's|^\./||' || true)"
  local allow=""
  for pair in "${ALLOWLIST[@]}"; do allow+="${pair%%:*}"$'\t'"${pair#*:}"$'\n'; done
  local remaining
  remaining="$(printf '%s' "$hits" | ALLOW="$allow" SCRUB="$SCRUB_PATTERNS" ROOT="$root" python3 -c '
import os, re, sys
scrub = re.compile(os.environ["SCRUB"], re.IGNORECASE)
root = os.environ["ROOT"]
allow = {}
for row in os.environ["ALLOW"].splitlines():
    if row.strip():
        f, pat = row.split("\t", 1)
        allow.setdefault(f, []).append(pat)
out = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    fname, _, rest = line.partition(":")
    content = rest.partition(":")[2]
    for pat in allow.get(fname, []):
        content = re.sub(re.escape(pat) + r"(?![\w-])", "", content)
    if scrub.search(content):
        out.append(line)
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != ".git"]
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, root)
        if scrub.search(rel):
            out.append("%s:0:<FILENAME> %s" % (rel, rel))
        try:
            with open(full, "rb") as fh:
                if b"\x00" in fh.read(8192):
                    out.append("%s:0:<BINARY> unreadable by the scrub gate" % rel)
        except OSError:
            pass
print("\n".join(out))
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
if [ -n "${EXPECT_VERSION:-}" ] && [ "$_pv" != "$EXPECT_VERSION" ]; then
  fail "version (gate C) — manifests read $_pv but this release expects $EXPECT_VERSION"
fi
say "gate C (version): $_pv${EXPECT_VERSION:+ (expected $EXPECT_VERSION ✓)}"

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
