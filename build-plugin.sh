#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# brain plugin builder — the ONE owner of "what ships".
# Packs exactly what belongs in the .plugin file. Nothing else.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# --list:        print the package manifest (one path per line) and exit.
# --list-public: the package manifest PLUS the public-repo extras (README,
#                CONTRIBUTING, MIGRATING, the test suite) — what
#                scripts/export-public-tree.sh materializes. Both views come
#                from this file, so there is no second manifest to drift.
MODE=build
case "${1:-}" in
  --list)        MODE=list;   shift ;;
  --list-public) MODE=public; shift ;;
esac
OUT="${1:-brain.plugin}"
cd "$DIR"

# ── OPT-IN (DISTRIBUTION-READINESS 5.9). Every path below is NAMED — by literal
# or by shape — and a tracked file nothing here names does not ship. Selection
# runs through `git ls-files`, so an UNTRACKED file cannot ship even when its
# shape matches: a scratch DB or a secret dropped into a shipped dir is never
# packaged.
#
# Shapes are for CODE, whose reachability tests hold the other half of the
# bargain (test_deploy_contract.py: a shipped hooks/scripts file must be wired,
# a shipped servers/ module must be imported from an entrypoint) — and because
# a hand-list of modules rots: the last one fell 62 files behind and shipped a
# brain_mcp.py whose imports were not in the package. Data, docs, fixtures and
# launchers are named one by one — those are the shapes personal material has
# actually taken (session logs as JSON fixtures, gold corpora, dev harnesses
# defaulting to the author's machine, dev notes).
#
# A literal that matches nothing FAILS the build (renamed? removed?), so a named
# path cannot rot silently either.
FILES=()
_ship() {   # _ship <what> <pathspec>... — append the tracked files the pathspecs name
  local what="$1"; shift
  local found
  found="$(git ls-files -- "$@")"
  if [ -z "$found" ]; then
    echo "MISSING: $what — no tracked file matches (renamed/removed?)"
    exit 1
  fi
  while IFS= read -r _f; do FILES+=("$_f"); done <<< "$found"
}
_name() {   # _name <path>... — literal entries, each asserted tracked on its own
  local p; for p in "$@"; do _ship "$p" "$p"; done
}

# The notice travels with the copies. plugin.json declares Elastic-2.0;
# without the text in the package, a zip-channel install gets the claim
# and not the license. (Repo-clone installs pick up the root LICENSE for free —
# this closes the upload path.) LICENSE contains the full current grant.
# marketplace.json makes the unzipped package a self-contained marketplace:
# `claude plugin marketplace add <unzip-dir>` works with no repo access (source
# "./" resolves to the unzip dir itself). The Codex host reads
# .codex-plugin/plugin.json FIRST (before .claude-plugin/) and treats it as the
# whole manifest, so the same package installs on both hosts;
# hooks/hooks.codex.json ships with hooks/ below.
_name LICENSE \
      .claude-plugin/plugin.json \
      .claude-plugin/marketplace.json \
      .codex-plugin/plugin.json \
      .mcp.json \
      requirements.txt

# dashboard/ — the read-only observer UI: Python plus the static web assets.
# Dev notes (*.md) are not named and do not ship.
_ship "dashboard/" ':(glob)dashboard/**/*.py' ':(glob)dashboard/**/*.js' \
                   ':(glob)dashboard/**/*.css' ':(glob)dashboard/**/*.html'

# servers/ — runtime Python. `/archive/` is excluded: retired units kept in-repo
# for reference (e.g. scales/s2/archive/reclassify.py) have no runtime caller
# and read as internal clutter to an outside installer. Architecture notes
# (*.md) are not named. The one data file is named on its own.
_ship "servers/ modules" ':(glob)servers/**/*.py' ':(exclude)servers/**/archive/**'
_name servers/scales/s2/aspects_v1.json

# hooks/ — the two hook manifests, the scripts they wire (shell + Python), the
# launchd plists, and the extensionless `brain-*` launchers that launchd and the
# user's settings exec (test_launchers_carry_their_role pins that naming).
# Dev notes (hooks/HOOKS.md) are not named. NO top-level bin/ in the package:
# claude.ai-hosted plugins reject bin/ executables (PATH-injected but invisible
# on the admin approval surface) — launchers live in hooks/scripts/; bin/ holds
# only the runtime-fetched uv (ensure-runtime.sh), never packaged. scripts/ is a
# dev dir — nothing in it ships (live seeding is servers/seed_pack.py).
_name hooks/hooks.json hooks/hooks.codex.json
_ship "hooks/scripts/" ':(glob)hooks/scripts/*.sh' ':(glob)hooks/scripts/*.py' \
                       ':(glob)hooks/scripts/*.plist' ':(glob)hooks/scripts/brain-*'
# Host adapter modules are MCP extensions, separate from event-hook scripts.
_ship "hooks/adapters/" ':(glob)hooks/adapters/*.py'

# skills/ — SKILL.md *is* the skill; .md is the payload here. A skill's
# references/ subtree ships with it.
_ship "skills/" ':(glob)skills/*/SKILL.md' ':(glob)skills/*/references/*.md'

if [ "$MODE" = public ]; then
  # ── Public-repo extras (D-8: runtime + tests ship). Never in the .plugin zip.
  # CHANGELOG.md is the PUBLIC changelog (0.9.x line); scripts/release.sh
  # refuses a release whose version has no entry in it.
  _name README.md CONTRIBUTING.md MIGRATING.md CHANGELOG.md
  # tests/ — the suite by shape, its infrastructure by name. Nothing else under
  # tests/ ships: fixtures, corpora, gold datasets, benchmarks, runners and notes
  # are exactly where personal material has landed before (tests/conversations
  # was real session logs; the gold sets carry real content; the bench/run
  # harnesses default to the author's machine), so each needs its own line.
  # tests/test_deploy_contract.py matches the shape and is removed by the
  # export's denylist: it is the gate itself, and carries the planted leaks that
  # prove gate B fires.
  _ship "tests/ suite" ':(glob)tests/test_*.py' ':(glob)tests/integration/test_*.py'
  _name tests/__init__.py \
        tests/conftest.py \
        tests/brain_test_base.py \
        tests/isolated_brain.py \
        tests/eval_optional.py \
        tests/interaction_override.py \
        tests/integration/__init__.py
fi

if [ "$MODE" != build ]; then
  printf '%s\n' "${FILES[@]}"
  exit 0
fi

# Verify all files exist before packing
missing=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "Aborting — fix missing files first."
  exit 1
fi

rm -f "$OUT"
zip "$OUT" "${FILES[@]}"

size=$(du -h "$OUT" | cut -f1)
count=${#FILES[@]}
echo "✓ Built $OUT — $count files, $size"
