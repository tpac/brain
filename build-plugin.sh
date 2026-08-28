#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# brain plugin builder
# Packs exactly what belongs in the .plugin file. Nothing else.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# --list: print the package manifest (one path per line) and exit — the ONE
# owner of "what ships"; the public-tree export (scripts/export-public-tree.sh)
# consumes this instead of re-deriving the list.
LIST_ONLY=0
if [ "${1:-}" = "--list" ]; then
  LIST_ONLY=1
  shift
fi
OUT="${1:-brain.plugin}"

# Explicit file manifest — if it's not listed, it doesn't ship
FILES=(
  # MIT requires the notice travel with the copies. plugin.json DECLARES
  # "license": "MIT"; without the text in the package, a zip-channel install
  # gets the claim and not the license. (Repo-clone installs pick up the
  # root LICENSE for free — this closes the upload path.)
  LICENSE
  .claude-plugin/plugin.json
  # marketplace.json makes the unzipped package a self-contained marketplace:
  # `claude plugin marketplace add <unzip-dir>` works with no repo access
  # (source "./" resolves to the unzip dir itself).
  .claude-plugin/marketplace.json
  .mcp.json
  requirements.txt
  # servers/ ships in FULL via `git ls-files servers` below — do NOT hand-list.
  # The explicit list rotted 62 files behind reality and shipped a brain_mcp.py
  # that imported modules (dispatch_common, frame, scouts/, …) which never got
  # packaged → the per-session MCP server crashed on import. git ls-files ends
  # that class of bug: tracked-only (no scratch leaks), new modules auto-ship.
  # hooks/ skills/ bin/ ship in FULL via `git ls-files` below — do NOT hand-list.
  # Same rot class as servers/: the hand-list silently dropped brain-daemon
  # (daemon launcher — dead on clean installs) and watch/SKILL.md before it.
  # scripts/ is a dev dir — nothing in it ships (live seeding is
  # servers/seed_pack.py, in the servers/ subtree).
)

cd "$DIR"

# Dashboard (read-only observer UI) — ship the git-TRACKED subtree only, via
# `git ls-files`, so a developer's untracked scratch / DB dump / secret in
# dashboard/ can never leak into the distributed package (the explicit-manifest
# safety the rest of this list enforces). New tracked files are picked up
# automatically (no per-file drift); a vanished dir fails loudly. Dev notes excluded.
_dash_files="$(git ls-files dashboard | grep -vE '/(DASHBOARD-NEXT|Dashboard-nextwork)\.md$' || true)"
if [ -z "$_dash_files" ]; then
  echo "MISSING: dashboard/ — no tracked files found (dir renamed/removed?)"
  exit 1
fi
while IFS= read -r _f; do FILES+=("$_f"); done <<< "$_dash_files"

# servers/ — ship the git-TRACKED tree in full (same tracked-only safety as
# dashboard above). Replaces the hand-maintained allowlist that rotted 62 files
# behind reality. Dev architecture notes (*.md) are excluded — runtime code only.
# `/archive/` is excluded: retired units kept in-repo for reference (e.g.
# scales/s2/archive/reclassify.py) have no runtime caller and read as internal
# clutter to an outside installer. In-repo, not in-package.
_srv_files="$(git ls-files servers | grep -vE '\.md$|/archive/' || true)"
if [ -z "$_srv_files" ]; then
  echo "MISSING: servers/ — no tracked files found (dir renamed/removed?)"
  exit 1
fi
while IFS= read -r _f; do FILES+=("$_f"); done <<< "$_srv_files"

# hooks/ skills/ — git-TRACKED runtime code, shipped in full (same pattern
# as dashboard/servers above). This closes the brain-daemon / watch-SKILL.md
# class of omission. NO top-level bin/ in the package: claude.ai-hosted plugins
# reject bin/ executables (PATH-injected but invisible on the admin approval
# surface) — launchers live in hooks/scripts/; bin/ holds only the
# runtime-fetched uv (ensure-runtime.sh), never packaged.
#   hooks/  : exclude *.md (dev notes like hooks/HOOKS.md — not runtime).
#   skills/ : KEEP *.md — SKILL.md *is* the skill; .md is the payload.
for _dir in hooks skills; do
  if [ "$_dir" = "skills" ]; then
    _files="$(git ls-files "$_dir" || true)"
  else
    _files="$(git ls-files "$_dir" | grep -vE '\.md$' || true)"
  fi
  if [ -z "$_files" ]; then
    echo "MISSING: $_dir/ — no tracked files found (dir renamed/removed?)"
    exit 1
  fi
  while IFS= read -r _f; do FILES+=("$_f"); done <<< "$_files"
done

if [ "$LIST_ONLY" -eq 1 ]; then
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
