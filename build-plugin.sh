#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# brain v3.2 — Serverless plugin builder
# Packs exactly what belongs in the .plugin file. Nothing else.
#
# v3.0: Python serverless — no Node.js, no HTTP server.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-brain.plugin}"

# Explicit file manifest — if it's not listed, it doesn't ship
FILES=(
  .claude-plugin/plugin.json
  # Python brain module (replaces Node.js server)
  servers/__init__.py
  servers/brain.py
  servers/schema.py
  servers/embedder.py
  servers/migrate.py
  # Hook scripts (all serverless — no HTTP calls)
  hooks/hooks.json
  hooks/scripts/boot-brain.sh
  hooks/scripts/pre-compact-save.sh
  hooks/scripts/pre-edit-suggest.sh
  hooks/scripts/pre-response-recall.sh
  hooks/scripts/extract-session-log.py
  # Skill definition
  skills/brain/SKILL.md
  skills/brain/references/detailed-api.md
  # Test framework
  tests/__init__.py
  tests/metrics.py
  tests/generate_golden.py
  tests/eval_runner.py
  tests/run_tests.py
  tests/golden_dataset.json
  tests/transcript_parser.py
  tests/relearning.py
)

cd "$DIR"

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
echo ""
echo "Built $OUT ($size)"
