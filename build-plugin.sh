#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# brain v5.2.0 — Serverless plugin builder
# Packs exactly what belongs in the .plugin file. Nothing else.
#
# v5.2: Critical flag, PreToolUse(Bash) safety, vocabulary expansion,
#        inf bug fix. Plus v5.1 mixin architecture, confidence dynamics.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-brain.plugin}"

# Explicit file manifest — if it's not listed, it doesn't ship
FILES=(
  .claude-plugin/plugin.json
  # Python brain module — core + 11 mixin modules
  servers/__init__.py
  servers/brain.py
  servers/brain_absorb.py
  servers/brain_connections.py
  servers/brain_consciousness.py
  servers/brain_constants.py
  servers/brain_dreams.py
  servers/brain_engineering.py
  servers/brain_evolution.py
  servers/brain_recall.py
  servers/brain_remember.py
  servers/brain_surface.py
  servers/brain_vocabulary.py
  servers/schema.py
  servers/embedder.py
  servers/migrate.py
  servers/dal.py
  servers/daemon.py
  servers/metrics.py
  # Hook scripts
  hooks/hooks.json
  hooks/scripts/boot-brain.sh
  hooks/scripts/pre-compact-save.sh
  hooks/scripts/pre-edit-suggest.sh
  hooks/scripts/pre-bash-safety.sh
  hooks/scripts/pre-response-recall.sh
  hooks/scripts/post-response-track.sh
  hooks/scripts/idle-maintenance.sh
  hooks/scripts/post-compact-reboot.sh
  hooks/scripts/session-end.sh
  hooks/scripts/stop-failure-log.sh
  hooks/scripts/config-change-host.sh
  hooks/scripts/post-bash-host-check.sh
  hooks/scripts/worktree-context.sh
  hooks/scripts/worktree-cleanup.sh
  hooks/scripts/resolve-brain-db.sh
  hooks/scripts/daemon-client.sh
  hooks/scripts/brain-client.sh
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
