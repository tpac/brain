#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# brain plugin builder
# Packs exactly what belongs in the .plugin file. Nothing else.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-brain.plugin}"

# Explicit file manifest — if it's not listed, it doesn't ship
FILES=(
  .claude-plugin/plugin.json
  .mcp.json
  requirements.txt
  # Core brain
  servers/__init__.py
  servers/brain.py
  servers/brain_assembly.py
  servers/brain_connections.py
  servers/brain_constants.py
  servers/brain_recall.py
  servers/brain_remember.py
  servers/brain_reminders.py
  servers/brain_voice.py
  servers/contract.py
  servers/pipeline_contract.py
  servers/recall_scoring.py
  servers/text_processing.py
  servers/embedder.py
  servers/embed_queue.py
  servers/schema.py
  # DAL
  servers/dal.py
  servers/dal_metadata.py
  servers/dal_signal_queue.py
  # Daemon
  servers/daemon_config.py
  servers/daemon_server.py
  servers/daemon_client.py
  servers/daemon_dispatch.py
  servers/daemon_hooks.py
  # MCP + signals
  servers/brain_mcp.py
  servers/signal_producers.py
  servers/surface_assembler.py
  servers/health_check.py
  servers/interaction_seed.py
  servers/session_context.py
  servers/trace_contract.py
  servers/brain_cli.py
  # Aspects (unified taxonomy — replaces former families layer)
  servers/aspects.py
  servers/aspect_migration.py
  # Scale modules
  servers/scales/__init__.py
  servers/scales/dispatch.py
  servers/scales/runner.py
  servers/scales/s0/__init__.py
  servers/scales/s0/conversation.py
  servers/scales/s1/__init__.py
  servers/scales/s1/surface.py
  servers/scales/s1/surface_contract.py
  servers/scales/s1/encode.py
  servers/scales/s1/encode_contract.py
  servers/scales/s2/__init__.py
  servers/scales/s2/base.py
  servers/scales/s2/coordinator.py
  servers/scales/s2/community.py
  servers/scales/s2/community_contract.py
  servers/scales/s2/community_decoder.py
  servers/scales/s2/community_encoder.py
  servers/scales/s2/community_enrichment_prompt.py
  servers/scales/s2/consolidation.py
  servers/scales/s2/consolidation_contract.py
  servers/scales/s2/consolidation_decoder.py
  servers/scales/s2/consolidation_encoder.py
  servers/scales/s2/consolidation_enrichment_prompt.py
  servers/scales/s2/aspects_v1.json
  servers/scales/s2/healer.py
  servers/scales/s2/healer_contract.py
  servers/scales/s2/healer_decoder.py
  servers/scales/s2/healer_encoder.py
  servers/scales/s2/healer_prompt.py
  servers/scales/s2/reclassify.py
  # Migrations directory removed — schema migrations handled by schema.py's
  # diff-based ALTER TABLE mechanism (see 7d6caeb1).
  # Hook scripts — bash shims
  hooks/hooks.json
  hooks/scripts/ensure-runtime.sh
  hooks/scripts/brain-env.sh
  hooks/scripts/mcp-launch.sh
  hooks/scripts/boot-brain.sh
  hooks/scripts/pre-edit-suggest.sh
  hooks/scripts/pre-bash-safety.sh
  hooks/scripts/pre-response-recall.sh
  hooks/scripts/post-response-track.sh
  hooks/scripts/idle-maintenance.sh
  hooks/scripts/session-end.sh
  hooks/scripts/stop-failure-log.sh
  hooks/scripts/config-change-host.sh
  hooks/scripts/post-bash-host-check.sh
  hooks/scripts/worktree-context.sh
  hooks/scripts/worktree-cleanup.sh
  hooks/scripts/restart-daemon.sh
  hooks/scripts/encoding-hook.sh
  hooks/scripts/brain-statusline.sh
  hooks/scripts/resolve-brain-db.sh
  hooks/scripts/daemon-client.sh
  hooks/scripts/brain-client.sh
  # Hook scripts — Python logic
  hooks/scripts/hook_common.py
  hooks/scripts/boot_brain.py
  hooks/scripts/pre_edit_suggest.py
  hooks/scripts/pre_bash_safety.py
  hooks/scripts/pre_response_recall.py
  hooks/scripts/post_response_track.py
  hooks/scripts/idle_maintenance.py
  hooks/scripts/session_end.py
  hooks/scripts/stop_failure_log.py
  hooks/scripts/config_change_host.py
  hooks/scripts/post_bash_host_check.py
  hooks/scripts/worktree_context.py
  hooks/scripts/worktree_cleanup.py
  hooks/scripts/encoding_hook.py
  hooks/scripts/post_tool_trace.py
  hooks/scripts/agent-bridge.py
  # Skill
  skills/brain/SKILL.md
  skills/brain/references/detailed-api.md
  # Data
  data/common_words_10k.txt
  scripts/seed_brain.py
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
count=${#FILES[@]}
echo "✓ Built $OUT — $count files, $size"
