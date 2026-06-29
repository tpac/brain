"""
brain — Daemon Command Dispatch

Table-driven command routing. Each command maps to a handler function with
read/write classification for lock management. Handlers live in the
dispatch_* domain modules; this file is the single registry (COMMAND_TABLE)
that daemon_server.py looks up, plus re-exports so handlers stay importable
from servers.daemon_dispatch.

Commands are plain functions: handler(brain, args, graph_changes) -> dict
"""

from typing import Dict

from .dispatch_common import CmdEntry, check_unknown_keys, _resolve_id, _pop_session_ctx
from .dispatch_write import (
    _handle_remember, _handle_remember_batch, _handle_revise, _handle_revise_batch,
    _handle_brain_batch, _handle_connect, _handle_connect_batch, _handle_revise_edge,
    _handle_enrich,
    _validate_source_refs, _maybe_warn_source_refs_hex_format,
    _maybe_warn_source_refs_sparseness,
)
from .dispatch_read import (
    _enrich_recall_results, _handle_recall, _handle_recall_batch, _handle_get_node,
    _handle_get_nodes, _handle_find_node_by_title, _handle_filter_nodes,
    _handle_graph_expand, _handle_context_boot,
)
from .dispatch_observability import (
    _handle_trace_append, _handle_get_trace, _handle_get_traces, _handle_query_traces,
    _handle_recall_episodes,
    _handle_count_traces, _handle_query_logs,
    _handle_clear_errors, _handle_log_debug, _handle_list_interactions,
    _handle_get_interaction, _handle_set_interaction_active, _handle_register_interaction,
)
from .dispatch_ops import (
    _handle_ping, _handle_health_check, _handle_validate_config,
    _handle_scan_host, _handle_procedure_trigger, _handle_get_config,
    _handle_get_debug_status, _handle_enrichment_coverage, _handle_pre_edit,
    _handle_save, _handle_reset_session, _handle_set_config,
    _handle_promote_staged, _handle_backfill_summaries, _handle_backfill_vectors,
    _handle_diagnose, _handle_eval,
    _handle_drop_sys_revision_history,
)
from .dispatch_self import (
    _handle_self_presence, _handle_self_peek, _handle_self_send, _handle_self_inbox,
    _handle_self_inbox_peek, _handle_self_outbox)


COMMAND_TABLE: Dict[str, CmdEntry] = {
    # Removed 2026-04-13: engineering_context, correction_patterns, last_synthesis,
    # dreams, staged, suggest_metrics, get_active_evolutions, assess_developmental_stage,
    # instinct_check, prompt_reflection, self_reflection, consolidate, dream,
    # auto_heal, auto_tune, synthesize_session.
    # Removed 2026-04-06: record_divergence, learn_vocabulary.

    # ── Reads (no lock needed) ──
    "ping":                     CmdEntry(_handle_ping,                 is_write=False),
    "context_boot":             CmdEntry(_handle_context_boot,         is_write=False),
    "recall":                   CmdEntry(_handle_recall,               is_write=False),
    "validate_config":          CmdEntry(_handle_validate_config,      is_write=False),
    "health_check":             CmdEntry(_handle_health_check,         is_write=False),
    "scan_host":                CmdEntry(_handle_scan_host,            is_write=False),
    "procedure_trigger":        CmdEntry(_handle_procedure_trigger,    is_write=False),
    "get_config":               CmdEntry(_handle_get_config,           is_write=False),
    "get_debug_status":         CmdEntry(_handle_get_debug_status,     is_write=False),
    "enrichment_coverage":      CmdEntry(_handle_enrichment_coverage,  is_write=False),
    "pre_edit":                 CmdEntry(_handle_pre_edit,             is_write=False),

    # ── Self channel — presence (read-only) + signal (writes brain_logs.db,
    #    lock-guarded inside signal.py, so is_write=False at the daemon layer) ──
    "self_presence":            CmdEntry(_handle_self_presence,        is_write=False),
    "self_peek":                CmdEntry(_handle_self_peek,            is_write=False),
    "self_send":                CmdEntry(_handle_self_send,            is_write=False),
    "self_inbox":               CmdEntry(_handle_self_inbox,           is_write=False),
    "self_inbox_peek":          CmdEntry(_handle_self_inbox_peek,      is_write=False),
    "self_outbox":              CmdEntry(_handle_self_outbox,          is_write=False),

    # ── Writes (exclusive lock) ──
    "save":                CmdEntry(_handle_save,               is_write=True, marks_dirty=False),
    "reset_session":       CmdEntry(_handle_reset_session,      is_write=True, marks_dirty=True),
    "set_config":          CmdEntry(_handle_set_config,         is_write=True, marks_dirty=True),
    "log_debug":           CmdEntry(_handle_log_debug,          is_write=True, marks_dirty=True),
    "promote_staged":      CmdEntry(_handle_promote_staged,     is_write=True, marks_dirty=True),
    "backfill_summaries":  CmdEntry(_handle_backfill_summaries, is_write=True, marks_dirty=True),
    "backfill_vectors":    CmdEntry(_handle_backfill_vectors,   is_write=True, marks_dirty=True),
    "remember":              CmdEntry(_handle_remember,             is_write=True, marks_dirty=True),
    "remember_batch":        CmdEntry(_handle_remember_batch,      is_write=True, marks_dirty=True),
    "revise":                CmdEntry(_handle_revise,               is_write=True, marks_dirty=True),
    "revise_batch":          CmdEntry(_handle_revise_batch,         is_write=True, marks_dirty=True),
    "find_node_by_title":    CmdEntry(_handle_find_node_by_title,  is_write=False, marks_dirty=False),
    "filter_nodes":          CmdEntry(_handle_filter_nodes,        is_write=False, marks_dirty=False),
    "query_logs":            CmdEntry(_handle_query_logs,          is_write=False, marks_dirty=False),
    "clear_errors":          CmdEntry(_handle_clear_errors,        is_write=True,  marks_dirty=False),
    "query_traces":          CmdEntry(_handle_query_traces,        is_write=False, marks_dirty=False),
    "recall_episodes":       CmdEntry(_handle_recall_episodes,     is_write=False, marks_dirty=False),
    "count_traces":          CmdEntry(_handle_count_traces,        is_write=False, marks_dirty=False),
    "list_interactions":     CmdEntry(_handle_list_interactions,   is_write=False, marks_dirty=False),
    "get_interaction":       CmdEntry(_handle_get_interaction,     is_write=False, marks_dirty=False),
    "register_interaction":  CmdEntry(_handle_register_interaction,is_write=True,  marks_dirty=False),
    "set_interaction_active": CmdEntry(_handle_set_interaction_active, is_write=True,  marks_dirty=False),
    "trace_append":          CmdEntry(_handle_trace_append,        is_write=True,  marks_dirty=False),
    "get_node":              CmdEntry(_handle_get_node,             is_write=False, marks_dirty=False),
    "get_nodes":             CmdEntry(_handle_get_nodes,            is_write=False, marks_dirty=False),
    "get_trace":             CmdEntry(_handle_get_trace,            is_write=False, marks_dirty=False),
    "get_traces":            CmdEntry(_handle_get_traces,           is_write=False, marks_dirty=False),
    "recall_batch":          CmdEntry(_handle_recall_batch,         is_write=False, marks_dirty=False),
    "graph_expand":          CmdEntry(_handle_graph_expand,         is_write=False, marks_dirty=False),
    "connect":               CmdEntry(_handle_connect,             is_write=True, marks_dirty=True,
                                      accepts=frozenset({"source_id", "target_id", "relation",
                                                         "weight", "description", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "connect_batch":         CmdEntry(_handle_connect_batch,       is_write=True, marks_dirty=True,
                                      accepts=frozenset({"connections", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "revise_edge":           CmdEntry(_handle_revise_edge,         is_write=True, marks_dirty=True,
                                      accepts=frozenset({"source_id", "target_id", "relation",
                                                         "new_relation", "description", "weight",
                                                         "encoding_source", "chain_id", "session_id",
                                                         "reason"})),
    "brain_batch":           CmdEntry(_handle_brain_batch,         is_write=True, marks_dirty=True,
                                      accepts=frozenset({"operations", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "enrich":                CmdEntry(_handle_enrich,              is_write=True, marks_dirty=True),
    "eval":                  CmdEntry(_handle_eval,                is_write=True, marks_dirty=True),
    "diagnose":              CmdEntry(_handle_diagnose,            is_write=False, marks_dirty=False),
    "drop_sys_revision_history": CmdEntry(_handle_drop_sys_revision_history, is_write=True, marks_dirty=True),
}

# "shutdown" is handled directly by daemon_server (needs to set self.running)
# "hook_*" commands are dispatched via HOOK_TABLE in daemon_server
