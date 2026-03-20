#!/usr/bin/env python3
"""
Split brain.py monolith into mixin modules.

This script:
1. Reads brain.py and parses all method definitions
2. Assigns each method to a target module based on the MODULE_MAP
3. Creates mixin files with the extracted methods
4. Rewrites brain.py as a thin assembler that inherits from all mixins

Run from repo root: python3 scripts/split_monolith.py
"""

import ast
import os
import re
import sys

BRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "servers", "brain.py")

# ── Module assignments ──
# Each method name → target module
# Methods not listed stay in brain.py (core: __init__, save, close, etc.)

MODULE_MAP = {
    # brain_recall.py — search, retrieval, scoring
    "brain_recall": [
        "recall",
        "recall_with_embeddings",
        "semantic_recall",
        "backfill_embeddings",
        "spread_activation",
        "_search_keywords",
        "_mark_accessed",
        "_hebbian_strengthen",
        "_log_recall",
        "_get_recent",
        "_matches_temporal_filter",
    ],

    # brain_remember.py — node creation, TF-IDF indexing
    "brain_remember": [
        "remember",
        "remember_rich",
        "validate_node",
        "get_node_with_metadata",
        "backfill_summaries",
        "recall_expand",
        "_generate_summary",
        "_extract_keywords",
        "_bridge_at_store_time",
        "set_personal",
        "get_personal_nodes",
        "enrich_keywords",
        "store_embedding",
        # TF-IDF methods
        "_tfidf_tokenize",
        "_compute_tf",
        "_store_tfidf_vector",
        "_tfidf_score",
        "_batch_tfidf_scores",
        "_rebuild_tfidf_index",
    ],

    # brain_connections.py — edges, bridging, graph ops
    "brain_connections": [
        "connect",
        "connect_typed",
        "_find_bridge_candidates",
        "_create_bridge",
        "_bridge_at_consolidation",
        "_propose_bridge",
        "_mature_bridge_proposals",
        "_random_walk",
        "_get_node_title",
    ],

    # brain_evolution.py — tensions, hypotheses, patterns, auto-discover, auto-heal, auto-tune
    "brain_evolution": [
        "auto_discover_evolutions",  # NOTE: there are TWO definitions — we keep the SECOND (later) one
        "auto_heal",
        "auto_tune",
        "prompt_reflection",
        "create_tension",
        "create_hypothesis",
        "create_pattern",
        "create_catalyst",
        "create_aspiration",
        "resolve_evolution",
        "get_active_evolutions",
        "confirm_evolution",
        "dismiss_evolution",
        "get_relevant_aspirations",
        "check_hypothesis_relevance",
        "detect_catalyst",
    ],

    # brain_engineering.py — engineering memory, code cognition, divergence, synthesis
    "brain_engineering": [
        "remember_purpose",
        "remember_mechanism",
        "remember_impact",
        "remember_constraint",
        "remember_convention",
        "remember_lesson",
        "remember_mental_model",
        "remember_uncertainty",
        "record_reasoning_trace",
        "update_file_inventory",
        "get_file_inventory",
        "detect_file_changes",
        "update_system_purpose",
        "get_engineering_context",
        "get_change_impact",
        "record_divergence",
        "record_validation",
        "get_correction_patterns",
        "track_session_event",
        "synthesize_session",
        "get_last_synthesis",
        # Code cognition helpers
        "create_fn_reasoning",
        "create_param_influence",
        "create_code_concept",
        "create_arch_constraint",
        "create_causal_chain",
        "create_bug_lesson",
        "create_comment_anchor",
        "create_failure_mode",
        "create_performance",
        "create_capability",
        "create_interaction",
        "create_meta_learning",
        "set_reminder",
        "create_reminder",
        "get_due_reminders",
        "auto_generate_self_reflection",
    ],

    # brain_dreams.py — dreaming, consolidation
    "brain_dreams": [
        "dream",
        "_spawn_thought",
        "consolidate",
    ],

    # brain_vocabulary.py — vocabulary system
    "brain_vocabulary": [
        "learn_vocabulary",
        "_clear_vocabulary_gap",
        "_connect_vocabulary",
        "resolve_vocabulary",
    ],

    # brain_surface.py — suggest, context_boot, pre_edit, staged, health
    "brain_surface": [
        "suggest",
        "_suggest_reason",
        "context_boot",
        "validate_config",
        "health_check",
        "pre_edit",
        "procedure_trigger",
        "list_staged",
        "confirm_staged",
        "dismiss_staged",
        "auto_promote_staged",
        "get_suggest_metrics",
    ],

    # brain_absorb.py — brain-to-brain transfer
    "brain_absorb": [
        "absorb",
        "_match_existing",
        "_absorb_connections",
    ],
}

# Methods that stay in brain.py (core infrastructure)
CORE_METHODS = {
    "__init__", "get_instance", "clear_instances",
    "_post_schema_init", "_init_rate_limiter", "_init_file_logger",
    "_check_rate_limit", "_write_to_file_log", "_check_logs_db_size",
    "now", "_generate_id",
    "_recency_score", "_frequency_score", "_combined_score",
    "_classify_intent",
    "_get_node_count", "_get_edge_count", "_get_locked_count",
    "_get_session_activity", "_update_session_activity",
    "reset_session_activity", "record_remember", "record_message", "record_edit_check",
    "get_encoding_heartbeat",
    "_log_error", "get_recent_errors", "log_debug",
    "_get_tunable", "_set_tunable", "get_config", "set_config",
    "_get_embedder_config", "set_embedder_config",
    "get_debug_status",
    "scan_host_environment",
    "get_pruning_adjustments",
    "log_communication", "get_communication_stats",
    "save", "close",
    # Stub for removed consciousness
    "_STUB_consciousness_removed",
}

# Constants that belong to specific modules
MODULE_CONSTANTS = {
    "brain_recall": [
        "EMBEDDING_PRIMARY_WEIGHT", "KEYWORD_FALLBACK_WEIGHT",
        "TFIDF_SEMANTIC_WEIGHT", "TFIDF_KEYWORD_WEIGHT", "TFIDF_STOP_WORDS",
        "INTENT_PATTERNS", "INTENT_TYPE_BOOSTS", "TEMPORAL_PATTERNS",
    ],
    "brain_connections": [
        "EDGE_TYPES", "SPREAD_DECAY", "MAX_HOPS", "MAX_NEIGHBORS", "STABILITY_BOOST",
    ],
    "brain_dreams": [
        "DREAM_WALK_LENGTH", "DREAM_COUNT", "DREAM_MIN_NOVELTY",
    ],
    "brain_evolution": [
        "REASONING_STEP_TYPES", "CURIOSITY_MAX_PROMPTS",
        "CURIOSITY_CHAIN_GAP_THRESHOLD", "CURIOSITY_DECAY_WARNING_HOURS",
    ],
}


def main():
    with open(BRAIN_PY) as f:
        source = f.read()

    lines = source.split('\n')
    tree = ast.parse(source)

    # Find Brain class
    brain_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Brain':
            brain_class = node
            break

    if not brain_class:
        print("ERROR: Brain class not found")
        sys.exit(1)

    # Map method name → (start_line, end_line, source_text)
    # For duplicate names (auto_discover_evolutions), keep both
    methods = {}
    duplicates = {}
    for item in brain_class.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(item, 'end_lineno', item.lineno)
            method_source = '\n'.join(lines[item.lineno - 1:end])
            if item.name in methods:
                # Duplicate — store first as dead code
                duplicates[item.name] = methods[item.name]
                methods[item.name] = (item.lineno, end, method_source)
            else:
                methods[item.name] = (item.lineno, end, method_source)

    print("Found %d methods (%d with duplicates)" % (len(methods), len(duplicates)))

    # Verify all mapped methods exist
    all_mapped = set()
    for module, method_list in MODULE_MAP.items():
        for m in method_list:
            if m not in methods and m not in duplicates:
                print("WARNING: Method '%s' (assigned to %s) not found in Brain class" % (m, module))
            all_mapped.add(m)

    # Check for unmapped methods
    unmapped = set(methods.keys()) - all_mapped - CORE_METHODS
    if unmapped:
        print("WARNING: Unmapped methods (will stay in brain.py): %s" % sorted(unmapped))

    # Build method → line range mapping for removal from brain.py
    lines_to_remove = set()  # set of (start, end) ranges

    # For each module, collect methods and generate mixin file
    servers_dir = os.path.dirname(BRAIN_PY)

    for module_name, method_list in MODULE_MAP.items():
        mixin_class_name = ''.join(part.capitalize() for part in module_name.split('_')) + 'Mixin'
        # e.g. brain_recall -> BrainRecallMixin

        print("\n=== %s (%s) ===" % (module_name, mixin_class_name))

        # Collect method sources
        method_sources = []
        for m in method_list:
            if m in methods:
                start, end, src = methods[m]
                method_sources.append((m, start, end, src))
                lines_to_remove.add((start, end))
                print("  %s (lines %d-%d)" % (m, start, end))

                # For auto_discover_evolutions, also mark the dead first copy
                if m in duplicates:
                    dstart, dend, dsrc = duplicates[m]
                    lines_to_remove.add((dstart, dend))
                    print("  %s [DEAD COPY] (lines %d-%d) — REMOVING" % (m, dstart, dend))

        if not method_sources:
            print("  (no methods found, skipping)")
            continue

        # Sort by line number
        method_sources.sort(key=lambda x: x[1])

        # Determine imports needed
        imports = set()
        combined_src = '\n'.join(src for _, _, _, src in method_sources)
        if 'embedder.' in combined_src or 'embedder.embed' in combined_src:
            imports.add('from . import embedder')
        if 'json.' in combined_src or 'json.loads' in combined_src or 'json.dumps' in combined_src:
            imports.add('import json')
        if 'math.' in combined_src:
            imports.add('import math')
        if 're.' in combined_src or 're.compile' in combined_src:
            imports.add('import re')
        if 'uuid.' in combined_src:
            imports.add('import uuid')
        if 'time.' in combined_src:
            imports.add('import time')
        if 'random.' in combined_src:
            imports.add('import random')
        if 'struct.' in combined_src:
            imports.add('import struct')
        if 'datetime' in combined_src:
            imports.add('from datetime import datetime')
        if 'Dict' in combined_src or 'List' in combined_src or 'Optional' in combined_src or 'Any' in combined_src or 'Tuple' in combined_src or 'Set' in combined_src:
            typing_imports = []
            for t in ['Dict', 'List', 'Tuple', 'Optional', 'Any', 'Set']:
                if t in combined_src:
                    typing_imports.append(t)
            imports.add('from typing import %s' % ', '.join(sorted(typing_imports)))

        # Build the mixin file
        mixin_lines = []
        mixin_lines.append('"""')
        mixin_lines.append('brain — %s' % mixin_class_name.replace('Mixin', ' Mixin'))
        mixin_lines.append('')
        mixin_lines.append('Extracted from brain.py monolith. Methods are mixed into the Brain class')
        mixin_lines.append('via multiple inheritance. All methods reference self.conn, self.get_config, etc.')
        mixin_lines.append('which are provided by Brain.__init__.')
        mixin_lines.append('"""')
        mixin_lines.append('')

        for imp in sorted(imports):
            mixin_lines.append(imp)
        mixin_lines.append('')

        # Add module-specific constants
        if module_name in MODULE_CONSTANTS:
            mixin_lines.append('')
            mixin_lines.append('# ── Constants ──')
            for const_name in MODULE_CONSTANTS[module_name]:
                # Find the constant in the original source
                pattern = re.compile(r'^(%s\s*=\s*)' % re.escape(const_name), re.MULTILINE)
                match = pattern.search(source)
                if match:
                    # Extract the full constant definition (may span multiple lines)
                    const_start = source[:match.start()].count('\n')
                    # Find end of constant (next line that starts at column 0 with a non-space char)
                    remaining = source[match.start():]
                    const_lines_raw = remaining.split('\n')
                    const_end_offset = 1
                    for i, cl in enumerate(const_lines_raw[1:], 1):
                        if cl and not cl[0].isspace() and not cl.startswith('#'):
                            break
                        const_end_offset = i + 1
                    const_text = '\n'.join(const_lines_raw[:const_end_offset]).rstrip()
                    mixin_lines.append(const_text)
                    mixin_lines.append('')

        mixin_lines.append('')
        mixin_lines.append('class %s:' % mixin_class_name)
        mixin_lines.append('    """%s methods for Brain."""' % module_name.replace('brain_', '').capitalize())
        mixin_lines.append('')

        for m, start, end, src in method_sources:
            # Ensure proper indentation (methods should be at 4-space indent)
            mixin_lines.append(src)
            mixin_lines.append('')

        # Write the mixin file
        mixin_path = os.path.join(servers_dir, module_name + '.py')

        # Don't overwrite existing consciousness mixin
        if module_name == 'brain_consciousness':
            print("  SKIP: %s already exists" % mixin_path)
            continue

        with open(mixin_path, 'w') as f:
            f.write('\n'.join(mixin_lines))

        total_lines = sum(end - start + 1 for _, start, end, _ in method_sources)
        print("  Created %s (%d lines, %d methods)" % (mixin_path, total_lines, len(method_sources)))

    # Report
    print("\n=== Summary ===")
    total_removed = sum(end - start + 1 for start, end in lines_to_remove)
    print("Lines to remove from brain.py: %d" % total_removed)
    print("Original brain.py: %d lines" % len(lines))
    print("Expected brain.py after: ~%d lines" % (len(lines) - total_removed))

    # Don't rewrite brain.py automatically — too risky
    # Instead, output the line ranges to remove
    print("\nLine ranges to remove (sorted):")
    for start, end in sorted(lines_to_remove):
        print("  %d-%d (%d lines)" % (start, end, end - start + 1))

    print("\nMixin files created. Next step: manually update brain.py to:")
    print("1. Import all new mixins")
    print("2. class Brain(ConsciousnessMixin, RecallMixin, ..., etc)")
    print("3. Remove extracted method bodies (replace with pass or delete)")
    print("4. Run tests to verify")


if __name__ == '__main__':
    main()
