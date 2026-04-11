#!/usr/bin/env python3
"""Recover lost open metadata fields from S1E encoding traces.

Background:
    daemon_dispatch.py had a bug where _handle_remember and _handle_remember_batch
    filtered out all fields not in the contract's known fields (get_remember_fields()).
    This silently dropped open fields like `assumed`, `reality`, `trigger`,
    `emotional_context`, `impact_scope` etc. that the S1E encoding agent tried
    to write via brain_batch operations.

    The bug: `cleaned = {k: v for k, v in spec.items() if k in accepted_fields}`
    The fix: `cleaned = {k: v for k, v in spec.items() if v is not None}`

Data availability:
    The original tool call inputs (with the open field values) are NOT stored
    anywhere recoverable. The trace system (runner.py) only logs action summaries
    and node IDs in delta events, not the full tool call parameters. The Anthropic
    API responses are not cached. So the exact values are unrecoverable from traces.

    However, the encoding agent sometimes embedded structured field labels in node
    content (e.g. "Trigger: ..." or "**impact_scope:** ..."). This script scans
    for those patterns.

Recovery approach:
    1. Scan all encoder-created nodes for explicit field-label patterns in content
    2. Extract only high-confidence matches (labels at line start, not inline narrative)
    3. Identify correction/lesson/moment nodes that likely SHOULD have had open fields
       but don't — flag these as candidates for manual review or re-encoding
    4. Report findings and optionally write recoverable fields to node_metadata_kv

Usage:
    python3 scripts/recover_s1e_open_fields.py --dry-run     # report only
    python3 scripts/recover_s1e_open_fields.py --apply        # write recoverable fields
    python3 scripts/recover_s1e_open_fields.py --dry-run -v   # verbose + candidates list
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Optional


# ── Configuration ──

DB_DIR = os.environ.get('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))
BRAIN_DB = os.path.join(DB_DIR, 'brain.db')
LOGS_DB = os.path.join(DB_DIR, 'brain_logs.db')

# Open fields the encoding agent was told to use (encoding-agent-v3.md line 118)
DOCUMENTED_OPEN_FIELDS = {
    'assumed', 'reality', 'trigger', 'emotional_context', 'impact_scope',
}

# Fields already in the contract — never extract these from content
CONTRACT_FIELDS = {
    'type', 'title', 'content', 'keywords', 'confidence', 'locked',
    'archived', 'critical', 'emotion', 'emotion_label', 'project',
    'personal', 'personal_context', 'evolution_status', 'source_turn_id',
    'encoding_source', 'situation', 'id',
    # Promoted metadata
    'reasoning', 'user_raw_quote', 'anchor_raw_quote',
    'correction_of', 'correction_pattern', 'source_context',
    'confidence_rationale', 'alternatives', 'change_impacts',
    'source_attribution', 'scope',
    # Internal
    'revision_history', 'validation_count', 'metadata_created_at',
    'community_idx', 'last_validated',
}

# Node types that are strong candidates for having had open fields
CANDIDATE_TYPES_CORRECTION = {'correction', 'bug'}
CANDIDATE_TYPES_EMOTIONAL = {'moment', 'identity', 'lesson'}


# ── Extraction functions ──
# All extractors require explicit field-label patterns at line start.
# We do NOT infer fields from narrative content — too many false positives.

def extract_open_fields_from_content(content: str, node_type: str) -> Dict[str, str]:
    """Extract open metadata fields from explicit labels in node content.

    Only matches the documented open fields from encoding-agent-v3.md line 118:
    assumed, reality, trigger, emotional_context, impact_scope.

    The generic snake_case scanner was removed because brain content heavily
    uses code identifiers (function names, table names, edge types) as section
    labels, producing too many false positives. Only the five documented open
    fields are searched.

    For assumed/reality: requires BOTH to be present (pair validation) and
    only on correction/bug type nodes.

    Returns: {field_name: value}
    """
    fields = {}

    # assumed/reality pair — only for correction-type nodes, require both
    if node_type in CANDIDATE_TYPES_CORRECTION:
        assumed_m = re.search(
            r'(?:^|\n)\s*(?:[-*]*\s*)?(?:\*{0,2})assumed(?:\*{0,2})\s*:\s*(.+?)(?:\n|$)',
            content, re.IGNORECASE | re.MULTILINE)
        reality_m = re.search(
            r'(?:^|\n)\s*(?:[-*]*\s*)?(?:\*{0,2})reality(?:\*{0,2})\s*:\s*(.+?)(?:\n|$)',
            content, re.IGNORECASE | re.MULTILINE)
        if assumed_m and reality_m:
            a_val = assumed_m.group(1).strip().strip('"\'')
            r_val = reality_m.group(1).strip().strip('"\'')
            if len(a_val) > 15 and len(r_val) > 15:
                fields['assumed'] = a_val[:500]
                fields['reality'] = r_val[:500]

    # trigger, emotional_context, impact_scope — any node type
    # Must NOT be inside a bullet list (- Trigger: / * Trigger:) — those are
    # content structure, not standalone metadata labels. Require the label to
    # be at the very start of a line with no bullet prefix.
    for field_name in ('trigger', 'emotional_context', 'impact_scope'):
        label_pattern = field_name.replace('_', '[_ ]')
        fm = re.search(
            r'(?:^|\n)(?:\*{0,2})' + label_pattern +
            r'(?:\*{0,2})\s*:\s*(.+?)(?:\n|$)',
            content, re.IGNORECASE | re.MULTILINE)
        if fm:
            val = fm.group(1).strip().strip('"\'')
            if len(val) > 10:
                fields[field_name] = val[:300]

    return fields


# ── S1E trace analysis ──

def get_s1e_trace_nodes(logs_conn: sqlite3.Connection) -> Dict[str, dict]:
    """Get node IDs referenced in S1E delta traces.

    Returns: {node_id: {chain_id, session_id, created_at, was_created, was_revised}}
    """
    rows = logs_conn.execute("""
        SELECT chain_id, session_id, summary, metadata, created_at
        FROM trace_events
        WHERE scale = 's1' AND event_type = 'delta' AND ref_type = 'encoding_run'
        ORDER BY created_at ASC
    """).fetchall()

    node_map = {}
    for chain_id, session_id, summary, metadata_json, created_at in rows:
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        created_ids = set(meta.get('created', []))
        revised_ids = set(meta.get('revised', []))
        summary_ids = set(re.findall(r'`([a-f0-9]{8})`', summary or ''))

        for nid in (created_ids | revised_ids | summary_ids):
            node_map[nid] = {
                'chain_id': chain_id,
                'session_id': session_id,
                'created_at': created_at,
                'was_created': nid in created_ids,
                'was_revised': nid in revised_ids,
            }

    return node_map


def identify_candidate_nodes(brain_conn: sqlite3.Connection) -> List[dict]:
    """Identify nodes that likely SHOULD have had open fields set.

    These are nodes where:
    - Type is correction/bug and content describes an assumption/reality gap
      but no 'assumed'/'reality' metadata exists
    - Type is moment/identity/lesson and content has emotional language
      but no 'emotional_context' metadata exists
    - Content describes a trigger/activation pattern
      but no 'trigger' metadata exists

    Returns: [{node_id, title, type, missing_fields: [str], evidence: str}]
    """
    # Get all encoder nodes with their metadata keys
    nodes = brain_conn.execute("""
        SELECT n.id, n.type, n.title, n.content
        FROM nodes n
        WHERE n.encoding_source = 'encoder:sonnet'
          AND n.archived = 0
        ORDER BY n.created_at DESC
    """).fetchall()

    meta_keys = {}
    rows = brain_conn.execute("""
        SELECT node_id, key FROM node_metadata_kv
        WHERE node_id IN (
            SELECT id FROM nodes WHERE encoding_source = 'encoder:sonnet' AND archived = 0
        )
    """).fetchall()
    for nid, key in rows:
        meta_keys.setdefault(nid, set()).add(key)

    candidates = []

    for node_id, node_type, title, content in nodes:
        if not content:
            continue

        existing = meta_keys.get(node_id, set())
        missing = []
        evidence_parts = []

        # Correction nodes without assumed/reality
        if node_type in CANDIDATE_TYPES_CORRECTION:
            if 'assumed' not in existing and 'reality' not in existing:
                # Check if content describes an assumption gap
                has_assumption = bool(re.search(
                    r'\b(?:assumed|claimed|thought|believed|wrong|incorrect|'
                    r'mistake|bug|actually|reality|in fact)\b',
                    content, re.IGNORECASE))
                if has_assumption:
                    missing.append('assumed/reality')
                    # Extract a brief evidence snippet
                    for sentence in re.split(r'[.!?\n]+', content[:500]):
                        s = sentence.strip()
                        if re.search(r'\b(?:assumed|claimed|wrong|actually|reality)\b',
                                     s, re.IGNORECASE) and len(s) > 20:
                            evidence_parts.append(s[:100])
                            break

        # Emotional nodes without emotional_context
        if node_type in CANDIDATE_TYPES_EMOTIONAL:
            if 'emotional_context' not in existing:
                emotional_words = re.findall(
                    r'\b(?:frustrated|frustration|angry|hurt|moved|emotional|'
                    r'excited|surprised|disappointed|proud|sad|anxious|'
                    r'vulnerable|felt like|feels like|I feel|honor|gift|'
                    r'losing a partner|relationship)\b',
                    content, re.IGNORECASE)
                if len(emotional_words) >= 2:
                    missing.append('emotional_context')
                    evidence_parts.append('emotional words: %s' % ', '.join(emotional_words[:4]))

        # Any node with clear impact scope language
        if 'impact_scope' not in existing:
            scope_match = re.search(
                r'\b(?:all concurrent|every session|entire pipeline|'
                r'all (?:recall|encoding|surface)|affects everything|'
                r'system-wide|brain-wide)\b',
                content, re.IGNORECASE)
            if scope_match:
                missing.append('impact_scope')
                evidence_parts.append('scope: "%s"' % scope_match.group(0))

        if missing:
            candidates.append({
                'node_id': node_id,
                'title': title,
                'type': node_type,
                'missing_fields': missing,
                'evidence': '; '.join(evidence_parts) if evidence_parts else '',
            })

    return candidates


# ── Reporting ──

def print_report(recoveries: List[dict], candidates: List[dict],
                 verbose: bool = False):
    """Print recovery report."""

    print("\n" + "=" * 70)
    print("RECOVERY REPORT — Open Fields Lost by Dispatch Filter Bug")
    print("=" * 70)

    # Section 1: Recoverable fields
    if recoveries:
        field_counts = {}
        for r in recoveries:
            for k in r['fields']:
                field_counts[k] = field_counts.get(k, 0) + 1

        print("\n--- RECOVERABLE (explicit labels found in content) ---")
        print("Nodes: %d  |  Fields: %d" % (
            len(recoveries),
            sum(len(r['fields']) for r in recoveries)))
        print("\nField distribution:")
        for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
            print("  %-25s  %d nodes" % (field, count))

        print("\nDetails:")
        for r in recoveries:
            print("\n  [%s] %s  (type: %s)" % (
                r['node_id'], r['title'][:55], r['type']))
            if verbose and r.get('trace_chain'):
                print("    trace: %s" % r['trace_chain'])
            for k, v in r['fields'].items():
                display = v if len(v) <= 70 else v[:67] + '...'
                print("    + %-22s = %s" % (k, display))
    else:
        print("\n--- RECOVERABLE ---")
        print("None. The encoding agent passed open fields as tool parameters,")
        print("not as content labels. The dispatch filter dropped them silently")
        print("and the original tool call data is not stored in traces.")

    # Section 2: Candidates for re-encoding
    if verbose and candidates:
        print("\n\n--- CANDIDATES FOR RE-ENCODING ---")
        print("These nodes likely SHOULD have had open fields but don't.")
        print("They cannot be recovered automatically — the original values")
        print("are lost. Options: re-encode via S1E, or add manually via MCP.")
        print("\nTotal candidates: %d" % len(candidates))

        # Group by missing field
        by_field = {}
        for c in candidates:
            for f in c['missing_fields']:
                by_field.setdefault(f, []).append(c)

        for field, nodes in sorted(by_field.items(), key=lambda x: -len(x[1])):
            print("\n  Missing '%s': %d nodes" % (field, len(nodes)))
            for c in nodes[:10]:  # Show at most 10 per field
                evidence = ('  -- %s' % c['evidence']) if c['evidence'] else ''
                print("    [%s] %s%s" % (
                    c['node_id'], c['title'][:50], evidence))
            if len(nodes) > 10:
                print("    ... and %d more" % (len(nodes) - 10))

    elif verbose:
        print("\n\n--- CANDIDATES FOR RE-ENCODING ---")
        print("No strong candidates identified.")

    # Section 3: Summary
    print("\n\n--- IMPACT ASSESSMENT ---")
    print("The dispatch filter bug affected ALL encoder-created nodes.")
    print("Any open field passed by the encoding agent via brain_batch was dropped.")
    print("The encoding agent prompt (v3, line 118) explicitly tells the encoder")
    print("to use open fields: assumed, reality, trigger, emotional_context,")
    print("impact_scope, and any other descriptive key.")
    print("\nSince the tool call inputs are not logged in traces (only action")
    print("summaries and node IDs), the exact values cannot be recovered.")
    print("The fix in daemon_dispatch.py prevents future loss.")
    if candidates:
        print("\n%d nodes are strong candidates for having lost open fields." %
              len(candidates))
        print("Use --dry-run -v to see the full candidate list.")

    print("\n" + "=" * 70)


def apply_recoveries(brain_conn: sqlite3.Connection, recoveries: List[dict]) -> int:
    """Write recovered fields to node_metadata_kv. Returns count written."""
    total = 0
    for r in recoveries:
        for k, v in r['fields'].items():
            brain_conn.execute(
                'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) '
                'VALUES (?, ?, ?)',
                (r['node_id'], k, v))
            total += 1
    brain_conn.commit()
    return total


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description='Recover lost open metadata fields from S1E encoding')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true',
                       help='Report what would be recovered (no writes)')
    group.add_argument('--apply', action='store_true',
                       help='Write recovered fields to brain.db (backs up first)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show candidates for re-encoding and trace details')
    args = parser.parse_args()

    if not os.path.exists(BRAIN_DB):
        print("ERROR: brain.db not found at %s" % BRAIN_DB, file=sys.stderr)
        print("Set BRAIN_DB_DIR if brain is elsewhere.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(LOGS_DB):
        print("ERROR: brain_logs.db not found at %s" % LOGS_DB, file=sys.stderr)
        sys.exit(1)

    print("Brain DB: %s" % BRAIN_DB)
    print("Logs DB:  %s" % LOGS_DB)

    brain_conn = sqlite3.connect(BRAIN_DB)
    brain_conn.execute("PRAGMA journal_mode=WAL")
    logs_conn = sqlite3.connect(LOGS_DB)
    logs_conn.execute("PRAGMA journal_mode=WAL")

    try:
        # Baseline counts
        total_encoder = brain_conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE encoding_source = 'encoder:sonnet' AND archived = 0"
        ).fetchone()[0]
        total_meta = brain_conn.execute(
            "SELECT COUNT(*) FROM node_metadata_kv"
        ).fetchone()[0]
        print("\nEncoder nodes (non-archived): %d" % total_encoder)
        print("Existing metadata entries:     %d" % total_meta)

        # Phase 1: Scan for recoverable fields in content
        print("\nPhase 1: Scanning content for explicit open-field labels...")
        s1e_nodes = get_s1e_trace_nodes(logs_conn)

        nodes = brain_conn.execute("""
            SELECT id, type, title, content, created_at
            FROM nodes
            WHERE encoding_source = 'encoder:sonnet' AND archived = 0
            ORDER BY created_at ASC
        """).fetchall()

        existing_meta = {}
        for nid, key in brain_conn.execute("""
            SELECT node_id, key FROM node_metadata_kv
            WHERE node_id IN (
                SELECT id FROM nodes WHERE encoding_source = 'encoder:sonnet' AND archived = 0
            )
        """).fetchall():
            existing_meta.setdefault(nid, set()).add(key)

        recoveries = []
        for node_id, node_type, title, content, created_at in nodes:
            if not content:
                continue

            node_existing = existing_meta.get(node_id, set())
            fields = extract_open_fields_from_content(content, node_type)

            # Remove fields that already exist
            fields = {k: v for k, v in fields.items() if k not in node_existing}

            if fields:
                trace_info = s1e_nodes.get(node_id, {})
                recoveries.append({
                    'node_id': node_id,
                    'title': title,
                    'type': node_type,
                    'created_at': created_at,
                    'fields': fields,
                    'trace_chain': trace_info.get('chain_id', ''),
                })

        # Phase 2: Identify candidates that likely lost open fields
        print("Phase 2: Identifying candidates for re-encoding...")
        candidates = identify_candidate_nodes(brain_conn)

        # Report
        print_report(recoveries, candidates, verbose=args.verbose)

        # Apply if requested
        if args.apply and recoveries:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            backup_path = BRAIN_DB + '.bak-' + timestamp
            print("\nBacking up brain.db to %s..." % backup_path)
            shutil.copy2(BRAIN_DB, backup_path)
            print("Backup: %d bytes" % os.path.getsize(backup_path))

            total_written = apply_recoveries(brain_conn, recoveries)
            new_meta = brain_conn.execute(
                "SELECT COUNT(*) FROM node_metadata_kv"
            ).fetchone()[0]
            print("\nWritten: %d fields across %d nodes" % (
                total_written, len(recoveries)))
            print("Metadata entries: %d -> %d (+%d)" % (
                total_meta, new_meta, new_meta - total_meta))

        elif args.apply:
            print("\nNothing recoverable to apply.")

        elif args.dry_run:
            total_fields = sum(len(r['fields']) for r in recoveries)
            if total_fields:
                print("\nDry run: would write %d fields across %d nodes." % (
                    total_fields, len(recoveries)))
            else:
                print("\nDry run: nothing recoverable from content patterns.")

    finally:
        brain_conn.close()
        logs_conn.close()


if __name__ == '__main__':
    main()
