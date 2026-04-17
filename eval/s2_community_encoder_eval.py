#!/usr/bin/env python3
"""S2 Community ENCODER Eval — Real Sonnet, rerun architecture.

Runs the full decoder→encoder→decoder→encoder loop with:
1. Fingerprint suppression between cycles
2. Priority-ordered proposals (merge > new_community > add_to_existing > health > drift)
3. Richer encoder context (inline representative content, no get_nodes round trip)
4. Full decoder rerun between encoder cycles (fresh proposals against current graph)

Usage:
    python3 eval/s2_community_encoder_eval.py                    # 2 cycles
    python3 eval/s2_community_encoder_eval.py --cycles 3         # 3 cycles
    python3 eval/s2_community_encoder_eval.py --max-proposals 20 # proposals per cycle
    python3 eval/s2_community_encoder_eval.py --keep             # Keep temp dir
    python3 eval/s2_community_encoder_eval.py --save report.json
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse decoder and fingerprint infrastructure from the decoder eval
from eval.s2_community_decoder_eval import (
    create_rejection_table,
    compute_fingerprint,
    get_proposed_ids,
    filter_rejected,
    record_rejections,
    run_new_decoder,
    compute_cross_run_metrics,
)


# ═══════════════════════════════════════════════════════════════
# RICHER NODE FORMAT — more generous for 20-proposal batches
# ═══════════════════════════════════════════════════════════════

S2CE_RICH_NODE_FORMAT = {
    'content_limit': 500,       # Up from 300 — encoder needs the gist
    'edge_limit': 5,            # Up from 4 — relations are the story
    'metadata_limit': 200,      # Up from 150
    'time_format': 'relative',
}

S2CE_RICH_COMMUNITY_FORMAT = {
    'content_limit': 300,       # Up from 150 — brief but informative
    'edge_limit': 0,
    'metadata_limit': 100,      # Show key community metadata
    'time_format': 'relative',
}


# ═══════════════════════════════════════════════════════════════
# PRIORITY ORDERING
# ═══════════════════════════════════════════════════════════════

TYPE_PRIORITY = {
    'merge_communities': 0,     # Simplifies landscape first
    'new_community': 1,         # Creates structure for future Phase 1
    'add_to_existing': 2,       # Extends communities at frontier
    'health_update': 3,         # Corrects metadata
    'drift': 4,                 # Smallest structural impact
}


def sort_proposals(proposals):
    """Sort by type priority, then by confidence metric within type."""
    def _sort_key(p):
        type_rank = TYPE_PRIORITY.get(p['type'], 99)

        # Within-type ranking (higher = better, negate for descending)
        if p['type'] == 'new_community':
            confidence = p.get('internal_fraction', 0)
        elif p['type'] == 'add_to_existing':
            confidence = p.get('affinity', 0)
        elif p['type'] == 'merge_communities':
            confidence = p.get('overlap_pct', 0)
        elif p['type'] == 'health_update':
            # Dead communities first, then degrading
            signal = p.get('signal', '')
            confidence = {'dead': 1.0, 'degrading': 0.5, 'corridor_maturing': 0.3}.get(signal, 0)
        elif p['type'] == 'drift':
            foreign = p.get('foreign', [{}])
            confidence = foreign[0].get('affinity', 0) if foreign else 0
        else:
            confidence = 0

        return (type_rank, -confidence)

    return sorted(proposals, key=_sort_key)


def match_proposals_to_actions(sent_proposals, action_details):
    """Walk brain_batch operations to determine which proposals the encoder acted on.

    Returns (acted_on, skipped) lists. A proposal is "acted on" if the encoder
    made ANY operation targeting its constituent nodes (even if that op failed
    — the encoder still saw and judged it).

    Matching rules per proposal type:
    - new_community: remember op (type=community) whose connection target_ids
      overlap >= 50% with the proposal's member set
    - add_to_existing: connect op (community_member) with matching (source, target)
    - drift (accept): connect op to foreign community
    - drift (reject): revise op with _sys_drift_threshold on the node
    - health_update: archive or revise (community_maturity) on the community
    - merge_communities: archive op on smaller_id
    """
    acted_idx = set()

    for action in action_details:
        if action.get('tool') != 'brain_batch':
            continue
        operations = action.get('input', {}).get('operations', [])
        for op_spec in operations:
            if not isinstance(op_spec, dict):
                continue
            op = op_spec.get('op', '')

            if op == 'remember' and op_spec.get('type') == 'community':
                # new_community match: >= 50% of proposal members in connections
                conn_targets = {
                    c.get('target_id') for c in op_spec.get('connections', [])
                    if isinstance(c, dict)
                    and c.get('relation') == 'community_member'
                    and c.get('target_id')
                }
                if not conn_targets:
                    continue
                for i, p in enumerate(sent_proposals):
                    if p.get('type') != 'new_community':
                        continue
                    members = set(p.get('members', []))
                    if not members:
                        continue
                    overlap = len(conn_targets & members) / len(members)
                    if overlap >= 0.5:
                        acted_idx.add(i)

            elif op == 'connect' and op_spec.get('relation') == 'community_member':
                src = op_spec.get('source_id')
                tgt = op_spec.get('target_id')
                if not (src and tgt):
                    continue
                for i, p in enumerate(sent_proposals):
                    if p.get('type') == 'add_to_existing':
                        if p.get('community_id') == src and p.get('node_id') == tgt:
                            acted_idx.add(i)
                    elif p.get('type') == 'drift':
                        # Drift accept: connected to foreign community
                        if p.get('node_id') == tgt:
                            for f in p.get('foreign', []):
                                if isinstance(f, dict) and f.get('id') == src:
                                    acted_idx.add(i)
                                    break

            elif op == 'revise':
                nid = op_spec.get('node_id')
                if not nid:
                    continue
                # drift rejection: threshold raise on the node
                if '_sys_drift_threshold' in op_spec:
                    for i, p in enumerate(sent_proposals):
                        if p.get('type') == 'drift' and p.get('node_id') == nid:
                            acted_idx.add(i)
                # health_update: maturity change on the community
                if 'community_maturity' in op_spec:
                    for i, p in enumerate(sent_proposals):
                        if (p.get('type') == 'health_update'
                                and p.get('community_id') == nid):
                            acted_idx.add(i)
                # merge: revise on larger (narrative update)
                for i, p in enumerate(sent_proposals):
                    if (p.get('type') == 'merge_communities'
                            and p.get('larger_id') == nid):
                        acted_idx.add(i)

            elif op == 'archive':
                nid = op_spec.get('node_id')
                if not nid:
                    continue
                for i, p in enumerate(sent_proposals):
                    if p.get('type') == 'health_update' and p.get('community_id') == nid:
                        acted_idx.add(i)
                    elif (p.get('type') == 'merge_communities'
                            and p.get('smaller_id') == nid):
                        acted_idx.add(i)

    acted_on = [sent_proposals[i] for i in sorted(acted_idx)]
    skipped = [p for i, p in enumerate(sent_proposals) if i not in acted_idx]
    return acted_on, skipped


# ═══════════════════════════════════════════════════════════════
# RICHER PROPOSAL FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_proposals_rich(brain, proposals):
    """Format proposals with inline representative content.

    For new_community: render top 3 representative nodes inline.
    For add_to_existing: render the node + community narrative.
    No get_nodes() round trip needed — everything is in the prompt.
    """
    from servers.contract import render_rich_node
    from servers.pipeline_contract import get_rich_node
    from servers.scales.s1.surface_contract import _relative_time

    lines = ['PROPOSALS (%d):\n' % len(proposals)]

    for i, prop in enumerate(proposals):
        ptype = prop['type'].upper().replace('_', ' ')
        lines.append('[%d] %s' % (i + 1, ptype))

        if prop['type'] == 'new_community':
            lines.append('    %d members, int_frac=%.0f%%' % (
                prop.get('member_count', 0),
                prop.get('internal_fraction', 0) * 100))

            sig = prop.get('edge_signature', {})
            if sig:
                parts = ['%s(%.0f%%)' % (f, p * 100)
                         for f, p in sorted(sig.items(), key=lambda x: -x[1])[:4]]
                lines.append('    Signature: %s' % ', '.join(parts))

            # Sample internal edges — show the relationships
            for se in prop.get('sample_edges', []):
                lines.append('    Edge: "%s" %s "%s" — %s' % (
                    se.get('source', '?'), se.get('relation', '?'),
                    se.get('target', '?'), se.get('description', '')[:80]))

            # All members (compact)
            all_members = prop.get('all_members', [])
            if all_members:
                lines.append('    Members (%d):' % len(all_members))
                for m in all_members:
                    age = _relative_time(m.get('date', '')) or m.get('date', '?')
                    lines.append('      [%s] "%s" (id:%s, %s)' % (
                        m.get('type', '?'), m.get('title', '?'),
                        m.get('id', '?')[:8], age))

            # Inline representative nodes — the key decision context
            reps = prop.get('representatives', [])
            if reps:
                rep_ids = [r['id'] for r in reps[:3]]
                rich_nodes = get_rich_node(brain, rep_ids)
                if rich_nodes:
                    lines.append('    Key members (full context):')
                    for rid in rep_ids:
                        if rid in rich_nodes:
                            rendered = render_rich_node(
                                rich_nodes[rid], S2CE_RICH_NODE_FORMAT)
                            # Indent each line
                            for line in rendered.split('\n'):
                                lines.append('      %s' % line)
                            lines.append('')

        elif prop['type'] == 'add_to_existing':
            node_id = prop.get('node_id', '?')
            lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                prop.get('node_type', '?'),
                prop.get('node_title', '?'),
                node_id[:8] if node_id else '?'))

            # Inline the node's content for better judgment
            if node_id and node_id != '?':
                rich = get_rich_node(brain, node_id)
                if rich:
                    rendered = render_rich_node(rich, S2CE_RICH_NODE_FORMAT)
                    for line in rendered.split('\n'):
                        lines.append('      %s' % line)

            if prop.get('source') == 'overlap_check':
                lines.append('    (Algorithmic placement)')

            comm_id = prop.get('community_id')
            comm_title = prop.get('community_title', '?')
            aff = prop.get('affinity', 0)
            lines.append('    → connect to "%s" (community_id: %s, affinity: %.0f%%)' % (
                comm_title, comm_id[:8] if comm_id else '?', aff * 100))

            # Brief community narrative for context
            if comm_id:
                comm_rich = get_rich_node(brain, comm_id)
                if comm_rich:
                    rendered = render_rich_node(comm_rich, S2CE_RICH_COMMUNITY_FORMAT)
                    lines.append('    Community context:')
                    for line in rendered.split('\n')[:5]:
                        lines.append('      %s' % line)

        elif prop['type'] == 'drift':
            node_id = prop.get('node_id', '?')
            lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                prop.get('node_type', '?'),
                prop.get('node_title', '?'),
                node_id[:8] if node_id else '?'))
            lines.append('    Home: "%s" (affinity: %.0f%%)' % (
                prop.get('home_community', '?'),
                prop.get('home_affinity', 0) * 100))
            lines.append('    Current threshold: %.1fx' % (
                prop.get('current_drift_threshold', 1.5)))
            for f in prop.get('foreign', []):
                lines.append('    Drifting toward: "%s" (community_id: %s, affinity: %.0f%%)' % (
                    f.get('title', '?'),
                    f.get('id', '?')[:8],
                    f.get('affinity', 0) * 100))

        elif prop['type'] == 'health_update':
            lines.append('    Community: "%s" (community_id: %s)' % (
                prop.get('community_title', '?'),
                prop.get('community_id', '?')[:8]))
            lines.append('    Signal: %s (int_frac %.2f → %.2f)' % (
                prop.get('signal', '?'),
                prop.get('old_fraction', 0),
                prop.get('new_fraction', 0)))

        elif prop['type'] == 'merge_communities':
            lines.append('    Larger: "%s" (%d members, id:%s)' % (
                prop.get('larger_title', '?'),
                prop.get('larger_size', 0),
                prop.get('larger_id', '?')[:8]))
            lines.append('    Smaller: "%s" (%d members, id:%s)' % (
                prop.get('smaller_title', '?'),
                prop.get('smaller_size', 0),
                prop.get('smaller_id', '?')[:8]))
            lines.append('    Overlap: %d shared (%.0f%% of smaller), %d unique in smaller' % (
                prop.get('shared_count', 0),
                prop.get('overlap_pct', 0) * 100,
                prop.get('unique_in_smaller', 0)))

        lines.append('')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# ENCODER WRAPPER — uses real Sonnet
# ═══════════════════════════════════════════════════════════════

HAIKU_COMMUNITY_PROMPT = """You are a community encoder for a persistent brain shared between an operator (Tom) and an AI assistant (Anchor).

Read the proposals. Call brain_batch to act on them. Every proposal needs a decision: accept (create/connect) or reject (skip and note in journal).

## Rules
- Round 1 (optional): call get_nodes() to inspect specific members before deciding.
- Round 2: call brain_batch with ALL your actions for ALL proposals. One call.
- After brain_batch: write a short journal and DONE.
- Do NOT analyze or explain. Tool calls then journal.

## NEW COMMUNITY

Community nodes are first-class nodes — they use the same fields as any other node. No special prefixed fields.

Example:
```
brain_batch({operations: [
  {op: "remember", type: "community",
   title: "Hook Latency: From Timeouts to Daemon Profiling",
   content: "The hook latency story begins with 14s recall timeouts (id:577119fd) and traces through daemon profiling (id:854b4bc3). The pattern: every investigation started at the hook layer but the root cause was always deeper — daemon thread pool, embedder cold start, or API latency. Lesson: hook timeouts are symptoms, not causes.",
   situation: "When debugging response latency or hook timeouts — start at the daemon, not the hook",
   keywords: "hook latency timeout daemon profiling",
   confidence: 0.85, auto_connect: false,
   connections: [
     {target_id: "577119fd", relation: "community_member", weight: 0.3},
     {target_id: "0fce53be", relation: "community_member", weight: 0.3},
     {target_id: "854b4bc3", relation: "community_member", weight: 0.3}
   ],
   community_key_decisions: "577119fd: Hook pipeline latency, 854b4bc3: Gemma resolution",
   community_members: "577119fd: Hook pipeline latency, 0fce53be: 20s root cause, 854b4bc3: Gemma resolution",
   community_maturity: "settled",
   community_dominant_type: "finding",
   community_size: "3", community_internal_fraction: "0.89", community_is_corridor: "false"}
]})
```

Key rules:
- ALL members from the proposal go in `connections` — not just a few, ALL of them
- Reference node IDs in content as `(id:XXXXXXXX)`
- `auto_connect: false` always
- `content` is INSIGHT, not summary. What pattern do these nodes reveal together that no single node names?
- `situation` answers: "When would a future Anchor need this community?"
- Do NOT write community_narrative or community_open_questions — content IS the narrative, and a healer process fills question/situation later

Community-specific metadata (string values):
- `community_key_decisions` — "id: title" pairs (3-5 defining nodes)
- `community_members` — ALL member IDs as "id: title" pairs
- `community_latest_development` — newest node + what it means for trajectory (one sentence with id)
- `community_maturity` — "forming" / "active" / "settled" / "corridor"
- `community_dominant_type` — most common node type
- `community_size`, `community_internal_fraction`, `community_is_corridor` — structural

## ADD TO EXISTING

Connect node to community. Update size.

Example:
```
brain_batch({operations: [
  {op: "connect", source_id: "comm1234", target_id: "node5678", relation: "community_member", weight: 0.3},
  {op: "revise", node_id: "comm1234", reason: "member added: Node Title", community_size: "15"}
]})
```

## DRIFT

Move node to new community (accept) or raise threshold (reject).

Accept: `{op: "connect", source_id: "<new_community>", target_id: "<node_id>", relation: "community_member", weight: 0.3}`
Reject: `{op: "revise", node_id: "<node_id>", reason: "drift rejected", _sys_drift_threshold: "<current + 0.1>"}`

## HEALTH UPDATE

Dead (int_frac<5%): `{op: "archive", node_id: "<community_id>", reason: "dead — members dispersed"}`
Degrading: `{op: "revise", node_id: "<community_id>", reason: "health update", community_maturity: "forming"}`
Maturing corridor: `{op: "revise", node_id: "<community_id>", reason: "corridor maturing", community_maturity: "active"}`

## MERGE

Absorb smaller into larger. Combined narrative. Archive smaller.

Example:
```
brain_batch({operations: [
  {op: "revise", node_id: "larger_id", reason: "merged with Smaller Title",
   content: "Combined narrative...", community_size: "25"},
  {op: "connect", source_id: "larger_id", target_id: "unique_member", relation: "community_member", weight: 0.3},
  {op: "archive", node_id: "smaller_id", reason: "merged into Larger Title"}
]})
```

## Journal (after ALL tool calls)

```
ACCEPTED: [community titles with member counts]
REJECTED: [what and why — one line each]
DONE
```"""


HAIKU_RICH_PROMPT = """Community encoder for a persistent brain shared with operator Tom.

Every proposal gets a decision: accept (tool call) or reject (journal line).

2 rounds max: ONE optional get_nodes call (with ALL the IDs you need in one list), then brain_batch, then journal. DONE.

CRITICAL: Never make parallel get_nodes calls. ONE call with multiple IDs, not multiple calls with one ID each. Multiple parallel calls blow up the context and cause failures.

## NEW COMMUNITY

Community nodes are first-class nodes — they use the same fields as any other node.

Example:
```
brain_batch({operations: [
  {op: "remember", type: "community",
   title: "Hook Latency: From Timeouts to Daemon Profiling",
   content: "The hook latency story begins with 14s recall timeouts (id:577119fd) and traces through daemon profiling (id:854b4bc3). The pattern: every investigation started at the hook layer but the root cause was always deeper — daemon thread pool, embedder cold start, or API latency. Lesson: hook timeouts are symptoms, not causes.",
   situation: "When debugging response latency or hook timeouts — start at the daemon, not the hook",
   keywords: "hook latency timeout daemon profiling",
   confidence: 0.85, auto_connect: false,
   connections: [
     {target_id: "577119fd", relation: "community_member", weight: 0.3},
     {target_id: "0fce53be", relation: "community_member", weight: 0.3},
     {target_id: "854b4bc3", relation: "community_member", weight: 0.3}
   ],
   community_key_decisions: "577119fd: Hook pipeline latency, 854b4bc3: Gemma resolution",
   community_members: "577119fd: Hook pipeline latency, 0fce53be: 20s root cause, 854b4bc3: Gemma resolution",
   community_latest_development: "Gemma resolution (id:854b4bc3) confirmed daemon-level bottleneck — embedder swap resolved the API latency path",
   community_maturity: "settled",
   community_dominant_type: "finding",
   community_size: "3", community_internal_fraction: "0.89", community_is_corridor: "false"}
]})
```

Key rules:
- ALL members from the proposal go in `connections` — not just a few, ALL of them
- Reference node IDs in content as `(id:XXXXXXXX)`
- `auto_connect: false` always
- Do NOT write community_narrative or community_open_questions — content IS the narrative, a healer fills question/situation later
- DO write community_latest_development — the newest node and what it means for this area's trajectory

## Writing Good Titles

Title is the arc in one line. Pattern: "Topic: From X to Y" or "Topic: Principle".

BAD (flat label): "Partnership Philosophy"
GOOD (arc): "Partnership Philosophy: From User-Tool to Shared Identity"

BAD (topic only): "File Reload Patterns"
GOOD (tension named): "File Reload Patterns: The Cost of Blanket Freshness"

## Writing Good Content

Every community content has four moves:

1. OPEN with the arc — "The [topic] story begins with..." or "[Topic] crystallized through..."
2. REFERENCE specific IDs in narrative flow — "(id:xxxxxxxx) traces through (id:yyyyyyyy)"
3. NAME the pattern explicitly — "The pattern: X happens when Y"
4. EXTRACT the lesson — one quotable sentence for future-you

BAD (lists facts): "v5.2 focused on behavior. v5.3 achieved richness. v5.4 migrated to MCP."
GOOD (all four moves): "Encoding behavior can't be specified — it has to be discovered. Every SKILL.md version was a hypothesis that eval either confirmed or killed. The progression from v5.2 (id:41ba0ce4) through v5.3 (id:e4cdab4e) proves: measurement drives behavior, not prompts."

When nodes carry Tom's exact words or reflections — weave them into content. Quotes are highest-signal.

## Latest Development

`community_latest_development` captures what's MOVING in this area — not history, but the newest node and what it represents for the community's trajectory. One sentence, references the latest node's ID.

Example: "Gemma resolution (id:854b4bc3) confirmed daemon-level bottleneck — embedder swap resolved the API latency path"

## Using Proposal Data

**Internal fraction** guides confidence: 80%+ = tight, write confidently. 40-70% = connected but broader. <20% = corridor, forming.
**Edge signature** reveals the story type: extension_refinement = growth arc, correction = learning arc, problem_solution = diagnosis.
**Timeline** spread: same week = single investigation, weeks apart = evolving thread.

## Situation Field

Write situation as: "When [specific trigger] — [what this community teaches]"
Example: "When debugging response latency or hook timeouts — start at the daemon, not the hook"

Community-specific metadata (string values):
- `community_key_decisions` — "id: title" pairs (3-5 defining nodes)
- `community_members` — ALL member IDs as "id: title" pairs
- `community_latest_development` — newest node + what it means for trajectory (one sentence with id)
- `community_maturity` — "forming" / "active" / "settled" / "corridor"
- `community_dominant_type` — most common node type
- `community_size`, `community_internal_fraction`, `community_is_corridor` — structural

## ADD TO EXISTING

Connect node to community. Update size.

```
brain_batch({operations: [
  {op: "connect", source_id: "comm1234", target_id: "node5678", relation: "community_member", weight: 0.3},
  {op: "revise", node_id: "comm1234", reason: "member added: Node Title", community_size: "15"}
]})
```

## DRIFT

Accept: `{op: "connect", source_id: "<new_community>", target_id: "<node_id>", relation: "community_member", weight: 0.3}`
Reject: `{op: "revise", node_id: "<node_id>", reason: "drift rejected", _sys_drift_threshold: "<current + 0.1>"}`

## HEALTH UPDATE

Dead (int_frac<5%): `{op: "archive", node_id: "<community_id>", reason: "dead — members dispersed"}`
Degrading: `{op: "revise", node_id: "<community_id>", reason: "health update", community_maturity: "forming"}`
Maturing: `{op: "revise", node_id: "<community_id>", reason: "corridor maturing", community_maturity: "active"}`

## MERGE

```
brain_batch({operations: [
  {op: "revise", node_id: "larger_id", reason: "merged with Smaller Title",
   content: "Combined narrative...", community_size: "25"},
  {op: "connect", source_id: "larger_id", target_id: "unique_member", relation: "community_member", weight: 0.3},
  {op: "archive", node_id: "smaller_id", reason: "merged into Larger Title"}
]})
```

## Journal (after ALL tool calls)

```
ACCEPTED: [community titles with member counts]
REJECTED: [what and why — one line each]
DONE
```

## YOUR ROLE

You are an ENCODER, not an analyst. Your value is DECISIVE ACTION, not thorough inspection.

- You have 2 rounds. ONE optional get_nodes, then brain_batch. That's it.
- After any get_nodes call, your NEXT response MUST be brain_batch — no more inspection.
- If you're uncertain about a proposal, reject it in the journal with a one-line reason. Do NOT inspect more.
- Every proposal in the batch needs a decision. Accept → tool call. Reject → journal line. No "let me think more."
- The quality guidance above is for HOW you write content when you act — not license to delay action.

Partial action beats complete analysis. A community created with decent content and missing polish will be improved by the healer. A community that was never created because you wanted more inspection is permanently lost."""


def run_encoder(brain, proposals, community_state, max_proposals=20, model=None,
                prompt_variant=None):
    """Run encoder on priority-sorted, fingerprint-filtered proposals.

    Args:
        model: Override model ID (e.g. 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001').
               If None, uses contract default.
    """
    from servers.scales.s2.community_encoder import CommunityEncoder
    from servers.scales.s2.community_contract import COMMUNITY_DETECTION

    config = dict(COMMUNITY_DETECTION)
    config['max_proposals_per_call'] = max_proposals
    config['max_actionable_per_run'] = max_proposals  # Single batch
    config['max_rounds'] = 2  # brain_batch + possible get_nodes if needed
    if model:
        config['model'] = model

    encoder = CommunityEncoder(brain, config=config)

    # Override model and prompt in the interaction config
    if model or prompt_variant:
        import json
        existing = brain.get_interaction_config('s2_community_enrichment') or {}
        if model:
            existing['model'] = model
        existing['max_tokens'] = 32768

        # Select prompt based on variant
        if prompt_variant == 'lean':
            template = HAIKU_COMMUNITY_PROMPT
        elif prompt_variant == 'rich':
            template = HAIKU_RICH_PROMPT
        elif model and 'haiku' in model:
            # Default for Haiku: rich prompt (now production as of v11)
            template = HAIKU_RICH_PROMPT
        else:
            template = brain.get_interaction_prompt('s2_community_enrichment') or ''

        # Write temporarily — IsolatedBrain, so no production impact
        brain._interaction_dal.register(
            's2_community_enrichment',
            template=template,
            parameters=json.dumps(existing),
            created_by='eval:model_override')

    # Sort and cap
    sorted_proposals = sort_proposals(proposals)[:max_proposals]

    model_name = model or config.get('model', 'default')
    print('\n  Encoder [%s] receiving %d proposals (of %d surviving):' % (
        model_name, len(sorted_proposals), len(proposals)))
    type_counts = Counter(p['type'] for p in sorted_proposals)
    for t, c in sorted(type_counts.items(), key=lambda x: TYPE_PRIORITY.get(x[0], 99)):
        print('    %-20s %d' % (t, c))

    t0 = time.time()
    result = encoder.run(sorted_proposals, community_state)
    elapsed = time.time() - t0

    if result:
        result['elapsed_s'] = round(elapsed, 1)
        result['proposals_sent'] = len(sorted_proposals)
        result['proposals_by_type'] = dict(type_counts)
        result['model'] = model_name
    else:
        result = {'elapsed_s': round(elapsed, 1), 'error': 'encoder failed',
                  'proposals_sent': len(sorted_proposals), 'model': model_name}

    return result, sorted_proposals


# ═══════════════════════════════════════════════════════════════
# FULL CYCLE: decode → filter → encode → record rejections
# ═══════════════════════════════════════════════════════════════

def run_cycle(brain, cycle_num, max_proposals=20, config=None, model=None, **kwargs):
    """One full decode→encode cycle.

    Returns dict with decoder stats, encoder stats, and actions taken.
    """
    model_label = model or 'default'
    print('\n' + '=' * 60)
    print('CYCLE %d [%s]' % (cycle_num, model_label))
    print('=' * 60)

    # Decode
    print('\n  Decoding...')
    decode_result = run_new_decoder(brain, config)
    proposals = decode_result['proposals']
    stats = decode_result['stats']

    print('  Unplaced: %d | Communities: %d' % (
        stats['unplaced'], stats['communities']))
    print('  Raw: %d | Suppressed: %d | Surviving: %d' % (
        stats['raw_proposals'], stats['suppressed_count'],
        stats['total_proposals']))
    if stats.get('suppressed_by_type'):
        print('  Suppressed: %s' % '  '.join(
            '%s=%d' % (t, c) for t, c in
            sorted(stats['suppressed_by_type'].items(), key=lambda x: -x[1])))

    if not proposals:
        print('  No proposals after suppression — converged.')
        return {
            'cycle': cycle_num,
            'decoder': stats,
            'encoder': None,
            'converged': True,
        }

    # Encode
    print('\n  Encoding with %s...' % (model or 'default model'))
    encode_result, sent_proposals = run_encoder(
        brain, proposals, decode_result['community_state'],
        max_proposals=max_proposals, model=model,
        prompt_variant=kwargs.get('prompt_variant'))

    actions = encode_result.get('write_actions', 0)
    rounds = encode_result.get('rounds', 0)
    elapsed = encode_result.get('elapsed_s', 0)

    print('\n  Encoder result: %d actions (%d writes) in %d rounds, %.1fs' % (
        encode_result.get('actions', 0), actions, rounds, elapsed))

    # Precise matching: determine which proposals the encoder actually acted on
    # by walking brain_batch operations from action_details. Only stamp the
    # skipped ones. Accepted proposals auto-invalidate on graph change anyway.
    total_sent = len(sent_proposals)
    acted_proposals, skipped_proposals = match_proposals_to_actions(
        sent_proposals, encode_result.get('action_details', []))

    # Detect encoder failure: API error, max_tokens explosion, or no rounds completed.
    # On failure, do NOT stamp — proposals deserve another chance next cycle.
    final_text = encode_result.get('final_text', '') or ''
    encoder_failed = (
        encode_result.get('rounds', 0) == 0 or
        bool(encode_result.get('error')) or
        'FAILED' in final_text or
        'ERROR' in final_text[:200]
    )

    if encoder_failed:
        print('  [!] Encoder failed or incomplete - NOT stamping proposals')
    else:
        print('  Matched: %d acted on, %d skipped (stamping skipped only)' % (
            len(acted_proposals), len(skipped_proposals)))
        if skipped_proposals:
            record_rejections(brain, skipped_proposals)

    return {
        'cycle': cycle_num,
        'decoder': stats,
        'encoder': {
            'proposals_sent': total_sent,
            'proposals_by_type': encode_result.get('proposals_by_type', {}),
            'actions': encode_result.get('actions', 0),
            'write_actions': actions,
            'rounds': rounds,
            'elapsed_s': elapsed,
            'final_text': encode_result.get('final_text', '')[:2000],
            'action_details': encode_result.get('action_details', []),
            'acted_on_count': len(acted_proposals),
            'skipped_count': len(skipped_proposals),
        },
        'converged': False,
    }


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(cycles, brain):
    """Print comprehensive eval report."""
    SEP = '=' * 70

    node_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
    comm_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
    ).fetchone()[0]
    rejection_count = brain.conn.execute(
        "SELECT COUNT(*) FROM s2_rejections").fetchone()[0]

    print('\n' + SEP)
    print('S2 COMMUNITY ENCODER EVAL — SUMMARY')
    print(SEP)

    print('\nFinal state: %d nodes, %d communities' % (node_count, comm_count))
    print('Rejection table: %d entries' % rejection_count)

    # Trajectory
    print('\nPer-cycle trajectory:')
    print('  %-8s %-10s %-10s %-10s %-10s %-10s %s' % (
        'Cycle', 'Unplaced', 'Raw', 'Suppressed', 'Surviving', 'Actions', 'Time'))
    print('  ' + '-' * 70)
    for c in cycles:
        d = c['decoder']
        e = c.get('encoder') or {}
        print('  %-8d %-10d %-10d %-10d %-10d %-10d %s' % (
            c['cycle'],
            d['unplaced'],
            d['raw_proposals'],
            d['suppressed_count'],
            d['total_proposals'],
            e.get('write_actions', 0),
            '%.1fs' % e.get('elapsed_s', 0) if e else 'converged'))

    # Encoder journal excerpts
    for c in cycles:
        e = c.get('encoder')
        if e and e.get('final_text'):
            print('\n--- Cycle %d encoder journal ---' % c['cycle'])
            # Show first 1000 chars of journal
            journal = e['final_text'][:1000]
            for line in journal.split('\n'):
                print('  %s' % line)

    print('\n' + SEP)


def save_report(path, cycles):
    """Save report as JSON."""
    with open(path, 'w') as f:
        json.dump({
            'timestamp': datetime.utcnow().isoformat(),
            'cycles': cycles,
        }, f, indent=2, default=str)
    print('Saved to %s' % path)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='S2 Community Encoder Eval — rerun architecture with model comparison')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of decode→encode cycles (default: 2)')
    parser.add_argument('--max-proposals', type=int, default=20,
                        help='Max proposals per encoder call (default: 20)')
    parser.add_argument('--model', default=None,
                        help='Model override (e.g. claude-sonnet-4-6, claude-haiku-4-5-20251001)')
    parser.add_argument('--ab', action='store_true',
                        help='A/B test: cycle 1 with Sonnet 4.6, cycle 2 with Haiku 4.5')
    parser.add_argument('--ab-prompts', action='store_true',
                        help='A/B test: Haiku lean prompt vs Haiku rich prompt')
    parser.add_argument('--keep', action='store_true',
                        help='Keep temp directory for inspection')
    parser.add_argument('--save', help='Save report to JSON file')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    AB_MODELS = [
        'claude-sonnet-4-6',
        'claude-haiku-4-5-20251001',
    ]

    # Prompt A/B labels
    AB_PROMPTS = {
        'haiku-lean': ('claude-haiku-4-5-20251001', 'lean'),
        'haiku-rich': ('claude-haiku-4-5-20251001', 'rich'),
    }

    print('Setting up isolated brain copy...')
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        print('Isolated brain at: %s' % env.db_dir)

        # Create rejection table
        create_rejection_table(brain)

        cycles = []

        if args.ab:
            # A/B test: decode ONCE on the shared brain, then run each model
            # on a FRESH copy with the same proposals. This ensures both models
            # see identical input.
            print('\n*** A/B TEST: Sonnet 4.6 vs Haiku 4.5 ***')
            print('Decoding once, then each model gets identical proposals.\n')

            # Decode once to get proposals
            print('Decoding proposals...')
            decode_result = run_new_decoder(brain)
            all_proposals = decode_result['proposals']
            community_state = decode_result['community_state']
            stats = decode_result['stats']

            # Sort and cap to the proposals both models will see
            sorted_proposals = sort_proposals(all_proposals)[:args.max_proposals]
            type_counts = Counter(p['type'] for p in sorted_proposals)

            print('  Unplaced: %d | Communities: %d' % (
                stats['unplaced'], stats['communities']))
            print('  Raw: %d | Surviving: %d | Sending: %d' % (
                stats['raw_proposals'], stats['total_proposals'],
                len(sorted_proposals)))
            for t, c in sorted(type_counts.items(),
                               key=lambda x: TYPE_PRIORITY.get(x[0], 99)):
                print('    %-20s %d' % (t, c))

            for model in AB_MODELS:
                # Fresh brain copy — encoder writes don't contaminate the other
                with IsolatedBrain(cleanup=not args.keep) as ab_env:
                    ab_brain = ab_env.brain
                    create_rejection_table(ab_brain)

                    print('\n' + '=' * 60)
                    print('MODEL: %s' % model)
                    print('=' * 60)
                    print('  Isolated at: %s' % ab_env.db_dir)

                    # Re-read community state from this copy (identical to original)
                    from servers.scales.s2.community_decoder import CommunityDecoder
                    ab_decoder = CommunityDecoder(ab_brain)
                    ab_community_state = ab_decoder._read_community_state()

                    print('  Encoding %d proposals with %s...' % (
                        len(sorted_proposals), model))

                    encode_result, sent = run_encoder(
                        ab_brain, all_proposals, ab_community_state,
                        max_proposals=args.max_proposals, model=model)

                    e = encode_result
                    print('\n  Result: %d actions (%d writes) in %d rounds, %.1fs' % (
                        e.get('actions', 0), e.get('write_actions', 0),
                        e.get('rounds', 0), e.get('elapsed_s', 0)))
                    print('  Tokens: %d in / %d out' % (
                        e.get('input_tokens', 0), e.get('output_tokens', 0)))

                    cycles.append({
                        'cycle': 1,
                        'model': model,
                        'decoder': stats,
                        'encoder': e,
                        'converged': False,
                    })

            # Print A/B comparison
            print('\n' + '=' * 70)
            print('A/B COMPARISON')
            print('=' * 70)
            print('\n  %-30s %-20s %-20s' % ('', AB_MODELS[0], AB_MODELS[1]))
            print('  ' + '-' * 70)

            def _get(idx, key, default=0):
                e = cycles[idx].get('encoder') or {}
                return e.get(key, default)

            print('  %-30s %-20s %-20s' % ('Actions (writes)',
                '%d (%d)' % (_get(0, 'actions'), _get(0, 'write_actions')),
                '%d (%d)' % (_get(1, 'actions'), _get(1, 'write_actions'))))
            print('  %-30s %-20s %-20s' % ('Rounds',
                _get(0, 'rounds'), _get(1, 'rounds')))
            print('  %-30s %-20s %-20s' % ('Time',
                '%.1fs' % _get(0, 'elapsed_s'),
                '%.1fs' % _get(1, 'elapsed_s')))
            print('  %-30s %-20s %-20s' % ('Input tokens',
                _get(0, 'input_tokens'), _get(1, 'input_tokens')))
            print('  %-30s %-20s %-20s' % ('Output tokens',
                _get(0, 'output_tokens'), _get(1, 'output_tokens')))
            print('  %-30s %-20s %-20s' % ('Truncations',
                len(_get(0, 'truncations', [])), len(_get(1, 'truncations', []))))

            for i, model in enumerate(AB_MODELS):
                e = cycles[i].get('encoder') or {}
                journal = e.get('final_text', '')
                if journal:
                    print('\n--- %s journal ---' % model)
                    for line in journal[:1500].split('\n'):
                        print('  %s' % line)
        elif args.ab_prompts:
            # Prompt A/B: same model (Haiku), lean vs rich prompt
            print('\n*** PROMPT A/B: Haiku lean vs Haiku rich ***')
            print('Same brain, same proposals, different system prompts.\n')

            # Decode once
            print('Decoding proposals...')
            decode_result = run_new_decoder(brain)
            all_proposals = decode_result['proposals']
            stats = decode_result['stats']
            sorted_proposals = sort_proposals(all_proposals)[:args.max_proposals]
            type_counts = Counter(p['type'] for p in sorted_proposals)

            print('  Unplaced: %d | Sending: %d proposals' % (
                stats['unplaced'], len(sorted_proposals)))
            for t, c in sorted(type_counts.items(),
                               key=lambda x: TYPE_PRIORITY.get(x[0], 99)):
                print('    %-20s %d' % (t, c))

            prompt_variants = [
                ('haiku-lean', 'claude-haiku-4-5-20251001', 'lean'),
                ('haiku-rich', 'claude-haiku-4-5-20251001', 'rich'),
            ]

            for label, model_id, variant in prompt_variants:
                with IsolatedBrain(cleanup=not args.keep) as ab_env:
                    ab_brain = ab_env.brain
                    create_rejection_table(ab_brain)

                    from servers.scales.s2.community_decoder import CommunityDecoder
                    ab_community_state = CommunityDecoder(ab_brain)._read_community_state()

                    print('\n' + '=' * 60)
                    print('PROMPT: %s' % label)
                    print('=' * 60)
                    print('  Isolated at: %s' % ab_env.db_dir)

                    encode_result, sent = run_encoder(
                        ab_brain, all_proposals, ab_community_state,
                        max_proposals=args.max_proposals,
                        model=model_id, prompt_variant=variant)

                    e = encode_result
                    print('\n  Result: %d actions (%d writes) in %d rounds, %.1fs' % (
                        e.get('actions', 0), e.get('write_actions', 0),
                        e.get('rounds', 0), e.get('elapsed_s', 0)))

                    cycles.append({
                        'cycle': 1, 'model': label,
                        'decoder': stats, 'encoder': e,
                        'converged': False,
                    })

            # Print prompt comparison
            print('\n' + '=' * 70)
            print('PROMPT A/B COMPARISON')
            print('=' * 70)
            print('\n  %-25s %-25s %-25s' % ('', 'haiku-lean', 'haiku-rich'))
            print('  ' + '-' * 70)

            def _pget(idx, key, default=0):
                return (cycles[idx].get('encoder') or {}).get(key, default)

            print('  %-25s %-25s %-25s' % ('Actions (writes)',
                '%d (%d)' % (_pget(0, 'actions'), _pget(0, 'write_actions')),
                '%d (%d)' % (_pget(1, 'actions'), _pget(1, 'write_actions'))))
            print('  %-25s %-25s %-25s' % ('Rounds',
                _pget(0, 'rounds'), _pget(1, 'rounds')))
            print('  %-25s %-25s %-25s' % ('Time',
                '%.1fs' % _pget(0, 'elapsed_s'),
                '%.1fs' % _pget(1, 'elapsed_s')))
            print('  %-25s %-25s %-25s' % ('Output tokens',
                _pget(0, 'output_tokens'), _pget(1, 'output_tokens')))

            for i, (label, _, _) in enumerate(prompt_variants):
                e = cycles[i].get('encoder') or {}
                journal = e.get('final_text', '')
                if journal:
                    print('\n--- %s journal ---' % label)
                    for line in journal[:1500].split('\n'):
                        print('  %s' % line)

        else:
            for i in range(1, args.cycles + 1):
                result = run_cycle(
                    brain, i,
                    max_proposals=args.max_proposals,
                    model=args.model)
                cycles.append(result)

                if result.get('converged'):
                    print('\nConverged after %d cycles.' % i)
                    break

        print_report(cycles, brain)

        if args.save:
            save_report(args.save, cycles)

        if args.keep:
            print('\nTemp dir preserved: %s' % env.db_dir)


if __name__ == '__main__':
    main()
