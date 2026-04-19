"""S2 Rejection Table — fingerprint-based suppression for integration units.

Solves the re-proposal bug: when an encoder rejects a proposal, nothing in
the graph changes, so the decoder keeps producing the same proposal every
run. Without a rejection memory, the encoder judges the same thing forever.

Design:
- Proposal fingerprint = hash of parameters the encoder actually judges
  (not implementation artifacts). When graph state changes in a way that
  would alter the proposal's inputs, the fingerprint changes and the old
  rejection no longer applies.
- Table `s2_rejections` in brain.db holds (fingerprint, integration_unit,
  proposal_type, proposed_ids, created_at). Proposed IDs enable S3 pattern
  analysis.
- Table schema lives in servers/schema.py (single source of truth).
  ensure_schema() creates it at Brain startup — do not CREATE TABLE here.

Used by S2 Community Detection and Consolidation. Pattern for any S2 unit:
- Informational outcomes (CONSOLIDATE/EVOLVE/KEEP/community placement)
  write semantic edges — state is discoverable from the graph via JOIN.
- Marker-only outcomes (SKIP, "looked at, didn't act") record a fingerprint
  here so the decoder filters the proposal on subsequent runs.

Decoder-side: call filter_rejected(brain, proposals) before sending to encoder.
Encoder-side: call record_rejections(brain, skipped, integration_unit=...)
after the encoder decides a proposal isn't worth a semantic edge.
"""
import hashlib
import json
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# FINGERPRINT COMPUTATION
# ═══════════════════════════════════════════════════════════════
#
# Each fingerprint captures what the encoder actually judges on.
# Implementation artifacts (internal fraction, shared_count, exact
# affinity) are excluded so proposals aren't falsely invalidated by
# cosmetic changes. When real inputs change (community grew toward
# the node, cluster composition changed, drift thresholds raised),
# the fingerprint naturally changes.

def compute_fingerprint(proposal):
    """Stable fingerprint for a proposal.

    - add_to_existing: node_id + community_id + affinity_tier
      (tier captures the regime where encoder judgment would change:
      borderline <0.40, moderate <0.65, strong >=0.65)
    - new_community: top 40% of representative IDs (structural hubs
      define cluster identity; peripheral member changes don't
      invalidate the rejection)
    - drift: node_id + foreign_community_id
    - health_update: community_id + signal type (dead/degrading/maturing)
    - merge_communities: larger_id + smaller_id
    """
    ptype = proposal.get('type', '')

    if ptype == 'add_to_existing':
        aff = proposal.get('affinity', 0)
        if aff >= 0.65:
            tier = 'strong'
        elif aff >= 0.40:
            tier = 'moderate'
        else:
            tier = 'borderline'
        raw = 'add:%s:%s:%s' % (
            proposal.get('node_id', ''),
            proposal.get('community_id', ''),
            tier)

    elif ptype == 'new_community':
        # Structural hubs (top 40% by internal edges) define cluster identity.
        # Falls back to sorted member IDs if representatives not populated.
        reps = proposal.get('representatives', [])
        if reps:
            rep_ids = sorted(r.get('id', '') for r in reps if r.get('id'))
            n_keep = max(2, int(len(rep_ids) * 0.4 + 0.5))
            core = rep_ids[:n_keep]
        else:
            members = sorted(proposal.get('members', []))
            n_keep = max(2, int(len(members) * 0.4 + 0.5))
            core = members[:n_keep]
        raw = 'new:' + ':'.join(core)

    elif ptype == 'drift':
        foreign = proposal.get('foreign', [{}])
        foreign_id = foreign[0].get('id', '') if foreign else ''
        raw = 'drift:%s:%s' % (proposal.get('node_id', ''), foreign_id)

    elif ptype == 'health_update':
        raw = 'health:%s:%s' % (
            proposal.get('community_id', ''),
            proposal.get('signal', ''))

    elif ptype == 'merge_communities':
        raw = 'merge:%s:%s' % (
            proposal.get('larger_id', ''),
            proposal.get('smaller_id', ''))

    elif ptype == 'consolidation_cluster':
        # Cluster identity = sorted (member_id, member_updated_at) pairs.
        # Members alone aren't enough: a small content edit can leave
        # similarity above threshold so the cluster re-surfaces with the
        # same IDs, and the old rejection would block re-evaluation even
        # though the content the encoder would judge has changed.
        # Including updated_at makes any revise()/remember() on a member
        # invalidate the rejection automatically — no bookkeeping needed.
        members = sorted(proposal.get('members', []))
        updated_at = proposal.get('member_updated_at')

        ts_by_id = None
        if isinstance(updated_at, dict) and updated_at:
            ts_by_id = updated_at
        elif (isinstance(updated_at, list)
              and len(updated_at) == len(proposal.get('members', []))
              and updated_at):
            ts_by_id = dict(zip(proposal.get('members', []), updated_at))
        # else: absent, empty, or malformed → fall back to id-only format
        # so pre-existing fingerprints (written before this change) still
        # match legacy proposals lacking the field.

        if ts_by_id is not None:
            parts = [f'{m}|{ts_by_id.get(m, "")}' for m in members]
        else:
            parts = members
        raw = 'consol:' + ':'.join(parts)

    else:
        raw = '%s:%s' % (ptype, proposal.get('node_id', ''))

    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_proposed_ids(proposal):
    """Extract all node IDs involved in a proposal (for S3 analysis)."""
    ids = []
    if proposal.get('node_id'):
        ids.append(proposal['node_id'])
    if proposal.get('members'):
        ids.extend(proposal['members'])
    if proposal.get('community_id'):
        ids.append(proposal['community_id'])
    if proposal.get('larger_id'):
        ids.append(proposal['larger_id'])
    if proposal.get('smaller_id'):
        ids.append(proposal['smaller_id'])
    for f in proposal.get('foreign', []):
        if isinstance(f, dict) and f.get('id'):
            ids.append(f['id'])
    return ids


# ═══════════════════════════════════════════════════════════════
# FILTERING AND RECORDING
# ═══════════════════════════════════════════════════════════════

def filter_rejected(brain, proposals):
    """Remove proposals that match a previous rejection fingerprint.

    Returns (surviving_proposals, suppressed_count).
    """
    if not proposals:
        return proposals, 0

    fps_with_proposal = [(p, compute_fingerprint(p)) for p in proposals]
    all_fp = [fp for _, fp in fps_with_proposal]

    rejected_set = set()
    for chunk_start in range(0, len(all_fp), 900):
        chunk = all_fp[chunk_start:chunk_start + 900]
        if not chunk:
            continue
        placeholders = ','.join('?' * len(chunk))
        rows = brain.conn.execute(
            "SELECT fingerprint FROM s2_rejections WHERE fingerprint IN (%s)" % placeholders,
            chunk).fetchall()
        for row in rows:
            rejected_set.add(row[0])

    surviving = []
    suppressed = 0
    for p, fp in fps_with_proposal:
        if fp in rejected_set:
            suppressed += 1
        else:
            surviving.append(p)
    return surviving, suppressed


def record_rejections(brain, proposals, integration_unit='s2:community_detection'):
    """Write rejected proposal fingerprints to s2_rejections.

    Uses INSERT OR IGNORE — duplicate fingerprints from repeated rejection
    are silently deduplicated.
    """
    if not proposals:
        return 0
    ts = datetime.now(timezone.utc).isoformat()
    count = 0
    for p in proposals:
        fp = compute_fingerprint(p)
        ids_json = json.dumps(get_proposed_ids(p))
        brain.conn.execute(
            "INSERT OR IGNORE INTO s2_rejections "
            "(fingerprint, integration_unit, proposal_type, proposed_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fp, integration_unit, p.get('type', ''), ids_json, ts))
        count += 1
    brain.conn.commit()
    return count


# ═══════════════════════════════════════════════════════════════
# MATCHER — proposal → encoder action
# ═══════════════════════════════════════════════════════════════

def match_proposals_to_actions(sent_proposals, action_details):
    """Walk brain_batch operations to determine which proposals were acted on.

    Returns (acted_on, skipped) lists. A proposal is "acted on" if the
    encoder made ANY operation targeting its constituent nodes (success
    or failure — the encoder still saw and judged it).

    Matching rules per proposal type:
    - new_community: remember op (type=community) whose connection target_ids
      overlap >= 50% with the proposal's member set
    - add_to_existing: connect op (community_member) with matching (source, target)
    - drift (accept): connect op to foreign community
    - drift (reject): revise op with _sys_drift_threshold on the node
    - health_update: archive or revise (community_maturity) on the community
    - merge_communities: archive op on smaller_id, or revise on larger_id
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
                        if p.get('node_id') == tgt:
                            for f in p.get('foreign', []):
                                if isinstance(f, dict) and f.get('id') == src:
                                    acted_idx.add(i)
                                    break

            elif op == 'revise':
                nid = op_spec.get('node_id')
                if not nid:
                    continue
                if '_sys_drift_threshold' in op_spec:
                    for i, p in enumerate(sent_proposals):
                        if p.get('type') == 'drift' and p.get('node_id') == nid:
                            acted_idx.add(i)
                if 'community_maturity' in op_spec:
                    for i, p in enumerate(sent_proposals):
                        if (p.get('type') == 'health_update'
                                and p.get('community_id') == nid):
                            acted_idx.add(i)
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
# PRIORITY ORDERING
# ═══════════════════════════════════════════════════════════════
#
# TYPE_PRIORITY lives in community_contract.py (source of truth).
# Community-specific: the proposal types belong to the community unit.

from .community_contract import TYPE_PRIORITY  # noqa: E402


def sort_proposals_by_priority(proposals):
    """Sort proposals by type priority, then by confidence within type (descending)."""
    def _sort_key(p):
        type_rank = TYPE_PRIORITY.get(p.get('type', ''), 99)
        ptype = p.get('type', '')
        if ptype == 'new_community':
            confidence = p.get('internal_fraction', 0)
        elif ptype == 'add_to_existing':
            confidence = p.get('affinity', 0)
        elif ptype == 'merge_communities':
            confidence = p.get('overlap_pct', 0)
        elif ptype == 'health_update':
            signal = p.get('signal', '')
            confidence = {'dead': 1.0, 'degrading': 0.5, 'corridor_maturing': 0.3}.get(signal, 0)
        elif ptype == 'drift':
            foreign = p.get('foreign', [{}])
            confidence = foreign[0].get('affinity', 0) if foreign else 0
        else:
            confidence = 0
        return (type_rank, -confidence)

    return sorted(proposals, key=_sort_key)
