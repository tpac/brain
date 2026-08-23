"""Encoding-quality probes — measure WHAT the encoder wrote, not just
whether the answer was right.

Each probe is read-only against a brain that has just been replayed against
an item. Returns a JSON-shaped dict with a `score` (0-1 where applicable),
the raw measurements, and `evidence` (node ids, snippets) for failure
attachment in reports.

Composable: any probe can be skipped without affecting others. Probes that
need item-side ground truth (brain_presence, specificity_preservation) take
the item dict; structural probes (source_refs_coverage, atomization_shape,
edge_structure, voice_balance) take only the brain.
"""
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Brain inspection helpers
# ─────────────────────────────────────────────────────────────────

def _live_nodes(brain, encoder_only: bool = True) -> List[Dict[str, Any]]:
    """Return non-archived nodes with their kv-metadata (situation, reasoning,
    their_raw_quote, my_raw_quote) attached.

    Nodes table holds the structural columns; per-node free-form fields live
    in `node_metadata_kv` (key-value rows). MetadataDAL.get_all_bulk hydrates
    them in one query.

    `encoder_only=True` filters to nodes that came from haystack replay —
    excludes the 16 seed-pack nodes ('anchor:seed') and any S2 unit work
    ('s2:*') that fired downstream of the encoder. Different encoder
    versions stamp the encoding_source field differently:
      - v19/v21: 'anchor' (the brain.remember default — encoder doesn't set it)
      - v22+   : 'encoder:sonnet' (explicit when set)
    The filter accepts BOTH so probes work uniformly across versions.
    """
    where = "WHERE archived = 0"
    if encoder_only:
        where += (" AND encoding_source != 'anchor:seed' "
                  "AND (encoding_source IS NULL OR encoding_source NOT LIKE 's2:%')")
    rows = brain.conn.execute(
        f"SELECT id, type, title, content, encoding_source, created_at "
        f"FROM nodes {where} ORDER BY created_at DESC"
    ).fetchall()
    if not rows:
        return []
    node_ids = [r[0] for r in rows]
    from servers.dal_metadata import MetadataDAL
    meta_dal = MetadataDAL(brain.conn)
    meta_by_id = meta_dal.get_all_bulk(node_ids)
    out = []
    for r in rows:
        nid, ntype, title, content, enc_source, created_at = r
        meta = meta_by_id.get(nid, {})
        out.append({
            'id': nid, 'type': ntype, 'title': title, 'content': content or '',
            'situation': meta.get('situation', '') or '',
            'reasoning': meta.get('reasoning', '') or '',
            'their_raw_quote': meta.get('their_raw_quote', '') or '',
            'my_raw_quote': meta.get('my_raw_quote', '') or '',
            'encoding_source': enc_source, 'created_at': created_at,
        })
    return out


def _node_source_refs(conn: sqlite3.Connection, node_id: str) -> List[str]:
    rows = conn.execute(
        "SELECT trace_id FROM node_source_refs WHERE node_id = ? "
        "ORDER BY position ASC", (node_id,)).fetchall()
    return [r[0] for r in rows]


def _node_edges(conn: sqlite3.Connection, node_id: str) -> List[Dict[str, Any]]:
    """All non-archived edge relations touching node_id (either direction)."""
    rows = conn.execute(
        "SELECT e.source_id, e.target_id, er.relation, er.description, "
        "       er.encoding_source FROM edges e "
        "JOIN edge_relations er ON er.edge_id = e.edge_id "
        "WHERE (e.source_id = ? OR e.target_id = ?) AND er.archived = 0",
        (node_id, node_id)).fetchall()
    return [
        {'source_id': r[0], 'target_id': r[1], 'relation': r[2],
         'description': r[3] or '', 'encoding_source': r[4] or ''}
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────

def probe_brain_presence(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """Does any encoded node contain the gold answer's atomic value?

    Looks for the gold answer as a substring in node title/content/situation
    /reasoning/voice fields. Returns the best match by string-presence +
    field weight. This is a coarse "is it findable at all" signal; a richer
    embedding-based version is a future extension.

    Returns:
        {found: bool, score: 0-1, best_match: {node_id, field, snippet},
         partial_matches: [...]}
    """
    gold = (item.get('answer') or '').strip()
    if not gold:
        return {'found': False, 'score': 0.0, 'reason': 'no gold answer'}

    # Extract distinctive atoms — words >=4 chars, digit groups, dates
    atoms = set()
    for word in re.findall(r'\b\w{4,}\b', gold):
        atoms.add(word.lower())
    for num in re.findall(r'\d+', gold):
        atoms.add(num)

    nodes = _live_nodes(brain)
    field_weights = {'title': 1.0, 'content': 0.85, 'situation': 0.70,
                     'their_raw_quote': 0.90, 'my_raw_quote': 0.90,
                     'reasoning': 0.40}
    best = {'score': 0.0}
    partials = []
    for n in nodes:
        for field, w in field_weights.items():
            blob = (n.get(field) or '').lower()
            if not blob:
                continue
            matched_atoms = sum(1 for a in atoms if a in blob)
            if matched_atoms == 0:
                continue
            atom_score = matched_atoms / max(1, len(atoms))
            cell_score = atom_score * w
            if cell_score > best['score']:
                best = {
                    'score': cell_score, 'node_id': n['id'], 'field': field,
                    'snippet': blob[:200], 'matched_atoms': matched_atoms,
                    'total_atoms': len(atoms),
                }
            if cell_score > 0.3:
                partials.append({'node_id': n['id'], 'field': field,
                                 'score': cell_score, 'matched_atoms': matched_atoms})

    return {
        'found': best['score'] >= 0.5,
        'score': round(best['score'], 3),
        'gold_answer': gold[:200],
        'gold_atoms_total': len(atoms),
        'best_match': best if best['score'] > 0 else None,
        'partial_matches': partials[:5],
        'nodes_encoded': len(nodes),
    }


def probe_specificity_preservation(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """Are numerics from the haystack preserved (not smoothed)?

    Two preservation paths counted separately:
      - **in_content**: numeric appears verbatim in some encoded node's
        title/content/situation/voice/reasoning. The direct-recall path.
      - **via_substrate**: numeric appears in a trace_events.summary that
        some encoded node's source_refs point at. The joint-reactivation
        path — recoverable via Pure-reference (decision 25).

    `combined_score` is the union — a numeric counts as preserved if EITHER
    path holds it. This is the architecturally honest measurement: source_refs
    aren't lossy storage, they're indexed pointers; recall renders both.

    Reports both individually so the encoder's atomization-vs-pure-reference
    judgment is visible in the data, not collapsed.
    """
    haystacks = item.get('haystack_sessions') or []
    haystack_text = ''
    for session in haystacks:
        for turn in session:
            haystack_text += (turn.get('content') or '') + '\n'

    numerics = set()
    numerics.update(re.findall(r'\b\d+\.\d+\b', haystack_text))
    numerics.update(re.findall(r'\b\d+\s*[-–]\s*\d+\b', haystack_text))
    numerics.update(re.findall(r'\b\d+\s*%\b', haystack_text))
    numerics.update(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', haystack_text))
    numerics.update(re.findall(r'\b\d{1,4}\b', haystack_text))
    numerics = {n for n in numerics if not (n.isdigit() and len(n) == 1)}

    if not numerics:
        return {'score': 1.0, 'in_content': 0, 'via_substrate': 0,
                'combined': 0, 'total': 0, 'note': 'no numerics in haystack'}

    nodes = _live_nodes(brain)
    node_blob = '\n'.join(
        n.get(f) or '' for n in nodes
        for f in ('title', 'content', 'situation', 'their_raw_quote',
                  'my_raw_quote', 'reasoning'))

    # in-content path
    in_content = {n for n in numerics if n in node_blob}

    # via-substrate path — gather trace_events.summary for every trace_id
    # referenced by any encoded node's source_refs
    ref_trace_ids = set()
    for n in nodes:
        for r in _node_source_refs(brain.conn, n['id']):
            ref_trace_ids.add(r)
    substrate_blob = ''
    if ref_trace_ids:
        placeholders = ','.join('?' * len(ref_trace_ids))
        rows = brain.logs_conn.execute(
            f"SELECT summary, metadata FROM trace_events WHERE id IN ({placeholders})",
            list(ref_trace_ids)).fetchall()
        # Full turn content lives in metadata.content (summary is truncated to 200c).
        # Defensive: metadata may be JSON string OR double-encoded JSON; handle both
        # via _decode_metadata-style fallback.
        import json as _json
        pieces = []
        for summary, metadata in rows:
            if summary:
                pieces.append(summary)
            if metadata:
                payload = metadata
                for _ in range(2):  # tolerate double-encoded JSON
                    if isinstance(payload, str):
                        try:
                            payload = _json.loads(payload)
                        except Exception:
                            payload = None
                            break
                if isinstance(payload, dict) and payload.get('content'):
                    pieces.append(payload['content'])
        substrate_blob = '\n'.join(pieces)
    via_substrate = {n for n in numerics if n in substrate_blob}

    combined = in_content | via_substrate
    dropped = sorted(set(numerics) - combined)
    only_substrate = sorted(via_substrate - in_content)

    return {
        'score': round(len(combined) / len(numerics), 3),
        'in_content_score': round(len(in_content) / len(numerics), 3),
        'via_substrate_score': round(len(via_substrate) / len(numerics), 3),
        'in_content': len(in_content),
        'via_substrate': len(via_substrate),
        'combined': len(combined),
        'total': len(numerics),
        'preserved_via_substrate_only': only_substrate[:8],
        'dropped_examples': dropped[:8],
        'n_refs_consulted': len(ref_trace_ids),
    }


def probe_source_refs_coverage(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """What % of nodes carry source_refs? Sparsity distribution. Hex-format
    failures. Coverage signal for v22 source_refs teaching."""
    nodes = _live_nodes(brain)
    if not nodes:
        return {'score': 0.0, 'nodes_encoded': 0,
                'reason': 'no nodes encoded'}

    HEX_RE = re.compile(r'^[0-9a-f]{8}$')

    nodes_with_refs = 0
    ref_counts = []
    hex_format_failures = 0
    sparsity_violations = 0  # >5 refs/node, v22 §7.5 threshold

    for n in nodes:
        refs = _node_source_refs(brain.conn, n['id'])
        if refs:
            nodes_with_refs += 1
            ref_counts.append(len(refs))
            if len(refs) > 5:
                sparsity_violations += 1
            for r in refs:
                if not HEX_RE.match(r):
                    hex_format_failures += 1

    coverage = nodes_with_refs / len(nodes)
    avg_refs = sum(ref_counts) / max(1, len(ref_counts))
    return {
        'score': round(coverage, 3),
        'nodes_encoded': len(nodes),
        'nodes_with_refs': nodes_with_refs,
        'coverage_pct': round(coverage * 100, 1),
        'avg_refs_per_node_with_refs': round(avg_refs, 2),
        'ref_count_distribution': {
            '1': ref_counts.count(1), '2': ref_counts.count(2),
            '3': ref_counts.count(3), '4': ref_counts.count(4),
            '5': ref_counts.count(5), '>5': sum(1 for c in ref_counts if c > 5),
        },
        'sparsity_violations_gt5': sparsity_violations,
        'hex_format_failures': hex_format_failures,
    }


def probe_atomization_shape(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """How many nodes per turn? Bundled vs atomized. Heuristic: total
    nodes encoded / total turns in haystack."""
    nodes = _live_nodes(brain)
    total_turns = sum(len(s) for s in item.get('haystack_sessions') or [])
    if total_turns == 0:
        return {'score': 0.0, 'reason': 'no turns'}

    # Type diversity = atomization signal
    types = [n['type'] for n in nodes]
    type_counts: Dict[str, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    nodes_per_turn = len(nodes) / total_turns
    # Sweet spot empirically ~0.3-0.8 (1 node per 1-3 turns)
    # Too low = under-encoding, too high = fragmentation
    if 0.3 <= nodes_per_turn <= 0.8:
        score = 1.0
    elif nodes_per_turn < 0.3:
        score = nodes_per_turn / 0.3
    else:  # > 0.8
        score = max(0.0, 1.0 - (nodes_per_turn - 0.8) / 1.2)

    return {
        'score': round(score, 3),
        'total_nodes': len(nodes),
        'total_turns': total_turns,
        'nodes_per_turn': round(nodes_per_turn, 3),
        'unique_types': len(type_counts),
        'type_distribution': dict(sorted(type_counts.items(),
                                          key=lambda x: -x[1])[:10]),
    }


def probe_edge_structure(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """Edge counts by relation. Aspect coverage. related_to overuse signal.
    co_anchored auto-edge presence."""
    nodes = _live_nodes(brain)
    node_ids = {n['id'] for n in nodes}

    # Pull all edges touching this run's nodes
    relation_counts: Dict[str, int] = {}
    co_anchored_pairs = set()
    typed_connect_pairs = set()
    related_to_count = 0

    seen_pairs = set()
    for n in nodes:
        for e in _node_edges(brain.conn, n['id']):
            src, tgt, rel = e['source_id'], e['target_id'], e['relation']
            if tgt not in node_ids and src not in node_ids:
                continue
            pair = tuple(sorted([src, tgt]))
            if (pair, rel) in seen_pairs:
                continue
            seen_pairs.add((pair, rel))
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
            if rel == 'co_anchored':
                co_anchored_pairs.add(pair)
            elif rel == 'related_to':
                related_to_count += 1
            elif e['encoding_source'] != 'dispatch:co_anchored':
                typed_connect_pairs.add(pair)

    # Aspects: load aspect registry to bucket relations
    try:
        aspect_buckets: Dict[str, int] = {}
        for rel, count in relation_counts.items():
            aspect = brain.aspects.by_edge_relation(rel)
            bucket = aspect.name if aspect else 'unmapped'
            aspect_buckets[bucket] = aspect_buckets.get(bucket, 0) + count
    except Exception:
        aspect_buckets = {}

    total_edges = sum(relation_counts.values())
    related_to_pct = (related_to_count / total_edges * 100) if total_edges else 0

    return {
        'total_edges': total_edges,
        'unique_relations': len(relation_counts),
        'co_anchored_pairs': len(co_anchored_pairs),
        'typed_connect_pairs': len(typed_connect_pairs),
        'related_to_count': related_to_count,
        'related_to_pct': round(related_to_pct, 1),
        'relation_distribution': dict(sorted(relation_counts.items(),
                                              key=lambda x: -x[1])[:15]),
        'aspect_buckets': aspect_buckets,
    }


def probe_voice_balance(brain, item: Dict[str, Any]) -> Dict[str, Any]:
    """their_raw_quote vs my_raw_quote presence. Symmetry on identity /
    correction / decision types — the dims where D7 anchor_voice_symmetry
    fires hardest."""
    nodes = _live_nodes(brain)
    if not nodes:
        return {'score': 0.0, 'reason': 'no nodes'}

    IDENTITY_BEARING_TYPES = {'principle', 'identity', 'vision', 'rule',
                              'correction', 'decision', 'lesson', 'insight',
                              'pattern', 'moment'}

    total = len(nodes)
    with_user = sum(1 for n in nodes if (n.get('their_raw_quote') or '').strip())
    with_anchor = sum(1 for n in nodes if (n.get('my_raw_quote') or '').strip())

    # Identity-bearing subset (where symmetry matters most)
    id_nodes = [n for n in nodes if n['type'] in IDENTITY_BEARING_TYPES]
    id_total = len(id_nodes)
    id_with_user = sum(1 for n in id_nodes if (n.get('their_raw_quote') or '').strip())
    id_with_anchor = sum(1 for n in id_nodes if (n.get('my_raw_quote') or '').strip())

    # Symmetry score = min/max on identity-bearing nodes — high when balanced
    if id_total == 0:
        symmetry = None
    else:
        a, b = id_with_user, id_with_anchor
        if max(a, b) == 0:
            symmetry = 0.0
        else:
            symmetry = round(min(a, b) / max(a, b), 3)

    return {
        'score': symmetry if symmetry is not None else 0.0,
        'total_nodes': total,
        'with_their_raw_quote': with_user,
        'with_my_raw_quote': with_anchor,
        'user_pct': round(with_user / total * 100, 1),
        'anchor_pct': round(with_anchor / total * 100, 1),
        'identity_bearing_total': id_total,
        'identity_bearing_with_user': id_with_user,
        'identity_bearing_with_anchor': id_with_anchor,
        'identity_bearing_symmetry': symmetry,
    }


# ─────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────

ALL_PROBES = {
    'brain_presence': probe_brain_presence,
    'specificity_preservation': probe_specificity_preservation,
    'source_refs_coverage': probe_source_refs_coverage,
    'atomization_shape': probe_atomization_shape,
    'edge_structure': probe_edge_structure,
    'voice_balance': probe_voice_balance,
}


def run_all_probes(brain, item: Dict[str, Any],
                   skip: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run every probe against (brain, item). Returns {probe_name: result}.

    `skip` lets the caller drop heavy probes during smoke-test runs.
    A failing probe doesn't fail the whole batch — its result carries
    `error` and `score=0.0`.
    """
    skip = set(skip or [])
    results = {}
    for name, fn in ALL_PROBES.items():
        if name in skip:
            results[name] = {'skipped': True}
            continue
        try:
            results[name] = fn(brain, item)
        except Exception as e:
            results[name] = {'score': 0.0, 'error': repr(e)}
    return results
