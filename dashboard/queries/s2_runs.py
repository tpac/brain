"""S2 unit runs — consolidation, community detection, and healer.

Chain IDs follow the format `s2-{date}-{unit_name}`; we filter by substring
because there's no scale+ref_type combination that uniquely identifies them
yet. Survived schema v29 (hex `trace_events.id`) because chain_id is still
string-formatted.

This file used to have three near-identical functions all running the same
"pull O/K/delta from logs_db, enrich per-delta from brain.db" skeleton. Now
that pattern lives in `_query_s2_unit_runs` and each unit declares its own
enrichment callback. Healer stays separate — it's logs-only with no
brain-side enrichment, a different shape.
"""

import json
from typing import Callable, List

from ..clock import iso_window_around, utc_cutoff
from ..db import brain_db_path, logs_db_path, ro_connect
from ..log import warn


def _fetch_ok_deltas(conn, unit_keyword: str, hours: int, delta_columns: str,
                     ok_extra_columns: str = ''):
    """Pull (delta_rows, ok_by_chain) for a given S2 unit from logs_db.

    `delta_columns` is the column list selected for delta rows — callers
    differ on whether they need `ref_type`. `ok_extra_columns` lets a
    caller pull extra fields off the O/K rows (e.g. metadata).

    Both queries share the same `chain_id LIKE '%X%'` substring filter
    and the same time-window cutoff. Brittle but works while chain_ids
    are string-keyed.
    """
    since = utc_cutoff(hours=hours)
    delta_rows = conn.execute(
        "SELECT %s FROM trace_events WHERE chain_id LIKE ? "
        "AND event_type = 'delta' AND created_at > ? ORDER BY created_at DESC"
        % delta_columns,
        ('%' + unit_keyword + '%', since),
    ).fetchall()
    ok_select = "chain_id, event_type, summary"
    if ok_extra_columns:
        ok_select += ', ' + ok_extra_columns
    ok_rows = conn.execute(
        "SELECT %s FROM trace_events WHERE chain_id LIKE ? "
        "AND event_type IN ('O', 'K') AND created_at > ? ORDER BY created_at DESC"
        % ok_select,
        ('%' + unit_keyword + '%', since),
    ).fetchall()
    return delta_rows, ok_rows


def _query_s2_unit_runs(unit_keyword: str, hours: int, delta_columns: str,
                        enricher: Callable, ok_extra_columns: str = '',
                        component: str = 'queries.s2_runs') -> List[dict]:
    """Shared skeleton for consolidation + community runs.

    Steps:
      1. Open logs_db, pull O/K and delta rows for `unit_keyword`.
      2. Build `ok_by_chain[chain_id][event_type]` payload dict.
      3. Open brain_db. For each delta, call `enricher(bconn, delta_row,
         ok_payload) -> dict` and append to the result list.

    Either DB unreachable → return []. The enricher's job is to produce the
    per-run dict (chain_id, timestamp, summary, etc + unit-specific fields).
    Loud-by-default: any failure logs via warn() before returning [].
    """
    runs: List[dict] = []

    try:
        with ro_connect(logs_db_path()) as conn:
            if conn is None:
                return []
            delta_rows, ok_rows = _fetch_ok_deltas(
                conn, unit_keyword, hours, delta_columns, ok_extra_columns)
    except Exception as e:
        warn(component, '%s logs_db pull failed' % unit_keyword, exc=e)
        return []

    # Build chain → event_type → payload index.
    ok_by_chain: dict = {}
    for r in ok_rows:
        chain, et, summary = r[0], r[1], r[2] or ''
        meta = r[3] if len(r) > 3 else None
        ok_by_chain.setdefault(chain, {})[et] = {'summary': summary, 'metadata': meta}

    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return []
            for delta_row in delta_rows:
                ok_payload = ok_by_chain.get(delta_row[0], {})
                try:
                    runs.append(enricher(bconn, delta_row, ok_payload))
                except Exception as e:
                    # One delta failing shouldn't drop the whole feed.
                    warn(component, '%s enricher failed for chain %s' % (
                        unit_keyword, delta_row[0] if delta_row else '?'), exc=e)
    except Exception as e:
        warn(component, '%s brain_db enrichment failed' % unit_keyword, exc=e)
        return []

    return runs


# ── Consolidation enricher ──────────────────────────────────────────────────

def _enrich_consolidation(bconn, delta_row, ok_payload) -> dict:
    chain_id, summary, meta_raw, created_at = delta_row

    journal = ''
    try:
        meta = json.loads(meta_raw) if meta_raw else {}
        journal = meta.get('final_text', '')
    except Exception:
        pass

    # ±60-minute window around the delta. iso_window_around handles
    # midnight/hour rollovers (the old string-clamp did not).
    #
    # This view stays window-based ON PURPOSE — unlike the S1 encoding-runs
    # view, consolidation runs leave their delta `created` bucket EMPTY (verified:
    # 564 deltas, 0 with meta.created, yet 102 s2:consolidation nodes exist). The
    # delta DOES carry action_details — `created` is empty because consolidation
    # synthesizes via consolidate/evolve ops, not the remember->'created' path the
    # runner buckets. So reading trace ids here would show 0 synth for every run.
    # The window is sound for consolidation because S2 tags reliably (base.py
    # stamps encoding_source unconditionally) and runs are idle-gated/spaced, so
    # cross-run overlap is unlikely. The authoritative fix lives in the
    # consolidation encoder (bucket synthesis ids into `created`), not here.
    ts_lo, ts_hi = iso_window_around(created_at, minutes=60)

    synth_nodes = bconn.execute(
        "SELECT id, type, title, substr(content,1,500), confidence "
        "FROM nodes WHERE encoding_source = 's2:consolidation' "
        "AND created_at BETWEEN ? AND ? AND archived = 0 ORDER BY created_at",
        (ts_lo, ts_hi),
    ).fetchall()

    # Forensic view — show originals even though their edges are archived.
    archived_nodes = []
    for sn in synth_nodes:
        originals = bconn.execute(
            "SELECT n.id, n.type, n.title, substr(n.content,1,150) "
            "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
            "JOIN nodes n ON n.id = e.target_id "
            "WHERE e.source_id = ? AND er.relation = 'consolidated_into' "
            "AND n.archived = 1",
            (sn[0],),
        ).fetchall()
        for o in originals:
            archived_nodes.append({"id": o[0], "type": o[1], "title": o[2], "content": o[3]})

    evolved_archived = bconn.execute(
        "SELECT n.id, n.type, n.title, substr(n.content,1,300) "
        "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
        "JOIN nodes n ON n.id = e.target_id "
        "WHERE er.relation = 'supersedes' AND e.created_at BETWEEN ? AND ? "
        "AND n.archived = 1",
        (ts_lo, ts_hi),
    ).fetchall()
    for o in evolved_archived:
        if not any(a['id'] == o[0] for a in archived_nodes):
            archived_nodes.append({"id": o[0], "type": o[1], "title": o[2], "content": o[3]})

    kept_edges = bconn.execute(
        "SELECT n1.title, n2.title, er.relation, er.description "
        "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
        "JOIN nodes n1 ON n1.id = e.source_id "
        "JOIN nodes n2 ON n2.id = e.target_id "
        "WHERE er.relation = 'similar_to' AND e.created_at BETWEEN ? AND ? "
        "AND er.archived = 0 ORDER BY e.created_at",
        (ts_lo, ts_hi),
    ).fetchall()

    evolved_edges = bconn.execute(
        "SELECT n1.title, n2.title, er.description "
        "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
        "JOIN nodes n1 ON n1.id = e.source_id "
        "JOIN nodes n2 ON n2.id = e.target_id "
        "WHERE er.relation = 'supersedes' AND e.created_at BETWEEN ? AND ? "
        "AND n2.archived = 1 ORDER BY e.created_at",
        (ts_lo, ts_hi),
    ).fetchall()

    return {
        "chain_id": chain_id,
        "timestamp": created_at,
        "summary": summary or '',
        "o_summary": ok_payload.get('O', {}).get('summary', ''),
        "k_summary": ok_payload.get('K', {}).get('summary', ''),
        "journal": journal[:1000],
        "synthesized": [
            {"id": n[0], "type": n[1], "title": n[2], "content": n[3], "confidence": n[4]}
            for n in synth_nodes
        ],
        "archived": archived_nodes,
        "kept": [
            {"source": e[0], "target": e[1], "description": e[3] or ''}
            for e in kept_edges
        ],
        "evolved": [
            {"survivor": e[0], "archived": e[1], "description": e[2] or ''}
            for e in evolved_edges
        ],
    }


def query_consolidation_runs(hours: int = 24):
    """S2 consolidation runs — synthesized / archived / kept / evolved per run."""
    return _query_s2_unit_runs(
        'consolidation', hours,
        delta_columns='chain_id, summary, metadata, created_at',
        enricher=_enrich_consolidation,
        ok_extra_columns='metadata',
    )


# ── Community enricher ──────────────────────────────────────────────────────
# Community is a bit different: the community-node list is fetched ONCE
# (not per-delta), so the enricher reads it from a closure rather than
# requerying. We bind that via a factory.

def _make_community_enricher(community_list):
    def _enrich(bconn, delta_row, ok_payload):
        chain_id, summary, _meta_raw, created_at, ref_type = delta_row
        # `created_count` mirrors what query_community_runs reported before:
        # count of `community_created` deltas in the same chain. We don't
        # have the full delta_rows here, so 1 if THIS row was the trigger,
        # otherwise 0. The UI uses .length on `communities` for the real
        # count — this field is just a hint.
        created_count = 1 if ref_type == 'community_created' else 0
        return {
            "chain_id": chain_id,
            "timestamp": created_at,
            "summary": summary or '',
            "o_summary": ok_payload.get('O', {}).get('summary', ''),
            "k_summary": ok_payload.get('K', {}).get('summary', ''),
            "created_count": created_count,
            "communities": community_list[:15],
        }
    return _enrich


def _fetch_community_nodes(bconn) -> list:
    """Pull the 30 most-recent community nodes with their KV metadata."""
    communities = bconn.execute(
        "SELECT id, title, substr(content,1,400), confidence, created_at "
        "FROM nodes WHERE type = 'community' AND archived = 0 "
        "AND encoding_source = 's2:community_detection' "
        "ORDER BY created_at DESC LIMIT 30"
    ).fetchall()
    out = []
    for c in communities:
        cid, title, content, conf, created = c
        meta = dict(bconn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (cid,),
        ).fetchall())
        member_count = bconn.execute(
            "SELECT COUNT(*) FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE (e.source_id = ? OR e.target_id = ?) "
            "AND er.relation = 'community_member' "
            "AND er.archived = 0",
            (cid, cid),
        ).fetchone()[0]
        out.append({
            "id": cid, "title": title, "content": content,
            "confidence": conf, "created_at": created,
            "members": member_count,
            "maturity": meta.get('community_maturity', '?'),
            "narrative": (meta.get('community_narrative') or '')[:300],
            "open_questions": (meta.get('community_open_questions') or '')[:200],
            "latest": (meta.get('community_latest_development') or '')[:150],
        })
    return out


def query_community_runs(hours: int = 24):
    """S2 community detection runs — one entry per delta, sharing a snapshot
    of the 30 most-recent community nodes."""
    # We need to fetch the community node list BEFORE _query_s2_unit_runs
    # opens its brain.db connection, so do it in our own ro_connect first.
    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return []
            community_list = _fetch_community_nodes(bconn)
    except Exception as e:
        warn('queries.s2_runs', 'community node pre-fetch failed', exc=e)
        return []

    # Dedupe across deltas: render each chain once. This mirrors the prior
    # behavior (seen_chains set) and prevents the same community list from
    # repeating once per delta event in the same chain.
    seen = set()
    runs = _query_s2_unit_runs(
        'community_detection', hours,
        delta_columns='chain_id, summary, metadata, created_at, ref_type',
        enricher=_make_community_enricher(community_list),
    )
    deduped = []
    for r in runs:
        if r['chain_id'] in seen:
            continue
        seen.add(r['chain_id'])
        deduped.append(r)
    return deduped


# ── Healer (different shape: logs-only, single forward pass) ────────────────

def query_healer_runs(hours: int = 24, limit: int = 30):
    """S2 Healer passes — one card per `healer_generated` delta event.

    Healer can run multiple times per day; the chain_id is `s2-{date}-healer`,
    so chains aren't unique per pass. Pair each delta with its nearest
    preceding O/K events in the same chain via a single forward pass —
    different enough from consolidation/community that sharing the helper
    would distort it.
    """
    rows = []
    try:
        with ro_connect(logs_db_path()) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT chain_id, event_type, ref_type, summary, metadata, created_at "
                "FROM trace_events WHERE chain_id LIKE '%healer%' "
                "AND created_at > ? ORDER BY created_at ASC",
                (utc_cutoff(hours=hours),),
            ).fetchall()
    except Exception as e:
        warn('queries.s2_runs', 'healer pull failed', exc=e)
        return []

    # Walk forward; remember most-recent O and K per chain, snap each delta
    # to whatever O/K preceded it. Linear time, no resort needed.
    last_o: dict = {}
    last_k: dict = {}
    runs = []
    for chain_id, event_type, ref_type, summary, _meta, created_at in rows:
        if event_type == 'O':
            last_o[chain_id] = (ref_type or '', summary or '')
        elif event_type == 'K':
            last_k[chain_id] = (ref_type or '', summary or '')
        elif event_type == 'delta':
            o = last_o.get(chain_id, ('', ''))
            k = last_k.get(chain_id, ('', ''))
            runs.append({
                'chain_id': chain_id,
                'timestamp': created_at,
                'summary': summary or '',
                'ref_type': ref_type or '',
                'o_summary': o[1],
                'o_ref_type': o[0],
                'k_summary': k[1],
                'k_ref_type': k[0],
            })

    # Newest first — matches the consolidation/community convention.
    runs.sort(key=lambda r: r['timestamp'], reverse=True)
    return runs[:limit]
