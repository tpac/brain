"""Encoding activity + encoding runs.

`query_encoding_activity` is the flat stream: new nodes, revisions, edges,
enrichments — ordered by timestamp.

`query_encoding_runs` is the grouped view: one card per S1E run, with the
encoder prompt (read from the tmp file the encoder writes) plus the nodes
and edges created inside the run's time window.
"""

import json
import os

from ..clock import utc_cutoff
from ..db import brain_db_path, logs_db_path, ro_connect
from ..log import warn
from ..query import safe_query


@safe_query('queries.encoding', brain_db_path)
def query_encoding_activity(conn, since_ts: str = "", limit: int = 30):
    """Flat encoding activity stream — new nodes, revisions, connections, enrichments."""
    events = []
    where = "WHERE created_at > ?" if since_ts else "WHERE 1=1"
    args_base = (since_ts,) if since_ts else ()

    # New nodes
    rows = conn.execute(
        f"SELECT id, type, title, content, confidence, encoding_source, locked, created_at "
        f"FROM nodes {where} ORDER BY created_at DESC LIMIT ?",
        args_base + (limit,),
    ).fetchall()
    for r in rows:
        events.append({
            "kind": "created", "id": r[0], "type": r[1], "title": r[2],
            "content": (r[3] or "")[:300], "confidence": r[4],
            "encoding_source": r[5], "locked": bool(r[6]), "timestamp": r[7],
        })

    # Revised nodes
    rows = conn.execute(
        "SELECT id, type, title, content, confidence, revised_at, encoding_source "
        "FROM nodes WHERE revised_at IS NOT NULL AND revised_at > ? "
        "ORDER BY revised_at DESC LIMIT ?",
        (since_ts or "1970-01-01", limit),
    ).fetchall()
    for r in rows:
        events.append({
            "kind": "revised", "id": r[0], "type": r[1], "title": r[2],
            "content": (r[3] or "")[:300], "confidence": r[4], "timestamp": r[5],
            "encoding_source": r[6],
        })

    # New connections — active-only timeline (v25 archived=0 filter).
    rows = conn.execute(
        f"SELECT e.source_id, e.target_id, er.relation, e.weight, e.created_at, "
        f"n1.title, n2.title, n1.type, n2.type "
        f"FROM edges e "
        f"JOIN edge_relations er ON er.edge_id = e.edge_id "
        f"LEFT JOIN nodes n1 ON n1.id = e.source_id "
        f"LEFT JOIN nodes n2 ON n2.id = e.target_id "
        f"{where.replace('created_at', 'e.created_at')} "
        f"AND er.archived = 0 "
        f"AND er.relation NOT IN ('co_accessed', 'emergent_bridge') "
        f"ORDER BY e.created_at DESC LIMIT ?",
        args_base + (limit,),
    ).fetchall()
    for r in rows:
        events.append({
            "kind": "connected",
            "source_title": r[5] or r[0][:12],
            "target_title": r[6] or r[1][:12],
            "relation": r[2], "weight": r[3], "timestamp": r[4],
            "source_type": r[7] or '', "target_type": r[8] or '',
            "source_id": r[0], "target_id": r[1],
        })

    # Enrichments
    rows = conn.execute(
        f"SELECT ne.node_id, ne.vector_type, ne.text, ne.created_at, n.title "
        f"FROM node_enrichments ne "
        f"LEFT JOIN nodes n ON n.id = ne.node_id "
        f"{where.replace('created_at', 'ne.created_at')} "
        f"ORDER BY ne.created_at DESC LIMIT ?",
        args_base + (limit,),
    ).fetchall()
    # For _situation events, text column is deprecated — pull from KV.
    sit_ids = [r[0] for r in rows if r[1] == '_situation']
    if sit_ids:
        kv_rows = conn.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key='situation' AND node_id IN (%s)"
            % ','.join('?' * len(sit_ids)),
            sit_ids,
        ).fetchall()
        kv_sit = dict(kv_rows)
    else:
        kv_sit = {}
    for r in rows:
        text = kv_sit.get(r[0], '') if r[1] == '_situation' else (r[2] or '')
        events.append({
            "kind": "enriched", "node_title": r[4] or r[0][:12],
            "vector_type": r[1], "text": text[:200], "timestamp": r[3],
        })

    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


@safe_query('queries.encoding', logs_db_path)
def _query_encoding_chains(conn, limit: int, session_id: str, hours: int):
    """Phase 1 of query_encoding_runs: pull S1E chain skeletons from logs_db.

    Split into its own decorated function so the logs-db connection is
    managed cleanly by @safe_query. Brain-db enrichment happens in a second
    pass (manual ro_connect) below, because @safe_query is single-DB.
    """
    conditions = [
        "scale = 's1'",
        "event_type = 'O'",
        "ref_type = 'encoding_prompt'",
        "created_at > ?",
    ]
    params = [utc_cutoff(hours=hours)]
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    where = ' AND '.join(conditions)
    chain_rows = conn.execute(
        "SELECT chain_id, ref_id, summary, metadata, session_id, created_at "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()

    runs = []
    for row in chain_rows:
        chain_id = row[0]
        prompt_file = row[1] or ''
        prompt_info = row[2] or ''
        session = row[4] or ''
        timestamp = row[5] or ''
        # Chain ID format: s1e-{session_short}-{counter}. Trailing token is
        # the counter. Keep this string-split — it's the only chain_id parse
        # left after the hex migration; chain_ids themselves stay string-
        # formatted even after trace_events.id went hex.
        counter = chain_id.split('-')[-1] if chain_id else ''

        k_row = conn.execute(
            "SELECT summary, ref_id FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'K'",
            (chain_id,),
        ).fetchone()
        catalog_info = k_row[0] if k_row else ''

        d_row = conn.execute(
            "SELECT summary, created_at, metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'delta'",
            (chain_id,),
        ).fetchone()
        summary = d_row[0] if d_row else '(encoding in progress or no actions)'
        delta_ts = d_row[1] if d_row else ''

        # The run records the exact node ids it created/revised in its delta
        # metadata (build_delta_metadata, populated for both remember_batch and
        # brain_batch). Read that authoritative list rather than reconstructing
        # it from encoding_source + a time window — the reconstruction drifts the
        # moment the tag or the run duration changes; the trace never does.
        created_ids, revised_ids = [], []
        if d_row and d_row[2]:
            try:
                dmeta = json.loads(d_row[2])
                created_ids = [i for i in (dmeta.get('created') or []) if i]
                revised_ids = [i for i in (dmeta.get('revised') or []) if i]
            except (ValueError, TypeError):
                pass

        encoder_prompt = None
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file) as f:
                    encoder_prompt = json.load(f).get("user_content")
            except Exception:
                # Inner row-level failure — silent on purpose.
                pass

        runs.append({
            "chain_id": chain_id,
            "counter": counter,
            "start_ts": timestamp,
            "delta_ts": delta_ts,
            "session_id": session,
            "summary": summary[:500],
            "prompt_info": prompt_info,
            "catalog_info": catalog_info,
            "created_ids": created_ids,
            "revised_ids": revised_ids,
            "nodes": [],
            "edges": [],
            "encoder_prompt": encoder_prompt,
        })
    return runs


def query_encoding_runs(limit: int = 10, session_id: str = '', hours: int = 24):
    """S1E runs — one chain per run, enriched with the exact nodes the run
    recorded, plus the edges touching them.

    Two-pass: chain skeletons from logs_db (via @safe_query), then per-run
    enrichment from brain.db (manual second connection, @safe_query is single-DB).

    The nodes shown are the precise ids the run recorded in its delta trace
    (`created`/`revised`), fetched by id — NOT reconstructed from
    encoding_source + a time window. The trace is the run's own statement of what
    it did, so the view can't drift when tagging changes (e.g. a run writing via
    brain_batch instead of remember_batch) or when a run runs longer than the old
    ±2min guess. A run that recorded no node writes shows empty — truthfully.
    """
    runs = _query_encoding_chains(limit, session_id, hours)
    if not runs:
        return runs
    # Strip the internal id scaffolding up front so it never leaks into the
    # response on the bconn-None / exception paths (the per-run loop below
    # consumes from this map instead of from the run dict).
    ids_by_chain = {
        r['chain_id']: ((r.pop('created_ids', []) or []), (r.pop('revised_ids', []) or []))
        for r in runs
    }
    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return runs
            for run in runs:
                created_ids, revised_ids = ids_by_chain.get(run['chain_id'], ([], []))
                node_ids = created_ids + revised_ids
                if not node_ids:
                    continue  # nothing recorded for this run — leave it empty

                placeholders = ','.join('?' * len(node_ids))
                rows = bconn.execute(
                    "SELECT id, type, title, substr(content,1,200), created_at, encoding_source "
                    "FROM nodes WHERE id IN (%s)" % placeholders,
                    node_ids,
                ).fetchall()
                by_id = {r[0]: r for r in rows}

                run['nodes'] = []
                seen = set()
                for nid in created_ids:
                    r = by_id.get(nid)
                    if r and nid not in seen:
                        seen.add(nid)
                        run['nodes'].append({
                            "id": r[0], "type": r[1], "title": r[2],
                            "content": r[3], "timestamp": r[4], "encoding_source": r[5],
                        })
                for nid in revised_ids:
                    r = by_id.get(nid)
                    if r and nid not in seen:
                        seen.add(nid)
                        run['nodes'].append({
                            "id": r[0], "type": r[1], "title": r[2],
                            "content": r[3], "timestamp": r[4],
                            "encoding_source": r[5], "kind": "revised",
                        })

                # Edges the run formed: those touching its own node set, bounded
                # by the run's real span (start trace → delta trace). Derived from
                # the authoritative node set, so no fixed-window or tag guessing.
                start_ts = run.get('start_ts', '')
                end_ts = run.get('delta_ts') or start_ts
                if start_ts and end_ts:
                    nid_ph = ','.join('?' * len(node_ids))
                    edges = bconn.execute(
                        "SELECT e.source_id, e.target_id, er.relation, e.weight, e.created_at, "
                        "n1.title, n2.title "
                        "FROM edges e "
                        "JOIN edge_relations er ON er.edge_id = e.edge_id "
                        "LEFT JOIN nodes n1 ON n1.id = e.source_id "
                        "LEFT JOIN nodes n2 ON n2.id = e.target_id "
                        "WHERE e.created_at BETWEEN ? AND ? "
                        "AND er.archived = 0 "
                        "AND er.relation NOT IN ('co_accessed', 'emergent_bridge') "
                        "AND (e.source_id IN (%s) OR e.target_id IN (%s)) "
                        "ORDER BY e.created_at" % (nid_ph, nid_ph),
                        [start_ts, end_ts] + node_ids + node_ids,
                    ).fetchall()
                    run['edges'] = [
                        {"relation": e[2], "weight": e[3],
                         "source_title": e[5] or e[0][:12],
                         "target_title": e[6] or e[1][:12],
                         "timestamp": e[4]}
                        for e in edges
                    ]
    except Exception as e:
        warn('queries.encoding', 'enriching runs with nodes/edges failed', exc=e)
    return runs
