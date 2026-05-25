"""Encoding activity + encoding runs.

`query_encoding_activity` is the flat stream: new nodes, revisions, edges,
enrichments — ordered by timestamp.

`query_encoding_runs` is the grouped view: one card per S1E run, with the
encoder prompt (read from the tmp file the encoder writes) plus the nodes
and edges created inside the run's time window.
"""

import json
import os

from ..clock import iso_window_around, utc_cutoff
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
            "SELECT summary, created_at FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'delta'",
            (chain_id,),
        ).fetchone()
        summary = d_row[0] if d_row else '(encoding in progress or no actions)'
        delta_ts = d_row[1] if d_row else ''

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
            "nodes": [],
            "edges": [],
            "encoder_prompt": encoder_prompt,
        })
    return runs


def query_encoding_runs(limit: int = 10, session_id: str = '', hours: int = 24):
    """S1E runs — one chain per run, enriched with nodes/edges from brain.db.

    Two-pass: chain skeletons from logs_db (via @safe_query), then per-run
    enrichment from brain.db (manual second connection). Manual second pass
    because @safe_query is single-DB by design.
    """
    runs = _query_encoding_chains(limit, session_id, hours)
    if not runs:
        return runs
    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return runs
            for run in runs:
                ts = run.get('delta_ts') or run.get('start_ts', '')
                if not ts:
                    continue
                # ±2-minute window around the run timestamp. Rolls hours
                # and midnight correctly (the old string-clamp version did not).
                ts_lo, ts_hi = iso_window_around(ts, minutes=2)

                nodes = bconn.execute(
                    "SELECT id, type, title, substr(content,1,200), created_at "
                    "FROM nodes WHERE encoding_source = 'encoder:sonnet' "
                    "AND created_at BETWEEN ? AND ? ORDER BY created_at",
                    (ts_lo, ts_hi),
                ).fetchall()
                run['nodes'] = [
                    {"id": n[0], "type": n[1], "title": n[2], "content": n[3], "timestamp": n[4]}
                    for n in nodes
                ]

                revised = bconn.execute(
                    "SELECT id, type, title, substr(content,1,200), revised_at "
                    "FROM nodes WHERE encoding_source = 'encoder:sonnet' "
                    "AND revised_at BETWEEN ? AND ? ORDER BY revised_at",
                    (ts_lo, ts_hi),
                ).fetchall()
                for r in revised:
                    if not any(n['id'] == r[0] for n in run['nodes']):
                        run['nodes'].append({
                            "id": r[0], "type": r[1], "title": r[2],
                            "content": r[3], "timestamp": r[4], "kind": "revised",
                        })

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
                    "ORDER BY e.created_at",
                    (ts_lo, ts_hi),
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
