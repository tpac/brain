"""S1 recall events — the live decoding feed.

Reads from `trace_events` (single source of truth since 2026-04-05): joins
O / K / delta rows by chain_id to produce one summary per recall.

Cursor is timestamp-based (`since_ts`), not integer rowid: `trace_events.id`
is now an 8-char hex string under schema v29, so integer ordering no longer
applies. `created_at` is monotonic per writer.
"""

import json
import os

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query


def read_judge_file(recall_ref: str):
    """Read judge data from the temp file the hook writes."""
    # BRAIN_TMP_DIR env protocol — must match the WRITER
    # (servers.daemon_config.brain_tmp_dir(); default /tmp). The dashboard is a
    # separate, deliberately servers-decoupled process, so it reads the same env
    # var directly rather than importing servers.
    path = os.path.join(os.environ.get('BRAIN_TMP_DIR', '/tmp'),
                        "brain-judge-result-%s.json" % recall_ref)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            data = json.load(f)
        return (
            data.get("surface_prompt") or data.get("judge_prompt"),
            data.get("surface_output") or data.get("judge_output"),
        )
    except Exception:
        # Inner row-level failure — silent on purpose. Old-format files exist
        # and we'd spam stderr if every one logged. P0.4 unifies this via
        # io.safe_json_file().
        return None, None


def query_recall_prompt(recall_ref: str = ""):
    """Lazy-load one recall's full surface prompt on card expand.

    Split out of query_recall_log so the polled decoding feed stays small: the
    ~35KB prompt is fetched only when the operator expands a card's "Show
    Prompt". Reads the same /tmp judge-result file read_judge_file uses.
    Returns {"judge_prompt": str} or {"error": str} for the UI fallback."""
    # recall_ref lands in a filename — reject anything that could traverse out
    # of the tmp dir (localhost-only dashboard, but cheap defense).
    if not recall_ref or '/' in recall_ref or '..' in recall_ref:
        return {"error": "bad recall_ref"}
    prompt, _out = read_judge_file(recall_ref)
    if not prompt:
        return {"error": "no prompt file for %s" % recall_ref}
    return {"judge_prompt": prompt}


@safe_query('queries.recalls', logs_db_path)
def query_recall_log(conn, since_ts: str = '', limit: int = 50, session_id: str = ''):
    """Read recall events from S1 traces, joined into one row per chain."""
    conditions = ["scale = 's1'", "event_type = 'O'", "ref_type = 'recall'"]
    params = []
    if since_ts:
        conditions.append("created_at > ?")
        params.append(since_ts)
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    where = ' AND '.join(conditions)
    rows = conn.execute(
        "SELECT id, chain_id, ref_id, summary, metadata, session_id, created_at "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()

    results = []
    for r in rows:
        trace_id = r[0]
        chain_id = r[1]
        recall_ref = r[2] or ''
        summary = r[3] or ''
        session_id_row = r[5] or ''
        timestamp = r[6] or ''

        candidates = []
        query = ''
        candidate_count = 0
        try:
            if summary and 'candidates for:' in summary:
                candidate_count = int(summary.split(' candidates')[0])
                query = summary.split('for: ', 1)[1] if 'for: ' in summary else ''
        except (ValueError, IndexError):
            pass
        human_identity = ''
        agent_identity = ''
        try:
            meta = json.loads(r[4]) if r[4] else {}
            if not query:
                query = meta.get('query', '')
            # Identity stamping (75075eb / 65bf483) records who was
            # speaking when the trace was written. Surface for the UI.
            human_identity = meta.get('human_identity', '') or ''
            agent_identity = meta.get('agent_identity', '') or ''
            for cand_str in meta.get('candidates', []):
                parts = cand_str.split('|')
                if len(parts) >= 4:
                    candidates.append({
                        'id': parts[0],
                        'title': '|'.join(parts[1:-2]),
                        'score': parts[-2],
                        'type': parts[-1],
                    })
        except Exception:
            pass

        # K event (judge-selected + activation expansion) in the same chain.
        # ref_id is the JSON-encoded list of short ids the Haiku judge picked;
        # metadata.activations is the spread-activation expansion (post-
        # selection neighbors that lit up enough to enter additionalContext).
        # Both are short (8-char) ids — graph node ids share that format, so
        # they reconcile directly with no remapping.
        selected_ids = []
        activation_ids = []
        k_row = conn.execute(
            "SELECT ref_id, summary, metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'K'",
            (chain_id,),
        ).fetchone()
        if k_row:
            try:
                selected_ids = json.loads(k_row[0]) if k_row[0] else []
            except Exception:
                pass
            try:
                k_meta = json.loads(k_row[2]) if k_row[2] else {}
                activation_ids = [
                    entry.get('id') for entry in (k_meta.get('activations') or [])
                    if entry.get('id')
                ]
            except Exception:
                pass

        # Δ event (additionalContext) in same chain
        judge_output = None
        d_row = conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'delta'",
            (chain_id,),
        ).fetchone()
        if d_row:
            try:
                d_meta = json.loads(d_row[0]) if d_row[0] else {}
                judge_output = d_meta.get('content', '')
            except Exception:
                pass

        j_prompt, j_output_file = read_judge_file(recall_ref)
        titles = {c['id']: c['title'] for c in candidates}

        results.append({
            "id": trace_id,
            "session_id": session_id_row,
            "query": query,
            "returned_ids": [c['id'] for c in candidates],
            "returned_count": candidate_count or len(candidates),
            "titles": titles,
            "snippets": {},
            "timestamp": timestamp,
            "source": "hook",
            "used_ids": selected_ids,
            "used_count": len(selected_ids),
            "activation_ids": activation_ids,
            # judge_prompt (the full surface prompt, ~35KB each) is NOT shipped
            # in the list — it was 75% of a 2.3MB polled payload, hidden behind
            # "Show Prompt" anyway. The frontend lazy-loads it per-card on expand
            # via query_recall_prompt(recall_ref). has_prompt tells the UI whether
            # to render the button without shipping the bytes.
            "has_prompt": bool(j_prompt),
            "recall_ref": recall_ref,
            # judge_output (the additionalContext) STAYS inline — it's the card's
            # displayed content, and it's an order of magnitude smaller (~8KB).
            # Prefer the /tmp surface-result file over trace metadata — the trace
            # truncates at 4000 chars, but the file holds the full context.
            "judge_output": j_output_file or judge_output,
            "human_identity": human_identity,
            "agent_identity": agent_identity,
        })
    return results
