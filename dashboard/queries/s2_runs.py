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
from ..db import brain_db_path, fetch_by_id, logs_db_path, ro_connect
from ..log import warn


# Mirror of servers/trace_contract.RESIDUE_REF_TYPES + EMITTER_REF_TYPES. The
# dashboard disconnection contract forbids importing servers.* (see
# queries/aspects.py), so we replicate. A consistency test pins these against the
# server-side constants — mirror-and-pin, never import.
#
# Both families are event_type='delta' on the unit's chain without being the
# unit's per-RUN integration delta, so an unfiltered pull renders them as phantom
# run cards: residue is encoder *notes* (journal_note); emitter rows are per-WRITE
# mutations (one per node/edge touched).
_RESIDUE_REF_TYPES = ('journal_note',)
_EMITTER_REF_TYPES = (
    'node_created', 'node_archived', 'node_deleted',
    'node_revised', 'edge_relation_revised',
)
# Everything the run-card queries must not mistake for a run.
_NON_RUN_REF_TYPES = _RESIDUE_REF_TYPES + _EMITTER_REF_TYPES


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
    excl = ','.join(['?'] * len(_NON_RUN_REF_TYPES))
    delta_rows = conn.execute(
        "SELECT %s FROM trace_events WHERE chain_id LIKE ? "
        "AND event_type = 'delta' AND (ref_type IS NULL OR ref_type NOT IN (%s)) "
        "AND created_at > ? ORDER BY created_at DESC"
        % (delta_columns, excl),
        ('%' + unit_keyword + '%', *_NON_RUN_REF_TYPES, since),
    ).fetchall()
    ok_select = "chain_id, event_type, summary, created_at"
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
      2. Build a per-chain, time-ordered O/K index + a `nearest_ok(chain,
         delta_ts)` closure that snaps a delta to the O/K immediately
         preceding it.
      3. Open brain_db. For each delta, call `enricher(bconn, delta_row,
         nearest_ok) -> dict` and append to the result list.

    Either DB unreachable → return []. The enricher's job is to produce the
    per-run dict (chain_id, timestamp, summary, etc + unit-specific fields);
    it calls `nearest_ok(chain_id, delta_created_at)` to resolve its run's
    O/K summaries.
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

    # Historical chain_ids are shared by every run on the same day
    # (s2-{date}-{unit}); new chains are per-run unique (s2-{ts}-{unit},
    # seconds-stamped), but this code still serves the date-based historical
    # rows where one chain holds many O/K/delta triples. Keying
    # O/K by chain_id alone collapses all runs onto a single (oldest) O/K
    # pair — which made every consolidation/community card show identical
    # cluster counts. Instead build a per-chain, time-ordered O/K list and
    # snap each delta to the O and K that immediately precede it.
    ok_events: dict = {}
    for r in ok_rows:
        chain, et, summary, ts = r[0], r[1], r[2] or '', r[3]
        meta = r[4] if len(r) > 4 else None
        ok_events.setdefault(chain, []).append(
            {'et': et, 'ts': ts, 'summary': summary, 'metadata': meta})
    for evs in ok_events.values():
        evs.sort(key=lambda e: e['ts'])  # ascending; ISO-T strings compare lexically

    def nearest_ok(chain_id: str, delta_ts) -> dict:
        """Most-recent O and K in `chain_id` at or before `delta_ts`.

        Returns {event_type: {'summary', 'metadata'}} — the same shape the
        enrichers already consume via .get('O'/'K', {}).get('summary')."""
        payload: dict = {}
        for e in ok_events.get(chain_id, []):
            if e['ts'] <= delta_ts:
                payload[e['et']] = {'summary': e['summary'], 'metadata': e['metadata']}
            else:
                break  # list is ascending — nothing after this can precede the delta
        return payload

    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return []
            for delta_row in delta_rows:
                try:
                    runs.append(enricher(bconn, delta_row, nearest_ok))
                except Exception as e:
                    # One delta failing shouldn't drop the whole feed.
                    warn(component, '%s enricher failed for chain %s' % (
                        unit_keyword, delta_row[0] if delta_row else '?'), exc=e)
    except Exception as e:
        warn(component, '%s brain_db enrichment failed' % unit_keyword, exc=e)
        return []

    return runs


# ── Consolidation enricher ──────────────────────────────────────────────────

def _short(nid) -> str:
    """8-char id label for a link endpoint we couldn't resolve to a title.
    str() guards against a malformed delta carrying a non-string id."""
    return str(nid)[:8]


# Hebbian/system relations are never encoder-emitted in a consolidation batch,
# but exclude them defensively so a stray one can't pollute the links list.
_NON_DECISION_RELATIONS = {'co_accessed', 'emergent_bridge'}


def _deconstruct_consolidation_ops(meta: dict):
    """Read what a consolidation run produced from its OWN delta record.

    Modern consolidation never CREATES nodes — by design it folds clusters into
    existing survivors (consolidation_enrichment_prompt.py: "Consolidation does
    not create new nodes. It strengthens existing ones."). A run's product is
    therefore enriched survivors, folded-in (archived) originals, and the
    suppression/teaching links it draws between nodes it KEEPS.

    Two id sources, by reliability:
      • survivors (enriched) + archived (folded-in): prefer the ok-gated
        top-level `revised`/`archived` buckets that build_delta_metadata
        populates (the dispatch `absorb` op returns affected survivor→`revised`,
        absorbed→`archived`, and SKIPS failed ops). Fall back to deconstructing
        the recorded op INPUT for pre-fix historical deltas, whose buckets are
        empty. The op INPUT is what was *requested*, so it isn't ok-gated — the
        buckets are strictly better when present, which is why they win.
      • links: every `connect` op — KEEP/SKIP (`similar_to`), SUPERSESSION
        (`supersedes`), CONTRADICTION (`corrects`), partition (`depends_on`),
        … . Connect ops aren't bucketed as node *pairs*, so links can only come
        from the recorded op input; capturing every relation (not just
        `similar_to`) is what keeps supersession/contradiction decisions
        visible and keeps a connect-only run off the legacy time window.

    Returns (synth_ids, archived_ids, links); links = list of
    {source_id, target_id, relation, description}.
    """
    op_survivors, op_archived, links = [], [], []
    for ad in (meta.get('action_details') or []):
        if not isinstance(ad, dict):
            continue
        for op in ((ad.get('input') or {}).get('operations') or []):
            if not isinstance(op, dict):
                continue
            name = op.get('op')
            if name == 'absorb':
                if op.get('survivor_id'):
                    op_survivors.append(op['survivor_id'])
                if op.get('absorbed_id'):
                    op_archived.append(op['absorbed_id'])
            elif name == 'revise':
                if op.get('node_id'):
                    op_survivors.append(op['node_id'])
            elif name == 'archive':
                if op.get('node_id'):
                    op_archived.append(op['node_id'])
            elif name == 'connect':
                rel = op.get('relation')
                src, tgt = op.get('source_id'), op.get('target_id')
                if rel and rel not in _NON_DECISION_RELATIONS and src and tgt:
                    links.append({'source_id': src, 'target_id': tgt,
                                  'relation': rel,
                                  'description': op.get('description', '') or ''})
    # ok-gated buckets win when present (post-fix deltas); op-input is the
    # retroactive fallback for pre-fix history.
    synth_ids = [i for i in (meta.get('revised') or []) if i] or op_survivors
    archived_ids = [i for i in (meta.get('archived') or []) if i] or op_archived
    return synth_ids, archived_ids, links


def _enrich_consolidation(bconn, delta_row, nearest_ok) -> dict:
    chain_id, summary, meta_raw, created_at = delta_row
    ok_payload = nearest_ok(chain_id, created_at)

    meta, journal = {}, ''
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            journal = meta.get('final_text', '')
        except (ValueError, TypeError) as e:
            # Loud-by-default: a corrupt delta payload silently fell back to the
            # ±60min window before. Surface it (stderr + Logs tab) and still
            # degrade gracefully to the window rather than dropping the run.
            warn('queries.s2_runs',
                 'consolidation delta %s has unparseable metadata; '
                 'falling back to window' % (chain_id or '?'), exc=e)

    # Trace-authoritative: read the exact node ids the run recorded touching
    # (from its delta's recorded ops/buckets) and fetch them by id — mirroring
    # query_encoding_runs in encoding.py, NOT reconstructing from encoding_source
    # + a time window. The window drifted the moment consolidation switched from
    # creating synth nodes to folding clusters into survivors via `absorb`; the
    # trace never does.
    synth_ids, archived_ids, link_specs = _deconstruct_consolidation_ops(meta)

    synthesized, archived_nodes, links = [], [], []
    node_ids = list({*synth_ids, *archived_ids,
                     *(s for k in link_specs
                       for s in (k['source_id'], k['target_id']))})

    if node_ids:
        by_id = fetch_by_id(
            bconn, 'nodes',
            'id, type, title, substr(content,1,500), confidence, archived',
            node_ids)

        # Liveness is the DB `archived` flag, not op ordering. A node that was a
        # survivor in one op and absorbed in a later op (chain merge), or a
        # survivor archived by a later run, is shown as archived — never as a
        # live survivor. The shared `seen` set dedups one node to one section.
        seen = set()
        for nid in synth_ids:
            if nid in seen:
                continue
            r = by_id.get(nid)
            if not r:
                continue
            seen.add(nid)
            node = {"id": r[0], "type": r[1], "title": r[2], "content": r[3]}
            if r[5]:  # archived flag → it didn't survive
                archived_nodes.append(node)
            else:
                node["confidence"] = r[4]
                synthesized.append(node)
        for nid in archived_ids:
            if nid in seen:
                continue
            r = by_id.get(nid)
            if not r:
                continue
            seen.add(nid)
            archived_nodes.append({"id": r[0], "type": r[1], "title": r[2],
                                   "content": r[3]})
        for k in link_specs:
            rs, rt = by_id.get(k['source_id']), by_id.get(k['target_id'])
            links.append({"source": rs[2] if rs else _short(k['source_id']),
                          "target": rt[2] if rt else _short(k['target_id']),
                          "relation": k['relation'],
                          "description": k['description']})
    else:
        # Legacy-only fallback. Pre-absorb deltas created synth nodes via
        # `remember`, whose generated ids aren't in the op input, so there's
        # nothing to deconstruct — reconstruct from encoding_source + a ±60min
        # window. This is the ONLY remaining window-based path; it fires solely
        # for the April-era deltas that predate survive-and-absorb (and the live
        # dashboard's 12–24h horizon). Modern runs never reach here.
        synthesized, archived_nodes, links = \
            _legacy_window_consolidation(bconn, created_at)

    return {
        "chain_id": chain_id,
        "timestamp": created_at,
        "summary": summary or '',
        "o_summary": ok_payload.get('O', {}).get('summary', ''),
        "k_summary": ok_payload.get('K', {}).get('summary', ''),
        "journal": journal[:1000],
        "synthesized": synthesized,
        "archived": archived_nodes,
        "links": links,
    }


def _legacy_window_consolidation(bconn, created_at):
    """Pre-absorb fallback (see _enrich_consolidation). Reconstructs a run's
    synthesized/archived nodes + links from encoding_source + a ±60min window —
    the original heuristic, kept so April-era forensic views don't regress.
    Returns (synthesized, archived, links) in the same shapes the
    trace-authoritative path produces (similar_to + supersedes edges → links).
    """
    # ±60-minute window. iso_window_around handles midnight/hour rollovers.
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

    return (
        [{"id": n[0], "type": n[1], "title": n[2], "content": n[3], "confidence": n[4]}
         for n in synth_nodes],
        archived_nodes,
        [{"source": e[0], "target": e[1], "relation": "similar_to",
          "description": e[3] or ''} for e in kept_edges]
        + [{"source": e[0], "target": e[1], "relation": "supersedes",
            "description": e[2] or ''} for e in evolved_edges],
    )


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
    def _enrich(bconn, delta_row, nearest_ok):
        chain_id, summary, _meta_raw, created_at, ref_type = delta_row
        ok_payload = nearest_ok(chain_id, created_at)
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

    Healer can run multiple times per day; historical chain_ids are
    `s2-{date}-healer` (not unique per pass) — new ones are seconds-stamped
    and unique, but this serves the historical rows. Pair each delta with its nearest
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
            # Only the unit's per-RUN delta makes a card. Unlike the other run
            # queries this one filters in Python (it needs the O/K rows in the
            # same forward pass, so the SQL can't pre-filter by ref_type).
            # Without this the docstring was a lie: journal_note residue already
            # rendered as phantom passes, and per-write mutation rows would add
            # one card per healed field.
            if (ref_type or '') in _NON_RUN_REF_TYPES:
                continue
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
