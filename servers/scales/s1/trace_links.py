"""trace↔node links — what S1 surfaced and encoded, joined to traces.

Scale: S1 (turn). This composes S1's OWN behavioral output — S1R surface
selections and S1E encode runs — into a per-trace view of which nodes the brain
touched around each turn. It is the S1 sibling of `scales/s0/conversation.py`:
S0 composes s0 traces into the conversation; this composes s1 traces into the
node-links over that conversation. Recall lives at the brain level (it spans
every scale); surface+encode are about the turn, so they live here.

It is a consumer-NEUTRAL link, not "provenance" (which is one reader's lens).
The capability returns, per target trace, the nodes linked to it by relation:

    trace_id -> {
        'surfaced':   [node_id, ...],   # recall surfaced AND Haiku picked (S1R)
        'encoded':    [node_id, ...],   # what the owning run wrote    (S1E)
        'encoded_by': trace_id | None,  # the encoding_run trace; None = unencoded
        'authored':   [node_id, ...],   # what Anchor's own tools wrote (S0, anchor_touched)
        'recalled':   [node_id, ...],   # what Anchor deliberately looked up (S0)
        'endo':       [node_id, ...],   # endo-surfaced this turn (S0, empty until wired)
        'dropped':    [node_id, ...],   # offered to Haiku, NOT picked (S1R pool − surfaced)
    }

Two readers, ONE join (do not fork a sibling join per reader — that is how the
2-vs-3-tuple drift happened):
  • The S1 encoder reads a link as "already handled here → revise, don't dupe,"
    and reads `encoded_by is None` as the emergent unencoded boundary.
  • The LAF episodic layers read the SAME link at similar past moments:
    `surfaced` (+act picked), `encoded` (+act learned), `dropped` (−inhibit —
    offered in a situation like this and consistently not chosen).

THE JOIN IS STRUCTURAL, by stop_counter — never by timestamp proximity. Every
chain for turn N ends in `-N` (`s0-{short}-N`, `s1r-{short}-N`, `s1e-{short}-N`),
so `_stop_of` extracts the join key from any of them. surfaced is 1:1 with a turn
(same stop); encode is per-RUN (one run at stop S closes the turn-range ending
at S), so a turn's owning run is the first run whose stop >= the turn's stop, and
`encoded_by` disambiguates which run / marks the boundary.

TWO LAYERS so it is robust to a raw trace dataset run sequentially:
  • nodes_for_traces(...) — PURE. Plain trace records in, link map out. No brain,
    no DB. This is what eval replay (Frozen Corpus), unit tests, and any
    historical/sequential dataset drive directly; it assumes nothing about storage.
  • gather(brain, session_id, streams=...) — thin live adapter. Pulls the named
    trace streams via the public `query_traces` door (no bespoke DAL) and returns
    a DICT keyed by stream name — never a bare tuple (tuple arity is unauditable
    drift; a named dict extends without breaking any consumer). Eval/tests skip it.

Anchor's own actions (`authored`/`recalled`) come from the S0 `anchor_touched`
delta — a per-turn aggregate the daemon flushes at the Stop boundary (the S0
mirror of the S1 encode delta). It reuses the encode delta's `created`/`revised`
keys, so `_delta_ids` parses both with no second path. NOT sourced from
`encoding_source` (in-flux, encoder-invisible) — the touched delta is captured
structurally at dispatch, where only Anchor's TCP calls flow. `endo` rides the
same delta when endo recall is wired.
"""
import json


def _stop_of(chain_id):
    """The stop_counter that names a chain — the trace↔trace join key.

    Chain ids are `{scale}-{session_short}-{N}`; the trailing segment is the
    stop. Returns the int, or None for a missing/odd-shaped chain (those targets
    simply get an empty link rather than crashing the join)."""
    if not chain_id:
        return None
    tail = str(chain_id).rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def _surface_ids(ref_id):
    """Node ids a surface_selected trace recorded — its ref_id is a JSON list.
    Malformed → [] (one bad row never sinks the join; the live writer validates
    on its own side, and this layer stays pure with nothing to log to)."""
    try:
        v = json.loads(ref_id or '[]')
        return list(v) if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def _dedup(seq):
    """Order-preserving dedup — node ids stay in first-seen order (the codebase
    idiom; O(n))."""
    return list(dict.fromkeys(seq))


def _delta_ids(meta, *keys):
    """Node ids out of a delta's id-list fields — the ONE parser shared by the S1
    encode delta and the S0 anchor_touched delta. Both carry the same
    `created`/`revised`/`archived` keys (the symmetry), so neither needs its own
    parsing path; the S0 delta just has extra keys (`recalled`/`endo`) this reads
    on demand. Concatenates the requested keys, order-preserving dedup."""
    out = []
    for k in keys:
        out.extend((meta or {}).get(k) or [])
    return _dedup(out)


def _candidate_outcomes(meta):
    """(candidate_ids, dropped_ids) from a candidate-pool trace's metadata. PURE.

    Reads, in priority order, the two trace shapes that carry the candidate pool:
      1. Δ `additionalContext` — `outcomes_per_candidate {short_id: verdict}` is
         the authoritative Haiku-resolved pool; candidates = its keys, dropped =
         keys whose verdict == 'dropped'. (Falls back to the `dropped`/`selected`
         mirror lists if outcomes is absent but those are present.)
      2. O `recall` — `candidates ["short_id|title|score|type", ...]` is the raw
         pool with no verdict; returns (candidate_ids, None) — the caller derives
         dropped as candidates − surfaced picks.
    Returns (candidate_ids, dropped_ids_or_None). Empty/odd metadata → ([], None),
    so a malformed row contributes nothing rather than crashing the join."""
    if not isinstance(meta, dict):
        return [], None
    outcomes = meta.get('outcomes_per_candidate')
    if isinstance(outcomes, dict) and outcomes:
        cands = list(outcomes.keys())
        dropped = [cid for cid, v in outcomes.items() if v == 'dropped']
        return cands, dropped
    # Δ trace mirror lists (outcomes absent but selected/dropped present)
    sel = meta.get('selected')
    drp = meta.get('dropped')
    if isinstance(sel, list) or isinstance(drp, list):
        sel = list(sel or [])
        drp = list(drp or [])
        return _dedup(sel + drp), drp
    # O recall trace: "short_id|title|score|type" rows, no verdict
    raw = meta.get('candidates')
    if isinstance(raw, list) and raw:
        cands = [str(r).split('|', 1)[0] for r in raw if str(r).split('|', 1)[0]]
        return _dedup(cands), None
    return [], None


def nodes_for_traces(surface_traces, encode_traces, target_traces,
                     touched_traces=None, recall_traces=None):
    """Join S1 surface/encode + S0 anchor_touched + S1R candidate-pool traces to
    target traces by stop. PURE.

    Args:
        surface_traces: S1R `surface_selected` trace records (chain_id + ref_id).
        encode_traces:  S1E `encoding_run` trace records (chain_id + metadata
                        with `created`/`revised` node-id lists).
        target_traces:  the traces to attach links to — each a record with `id`
                        (the map key) and `chain_id` (the stop to join on).
                        Usually the turns' user_message traces; a recall layer
                        passes embedding-matched traces instead.
        touched_traces: S0 `anchor_touched` trace records (chain_id + metadata) —
                        what Anchor's OWN tools touched. Optional (the feed may not
                        be wired); when omitted, authored/recalled/endo are empty.
        recall_traces:  S1R candidate-POOL trace records — either Δ
                        `additionalContext` rows (metadata.outcomes_per_candidate,
                        the authoritative Haiku-resolved pool + verdict) or O
                        `recall` rows (metadata.candidates, the raw pool). Mixed is
                        fine; per stop the verdict shape wins. Optional; when
                        omitted, `dropped` is empty.

    Returns:
        {target_trace_id: {'surfaced': [ids], 'encoded': [ids],
                           'encoded_by': run_trace_id | None,
                           'authored': [ids], 'recalled': [ids], 'endo': [ids],
                           'dropped': [ids]}}

    surfaced / authored / recalled / endo / dropped are 1:1 with the turn (same
    stop); encoded is per-RUN (the owning run). `dropped` = the turn's candidate
    pool MINUS surfaced (explicit verdict wins over the derived subtraction; a
    node never appears in both — surfaced wins). surfaced/dropped ids are what
    the surface traces record (8-char short ids); encoded/authored are full —
    verbatim either way, width resolution is the consumer's job.
    """
    # surfaced, indexed by stop (1:1 with a turn; merge if a stop ever repeats).
    surf_by_stop = {}
    for t in surface_traces:
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        bucket = surf_by_stop.setdefault(stop, [])
        bucket.extend(_surface_ids(t.get('ref_id')))

    # anchor_touched, indexed by stop (1:1 with a turn, like surfaced). authored =
    # created∪revised (live nodes Anchor wrote; archived is recorded in the trace
    # but not surfaced as a link — the node is gone). recalled = deliberate
    # lookups; endo = endo-surface (empty until wired). Shared _delta_ids parser.
    touched_by_stop = {}
    for t in (touched_traces or []):
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        meta = t.get('metadata')
        e = touched_by_stop.setdefault(stop, {'authored': [], 'recalled': [], 'endo': []})
        e['authored'].extend(_delta_ids(meta, 'created', 'revised'))
        e['recalled'].extend(_delta_ids(meta, 'recalled'))
        e['endo'].extend(_delta_ids(meta, 'endo'))

    # candidate pool + explicit-dropped (when the trace carries a verdict),
    # indexed by stop. A stop may have both an O row (raw pool) and a Δ row
    # (verdict) — keep the union of candidates and any explicit dropped set.
    cands_by_stop = {}
    explicit_dropped_by_stop = {}
    for t in (recall_traces or []):
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        cands, dropped = _candidate_outcomes(t.get('metadata') or {})
        if cands:
            cands_by_stop.setdefault(stop, []).extend(cands)
        if dropped is not None:
            explicit_dropped_by_stop.setdefault(stop, []).extend(dropped)

    # encode runs as (stop, trace_id, [created+revised]), ascending by stop.
    runs = []
    for t in encode_traces:
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        ids = _delta_ids(t.get('metadata'), 'created', 'revised')  # shared parser
        runs.append((stop, t.get('id'), ids))
    runs.sort(key=lambda r: r[0])

    out = {}
    for tt in target_traces:
        tid = tt.get('id')
        if tid is None:
            continue
        stop = _stop_of(tt.get('chain_id'))
        surfaced = _dedup(surf_by_stop.get(stop, [])) if stop is not None else []
        encoded, encoded_by = [], None
        if stop is not None:
            # owning run = first run that closes a range at/after this turn.
            # LIMIT (proximity is heuristic): the encoder reads a sliding message
            # window, so a single failed run's turns are normally re-covered by
            # the next run's lookback (correct). But if 2+ consecutive runs fail,
            # a turn can be attributed to a later run that never saw it — a false
            # "already encoded." Traces carry no per-run input-range to fix this
            # precisely; the encoder's "revise if shifted, don't blind-skip"
            # how-to-read is the mitigation. Rare (needs consecutive failures).
            for rstop, rtid, rids in runs:
                if rstop >= stop:
                    encoded, encoded_by = list(rids), rtid
                    break
        tch = touched_by_stop.get(stop) or {} if stop is not None else {}
        # dropped: explicit verdict wins over the derived pool−surfaced; either way
        # a node can never be both surfaced and dropped at one stop — surfaced wins.
        surfaced_set = set(surfaced)
        if stop is None:
            pool = []
        elif stop in explicit_dropped_by_stop:
            pool = explicit_dropped_by_stop[stop]
        else:
            pool = cands_by_stop.get(stop, [])
        dropped = [d for d in _dedup(pool) if d not in surfaced_set]
        out[tid] = {
            'surfaced': surfaced, 'encoded': encoded, 'encoded_by': encoded_by,
            'authored': _dedup(tch.get('authored', [])),
            'recalled': _dedup(tch.get('recalled', [])),
            'endo': _dedup(tch.get('endo', [])),
            'dropped': dropped,
        }
    return out


def session_node_ids(encode_traces, touched_traces):
    """Session-level UNION of the nodes the brain wrote/Anchor touched — the
    catalog's view (vs nodes_for_traces' per-turn view). PURE. No turn keying:
    the widened catalog wants every id that appears anywhere in the session's
    encode + touched traces, so it can hold each body once. Same `_delta_ids`
    parser as everywhere else.

    Returns {'encoded': set, 'authored': set, 'recalled': set} — `encoded` from
    the S1 encode runs (created∪revised), `authored`/`recalled` from the S0
    anchor_touched deltas. `surfaced` is NOT here: the catalog already has it from
    the Haiku judge outputs. `endo` is omitted until the stream is wired.
    """
    encoded, authored, recalled = set(), set(), set()
    for t in (encode_traces or []):
        encoded.update(_delta_ids(t.get('metadata'), 'created', 'revised'))
    for t in (touched_traces or []):
        authored.update(_delta_ids(t.get('metadata'), 'created', 'revised'))
        recalled.update(_delta_ids(t.get('metadata'), 'recalled'))
    return {'encoded': encoded, 'authored': authored, 'recalled': recalled}


# The streams gather() can pull: name → (ref_type, scale). One registry, so a
# new stream is one line here and zero signature changes anywhere.
GATHER_STREAMS = {
    'surface': ('surface_selected', 's1'),    # S1R picks (the `surfaced` role)
    'encode':  ('encoding_run', 's1'),        # S1E deltas (the `encoded` role)
    'touched': ('anchor_touched', 's0'),      # Anchor's own tool actions
    'recall':  ('additionalContext', 's1'),   # candidate pool + verdict (`dropped`)
}


def gather(brain, session_id, streams=('surface', 'encode', 'touched'), limit=500):
    """Live adapter: fetch named trace streams for a session. Thin.

    Composes the public `query_traces` door (no bespoke DAL, per the §10.2
    trace-query law). Returns {stream_name: [trace records]} for the requested
    `streams` — a NAMED dict, never a bare tuple: tuple arity drifts silently
    when two consumers evolve the stream list apart (the 2-vs-3-tuple lesson);
    a keyed dict extends without breaking anyone. Eval/tests bypass this and
    feed records to nodes_for_traces directly.

    Defaults to the encoder's three streams. The episodic layers pass
    ('surface', 'encode', 'recall') — `recall` is the Δ `additionalContext`
    candidate-pool rows carrying `metadata.outcomes_per_candidate` (the
    authoritative Haiku-resolved pool + verdict; `_candidate_outcomes` also
    reads the O `recall` raw-pool shape for datasets that only logged that).

    `limit` is the session trace-pull bound (created_at DESC, so it keeps the
    most-recent traces — what the recent-turn window needs). Distinct from the
    timeline's LIVED_SEQUENCE_PULL; a reusable adapter default, not a per-piece
    constant. A consumer with a different need passes its own.
    """
    out = {}
    for name in streams:
        ref_type, scale = GATHER_STREAMS[name]   # unknown name = caller bug, loud
        out[name] = brain.query_traces(
            ref_type=ref_type, scale=scale,
            session_id=session_id, hours=None, limit=limit).get('events', [])
    return out
