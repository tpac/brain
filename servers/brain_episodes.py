"""Episodic recall — decode over the trace substrate.

`recall_episodes` is the decode-over-traces sibling of `recall` (decode-over-
nodes). Traces are the universal record of the whole fractal — S0 exchanges,
S1 runs, S2 runs, all of it — so this is a brain-level *capability*: it spans
every scale, owned by none, and is composed onto Brain as a mixin alongside
BrainRecallMixin. The substrate access (filter_events / filter_event_vectors)
lives in TraceDAL; this layer is the orchestration (rank, hydrate, two paths).

Two needles, composable:
  - query    — semantic, ranks candidates by cosine against the EXISTING
               trace_embeddings (no new embedding). Currently covers s0 only,
               since the embed-reconciliation worker vectorizes s0 traces.
  - contains — exact substring (SQL LIKE over summary + metadata).

ref_type default: tool_result is kept OUT of the default ranked list — not
because it's noise (a file edit / command run is real "what I did" signal) but
because it's high-volume and would flood the common "what was said" query. It's
one explicit step away (ref_type='tool_result'), advertised in the tool. Expand,
not overload: it can join the default later; un-flooding a default can't.
"""
import re
from typing import Any, Dict

from . import embedder
from .clock import iso_cutoff
from .brain_constants import (
    EPISODE_DEFAULT_LIMIT, EPISODE_DEFAULT_WINDOW_DAYS,
    EPISODE_SEMANTIC_CANDIDATE_CAP)


def _resolve_time_bound(value):
    """Resolve a recall_episodes time bound to an ISO created_at string.

    Accepts relative shorthand ('30m', '2h', '3d', '1w', case-insensitive) →
    that-much-ago via iso_cutoff, or an ISO timestamp literal (YYYY-MM-DD…,
    passed through). Returns '' for falsy input. Anything else raises
    ValueError — a malformed bound must fail loud, not silently bind a
    non-timestamp into a lexical comparison (empty result, or a no-op that
    disables the filter). Parsing is a presentation concern, kept out of the
    DAL, which only ever sees resolved ISO bounds.
    """
    if not value:
        return ''
    s = str(value).strip()
    m = re.fullmatch(r'(\d+)\s*([mhdw])', s, re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        kw = {'m': {'minutes': n}, 'h': {'hours': n},
              'd': {'days': n}, 'w': {'days': 7 * n}}[unit]
        return iso_cutoff(**kw)
    # ISO timestamp literal: validate AND normalize via fromisoformat. A bare
    # date-prefix check would pass a space-separated bound ('2026-06-14 12:00')
    # or an out-of-range date verbatim, and a space-separated bound lex-mis-
    # compares against ISO-T storage (the documented 'T'(0x54) > ' '(0x20)
    # hazard). fromisoformat rejects junk/impossible dates and re-emits ISO-T.
    try:
        from datetime import datetime as _dt, timezone as _tz
        dt = _dt.fromisoformat(s)
        if dt.tzinfo is None:        # storage is tz-aware ('+00:00') — match it
            dt = dt.replace(tzinfo=_tz.utc)
        return dt.isoformat()
    except ValueError:
        raise ValueError(
            "time bound %r not understood — use relative shorthand "
            "('30m','2h','3d','1w') or an ISO timestamp ('2026-06-14T12:00:00')"
            % value)


class BrainEpisodesMixin:
    """Brain-level capability: semantic and/or lexical recall over traces,
    returning full episode records. Composed onto Brain."""

    def recall_episodes(self, query: str = None, contains: str = None,
                        session_id: str = None, session_ids=None,
                        scale: str = 's0', event_type: str = None,
                        ref_type=None, older_than: str = None,
                        younger_than: str = None, sort_order: str = 'desc',
                        limit: int = None) -> Dict[str, Any]:
        """Search/filter trace_events and return full episode records — the
        decode-over-traces sibling of recall.

        Needles (composable): query (semantic — ranks vs trace_embeddings;
        s0-only today) and/or contains (substring over summary + metadata).
        Frame: session_id | session_ids, scale (default 's0'; '' = all),
        event_type, ref_type, time bounds (older_than / younger_than accept ISO
        or relative shorthand). With no session scope and no time bound at all,
        a default EPISODE_DEFAULT_WINDOW_DAYS lower bound is applied so an
        unbounded scan can't walk the whole append-only table; an explicit
        older_than is left as-is (no forced floor).

        ref_type: a str (one type) or a list (several). UNSET → the conversation
        default, sourced from the trace_contract dial (CONVERSATIONAL_REF_TYPES)
        at s0 — so the default tracks the contract and can't drift, and
        non-conversational s0 traffic (tool_result, heartbeat, structural
        deltas) stays out of the common query. Pass ref_type='tool_result' for
        the "what I did with files/commands" lens, or the full list (e.g.
        ['user_message','assistant_message','tool_result']) for the interleaved
        said+did timeline. Non-s0 scales have no conversational notion, so unset
        means all ref_types at that scale.

        Ordering: with query, ranked by relevance (each episode carries _score);
        otherwise by created_at (sort_order 'desc' default = latest first).

        Returns {'episodes': [<full trace records>], 'ranked_by':
                 'relevance'|'time', 'truncated': bool} — truncated is True when
                 more matched than were returned/ranked (hit limit, or the
                 semantic candidate cap), so the caller knows it didn't see all.
        """
        from .trace_contract import CONVERSATIONAL_REF_TYPES
        limit = EPISODE_DEFAULT_LIMIT if limit is None else int(limit)
        younger_iso = _resolve_time_bound(younger_than)
        older_iso = _resolve_time_bound(older_than)
        if (not younger_iso and not older_iso
                and not session_id and not session_ids):
            younger_iso = iso_cutoff(days=EPISODE_DEFAULT_WINDOW_DAYS)

        # ref_type whitelist: explicit (str→[str] or list) wins; else the s0
        # conversation default comes from the contract dial (no hardcoded list
        # to drift); other scales have no conversational notion → no filter.
        if ref_type:
            ref_types = [ref_type] if isinstance(ref_type, str) else list(ref_type)
        elif scale == 's0':
            ref_types = list(CONVERSATIONAL_REF_TYPES)
        else:
            ref_types = None
        common = dict(
            contains=contains or '', scale=scale or '',
            event_type=event_type or '', ref_types=ref_types,
            session_id=session_id or '', session_ids=session_ids,
            younger_than=younger_iso, older_than=older_iso)

        # Semantic path: rank a lean candidate scan (id+vector via one JOIN, no
        # metadata decode) by cosine, then batch-hydrate full records for ONLY
        # the top-k. Degrades to the time path on any non-ValueError error
        # (mirrors _trace_chain_candidates); ValueError (e.g. both session
        # forms) surfaces.
        if query:
            try:
                qvec = embedder.embed_query(query)
                if qvec:
                    cands = self._trace_dal.filter_event_vectors(
                        limit=EPISODE_SEMANTIC_CANDIDATE_CAP, **common)
                    if cands:
                        scored = sorted(
                            ((embedder.cosine_similarity(qvec, vec), tid)
                             for tid, vec in cands),
                            key=lambda x: -x[0])[:limit]
                        recs = {r['id']: r for r in self._trace_dal.get_by_ids(
                            [tid for _, tid in scored])}
                        episodes = [dict(recs[tid], _score=round(score, 4))
                                    for score, tid in scored if tid in recs]
                        return {'episodes': episodes,
                                'truncated': len(cands) > limit,
                                'ranked_by': 'relevance'}
                else:
                    # embed_query returned no vector → embedder unavailable.
                    # A no-MATCH (cands empty) is legit and stays quiet; an
                    # un-embeddable semantic request is a real degrade — surface
                    # it instead of silently answering by recency.
                    self._log_error(
                        'recall_episodes_embed_unavailable',
                        RuntimeError('embed_query returned no vector'),
                        'semantic query requested but not embeddable (embedder '
                        'unavailable?); degraded to time path')
            except ValueError:
                raise
            except Exception as e:
                self._log_error('recall_episodes_semantic', e,
                                'semantic rerank failed; degraded to time path')
            # qvec unavailable / nothing embedded / degraded → time path.

        # Time path: indexed WHERE + ORDER BY created_at + LIMIT early-exits, so
        # only `limit` rows are fetched and decoded.
        order = sort_order if sort_order in ('asc', 'desc') else 'desc'
        episodes = self._trace_dal.filter_events(
            sort_order=order, limit=limit, **common)
        return {'episodes': episodes, 'truncated': len(episodes) >= limit,
                'ranked_by': 'time'}
