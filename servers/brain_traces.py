"""The traces functional layer — every trace read (and API-level write) on Brain.

One rule, no judgment: reading or writing traces through the API? It's a
`brain.` method, and it lives HERE. Only this file touches TraceDAL; the
vocabulary (ref types, turn classification, journal parse/render) stays in
trace_contract.py. Sanctioned direct-DAL exceptions: the recall scoring
engines' vector-substrate pulls (`event_vector_rows` — brain_recall's
trace-chain lane, recall_laf's episodic matrix) and the read-only dashboard.

Sections:
- Generic door   — query_traces, get_trace, get_traces, count_traces
- Journal + arc  — journal_notes, write_journal_notes, write_session_arc
- Episodic       — recall_episodes (decode-over-traces sibling of recall)
- Conversation   — get_conversation, turns_since_last_encode,
                   get_conversation_around (+ JSONL fallback for pre-trace
                   history)

Traces are the universal record of the whole fractal — S0 exchanges, S1 runs,
S2 runs — so these are brain-level capabilities: they span every scale, owned
by none. Scale packages (scales/s0|s1|s2) host integration units, never data
access; scale is a tag in the substrate, not a boundary in the read path.

Design: docs/TRACES-LAYER-DESIGN.md
"""
import json
import os
import re
import time
from bisect import bisect_left
from functools import lru_cache
from typing import Any, Dict, List

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


class BrainTracesMixin:
    """Brain-level trace capabilities: the generic query door, journal/arc
    residue, episodic recall, and conversation reads. Composed onto Brain."""

    # ── Generic door ──

    def get_trace(self, trace_id):
        """Single trace_event point lookup. Returns the full row dict
        (id/chain_id/scale/event_type/ref_type/ref_id/summary/metadata/
        session_id/created_at) or None if not found.

        Convention mirrors brain.get_node — same single-input shape so
        callers don't have to think about batch vs point at the API
        edge. For batch lookups use brain.get_traces.
        """
        if trace_id is None:
            return None
        rows = self._trace_dal.get_by_ids([trace_id])
        return rows[0] if rows else None

    def get_traces(self, trace_ids):
        """Batch trace_event lookup. Returns a list of full row dicts
        in ascending-id order; missing ids are silently skipped.

        Use this when a caller has a list of trace_ids (e.g., expanding
        node.source_refs at render time, or get_traces tool exposure
        for the encoder). Single point lookup → use brain.get_trace.
        """
        if not trace_ids:
            return []
        return self._trace_dal.get_by_ids(list(trace_ids))

    def query_traces(self, scale: str = '', hours: int = 24,
                     event_type: str = '', chain_id: str = '',
                     session_id: str = '', session_ids=None,
                     ref_type: str = '', ref_id: str = '', chain_suffix: str = '',
                     exclude_ref_types=None,
                     grouped: bool = False, limit: int = 100,
                     older_than: str = ''):
        """Query trace events — the fractal learning loop data.

        Modes:
        - chain_id set: return single chain with all events
        - ref_type set: filter events by ref_type (+ optional ref_id to scope to
          one subject, chain_suffix to scope to one S2 unit's chains, session_id
          to scope to one session, hours=None to disable the time window)
        - grouped=True + session_id: return chains grouped with nested events
          (NOTE: unlike the flat single-session pull, the grouped path IS
          hours-bound — pass a wide hours for historical sessions)
        - session_ids (list) set: cross-session pull; hours ignored
        - session_id (str) set: single-session pull; hours ignored
        - default: return flat recent events (hours-bound; + optional chain_suffix
          to scope to one S2 unit, exclude_ref_types to drop residue like
          journal_note, hours=None to disable the window)

        older_than (ISO, strict `created_at <`) positions the newest-first
        LIMIT window at a historical instant — the replay as-of bound,
        pushed into SQL so the limit clips the right end of the ordering.
        Applies to the flat modes (ref_type / default); chain_id and grouped
        pulls are whole-chain reads and don't take it.
        """
        if chain_id:
            return {'chain': self._trace_dal.get_chain(chain_id)}
        if ref_type:
            return {'events': self._trace_dal.get_by_ref_type(
                ref_type=ref_type, scale=scale, hours=hours, limit=limit,
                session_id=session_id, ref_id=ref_id, chain_suffix=chain_suffix,
                older_than=older_than)}
        if grouped and session_id:
            return {'chains': self._trace_dal.get_chains(
                session_id=session_id, scale=scale, hours=hours, limit=limit)}
        # Single or multi session pulls — both authoritative, both ignore hours.
        # get_recent raises ValueError if both are set; we don't second-guess.
        return {'events': self._trace_dal.get_recent(
            scale=scale, hours=hours, event_type=event_type,
            session_id=session_id, session_ids=session_ids, limit=limit,
            chain_suffix=chain_suffix, exclude_ref_types=exclude_ref_types,
            older_than=older_than)}

    def count_traces(self, field: str, scale: str = '', hours: int = 24):
        """Count trace events grouped by a field."""
        return self._trace_dal.count_by(field=field, scale=scale, hours=hours)

    # ── Journal + arc (encoder residue) ──

    def journal_notes(self, *, subject: str = '', scale: str = '',
                      session_id: str = '', unit: str = '',
                      k=None, limit: int = 200):
        """Read encoder journal notes — through the trace API, never raw SQL.

        Composes query_traces(ref_type='journal_note', ...) — the public door;
        TraceDAL stays underneath it. Two modes:
        • subject set → every note ABOUT that subject (ref_id), newest first
          (the hotspot view: N notes on one subject).
        • else → continuity: notes from the last K note-bearing RUNS of an
          encoder, scoped by scale + (session_id for S1 | unit for S2). K
          defaults to JOURNAL_CONTINUITY_RUNS[encoder] (s1e / unit) → DEFAULT.

        Runs group by chain_id (per-run-unique at both scales). Returns note
        dicts {tag, note, subject, chain_id, created_at}, newest first.
        Subject→title resolution is left to the render layer — it avoids a
        heavy get_node per note here, and the consumer already holds the node.
        """
        from .trace_contract import (JOURNAL_CONTINUITY_RUNS,
                                      JOURNAL_CONTINUITY_RUNS_DEFAULT,
                                      JOURNAL_RESOLVE_TAGS, JOURNAL_OPEN_TAGS,
                                      JOURNAL_OPEN_PIN_CAP)
        events = self.query_traces(
            ref_type='journal_note', scale=scale, ref_id=subject,
            session_id=session_id, chain_suffix=unit, hours=None, limit=limit,
        ).get('events', [])
        open_meta = {}   # id(event) → {'open_runs': N, 'first_seen': iso}
        if not subject:  # continuity: resolve-filter + K runs + open pins
            if k is None:
                key = 's1e' if scale == 's1' else unit
                k = JOURNAL_CONTINUITY_RUNS.get(key, JOURNAL_CONTINUITY_RUNS_DEFAULT)

            def _tag(e):
                return ((e.get('metadata') or {}).get('tag') or '').strip().casefold()

            def _subj(e):
                return (e.get('ref_id') or '').strip().casefold()

            # Pass 1 — resolve-filtering, newest→oldest across ALL fetched
            # events: a `resolved`/`retire` note retires every strictly-older
            # note with the same (normalized) subject. Read-time only; the
            # trace rows are untouched. The resolve note itself stays until it
            # ages out — it documents the resolution.
            resolved_seen, alive = set(), []
            for e in events:                       # created_at DESC
                s = _subj(e)
                if s and s in resolved_seen:
                    continue
                if s and _tag(e) in JOURNAL_RESOLVE_TAGS:
                    resolved_seen.add(s)
                alive.append(e)

            # Pass 2 — the K-run window over surviving notes.
            seen, window = [], []
            for e in alive:
                ch = e.get('chain_id') or ''
                if ch not in seen:
                    if len(seen) >= k:
                        break
                    seen.append(ch)
                window.append(e)

            # Pass 3 — open pins: the newest surviving `open`-tagged note per
            # subject stays visible beyond the window until resolved (capped).
            # ×N persistence = distinct runs mentioning the subject, computed
            # over the post-resolve SURVIVORS — a resolve closes the epoch, so
            # a re-opened subject starts at ×1 with a fresh first_seen. The
            # reader does the bumping, never the encoder.
            # Horizon note: everything here operates within the `limit` newest
            # fetched events (~dozens of runs). That bound is the backstop of
            # last resort — the ×N nudge fires at JOURNAL_OPEN_NUDGE_RUNS,
            # long before any pin could age past the horizon.
            runs_by_subj, first_seen = {}, {}
            for e in alive:
                s = _subj(e)
                if not s:
                    continue
                if _tag(e) in JOURNAL_RESOLVE_TAGS:
                    continue  # a resolve closes an epoch; it doesn't open one
                ch = e.get('chain_id')
                if ch:
                    runs_by_subj.setdefault(s, set()).add(ch)
                c = e.get('created_at') or ''
                if c and (s not in first_seen or c < first_seen[s]):
                    first_seen[s] = c

            in_window = {id(e) for e in window}
            pinned, pinned_subjects, pins_dropped = [], set(), 0
            for e in alive:                        # newest first
                if _tag(e) not in JOURNAL_OPEN_TAGS:
                    continue
                s = _subj(e)
                if not s or s in pinned_subjects:
                    continue
                pinned_subjects.add(s)
                open_meta[id(e)] = {
                    'open_runs': len(runs_by_subj.get(s, ())) or 1,
                    'first_seen': first_seen.get(s, ''),
                }
                if id(e) not in in_window:
                    if len(pinned) < JOURNAL_OPEN_PIN_CAP:
                        pinned.append(e)
                    else:
                        pins_dropped += 1
            if pins_dropped:
                # Loud by default: a dropped pin is an unresolved open item
                # silently leaving the encoder's sight.
                self._log_warning(
                    'journal_open_pin_overflow',
                    '%s/%s: %d open item(s) beyond the %d-pin cap dropped from '
                    'continuity — resolve or promote some' % (
                        scale, unit or session_id, pins_dropped,
                        JOURNAL_OPEN_PIN_CAP))

            events = window + pinned
        return [{
            'tag': (e.get('metadata') or {}).get('tag', ''),
            'note': (e.get('metadata') or {}).get('note', ''),
            'subject': e.get('ref_id', ''),
            'chain_id': e.get('chain_id', ''),
            'created_at': e.get('created_at', ''),
            **open_meta.get(id(e), {}),
        } for e in events]

    def write_journal_notes(self, *, final_text, chain_id, scale, session_id=''):
        """Write door — the mirror of journal_notes (read). Extract the
        encoder's `## Review` fenced block, parse it, and write each note as its
        own journal_note trace row (event_type='delta', ref_id=subject), all
        sharing the run's chain_id.

        Returns a structured result so the caller (and the trace) can see what
        happened: `{'written': int, 'malformed': int, 'status': str}` where
        status is one of:
          • 'ok'                 — a non-empty review processed (counts tell the rest)
          • 'salvaged'           — no `## Review` heading, but a heading-less fence
                                   of valid notes was harvested (drift, logged loud)
          • 'empty_review'       — a fenced review that was empty (a legit clean run)
          • 'no_review_section'  — the encoder emitted no `## Review` at all
          • 'no_review_extracted'— `## Review` present but no parseable fence (drift)
          • 'error'              — an unexpected failure (isolated; see below)

        LOUD BY DEFAULT — nothing is dropped silently. The encoder is expected
        to ALWAYS emit a `## Review` section (empty fence on a clean run), so a
        missing section or a broken fence is real drift and gets a warning. A
        malformed line, a subject-less note, or unbuildable metadata each logs
        loud and is skipped — one bad note never sinks the rest. If notes parsed
        but none survived, that's an accidental full drop → loud. The whole body
        is failure-isolated: any unexpected error is logged loud and swallowed —
        a journal write must never break or roll back the encoder's actual run.
        """
        from .trace_contract import (extract_review_block, parse_journal_notes,
                                      build_journal_note_metadata,
                                      salvage_review_fence,
                                      JOURNAL_REVIEW_MARKER)
        try:
            salvaged = False
            block = extract_review_block(final_text)
            if block is None:
                # Drift salvage: the encoder sometimes writes a valid notes
                # fence but drops the heading — harvest it (strict gate in
                # salvage_review_fence) rather than lose the batch's residue.
                # Still a warning: drift stays visible, just no longer lossy.
                block = salvage_review_fence(final_text)
                if block is not None:
                    salvaged = True
                    self._log_warning(
                        'journal_note_review_salvaged',
                        'chain=%s: no %r heading, but a valid heading-less notes '
                        'fence was found — salvaged (format drift)'
                        % (chain_id, JOURNAL_REVIEW_MARKER))
                elif JOURNAL_REVIEW_MARKER in (final_text or ''):
                    self._log_warning(
                        'journal_note_no_review_extracted',
                        'chain=%s: %r present but no parseable fenced block'
                        % (chain_id, JOURNAL_REVIEW_MARKER))
                    return {'written': 0, 'malformed': 0,
                            'status': 'no_review_extracted'}
                else:
                    self._log_warning(
                        'journal_note_no_review_section',
                        'chain=%s: encoder final_text (%d chars) has no %r section'
                        % (chain_id, len(final_text or ''), JOURNAL_REVIEW_MARKER))
                    return {'written': 0, 'malformed': 0,
                            'status': 'no_review_section'}
            if block == '':
                # Fenced review present but empty — the legit "clean run, nothing
                # to note" case. Visible (debug), not an alarm.
                self.log_debug('journal_note_empty_review', 'write_journal_notes',
                               chain_id=chain_id)
                return {'written': 0, 'malformed': 0, 'status': 'empty_review'}

            notes, malformed = parse_journal_notes(block)
            for raw in malformed:
                self._log_warning('journal_note_malformed',
                                  'chain=%s: %s' % (chain_id, raw[:200]))
            events = []
            for n in notes:
                subject = (n.get('subject') or '').strip()
                if not subject:
                    self._log_warning('journal_note_no_subject',
                                      'chain=%s: %s' % (chain_id, str(n)[:200]))
                    continue
                try:
                    meta = build_journal_note_metadata(note=n['note'],
                                                       tag=n.get('tag', ''))
                except ValueError as e:
                    self._log_warning('journal_note_build_failed',
                                      'chain=%s: %s | %s' % (chain_id, e, str(n)[:160]))
                    continue
                events.append({
                    'chain_id': chain_id, 'scale': scale, 'event_type': 'delta',
                    'ref_type': 'journal_note', 'ref_id': subject,
                    'summary': meta['note'][:80], 'metadata': meta,
                    'session_id': session_id,
                })
            if events:
                self._trace_dal.append_batch(events)
            if notes and not events:
                self._log_warning(
                    'journal_note_all_dropped',
                    'chain=%s: parsed %d notes but wrote 0 (all failed subject/build)'
                    % (chain_id, len(notes)))
            return {'written': len(events), 'malformed': len(malformed),
                    'status': 'salvaged' if salvaged else 'ok'}
        except Exception as e:
            self._log_error('journal_note_write_failed', e, 'chain=%s' % chain_id)
            return {'written': 0, 'malformed': 0, 'status': 'error'}

    def write_session_arc(self, *, final_text, session_id, limit=800):
        """Write door for the session arc — the journal mechanism's second
        component (design §7.2: Encode → Arc → Review). Extract the encoder's
        `## Arc` fenced one-liner and ACCUMULATE it onto the per-session
        digest at `session_context_{session_id}` (read back by
        session_context_for: the Frame's 'Current focus', the next encoding
        run's context block, and recall ranking).

        `limit` bounds the accumulated digest (chars); truncation drops the
        OLDEST lines from the front — the digest is a rolling journey, newest
        movement last. Caller passes its contract value (S1E:
        ENCODING_AGENT['session_context_limit']).

        Returns {'written': bool, 'status': str} where status is one of:
          • 'ok'              — a non-empty arc line appended
          • 'empty_arc'       — fenced arc present but empty (legit: nothing
                                progressed this run)
          • 'no_arc_section'  — no `## Arc` at all (drift — the block was
                                injected, the encoder ignored it)
          • 'no_arc_extracted'— `## Arc` present but no parseable fence (drift)
          • 'error'           — unexpected failure (isolated; never breaks
                                the encoder's run)

        LOUD BY DEFAULT — an opted-in encoder is expected to ALWAYS emit
        `## Arc` (empty fence when nothing moved), so a missing section or a
        broken fence logs a warning. An empty fence is visible (debug), not
        an alarm.
        """
        from .trace_contract import extract_arc_block, JOURNAL_ARC_MARKER
        try:
            if not session_id:
                return {'written': False, 'status': 'error'}
            block = extract_arc_block(final_text)
            if block is None:
                if JOURNAL_ARC_MARKER in (final_text or ''):
                    self._log_warning(
                        'session_arc_no_arc_extracted',
                        'session=%s: %r present but no parseable fenced block'
                        % (session_id[:8], JOURNAL_ARC_MARKER))
                    return {'written': False, 'status': 'no_arc_extracted'}
                self._log_warning(
                    'session_arc_no_arc_section',
                    'session=%s: encoder final_text (%d chars) has no %r section'
                    % (session_id[:8], len(final_text or ''), JOURNAL_ARC_MARKER))
                return {'written': False, 'status': 'no_arc_section'}
            if block == '':
                self.log_debug('session_arc_empty', 'write_session_arc',
                               session_id=session_id)
                return {'written': False, 'status': 'empty_arc'}
            # Accumulate: newline-joined journey, oldest lines truncated from
            # the front at a line boundary (same shape _save_session_context
            # kept for the legacy SESSION_CONTEXT: path).
            new_line = block.splitlines()[0].strip()  # ONE line by contract
            existing = self.session_context_for(session_id)
            combined = (existing + '\n' + new_line) if existing else new_line
            if len(combined) > limit:
                truncated = combined[len(combined) - limit:]
                nl_idx = truncated.find('\n')
                if 0 <= nl_idx < 60:
                    truncated = truncated[nl_idx + 1:]
                combined = truncated
            self.set_config('session_context_' + session_id, combined)
            return {'written': True, 'status': 'ok'}
        except Exception as e:
            self._log_error('session_arc_write_failed', e,
                            'session=%s' % (session_id or '')[:8])
            return {'written': False, 'status': 'error'}

    # ── Episodic recall ──

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
        the "what I did with files/commands" lens, or SAID_AND_DID_REF_TYPES for
        the interleaved said+did timeline. Non-s0 scales have no conversational
        notion, so unset means all ref_types at that scale.

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

    # ── Conversation ──

    def get_conversation(self, session_id: str, limit: int = 20,
                         with_judge_output: bool = True,
                         with_surfaced: bool = False,
                         exclude_trace_id: str = None,
                         older_than: str = None) -> List[Dict]:
        """Get recent conversation turns for a session.

        The simple path — S1E, scribe_due, the surface window, the LAF
        moment stack: anything that knows its session_id and wants the last
        N turns. No timestamp resolution, no JSONL fallback (historic
        center-on-a-moment lookups are get_conversation_around's job).

        Returns: [{role, content, timestamp, trace_id, judge_output}]
            trace_id: 8-char hex id from trace_events (v29) — used by S1 encoder
                      to populate source_refs via `[trace:<hex>]` inline markers.
            judge_output: surface selection from S1R for the user turn (if any).
                      Filling it costs an extra query over the window's recall
                      chains — callers that only read role/content (the
                      scribe_due poll) pass with_judge_output=False.
            with_surfaced=True adds `surfaced` per user turn — the memories
                      the surface selected ([{id, title}]); the v13 XML
                      surface layout's <shown> source.
            exclude_trace_id drops one row in SQL — mid-turn readers that
                      want PREVIOUS turns only pass the current prompt's
                      trace id (see get_session_turns for the interrupt
                      subtlety).
            older_than: ISO strict `created_at <` bound, applied in SQL —
                      the replay as-of cut: "the last N turns as of that
                      instant", not "the last N turns now, minus the future".
        """
        try:
            turns = self._trace_dal.get_session_turns(
                session_id, limit=limit, with_judge_output=with_judge_output,
                with_surfaced=with_surfaced, exclude_trace_id=exclude_trace_id,
                older_than=older_than)
            out = []
            for t in turns:
                row = {'role': t['role'],
                       'trace_id': t.get('trace_id'),
                       'content': t.get('content', ''),
                       'timestamp': t.get('timestamp', ''),
                       'judge_output': t.get('judge_output', '')}
                if with_surfaced:
                    row['surfaced'] = t.get('surfaced', [])
                out.append(row)
            return out
        except Exception as e:
            # Empty-list degrade keeps every consumer alive (Scribe cadence,
            # surface window, moment stack) — but never silently.
            self._log_error('get_conversation', e,
                            'session=%s' % (session_id or '')[:8])
            return []

    def turns_since_last_encode(self, session_id: str) -> int:
        """Conversational turns this session has had since its last S1 encode.

        The S1 Scribe's cadence signal, read LIVE from traces instead of a
        maintained counter. The old `conversational_count` desynced across
        resume/restart (boot reset it while the traces stayed truthful), which
        starved the Scribe — so the gate now derives the count from the event
        log that never lies.

        Anchors on the most recent SUCCESSFUL run — the latest `encoding_run`
        delta — not the latest attempt. Anchoring on `encoding_prompt` (the
        old form) meant a run that failed AFTER writing its prompt trace
        silently reset the cadence: its turns were skipped, not retried, and
        a failed tail encode was never retried at all (found live 2026-07-28,
        fb78aab9 #38). The count anchors at that successful run's START (its
        chain's encoding_prompt timestamp), so turns that arrived while it ran
        still count as unencoded. Failed attempts stay "due"; the daemon's
        retry cooldown paces the re-fire and scribe_repeated_failure escalates
        a wedged session. No prior successful encode → counts all turns, so a
        fresh session fires at the threshold. Same turn definition the encoder
        reads (trace_contract).
        """
        if not session_id:
            return 0
        last = self._trace_dal.get_by_ref_type(
            'encoding_run', scale='s1', session_id=session_id,
            hours=None, limit=1)
        since = ''
        if last:
            chain = last[0].get('chain_id') or ''
            prompts = self._trace_dal.get_by_ref_type(
                'encoding_prompt', scale='s1', session_id=session_id,
                hours=None, limit=50)
            since = next((p['created_at'] for p in prompts
                          if p.get('chain_id') == chain),
                         last[0]['created_at'])
        return self._trace_dal.conversational_turns_since(session_id, since)

    def get_conversation_around(self, node_id: str = None,
                                session_id: str = None,
                                timestamp: str = None,
                                before: int = 10, after: int = 5) -> List[Dict]:
        """Get conversation exchanges around a point in time.

        Resolution order:
        1. If session_id + timestamp given: query that session directly
        2. If node_id given: find encoding trace → get session_id + timestamp
        3. If only timestamp given: find nearest session from traces
        4. If traces fail: fall back to JSONL conversation logs

        Args:
            node_id: Node ID — resolves to the conversation that created it
            session_id: Full session UUID — skip searching, query directly
            timestamp: ISO timestamp to center the window on
            before: Exchanges before the timestamp (default 10)
            after: Exchanges after the timestamp (default 5)

        Returns: [{role: 'user'|'assistant', content: str, timestamp: str}]
                 Chronological order. Empty list if no conversation found.
        """
        resolved_session = session_id
        resolved_timestamp = timestamp

        if node_id and not timestamp:
            resolved_timestamp = self._resolve_node_timestamp(node_id)

        if not resolved_timestamp:
            return []

        # If we have node_id but no session, find the encoding session
        if node_id and not resolved_session:
            resolved_session, enc_ts = self._find_encoding_session(
                node_id, resolved_timestamp)
            if enc_ts:
                # Use encoding timestamp, more precise than created_at
                resolved_timestamp = enc_ts

        # Strategy 1: S0 traces (post-April 5)
        if resolved_session:
            turns = self._conversation_by_session(
                resolved_session, resolved_timestamp, before, after)
            if turns:
                return turns

        # Strategy 2: Find session by timestamp proximity
        turns = self._conversation_by_timestamp(
            resolved_timestamp, before, after)
        if turns:
            return turns

        # Strategy 3: JSONL conversation logs (pre-April 5)
        return _from_jsonl(resolved_timestamp, before, after)

    def _resolve_node_timestamp(self, node_id):
        """Get a node's created_at timestamp (full id or short-id prefix) —
        through NodeDAL, so prefix-resolution semantics live in one place."""
        nid = (self._nodes.resolve_id(node_id)
               or self._nodes.resolve_id(node_id[:8]))
        if not nid:
            return None
        node = self._nodes.get_naked_node(nid)
        return node['created_at'] if node else None

    def _find_encoding_session(self, node_id, node_created_at):
        """Find which session encoded this node.

        Checks S1E traces for this node_id. Returns (session_id,
        encoding_timestamp).
        """
        short_id = node_id[:8]
        try:
            # Direct match: S1E trace metadata contains this node ID
            hit = self._trace_dal.find_by_metadata_substring(
                's1', 'encoding_run', short_id)
            if hit and hit['session_id']:
                return hit['session_id'], hit['created_at']

            # Fallback: nearest S1E trace before node creation, SAME DAY
            # (prevents matching traces from completely different sessions)
            node_date = node_created_at[:10]
            hit = self._trace_dal.latest_in_window(
                's1', 'encoding_run', node_created_at, node_date + 'T00:00:00')
            if hit and hit['session_id']:
                return hit['session_id'], hit['created_at']

        except Exception:
            pass

        return None, None

    def _conversation_by_session(self, session_id, timestamp, before, after):
        """Get conversation from S0 traces for a specific session."""
        try:
            turns = self._trace_dal.get_session_turns(
                session_id,
                around_timestamp=timestamp,
                before=before,
                after=after,
                with_judge_output=False,
            )
            if turns:
                return [{'role': t['role'], 'content': t.get('content', ''),
                          'timestamp': t.get('timestamp', '')} for t in turns]
        except Exception:
            pass
        return []

    def _conversation_by_timestamp(self, timestamp, before, after):
        """Find the session active at a timestamp and get its conversation."""
        try:
            # Find the S0 trace closest to this timestamp
            hit = self._trace_dal.latest_in_window(
                's0', 'user_message', timestamp, timestamp[:10] + 'T00:00:00')
            if hit and hit['session_id']:
                return self._conversation_by_session(
                    hit['session_id'], timestamp, before, after)
        except Exception:
            pass
        return []

    # ═══════════════════════════════════════════════════════════
    # Payload recorder (docs/TRACE-MODES-DESIGN.md)
    # ═══════════════════════════════════════════════════════════
    # Fat payloads (full prompts, per-round payloads, failed-run conversations)
    # live in FILES under {db_dir}/payloads/ — never in trace rows. The trace
    # row stays the authoritative index (bounded forensics + this pointer);
    # a deleted file degrades to a clean "pruned" read, so the user deleting
    # payloads from outside is always safe.

    def _payload_root(self):
        """{db_dir}/payloads — derived from THIS brain instance's db_path,
        never a global env var (a88343d6: env-resolved writer/reader path
        seams split-brain silently). IsolatedBrain and eval's fresh brains
        get their own root by construction."""
        return os.path.join(os.path.dirname(os.path.abspath(self.db_path)),
                            'payloads')

    def _payload_chain_dir(self, chain_id):
        """Date-first, chain-second. The date dir is the chain's FIRST
        payload's day: a chain dir already existing under yesterday is reused
        so a run straddling midnight stays one `ls`."""
        from .clock import iso_cutoff, iso_now
        today = iso_now()[:10]
        yesterday = iso_cutoff(days=1)[:10]
        for day in (today, yesterday):
            d = os.path.join(self._payload_root(), day, chain_id)
            if os.path.isdir(d):
                return d, day
        d = os.path.join(self._payload_root(), today, chain_id)
        os.makedirs(d, exist_ok=True)
        return d, today

    # Gate-config TTL (seconds). Per-round/per-prompt kinds put the gate
    # lookup on every agent round and every user prompt (judge), so the
    # config read is TTL-cached (the LAFEngine.config pattern — performance
    # charter, docs/TRACE-MODES-DESIGN.md). set_interaction_active
    # invalidates on a `trace_recording` flip, so "entering debug" still
    # bites on the very next write, not a TTL later.
    TRACE_RECORDING_CFG_TTL_S = 60.0

    def _trace_recording_config(self):
        """Active `trace_recording` config dict, TTL-cached."""
        now = time.monotonic()
        if (getattr(self, '_trace_rec_cfg', None) is not None
                and now - getattr(self, '_trace_rec_cfg_ts', 0.0)
                < self.TRACE_RECORDING_CFG_TTL_S):
            return self._trace_rec_cfg
        self._trace_rec_cfg = self.get_interaction_config('trace_recording')
        self._trace_rec_cfg_ts = now
        return self._trace_rec_cfg

    def invalidate_trace_recording_cache(self):
        """Drop the cached gate config — called by set_interaction_active
        when the `trace_recording` pointer flips."""
        self._trace_rec_cfg = None

    def _payload_kind_enabled(self, kind, chain_id):
        """Gate resolution for one payload kind. Effective policy = the
        contract's NORMAL defaults (complete by construction over
        PAYLOAD_KIND_EXT) overlaid with the active `trace_recording` config —
        so a kind added to the contract AFTER a brain's config was seeded
        still resolves (to its contract default) instead of silently never
        recording; the config-missing gap is loud-logged (rate-limited by
        _log_error's fingerprint dedup, so a wired call site reminds
        periodically rather than spamming or going quiet forever)."""
        from .trace_contract import TRACE_RECORDING_NORMAL
        cfg = self._trace_recording_config()
        cfg_kinds = cfg.get('kinds')
        effective = dict(TRACE_RECORDING_NORMAL['kinds'])
        if isinstance(cfg_kinds, dict):
            if kind not in cfg_kinds:
                self._log_error(
                    'record_payload_config_missing_kind',
                    ValueError('kind %r absent from active trace_recording '
                               'config — using contract default' % (kind,)),
                    context='chain=%s' % chain_id)
            effective.update(cfg_kinds)
        elif cfg:
            # Config exists but `kinds` is malformed — degrading to the
            # contract defaults must be loud, not the silent branch in an
            # otherwise loud function ("the interesting run is always the
            # one you weren't capturing").
            self._log_error(
                'record_payload_malformed_config',
                ValueError('trace_recording config has non-dict kinds: %r'
                           % (cfg_kinds,)),
                context='chain=%s' % chain_id)
        return bool(effective.get(kind))

    def record_payload(self, chain_id, kind, content, *, seq=None):
        """Write one payload file for a chain; return its pointer (path
        RELATIVE to db_dir) or None (gated off / unknown kind / empty /
        write failed — failures loud-log, never raise into the caller).

        The ONE capture writer — call sites hold zero knowledge of gates,
        paths, or formats. Gating: the `trace_recording` K-store interaction
        (per-kind on/off; modes are named config versions — see
        TRACE_RECORDING_NORMAL/DEBUG in trace_contract.py). Performance
        charter: compact JSON (no indent — the C json encoder holds the GIL;
        `jq .` the file instead), and this is never called on the recall hot
        path.

        Chain dirs are APPEND-ONLY: files open with O_EXCL and collisions get
        an attempt ordinal (`000-prompt.2.md`) — a Scribe idle-tail retry
        reuses the same chain_id, and overwriting would destroy exactly the
        failed attempt's forensics. The retention pass is the only deleter.
        chain_id is sanitized into a path segment: the writer must be as
        traversal-proof as read_payload's guard, or a chain containing '/'
        creates dirs outside payloads/ that the pruner never sees.
        """
        from .trace_contract import PAYLOAD_KIND_EXT
        pointer = None
        with self.loud('record_payload',
                       'chain=%s kind=%s' % (chain_id, kind)):
            if not chain_id or content in (None, '', {}, []):
                return None
            ext = PAYLOAD_KIND_EXT.get(kind)
            if ext is None:
                # A typo'd kind at a wired call site must be visible —
                # _log_error's fingerprint rate-limit dedups repeats.
                # That dedup RELIES on the fingerprint being
                # source:type:message[:100] and EXCLUDING context (which
                # carries the per-chain id) — folding context in would
                # defeat it and log one row per payload write.
                self._log_error('record_payload_unknown_kind',
                                ValueError('unknown payload kind %r'
                                           % (kind,)),
                                context='chain=%s' % chain_id)
                return None
            if not self._payload_kind_enabled(kind, chain_id):
                return None
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False,
                                     default=str)
            safe_chain = re.sub(r'[^A-Za-z0-9._-]', '_', str(chain_id))
            if not re.search(r'[A-Za-z0-9]', safe_chain):
                # Dot/punctuation-only names ('.', '..') are path syntax,
                # not names — they'd land files outside the date layout
                # where the retention pass never looks.
                safe_chain = 'chain-' + (safe_chain.replace('.', '_')
                                         or 'empty')
            chain_dir, day = self._payload_chain_dir(safe_chain)
            base = '%03d-%s' % (int(seq or 0), kind)
            for attempt in range(1, 100):
                name = ('%s.%s' % (base, ext) if attempt == 1
                        else '%s.%d.%s' % (base, attempt, ext))
                path = os.path.join(chain_dir, name)
                try:
                    fd = os.open(path,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                except FileExistsError:
                    continue
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                pointer = os.path.join('payloads', day, safe_chain, name)
                return pointer
            raise RuntimeError('100 attempt ordinals exhausted for %s' % base)
        return pointer  # loud() swallowed a failure → None

    def round_recorder(self, chain_id, seq_base=0):
        """Build the per-round capture closure for run_llm_loop's
        `record_round_fn` — the runner has no brain by design, so each
        caller hands it this closure over (brain, chain_id). The layer owns
        the payload shape (build_round_payload) and the gate; with the
        `round_payload` kind off (the normal config) the per-round cost is
        one TTL-cached gate lookup. Never raises into the caller's loop.

        `seq_base` — file-seq offset for multi-batch encoders that run
        several run_llm_loop calls on ONE run chain (consolidation,
        community): without it every batch's round 0 collides on
        000-round_payload.json and batch identity dies in an attempt
        ordinal. Pass batch_num*100 (rounds are single digits); the
        payload's `round` field stays the in-batch index."""
        from .trace_contract import build_round_payload

        def _record(round_idx, parts):
            try:
                self.record_payload(
                    chain_id, 'round_payload',
                    build_round_payload(label=chain_id, round_idx=round_idx,
                                        seq=seq_base + round_idx, **parts),
                    seq=seq_base + round_idx)
            except Exception as e:
                self._log_error('round_recorder', e,
                                context='chain=%s round=%s'
                                        % (chain_id, round_idx))
        return _record

    def record_failed_run(self, chain_id, error):
        """Record the `failed_run` payload for a dead agent run — the full
        conversation at failure time. The LAYER owns the payload shape
        ({'error', 'messages'}) and its cap; consumers hand over the raw
        exception (RunLoopError carries `.msgs`, already bounded by
        tool_result_cap). Round-0 / unwrapped failures have no msgs and
        record nothing — the prompt kind is that half of the story.
        Returns the pointer or None."""
        from .trace_contract import FAILED_RUN_ERROR_CAP
        msgs = getattr(error, 'msgs', None)
        if not msgs:
            return None
        return self.record_payload(
            chain_id, 'failed_run',
            {'error': str(error)[:FAILED_RUN_ERROR_CAP], 'messages': msgs})

    def prune_payloads_if_due(self, now=None):
        """Age-prune {db_dir}/payloads/ date dirs older than retention_days
        (from the trace_recording config). Self-gated — deliberately runs
        AHEAD of the S2 fire conditions (a keyless brain still prunes): an
        in-memory hourly throttle keeps the per-poll cost at ~zero and the
        `s2_payload_prune_last_ts` brain_meta stamp enforces once-per-day.
        The stamp is written only after a FULLY successful prune — both
        exceptions and partially-failed removals retry within the hour,
        loudly, instead of silently waiting a day. Wall-clock deliberately
        (system bookkeeping, exempt from conversation-time). Never raises;
        returns date-dirs actually removed — failed removals loud-log and
        don't count."""
        import shutil
        import time as _time
        from datetime import datetime, timezone
        from .clock import iso_cutoff
        from .trace_contract import TRACE_RECORDING_NORMAL
        try:
            now = now if now is not None else _time.time()
            if now - getattr(self, '_payload_prune_checked', 0) < 3_600:
                return 0
            self._payload_prune_checked = now
            last = float(self.get_config('s2_payload_prune_last_ts') or 0)
            if now - last < 86_400:
                return 0
            days = self.get_interaction_config(
                'trace_recording').get('retention_days')
            if days is None:
                days = TRACE_RECORDING_NORMAL['retention_days']
            days = int(days)
            if days < 0:
                # A negative value would compute a FUTURE cutoff and delete
                # today's dirs out from under live runs — refuse loudly.
                self._log_error('payload_prune_bad_retention',
                                ValueError('retention_days=%d — using '
                                           'default' % days))
                days = TRACE_RECORDING_NORMAL['retention_days']
            # days=0 legitimately means "keep only today": cutoff is today,
            # and the strict `<` below never touches today's dir.
            cutoff = iso_cutoff(
                days=days,
                at=datetime.fromtimestamp(now, tz=timezone.utc))[:10]
            root = self._payload_root()
            removed, failed = 0, []
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if not (re.fullmatch(r'\d{4}-\d{2}-\d{2}', name)
                            and name < cutoff):
                        continue
                    target = os.path.join(root, name)
                    shutil.rmtree(target, ignore_errors=True)
                    if os.path.exists(target):
                        failed.append(name)   # counter must not lie
                    else:
                        removed += 1
            if failed:
                # No stamp on partial failure: the undeletable dirs retry
                # within the hour (loudly, fingerprint-capped) instead of
                # silently waiting a day.
                self._log_error(
                    'payload_prune_failed_dirs',
                    RuntimeError('could not remove: %s' % ', '.join(failed)))
            else:
                self.set_config('s2_payload_prune_last_ts', str(now))
            return removed
        except Exception as e:
            self._log_error('payload_prune', e, context='')
            return 0

    def read_payload(self, pointer):
        """Read a payload by its relative pointer → str, or None (pruned /
        missing / never recorded). The pointer must stay inside db_dir —
        absolute paths and traversal are rejected (pointers come from trace
        metadata, which is data, not a path authority). Only a MISSING file
        is the silent 'pruned' answer; I/O failures (permissions, EIO) loud-
        log — 'pruned' must never mask an outage sitting on intact files.
        Bytes that don't decode (a crash mid-write splitting a multibyte
        char) come back with replacement chars rather than raising —
        degraded forensics beat none."""
        if not pointer or not isinstance(pointer, str):
            return None
        norm = os.path.normpath(pointer)
        if (os.path.isabs(norm) or norm.startswith('..')
                or not norm.startswith('payloads' + os.sep)):
            return None
        path = os.path.join(os.path.dirname(os.path.abspath(self.db_path)),
                            norm)
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            self._log_error('read_payload', e, context=norm)
            return None


# ═══════════════════════════════════════════════════════════════
# JSONL Conversation Log Support (pre-trace history)
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_conv_dir():
    """Locate the conversations/ dir under the brain repo root (or '' if
    absent). __file__ is servers/brain_traces.py → repo root is TWO dirname
    hops up. Path depth is load-bearing: a wrong hop count silently kills the
    JSONL fallback (empty dir, no error). Cached once per process
    (cache_clear() resets — e.g. a test pointing at a temp dir)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, 'conversations')
    return candidate if os.path.isdir(candidate) else ''


def _from_jsonl(timestamp, before, after):
    """Get conversation from JSONL log files."""
    conv_dir = _get_conv_dir()
    if not conv_dir:
        return []

    target_file = _find_conversation_file(conv_dir, timestamp)
    if not target_file:
        return []

    return _extract_window(target_file, timestamp, before, after)


def _find_conversation_file(conv_dir, timestamp):
    """Find which JSONL file covers a timestamp.

    Checks files by their internal message timestamps, not just filenames.
    Caches time ranges to avoid re-scanning.
    """
    target_date = timestamp[:10]
    best_file = None
    best_distance = float('inf')

    for fname in os.listdir(conv_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(conv_dir, fname)

        first_ts, last_ts = _get_file_time_range(path)
        if not first_ts or not last_ts:
            continue

        # Check if target falls within this file's range
        if first_ts[:10] <= target_date <= last_ts[:10]:
            return path  # Exact match

        # Track closest file for near-misses
        if first_ts[:10] <= target_date:
            distance = ord(target_date[9]) - ord(last_ts[9]) if last_ts else 99
            if distance < best_distance:
                best_distance = distance
                best_file = path

    return best_file


def _get_file_time_range(path):
    """Get first and last message timestamps from a JSONL file."""
    first_ts = None
    last_ts = None

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i > 50:
                    break
                try:
                    obj = json.loads(line.strip())
                    if obj.get('type') in ('user', 'assistant', 'human'):
                        ts = obj.get('timestamp', '')
                        if ts and not first_ts:
                            first_ts = ts
                except (json.JSONDecodeError, KeyError):
                    pass

        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20480))
            tail = f.read().decode('utf-8', errors='ignore')
            for line in reversed(tail.split('\n')):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get('type') in ('user', 'assistant', 'human'):
                        ts = obj.get('timestamp', '')
                        if ts:
                            last_ts = ts
                            break
                except (json.JSONDecodeError, KeyError):
                    pass
    except (IOError, OSError):
        pass

    return first_ts, last_ts


def _extract_window(path, timestamp, before, after):
    """Extract conversation messages around a timestamp from JSONL file."""
    messages = []

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get('type') not in ('user', 'assistant', 'human'):
                        continue

                    ts = obj.get('timestamp', '')
                    if not ts:
                        continue

                    role = 'user' if obj['type'] in ('user', 'human') else 'assistant'

                    msg = obj.get('message', {})
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                    else:
                        content = obj.get('content', '')

                    if isinstance(content, list):
                        texts = [p.get('text', '') for p in content
                                 if isinstance(p, dict) and p.get('type') == 'text']
                        content = ' '.join(texts)

                    if not content or len(content.strip()) < 2:
                        continue

                    messages.append({
                        'role': role,
                        'content': content[:500],
                        'timestamp': ts,
                    })

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except (IOError, OSError):
        return []

    if not messages:
        return []

    # Find message closest to target timestamp
    timestamps = [m['timestamp'] for m in messages]
    idx = bisect_left(timestamps, timestamp)
    idx = min(idx, len(messages) - 1)

    # Window: before × 2 and after × 2 (user + assistant = 2 per exchange)
    start = max(0, idx - before * 2)
    end = min(len(messages), idx + after * 2 + 1)

    return messages[start:end]
