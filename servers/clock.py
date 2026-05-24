"""Single source of truth for "now" across S1/S2/encoder/scouts.

Why this file exists
--------------------
Bug surfaced 2026-05-11 during temporal-eval analysis: build_muster_context()
defaulted current_date to _dt.date.today() when not supplied. In eval replays
(encoder running over a historical conversation dated 2023-03-19), the
temporal scout resolved "today/yesterday/last Tuesday" against the wall-clock
(2026-05-11) instead of the conversation date. Trace evidence on
gpt4_b0863698: candidate_handles included 2026-05-11, 2026-03-05, 2026-05-10
— all relative phrases mis-anchored to NOW.

The architectural fix: ONE function for "now". Callers state which "now" they
want — wall-clock (operator's TZ) or conversation-anchored. Eval inherits the
fix because every code path uses the same helper.

See brain memory dcb5b951 for the architecture decision and 6d5b789e for the
bug being fixed.

Resolution priority (brain_now)
-------------------------------
  1. Explicit `tz` parameter (callers that know better)
  2. Brain config 'operator_tz' (when wired through userConfig)
  3. Host wall-clock (the daemon runs on the operator's machine)
  4. UTC fallback — logged loudly so we know if we got here

UTC-internal is the better long-term architecture (storage UTC, render in
operator TZ). Deferred for now — see BACKLOG.md "UTC-internal clock refactor".

Contract
--------
DO NOT call `datetime.now()`, `date.today()`, or `time.time()` directly in
S1/S2 code or scouts. Use brain_now() / conversation_now(). A contract test
(tests/test_clock_contract_sync.py) scans for direct calls and fails.
Exempt: telemetry/perf timers (use time.monotonic explicitly), schema files.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
import sys as _sys
from typing import Any, Iterable, Optional


# ─── Public API ─────────────────────────────────────────────────────────


def brain_now(brain=None, tz: Optional[str] = None) -> _dt.datetime:
    """Return the current 'now' in the operator's frame as a timezone-aware
    datetime.

    Resolution priority:
      1. `tz` parameter (e.g. "America/New_York")
      2. brain.config.get('operator_tz') if brain is provided
      3. Host system TZ (daemon runs on operator's machine, so this IS
         operator TZ in practice)
      4. UTC fallback with stderr warning

    Callers should pass `brain` when they have one — lets brain config drive
    TZ when the userConfig wires that in later.
    """
    target_tz = _resolve_tz(brain=brain, tz=tz)
    if target_tz is None:
        # Final UTC fallback — log so we know.
        print('[clock] WARN brain_now falling back to UTC (no host TZ, no '
              'operator_tz, no explicit tz)', file=_sys.stderr)
        return _dt.datetime.now(_dt.timezone.utc)
    return _dt.datetime.now(target_tz)


def brain_today(brain=None, tz: Optional[str] = None) -> _dt.date:
    """Return today's date in the operator's frame. Convenience wrapper."""
    return brain_now(brain=brain, tz=tz).date()


def iso_cutoff(hours: float = 0, minutes: float = 0, days: float = 0) -> str:
    """Return an ISO-format UTC cutoff timestamp suitable for SQL WHERE clauses.

    Use bound as a parameter against ISO-T-formatted timestamp columns:

        cur.execute(
            "SELECT ... WHERE created_at > ?",
            (iso_cutoff(hours=24),))

    Why this exists
    ---------------
    SQLite's ``datetime('now', '-N hours')`` returns a space-separated format
    without microseconds or timezone (e.g. ``'2026-05-24 17:07:13'``). Brain
    stores ISO with a 'T' separator (e.g. ``'2026-05-24T17:07:13.123456+00:00'``
    or with a 'Z' suffix). Lexicographic comparison breaks because
    ``'T' (0x54) > ' ' (0x20)`` — same-day-earlier rows incorrectly look
    "greater than" a same-day cutoff. Replace any
    ``WHERE col > datetime('now', '-N units')`` in SQL with a bound
    ``iso_cutoff(...)`` to compare apples to apples.

    Two storage formats coexist (latent, currently safe)
    ----------------------------------------------------
    The codebase has two "ISO" suffixes in flight:

      - ``Brain.now()``        → ``'…T…ffffffZ'``   (nodes.created_at,
                                                     nodes.last_accessed)
      - ``TraceDAL`` inserts   → ``'…T…ffffff+00:00'`` (trace_events.created_at,
                                                       debug_log.created_at,
                                                       hook_errors.created_at)

    They are NOT lex-equivalent — ``'Z' (0x5A) > '+' (0x2B)`` — so a Z-stored
    string is "greater than" a +00:00 string at the same instant.

    **Invariant: queries stay within a single column.** Every WHERE-clause
    timestamp comparison in the codebase reads from ONE column with ONE
    consistent format. ``iso_cutoff`` returns ``+00:00`` form; both stored
    formats compare correctly against it whenever the date/time/microsecond
    prefix differs (i.e. effectively always — only an exact-microsecond
    match would expose the suffix mismatch). If you ever need to compare
    timestamps across columns with different suffix conventions, normalize
    first — don't rely on raw string comparison.

    Args:
        hours, minutes, days: subtracted from current UTC time. All optional.

    Returns:
        ISO-8601 string in UTC, ``'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'``.
    """
    dt = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        hours=hours, minutes=minutes, days=days)
    return dt.isoformat()


def conversation_now(messages: Optional[Iterable[Any]] = None,
                      session_started_at: Optional[Any] = None,
                      brain=None,
                      tz: Optional[str] = None) -> _dt.datetime:
    """Return the 'now' this CONVERSATION thinks it's happening.

    Used by encoder/scout for resolving relative dates ("today", "yesterday",
    "last Tuesday") in the conversation under encoding. Always returns a
    timezone-aware datetime.

    Resolution priority:
      1. `[Current date: ...]` prefix in the most recent user message
         (eval replay injects this — see eval/longmem/replay.py)
      2. `session_started_at` — typically the SessionContext's start
         timestamp or the haystack's session date
      3. brain_now() — operator wall-clock (the production path; the
         conversation is happening NOW)

    Args:
        messages: iterable of message dicts {role, content, ...} or strings;
                  the function looks for the eval's date-prefix marker.
        session_started_at: ISO timestamp string OR datetime; the session's
                  notional start time.
        brain: for TZ resolution (see brain_now).
        tz: explicit TZ override.

    Never raises — always returns a valid datetime.
    """
    # 1. Eval replay prefix on last user message
    prefix_date = _extract_replay_date_prefix(messages)
    if prefix_date is not None:
        return prefix_date

    # 2. Session start
    if session_started_at is not None:
        parsed = _parse_to_datetime(session_started_at, tz=tz, brain=brain)
        if parsed is not None:
            return parsed

    # 3. Wall-clock fallback (production path)
    return brain_now(brain=brain, tz=tz)


def conversation_today(messages: Optional[Iterable[Any]] = None,
                        session_started_at: Optional[Any] = None,
                        brain=None,
                        tz: Optional[str] = None) -> _dt.date:
    """Date form of conversation_now. Same resolution priority."""
    return conversation_now(messages=messages,
                             session_started_at=session_started_at,
                             brain=brain, tz=tz).date()


# ─── Internals ──────────────────────────────────────────────────────────


# Eval replay prepends "[Current date: 2023/03/19 (Sun) 07:16]" or
# "[Current date: 2023-03-19]" to user messages. Match either shape.
_REPLAY_DATE_RE = _re.compile(
    r'\[Current date:\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})'
    r'(?:\s+\([^)]*\))?'              # optional "(Sun)" weekday
    r'(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?'  # optional HH:MM[:SS]
    r'\s*\]',
    _re.IGNORECASE,
)


def _extract_replay_date_prefix(messages) -> Optional[_dt.datetime]:
    """If the most recent user message starts with '[Current date: ...]',
    parse and return that datetime (UTC-aware). Else None.

    Eval replay injects this prefix — see eval/longmem/replay.py:109.
    """
    if not messages:
        return None
    # Find last user message; if `messages` is non-list, snapshot to list.
    try:
        msg_list = list(messages)
    except TypeError:
        return None

    for m in reversed(msg_list):
        if isinstance(m, dict):
            if m.get('role') != 'user':
                continue
            content = m.get('content') or m.get('text') or ''
        elif isinstance(m, str):
            content = m
        else:
            continue
        if not content:
            continue
        match = _REPLAY_DATE_RE.search(content)
        if not match:
            # Only inspect the LAST user message — eval prepends per turn
            # but we want the most recent one, which should be sufficient.
            return None
        date_str = match.group(1).replace('/', '-')
        time_str = match.group(2) or '00:00:00'
        try:
            # Pad single-digit components for fromisoformat
            y, mo, d = date_str.split('-')
            iso = '%s-%s-%sT%s' % (y, mo.zfill(2), d.zfill(2),
                                     time_str if len(time_str.split(':')) >= 2 else (time_str + ':00:00'))
            # Ensure HH:MM:SS
            parts = time_str.split(':')
            if len(parts) == 2:
                time_str = time_str + ':00'
            iso = '%s-%s-%sT%s' % (y, mo.zfill(2), d.zfill(2), time_str)
            dt = _dt.datetime.fromisoformat(iso)
            # Assume operator TZ for the conversation timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz() or _dt.timezone.utc)
            return dt
        except (ValueError, IndexError):
            return None
    return None


def _parse_to_datetime(val: Any, tz: Optional[str] = None,
                        brain=None) -> Optional[_dt.datetime]:
    """Parse a value (str / datetime / date / int epoch) to a tz-aware
    datetime. Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, _dt.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                or _dt.timezone.utc)
        return val
    if isinstance(val, _dt.date):
        target = _resolve_tz(brain=brain, tz=tz) or _dt.timezone.utc
        return _dt.datetime.combine(val, _dt.time(0, 0), tzinfo=target)
    if isinstance(val, (int, float)):
        return _dt.datetime.fromtimestamp(val, tz=_dt.timezone.utc)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        # Try ISO first
        try:
            dt = _dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                  or _dt.timezone.utc)
            return dt
        except ValueError:
            pass
        # Fall back to dateutil if available
        try:
            from dateutil import parser as _dp
            dt = _dp.parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                  or _dt.timezone.utc)
            return dt
        except Exception:
            return None
    return None


def _resolve_tz(brain=None, tz: Optional[str] = None) -> Optional[_dt.tzinfo]:
    """Resolve a tzinfo per the priority chain:
       1. Explicit tz string  2. brain.config['operator_tz']
       3. Host system TZ      4. None (caller falls back to UTC)
    """
    candidates = [tz]
    if brain is not None:
        try:
            cfg = getattr(brain, 'config', None)
            if cfg is not None:
                if hasattr(cfg, 'get'):
                    candidates.append(cfg.get('operator_tz'))
                else:
                    candidates.append(getattr(cfg, 'operator_tz', None))
        except Exception:
            pass

    for c in candidates:
        if not c:
            continue
        try:
            from zoneinfo import ZoneInfo  # 3.9+
            return ZoneInfo(c)
        except Exception:
            continue

    # Host TZ — astimezone(None) attaches the local one.
    try:
        local = _dt.datetime.now().astimezone().tzinfo
        if local is not None:
            return local
    except Exception:
        pass
    return None


__all__ = ['brain_now', 'brain_today', 'conversation_now', 'conversation_today']
