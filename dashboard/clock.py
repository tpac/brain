"""ISO-format timestamp helpers for the dashboard.

Mirrors servers/clock.py semantics: brain stores ISO-T with the
trailing +00:00 zone marker, NOT SQLite's native space-separated
wall-clock string. Time-window queries must use this format or lex
comparison silently breaks at midnight.
"""

from datetime import datetime, timezone, timedelta


def utc_cutoff(hours: float = 0, minutes: float = 0, days: float = 0) -> str:
    """Return an ISO cutoff timestamp compatible with stored created_at values."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours, minutes=minutes, days=days)
    return dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')


def iso_window_around(ts: str, minutes: int) -> tuple:
    """Return (lo, hi) ISO strings that bracket `ts` by ±`minutes`.

    Replaces the hand-rolled string-slicing in encoding.py and s2_runs.py
    (the latter clamped by hour, not minute — this function accepts any
    positive int, so both sites consolidate here). That old code did:

        ts_clean = ts.replace('+00:00', '').replace('Z', '').split('.')[0]
        ts_lo = ts_clean[:10] + 'T' + ts_clean[11:13] + ':' + '%02d' % max(0, int(ts_clean[14:16]) - 2) + ':00'
        ts_hi = ts_clean[:10] + 'T' + ts_clean[11:13] + ':' + '%02d' % min(59, int(ts_clean[14:16]) + 2) + ':59'

    Two bugs in that pattern, both silently. **Hour rollover**: an event at
    08:01 minus 2 min produces "08:-1:00" → clamped to "08:00:00" — loses the
    real window into the previous hour. **Midnight rollover**: clamps minute
    only, never decrements the hour or day. Worst case: an event at 00:01
    queried for ±5 min returns rows from 00:00 of the SAME day only,
    missing the 5 minutes from the prior day.

    This implementation parses the timestamp into a real `datetime`, does
    arithmetic, and re-formats. Same return shape as before so callers
    don't change.

    Returns two ISO-T strings WITHOUT the +00:00 suffix — that matches the
    callers' `BETWEEN ? AND ?` shape against `created_at`. SQLite's
    string comparison handles both stored formats because the prefix is
    identical up to the first non-numeric char.
    """
    # Strip the timezone suffix and any sub-second precision, then parse.
    clean = ts.replace('+00:00', '').replace('Z', '').split('.')[0]
    try:
        dt = datetime.strptime(clean, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        # Malformed input — return the original ts in both slots so a
        # BETWEEN query degenerates to "equals exactly this moment" and
        # returns nothing rather than a wide accidental sweep.
        return (ts, ts)
    lo = dt - timedelta(minutes=minutes)
    hi = dt + timedelta(minutes=minutes)
    fmt = '%Y-%m-%dT%H:%M:%S'
    return (lo.strftime(fmt), hi.strftime(fmt))
