"""Presence — the self-channel awareness layer (PULL, not push).

Who else is awake right now (roster) and where they are (peek). Composition
only — no raw SQL here: the session_state reads live on the brain
(`brain.present_streams`), a stream's focus is `brain.session_context_for`, and
the line is `self_contract.render_presence`.

This is the PULL primitive: I look at my other streams of thought when interest
calls (Tom: "if interest calls you can pull to learn more"). The AUTOMATIC
surfacing of the presence line at boot / hook fire is a separate, eval-gated
step (docs/BOOT-REIGNITION.md) — build it on top of build_presence().
"""

from datetime import datetime, timezone

from servers.scales.self_channel import self_contract


def _age_min(iso_ts):
    """Minutes since `iso_ts` — wall-clock (presence is real-time, exempt from the
    conversation_now rule, like present_streams). Empty/unparseable → a large age
    (treated as lost)."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return 1e9


def _first_line(focus, max_chars=100):
    """One-line roster focus — first line of the focus string present_streams
    hands us (the stream's latest user_message summary). Pure render, no read."""
    s = (focus or '').strip()
    if not s:
        return ''
    return s.splitlines()[0].strip()[:max_chars].rstrip()


def build_presence(brain, my_session_id='', limit=None):
    """Roster of streams of thought awake now, each with a one-line focus.

    Bounded by `limit` (default: PRESENCE_MAX_STREAMS) and the wall-clock window
    — ranked by recency, never enumerated (imagine 20 streams). Returns:

        {'streams': [{'session_id','short','focus','updated_at'}, ...],
         'line': <rendered presence line>}

    `line` is what a future boot/hook surfaces; the structured list is what
    `peek` drills into.
    """
    cap = limit or self_contract.PRESENCE_MAX_STREAMS
    # Fetch out to the lost-grace window so a recently-gone stream is visible, not
    # silently dropped at the live edge; headroom on the limit so lost entries
    # don't crowd out the live cap.
    window = self_contract.ROSTER_LIVE_WINDOW_MIN + self_contract.ROSTER_LOST_GRACE_MIN
    raw = brain.present_streams(
        exclude_session=my_session_id or '',
        window_min=window,
        limit=cap + 5)
    live, lost = [], []
    for r in raw:
        sid = r.get('session_id', '')
        entry = {
            'session_id': sid,
            'short': sid[:8],
            'focus': _first_line(r.get('focus', '')),
            'updated_at': r.get('updated_at', ''),
            'state': self_contract.classify_liveness(_age_min(r.get('updated_at', ''))),
        }
        if entry['state'] == 'lost':
            lost.append(entry)
        elif len(live) < cap:
            live.append(entry)
    line = self_contract.render_presence(
        [(s['short'], s['focus'], s['state']) for s in live],
        lost=[(s['short'], s['focus']) for s in lost])
    return {'streams': live, 'lost': lost, 'line': line}


def peek(brain, session_id):
    """Look into one stream of thought — its full current focus (the session arc).

    The interest-driven pull: 'where is that stream right now?' — read-only, no
    interruption. Returns the full arc (not the one-line roster summary).
    """
    if not session_id:
        return {'session_id': '', 'short': '', 'focus': '', 'found': False}
    arc = (brain.session_context_for(session_id) or '').strip()
    return {'session_id': session_id, 'short': session_id[:8],
            'focus': arc, 'found': bool(arc)}
