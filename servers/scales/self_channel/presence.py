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

from servers.scales.self_channel import self_contract


def _focus_line(brain, session_id, max_chars=100):
    """One-line current focus for a stream — the first line of its session arc."""
    arc = (brain.session_context_for(session_id) or '').strip()
    if not arc:
        return ''
    return arc.splitlines()[0].strip()[:max_chars].rstrip()


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
    raw = brain.present_streams(
        exclude_session=my_session_id or '',
        window_min=self_contract.ROSTER_LIVE_WINDOW_MIN,
        limit=cap)
    streams = []
    for r in raw:
        sid = r.get('session_id', '')
        streams.append({
            'session_id': sid,
            'short': sid[:8],
            'focus': _focus_line(brain, sid),
            'updated_at': r.get('updated_at', ''),
        })
    line = self_contract.render_presence(
        [(s['short'], s['focus']) for s in streams])
    return {'streams': streams, 'line': line}


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
