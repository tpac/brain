"""SessionContext — per-request session identity.

The brain is a singleton. Sessions are not. SessionContext flows with every
brain call — hooks, MCP, encoding. The brain serves requests tagged with
context, like a database server.

SessionContext carries:
- session_id: identity (from Claude Code hook args)
- stop_counter: which stop we're on
- fatigue: {node_id: access_count} for synaptic fatigue (resets between sessions)

Usage:
    # In a hook:
    ctx = SessionContext.from_hook_args(args)
    chain = ctx.s0_chain()  # 's0-{session_short}-{stop}'

    # Persist across daemon restarts:
    ctx.save(conn)
    ctx = SessionContext.load(conn, session_id)
"""
import json
import uuid
import sqlite3
from typing import Dict
from datetime import datetime, timezone


class SessionContext:
    """Per-session state that flows with every brain call."""

    def __init__(self, session_id: str = '', stop_counter: int = 0):
        self.session_id = session_id or uuid.uuid4().hex
        self.stop_counter = stop_counter
        self.fatigue: Dict[str, int] = {}  # {node_id: access_count} — resets between sessions
        self.edge_fatigue: Dict[str, int] = {}  # {target_node_id: surface_count} — edge rotation

    @classmethod
    def from_hook_args(cls, args: dict) -> 'SessionContext':
        """Create from Claude Code hook JSON input.

        Every hook receives session_id in its stdin JSON.
        If missing, generates a fallback UUID.
        """
        session_id = args.get('session_id', '') or ''
        return cls(session_id=session_id)

    @property
    def session_short(self) -> str:
        """First 8 chars of session_id — used in chain IDs."""
        return self.session_id[:8]

    def increment_stop(self):
        """Increment stop counter. Called by the Stop hook."""
        self.stop_counter += 1

    # ── Chain ID generators ──

    def s0_chain(self) -> str:
        """S0 chain: one per stop — messages + tools."""
        return 's0-%s-%d' % (self.session_short, self.stop_counter)

    def s1r_chain(self) -> str:
        """S1 recall chain: recall/surface for this stop."""
        return 's1r-%s-%d' % (self.session_short, self.stop_counter)

    def s1e_chain(self) -> str:
        """S1 encode chain: encoding run triggered at this stop."""
        return 's1e-%s-%d' % (self.session_short, self.stop_counter)

    # ── Frame ──

    def get_frame(self, brain) -> str:
        """Build and return Anchor's structured awareness Frame for this session.

        Phase 2 (2026-05-02): Frame Constructor — markdown text with five
        sections (Operator / Partnership / Active threads / Current focus /
        Recent moves), composed deterministically from existing brain queries
        plus this session's encoder state.

        Brain is passed as a dependency rather than stored on SessionContext —
        SessionContext is a per-request data carrier, brain is the singleton.

        v1: rebuilt fresh on every call (no caching). The slow-changing slots
        (operator/partnership/active-threads come from brain state) and the
        fast-changing slots (current_focus/recent_moves come from session
        state) are already separated at the data-source layer in build_frame —
        a future split into brain-level vs session-level caching slots in
        cleanly without restructuring the renderer.

        See servers/scales/s1/frame.py and docs/FRAME-DESIGN.md.
        """
        from servers.scales.s1.frame import build_frame
        return build_frame(brain, self.session_id)

    # ── Persistence ──

    def increment_fatigue(self, node_id: str) -> int:
        """Increment fatigue for a node. Returns new count."""
        self.fatigue[node_id] = self.fatigue.get(node_id, 0) + 1
        return self.fatigue[node_id]

    def increment_edge_fatigue(self, target_id: str) -> int:
        """Increment edge fatigue for a target node. Returns new count."""
        self.edge_fatigue[target_id] = self.edge_fatigue.get(target_id, 0) + 1
        return self.edge_fatigue[target_id]

    def get_edge_fatigue(self, target_id: str) -> int:
        """Get current edge fatigue count for a target node."""
        return self.edge_fatigue.get(target_id, 0)

    def save(self, conn: sqlite3.Connection):
        """Save session context to DB. Creates or updates.

        Includes fatigue as JSON — single row replaces 52K per-node rows.
        """
        now = datetime.now(timezone.utc).isoformat()
        data = json.dumps({
            'stop_counter': self.stop_counter,
            'fatigue': self.fatigue,
            'edge_fatigue': self.edge_fatigue,
        })
        conn.execute(
            'INSERT OR REPLACE INTO session_state (session_id, key, node_id, value, updated_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (self.session_id, '_session_context', '', data, now))
        conn.commit()

    @classmethod
    def load(cls, conn: sqlite3.Connection, session_id: str) -> 'SessionContext':
        """Load session context from DB. Returns None if not found."""
        row = conn.execute(
            'SELECT value FROM session_state WHERE session_id = ? AND key = ?',
            (session_id, '_session_context')).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row[0])
            ctx = cls(
                session_id=session_id,
                stop_counter=data.get('stop_counter', 0),
            )
            ctx.fatigue = {k: int(v) for k, v in data.get('fatigue', {}).items()}
            ctx.edge_fatigue = {k: int(v) for k, v in data.get('edge_fatigue', {}).items()}
            return ctx
        except (json.JSONDecodeError, TypeError):
            return None
