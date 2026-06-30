"""SessionContext — per-request session identity.

The brain is a singleton. Sessions are not. SessionContext flows with every
brain call — hooks, MCP, encoding. The brain serves requests tagged with
context, like a database server.

SessionContext carries:
- session_id: identity (from Claude Code hook args)
- stop_counter: which stop we're on
- fatigue: {node_id: access_count} for synaptic fatigue (resets between sessions)
- activity counters (remember_count, message_count, edit_check_count,
  boot_time) — replaces the brain_meta globals that leaked across
  parallel sessions before 2026-05-17

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


class SessionContextCorrupt(Exception):
    """Raised by SessionContext.load when a session_state row EXISTS but its blob
    won't parse (corrupt) — as opposed to absent, which returns None. The
    brain-holding caller catches this and logs via the canonical
    brain._log_error; SessionContext stays a pure data carrier (no brain ref, no
    DAL reach, no logging of its own)."""


class SessionContext:
    """Per-session state that flows with every brain call."""

    def __init__(self, session_id: str = '', stop_counter: int = 0):
        self.session_id = session_id or uuid.uuid4().hex
        self.stop_counter = stop_counter
        # Turn classification (see trace_contract S0 TURN CLASSIFICATION):
        # last_recall_stop = the stop_counter value at which a real UserPromptSubmit
        # last ran hook_recall. A turn is "conversational" iff recall ran THIS stop
        # (last_recall_stop == stop_counter); a /watch wakeup skips recall
        # client-side, so its stop never updates this → it reads as a heartbeat.
        # -1 = recall has never run for this session.
        self.last_recall_stop: int = -1
        # Integration cadence (every Nth conversational turn → S1 Scribe) is NOT
        # a stored counter anymore — it's derived live from traces via
        # turns_since_last_encode(). A maintained counter desynced across resume
        # (boot reset it while the traces stayed truthful), starving the Scribe.
        # stop_counter stays a real counter: it names chain IDs and must be
        # assigned synchronously, before its trace exists.
        # Transient (not persisted): set by post_response_common each turn so the
        # Stop hook's Scribe gate only fires on conversational turns.
        self.last_turn_conversational: bool = True
        self.fatigue: Dict[str, int] = {}  # {node_id: access_count} — resets between sessions
        self.edge_fatigue: Dict[str, int] = {}  # {target_node_id: surface_count} — edge rotation
        # Per-turn "Anchor touched" accumulator — the S0 mirror of the encoder's
        # ops-delta. Anchor's own MCP tools append node ids here as they fire
        # (writes via dispatch `affected`, deliberate reads via get_node(s));
        # post_response_common flushes it as one `anchor_touched` S0 delta at the
        # turn boundary, then resets. Transient (not persisted) — a turn's touches
        # are flushed before the next turn. Keyset driven by ANCHOR_TOUCHED_KEYS
        # (the contract) so it can't drift from the builder it feeds via **touched.
        from .trace_contract import ANCHOR_TOUCHED_KEYS
        self.touched: Dict[str, list] = {k: [] for k in ANCHOR_TOUCHED_KEYS}
        # Per-session node activity — the parallel-session replacement for
        # global nodes.{activation, recency_score, last_accessed, access_count}
        # in reads that should be session-scoped (spreading-activation kernel,
        # recency filtering, live-session Frame composition). Global nodes
        # columns stay populated by the drain for S2 maintenance + dashboard
        # analytics. See bump_node_activity() for write semantics.
        # Shape: {node_id: {'activation': float, 'recency_score': float,
        #                   'access_count': int, 'last_accessed': iso_str}}
        self.node_activity: Dict[str, Dict[str, object]] = {}
        # Activity counters — were global brain_meta keys (leaked across
        # parallel sessions). Persisted in session_state alongside fatigue.
        self.remember_count: int = 0
        self.message_count: int = 0
        self.edit_check_count: int = 0
        self.boot_time: str = ''  # ISO timestamp; empty means not booted yet
        # Session env — the Claude-side facts (working dir + git branch) FED IN
        # from the boot hook; the daemon never introspects Claude to learn them.
        # Per-session identity (not activity), surfaced in peek so streams can
        # tell where each other is working. Stable for the session's life.
        self.cwd: str = ''
        self.branch: str = ''
        # The linked worktree this stream works in (CC WorktreeCreate `name`),
        # or '' for the main working tree. Per-session — replaces the global
        # `current_worktree` config that was last-writer-wins across parallel
        # streams (two streams in different worktrees clobbered each other).
        # Stamped at boot (derived from cwd) and refreshed on WorktreeCreate/Remove.
        self.worktree: str = ''
        # Segment / conversation-shift state. Were brain_meta keys
        # (`segment_*_{session_id}`); moved here 2026-05-17 because those
        # writes on the hook_recall hot path were saturating brain.db
        # locks and racing with concurrent writers (the `another row
        # available` SQL contract bug). In-memory, autosave persists.
        self.segment_id: int = 0
        self.segment_embeddings: list = []  # list[str] (base64-encoded vectors)
        self.segment_node_ids: list = []    # list[str] (node ids surfaced this segment)

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

        Frame Constructor — markdown text with three sections (What I've learned
        / Current focus / Recent moves), composed from existing brain queries
        plus this session's encoder state. The wisdom section is relevance-
        ranked mid-session and seeded-influence-sampled at boot (stable within a
        session). No LLM call.

        Brain is passed as a dependency rather than stored on SessionContext —
        SessionContext is a per-request data carrier, brain is the singleton.
        Rebuilt fresh on every call (no caching).

        See servers/scales/s1/frame.py.
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

    def bump_node_activity(self, node_id: str, ts: str) -> Dict[str, object]:
        """Mark a node as accessed by this session at time `ts`.

        Semantics mirror the global `nodes` UPDATE in recall_write_queue.drain:
          access_count += 1
          activation   = min(1.0, activation + 0.1)
          recency_score = 1.0  (reset on access)
          last_accessed = ts   (caller passes wall-clock or conversation-time)

        First access creates the entry with access_count=1, activation=1.0
        (capped at the bump-from-zero ceiling — matches `remember()` initial).
        Returns the updated record.
        """
        if not node_id:
            return {}
        rec = self.node_activity.get(node_id)
        if rec is None:
            rec = {
                'activation': 1.0,
                'recency_score': 1.0,
                'access_count': 1,
                'last_accessed': ts,
            }
        else:
            current = float(rec.get('activation', 0.0))
            rec['activation'] = min(1.0, current + 0.1)
            rec['recency_score'] = 1.0
            rec['access_count'] = int(rec.get('access_count', 0)) + 1
            # Monotonic ts: ISO-8601 strings sort lexicographically.
            existing_ts = rec.get('last_accessed', '')
            if not existing_ts or ts > existing_ts:
                rec['last_accessed'] = ts
        self.node_activity[node_id] = rec
        return rec

    def get_node_activity(self, node_id: str) -> Dict[str, object]:
        """Return this session's activity record for a node, or empty dict."""
        return self.node_activity.get(node_id, {})

    def set_env(self, cwd: str = '', branch: str = '', worktree=None) -> None:
        """Stamp the Claude-side session env — where this stream is working.

        Single mutator for the per-session identity fed in from the boot hook and
        the WorktreeCreate/Remove hooks (the daemon never introspects Claude). The
        per-session replacement for the global cwd/branch/worktree config that was
        last-writer-wins across parallel streams.

        cwd/branch refresh only on a truthy value — falsy (''/None) leaves the
        existing value, so a failed detection never clobbers a known one. worktree
        is three-state: None leaves it unchanged (detection failed — keep what we
        have), '' CLEARS it (main tree, or WorktreeRemove), a name SETS it. That
        None-vs-'' distinction is why detect_git_env returns None on git failure.
        """
        if cwd:
            self.cwd = cwd
        if branch:
            self.branch = branch
        if worktree is not None:
            self.worktree = worktree

    def save(self, conn: sqlite3.Connection):
        """Save session context to DB. Creates or updates.

        Includes fatigue + activity counters as JSON — single row replaces
        52K per-node rows and 6 leaky brain_meta globals.
        """
        data = json.dumps({
            'stop_counter': self.stop_counter,
            'last_recall_stop': self.last_recall_stop,
            'fatigue': self.fatigue,
            'edge_fatigue': self.edge_fatigue,
            'remember_count': self.remember_count,
            'message_count': self.message_count,
            'edit_check_count': self.edit_check_count,
            'boot_time': self.boot_time,
            'cwd': self.cwd,
            'branch': self.branch,
            'worktree': self.worktree,
            'segment_id': self.segment_id,
            'segment_embeddings': self.segment_embeddings,
            'segment_node_ids': self.segment_node_ids,
            'node_activity': self.node_activity,
        })
        # SessionStateDAL is the sole gateway to session_state; wrap the conn
        # we're handed (SessionContext has no Brain ref to reach a held DAL).
        from .dal import SessionStateDAL
        SessionStateDAL(conn).set(self.session_id, '_session_context', data)

    @classmethod
    def load(cls, conn: sqlite3.Connection, session_id: str) -> 'SessionContext':
        """Load session context from DB. Returns None if not found."""
        from .dal import SessionStateDAL
        value = SessionStateDAL(conn).get(session_id, '_session_context')
        if not value:
            return None
        try:
            data = json.loads(value)
            ctx = cls(
                session_id=session_id,
                stop_counter=data.get('stop_counter', 0),
            )
            ctx.last_recall_stop = int(data.get('last_recall_stop', -1))
            ctx.fatigue = {k: int(v) for k, v in data.get('fatigue', {}).items()}
            ctx.edge_fatigue = {k: int(v) for k, v in data.get('edge_fatigue', {}).items()}
            ctx.remember_count = int(data.get('remember_count', 0))
            ctx.message_count = int(data.get('message_count', 0))
            ctx.edit_check_count = int(data.get('edit_check_count', 0))
            ctx.boot_time = data.get('boot_time', '') or ''
            ctx.cwd = data.get('cwd', '') or ''
            ctx.branch = data.get('branch', '') or ''
            ctx.worktree = data.get('worktree', '') or ''
            ctx.segment_id = int(data.get('segment_id', 0))
            ctx.segment_embeddings = list(data.get('segment_embeddings', []) or [])
            ctx.segment_node_ids = list(data.get('segment_node_ids', []) or [])
            raw_activity = data.get('node_activity', {}) or {}
            # Coerce types — JSON round-trip preserves dicts but the inner
            # numerics need conversion when an older session_state row
            # predates this field (raw_activity will be {} then; loop no-op).
            for nid, rec in raw_activity.items():
                if isinstance(rec, dict):
                    ctx.node_activity[nid] = {
                        'activation': float(rec.get('activation', 0.0)),
                        'recency_score': float(rec.get('recency_score', 0.0)),
                        'access_count': int(rec.get('access_count', 0)),
                        'last_accessed': str(rec.get('last_accessed', '') or ''),
                    }
            return ctx
        except (json.JSONDecodeError, TypeError) as e:
            # The row EXISTED but won't parse — corrupt, not absent. Signal it so
            # the brain-holding caller logs via the canonical brain._log_error.
            # (Pure data carrier: no brain ref, no DAL reach, no logging here.)
            raise SessionContextCorrupt(
                'corrupt session_state blob for %s: %s' % ((session_id or '')[:8], e)
            ) from e
