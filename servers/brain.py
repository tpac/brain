"""brain Engine v7 — Python Port

Hebbian learning, Ebbinghaus decay, synaptic pruning, spreading activation.

This file contains the Brain class (core infrastructure, constructor, helpers)
and assembles all functionality via mixin inheritance:

  Brain(ConsciousnessMixin, RecallMixin, RememberMixin, ConnectionsMixin,
        EvolutionMixin, EngineeringMixin, DreamsMixin, AssemblyMixin)

Each mixin is in its own file (brain_recall.py, brain_remember.py, etc.)
and was extracted from the original 9000-line monolith.

Architecture: sqlite3 (WAL mode, FK on), fire-and-forget async embeddings,
TF-IDF semantic scoring, intent detection, temporal awareness.
"""

import sys
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
import sqlite3
import math
import uuid
import json
import time
import os
import struct
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from .schema import ensure_schema, ensure_logs_schema, migrate_logs_to_separate_db
from .dal import LogsDAL, MetaDAL
from .brain_consciousness import ConsciousnessMixin
from .brain_recall import BrainRecallMixin
from .brain_remember import BrainRememberMixin
from .brain_connections import BrainConnectionsMixin
from .brain_evolution import BrainEvolutionMixin
from .brain_engineering import BrainEngineeringMixin
from .brain_dreams import BrainDreamsMixin
from .brain_assembly import BrainAssemblyMixin
from . import embedder


from .brain_constants import (
    DECAY_HALF_LIFE,
    INTENT_PATTERNS, INTENT_TYPE_BOOSTS, TEMPORAL_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════
# BRAIN CLASS
# ═══════════════════════════════════════════════════════════════

class Brain(
    ConsciousnessMixin,
    BrainRecallMixin,
    BrainRememberMixin,
    BrainConnectionsMixin,
    BrainEvolutionMixin,
    BrainEngineeringMixin,
    BrainDreamsMixin,
    BrainAssemblyMixin,
):
    """
    Core brain engine.

    Manages:
    - Node storage (memories, thoughts, rules, etc.)
    - Edge connections (Hebbian learning)
    - Semantic recall (TF-IDF + embeddings)
    - Intent detection and temporal filtering
    - Session activity tracking

    Singleton pattern: Use Brain.get_instance(db_path) to reuse an existing
    warm Brain for the same db_path. Direct __init__ always creates a new instance
    (useful for tests, simulations, and fresh brains).
    """

    # ─── Singleton registry ───
    _instances: Dict[str, 'Brain'] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: str) -> 'Brain':
        """
        Get or create a singleton Brain for the given db_path.

        Returns an existing warm instance if one is already open for this path,
        avoiding repeated schema checks, TF-IDF rebuilds, and embedder loads.
        Thread-safe.

        Args:
            db_path: Path to brain.db file

        Returns:
            Brain instance (cached or newly created)
        """
        canonical = os.path.realpath(db_path)
        with cls._lock:
            instance = cls._instances.get(canonical)
            if instance is not None:
                # Verify the connection is still alive
                try:
                    instance.conn.execute('SELECT 1')
                    return instance
                except Exception:
                    # Connection died — remove stale entry and recreate
                    del cls._instances[canonical]

            instance = cls(db_path)
            cls._instances[canonical] = instance
            return instance

    @classmethod
    def clear_instances(cls):
        """
        Close and remove all cached Brain instances.
        Useful for test teardown or when switching brain files.
        """
        with cls._lock:
            for path, instance in list(cls._instances.items()):
                try:
                    instance.conn.commit()
                    instance.conn.close()
                except Exception:
                    pass
            cls._instances.clear()

    def __init__(self, db_path: str, skip_embedder: bool = False):
        """
        Initialize Brain with SQLite3 database.

        NOTE: For production use, prefer Brain.get_instance(db_path) which
        reuses warm instances. Direct __init__ always creates a fresh connection
        (appropriate for tests, simulations, and temporary brains).

        Args:
            db_path: Path to brain.db file
            skip_embedder: If True, skip loading the embedding model (~1GB).
                          Useful for tests that don't need semantic search.
        """
        self.db_path = db_path
        self._skip_embedder = skip_embedder

        # Open SQLite connection with WAL mode for concurrency
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA busy_timeout = 30000')
        self.conn.execute('PRAGMA foreign_keys=ON')

        # Create schema if needed
        ensure_schema(self.conn, db_path=db_path)

        # Open separate logs database (brain_logs.db)
        db_dir = os.path.dirname(db_path) or '.'
        self.logs_db_path = os.path.join(db_dir, 'brain_logs.db')
        self.logs_conn = sqlite3.connect(self.logs_db_path, check_same_thread=False)
        self.logs_conn.execute("PRAGMA journal_mode=WAL")
        self.logs_conn.execute("PRAGMA busy_timeout = 30000")
        ensure_logs_schema(self.logs_conn)

        # One-time migration: move log tables from brain.db to brain_logs.db
        migrate_logs_to_separate_db(self.conn, self.logs_conn)

        # DAL instances — incremental adoption, brain.py migrates one method at a time
        self._meta = MetaDAL(self.conn)
        self._logs_dal = LogsDAL(self.logs_conn)
        from .dal import TraceDAL, InteractionDAL
        self._trace_dal = TraceDAL(self.logs_conn)
        self._interaction_dal = InteractionDAL(self.logs_conn)

        # Init rate limiter for error logging (DDoS protection)
        self._init_rate_limiter()

        # Init human-readable log file (brain.log with rotation)
        self._init_file_logger(db_dir)

        # Post-schema initialization (TF-IDF rebuild if needed)
        self._post_schema_init()

        # Seed interactions if empty (first boot or cleared)
        try:
            from .interaction_seed import seed_interactions
            seed_interactions(self)
        except Exception as _e:
            print('[brain] WARNING: interaction seed failed: %s' % _e, flush=True)

        # v5: Session state accumulator for synthesis
        self._session_state = {
            'decisions': [], 'corrections': [], 'inflections': [],
            'model_updates': [], 'validations': [], 'open_questions': []
        }

        # Load embedder with config from brain_meta (falls back to plugin.json defaults)
        if not skip_embedder:
            try:
                embedder_config = self._get_embedder_config()
                embedder.load_model(embedder_config)
            except Exception as e:
                print(f'[brain] Embedder load failed (optional): {e}')

    def _post_schema_init(self):
        """
        Build TF-IDF index if node_vectors is empty but nodes exist.
        Called after ensure_schema() to handle runtime initialization.
        """
        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM node_vectors')
            vector_count = cursor.fetchone()[0]

            cursor = self.conn.execute('SELECT COUNT(*) FROM nodes')
            node_count = cursor.fetchone()[0]

            if node_count > 0 and vector_count == 0:
                print('[brain] Building TF-IDF index for existing nodes...')
                self._rebuild_tfidf_index()
                print('[brain] TF-IDF index built.')
        except Exception:
            # Tables might not exist yet on very first run
            pass

    def _init_rate_limiter(self):
        """Initialize in-memory rate limiter for error logging (DDoS protection)."""
        self._error_timestamps = {}    # source -> [monotonic timestamps]
        self._error_fingerprints = {}  # fingerprint -> (last_seen, count)
        self._error_suppressed = {}    # source -> suppressed_count
        self._circuit_open_until = {}  # source -> monotonic time when circuit closes
        self._last_db_size_check = 0.0
        # Limits (tunable via brain_meta)
        self._error_rate_window = 3600     # 1 hour
        self._error_max_per_source = 50    # per source per window
        self._error_max_global = 200       # across all sources per window
        self._error_dedup_window = 60      # seconds
        self._error_circuit_duration = 900 # 15 minutes
        self._max_logs_db_size = 50 * 1024 * 1024  # 50MB

    def _init_file_logger(self, db_dir: str):
        """Initialize rotating file logger for human-readable brain.log."""
        import logging
        from logging.handlers import RotatingFileHandler
        self._file_logger = logging.getLogger('brain_%s' % id(self))
        self._file_logger.setLevel(logging.DEBUG)
        # Avoid duplicate handlers on re-init
        if not self._file_logger.handlers:
            try:
                handler = RotatingFileHandler(
                    os.path.join(db_dir, 'brain.log'),
                    maxBytes=5 * 1024 * 1024,  # 5MB per file
                    backupCount=2,              # keep brain.log.1 and brain.log.2
                )
                handler.setFormatter(logging.Formatter('%(message)s'))
                self._file_logger.addHandler(handler)
            except Exception as e:
                print('[brain] Could not init file logger: %s' % e)

    def _check_rate_limit(self, source: str, fingerprint: str) -> bool:
        """Check if an error should be logged or suppressed.

        Returns True if the error should be SUPPRESSED (rate limited).
        Three layers: dedup window, per-source limit, global limit, circuit breaker.
        """
        now = time.monotonic()

        # Layer 0: Circuit breaker — if open, suppress everything from this source
        circuit_until = self._circuit_open_until.get(source, 0)
        if now < circuit_until:
            self._error_suppressed[source] = self._error_suppressed.get(source, 0) + 1
            return True
        elif circuit_until > 0 and now >= circuit_until:
            # Circuit just closed — log recovery
            del self._circuit_open_until[source]
            suppressed = self._error_suppressed.pop(source, 0)
            if suppressed > 0:
                self._write_to_file_log('INFO', source,
                    'Circuit breaker closed. %d errors suppressed.' % suppressed)

        # Layer 1: Dedup — same error within dedup window
        fp_entry = self._error_fingerprints.get(fingerprint)
        if fp_entry and (now - fp_entry[0]) < self._error_dedup_window:
            self._error_fingerprints[fingerprint] = (fp_entry[0], fp_entry[1] + 1)
            return True
        self._error_fingerprints[fingerprint] = (now, 1)

        # Layer 2: Per-source rate limit
        timestamps = self._error_timestamps.get(source, [])
        # Prune old timestamps
        cutoff = now - self._error_rate_window
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self._error_max_per_source:
            self._error_suppressed[source] = self._error_suppressed.get(source, 0) + 1
            self._error_timestamps[source] = timestamps
            # Check if circuit breaker should open (saturated 3 checks in a row)
            if self._error_suppressed.get(source, 0) >= self._error_max_per_source:
                self._circuit_open_until[source] = now + self._error_circuit_duration
                self._write_to_file_log('WARN', source,
                    'Circuit breaker OPENED for %ds. Source is flooding errors.' % self._error_circuit_duration)
            return True

        # Layer 3: Global rate limit
        global_count = sum(len(v) for v in self._error_timestamps.values())
        if global_count >= self._error_max_global:
            return True

        timestamps.append(now)
        self._error_timestamps[source] = timestamps
        return False

    def _write_to_file_log(self, level: str, source: str, message: str, traceback_str: str = ''):
        """Write a formatted entry to brain.log."""
        try:
            ts = self.now()
            parts = ['[%s] %s %s: %s' % (ts, level, source, message)]
            if traceback_str:
                for line in traceback_str.strip().split('\n')[-5:]:
                    parts.append('  ' + line)
            parts.append('---')
            self._file_logger.info('\n'.join(parts))
        except Exception:
            pass

    def _check_logs_db_size(self):
        """Rotate logs DB if it exceeds size limit. Checked at most once per minute."""
        now = time.monotonic()
        if now - self._last_db_size_check < 60:
            return
        self._last_db_size_check = now
        try:
            size = os.path.getsize(self.logs_db_path)
            if size > self._max_logs_db_size:
                # Delete entries older than 7 days
                self.logs_conn.execute(
                    "DELETE FROM debug_log WHERE created_at < datetime('now', '-7 days')")
                # access_log + recall_log tables dropped 2026-04-05
                self.logs_conn.execute(
                    "DELETE FROM dream_log WHERE created_at < datetime('now', '-7 days')")
                self.logs_conn.commit()
                self._write_to_file_log('INFO', 'logs_db', 'Pruned entries older than 7 days (DB was %dMB)' % (size // (1024*1024)))
        except Exception:
            pass

    def now(self) -> str:
        """Return current UTC ISO timestamp."""
        return datetime.utcnow().isoformat() + 'Z'

    def _generate_id(self, node_type: str = None) -> str:
        """Generate 8-char hex node ID. ~4.3B combinations, collision-free at brain scale."""
        return uuid.uuid4().hex[:8]

    # ─── Helper: Recency Scoring ───

    # _recency_score, _frequency_score, _combined_score REMOVED 2026-04-02.
    # Replaced by recall_scoring.unified_score() — a pure module with one formula
    # for both embedding and keyword paths. See recall_scoring.py for the
    # research-grounded formula (pattern completion + bounded modulators).

    def _classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent and extract metadata (type boosts, temporal filter, etc).

        Args:
            query: User query string

        Returns:
            Dict with:
                - intent: 'general' or specific intent name
                - typeBoosts: Dict of type→boost multipliers
                - temporalFilter: {'after': ISO, 'before': ISO} or None
                - followEdges: bool for deeper edge traversal
        """
        lower_query = query.lower()
        intent = 'general'
        type_boosts = {}
        temporal_filter = None
        follow_edges = False

        # Check intent patterns
        for intent_name, pattern in INTENT_PATTERNS.items():
            if pattern.search(lower_query):
                intent = intent_name
                type_boosts = INTENT_TYPE_BOOSTS.get(intent_name, {}).copy()
                break

        # Reasoning chains need deeper edge traversal
        if intent == 'reasoning_chain':
            follow_edges = True

        # Check temporal patterns
        for temporal in TEMPORAL_PATTERNS:
            pattern = temporal['pattern']
            match = pattern.search(lower_query)
            if match:
                range_fn = temporal['range_fn']
                try:
                    temporal_filter = range_fn(match)
                except TypeError:
                    temporal_filter = range_fn()
                if intent == 'general':
                    intent = 'temporal'
                type_boosts.update(INTENT_TYPE_BOOSTS.get('temporal', {}))
                break

        return {
            'intent': intent,
            'typeBoosts': type_boosts,
            'temporalFilter': temporal_filter,
            'followEdges': follow_edges
        }

    # ─── TF-IDF Methods ───







    # ─── Connection/Edge Management ───



    # ─── Embedding Integration ───




    # ─── Session Activity Tracking ───

    def _get_session_activity(self) -> Dict[str, Any]:
        """Read session activity from brain_meta via DAL."""
        try:
            return self._meta.get_session_activity()
        except Exception:
            return {}

    def _update_session_activity(self, key: str, value: Any):
        """Write session activity to brain_meta via DAL."""
        self._meta.set(key, str(value))

    @property
    def session_context(self):
        """Single accessor for session context (running journey summary).

        v9.2: Replaces 4 scattered get_config("session_context", "") reads
        across daemon_hooks.py and encoding_agent.py.
        """
        return self.get_config('session_context', '') or ''

    def get_interaction_config(self, name: str) -> dict:
        """Get the latest config for an interaction. Returns {} if not found."""
        interaction = self._interaction_dal.get_latest(name)
        if not interaction or not interaction.get('parameters'):
            return {}
        try:
            return json.loads(interaction['parameters'])
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_interaction_prompt(self, name: str) -> str:
        """Get the latest prompt text for an LLM interaction. Returns '' if not found."""
        interaction = self._interaction_dal.get_latest(name)
        if not interaction:
            return ''
        return interaction.get('template', '')

    def get_relations_for_families(self, *family_names) -> set:
        """Get all edge relation types belonging to the given families.

        Reads from s2_edge_families interaction (source of truth).
        Returns a set of relation strings.

        Usage:
            correction_rels = brain.get_relations_for_families(
                'correction_improvement', 'hierarchical_structure')
            # → {'corrects', 'corrected_by', 'supersedes', 'superseded_by', ...}
        """
        config = self.get_interaction_config('s2_edge_families')
        relations = set()
        for family in family_names:
            members = config.get(family, [])
            if isinstance(members, list):
                relations.update(members)
        return relations

    def get_interaction(self, name: str, version: int = 0) -> dict:
        """Get interaction by name. version=0 (default) returns latest, else specific version."""
        if version:
            return self._interaction_dal.get_version(name, version)
        return self._interaction_dal.get_latest(name)

    def get_or_create_session(self, session_id: str) -> 'SessionContext':
        """Get or create a SessionContext for a given session_id.

        This is the single entry point for session state. Hooks send
        session_id from Claude Code args. The brain holds the state.

        If the session exists in DB (daemon restarted), loads it.
        If new, creates and persists it.
        """
        from .session_context import SessionContext
        if not session_id:
            # Fallback: use brain_meta session_id (set at boot) rather than random UUID
            session_id = self.session_id
            if session_id == 'no_session':
                session_id = uuid.uuid4().hex
        ctx = SessionContext.load(self.logs_conn, session_id)
        if ctx:
            return ctx
        ctx = SessionContext(session_id=session_id)
        ctx.save(self.logs_conn)
        return ctx

    @property
    def session_id(self):
        """DEPRECATED — use get_or_create_session(session_id) instead.

        Kept for backward compatibility. Returns the last-seen session_id
        from brain_meta. New code should pass SessionContext through the
        call chain, not read from this singleton property.
        """
        if not hasattr(self, '_cached_session_id') or not self._cached_session_id:
            self._cached_session_id = self.get_config('session_id', '') or ''
        return self._cached_session_id or 'no_session'

    def reset_session_activity(self, session_id: str = ''):
        """Reset session counters. Session_id comes from hook args, not generated.

        Also creates/updates SessionContext in DB for persistence across
        daemon restarts.
        """
        sid = session_id or uuid.uuid4().hex
        self._update_session_activity('remember_count', 0)
        self._update_session_activity('edit_check_count', 0)
        self._update_session_activity('message_count', 0)
        self._update_session_activity('last_encode_at_message', 0)
        self._update_session_activity('boot_time', self.now())
        self._update_session_activity('session_id', sid)
        self._cached_session_id = sid
        # Persist SessionContext for daemon restart recovery
        from .session_context import SessionContext
        ctx = SessionContext(session_id=sid)
        ctx.save(self.logs_conn)
        # v5: Reset session state accumulator
        self._session_state = {
            'decisions': [], 'corrections': [], 'inflections': [],
            'model_updates': [], 'validations': [], 'open_questions': []
        }
        # v5.1: Reset segment state
        self.set_config('segment_id', '0')
        self.set_config('segment_embeddings', '[]')
        self.set_config('segment_node_ids', '[]')

    def check_segment_boundary(self, query_embedding):
        """Detect if a new message represents a context/topic shift.

        Compares the query embedding against the centroid of the last N
        message embeddings (sliding window). If similarity drops below
        threshold, declares a new segment boundary.

        Uses get_config/set_config directly (not _get_session_activity)
        because the DAL's session activity reader has a fixed key list.

        Args:
            query_embedding: bytes blob from embedder.embed()

        Returns:
            Dict with is_boundary, similarity, segment_id, segment_count
        """
        if not query_embedding:
            return {'is_boundary': False, 'segment_id': 0}

        import base64

        current_seg = int(self.get_config('segment_id', 0) or 0)

        # Decode stored embeddings from brain_meta
        stored_json = self.get_config('segment_embeddings', '[]') or '[]'
        try:
            stored_b64 = json.loads(stored_json)
        except Exception:
            stored_b64 = []

        stored_blobs = []
        for b64 in stored_b64:
            try:
                stored_blobs.append(base64.b64decode(b64))
            except Exception:
                pass

        # Warmup: need at least N messages before detecting boundaries
        window_size = int(self._get_tunable('segment_window_size', 2))
        if len(stored_blobs) < window_size:
            new_b64 = base64.b64encode(query_embedding).decode('ascii')
            stored_b64.append(new_b64)
            self.set_config('segment_embeddings', json.dumps(stored_b64))
            return {
                'is_boundary': False,
                'similarity': 1.0,
                'segment_id': current_seg,
                'segment_count': current_seg + 1,
            }

        # Compute centroid of sliding window
        centroid = embedder.compute_centroid(stored_blobs[-window_size:])
        if not centroid:
            return {'is_boundary': False, 'segment_id': current_seg}

        sim = embedder.cosine_similarity(query_embedding, centroid)
        threshold = float(self._get_tunable('segment_boundary_threshold', 0.74))

        is_boundary = sim < threshold

        if is_boundary:
            new_seg = current_seg + 1
            new_b64 = base64.b64encode(query_embedding).decode('ascii')
            self.set_config('segment_id', str(new_seg))
            self.set_config('segment_embeddings', json.dumps([new_b64]))
            self.set_config('segment_node_ids', '[]')
            return {
                'is_boundary': True,
                'similarity': round(sim, 3),
                'segment_id': new_seg,
                'segment_count': new_seg + 1,
            }
        else:
            new_b64 = base64.b64encode(query_embedding).decode('ascii')
            stored_b64.append(new_b64)
            stored_b64 = stored_b64[-window_size:]
            self.set_config('segment_embeddings', json.dumps(stored_b64))
            return {
                'is_boundary': False,
                'similarity': round(sim, 3),
                'segment_id': current_seg,
                'segment_count': current_seg + 1,
            }

    def get_current_segment_id(self):
        """Get the current conversation segment ID."""
        return int(self.get_config('segment_id', 0) or 0)

    def get_segment_node_ids(self):
        """Get node IDs created/accessed in the current segment."""
        raw = self.get_config('segment_node_ids', '[]') or '[]'
        try:
            return json.loads(raw)
        except Exception:
            return []

    def add_to_segment(self, node_id):
        """Add a node ID to the current segment's tracking list."""
        current = self.get_segment_node_ids()
        if node_id not in current:
            current.append(node_id)
            self.set_config('segment_node_ids', json.dumps(current))

    def record_remember(self):
        """Increment remember counter and mark last encode position."""
        activity = self._get_session_activity()
        count = activity.get('remember_count', 0) + 1
        self._update_session_activity('remember_count', count)
        # Track which message number the last encode happened at
        msg_count = activity.get('message_count', 0)
        self._update_session_activity('last_encode_at_message', msg_count)

    def record_message(self):
        """Increment message counter. Called by hooks on each user message."""
        activity = self._get_session_activity()
        count = activity.get('message_count', 0) + 1
        self._update_session_activity('message_count', count)

    def record_edit_check(self):
        """Increment edit check counter."""
        activity = self._get_session_activity()
        count = activity.get('edit_check_count', 0) + 1
        self._update_session_activity('edit_check_count', count)

    def get_encoding_heartbeat(self, nudge_threshold: int = 8) -> Optional[Dict[str, Any]]:
        """Check if Claude should be nudged to encode learnings.

        Returns a nudge dict if messages since last encode exceeds threshold,
        None otherwise. The nudge includes context about what's been missed.

        Args:
            nudge_threshold: Messages without encoding before nudging (default 8)
        """
        activity = self._get_session_activity()
        msg_count = int(activity.get('message_count', 0))
        remember_count = int(activity.get('remember_count', 0))
        last_encode_at = int(activity.get('last_encode_at_message', 0))

        messages_since_encode = msg_count - last_encode_at

        if messages_since_encode < nudge_threshold:
            return None

        # Build nudge with context
        nudge = {
            'messages_since_encode': messages_since_encode,
            'total_messages': msg_count,
            'total_encodes': remember_count,
            'severity': 'gentle' if messages_since_encode < 15 else 'urgent',
        }

        if remember_count == 0:
            nudge['message'] = '%d messages in session, nothing encoded yet. Decisions, corrections, or learnings to capture?' % msg_count
        else:
            nudge['message'] = '%d messages since last encode (%d total encodes). Any recent decisions or learnings worth preserving?' % (messages_since_encode, remember_count)

        return nudge

    # ─── Utilities ───

    def _get_node_count(self) -> int:
        """Get count of non-archived nodes."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM nodes WHERE archived = 0')
        return cursor.fetchone()[0]

    def _get_edge_count(self) -> int:
        """Get total edge count."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM edges')
        return cursor.fetchone()[0]

    def _get_locked_count(self) -> int:
        """Get count of locked nodes."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0')
        return cursor.fetchone()[0]

    # ─── REMEMBER: Store a new node with TF-IDF + embeddings ───




    # ─── v5 PHASE 2: Rich encoding API ───




    # ═══════════════════════════════════════════════════════════════
    # v5 SPRINT 2: Engineering Memory + Cognitive Layer
    # ═══════════════════════════════════════════════════════════════

    # ─── Engineering Memory: 7 kinds of understanding ───











    # ─── Cognitive Layer: Claude's own thoughts ───




    # ─── Project Maps: file inventory + change detection ───







    # ─── Phase 3: Self-Correction Traces + Positive Signals ───




    # ─── Phase 4: Session Synthesis Engine ───




    # ─── RECALL: v5 with TF-IDF + intent detection + temporal filtering + decay ───


    # ─── RECALL WITH EMBEDDINGS: Phase 0.5B — Embeddings-first recall ───


    # ─── SPREAD ACTIVATION: Multi-hop semantic activation ───


    # ─── Helper methods for remember/recall ───


































    # ─── v4: EVOLUTION TYPES ───
    # Tensions, hypotheses, patterns, catalysts, aspirations.
    # Forward-facing nodes that describe what is BECOMING, not what IS.








    # ─── v4: CODE COGNITION HELPERS ───
    # Semantic code understanding — not storing code, but understanding what it means.








    # ─── v4: SELF-REFLECTION TYPES ───
    # Brain looking inward — performance, failure modes, capabilities, interaction, meta-learning.






    # ─── v4: REMINDERS (due_date) ───




    # ─── v4: CONSCIOUSNESS — DELETED ───
    # get_consciousness_signals() deleted (2026-03-26).
    # Signals migrated to signal_producers.py + signal queue.
    # See: servers/signal_producers.py, servers/dal_signal_queue.py, servers/surface_assembler.py


    # ─── v4: PATTERN-INFORMED PRUNING ───
    # Confirmed patterns can adjust how the brain prunes. "Personal info is rare but
    # always significant" → protect low-frequency personal nodes.

    def get_pruning_adjustments(self) -> Dict[str, float]:
        """
        Read confirmed patterns and derive pruning adjustments.
        Returns dict of node_type → decay_multiplier.
        A multiplier > 1 means slower decay (more protection).
        """
        adjustments = {}
        try:
            # Find confirmed patterns that mention pruning or decay
            cursor = self.conn.execute(
                """SELECT content FROM nodes
                   WHERE type = 'pattern' AND evolution_status IN ('active', 'confirmed')
                     AND archived = 0
                     AND (content LIKE '%decay%' OR content LIKE '%prune%' OR content LIKE '%protect%'
                          OR content LIKE '%personal%' OR content LIKE '%important%')"""
            )
            for (content,) in cursor.fetchall():
                content_lower = content.lower() if content else ''
                # Simple heuristic: if pattern mentions protecting personal info
                if 'personal' in content_lower and ('protect' in content_lower or 'important' in content_lower):
                    adjustments['context'] = max(adjustments.get('context', 1), 3.0)
                    adjustments['concept'] = max(adjustments.get('concept', 1), 2.0)
                # If pattern mentions code being important
                if 'code' in content_lower and ('protect' in content_lower or 'important' in content_lower):
                    adjustments['code_concept'] = max(adjustments.get('code_concept', 1), 2.0)
                    adjustments['fn_reasoning'] = max(adjustments.get('fn_reasoning', 1), 2.0)
        except Exception:
            pass

        # Store adjustments for the decay function to read
        try:
            self.set_config('pruning_adjustments', json.dumps(adjustments))
        except Exception:
            pass

        return adjustments

    # ─── v4: COMMUNICATION FAILURE LOG ───
    # Track when Brain→Host signals are ignored. Learn how to talk to the host.

    def log_communication(self, node_id: str, signal_level: str, host_followed: bool,
                          context: Optional[str] = None):
        """
        Log whether the host acted on a brain signal.
        signal_level: 'high_priority', 'medium_priority', 'low_priority'
        host_followed: did the host act on it?
        Over time: brain learns signal force needed for compliance.
        """
        ts = self.now()
        key_yes = f'comm_{signal_level}_followed'
        key_no = f'comm_{signal_level}_ignored'
        key = key_yes if host_followed else key_no

        current = int(self.get_config(key, 0) or 0)
        self.set_config(key, current + 1)

        # Log individual event for pattern analysis
        try:
            self.conn.execute(
                """INSERT INTO brain_meta (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (f'comm_event_{int(time.time() * 1000)}',
                 json.dumps({'node_id': node_id, 'level': signal_level,
                             'followed': host_followed, 'context': context}),
                 ts)
            )
            self.conn.commit()
        except Exception:
            pass

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication compliance rates by signal level."""
        stats = {}
        for level in ('high_priority', 'medium_priority', 'low_priority'):
            followed = int(self.get_config(f'comm_{level}_followed', 0) or 0)
            ignored = int(self.get_config(f'comm_{level}_ignored', 0) or 0)
            total = followed + ignored
            stats[level] = {
                'followed': followed,
                'ignored': ignored,
                'total': total,
                'compliance_rate': followed / total if total > 0 else None,
            }
        return stats

    # ─── v4: FEEDBACK API (confirm/dismiss/refine conscious items) ───



    # ─── v4: DREAM SURPRISE SCORING FOR CONSCIOUSNESS ───

    def get_surfaceable_dreams(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get high-surprise dreams worth surfacing to consciousness.
        Filters: interest_score >= 4, created in last 48 hours, cross-cluster.
        """
        try:
            cursor = self.logs_conn.execute(
                """SELECT intuition_node_id, insight, seed_nodes, walk_path, created_at
                   FROM dream_log
                   WHERE created_at > datetime('now', '-48 hours')
                     AND intuition_node_id IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT 20"""
            )
            dreams = []
            for row in cursor.fetchall():
                # Look up node title/content from main DB
                node_row = self.conn.execute(
                    'SELECT title, content FROM nodes WHERE id = ?', (row[0],)
                ).fetchone()
                node_title = node_row[0] if node_row else ''
                node_content = node_row[1] if node_row else ''
                # Recompute surprise: longer walks = more distant = more surprising
                try:
                    walks = json.loads(row[3])
                    total_hops = sum(len(w) for w in walks)
                except Exception:
                    total_hops = 0

                if total_hops >= 4:  # Cross-cluster threshold
                    dreams.append({
                        'intuition_id': row[0],
                        'insight': row[1],
                        'title': node_title,
                        'content': node_content,
                        'created_at': row[4],
                        'total_hops': total_hops,
                    })

            return dreams[:limit]
        except Exception:
            return []

    # ─── v4: HOST AWARENESS ───

    def scan_host_environment(self) -> Dict[str, Any]:
        """
        Scan the current host environment and compare against last session.
        Returns: current environment state + diff from last session.
        """
        import platform

        env = {
            'python_version': platform.python_version(),
            'platform': platform.system(),
            'embedder_ready': embedder.is_ready(),
            'embedder_model': embedder.stats.get('model_name'),
            'embedder_dim': embedder.stats.get('embedding_dim'),
        }

        # Check fastembed version
        try:
            import fastembed
            env['fastembed_version'] = getattr(fastembed, '__version__', 'unknown')
        except ImportError:
            env['fastembed_version'] = None

        # Check for Cowork vs CLI vs other
        if os.path.exists('/sessions'):
            env['host_type'] = 'cowork'
        elif os.environ.get('CLAUDE_CODE'):
            env['host_type'] = 'claude_code'
        else:
            env['host_type'] = 'unknown'

        # Check proxy status
        env['proxy'] = os.environ.get('ALL_PROXY') or os.environ.get('HTTPS_PROXY') or None

        # Mounted directories (Cowork)
        mounts = []
        try:
            for d in os.listdir('/sessions'):
                mnt_path = f'/sessions/{d}/mnt'
                if os.path.isdir(mnt_path):
                    for item in os.listdir(mnt_path):
                        if not item.startswith('.'):
                            mounts.append(item)
        except Exception:
            pass
        env['mounted_dirs'] = mounts

        # Available pip packages relevant to brain
        for pkg in ['fastembed', 'brain_embedding', 'sqlite_vec', 'onnxruntime']:
            try:
                __import__(pkg.replace('-', '_'))
                env[f'pkg_{pkg}'] = True
            except ImportError:
                env[f'pkg_{pkg}'] = False

        # Compare against last session
        last_env_str = self.get_config('last_host_environment', '')
        diff = {}
        if last_env_str:
            try:
                last_env = json.loads(last_env_str)
                for key in set(list(env.keys()) + list(last_env.keys())):
                    if str(env.get(key)) != str(last_env.get(key)):
                        diff[key] = {'was': last_env.get(key), 'now': env.get(key)}
            except Exception:
                pass

        # Save current environment
        self.set_config('last_host_environment', json.dumps(env, default=str))

        # Flag if research needed (version changes, new packages)
        research_needed = []
        for key in diff:
            if 'version' in key and diff[key].get('now'):
                research_needed.append(f"Version change: {key} {diff[key].get('was')} → {diff[key].get('now')}")
            if key.startswith('pkg_') and diff[key].get('now') != diff[key].get('was'):
                research_needed.append(f"Package change: {key}")

        return {
            'environment': env,
            'diff': diff,
            'research_needed': research_needed,
        }

    # ─── v4: PROACTIVE BRAIN (Phase 3) ───




    # ─── v4: AUTO SELF-REFLECTION (Phase 4) ───


    # ─── v4: PERSONAL FLAG ───



    # ─── EMBEDDER CONFIG: Model-agnostic configuration ───

    # Default embedder config — used when brain_meta has no overrides.
    # These match plugin.json defaults. If plugin.json changes, update here too.
    _EMBEDDER_DEFAULTS = {
        'model_name': 'Snowflake/snowflake-arctic-embed-m-v1.5',
        'dim': 768,
        'pooling': 'cls',
        'model_file': 'onnx/model.onnx',
        'model_path': 'model-package/brain_embedding/model',
        'cache_dir': None,
    }

    def _get_embedder_config(self) -> Dict[str, Any]:
        """
        Read embedder config from brain_meta, falling back to defaults.
        Config keys: embedder_model_name, embedder_dim, embedder_pooling,
                     embedder_model_file, embedder_model_path, embedder_cache_dir.

        Users can override any of these via set_config() to switch models
        without changing code.
        """
        config = {}
        for key, default in self._EMBEDDER_DEFAULTS.items():
            meta_key = f'embedder_{key}'
            val = self.get_config(meta_key, default)
            # Handle None stored as string
            if val == 'None' or val == 'null':
                val = None
            config[key] = val
        return config

    def set_embedder_config(self, **kwargs) -> Dict[str, Any]:
        """
        Update embedder config in brain_meta. Takes effect on next boot.

        Example:
            brain.set_embedder_config(model_name="nomic-ai/nomic-embed-text-v1.5-Q", dim=768, pooling="mean")
        """
        updated = {}
        for key, value in kwargs.items():
            if key in self._EMBEDDER_DEFAULTS:
                meta_key = f'embedder_{key}'
                self.set_config(meta_key, value)
                updated[key] = value
        return {'updated': updated, 'takes_effect': 'next boot'}

    def set_config(self, key: str, value: Any) -> Dict[str, Any]:
        """Set a config value in brain_meta via DAL. Persists across restarts."""
        self._meta.set(key, str(value))
        return {'key': key, 'value': value, 'updated_at': self.now()}


    def get_debug_status(self) -> Dict[str, Any]:
        """Check if debug mode is enabled via DAL, falls back to env var."""
        try:
            val = self._meta.get('debug_enabled', '')
            if val:
                return {'debug_enabled': val == '1'}
        except Exception:
            pass
        return {'debug_enabled': os.environ.get('BRAIN_DEBUG') == '1'}

    # log_conflict + resolve_conflict REMOVED 2026-04-05 — conflict_log table dropped

    def _log_error(self, source: str, error: Exception, context: str = ''):
        """Log an error to brain_logs.db + brain.log with rate limiting.

        Replaces silent `except: pass` blocks. Errors are stored in the logs DB
        and surfaced at boot via consciousness signals.
        """
        try:
            import traceback
            error_str = str(error)
            error_type = type(error).__name__

            # Rate limit check — compute fingerprint
            fingerprint = '%s:%s:%s' % (source, error_type, error_str[:100])
            if self._check_rate_limit(source, fingerprint):
                return  # suppressed

            tb = traceback.format_exception(type(error), error, error.__traceback__)
            tb_short = ''.join(tb[-3:]) if len(tb) > 3 else ''.join(tb)

            # Write to logs DB
            self._check_logs_db_size()
            self.logs_conn.execute('''
                INSERT INTO debug_log
                  (session_id, event_type, source, metadata, created_at)
                VALUES (?, 'error', ?, ?, ?)
            ''', (
                self._get_session_activity().get('session_id', 'unknown'),
                source,
                json.dumps({
                    'error': error_str,
                    'type': error_type,
                    'context': context,
                    'traceback': tb_short[:500],
                }),
                self.now()
            ))

            # Write to human-readable log file
            self._write_to_file_log('ERROR', source,
                '%s: %s' % (error_type, error_str),
                tb_short)
        except Exception:
            # Last resort — can't even log the error. Print to stderr.
            print('brain: error in %s: %s (context: %s)' % (source, error, context),
                  file=sys.stderr)

    def get_recent_errors(self, hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors from brain_logs.db via DAL."""
        try:
            return self._logs_dal.get_recent_errors(hours=hours, limit=limit)
        except Exception:
            return []

    def log_debug(self, event_type: str, source: str, **kwargs) -> Dict[str, Any]:
        """Log a debug event to brain_logs.db + brain.log."""
        try:
            ts = self.now()
            self.logs_conn.execute('''
                INSERT INTO debug_log
                  (session_id, event_type, source, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', ('unknown', event_type, source, json.dumps(kwargs), ts))
            # Also write to file log for non-error events
            self._write_to_file_log('DEBUG', source, '%s: %s' % (event_type, json.dumps(kwargs)[:200]))
            return {'logged': True}
        except Exception as e:
            return {'logged': False, 'error': str(e)}


    # ─── Tunable Parameters ───
    # Brain-level parameters that can be self-tuned during healing.
    # Hardcoded module constants (DECAY_HALF_LIFE, etc.) serve as defaults.
    # Runtime values stored in brain_meta with 'tunable_' prefix.

    def _get_tunable(self, key: str, default: Any = None) -> Any:
        """Read a tunable parameter from brain_meta, falling back to hardcoded default."""
        stored = self.get_config(f'tunable_{key}')
        if stored is not None:
            if isinstance(stored, str):
                try:
                    return json.loads(stored)
                except (json.JSONDecodeError, TypeError) as _e:
                    self._log_error("_get_tunable", _e, f"parsing JSON for tunable key '{key}'")
            return stored
        return default

    def _set_tunable(self, key: str, value: Any, reason: str = '') -> None:
        """Write a tunable parameter to brain_meta and log the change to tuning_log."""
        old = self._get_tunable(key)
        ts = self.now()
        # Store as JSON if dict/list, else as string
        store_val = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        self.set_config(f'tunable_{key}', store_val)
        # tuning_log writes REMOVED 2026-04-05 — table dropped

    def get_config(self, key: str, default_val: Any = None) -> Any:
        """
        Get a config value from brain_meta via DAL.
        Auto-parses numbers and booleans based on default_val type.
        """
        try:
            val = self._meta.get(key, "")
            if not val:
                return default_val
            # Auto-parse numbers
            if default_val is not None and isinstance(default_val, (int, float)):
                try:
                    return float(val) if '.' in str(val) else int(val)
                except (ValueError, TypeError) as _e:
                    self._log_error("get_config", _e, "parsing numeric config value for key '%s'" % key)
            if val == 'true':
                return True
            if val == 'false':
                return False
            return val
        except Exception as _e:
            self._log_error("get_config", _e, "reading config key '%s' from brain_meta" % key)
        return default_val

    # ─── Pre-Edit Batch Method ───
    # Replaces the /pre-edit HTTP endpoint from index.js
    # Combines suggest + procedures + encoding health into one call



    # ═══════════════════════════════════════════════════════════════
    # ABSORB — Merge knowledge from another brain
    # ═══════════════════════════════════════════════════════════════




    def save(self, backup: bool = False):
        """
        Commit pending changes and optionally back up database.

        Args:
            backup: If True, create a backup copy
        """
        self.conn.commit()
        try:
            self.logs_conn.commit()
        except Exception:
            pass

        if backup and self.db_path:
            try:
                import shutil
                backup_path = f'{self.db_path}.backup-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}'
                shutil.copy2(self.db_path, backup_path)
            except Exception as e:
                print(f'[brain] Backup failed: {e}')

    def close(self):
        """Commit, close both database connections, and remove from singleton cache."""
        self.conn.commit()
        self.conn.close()
        try:
            self.logs_conn.commit()
            self.logs_conn.close()
        except Exception:
            pass
        # Remove from singleton registry if present
        canonical = os.path.realpath(self.db_path)
        with Brain._lock:
            if Brain._instances.get(canonical) is self:
                del Brain._instances[canonical]
