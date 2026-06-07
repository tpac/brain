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
from .dal import LogsDAL, BrainMetaDAL
from .db_backends.sqlite import commit_unless_batched
from .clock import iso_cutoff, iso_now
from .brain_recall import BrainRecallMixin
from .brain_remember import BrainRememberMixin
from .brain_connections import BrainConnectionsMixin
from .brain_reminders import BrainRemindersMixin
from .brain_assembly import BrainAssemblyMixin
from .brain_corrections import BrainCorrectionsMixin
from . import embedder


from .brain_constants import (
    DECAY_HALF_LIFE,
    INTENT_PATTERNS, INTENT_TYPE_BOOSTS, TEMPORAL_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════
# BRAIN CLASS
# ═══════════════════════════════════════════════════════════════

class Brain(
    BrainRecallMixin,
    BrainRememberMixin,
    BrainConnectionsMixin,
    BrainRemindersMixin,
    BrainAssemblyMixin,
    BrainCorrectionsMixin,
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
                    instance.conn.commit()  # commit-ok: teardown flush before close (clear_instances)
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

        # Batch state lives on the CONNECTION (self.conn.in_batch), set by
        # BatchAwareConnection below — not a brain-level flag. The batch
        # owner (_handle_brain_batch) flips conn.in_batch for the duration
        # of its BEGIN IMMEDIATE / COMMIT; DAL writers consult it via
        # commit_unless_batched() and brain._maybe_commit() does the same.
        # One source of truth on the resource it describes, so a writer
        # can't break batch atomicity by forgetting a kwarg. (Replaces the
        # old by-convention brain._batch_mode + per-call commit= kwarg.)

        # Serializes all writers to brain.db / brain_logs.db. The lock lives
        # on the brain (the resource it protects), not the daemon — this lets
        # any caller that holds a brain reference participate in
        # serialization: daemon dispatch, S2 encoder dispatch, embed_queue
        # backfill, autosave. RLock so a thread that already holds it can
        # call into a brain method that also wants to acquire it without
        # deadlock. TrackedRLock exposes the current holder for the
        # bg_writer stall watchdog — when a drain times out we want to
        # know which thread was holding when.
        from .tracked_lock import TrackedRLock
        self.write_lock = TrackedRLock()

        # Open SQLite connection with WAL mode for concurrency. Pragma
        # set comes from db_backends.current — single source for every
        # connection in the daemon.
        from . import db_backends
        self.conn = sqlite3.connect(
            db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.conn)

        # Background-writer connection (2026-05-18). Owned exclusively by the
        # embed_queue worker thread for batched, deferred, non-realtime writes:
        # temporal_extract intervals, vector backfill, recall-side access marks
        # (`_mark_accessed`), and Hebbian co-access strengthening (moved out
        # of the recall hot path at the surface layer). The recall hot path is
        # read-only at SQLite. Foreground writes via `self.conn` no longer race
        # with background batches at the WAL writer slot.
        #
        # Replaces the prior `self.conn_recall_write` design (Phase 8 cleanup,
        # 2026-05-18). The two-writer design split conn_recall_write from
        # `self.conn` to keep recall's frequent UPDATE+COMMIT off the long-
        # running write path; this is now obsolete because recall enqueues
        # instead of writing inline.
        #
        # Open failure here is a hard boot crash — the daemon cannot serve
        # writes without this connection. sqlite3 raises OperationalError,
        # which propagates to daemon_server boot path and surfaces in daemon.log.
        self.conn_bg_writer = sqlite3.connect(
            db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.conn_bg_writer)

        # Daemon boot timestamp — used by run_maintenance_if_due to enforce
        # MAINTENANCE_BOOT_GRACE_SECONDS so maintenance never fires during
        # the first N seconds after start. Boot already runs schema +
        # migrations + S0/S1 startup paths; piling consolidation on top
        # makes first-user-recall consistently time out.
        import time as _time
        self._boot_time = _time.time()

        # Create schema if needed
        ensure_schema(self.conn, db_path=db_path)

        # Open separate logs database (brain_logs.db)
        db_dir = os.path.dirname(db_path) or '.'
        self.logs_db_path = os.path.join(db_dir, 'brain_logs.db')
        self.logs_conn = sqlite3.connect(
            self.logs_db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.logs_conn)
        ensure_logs_schema(self.logs_conn)

        # One-time migration: move log tables from brain.db to brain_logs.db
        migrate_logs_to_separate_db(self.conn, self.logs_conn)

        # DAL instances — incremental adoption, brain.py migrates one method at a time
        self._meta = BrainMetaDAL(self.conn)
        self._logs_dal = LogsDAL(self.logs_conn)
        from .dal import TraceDAL, InteractionDAL, SessionStateDAL
        self._trace_dal = TraceDAL(self.logs_conn)
        self._interaction_dal = InteractionDAL(self.logs_conn)
        # logs-bound: session_state lives in brain_logs.db, not brain.db
        self._session_state = SessionStateDAL(self.logs_conn)

        # Repository aggregate (DAL cleanup Phase 2): hold the brain.db DALs
        # foreground-conn-bound so methods use them by construction instead of
        # re-instantiating XDAL(self.conn) ad hoc. The bg-writer path
        # (recall_write_queue) constructs its own GraphDAL on conn_bg_writer —
        # the one documented exception.
        from .dal import (NodeDAL, GraphDAL, Fts5DAL, TfIdfDAL, EntityDatesDAL,
                          SourceRefDAL)
        from .dal_metadata import MetadataDAL
        self._nodes = NodeDAL(self.conn)
        self._graph = GraphDAL(self.conn)
        self._meta_kv = MetadataDAL(self.conn)
        self._fts = Fts5DAL(self.conn)
        self._tfidf = TfIdfDAL(self.conn)
        self._source_refs = SourceRefDAL(self.conn)  # node_source_refs (not edges)
        # foreground-bound for readers (recall_by_time, cold-start gaps); the
        # backfill writer constructs EntityDatesDAL(conn_bg_writer) per its routing.
        self._entity_dates = EntityDatesDAL(self.conn)

        # Identity binding — concrete names of the human partner and the
        # agent at this brain's "current moment." Stamped onto every S0
        # trace event so each row independently records who said what
        # (matches biology's per-utterance speaker binding; survives
        # partner changes without rewriting history). Sourced from env
        # at boot; empty when unset → DAL skips stamping.
        #
        # The "identity missing" loud signal lives at the write boundary
        # (TraceDAL._maybe_warn_identity_unset), not here — boot is one
        # moment; trace writes are continuous, and that's where the gap
        # actually manifests.
        from .daemon_config import get_operator_name, get_agent_name
        self.operator_name = get_operator_name()
        self.agent_name = get_agent_name()
        self._trace_dal.set_identity(self.operator_name, self.agent_name)

        # Shared vector DAL — cache-backed by default. Set env
        # BRAIN_DISABLE_VECTOR_CACHE=1 to fall back to raw VectorDAL for
        # emergency rollback or A/B benchmarking. Brain consumers use
        # self._vec_dal either way.
        if os.environ.get('BRAIN_DISABLE_VECTOR_CACHE'):
            from .dal import VectorDAL
            self._vec_dal = VectorDAL(self.conn)
        else:
            from .dal_vector_cached import CachedVectorDAL
            self._vec_dal = CachedVectorDAL(self.conn)

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
        # _session_state removed 2026-04-13 — was used by deleted synthesize/track methods.

        # Load embedder with config from brain_meta (falls back to plugin.json defaults)
        if not skip_embedder:
            try:
                embedder_config = self._get_embedder_config()
                embedder.load_model(embedder_config)
            except Exception as e:
                print(f'[brain] Embedder load failed (optional): {e}')

        # AspectRegistry — first-class semantic-role API exposed as
        # brain.aspects. Eager validation: loads aspects from aspects_v1.json,
        # checks REQUIRED_ASPECTS present, logs a warning if any are missing
        # (doesn't block). Read-only Brain instances (skip_embedder=True for
        # background scale runs in runner.py) still validate — _load is a
        # cheap JSON read.
        #
        # MUST come before seed_baby_brain — seed creates 16 edges via
        # connect_typed, which hits _maybe_embed_edge_relation
        # (BrainConnectionsMixin), which dereferences brain.aspects to
        # compose edge text for the embedding. Pre-fix, AspectRegistry
        # was initialized AFTER seed_pack, so seed edges silently failed
        # to embed (caught in try/except, logged as
        # 'edge_embedding_write: AttributeError').
        try:
            from .aspects import AspectRegistry
            self.aspects = AspectRegistry(self)
        except Exception as _e:
            # Never let registry init crash Brain init. Log + leave aspects
            # unset; consumers that read brain.aspects will get AttributeError
            # which is louder than a silent empty registry.
            print('[brain] WARNING: AspectRegistry init failed: %s' % _e, flush=True)

        # Seed baby brain nodes if missing (runs AFTER embedder — remember() needs it)
        if not skip_embedder:
            try:
                from .seed_pack import seed_baby_brain
                seed_baby_brain(self)
            except Exception as _e:
                print('[brain] WARNING: seed_pack failed: %s' % _e, flush=True)

    def _post_schema_init(self):
        """
        Build TF-IDF index if node_vectors is empty but nodes exist.
        Called after ensure_schema() to handle runtime initialization.
        """
        try:
            # node_vectors empty? get_total_docs (COUNT DISTINCT node_id) is 0
            # iff the table is empty — equivalent to the old COUNT(*)==0 check.
            index_empty = self._tfidf.get_total_docs() == 0
            # count(archived=True) == COUNT(*) FROM nodes (all states), matching
            # the prior check exactly.
            node_count = self._nodes.count(archived=True)

            if node_count > 0 and index_empty:
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
        # SessionContext in-memory cache — keyed by session_id. Mutations
        # (fatigue, counters) live here across hooks within the same session;
        # autosave loop persists every AUTOSAVE_INTERVAL_SECONDS. Cleared
        # entries on SessionEnd hook.
        from typing import Dict as _Dict
        self._session_contexts: _Dict[str, 'SessionContext'] = {}
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
                cutoff_7d = iso_cutoff(days=7)
                self.logs_conn.execute(
                    "DELETE FROM debug_log WHERE created_at < ?", (cutoff_7d,))
                # access_log + recall_log tables dropped 2026-04-05
                self.logs_conn.execute(
                    "DELETE FROM dream_log WHERE created_at < ?", (cutoff_7d,))
                self.logs_conn.commit()
                self._write_to_file_log('INFO', 'logs_db', 'Pruned entries older than 7 days (DB was %dMB)' % (size // (1024*1024)))
        except Exception:
            pass

    def now(self) -> str:
        """Return current UTC ISO timestamp.

        Routes through ``iso_now()`` — single source of truth for the
        write-side timestamp format (``'…+00:00'``). Pre-2026-05-24 this
        emitted ``'…Z'``; historical rows with that suffix remain valid
        (consumers normalize via ``.replace('Z', '+00:00')`` before
        parsing). See ``servers/clock.py:iso_now``.
        """
        return iso_now()

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
    # Counters live on SessionContext, persisted via session_state.
    # See record_remember / record_message / record_edit_check /
    # get_encoding_heartbeat / reset_session_activity — all session-keyed.

    def session_context_for(self, session_id: str) -> str:
        """Per-session running journey summary (encoder's session arc).

        2026-05-02 (Frame Phase 2.5): replaces the global `session_context`
        property. The previous shape (single brain_meta key, leaked across
        parallel sessions — last-writer-wins, two Claude Code instances
        scrambled each other's arc) is gone. Each session writes/reads
        its own `session_context_{session_id}` key, mirroring the existing
        per-session `encoding_journal_{session_id}` pattern.

        Cross-session continuity (e.g., boot pulling the previous session's
        context) becomes a deliberate query — not an accidental side-effect
        of a leaky global. See docs/FRAME-DESIGN.md Phase 3 work.

        Returns empty string if no context for this session yet.
        """
        if not session_id:
            return ''
        return self.get_config('session_context_' + session_id, '') or ''

    def get_recent_encoding_journal(self, session_id: str, max_chars: int = 1500) -> str:
        """Read the most recent portion of the encoder's per-session journal.

        The journal is the encoder's running log of what it ENCODED / SKIPPED /
        is WATCHING this session, plus the SESSION CONTEXT field. It's
        prepended (newest first) with "--- Run N (stop #X) ---" delimiters,
        so the first max_chars naturally capture the most recent encoding
        pass.

        Used by surface (Phase 1 of Frame work, 2026-05-02) to give Haiku
        the encoder's current understanding of the session — not just the
        rolling 800-char session_context blob.

        Returns empty string if no journal exists for this session.
        """
        if not session_id:
            return ''
        full = self.get_config('encoding_journal_' + session_id, '') or ''
        return full[:max_chars] if full else ''

    def get_interaction_config(self, name: str) -> dict:
        """Get the active config for an interaction. Returns {} if not found.

        Reads the currently-active version via interaction_active pointer.
        Registering a new version does NOT change what this returns —
        call set_interaction_active() to flip the runtime to a new version.
        """
        interaction = self._interaction_dal.get_active(name)
        if not interaction or not interaction.get('parameters'):
            return {}
        try:
            return json.loads(interaction['parameters'])
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_interaction_prompt(self, name: str) -> str:
        """Get the active prompt text for an LLM interaction. Returns '' if not found.

        Reads the currently-active version via interaction_active pointer.
        See get_interaction_config() for the activation model.
        """
        interaction = self._interaction_dal.get_active(name)
        if not interaction:
            return ''
        return interaction.get('template', '')

    # get_relations_for_families — REMOVED 2026-05-04 (Step 12 of unified-aspects).
    # Replaced by brain.aspects.<name>.edge_relations (single name) or
    # brain.aspects.relations_in([names]) (multi-name union). All callers
    # migrated through Steps 7-11; the legacy s2_edge_families interaction
    # is no longer the source of truth.

    def get_interaction(self, name: str, version: int = 0) -> dict:
        """Get interaction by name. version=0 (default) returns active, else specific version."""
        if version:
            return self._interaction_dal.get_version(name, version)
        return self._interaction_dal.get_active(name)

    def set_interaction_active(self, name: str, version: int,
                                set_by: str = 'anchor') -> dict:
        """Flip the active version pointer for `name`. Runtime picks up
        the new active version on the next read of get_interaction_prompt
        or get_interaction_config. See InteractionDAL.set_active.
        """
        return self._interaction_dal.set_active(name, version, set_by)

    def get_or_create_session(self, session_id: str) -> 'SessionContext':
        """Get or create a SessionContext for a given session_id.

        Single entry point for session state. Hooks send session_id from
        Claude Code args. The brain holds the state.

        In-memory cache (2026-05-17): SessionContext instances are cached
        on `self._session_contexts` and returned by reference. Mutations
        to fatigue / counters live in memory across hooks in the same
        session; the autosave loop persists every
        `AUTOSAVE_INTERVAL_SECONDS` via `save_session_contexts()`. The
        cache entry is cleared on SessionEnd hook (clean shutdown of
        that session) — see daemon_hooks.hook_session_end.

        Race-safe on first load: two threads with the same brand-new
        session_id can call concurrently. The create path takes `write_lock`
        (the serializer for brain_logs.db writes) so the shared `logs_conn`
        can't hit concurrent transactions; `INSERT OR IGNORE` keeps the row
        itself idempotent. After first load, both threads operate on the same
        cached instance (Python attribute access — `write_lock` serializes
        actual mutations).
        """
        from .session_context import SessionContext
        import json as _json
        from datetime import datetime, timezone
        if not session_id:
            # Fallback: use brain_meta session_id (set at boot) rather than random UUID
            session_id = self.session_id
            if session_id == 'no_session':
                session_id = uuid.uuid4().hex
        # Fast path: already cached (lock-free — the common case).
        cached = self._session_contexts.get(session_id)
        if cached is not None:
            return cached
        # First touch: serialize the create under write_lock. INSERT OR IGNORE
        # is row-atomic, but self.logs_conn is shared across daemon request
        # threads (check_same_thread=False, deferred isolation) — two concurrent
        # .execute() calls auto-BEGIN on the same connection and collide with
        # "cannot start a transaction within a transaction". write_lock is the
        # documented serializer for brain_logs.db writes too (see __init__).
        with self.write_lock:
            cached = self._session_contexts.get(session_id)  # re-check under lock
            if cached is not None:
                return cached
            default_data = _json.dumps({'stop_counter': 0, 'fatigue': {}, 'edge_fatigue': {}})
            self._session_state.ensure_default(session_id, '_session_context', default_data)
            # Row is guaranteed to exist now — load reads our default or a
            # racing thread's already-modified state.
            ctx = SessionContext.load(self.logs_conn, session_id)
            if ctx is None:
                ctx = SessionContext(session_id=session_id)
            self._session_contexts[session_id] = ctx
            return ctx

    def save_session_contexts(self) -> int:
        """Persist all cached SessionContexts to session_state.

        Called by the daemon autosave loop every
        AUTOSAVE_INTERVAL_SECONDS. Cheap — one row write per active
        session, idempotent. Mutations (fatigue, counters) accumulate
        in memory between autosaves; this is the timely persistence
        boundary. Returns the count of saves attempted.
        """
        n = 0
        # Serialize the shared logs_conn writes. This runs on the autosave
        # thread, outside the dispatch write path, so without write_lock it can
        # collide with a concurrent get_or_create_session on the same
        # connection ("cannot start a transaction within a transaction").
        with self.write_lock:
            for ctx in list(self._session_contexts.values()):
                try:
                    ctx.save(self.logs_conn)
                    n += 1
                except Exception as _e:
                    try:
                        self._log_error('session_context_autosave', _e,
                                        'persisting cached SessionContext')
                    except Exception:
                        pass
        return n

    def present_streams(self, exclude_session: str = '',
                        window_min: float = 30, limit: int = 5) -> list:
        """Streams of thought awake RIGHT NOW — the self-channel presence roster.

        Distinct from `live_sessions()`: that one is "recent meaningful work"
        (≥min_messages, survives week-long gaps, for the Frame's cross-session
        slots). `present_streams` is WALL-CLOCK "who is awake this moment" —
        sessions whose session_state row updated within the last `window_min`
        minutes, newest first, excluding the caller.

        Wall-clock is correct here: presence is real-time, not conversation-time,
        so it's exempt from the conversation_now() rule like other bookkeeping
        reads. See docs/BOOT-REIGNITION.md (presence at scale).

        Liveness is sourced from real-turn S0 traces (TraceDAL), NOT
        session_state.updated_at — the latter is bumped by the autosave loop for
        every cached session, so it falsely marks idle/stale sids "live" (and a
        window relaunched under a new sid would linger forever). Traces only
        record actual turns, so the signal is honest.

        Returns [{'session_id': str, 'updated_at': iso, 'focus': str}], newest
        first. `updated_at` is the last real-turn time; `focus` is that
        session's latest conversational turn — user_message OR assistant_message
        per trace_contract.CONVERSATIONAL_REF_TYPES, excluding the wake-envelope
        marker (raw — render layer trims it).
        """
        from .clock import iso_cutoff
        try:
            rows = self._trace_dal.active_sessions_by_turn(
                iso_cutoff(minutes=window_min),
                exclude_session=exclude_session, limit=limit)
            return [{'session_id': r['session_id'], 'updated_at': r['last_turn'],
                     'focus': r['focus']} for r in rows]
        except Exception as e:
            try:
                self._log_error('present_streams_query', e,
                                'window_min=%s limit=%d' % (window_min, limit))
            except Exception:
                pass
            return []

    def session_activity(self, session_id: str, msg_limit: int = 2) -> dict:
        """Per-session activity snapshot for self_peek — first/last turn and the
        last conversational messages, from real S0 traces (TraceDAL). Mirrors
        present_streams (wall-clock, presence-adjacent, read-only). Returns {} on
        error so a peek degrades gracefully rather than raising."""
        if not session_id:
            return {}
        try:
            return self._trace_dal.session_activity(session_id, msg_limit=msg_limit)
        except Exception as e:
            try:
                self._log_error('session_activity_query', e,
                                'session=%s' % (session_id or '')[:8])
            except Exception:
                pass
            return {}

    def stamp_boot_liveness(self, session_id: str) -> None:
        """Write ONE S0 heartbeat trace at boot so a freshly-booted stream is
        visible in presence IMMEDIATELY — before it takes its first turn.

        present_streams reads real-turn S0 traces, so without a boot trace a
        stream is invisible until its first hook_recall/Stop — which is exactly
        why two just-booted streams can't find each other at boot (the
        rendezvous gap, 2026-06-06). 'heartbeat' is already counted by
        active_sessions_by_turn, so the read side is unchanged. Loud on failure,
        never blocks boot."""
        if not session_id:
            return
        try:
            ctx = self.get_or_create_session(session_id)
            self._trace_dal.append(
                chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                ref_type='heartbeat', session_id=session_id,
                summary='boot — stream online')
        except Exception as e:
            try:
                self._log_error('stamp_boot_liveness', e,
                                'session=%s' % (session_id or '')[:8])
            except Exception:
                pass

    def live_sessions(self, limit: int = 5, min_messages: int = 5) -> list:
        """Return the most-recently-updated session_ids with at least
        `min_messages` messages on their SessionContext.

        Used by `live_session_activity` for the Frame brain-wide slots'
        cross-session view. "Live" is defined as "recent meaningful work",
        not wall-clock-active — survives a week-long gap (Tom's intent:
        last X sessions with ≥Y messages, parallel sessions all count).

        Reads from `session_state` rather than `_session_contexts` cache
        so it surfaces sessions whose ctx isn't currently in memory
        (post-daemon-restart, SessionEnd'd sessions, etc.).
        """
        try:
            return self._session_state.sessions_by_message_count(
                '_session_context', min_messages, limit)
        except Exception as e:
            try:
                self._log_error('live_sessions_query', e,
                                'min_messages=%d limit=%d' % (min_messages, limit))
            except Exception:
                pass
            return []

    def live_session_activity(self, node_ids=None,
                              limit: int = 5,
                              min_messages: int = 5) -> dict:
        """Aggregate per-session node_activity across the live sessions.

        Returns: {node_id: {'last_accessed': max_ts,
                            'activation': max_activation,
                            'access_count': sum_across_sessions,
                            'session_count': N}}

        - last_accessed: MAX across live sessions (most recent wins).
        - activation: MAX (a node "warm" in any one live session counts).
        - access_count: SUM (cross-session usage signal).
        - session_count: how many live sessions touched this node.

        Hybrid read path:
          (1) `self._session_contexts` cache — freshest in-memory state for
              sessions currently running.
          (2) `SessionContext.load(self.logs_conn, sid)` — persisted state
              for sessions not in cache (covers daemon restart + ended
              sessions still in the "live" window).
          Sessions that fail both paths are silently skipped.

        Args:
          node_ids: optional iterable to filter aggregation. None = all
              nodes touched by any live session.
          limit / min_messages: passed to `live_sessions`.
        """
        from .session_context import SessionContext
        filter_ids = set(node_ids) if node_ids is not None else None
        aggregated: dict = {}
        for sid in self.live_sessions(limit=limit, min_messages=min_messages):
            ctx = self._session_contexts.get(sid)
            if ctx is None:
                try:
                    ctx = SessionContext.load(self.logs_conn, sid)
                except Exception as e:
                    try:
                        self._log_error('live_session_activity_load', e,
                                        'session=%s' % sid[:8])
                    except Exception:
                        pass
                    ctx = None
            if ctx is None:
                continue
            for nid, rec in ctx.node_activity.items():
                if filter_ids is not None and nid not in filter_ids:
                    continue
                agg = aggregated.setdefault(nid, {
                    'last_accessed': '',
                    'activation': 0.0,
                    'access_count': 0,
                    'session_count': 0,
                })
                ts = str(rec.get('last_accessed', '') or '')
                if ts > agg['last_accessed']:
                    agg['last_accessed'] = ts
                act = float(rec.get('activation', 0.0))
                if act > agg['activation']:
                    agg['activation'] = act
                agg['access_count'] += int(rec.get('access_count', 0))
                agg['session_count'] += 1
        return aggregated

    def discard_session_context(self, session_id: str) -> None:
        """Save + drop a session's cached SessionContext. Called from
        SessionEnd hook for clean shutdown of that session.
        """
        ctx = self._session_contexts.pop(session_id, None)
        if ctx is not None:
            try:
                with self.write_lock:  # serialize the shared logs_conn write
                    ctx.save(self.logs_conn)
            except Exception as _e:
                try:
                    self._log_error('session_context_discard', _e,
                                    'final save before dropping cache entry')
                except Exception:
                    pass

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

        Activity counters (remember_count, message_count, edit_check_count,
        last_encode_at_message, boot_time) live on SessionContext now.
        Two writes to brain_meta are retained as deprecated singleton
        fallbacks (`session_id`, `boot_time`) for callers that haven't yet
        been threaded with session_id — see XXX flags in _log_error,
        brain_remember.remember(), and brain_assembly.pre_edit_check.
        """
        sid = session_id or uuid.uuid4().hex
        self._cached_session_id = sid
        # Persist SessionContext with fresh counters + segment state
        # (segment_id=0, segment_embeddings=[], segment_node_ids=[] are
        # the SessionContext defaults). Save immediately and replace the
        # cache entry — operator-visible reset semantics should land in
        # DB right away, not wait for autosave.
        from .session_context import SessionContext
        ctx = SessionContext(session_id=sid)
        ctx.boot_time = self.now()
        # Serialize the brain_meta (brain.db) + session_state (logs_conn)
        # writes under write_lock — same shared-connection race as
        # get_or_create_session when this runs outside the dispatch write
        # path (e.g. direct boot/test calls).
        with self.write_lock:
            # XXX deprecated singleton fallback for un-threaded callers (see
            # brain.session_id property + _log_error/_log_warning). C-refactor
            # threads session_id through every call site and drops this write.
            self._meta.set('session_id', sid)
            ctx.save(self.logs_conn)
            self._session_contexts[sid] = ctx

    def check_segment_boundary(self, query_embedding, session_id: str):
        """Detect if a new message represents a context/topic shift.

        Compares the query embedding against the centroid of the last N
        message embeddings (sliding window). If similarity drops below
        threshold, declares a new segment boundary.

        State lives on the cached SessionContext (segment_id /
        segment_embeddings / segment_node_ids). All reads and writes are
        in-memory; autosave persists every AUTOSAVE_INTERVAL_SECONDS.

        Pre-2026-05-17 this method wrote 1-3 set_config calls to brain_meta
        on every hook_recall, saturating brain.db locks under parallel
        sessions and surfacing the `another row available` cursor race.
        Moving the state to SessionContext eliminates that hot-path write
        entirely.

        Args:
            query_embedding: bytes blob from embedder.embed()
            session_id: session this prompt belongs to

        Returns:
            Dict with is_boundary, similarity, segment_id, segment_count
        """
        if not query_embedding or not session_id:
            return {'is_boundary': False, 'segment_id': 0}

        import base64

        ctx = self.get_or_create_session(session_id)
        current_seg = ctx.segment_id

        stored_b64 = list(ctx.segment_embeddings)
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
            ctx.segment_embeddings = stored_b64
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
            ctx.segment_id = new_seg
            ctx.segment_embeddings = [new_b64]
            ctx.segment_node_ids = []
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
            ctx.segment_embeddings = stored_b64
            return {
                'is_boundary': False,
                'similarity': round(sim, 3),
                'segment_id': current_seg,
                'segment_count': current_seg + 1,
            }

    def get_segment_node_ids(self, session_id: str):
        """Get node IDs created/accessed in the current segment for a session."""
        if not session_id:
            return []
        ctx = self.get_or_create_session(session_id)
        return list(ctx.segment_node_ids)

    def add_to_segment(self, node_id, session_id: str):
        """Add a node ID to the current segment's tracking list (in-memory)."""
        if not session_id or not node_id:
            return
        ctx = self.get_or_create_session(session_id)
        if node_id not in ctx.segment_node_ids:
            ctx.segment_node_ids.append(node_id)

    def record_remember(self, ctx):
        """Increment remember counter and mark last encode position.

        Takes a SessionContext; mutates in place. Caller is responsible
        for ctx.save() at the transaction boundary (turn end / handler
        exit). None ctx is a silent no-op.
        """
        if ctx is None:
            return
        ctx.remember_count += 1
        ctx.last_encode_at_message = ctx.message_count

    def record_message(self, ctx):
        """Increment message counter. Mutates ctx in place."""
        if ctx is None:
            return
        ctx.message_count += 1

    def record_edit_check(self, ctx):
        """Increment edit check counter. Mutates ctx in place."""
        if ctx is None:
            return
        ctx.edit_check_count += 1

    def get_encoding_heartbeat(self, ctx,
                               nudge_threshold: int = 8) -> Optional[Dict[str, Any]]:
        """Check if Claude should be nudged to encode learnings.

        Read-only against SessionContext counters.

        Args:
            ctx: SessionContext for the session being inspected
            nudge_threshold: Messages without encoding before nudging (default 8)
        """
        if ctx is None:
            return None
        msg_count = ctx.message_count
        remember_count = ctx.remember_count
        last_encode_at = ctx.last_encode_at_message

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
        return self._nodes.count()

    def _get_edge_count(self) -> int:
        """Get total edge count."""
        return self._graph.count_total()

    def _get_locked_count(self) -> int:
        """Get count of locked nodes."""
        return self._nodes.count_locked()

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
            self._maybe_commit()  # gated commit (was bare self.conn.commit())
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



    # get_surfaceable_dreams removed 2026-04-13 — dream system removed.

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
        for pkg in ['fastembed', 'sqlite_vec', 'onnxruntime']:
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
    # Must match plugin.json. Switch via set_embedder_config() (takes effect
    # on next boot) or by editing plugin.json + clearing brain_meta overrides.
    _EMBEDDER_DEFAULTS = {
        'model_name': 'nomic-ai/nomic-embed-text-v1.5-Q',
        'dim': 768,
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
        """Set a config value in brain_meta via DAL. Persists across restarts.

        Holds write_lock. set_config is a foreground self.conn write, and S2
        maintenance calls it lock-free on a pool thread (gating timestamps,
        failure counters, journals). Without the lock those writes interleave
        with a concurrent client brain_batch on the shared connection: the
        INSERT joins the batch's open transaction and BrainMetaDAL.set's
        commit_unless_batched then no-ops (in_batch=True), so the config write
        is lost if the batch rolls back. The lock serializes set_config against
        brain_batch (which resets in_batch before releasing the lock), so a
        non-owner thread never observes in_batch. write_lock is an RLock, so a
        caller already holding it (encoder dispatch, a batch) re-acquires safely.
        """
        with self.write_lock:
            self._meta.set(key, str(value))
        return {'key': key, 'value': value, 'updated_at': self.now()}

    # ══════════════════════════════════════════════════════════════════
    # Maintenance scheduling — brain owns the decision, daemon owns only
    # the polling cadence. Previously this logic sat in daemon_server._serve
    # which mixed transport and domain concerns. State lives in brain_meta
    # (s2_last_run_ts) so it survives daemon restarts cleanly; in-memory
    # state in the daemon lost it on every process death.
    # ══════════════════════════════════════════════════════════════════

    # Thresholds imported from brain_constants (contract-first; tunables
    # live next to other brain-wide config, not as class constants).

    def _maintenance_last_run_ts(self) -> float:
        """Epoch of the most recent maintenance run. Persisted in brain_meta."""
        raw = self.get_config('s2_last_run_ts') or '0'
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _maintenance_set_last_run_ts(self, ts: float) -> None:
        self.set_config('s2_last_run_ts', str(ts))

    def run_maintenance_if_due(self, last_activity_ts: float,
                               now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Run S2 maintenance iff idle and min-interval conditions are met.

        Args:
            last_activity_ts: Epoch seconds of last request the daemon saw.
            now: Epoch seconds (default: time.time()). Injectable for tests.

        Returns: S2 coordinator results dict when a run fired, None when
            not due. Caller (daemon) is responsible for serializing calls;
            this method is safe to call frequently — it no-ops cheaply when
            not due.
        """
        import time as _time
        from .brain_constants import (
            MAINTENANCE_IDLE_THRESHOLD_SECONDS,
            MAINTENANCE_MIN_INTERVAL_SECONDS,
            MAINTENANCE_FORCE_FIRE_SECONDS,
            MAINTENANCE_BOOT_GRACE_SECONDS,
        )
        now = now if now is not None else _time.time()

        # Boot grace gate (2026-05-08): never fire maintenance for the first
        # N seconds after daemon start. Without this, the previous logic
        # below (idle == inf when last_activity_ts is 0.0) made maintenance
        # fire on the very first daemon poll, blocking the first user
        # recall behind a long consolidation cycle.
        boot_age = now - getattr(self, '_boot_time', now)
        if boot_age < MAINTENANCE_BOOT_GRACE_SECONDS:
            return None

        # last_activity_ts == 0.0 means "daemon just booted, no user
        # prompts yet" — treat as infinitely idle so S2 can fire
        # immediately (subject to min_interval). Logging idle_seconds = inf
        # is clearer than "1777647345s" (epoch literal).
        if last_activity_ts is None or last_activity_ts == 0.0:
            idle_seconds = float('inf')
        else:
            idle_seconds = now - last_activity_ts
        last_run_ts = self._maintenance_last_run_ts()
        since_last_run = now - last_run_ts if last_run_ts else float('inf')

        # Min-interval gate is absolute — never fire more often than this.
        if since_last_run < MAINTENANCE_MIN_INTERVAL_SECONDS:
            return None
        # Idle gate is the normal trigger, BUT a stale-S2 safety valve
        # overrides it: if maintenance hasn't fired in FORCE_FIRE_SECONDS
        # the graph is going stale regardless of whether the user is at
        # the keyboard, so we fire anyway.
        if (idle_seconds < MAINTENANCE_IDLE_THRESHOLD_SECONDS and
                since_last_run < MAINTENANCE_FORCE_FIRE_SECONDS):
            return None

        # Mark the run BEFORE executing so concurrent callers (via second
        # poll) see the same timestamp and skip. The daemon's coarse
        # _s2_running lock is belt-and-suspenders.
        self._maintenance_set_last_run_ts(now)

        from servers.scales.s2.coordinator import run_s2
        results = run_s2(self)
        return {
            'ran_at_epoch': now,
            'idle_seconds': idle_seconds,
            'units': results,
        }


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

    def _log_error(self, source: str, error: Exception, context: str = '',
                   ctx=None):
        """Log an error to brain_logs.db + brain.log with rate limiting.

        Replaces silent `except: pass` blocks. Errors are stored in the logs DB
        and surfaced at boot via consciousness signals.

        Parallel-session attribution: callers in hot paths (hook_recall,
        S1 encode, MCP dispatch) can pass `ctx` for correct session
        attribution on the log row. Callers without ctx fall back to the
        deprecated `self.session_id` singleton — last-writer-wins under
        parallel sessions, but log attribution is informational only.
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

            _sid = (ctx.session_id if ctx is not None else self.session_id) or 'unknown'
            # Write to logs DB
            self._check_logs_db_size()
            self.logs_conn.execute('''
                INSERT INTO debug_log
                  (session_id, event_type, source, metadata, created_at)
                VALUES (?, 'error', ?, ?, ?)
            ''', (
                _sid,
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

    def _log_warning(self, source: str, message: str, context: str = '',
                     ctx=None):
        """Log a non-blocking warning to brain_logs.db + brain.log.

        For signals that are worth surfacing but aren't errors — empty-husk
        required aspect, auto-heal events, deprecated path used, etc. Different
        from _log_error: takes a string message rather than an Exception, and
        writes event_type='warning' so consumers can distinguish signal severity.

        Rate-limited via the same machinery as _log_error. `ctx` parameter
        works the same way (per-session attribution; falls back to singleton).
        """
        try:
            # Rate limit check — compute fingerprint
            fingerprint = '%s:warning:%s' % (source, message[:100])
            if self._check_rate_limit(source, fingerprint):
                return  # suppressed

            _sid = (ctx.session_id if ctx is not None else self.session_id) or 'unknown'
            # Write to logs DB
            self._check_logs_db_size()
            self.logs_conn.execute('''
                INSERT INTO debug_log
                  (session_id, event_type, source, metadata, created_at)
                VALUES (?, 'warning', ?, ?, ?)
            ''', (
                _sid,
                source,
                json.dumps({
                    'message': message,
                    'context': context,
                }),
                self.now()
            ))

            # Write to human-readable log file
            self._write_to_file_log('WARNING', source, message, context)
        except Exception:
            # Last resort — can't even log. Print to stderr.
            print('brain: warning in %s: %s (context: %s)' % (source, message, context),
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
            # Bool first — bool is a subclass of int in Python, so the
            # isinstance(default_val, (int, float)) check below would catch
            # bool defaults and try to parse "true"/"false" as int, throwing
            # a benign-but-noisy error every read. Handle bool explicitly.
            if isinstance(default_val, bool):
                if val in ('true', '1', 'yes', 'on', True):
                    return True
                if val in ('false', '0', 'no', 'off', False):
                    return False
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




    def warm_up(self) -> Dict[str, Any]:
        """First-call-only work, paid at boot instead of on the first prompt.

        Daemon boot announces ready and binds the socket immediately. This
        method runs in a background thread (see `BrainDaemon._run_warmup`)
        so the user never waits on it. By the time their first prompt
        arrives — they have to open Claude Code, see the splash, type —
        warmup is typically already done.

        What's covered:

          1. SQLite mmap warmup. The recall scan reads every non-archived
             primary embedding row from `node_enrichments`. The first such
             read faults ~hundreds of MB of pages from disk into RSS (a
             ~2 GB transient was observed at brain.db=218 MB pre-fix).
             Subsequent reads find pages already in the OS page cache.
             We pre-touch every blob byte so SQLite actually decodes the
             BLOB column, not just the row metadata.

          2. Structural degree cache. Used by the fatigue formula in the
             cosine loop. A single SQL UNION ALL across edges × edge_relations.
             Pre-built here so the recall hot path skips the lazy guard.

          3. Anthropic SDK + httpx pool. The S1 surface step calls Haiku
             on every recall via the Anthropic Python SDK. The first call
             in any process pays import (~350ms) + httpx pool init + TLS
             handshake to api.anthropic.com. We pay that here, store the
             warm client on `self.anthropic_client`, and `surface.py`
             reuses it across the daemon's ThreadPoolExecutor workers
             (httpx.Client is documented thread-safe).

             Two sub-phases: (a) eager import + client construction;
             (b) `models.retrieve` — free, warms TLS + httpx pool + DNS.
             That's enough.

             A 1-token `messages.create` was tried (2026-05-09) as a
             "Haiku route warmup". Removed: phase-timer data showed the
             daemon's surface_haiku phase is a stable ~5s whether the
             route is warmed or not. The ping wasn't buying real work.

        Note: an in-memory edge-text embedding pre-warm was tried here
        (2026-05-09, removed same day). It populated `_DESC_VEC_CACHE`
        with 1500 highest-weight edge texts so the first surface_spread
        phase wouldn't pay fastembed cost. Removed because it treated
        the symptom (cold cache on restart) instead of the underlying
        asymmetry: nodes have stored embeddings (one-time write, free
        reads forever), but edges did not — every spread call recomputed
        them. The fix was to give edges first-class stored embeddings
        like nodes (Option B in the design discussion); see
        `_compose_enriched_edge_text` in surface_contract.py and the
        `embedding` column on `edge_relations`.

        Note: cache_control / prompt-caching warmup is NOT here. The
        surface system block is ~2390 tokens, below Haiku 4.5's 4096-token
        minimum cacheable prefix. A cache warmup call would silently
        no-op. If the surface prefix grows past 4096, add it back.

        Idempotent and safe to race against a concurrent first recall —
        the recall hot path's lazy paths are guards, not initializers, so
        if warmup hasn't finished when the prompt arrives, recall will
        finish the work itself and warmup becomes a no-op on its second
        try. Last-writer-wins on dict assignment is fine because the data
        is identical. The Anthropic phase is similarly graceful: surface.py
        falls back to constructing its own client if `anthropic_client`
        isn't set yet.

        Failures are caught and logged. A warmup failure must never
        affect the daemon — falling through to lazy paths is acceptable
        degradation.

        Returns a dict of timings/sizes for daemon.log telemetry.
        """
        import time as _time
        t0 = _time.monotonic()
        timings: Dict[str, Any] = {}

        # 1. Embeddings mmap warmup.
        try:
            t = _time.monotonic()
            active_model = embedder.stats.get('model_name') or None
            rows = self._vec_dal.get_all_with_context(
                exclude_archived=True, model=active_model)
            # Touch every blob's bytes so SQLite actually decodes the BLOB
            # column. Iterating the row list alone may not — some bindings
            # defer BLOB materialization until you read the bytes.
            total_bytes = 0
            for r in rows:
                blob = r.get('embedding')
                if blob:
                    total_bytes += len(blob)
            timings['embeddings_loaded'] = len(rows)
            timings['embeddings_bytes'] = total_bytes
            timings['embeddings_ms'] = int((_time.monotonic() - t) * 1000)
        except Exception as e:
            self._log_error(
                'warmup_embeddings', e, 'mmap warmup failed')
            timings['embeddings_error'] = str(e)

        # 2. Structural degree cache.
        try:
            t = _time.monotonic()
            self._ensure_structural_degree_cache()
            timings['degree_cache_size'] = len(
                getattr(self, '_structural_degree_cache', {}) or {})
            timings['degree_cache_ms'] = int((_time.monotonic() - t) * 1000)
        except Exception as e:
            self._log_error(
                'warmup_degree_cache', e, 'degree cache build failed')
            timings['degree_cache_error'] = str(e)

        # 3. Anthropic SDK + Haiku connection.
        try:
            t = _time.monotonic()
            # Eager import + lifetime client. httpx.Client (under the hood)
            # is documented thread-safe; one instance shared across the
            # daemon's ThreadPoolExecutor workers reuses the connection
            # pool instead of every recall thread building its own.
            #
            # load_env() resolves ANTHROPIC_API_KEY from the four
            # supported sources (runtime cache, legacy in-repo .env, real
            # shell env). The daemon is launched by launchd with no shell,
            # so without this call os.environ has no API key and the
            # client construction would itself succeed but fail on first
            # use with "Could not resolve authentication method." Other
            # callers (encoder, S2 units) follow the same import-and-call
            # pattern — see scales/s1/encode.py and scales/s2/base.py.
            from .scales.dispatch import load_env
            load_env()
            import anthropic
            from .scales.s1.surface_contract import SURFACE_MODEL
            self.anthropic_client = anthropic.Anthropic()
            timings['anthropic_client_ms'] = int(
                (_time.monotonic() - t) * 1000)

            # Free warmup: warms TLS handshake + httpx connection pool +
            # DNS to api.anthropic.com. Doesn't bill.
            t = _time.monotonic()
            self.anthropic_client.models.retrieve(SURFACE_MODEL)
            timings['anthropic_models_retrieve_ms'] = int(
                (_time.monotonic() - t) * 1000)
            # No Haiku route ping. The ping was here briefly (2026-05-09)
            # but phase-timer data showed surface_haiku is a stable ~5s
            # whether the route is "warmed" or not — the ping wasn't
            # buying real latency. models.retrieve() warms TLS + pool +
            # DNS, which is the only first-call-only work that matters.
        except Exception as e:
            # SDK warmup failure must not crash the daemon — surface.py's
            # graceful-fallback path will construct a fresh client on first
            # call and pay the cold-start tax. Same as pre-warmup behavior.
            self._log_error(
                'warmup_anthropic', e, 'Anthropic SDK warmup failed')
            timings['anthropic_error'] = str(e)
            self.anthropic_client = None

        # (Edge-text embedding pre-warm was here briefly on 2026-05-09;
        # removed when edges got first-class stored embeddings — see
        # the docstring above.)

        timings['total_ms'] = int((_time.monotonic() - t0) * 1000)
        return timings

    def _maybe_commit(self):
        """Commit self.conn unless we're inside a brain_batch transaction.

        Write methods on Brain (remember, revise, connect, archive_node)
        used to call self.conn.commit() directly. Inside a brain_batch
        with many ops, this meant N separate commits hitting the WAL
        writer slot — bad for parallel-session contention, and there was
        no rollback semantic if op #37 failed.

        Now those methods call _maybe_commit() instead. It defers to the
        connection's batch state (self.conn.in_batch): a no-op inside
        _handle_brain_batch (which owns one BEGIN IMMEDIATE / COMMIT with
        ROLLBACK on failure), a real commit on the normal single-op path.
        Same gate the DAL writers use — commit_unless_batched is the one
        source of truth.
        """
        commit_unless_batched(self.conn)

    def save(self, backup: bool = False):
        """
        Commit pending changes and optionally back up database.

        Holds write_lock around the self.conn commit. save() is called
        lock-free from the daemon's S2 idle-maintenance path (_run_idle_
        maintenance) on a pool thread — the SAME pool that handles client
        commands. Without the lock, its commit() can land mid-flight on a
        concurrent client brain_batch (BEGIN IMMEDIATE + many writes on the
        shared self.conn), committing the batch's PARTIAL transaction and
        breaking its all-or-nothing atomicity. write_lock serializes save
        against brain_batch; it's an RLock, so the primary autosave path
        (daemon_server, which already holds write_lock before calling save)
        re-acquires safely. logs_conn is a separate DB — no coordination with
        the foreground write lock is needed.

        Args:
            backup: If True, create a backup copy
        """
        with self.write_lock:
            self.conn.commit()  # commit-ok: explicit durability point (save/autosave)
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
        """Commit, close all database connections, and remove from singleton cache.

        Per-connection close is wrapped so a failure on one doesn't skip the
        others. Errors are logged via `_log_error` (no silent except: pass)
        with distinct origins per connection / phase so post-mortem can tell
        commit failures from close failures.
        """
        try:
            self.conn.commit()  # commit-ok: final commit on shutdown
        except Exception as e:
            self._log_error('brain_close_commit', e, 'primary conn final commit')
        try:
            self.conn.close()
        except Exception as e:
            self._log_error('brain_close', e, 'primary conn close')

        # Background-writer connection (temporal, access marks, hebbian).
        try:
            self.conn_bg_writer.commit()
        except Exception as e:
            self._log_error('bg_writer_close_commit', e,
                            'bg_writer final commit')
        try:
            self.conn_bg_writer.close()
        except Exception as e:
            self._log_error('bg_writer_close', e, 'bg_writer close')

        # logs_conn close — logged via stderr fallback because _log_error
        # itself writes to logs_conn. Closing it last keeps prior _log_error
        # calls in this method functional.
        try:
            self.logs_conn.commit()
            self.logs_conn.close()
        except Exception as e:
            import sys as _sys
            print('[brain] logs_conn close failed: %s' % e, file=_sys.stderr)
        # Remove from singleton registry if present
        canonical = os.path.realpath(self.db_path)
        with Brain._lock:
            if Brain._instances.get(canonical) is self:
                del Brain._instances[canonical]
