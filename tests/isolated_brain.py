"""Isolated brain environment for tests and evals.

Copies production databases to a temp directory so tests never touch live data.
All reads and writes go to the isolated copy. Cleaned up on exit.

Usage:
    from tests.isolated_brain import IsolatedBrain

    with IsolatedBrain() as env:
        # env.brain — Brain instance (isolated copy)
        # env.db_dir — temp directory with copied DBs
        # env.brain_db — path to isolated brain.db
        # env.logs_db — path to isolated brain_logs.db
        result = env.brain.recall(query="test", limit=5)

    # Or keep the temp dir for inspection:
    with IsolatedBrain(cleanup=False) as env:
        ...
        print("Inspect at:", env.db_dir)

    # Or with a dispatch function for tool calls:
    with IsolatedBrain() as env:
        result = env.dispatch("recall", {"query": "test", "limit": 5})
"""
import os
import shutil
import tempfile
import sqlite3


def _default_production_dir():
    """Resolve production brain DB directory."""
    env_dir = os.environ.get('BRAIN_DB_DIR', '')
    if env_dir and os.path.exists(os.path.join(env_dir, 'brain.db')):
        return env_dir
    default = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')
    if os.path.exists(os.path.join(default, 'brain.db')):
        return default
    return None


class IsolatedBrain:
    """Context manager that creates an isolated brain environment.

    Copies production brain.db and brain_logs.db to a temp directory.
    Creates a Brain instance pointing at the copies.
    All operations are isolated — production is never touched.
    """

    def __init__(self, production_dir=None, cleanup=True, load_env=True):
        """
        Args:
            production_dir: path to production brain DBs (auto-detected if None)
            cleanup: delete temp dir on exit (False to keep for inspection)
            load_env: load .env file for API keys (needed for Haiku/Sonnet calls)
        """
        self.production_dir = production_dir or _default_production_dir()
        self.cleanup = cleanup
        self.load_env = load_env
        self.db_dir = None
        self.brain = None
        self.brain_db = None
        self.logs_db = None

    def __enter__(self):
        if not self.production_dir:
            raise RuntimeError("Cannot find production brain.db. Set BRAIN_DB_DIR or pass production_dir.")

        # Create temp directory
        self.db_dir = tempfile.mkdtemp(prefix='brain_test_')

        # Copy databases
        src_brain = os.path.join(self.production_dir, 'brain.db')
        src_logs = os.path.join(self.production_dir, 'brain_logs.db')

        self.brain_db = os.path.join(self.db_dir, 'brain.db')
        self.logs_db = os.path.join(self.db_dir, 'brain_logs.db')

        # Consistent working clones via the online backup API — safe against
        # a live daemon mid-write, and each clone is one self-contained .db
        # (committed WAL tail included), so no -wal/-shm juggling. A raw
        # copy2 of db + -wal + -shm captures the three files at different
        # instants and can tear.
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from servers.db_backends import current as db_backend
        db_backend.snapshot_to(src_brain, self.brain_db)
        if os.path.exists(src_logs):
            db_backend.snapshot_to(src_logs, self.logs_db)

        # Load .env for API keys
        if self.load_env:
            _load_env()

        # Set environment so Brain resolves correctly. Both BRAIN_DB_DIR and
        # ASPECTS_JSON_PATH are saved here and restored in __exit__ so the
        # process env returns to its pre-context state — the temp dir is
        # rmtree'd on exit, so a leaked override would point at a deleted dir.
        self._orig_brain_db_dir = os.environ.get('BRAIN_DB_DIR')
        self._orig_aspects_json_path = os.environ.get('ASPECTS_JSON_PATH')
        self._orig_brain_tmp_dir = os.environ.get('BRAIN_TMP_DIR')

        # Aspect isolation. If the caller already pinned ASPECTS_JSON_PATH
        # (e.g. run_aspect_cycles_on_clone.py points it at its own work file),
        # honor it — they own the aspects location. Otherwise pin it into our
        # temp dir so the BRAIN_DB_DIR fallback in aspects_json_path() can't
        # leak heals into the live file, and seed it from the operator's grown
        # aspects (not a fresh repo seed) for production-faithful fidelity.
        #
        # The failure-prone copy runs BEFORE any os.environ mutation: if it
        # raises, __enter__ propagates and Python never calls __exit__, so a
        # half-set environment would leak. Env assignments (below) can't raise.
        pin_aspects = self._orig_aspects_json_path is None
        if pin_aspects:
            src_aspects = os.path.join(self.production_dir, 'aspects_v1.json')
            if os.path.exists(src_aspects):
                shutil.copy2(src_aspects,
                             os.path.join(self.db_dir, 'aspects_v1.json'))

        os.environ['BRAIN_DB_DIR'] = self.db_dir
        if pin_aspects:
            os.environ['ASPECTS_JSON_PATH'] = os.path.join(
                self.db_dir, 'aspects_v1.json')
        # Route brain's ephemeral /tmp files into this isolated run's dir so
        # concurrent eval/test processes don't collide on fixed filenames and
        # the files are cleaned up with the temp dir (see daemon_config.brain_tmp_dir).
        os.environ['BRAIN_TMP_DIR'] = self.db_dir

        # Create isolated Brain instance
        from servers.brain import Brain
        self.brain = Brain(self.brain_db)

        # Auto-drain embeddings after writes — same rationale as in
        # BrainTestBase._patch_writes_to_auto_drain. The production embed
        # queue worker fires every 5s; tests run faster, so a remember()
        # → recall() in the same scope returns nothing. Wrap the write
        # APIs to drain synchronously, restoring the pre-deferral
        # contract that test code is written against.
        from servers import embed_queue, recall_write_queue

        def _wrap(name):
            orig = getattr(self.brain, name, None)
            if orig is None or getattr(orig, '_drain_wrapped', False):
                return
            brain = self.brain

            def w(*args, **kwargs):
                result = orig(*args, **kwargs)
                try:
                    embed_queue._drain_once(brain)
                except Exception as e:
                    import sys
                    print('[isolated_brain] embed drain after %s '
                          'failed: %s' % (name, e), file=sys.stderr)
                try:
                    recall_write_queue.drain_once(brain)
                except Exception as e:
                    import sys
                    print('[isolated_brain] recall_write drain after %s '
                          'failed: %s' % (name, e), file=sys.stderr)
                return result
            w._drain_wrapped = True
            w.__wrapped__ = orig
            setattr(self.brain, name, w)

        # Wrap recall too (Phase 5, 2026-05-18): mark_accessed + hebbian
        # enqueue happens on recall; tests asserting on access_count need
        # the bg_writer drain to fire synchronously.
        for name in ('remember', 'revise', 'recall'):
            _wrap(name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.brain:
            try:
                self.brain.save()
                self.brain.close()
            except Exception:
                pass
        # Restore env overrides so they don't leak to later code (the temp dir
        # we pointed them at is about to be rmtree'd).
        if hasattr(self, '_orig_brain_db_dir'):
            if self._orig_brain_db_dir is None:
                os.environ.pop('BRAIN_DB_DIR', None)
            else:
                os.environ['BRAIN_DB_DIR'] = self._orig_brain_db_dir
        if hasattr(self, '_orig_aspects_json_path'):
            if self._orig_aspects_json_path is None:
                os.environ.pop('ASPECTS_JSON_PATH', None)
            else:
                os.environ['ASPECTS_JSON_PATH'] = self._orig_aspects_json_path
        if hasattr(self, '_orig_brain_tmp_dir'):
            if self._orig_brain_tmp_dir is None:
                os.environ.pop('BRAIN_TMP_DIR', None)
            else:
                os.environ['BRAIN_TMP_DIR'] = self._orig_brain_tmp_dir
        if self.cleanup and self.db_dir:
            shutil.rmtree(self.db_dir, ignore_errors=True)
        return False

    def dispatch(self, cmd, args=None):
        """Execute a daemon command against the isolated brain.

        Routes through `dispatch_command` — the same execution chokepoint the
        daemon and the encoder closure use, so eval/test runs exercise the real
        arg-contract validation and per-op loudness path. No TCP, no daemon.

        No write lock is taken: an isolated brain is single-threaded by
        construction, with no daemon, autosave or embed_queue to serialize
        against.
        """
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, cmd, args or {}, [])

    def node_count(self):
        """Quick check: how many non-archived nodes."""
        row = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()
        return row[0] if row else 0

    def recall_log_count(self):
        """How many recall_log entries (to verify test isolation)."""
        row = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM recall_log").fetchone()
        return row[0] if row else 0


def _load_env():
    """Load .env for API keys."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
