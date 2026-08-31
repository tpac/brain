"""Contract tests for daemon instance keying (BRAIN_INSTANCE).

Locks the multi-entity boundary: with BRAIN_INSTANCE unset every rendezvous
path is byte-identical to the historical form (production unaffected); with it
set, every path and the launchd label carry the key, so an eval entity can
never acquire production's daemon/maintenance locks or target its launchd job.
Also locks the deterministic background-loop triggers (s2_force /
drain_embeddings) that entity harnesses call instead of re-implementing the
daemon's loops.
"""
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import daemon_config as dc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_FUNCS = [dc.get_socket_path, dc.get_pid_path, dc.get_lock_path,
              dc.get_startup_lock_path, dc.get_status_path,
              dc.get_maintenance_path, dc.get_recovery_state_path]


class TestInstancePaths(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("BRAIN_INSTANCE")
        os.environ.pop("BRAIN_INSTANCE", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("BRAIN_INSTANCE", None)
        else:
            os.environ["BRAIN_INSTANCE"] = self._saved

    def test_no_instance_paths_are_legacy_exact(self):
        uid = os.getuid()
        expected = ["/tmp/brain-daemon-%d.sock" % uid,
                    "/tmp/brain-daemon-%d.pid" % uid,
                    "/tmp/brain-daemon-%d.lock" % uid,
                    "/tmp/brain-startup-%d.lock" % uid,
                    "/tmp/brain-status-%d.json" % uid,
                    "/tmp/brain-maintenance-%d.lock" % uid,
                    "/tmp/brain-recovery-%d.json" % uid]
        self.assertEqual([f() for f in PATH_FUNCS], expected,
                         "production paths must be byte-identical to the "
                         "pre-instance form")

    def test_instance_suffixes_every_path(self):
        os.environ["BRAIN_INSTANCE"] = "evalx"
        base = {f() for f in PATH_FUNCS}
        for p in base:
            self.assertIn("-evalx.", p, p)
        os.environ.pop("BRAIN_INSTANCE")
        self.assertTrue(base.isdisjoint({f() for f in PATH_FUNCS}),
                        "instance paths must never collide with production's")


class TestInstanceImportGuard(unittest.TestCase):
    """The guard fires at import — needs a fresh interpreter per case."""

    def _import_probe(self, env_extra, code="import servers.daemon_config"):
        env = {k: v for k, v in os.environ.items()
               if k not in ("BRAIN_INSTANCE", "BRAIN_DAEMON_PORT",
                            "BRAIN_DB_DIR")}
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, %r); %s"
             % (REPO, code)],
            env=env, capture_output=True, text=True, cwd=REPO)

    def test_instance_without_port_and_db_refuses(self):
        r = self._import_probe({"BRAIN_INSTANCE": "evalx"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires explicit", r.stderr)

    def test_bad_instance_name_refuses(self):
        r = self._import_probe({"BRAIN_INSTANCE": "eval x",
                                "BRAIN_DAEMON_PORT": "47999",
                                "BRAIN_DB_DIR": "/tmp/x"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("1-32 chars", r.stderr)

    def test_instance_with_port_and_db_imports_and_keys_label(self):
        r = self._import_probe(
            {"BRAIN_INSTANCE": "evalx", "BRAIN_DAEMON_PORT": "47999",
             "BRAIN_DB_DIR": "/tmp/x"},
            code="from servers.daemon_launch import LAUNCHD_LABEL; "
                 "print(LAUNCHD_LABEL)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "com.brain.daemon.evalx")

    def test_production_label_unchanged(self):
        r = self._import_probe(
            {}, code="from servers.daemon_launch import LAUNCHD_LABEL; "
                     "print(LAUNCHD_LABEL)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "com.brain.daemon")


class TestBackgroundLoopTriggers(unittest.TestCase):
    def test_commands_registered_lock_free(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        for cmd in ("s2_force", "drain_embeddings"):
            self.assertIn(cmd, COMMAND_TABLE)
            # Both own their serialization internally (run_s2 single-flight;
            # drain batches take brain.write_lock) — holding the dispatch
            # exclusive lock for a multi-minute S2 cycle would deadlock S2's
            # own encoder dispatch.
            self.assertFalse(COMMAND_TABLE[cmd].is_write, cmd)

    def test_drain_now_is_the_worker_tick(self):
        from servers import embed_queue
        self.assertTrue(callable(embed_queue.drain_now))


if __name__ == "__main__":
    unittest.main()
