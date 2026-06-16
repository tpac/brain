"""Regression: IsolatedBrain must NOT leak aspect-registry heals into the
live $BRAIN_DB_DIR/aspects_v1.json.

Bug (observed 2026-06-16): ASPECTS_JSON_PATH was a module-level constant
resolved at IMPORT time — before IsolatedBrain set BRAIN_DB_DIR in __enter__.
So a top-level `from servers.scales.s2... import ...` froze the path at the
LIVE user-dir file, and ensure_aspects_user_copy (run at Brain.__init__ via
AspectRegistry._load) healed the live file even though the DB was isolated.

Fix: resolve the aspects path at CALL time (aspects_json_path()), so a later
os.environ change takes effect. IsolatedBrain additionally pins
ASPECTS_JSON_PATH to its temp dir.

This test asserts the live file is byte-for-byte unchanged across an
IsolatedBrain session that builds a Frame (which loads the registry and runs
the seed/self-heal path).
"""
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s2.aspect_contract import aspects_json_path  # noqa: E402
from servers.scales.s1.frame import build_frame  # noqa: E402 — top-level import is the repro trigger
from tests.isolated_brain import IsolatedBrain, _default_production_dir  # noqa: E402


class TestAspectsPathIsolation(unittest.TestCase):
    def test_isolated_brain_does_not_touch_live_aspects_file(self):
        prod_dir = _default_production_dir()
        if not prod_dir:
            self.skipTest("no production brain.db to isolate")
        live_path = os.path.join(prod_dir, 'aspects_v1.json')
        if not os.path.exists(live_path):
            self.skipTest("no live aspects_v1.json present")

        with open(live_path, 'rb') as f:
            before = f.read()
        before_mtime = os.path.getmtime(live_path)

        with IsolatedBrain() as env:
            # The aspects path must now resolve INTO the temp dir, not the live file.
            self.assertNotEqual(
                os.path.abspath(aspects_json_path()),
                os.path.abspath(live_path),
                "aspects_json_path() still points at the live file inside IsolatedBrain")
            # Exercise the registry load + seed/heal path.
            build_frame(env.brain, session_id='isolation-test')

        with open(live_path, 'rb') as f:
            after = f.read()
        self.assertEqual(before, after,
                         "IsolatedBrain modified the live aspects_v1.json content")
        self.assertEqual(before_mtime, os.path.getmtime(live_path),
                         "IsolatedBrain touched the live aspects_v1.json mtime")

    def test_isolated_brain_honors_preset_aspects_override(self):
        """A caller that pins ASPECTS_JSON_PATH before entering IsolatedBrain
        owns the aspects location — IsolatedBrain must NOT clobber it.

        run_aspect_cycles_on_clone.py relies on this: it points
        ASPECTS_JSON_PATH at its own work file so the encoder classifies into
        that file. If IsolatedBrain overwrites the env var, the eval reads back
        an unchanged file and silently reports wrong results.
        """
        import tempfile
        if not _default_production_dir():
            self.skipTest("no production brain.db to isolate")
        work = os.path.join(tempfile.mkdtemp(prefix='aspects_override_'),
                            'aspects_v1.json')
        os.environ['ASPECTS_JSON_PATH'] = work
        try:
            with IsolatedBrain():
                self.assertEqual(os.environ.get('ASPECTS_JSON_PATH'), work,
                                 "IsolatedBrain clobbered the caller's "
                                 "ASPECTS_JSON_PATH override")
                self.assertEqual(os.path.abspath(aspects_json_path()),
                                 os.path.abspath(work))
            # Restored to the caller's value on exit (not popped).
            self.assertEqual(os.environ.get('ASPECTS_JSON_PATH'), work)
        finally:
            os.environ.pop('ASPECTS_JSON_PATH', None)
            shutil.rmtree(os.path.dirname(work), ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
