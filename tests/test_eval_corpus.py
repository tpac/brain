"""Contract tests for the Frozen Corpus helpers (eval/longmem/corpus.py).

Pure functions — no Brain, no LLM. Locks the content-addressing and S2-delta
aggregation that the two-stage eval (build_corpus → sweep) depends on.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.longmem.corpus import (
    corpus_config_hash, source_token, summarize_s2_deltas, merge_s2_totals,
)


class TestCorpusConfigHash(unittest.TestCase):
    def test_deterministic(self):
        cfg = {"s1e": "active", "ingest_surface": "active", "s2_every_n": 2,
               "oracle": "longmemeval_oracle.json", "qids": ["a", "b"]}
        self.assertEqual(corpus_config_hash(cfg), corpus_config_hash(dict(cfg)))

    def test_key_order_independent(self):
        a = {"s1e": "active", "s2_every_n": 2, "qids": ["a"]}
        b = {"qids": ["a"], "s2_every_n": 2, "s1e": "active"}
        self.assertEqual(corpus_config_hash(a), corpus_config_hash(b),
                         "dict key order must not change the hash (sort_keys)")

    def test_sensitive_to_inputs(self):
        base = {"s1e": "active", "s2_every_n": 2, "qids": ["a", "b"]}
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "s1e": "file:deadbeef"}))
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "s2_every_n": 3}))
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "qids": ["a", "c"]}))


class TestVariantAddressing(unittest.TestCase):
    """Locks the launcher-parity guard: builds refuse to run in an unpinned
    shell, and a non-baseline variant joins the content address (a baseline
    build keeps every pre-fix corpus hash)."""

    def setUp(self):
        from eval.longmem.build_corpus import (
            _effective_variant_pins, _address_variants)
        self._pins = _effective_variant_pins
        self._address = _address_variants
        self._saved = {k: os.environ.get(k) for k in
                       ("BRAIN_SURFACE_VARIANT", "BRAIN_RECALL_VARIANT")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_unpinned_shell_refuses(self):
        os.environ.pop("BRAIN_SURFACE_VARIANT", None)
        os.environ["BRAIN_RECALL_VARIANT"] = "laf_v1"
        with self.assertRaises(SystemExit):
            self._pins("active")

    def test_baseline_pins_leave_hash_stable(self):
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"
        os.environ["BRAIN_RECALL_VARIANT"] = "laf_v1"
        cfg = {"s1e": "active", "qids": ["a"]}
        before = corpus_config_hash(cfg)
        self._address(cfg, self._pins("active"))
        self.assertEqual(before, corpus_config_hash(cfg),
                         "baseline variants must not invalidate pre-fix corpora")

    def test_nonbaseline_variant_changes_hash(self):
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"
        os.environ["BRAIN_RECALL_VARIANT"] = "baseline"
        cfg = {"s1e": "active", "qids": ["a"]}
        before = corpus_config_hash(cfg)
        self._address(cfg, self._pins("active"))
        self.assertEqual(cfg.get("recall_variant"), "baseline")
        self.assertNotEqual(before, corpus_config_hash(cfg))

    def test_ingest_surface_override_forces_agentic(self):
        os.environ["BRAIN_SURFACE_VARIANT"] = "v4"
        os.environ["BRAIN_RECALL_VARIANT"] = "laf_v1"
        pins = self._pins("eval/surface_v12_prompt.txt")
        self.assertEqual(pins["surface_variant"], "v5_agentic",
                         "a surface override runs the agentic loop regardless "
                         "of the shell pin")


class TestSourceToken(unittest.TestCase):
    def test_active(self):
        self.assertEqual(source_token("active"), "active")
        self.assertEqual(source_token(None), "active")

    def test_missing_path(self):
        tok = source_token("/no/such/prompt_file.txt")
        self.assertTrue(tok.startswith("missing:"))

    def test_file_content_hash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("prompt body v1")
            p = f.name
        try:
            tok = source_token(p)
            self.assertTrue(tok.startswith("file:"))
            # Same content → same token; different content → different token.
            self.assertEqual(tok, source_token(p))
            with open(p, "w") as f:
                f.write("prompt body v2")
            self.assertNotEqual(tok, source_token(p))
        finally:
            os.unlink(p)


class TestSummarizeS2Deltas(unittest.TestCase):
    def test_aggregates_fires_work_actions(self):
        # Deltas are run_s2() returns: units nested, elapsed_ms alongside.
        deltas = [
            {"units": {"community_detection": {"actions": 2},
                       "healer": {"actions": 0}}, "elapsed_ms": 10},
            {"units": {"community_detection": {"actions": 1},
                       "healer": {"skipped": "no gaps"}}, "elapsed_ms": 5},
        ]
        s = summarize_s2_deltas(deltas)
        self.assertEqual(s["community_detection"]["fires"], 2)
        self.assertEqual(s["community_detection"]["did_work"], 2)
        self.assertEqual(s["community_detection"]["actions"], 3)
        self.assertEqual(s["healer"]["fires"], 2)
        self.assertEqual(s["healer"]["did_work"], 0)
        self.assertEqual(s["healer"]["skipped"], 1)

    def test_counts_errors_and_samples(self):
        deltas = [{"units": {"consolidation": {"error": "cannot start a transaction"}}}]
        s = summarize_s2_deltas(deltas)
        self.assertEqual(s["consolidation"]["errors"], 1)
        self.assertEqual(len(s["consolidation"]["sample_errors"]), 1)
        # An errored fire is neither did_work nor a successful action.
        self.assertEqual(s["consolidation"]["did_work"], 0)
        self.assertEqual(s["consolidation"]["actions"], 0)

    def test_ignores_elapsed_and_nondict(self):
        # elapsed_ms and a skipped cycle carry no units; non-dict unit results drop.
        s = summarize_s2_deltas([
            {"units": {}, "skipped": "already running"},
            {"units": {"x": "notadict"}, "elapsed_ms": 99},
        ])
        self.assertEqual(s, {})

    def test_empty(self):
        self.assertEqual(summarize_s2_deltas([]), {})
        self.assertEqual(summarize_s2_deltas(None), {})


class TestMergeS2Totals(unittest.TestCase):
    def test_sums_across_items(self):
        items = [
            {"s2_delta": {"healer": {"fires": 2, "did_work": 1, "actions": 1,
                                     "errors": 0, "skipped": 1}}},
            {"s2_delta": {"healer": {"fires": 3, "did_work": 2, "actions": 4,
                                     "errors": 1, "skipped": 0}}},
        ]
        t = merge_s2_totals(items)
        self.assertEqual(t["healer"]["fires"], 5)
        self.assertEqual(t["healer"]["actions"], 5)
        self.assertEqual(t["healer"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
