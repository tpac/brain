"""Tests for servers/scales/s1/scouts/contract.py — scout I/O contract.

Covers:
- SCOUT_NAMES / interaction_name() helpers
- build_shared_prefix(): block shape, cache breakpoint placement,
  byte-identical output across calls (caching invariant)
- validate_scout_output(): all branches — ok, missing envelope,
  wrong scout, missing candidate fields, soft truncation, coerced types
- format_scout_report_for_s1s(): all four scouts, empty and populated
- Cache-layout contract: per-scout task moved from user content to
  system prompt (1h TTL); assemble_call_content removed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s1.scouts import contract as sc


class TestScoutRegistry(unittest.TestCase):

    def test_scout_names_fixed_set(self):
        self.assertEqual(
            set(sc.SCOUT_NAMES),
            {"quote", "temporal", "facts"})

    def test_interaction_name(self):
        self.assertEqual(sc.interaction_name("quote"), "s1_scout_quote")
        with self.assertRaises(ValueError):
            sc.interaction_name("unknown")


class TestBuildSharedPrefix(unittest.TestCase):

    def _prefix(self, **overrides):
        defaults = dict(
            session_context="design discussion for scouts",
            current_date="2026-04-23",
            catalog_rendered="Node abc123: 'Example'",
            surfaced_by_turn_rendered="t1: [abc123]",
            conversation_rendered="t1: User: hi\nAssistant: hi",
        )
        defaults.update(overrides)
        return sc.build_shared_prefix(**defaults)

    def test_returns_list_of_text_blocks(self):
        blocks = self._prefix()
        self.assertIsInstance(blocks, list)
        self.assertTrue(all(b.get("type") == "text" for b in blocks))
        self.assertTrue(all("text" in b for b in blocks))

    def test_cache_breakpoint_on_last_block_only(self):
        blocks = self._prefix()
        # exactly one cache_control marker
        with_cache = [b for b in blocks if "cache_control" in b]
        self.assertEqual(len(with_cache), 1)
        # and it's on the last block
        self.assertIn("cache_control", blocks[-1])
        self.assertEqual(blocks[-1]["cache_control"], {"type": "ephemeral"})

    def test_contains_all_sections(self):
        blocks = self._prefix()
        joined = "\n".join(b["text"] for b in blocks)
        # orientation preamble
        self.assertIn("You are observing a conversation", joined)
        # section headers
        self.assertIn("## Session context", joined)
        self.assertIn("## Current date", joined)
        self.assertIn("## Node catalog", joined)
        self.assertIn("## Surfaced nodes per turn", joined)
        self.assertIn("## Conversation window", joined)

    def test_byte_identical_for_same_inputs(self):
        """Cache sharing across scouts depends on byte-identical prefix."""
        a = self._prefix()
        b = self._prefix()
        self.assertEqual(a, b)

    def test_empty_strings_get_placeholders(self):
        blocks = self._prefix(session_context="", catalog_rendered="",
                              surfaced_by_turn_rendered="")
        joined = "\n".join(b["text"] for b in blocks)
        self.assertIn("(empty)", joined)
        self.assertIn("(none)", joined)


class TestAssembleCallContentRemoved(unittest.TestCase):
    """assemble_call_content was removed when per-scout task moved from
    user content (5m-cached) to system (1h-cached, per-scout). The runner
    composes system prompt directly and passes build_shared_prefix output
    through as user content. Lock the new contract: the old symbol must
    not return without an intentional design change.
    """

    def test_assemble_call_content_not_exported(self):
        self.assertFalse(
            hasattr(sc, "assemble_call_content"),
            "assemble_call_content should be removed — the per-scout task "
            "is now part of the system prompt (1h cache), not appended to "
            "user content. If reintroducing, update the cache-layout docs "
            "in contract.py first.")

    def test_assemble_call_content_not_in_all(self):
        self.assertNotIn("assemble_call_content", sc.__all__)


class TestValidateScoutOutput(unittest.TestCase):
    """Validation decision tree — no LLM calls."""

    def _valid_quote(self):
        return {
            "scout": "quote",
            "category_statement": "Phrases echoed or load-bearing should be quote atoms",
            "candidates": [
                {
                    "handle": "tokens are for thinking",
                    "speaker": "operator",
                    "evidence_quote": "delegate grunt work — tokens are for thinking",
                    "evidence_turns": ["t3"],
                    "why_candidate": "Echoed 2x; grounds delegation principle",
                    "grounds_candidates": ["Delegate work principle"],
                    "echo_count": 2,
                    "catalog_match": None,
                }
            ],
            "scanned": {"turns": 10, "considered": 4, "passed_threshold": 1},
        }

    def test_valid_envelope_passes(self):
        ok, normalized, errors, warnings = sc.validate_scout_output(
            self._valid_quote(), "quote")
        self.assertTrue(ok)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(len(normalized["candidates"]), 1)

    def test_empty_candidates_is_valid(self):
        out = self._valid_quote()
        out["candidates"] = []
        ok, normalized, errors, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertEqual(normalized["candidates"], [])

    def test_missing_envelope_field_fails(self):
        out = self._valid_quote()
        del out["category_statement"]
        ok, _, errors, _ = sc.validate_scout_output(out, "quote")
        self.assertFalse(ok)
        self.assertTrue(any("category_statement" in e for e in errors))

    def test_wrong_scout_name_fails(self):
        out = self._valid_quote()
        out["scout"] = "temporal"
        ok, _, errors, _ = sc.validate_scout_output(out, "quote")
        self.assertFalse(ok)

    def test_not_a_dict_fails(self):
        ok, stub, errors, _ = sc.validate_scout_output("not a dict", "quote")
        self.assertFalse(ok)
        self.assertEqual(stub["scout"], "quote")
        self.assertEqual(stub["candidates"], [])

    def test_unknown_scout_raises_programmer_error(self):
        with self.assertRaises(sc.ScoutOutputError):
            sc.validate_scout_output(self._valid_quote(), "unknown")

    def test_candidate_missing_required_field_is_dropped(self):
        out = self._valid_quote()
        del out["candidates"][0]["handle"]
        ok, normalized, errors, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)  # envelope sound; bad candidate dropped not fatal
        self.assertEqual(normalized["candidates"], [])
        self.assertTrue(any("handle" in w for w in warnings))

    def test_scout_specific_required_field_missing(self):
        out = self._valid_quote()
        del out["candidates"][0]["speaker"]
        ok, normalized, errors, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertEqual(normalized["candidates"], [])
        self.assertTrue(any("speaker" in w for w in warnings))

    def test_char_limit_soft_truncation(self):
        out = self._valid_quote()
        out["candidates"][0]["handle"] = "x" * 500
        ok, normalized, _, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertEqual(len(normalized["candidates"][0]["handle"]),
                         sc.FIELD_LIMITS["handle"])
        self.assertTrue(any("handle truncated" in w for w in warnings))

    def test_category_statement_truncation(self):
        out = self._valid_quote()
        out["category_statement"] = "a" * 1000
        ok, normalized, _, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertLessEqual(len(normalized["category_statement"]),
                             sc.FIELD_LIMITS["category_statement"])

    def test_evidence_turns_str_coerced_to_list(self):
        out = self._valid_quote()
        out["candidates"][0]["evidence_turns"] = "t3"
        ok, normalized, _, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertEqual(normalized["candidates"][0]["evidence_turns"], ["t3"])
        self.assertTrue(any("coerced" in w for w in warnings))

    def test_scanned_missing_turns_defaulted(self):
        out = self._valid_quote()
        out["scanned"] = {}
        ok, normalized, _, warnings = sc.validate_scout_output(out, "quote")
        self.assertTrue(ok)
        self.assertEqual(normalized["scanned"]["turns"], 0)

    def test_facts_required_fields_enforced(self):
        base = {
            "scout": "facts",
            "category_statement": "entity-feature-value",
            "candidates": [
                {
                    "handle": "Book pages",
                    "evidence_quote": "320 pages",
                    "evidence_turns": ["t1"],
                    "why_candidate": "count",
                    # entity/feature/value missing
                }
            ],
            "scanned": {"turns": 5},
        }
        ok, normalized, _, warnings = sc.validate_scout_output(base, "facts")
        self.assertTrue(ok)
        self.assertEqual(normalized["candidates"], [])
        msg = " ".join(warnings)
        self.assertIn("entity", msg)
        self.assertIn("feature", msg)
        self.assertIn("value", msg)


    def test_temporal_required_source_phrase(self):
        base = {
            "scout": "temporal",
            "category_statement": "dates become time_anchor bridges",
            "candidates": [
                {
                    "handle": "2026-04-23",
                    "evidence_quote": "today I did X",
                    "evidence_turns": ["t1"],
                    "why_candidate": "today anchor",
                    # source_phrase missing
                }
            ],
            "scanned": {"turns": 2},
        }
        ok, normalized, _, warnings = sc.validate_scout_output(base, "temporal")
        self.assertTrue(ok)
        self.assertEqual(normalized["candidates"], [])


class TestFormatScoutReport(unittest.TestCase):

    def _outputs(self):
        return {
            "quote": {
                "scout": "quote",
                "category_statement": "Phrases that echo",
                "candidates": [{
                    "handle": "short phrase",
                    "speaker": "operator",
                    "evidence_quote": "short phrase in context",
                    "evidence_turns": ["t2"],
                    "why_candidate": "recurs",
                    "grounds_candidates": ["Principle A"],
                    "echo_count": 2,
                }],
                "scanned": {"turns": 8, "considered": 3, "passed_threshold": 1},
            },
            "temporal": {
                "scout": "temporal",
                "category_statement": "dates → anchors",
                "candidates": [],
                "scanned": {"turns": 8, "date_phrases_found": 0},
            },
            "facts": {
                "scout": "facts",
                "category_statement": "entity-feature-value",
                "candidates": [{
                    "handle": "Book pages",
                    "entity": "The Nightingale",
                    "feature": "pages",
                    "value": "320",
                    "evidence_quote": "it's a 320-page book",
                    "evidence_turns": ["t4"],
                    "why_candidate": "metadata handle",
                }],
                "scanned": {"turns": 8, "fact_claims_found": 1, "passed_threshold": 1},
            },
        }

    def test_contains_all_scouts(self):
        report = sc.format_scout_report_for_s1s(self._outputs())
        for scout in sc.SCOUT_NAMES:
            self.assertIn(f"### {scout}", report)

    def test_empty_candidates_rendered_explicitly(self):
        report = sc.format_scout_report_for_s1s(self._outputs())
        # temporal has empty candidates
        self.assertIn("(nothing qualified)", report)

    def test_populated_candidates_show_extra_fields(self):
        report = sc.format_scout_report_for_s1s(self._outputs())
        # facts scout extra fields should be visible
        self.assertIn("entity:", report)
        self.assertIn("feature:", report)
        self.assertIn("value:", report)
        # quote extras
        self.assertIn("echo_count:", report)

    def test_missing_scout_shown_as_did_not_run(self):
        outputs = self._outputs()
        del outputs["temporal"]
        report = sc.format_scout_report_for_s1s(outputs)
        self.assertIn("### temporal", report)
        self.assertIn("did not run", report)

    def test_scanned_footer_present(self):
        report = sc.format_scout_report_for_s1s(self._outputs())
        self.assertIn("(scanned:", report)


if __name__ == "__main__":
    unittest.main()
