"""Retired node fields must not be taught, advertised, or queried.

WHY THIS EXISTS: schema v28 dropped `nodes.keywords`, and every write
surface was cleaned — but nothing asserted ABSENCE. The encoder prompts
kept teaching the field, `additionalProperties` is deliberately open on
every write branch, and unknown keys route to node_metadata_kv. Result:
229 nodes silently accumulated a dead `keywords` KV row between July 1
and the audit, and the whole existing suite stayed green throughout.

Tests assert presence. This one asserts absence.

Note on scope: a general "unknown field" check is impossible by design —
open KV accepts any key (`emotional_context`, `event_time`, ...) and that
freedom is the point. So this pins the specific fields we RETIRED, which
is the case that actually regressed.

NOT covered here (different concept, live recall path — node 48d23822):
the Q/A/B/K enrichment vector also named 'keywords'
(store_enrichments / ENRICHMENT_VECTOR_TYPES / the FTS `_search_keywords`
lane). Those are legitimate and must keep working.
"""
import re
import unittest

from servers.contract import get_writable_fields, STRUCTURAL_FIELDS, PROMOTED_FIELDS
from servers.brain_recall import _NODE_COLUMNS

# Fields removed from the node schema. A field lands here when its column
# or KV slot is retired, and it must then be absent from every surface
# below. Keep the reason inline — it is the whole documentation.
RETIRED_NODE_FIELDS = {
    'keywords': 'nodes.keywords dropped in schema v28 (auto-extractor noise)',
}


def _seed_prompts():
    """The four encoder prompts, as (name, text). These mirror the
    production-ACTIVE interaction versions — test_prompt_sync.py pins that
    correspondence, so scanning the seeds scans what the encoders see."""
    from servers.scales.s1.encoding_prompt import SYSTEM_PROMPT as s1e
    from servers.scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as comm
    from servers.scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as cons
    from servers.scales.s2.healer_prompt import SYSTEM_PROMPT as heal
    return [
        ('s1e', s1e),
        ('s2_community_enrichment', comm),
        ('s2_consolidation_enrichment', cons),
        ('s2_healer', heal),
    ]


class TestRetiredFieldsNotTaught(unittest.TestCase):
    """The encoder must not be TOLD to write a field the schema dropped."""

    def test_retired_field_absent_from_every_encoder_prompt(self):
        for field, reason in RETIRED_NODE_FIELDS.items():
            for name, text in _seed_prompts():
                hits = [ln.strip() for ln in text.split('\n') if field in ln]
                self.assertEqual(
                    hits, [],
                    '%s prompt still references retired field %r (%s).\n'
                    'Offending lines:\n  %s'
                    % (name, field, reason, '\n  '.join(hits[:5])))


class TestRetiredFieldsNotAdvertised(unittest.TestCase):
    """The write schema is generated from the contract — a retired field
    left in the contract silently reappears in every MCP write tool."""

    def test_retired_field_not_writable(self):
        writable = get_writable_fields()
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(field, writable, '%s (%s)' % (field, reason))

    def test_retired_field_not_in_field_registries(self):
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(field, STRUCTURAL_FIELDS, '%s (%s)' % (field, reason))
            self.assertNotIn(field, PROMOTED_FIELDS, '%s (%s)' % (field, reason))


class TestRetiredFieldsNotQueryable(unittest.TestCase):
    """_NODE_COLUMNS drives dict-filter SQL. A dropped column left in it
    builds a query against a column that no longer exists."""

    def test_retired_field_not_in_node_columns(self):
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(
                field, _NODE_COLUMNS,
                '%s is in _NODE_COLUMNS but the column is gone (%s) — a dict '
                'filter on it would build SQL against a dropped column'
                % (field, reason))


class TestRetiredFieldsNotInMcpSchemas(unittest.TestCase):
    """The generated MCP tool schemas are a FOURTH surface that talks to a
    model without being code, prompts, or seed files.

    Node 75c661d5 recorded three audit surfaces for retiring a param (code,
    interactions-DB prompts, seed .py). A fourth bit us: `find_node_by_title`
    advertised "(content snippet, keywords)" as its return shape long after
    v28 stopped returning it — not an example, a PROMISE, in a description
    no other check reads.
    """

    # (tool, field) pairs where the word legitimately appears.
    #   recall  — query PHRASING advice ("semantic, not keyword")
    #   enrich  — the Q/A/B/K enrichment vector, a different concept on the
    #             live recall path (node 48d23822)
    ALLOWED = {
        ('recall', 'keywords'),
        ('enrich', 'keywords'),
    }

    @staticmethod
    def _declared_property_names(schema):
        names = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == 'properties' and isinstance(v, dict):
                        names.update(v.keys())
                    walk(v)
            elif isinstance(o, list):
                for item in o:
                    walk(item)

        walk(schema)
        return names

    def test_no_write_op_declares_a_retired_field(self):
        """A retired field declared as a parameter is unambiguous — the
        schema is inviting a model to send a field the write path drops."""
        from servers.brain_mcp import TOOLS
        for tool in TOOLS:
            declared = self._declared_property_names(tool.get('inputSchema', {}))
            for field in RETIRED_NODE_FIELDS:
                if (tool['name'], field) in self.ALLOWED:
                    continue
                self.assertNotIn(
                    field, declared,
                    'MCP tool %r declares retired field %r as a parameter'
                    % (tool['name'], field))

    def test_no_tool_description_mentions_a_retired_field(self):
        """Descriptions are prose a model reads as fact. A retired field
        named here is either a stale return-shape promise or an example
        teaching a dead write."""
        import json
        from servers.brain_mcp import TOOLS
        for tool in TOOLS:
            blob = json.dumps(tool)
            for field in RETIRED_NODE_FIELDS:
                if (tool['name'], field) in self.ALLOWED:
                    continue
                self.assertNotIn(
                    field, blob,
                    'MCP tool %r mentions retired field %r in its schema or '
                    'description. If the mention is legitimate (a different '
                    'concept sharing the name), add (%r, %r) to ALLOWED with '
                    'a reason.' % (tool['name'], field, tool['name'], field))


class TestEnrichmentVectorSurvives(unittest.TestCase):
    """Guard the scope boundary: the removal must not take the live
    Q/A/B/K enrichment vector with it (node 48d23822)."""

    def test_keywords_enrichment_vector_still_registered(self):
        from servers.brain_constants import ENRICHMENT_VECTOR_TYPES
        self.assertIn('keywords', ENRICHMENT_VECTOR_TYPES)

    def test_store_enrichments_still_accepts_keywords(self):
        import inspect
        from servers.brain_remember import BrainRememberMixin
        sig = inspect.signature(BrainRememberMixin.store_enrichments)
        self.assertIn('keywords', sig.parameters)


if __name__ == '__main__':
    unittest.main()
