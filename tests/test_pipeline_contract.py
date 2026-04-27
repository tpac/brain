"""Tests for pipeline_contract.py — surface prompt, output formatting, embedding groups.

Covers:
- EMBEDDING_GROUPS contract integrity
- format_candidate_for_surface output structure
- build_surface_prompt assembly
- format_surface_output with and without graph neighbors
- get_group_weight lookups
"""

import unittest
import sys

sys.path.insert(0, '.')

from servers.pipeline_contract import (
    EMBEDDING_GROUPS,
    EMBEDDING_SCORING_METHOD,
    EMBEDDING_SKIP_FIELDS,
    get_group_weight,
    get_group_fields,
    format_candidate_for_surface,
    build_surface_prompt,
    format_surface_output,
    SURFACE,
)


class TestEmbeddingGroups(unittest.TestCase):
    """Verify embedding group contract integrity."""

    def test_all_groups_present(self):
        self.assertEqual(set(EMBEDDING_GROUPS.keys()), {
            'title', 'blend', 'high_meta', 'other_meta', 'edge_context', 'question',
            'field_content', 'field_situation', 'field_reasoning',
            'field_user_raw_quote', 'field_anchor_raw_quote',
        })

    def test_cohort_assignment(self):
        legacy = {k for k, v in EMBEDDING_GROUPS.items() if v['cohort'] == 'legacy'}
        field = {k for k, v in EMBEDDING_GROUPS.items() if v['cohort'] == 'field'}
        self.assertEqual(legacy, {
            'title', 'blend', 'high_meta', 'other_meta', 'edge_context', 'question'
        })
        self.assertEqual(field, {
            'field_content', 'field_situation', 'field_reasoning',
            'field_user_raw_quote', 'field_anchor_raw_quote'
        })

    def test_cohort_weight_invariants(self):
        """Legacy weights > 0 (participate in recall top2_avg).
        Field weights = 0 (kernel reads directly, must not ripple into scoring)."""
        for name, group in EMBEDDING_GROUPS.items():
            cohort = group.get('cohort')
            if cohort == 'legacy':
                self.assertGreater(group['weight'], 0, f"{name}: legacy must have weight>0")
            elif cohort == 'field':
                self.assertEqual(group['weight'], 0, f"{name}: field must have weight=0")
            else:
                self.fail(f"{name}: unknown cohort {cohort!r}")

    def test_weights_ordered(self):
        """Title > blend > high_meta > other_meta."""
        self.assertGreater(EMBEDDING_GROUPS['title']['weight'],
                          EMBEDDING_GROUPS['blend']['weight'])
        self.assertGreater(EMBEDDING_GROUPS['blend']['weight'],
                          EMBEDDING_GROUPS['high_meta']['weight'])
        self.assertGreater(EMBEDDING_GROUPS['high_meta']['weight'],
                          EMBEDDING_GROUPS['other_meta']['weight'])

    def test_title_always_computed(self):
        self.assertTrue(EMBEDDING_GROUPS['title']['always_compute'])

    def test_blend_is_primary(self):
        self.assertEqual(EMBEDDING_GROUPS['blend']['vector_type'], '_primary')

    def test_scoring_method(self):
        self.assertEqual(EMBEDDING_SCORING_METHOD, 'top2_avg')

    def test_get_group_weight_known(self):
        self.assertEqual(get_group_weight('title'), 1.0)
        self.assertEqual(get_group_weight('high_meta'), 0.70)

    def test_get_group_weight_unknown(self):
        """Unknown types get other_meta weight."""
        self.assertEqual(get_group_weight('nonexistent'),
                         EMBEDDING_GROUPS['other_meta']['weight'])

    def test_get_group_fields(self):
        fields = get_group_fields('high_meta')
        self.assertIn('situation', fields)
        self.assertIn('user_raw_quote', fields)
        self.assertNotIn('_emergent', fields)

    def test_skip_fields(self):
        self.assertIn('metadata_created_at', EMBEDDING_SKIP_FIELDS)
        self.assertIn('validation_count', EMBEDDING_SKIP_FIELDS)


class TestFormatCandidateForSurface(unittest.TestCase):
    """Verify candidate formatting for the surface prompt."""

    def test_basic_candidate(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test rule',
             'content': 'Some content', 'confidence': 0.9, 'score': 0.75}
        result = format_candidate_for_surface(c, 1)
        self.assertIn('#1', result)
        self.assertIn('[rule]', result)
        self.assertIn('Test rule', result)
        self.assertIn('abc12345', result)

    def test_metadata_included_when_present(self):
        """Metadata fields are read from `_metadata` (get_rich_node shape)
        as of the contract refactor. The old top-level keys are not honored."""
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content',
             'situation': 'When debugging',  # situation is top-level for ergonomics
             '_metadata': {'reasoning': 'Important because...'}}
        result = format_candidate_for_surface(c, 1)
        self.assertIn('Situation:', result)
        self.assertIn('reasoning:', result.lower())

    def test_metadata_omitted_when_empty(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content'}
        result = format_candidate_for_surface(c, 1)
        self.assertNotIn('Situation:', result)
        self.assertNotIn('Reasoning:', result)

    def test_locked_flag(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content', 'locked': True}
        result = format_candidate_for_surface(c, 1)
        self.assertIn('locked', result)

    def test_edges_included(self):
        """Edges live in `connections` (get_rich_node shape) and render as
        natural-language relations like `this depends_on "Related node"`.

        Renderer contract (servers/contract.py:render_rich_node):
        - When connection has multiple relations → use `relations` array.
        - When single relation → use top-level `relation` + `description`
          on the connection (not nested in a `relations` list of one).

        Was: pinned on the literal substring `Related node` from a top-level
        `top_edges` array, which the formatter no longer reads.
        """
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content',
             'connections': [{
                 'id': 'def67890',
                 'title': 'Related node',
                 'type': 'concept',
                 'direction': 'outgoing',
                 # Single-relation form: top-level relation + description.
                 'relation': 'depends_on',
                 'description': 'because',
             }]}
        result = format_candidate_for_surface(c, 1)
        self.assertIn('Related node', result)
        self.assertIn('depends_on', result)


class TestBuildSurfacePrompt(unittest.TestCase):
    """Verify surface prompt assembly."""

    def test_prompt_includes_session_context(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test query',
            session_context='Working on decode pipeline')
        self.assertIn('Working on decode pipeline', prompt)

    def test_prompt_includes_user_message(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'how does the daemon work?')
        self.assertIn('how does the daemon work?', prompt)

    def test_prompt_includes_recently_recalled(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test',
            recently_recalled=[{'id': 'xyz', 'title': 'Some old node'}])
        self.assertIn('xyz', prompt)
        self.assertIn('Some old node', prompt)

    def test_prompt_includes_silence_instruction(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test')
        self.assertIn('Silence is better than noise', prompt)

    def test_prompt_includes_noise_rejection(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test')
        self.assertIn('coincidence', prompt)

    def test_max_tokens_from_config(self):
        _, max_tokens = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test')
        self.assertEqual(max_tokens, SURFACE['max_tokens'])


class TestFormatSurfaceOutput(unittest.TestCase):
    """Verify structured output formatting."""

    def test_empty_selected(self):
        result = format_surface_output([], [])
        self.assertEqual(result, "")

    def test_basic_output(self):
        selected = [{'id': 'abc12345', 'why': 'directly relevant'}]
        candidates = [{'id': 'abc12345', 'type': 'rule', 'title': 'Test rule',
                       'content': 'Some content', 'confidence': 0.9}]
        result = format_surface_output(selected, candidates)
        self.assertIn('Brain recalled 1 memories', result)
        self.assertIn('Test rule', result)
        self.assertIn('directly relevant', result)

    def test_graph_neighbors_included(self):
        selected = [{'id': 'abc12345', 'why': 'relevant'}]
        candidates = [{'id': 'abc12345', 'type': 'rule', 'title': 'Test',
                       'content': 'Content'}]
        neighbors = [{'type': 'mechanism', 'title': 'Neighbor node',
                      'edge_type': 'depends_on', 'edge_description': 'because X',
                      'content': 'Neighbor content'}]
        result = format_surface_output(selected, candidates, neighbors)
        self.assertIn('Related knowledge', result)
        self.assertIn('Neighbor node', result)
        self.assertIn('depends_on', result)

    def test_no_graph_neighbors(self):
        selected = [{'id': 'abc12345', 'why': 'relevant'}]
        candidates = [{'id': 'abc12345', 'type': 'rule', 'title': 'Test',
                       'content': 'Content'}]
        result = format_surface_output(selected, candidates)
        self.assertNotIn('Related knowledge', result)

    def test_selected_not_in_candidates(self):
        """Selected node not found in candidates — should be skipped."""
        selected = [{'id': 'missing', 'why': 'relevant'}]
        candidates = [{'id': 'abc12345', 'type': 'rule', 'title': 'Test',
                       'content': 'Content'}]
        result = format_surface_output(selected, candidates)
        self.assertIn("Brain recalled 1 memories", result)
        self.assertNotIn("Test", result)  # Candidate content shouldn't appear


if __name__ == '__main__':
    unittest.main()
