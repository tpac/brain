"""Tests for servers/scales/s1/orientation.py — shared orientation labels.

Guardrail: the orientation section labels are the single source of truth
for what the S1 pipeline's structural inputs MEAN (node catalog, surfaced
nodes, conversation window, current date, session context, encoding journal).
Scouts and S1S must both read the same field-level explanations.

These tests lock:
- Each label exports and is non-empty
- Composed preambles for scouts and S1S both include the shared labels
  from the same module (no forking text in two places)
- The scout preamble re-exported via contract.py is the same object as
  orientation.SCOUT_ORIENTATION_PREAMBLE
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s1 import orientation as ori
from servers.scales.s1.scouts import contract as sc


class TestLabelsExist(unittest.TestCase):

    def test_all_labels_present(self):
        for name in (
            'NODE_CATALOG_LABEL',
            'SURFACED_NODES_LABEL',
            'CONVERSATION_WINDOW_LABEL',
            'CURRENT_DATE_LABEL',
            'SESSION_CONTEXT_LABEL',
            'ENCODING_JOURNAL_LABEL',
        ):
            self.assertTrue(hasattr(ori, name),
                            f'orientation module missing {name}')
            self.assertIsInstance(getattr(ori, name), str)
            self.assertTrue(len(getattr(ori, name)) > 30,
                            f'{name} is suspiciously short')

    def test_labels_use_operator_not_user(self):
        """All seed labels use 'operator' for domain-neutrality."""
        for name in (
            'CONVERSATION_WINDOW_LABEL',
            'SURFACED_NODES_LABEL',
        ):
            text = getattr(ori, name).lower()
            # Heuristic: "user" as a standalone word shouldn't appear;
            # 'operator' should be present. Tolerate 'users' or 'use' words
            # by requiring whole-word match.
            import re
            self.assertIsNone(
                re.search(r'\buser\b', text),
                f'{name} uses "user"; seed labels should say "operator"')

    def test_no_dead_terminology_judge(self):
        """'judge' was renamed to 'surfacer' — the old term should not
        appear in any label (it would teach new brains outdated vocab)."""
        for name in (
            'NODE_CATALOG_LABEL',
            'SURFACED_NODES_LABEL',
            'CONVERSATION_WINDOW_LABEL',
        ):
            text = getattr(ori, name).lower()
            self.assertNotIn(
                'judge',
                text,
                f'{name} still mentions "judge" — that term was renamed '
                f'to "surfacer". Update the label.')

    def test_no_hardcoded_turn_count(self):
        """The window size is a config value; the prompt shouldn't hardcode
        '10 exchanges' — if the config changes the prompt becomes a lie."""
        text = ori.CONVERSATION_WINDOW_LABEL.lower()
        # "10 exchanges" or "last 10" would be hardcoded
        import re
        self.assertIsNone(
            re.search(r'\b10\s+(exchanges|turns)\b', text),
            'CONVERSATION_WINDOW_LABEL hardcodes 10 — use "N" instead so '
            'the label survives config changes.')


class TestComposedPreambles(unittest.TestCase):

    def test_scout_preamble_includes_all_scout_labels(self):
        p = ori.SCOUT_ORIENTATION_PREAMBLE
        # Scouts don't get the encoding journal — it's S1S-only
        self.assertNotIn(ori.ENCODING_JOURNAL_LABEL.strip(), p)
        # All other labels present
        for label in (
            ori.SESSION_CONTEXT_LABEL,
            ori.CURRENT_DATE_LABEL,
            ori.NODE_CATALOG_LABEL,
            ori.SURFACED_NODES_LABEL,
            ori.CONVERSATION_WINDOW_LABEL,
        ):
            self.assertIn(label.strip(), p.strip(),
                          f'scout preamble missing a shared label block')

    def test_scout_preamble_has_scout_voice(self):
        p = ori.SCOUT_ORIENTATION_PREAMBLE
        self.assertIn('scout', p.lower())
        self.assertIn('do not write nodes', p.lower())

    def test_s1s_what_you_receive_includes_encoding_journal(self):
        p = ori.S1S_WHAT_YOU_RECEIVE
        self.assertIn(ori.ENCODING_JOURNAL_LABEL.strip(), p)
        self.assertIn(ori.NODE_CATALOG_LABEL.strip(), p)
        self.assertIn(ori.CONVERSATION_WINDOW_LABEL.strip(), p)


class TestContractReExport(unittest.TestCase):
    """contract.py re-exports SCOUT_ORIENTATION_PREAMBLE from orientation.
    Assert no forked copy — both point at the same string."""

    def test_contract_re_export_matches_orientation_source(self):
        self.assertEqual(sc.SCOUT_ORIENTATION_PREAMBLE,
                         ori.SCOUT_ORIENTATION_PREAMBLE)

    def test_contract_has_no_local_preamble_definition(self):
        """contract.py must not define its own preamble string literal —
        that would fork from orientation.py and drift. This check searches
        the module source for the old 'You are observing a conversation'
        literal; if found OUTSIDE an import line, contract has a local
        copy again."""
        import inspect
        import os as _os
        src_path = inspect.getfile(sc)
        # Only flag actual assignments, not import statements
        with open(src_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if 'SCOUT_ORIENTATION_PREAMBLE' not in line:
                continue
            if line.strip().startswith(('from ', 'import ', '#')):
                continue
            if '=' in line and 'from' not in line:
                # This is an assignment in contract.py — the forked copy is back
                stripped = line.strip()
                if stripped.startswith('SCOUT_ORIENTATION_PREAMBLE ='):
                    raise AssertionError(
                        f'contract.py line {i}: {stripped!r} — local '
                        f'assignment means a forked copy. Import from '
                        f'orientation.py instead.')


if __name__ == '__main__':
    unittest.main()
