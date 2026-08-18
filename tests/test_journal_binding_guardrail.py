"""Guardrail: every S2 encoder LLM call carries the journal.

The healer and aspect units ran mute for months — no journal binding, so no
residue, and nothing failed (the silence was only visible in a corpus-level
audit, finding 78677e17). This holds the line structurally: any `_call_llm`
call site in an S2 unit module must opt into the journal (`journal=True`),
and any future S2 loop encoder must decorate + harvest like the shipped two.
A deliberate journal-exempt call would edit this test — which is the point:
exemption becomes a reviewed decision, not a default.
"""
import os
import re
import unittest

S2_DIR = os.path.join(os.path.dirname(__file__), '..',
                      'servers', 'scales', 's2')


def _s2_sources():
    for fname in sorted(os.listdir(S2_DIR)):
        if not fname.endswith('.py'):
            continue
        path = os.path.join(S2_DIR, fname)
        with open(path) as f:
            yield fname, f.read()


class TestJournalBindingGuardrail(unittest.TestCase):

    def test_every_call_llm_site_binds_journal(self):
        """A `self._call_llm(...)` in an S2 module without journal=True is a
        new mute unit in the making."""
        offenders = []
        for fname, src in _s2_sources():
            for m in re.finditer(r'self\._call_llm\(([^)]*)\)', src, re.S):
                if 'journal=True' not in m.group(1):
                    line = src[:m.start()].count('\n') + 1
                    offenders.append('%s:%d' % (fname, line))
        self.assertEqual(offenders, [], (
            '_call_llm without journal=True: %s — single-shot S2 runs must '
            'carry the journal binding (or edit this test with the reason '
            'for the exemption)' % offenders))

    def test_every_run_llm_loop_encoder_decorates_and_harvests(self):
        """Loop encoders bind via decorate_system + journal.harvest (the
        harvest call may live in the shared _fold_batch_result)."""
        for fname, src in _s2_sources():
            if 'run_llm_loop(' not in src or fname == 'base.py':
                continue
            self.assertIn('.journal.decorate_system(', src,
                          '%s runs the LLM loop without decorating the '
                          'system prompt with the journal blocks' % fname)
            self.assertTrue(
                '.journal.harvest(' in src or '_fold_batch_result(' in src,
                '%s runs the LLM loop without harvesting residue' % fname)


if __name__ == '__main__':
    unittest.main()
