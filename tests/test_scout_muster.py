"""Tests for servers/scales/s1/scouts/muster.py — parallel scout orchestrator.

Mocks SCOUT_RUNNERS so tests don't hit the network or real scout internals.
Exercises:
- Happy path: all scouts return valid envelopes, formatted report contains all
- Error isolation: one scout raises → stub, others continue
- Timeout isolation: one scout blocks → timeout stub, others continue
- Muster context: shared_prefix shape + byte-identical determinism
- Context building: message → turn conversion, surfaced_by_turn parse
- Trace emission: O and K events per scout on the s1e chain
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.scouts import contract as sc
from servers.scales.s1.scouts import muster as m


# ─── Fake scout runner factory ────────────────────────────────────────────

def _ok_envelope(scout_name, n_candidates=1, category='test'):
    """Build a valid scout envelope."""
    spec = sc.SCOUT_FIELD_SPECS[scout_name]
    scout_extras = {}
    # Fill scout-specific required fields with plausible stubs
    for field in spec['required']:
        if field == 'speaker':
            scout_extras[field] = 'operator'
        elif field == 'source_phrase':
            scout_extras[field] = 'today'
        elif field in ('entity', 'feature', 'value'):
            scout_extras[field] = 'x'
        elif field == 'turn_evidence':
            scout_extras[field] = [{'turn': 't1', 'note': 'n'}]
    candidates = []
    for i in range(n_candidates):
        c = {
            'handle': f'h{i}',
            'evidence_quote': 'q',
            'evidence_turns': ['t1'],
            'why_candidate': 'w',
            **scout_extras,
        }
        candidates.append(c)
    return {
        'scout': scout_name,
        'category_statement': category,
        'candidates': candidates,
        'scanned': {'turns': 5, 'considered': 3, 'passed_threshold': n_candidates},
        '_usage': {'input_tokens': 100, 'output_tokens': 20},
        '_latency_ms': 100,
        '_errors': [],
        '_warnings': [],
    }


def _make_fake_runners(results_by_scout, sleep_by_scout=None,
                      raise_by_scout=None):
    """Build a SCOUT_RUNNERS-shape dict with scripted behavior."""
    sleep_by_scout = sleep_by_scout or {}
    raise_by_scout = raise_by_scout or {}

    def runner_for(name):
        def _run(brain, ctx):
            if name in sleep_by_scout:
                time.sleep(sleep_by_scout[name])
            if name in raise_by_scout:
                raise raise_by_scout[name]
            return results_by_scout[name]
        return _run

    return {n: runner_for(n) for n in sc.SCOUT_NAMES}


def _basic_ctx(brain):
    """Minimal ctx that run_muster can consume."""
    return {
        'brain': brain,
        'session_id': 'abcdef12-test-session',
        'counter': 5,
        'turns': [{'turn_id': 't1', 'role': 'user', 'text': 'hi today'}],
        'catalog_nodes': [],
        'surfaced_by_turn': {},
        'session_context': 'test',
        'current_date': '2026-04-23',
        'shared_prefix': sc.build_shared_prefix(
            session_context='test', current_date='2026-04-23',
            catalog_rendered='', surfaced_by_turn_rendered='',
            conversation_rendered='t1: user: hi today'),
        'anthropic_client': None,
        'log_fn': None,
    }


# ─── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath(BrainTestBase):
    needs_embedder = False

    def test_all_scouts_return_report_has_all_blocks(self):
        fake = _make_fake_runners({
            'quote': _ok_envelope('quote'),
            'temporal': _ok_envelope('temporal'),
            'facts': _ok_envelope('facts'),
        })
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            report, outputs, metrics = m.run_muster(_basic_ctx(self.brain))
        for name in sc.SCOUT_NAMES:
            self.assertIn(f'### {name}', report)
            self.assertIn(name, outputs)
            self.assertEqual(outputs[name]['_errors'], [])

    def test_exclude_scouts_skips_runner_and_stubs_disabled(self):
        """exclude_scouts=('quote',) — the lived arm's retirement: the quote
        runner never executes, its slot pads with the 'disabled' stub, and the
        other scouts run normally."""
        ran = []

        def _tracking(name, envelope):
            def _r(brain, ctx):
                ran.append(name)
                return envelope
            return _r

        fake = {n: _tracking(n, _ok_envelope(n)) for n in sc.SCOUT_NAMES}
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            _, outputs, metrics = m.run_muster(
                _basic_ctx(self.brain), exclude_scouts=('quote',))
        self.assertNotIn('quote', ran)                    # never executed
        self.assertIn('temporal', ran)
        self.assertIn('facts', ran)
        self.assertIn('quote', outputs)                   # slot still shape-safe
        errs = outputs['quote']['_errors']
        self.assertTrue(any('disabled' in e.get('msg', '') for e in errs),
                        'excluded scout must carry the disabled stub: %r' % errs)
        self.assertEqual(outputs['quote']['candidates'], [])

    def test_metrics_capture_counts_and_latency(self):
        fake = _make_fake_runners({
            'quote': _ok_envelope('quote', n_candidates=3),
            'temporal': _ok_envelope('temporal', n_candidates=5),
            'facts': _ok_envelope('facts', n_candidates=0),
        })
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            _, _, metrics = m.run_muster(_basic_ctx(self.brain))
        self.assertEqual(metrics['total_candidates'], 8)
        self.assertEqual(metrics['total_errors'], 0)
        self.assertGreaterEqual(metrics['elapsed_ms'], 0)
        self.assertEqual(metrics['per_scout']['quote']['candidates'], 3)


class TestErrorIsolation(BrainTestBase):
    needs_embedder = False

    def test_one_scout_raises_others_continue(self):
        fake = _make_fake_runners(
            results_by_scout={
                'quote': _ok_envelope('quote'),
                'temporal': _ok_envelope('temporal'),
                'facts': _ok_envelope('facts'),
            },
            raise_by_scout={'quote': RuntimeError('boom')},
        )
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            report, outputs, metrics = m.run_muster(_basic_ctx(self.brain))

        # Quote stubbed with errors
        self.assertEqual(outputs['quote']['candidates'], [])
        self.assertTrue(any('muster_exception' in e['type']
                            or 'boom' in e.get('msg', '')
                            for e in outputs['quote']['_errors']))

        # Others unaffected
        for name in [n for n in sc.SCOUT_NAMES if n != 'quote']:
            self.assertEqual(outputs[name]['_errors'], [])
            self.assertEqual(len(outputs[name]['candidates']), 1)

        # Report still renders all sections
        for name in sc.SCOUT_NAMES:
            self.assertIn(f'### {name}', report)

        # Metrics reflect one error
        self.assertGreaterEqual(metrics['total_errors'], 1)

    def test_all_scouts_raise_returns_all_stubs(self):
        fake = _make_fake_runners(
            results_by_scout={n: _ok_envelope(n) for n in sc.SCOUT_NAMES},
            raise_by_scout={n: RuntimeError('x') for n in sc.SCOUT_NAMES},
        )
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            _, outputs, metrics = m.run_muster(_basic_ctx(self.brain))
        for name in sc.SCOUT_NAMES:
            self.assertEqual(outputs[name]['candidates'], [])
            self.assertTrue(len(outputs[name]['_errors']) >= 1)


class TestTimeoutIsolation(BrainTestBase):
    needs_embedder = False

    def test_one_scout_times_out_others_return(self):
        fake = _make_fake_runners(
            results_by_scout={n: _ok_envelope(n) for n in sc.SCOUT_NAMES},
            sleep_by_scout={'facts': 2.0},  # blocks past timeout
        )
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            _, outputs, _ = m.run_muster(_basic_ctx(self.brain), timeout_s=0.2)
        # facts timed out
        self.assertEqual(outputs['facts']['candidates'], [])
        self.assertTrue(any(e['type'] == 'muster_timeout'
                            for e in outputs['facts']['_errors']))
        # others returned normally
        for name in ('quote', 'temporal'):
            self.assertEqual(outputs[name]['_errors'], [])


class TestSharedPrefixDeterminism(BrainTestBase):
    """Byte-identical shared_prefix across scouts is THE caching guarantee.

    Since all scouts in one cycle receive the same ctx['shared_prefix'],
    this is easy to confirm — but let's lock the build_muster_context
    output shape so regressions surface here."""
    needs_embedder = False

    def test_shared_prefix_has_cache_breakpoint(self):
        ctx = m.build_muster_context(
            brain=self.brain,
            messages=[{'id': 't1', 'role': 'user', 'content': 'hi'}],
            session_id='s123', counter=1,
            catalog_rendered='', catalog_node_ids=set(),
            session_context='', current_date='2026-04-23')
        blocks = ctx['shared_prefix']
        # Exactly one block carries cache_control
        with_cache = [b for b in blocks if 'cache_control' in b]
        self.assertEqual(len(with_cache), 1)
        # And it's the last one
        self.assertIn('cache_control', blocks[-1])

    def test_ctx_turns_derived_from_messages(self):
        messages = [
            {'id': 't1', 'role': 'user', 'content': 'hi'},
            {'id': 't2', 'role': 'assistant', 'content': 'hello'},
            {'id': 't3', 'role': 'user', 'content': 'today I'},
        ]
        ctx = m.build_muster_context(
            brain=self.brain, messages=messages,
            session_id='s', counter=1,
            catalog_rendered='', catalog_node_ids=set())
        self.assertEqual(len(ctx['turns']), 3)
        self.assertEqual(ctx['turns'][0]['turn_id'], 't1')
        self.assertEqual(ctx['turns'][2]['text'], 'today I')

    def test_surfaced_by_turn_parsed_from_judge_output(self):
        messages = [
            {'id': 't1', 'role': 'user', 'content': 'hi',
             'judge_output': 'picked id:abc12345 and id:def67890'},
            {'id': 't2', 'role': 'user', 'content': 'more',
             'judge_output': '(no selection)'},
            {'id': 't3', 'role': 'user', 'content': 'ok',
             'judge_output': 'id:ghi11111'},
        ]
        ctx = m.build_muster_context(
            brain=self.brain, messages=messages,
            session_id='s', counter=1,
            catalog_rendered='', catalog_node_ids=set())
        self.assertIn('abc12345', ctx['surfaced_by_turn']['t1'])
        self.assertIn('def67890', ctx['surfaced_by_turn']['t1'])
        # no-selection / missing judge_output → not in index
        self.assertNotIn('t2', ctx['surfaced_by_turn'])
        self.assertEqual(ctx['surfaced_by_turn']['t3'], ['ghi11111'])


class TestTraceEmission(BrainTestBase):
    needs_embedder = False

    def test_emits_O_and_K_per_scout_on_s1e_chain(self):
        fake = _make_fake_runners({
            'quote': _ok_envelope('quote'),
            'temporal': _ok_envelope('temporal'),
            'facts': _ok_envelope('facts'),
        })
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            m.run_muster(_basic_ctx(self.brain))

        # Read trace events back from logs_conn
        rows = self.brain.logs_conn.execute(
            "SELECT chain_id, event_type, ref_type "
            "FROM trace_events "
            "WHERE scale = 's1' AND chain_id LIKE 's1e-%' "
            "AND ref_type IN ('scout_input', 'scout_findings')"
        ).fetchall()

        # 3 scouts * 2 event types each = 6 events
        self.assertEqual(len(rows), 6)
        inputs = [r for r in rows if r[2] == 'scout_input']
        findings = [r for r in rows if r[2] == 'scout_findings']
        self.assertEqual(len(inputs), 3)
        self.assertEqual(len(findings), 3)

    def test_scout_findings_carry_token_usage(self):
        """An LLM scout's per-call usage ('_usage' stub, API field names)
        rides into the K scout_findings metadata under the short USAGE_FIELDS
        names — the cost-tally contract. Scouts without usage (algo scouts,
        stubs) emit no token fields at all, so an all-agents tally never
        counts zero-rows from scouts that made no API call."""
        # Stub carries USAGE_FIELDS short names — produced by
        # runner.read_usage at capture (scouts/base step 6); muster copies
        # keys verbatim, no mapping anywhere.
        with_usage = _ok_envelope('facts')
        with_usage[m.SCOUT_TOKEN_USAGE_KEY] = {
            'input_tokens': 1200,
            'output_tokens': 340,
            'cache_creation_tokens': 800,
            'cache_read_tokens': 5600,
        }
        no_usage = _ok_envelope('temporal')       # algo scout — no LLM call
        no_usage.pop(m.SCOUT_TOKEN_USAGE_KEY, None)
        fake = _make_fake_runners({
            'quote': _ok_envelope('quote'),        # default stub usage (100/20)
            'temporal': no_usage,
            'facts': with_usage,
        })
        with patch.object(m, 'SCOUT_RUNNERS', fake):
            m.run_muster(_basic_ctx(self.brain))

        rows = self.brain.logs_conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE scale = 's1' AND ref_type = 'scout_findings'"
        ).fetchall()
        metas = [json.loads(r[0]) for r in rows]
        by_scout = {meta['scout']: meta for meta in metas}

        facts = by_scout['facts']
        self.assertEqual(facts['input_tokens'], 1200)
        self.assertEqual(facts['output_tokens'], 340)
        self.assertEqual(facts['cache_creation_tokens'], 800)
        self.assertEqual(facts['cache_read_tokens'], 5600)

        # Partial stubs spread verbatim — absent fields stay absent (the
        # tally's .get(field, 0) treats them as 0)
        quote = by_scout['quote']
        self.assertEqual(quote['input_tokens'], 100)
        self.assertEqual(quote.get('cache_read_tokens', 0), 0)

        # No usage stub → no token fields at all (tally never counts it)
        self.assertNotIn('input_tokens', by_scout['temporal'])
        self.assertNotIn('output_tokens', by_scout['temporal'])


if __name__ == '__main__':
    unittest.main()
