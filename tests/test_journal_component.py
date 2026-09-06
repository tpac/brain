"""Component tests for JournalBinding (servers/scales/journal.py) — the one
object that attaches the journal to any agent request.

Pins the Phase 2 contract: decorate order (arc → review → closure-if-loop),
harvest as write+strip (the single-shot envelope rule: journal sections are
stripped BEFORE the caller's JSON extraction, so a `]`/`}` inside a fence
can't corrupt extract_json's rfind-based scan), and failure isolation on the
continuity read.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase  # noqa: E402
from servers.scales.journal import JournalBinding  # noqa: E402
from servers.scales.runner import extract_json  # noqa: E402


FINAL_TEXT = (
    'Batch judged. Payload:\n[{"node": "abc", "score": 0.9}]\n\n'
    '## Review\n```\nfriction · abc · score capped at 0.9]\n```\n'
)


class TestHarvest(BrainTestBase):
    needs_embedder = False

    def _binding(self, **kw):
        kw.setdefault('scale', 's2')
        kw.setdefault('unit', 'consolidation')
        return JournalBinding(self.brain, **kw)

    def test_harvest_writes_notes_and_strips(self):
        b = self._binding()
        remainder = b.harvest(FINAL_TEXT, 's2-20260728000000-consolidation')

        notes = self.brain.journal_notes(scale='s2', unit='consolidation')
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['tag'], 'friction')
        self.assertEqual(notes[0]['subject'], 'abc')
        self.assertNotIn('## Review', remainder)

    def test_stripped_remainder_survives_extract_json(self):
        """The envelope rule, both belts: the harvest strip removes the
        journal fence before parsing, and extract_json is itself
        fence-robust (post-a108cfc-review it tries every fence + the
        fence-free remainder), so the payload parses from the stripped
        remainder AND from the raw text alike."""
        payload = [{"node": "abc", "score": 0.9}]
        self.assertEqual(extract_json(FINAL_TEXT), payload)

        b = self._binding()
        remainder = b.harvest(FINAL_TEXT, 's2-20260728000001-consolidation')
        self.assertNotIn('## Review', remainder)
        self.assertEqual(extract_json(remainder), payload)

    def test_arc_bound_harvest_writes_session_arc(self):
        text = ('done.\n\n## Arc\n```\nsurvivor ladder shipped\n```\n\n'
                '## Review\n```\n```\n')
        b = JournalBinding(self.brain, scale='s1', session_id='sess-arc',
                           arc=True)
        remainder = b.harvest(text, 's1e-sessarc-5')
        self.assertIn('survivor ladder shipped',
                      self.brain.get_config('session_context_sess-arc', ''))
        self.assertNotIn('## Arc', remainder)
        self.assertNotIn('## Review', remainder)

    def test_continuity_failure_isolated(self):
        """A broken notes read degrades to no continuity, never raises."""
        b = self._binding()
        orig = self.brain.journal_notes
        self.brain.journal_notes = None       # any call → TypeError
        try:
            self.assertEqual(b.continuity(), '')
        finally:
            self.brain.journal_notes = orig

    def test_continuity_round_trip(self):
        b = self._binding(unit='community_detection')
        b.harvest('## Review\n```\ndoubt · xyz · unsure about placement\n```',
                  's2-20260728000002-community_detection')
        prefix = b.continuity()
        self.assertIn('xyz', prefix)
        self.assertIn('RECENT REVIEW NOTES', prefix)


S1 = 'aaaaaaaa-1111-2222-3333-444444444444'
S2 = 'bbbbbbbb-1111-2222-3333-444444444444'
CHAIN = 's1e-aaaaaaaa-3'
ADDRESSED_TEXT = (
    'Encoded 3 nodes.\n\n## Review\n```\n'
    'friction · 1ca943af · turn-row projection built in two places\n'
    'tell · segment 6.a · you are proceeding on "I wonder if", not a yes\n'
    'ask · 7e6decd2 · milestone says merge pending; it merged — revise, or leave?\n'
    '```\n'
)


class TestAddressedRouting(BrainTestBase):
    """Step 13(b): a binding with a source files its tell/ask lines as
    Thalamus items (the non-LLM entrance to the one door) — never as
    journal_note rows; a session makes them directed, none makes them
    broadcast. Without a source the lines stay plain notes and warn. The
    filing rides the run chain (13d)."""
    needs_embedder = False

    def _scribe(self, source='encoder:sonnet', session_id=S1):
        return JournalBinding(self.brain, scale='s1', session_id=session_id,
                              source=source)

    def _items(self, **kw):
        from servers.channels.thalamus import thalamus
        kw.setdefault('source', 'encoder:sonnet')
        kw.setdefault('target_session', S1)
        return thalamus.list_items(self.brain, **kw)['items']

    def _warnings(self, source):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='warning' "
            "AND source = ?", (source,)).fetchone()[0]

    def test_tell_and_ask_become_items_not_notes(self):
        self._scribe().harvest(ADDRESSED_TEXT, CHAIN)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        self.assertEqual([n['tag'] for n in notes], ['friction'])
        items = {i['body']: i for i in self._items()}
        self.assertEqual(len(items), 2)
        tell = next(i for b, i in items.items() if b.startswith('you are'))
        ask = next(i for b, i in items.items() if b.startswith('milestone'))
        # The subject is always the dedup key ("one line per subject" holds
        # literally); only a node-id subject is also a ref.
        self.assertEqual((tell['needs_answer'], tell['dedup_key'], tell['refs']),
                         (False, 'segment 6.a', []))
        self.assertEqual((ask['needs_answer'], ask['dedup_key'], ask['refs']),
                         (True, '7e6decd2', ['7e6decd2']))
        self.assertTrue(all(i['target_session'] == S1 for i in items.values()))
        filed = [e for e in self.brain.query_traces(chain_id=CHAIN)['chain']
                 if e['ref_type'] == 'thalamus_filed']
        self.assertEqual(sorted(e['ref_id'] for e in filed),
                         sorted(i['id'] for i in items.values()))

    def test_directed_ask_reaches_the_session_at_stop(self):
        from servers.channels.thalamus import thalamus
        self._scribe().harvest(ADDRESSED_TEXT, CHAIN)
        block, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 2)
        self.assertIn('encoder:sonnet asks', block)
        self.assertIn('encoder:sonnet notes', block)

    def test_no_source_keeps_them_as_notes_and_warns(self):
        # No source (a bare eval binding) → plain residue, one loud warning.
        JournalBinding(self.brain, scale='s1', session_id=S1).harvest(
            ADDRESSED_TEXT, CHAIN)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        self.assertEqual(sorted(n['tag'] for n in notes),
                         ['ask', 'friction', 'tell'])
        self.assertEqual(self._items(), [])
        b = JournalBinding(self.brain, scale='s1', session_id=S1)
        self.assertEqual(self._warnings(b._log_key('addressed_unbound')), 1)

    def test_s2_unit_files_broadcast_delivered_at_boot(self):
        """An S2 binding has a source and no session: its ask is broadcast,
        due at every session's boot (not Stop); its tell reaches the first
        session to boot. Same instruction, same door, other audience."""
        from servers.channels.thalamus import thalamus
        JournalBinding(self.brain, scale='s2', unit='consolidation',
                       source='s2:consolidation').harvest(
            ADDRESSED_TEXT, 's2-20260905120000-consolidation')
        items = self._items(source='s2:consolidation', target_session='')
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i['target_session'] == '' for i in items))
        _, n_stop = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n_stop, 1)          # the tell (a notice) only
        _, n_boot = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_boot, 1)          # the ask
        _, n_other = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n_other, 1)         # every session sees the ask once
        filed = [e for e in self.brain.query_traces(
            chain_id='s2-20260905120000-consolidation')['chain']
                 if e['ref_type'] == 'thalamus_filed']
        self.assertEqual(len(filed), 2)
        self.assertTrue(all(e['scale'] == 's2' for e in filed))

    def test_rejection_is_loud_and_kept_as_residue(self):
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            thalamus.file(self.brain, 'encoder:sonnet', 'fill %d' % i,
                          for_whom=S1)
        b = self._scribe()
        b.harvest(ADDRESSED_TEXT, CHAIN)
        self.assertEqual(len(self._items()), tc.MAX_OPEN_PER_SOURCE)
        self.assertEqual(self._warnings(b._log_key('addressed_rejected')), 2)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        kept = [n for n in notes if n['undelivered']]
        self.assertEqual(sorted(n['tag'] for n in kept), ['ask', 'tell'])
        self.assertTrue(all('budget' in n['undelivered'] for n in kept))
        # The note is the line the encoder wrote — the reason is a field.
        self.assertIn('you are proceeding on "I wonder if", not a yes',
                      [n['note'] for n in kept])

    def test_resolved_line_withdraws_the_open_item(self):
        """The encoder's own verb closes its own item — with the read side's
        tolerance: an echoed `tag · subject · …` head (the subject lands in
        the note) and a case change still find the open key."""
        self._scribe().harvest(ADDRESSED_TEXT, CHAIN)
        self.assertEqual(len(self._items()), 2)
        self._scribe().harvest(
            'done.\n\n## Review\n```\n'
            'resolved · ask · 7e6decd2 · Tom said revise\n'
            'resolved · Segment 6.A · confirmed\n'
            '```\n', 's1e-aaaaaaaa-4')
        self.assertEqual(self._items(), [])
        states = {i['dedup_key']: i['state']
                  for i in self._items(include_closed=True)}
        self.assertEqual(states, {'7e6decd2': 'withdrawn',
                                  'segment 6.a': 'withdrawn'})

    def test_routing_failure_is_isolated_and_the_arc_survives(self):
        """A Thalamus read hiccup during routing must neither escape harvest
        (S2's batch loop relies on it) nor cost the session arc."""
        from unittest import mock
        from servers.channels.thalamus import thalamus
        b = JournalBinding(self.brain, scale='s1', session_id=S1,
                           source='encoder:sonnet', arc=True)
        text = ('done.\n\n## Arc\n```\nrouting survived\n```\n\n' +
                ADDRESSED_TEXT)
        with mock.patch.object(thalamus, 'file',
                               side_effect=RuntimeError('database is locked')):
            remainder = b.harvest(text, CHAIN)
        self.assertNotIn('## Review', remainder)
        self.assertIn('routing survived',
                      self.brain.session_context_for(S1) or '')
        # Every filing failed loudly and was kept as residue.
        b2 = self._scribe()
        self.assertEqual(self._warnings(b2._log_key('addressed_rejected')), 2)
        kept = [n for n in self.brain.journal_notes(scale='s1', session_id=S1)
                if n['undelivered']]
        self.assertEqual(len(kept), 2)

    def test_same_subject_in_two_sessions_is_two_items(self):
        """The Scribe's source is one string for every session, so the
        identity a re-file matches on carries the target: session B's ask
        under the same subject is B's item, not a retarget of A's."""
        self._scribe().harvest(ADDRESSED_TEXT, CHAIN)
        JournalBinding(self.brain, scale='s1', session_id=S2,
                       source='encoder:sonnet').harvest(
            ADDRESSED_TEXT, 's1e-bbbbbbbb-3')
        a = {i['dedup_key']: i for i in self._items(target_session=S1)}
        bb = {i['dedup_key']: i for i in self._items(target_session=S2)}
        self.assertEqual(set(a), set(bb))
        self.assertNotEqual(a['7e6decd2']['id'], bb['7e6decd2']['id'])
        # B's resolve closes B's item only.
        JournalBinding(self.brain, scale='s1', session_id=S2,
                       source='encoder:sonnet').harvest(
            '## Review\n```\nresolved · 7e6decd2 · done\n```\n',
            's1e-bbbbbbbb-4')
        self.assertIn('7e6decd2', {i['dedup_key'] for i in
                                   self._items(target_session=S1)})
        self.assertNotIn('7e6decd2', {i['dedup_key'] for i in
                                      self._items(target_session=S2)})

    def test_dark_ship_no_addressed_lines_no_items(self):
        """Without the prompt paragraph no encoder writes the verbs — a
        review block with none files nothing and traces nothing."""
        self._scribe().harvest(
            'Encoded.\n\n## Review\n```\nfriction · 1ca943af · drift risk\n```\n',
            CHAIN)
        self.assertEqual(self._items(), [])
        self.assertEqual(
            [e for e in self.brain.query_traces(chain_id=CHAIN)['chain']
             if e['ref_type'] == 'thalamus_filed'], [])


class TestSingleShotCallLlm(BrainTestBase):
    """The Phase-3 wiring: _call_llm(journal=True) is the single attachment
    point for single-shot units (healer, aspect) — review block on the system
    tail (no closure, no arc), harvest before extract_json."""
    needs_embedder = False

    HEALING_JSON = '[{"node_id": "abcd1234", "question": "why does x?"}]'
    RESPONSE = (HEALING_JSON +
                '\n\n## Review\n```\ndoubt · abcd1234 · thin conversation]\n```\n')

    def _unit(self):
        from servers.scales.s2.healer_encoder import HealerEncoder
        return HealerEncoder(self.brain)

    def _patch_llm(self, response_text):
        """Swap run_llm_once in base's namespace; returns the capture dict."""
        from servers.scales.s2 import base as base_mod
        captured = {}

        def fake_run_llm_once(client, model, max_tokens, system_prompt,
                              user_content):
            captured['system'] = system_prompt
            captured['user'] = user_content
            return response_text, {'elapsed_ms': 1, 'input_tokens': 1,
                                   'output_tokens': 1, 'cache_read_tokens': 0,
                                   'cache_creation_tokens': 0}

        self._orig_run_once = base_mod.run_llm_once
        base_mod.run_llm_once = fake_run_llm_once
        self.addCleanup(lambda: setattr(base_mod, 'run_llm_once',
                                        self._orig_run_once))
        # _llm_client builds a real Anthropic client — stub it out too.
        self._orig_make_client = base_mod.make_client
        base_mod.make_client = lambda: object()
        self.addCleanup(lambda: setattr(base_mod, 'make_client',
                                        self._orig_make_client))
        return captured

    def test_journal_true_decorates_and_harvests(self):
        from servers.trace_contract import JOURNAL_REVIEW_INSTRUCTION
        unit = self._unit()
        captured = self._patch_llm(self.RESPONSE)

        parsed, tel = unit._call_llm('s2_healer', 'payload', journal=True)

        # Decoration: review block at the system tail; single-shot = no
        # closure, no arc.
        base_prompt = self.brain.get_interaction_prompt('s2_healer')
        self.assertTrue(captured['system'].startswith(base_prompt.rstrip()))
        self.assertIn(JOURNAL_REVIEW_INSTRUCTION, captured['system'])
        self.assertNotIn('## Finishing', captured['system'])
        self.assertNotIn('## Arc', captured['system'])

        # Harvest ordering: the fence's `]` would poison extract_json on the
        # raw text — the parsed payload proves strip-first.
        self.assertEqual(parsed, [{"node_id": "abcd1234",
                                   "question": "why does x?"}])

        # The note landed on THIS run's chain, scoped to the unit.
        notes = self.brain.journal_notes(scale='s2', unit='healer')
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['tag'], 'doubt')
        self.assertEqual(notes[0]['chain_id'], unit.chain_id())

    def test_journal_default_is_byte_identical_and_silent(self):
        unit = self._unit()
        captured = self._patch_llm(self.HEALING_JSON)

        parsed, tel = unit._call_llm('s2_healer', 'payload')

        self.assertEqual(captured['system'],
                         self.brain.get_interaction_prompt('s2_healer'))
        self.assertEqual(parsed, [{"node_id": "abcd1234",
                                   "question": "why does x?"}])
        self.assertEqual(self.brain.journal_notes(scale='s2', unit='healer'),
                         [])

    def test_healer_run_binds_journal(self):
        """End-to-end through HealerEncoder.run(): continuity prepended,
        journal=True on the call, residue persisted alongside the field write."""
        nid = self.brain.remember(
            type='fact', title='journal probe target',
            content='node missing a question')['id']
        unit = self._unit()
        # Seed one prior note so continuity has something to prepend.
        unit.journal.harvest(
            '## Review\n```\nfriction · prior · earlier run note\n```',
            's2-20260101000000-healer')
        response = ('[{"node_id": "%s", "question": "why probe?"}]' % nid[:8] +
                    '\n\n## Review\n```\nsurprise · %s · conversation absent\n```'
                    % nid[:8])
        captured = self._patch_llm(response)

        result = unit.run([{'node_id': nid, 'needs_question': True,
                            'rich_node': {'title': 'journal probe target',
                                          'content': 'node missing a question'},
                            'conversation': []}])

        self.assertEqual(result['nodes_healed'], 1)
        self.assertIn('RECENT REVIEW NOTES', captured['user'])
        self.assertIn('earlier run note', captured['user'])
        notes = self.brain.journal_notes(scale='s2', unit='healer')
        tags = {n['tag'] for n in notes}
        self.assertIn('surprise', tags)

    def test_out_of_batch_healing_dropped(self):
        """Review a108cfc finding #1: the continuity prefix shows ids from
        past runs; a healing for an id outside the batch has no needs_* flags
        and must be dropped, never written."""
        victim = self.brain.remember(
            type='fact', title='out-of-batch victim',
            content='node with an earned reasoning',
            reasoning='earned reasoning that must survive')['id']
        unit = self._unit()
        response = ('[{"node_id": "%s", "reasoning": "hallucinated"}]'
                    % victim[:8])
        self._patch_llm(response)

        # The batch proposes a DIFFERENT node — the victim id arrives only
        # in the model's output (as if read from the continuity prefix).
        other = self.brain.remember(type='fact', title='batch member',
                                    content='the actual proposal')['id']
        result = unit.run([{'node_id': other, 'needs_question': True,
                            'rich_node': {'title': 'batch member',
                                          'content': 'the actual proposal'},
                            'conversation': []}])

        self.assertEqual(result['nodes_healed'], 0)
        self.assertEqual(result['skipped'], 1)
        node = self.brain.get_node(victim)
        self.assertEqual(node['_metadata'].get('reasoning'),
                         'earned reasoning that must survive')

    def test_headingless_fence_payload_survives(self):
        """Review a108cfc finding #2: a notes fence that lost its ## Review
        heading (the documented Haiku drift) must not poison the payload —
        extract_json now tries every fence and the fence-free remainder."""
        from servers.scales.runner import extract_json
        drifted = ('[{"node_id": "ab12cd34", "question": "why?"}]\n\n'
                   '```\ndoubt · ab12cd34 · thin conversation]\n```')
        self.assertEqual(extract_json(drifted),
                         [{"node_id": "ab12cd34", "question": "why?"}])

    def test_salvage_refuses_json_payload_fence(self):
        """A single-shot payload fence whose strings contain '·' must not be
        harvested as residue notes."""
        from servers.trace_contract import salvage_review_fence
        text = ('```json\n[{"category": "node_types", "value": "probe", '
                '"rationale": "generative · lesson-bearing · keep"}]\n```')
        self.assertIsNone(salvage_review_fence(text))

    def test_harvest_failure_degrades_to_raw(self):
        """A journal-layer fault must never discard a paid response — the
        payload parses from the unstripped raw."""
        from servers.scales.journal import JournalBinding

        class _BoomBinding(JournalBinding):
            def harvest(self, *a, **kw):
                raise RuntimeError('journal layer down')

        unit = self._unit()
        self._patch_llm(self.RESPONSE)
        unit._journal_binding = _BoomBinding(self.brain, scale='s2',
                                             unit='healer')
        parsed, tel = unit._call_llm('s2_healer', 'payload', journal=True)
        self.assertEqual(parsed, [{"node_id": "abcd1234",
                                   "question": "why does x?"}])

    def test_aspect_run_binds_journal(self):
        """AspectEncoder.run() reaches _call_llm with journal=True and the
        continuity prefix on the user content."""
        from servers.scales.s2.aspect_encoder import AspectEncoder
        unit = AspectEncoder(self.brain)
        seen = {}

        def fake_call(name, user_content, journal=False):
            seen['name'] = name
            seen['journal'] = journal
            seen['user'] = user_content
            return None, {'elapsed_ms': 0}

        unit._call_llm = fake_call
        unit.journal.harvest(
            '## Review\n```\nopen · taxonomy · single-example candidates\n```',
            's2-20260101000000-aspect_integration')

        unit.run([{'category': 'node_types', 'value': 'probe_type',
                   'count': 1, 'examples': []}])

        self.assertEqual(seen['name'], 's2_aspects')
        self.assertTrue(seen['journal'])
        self.assertIn('single-example candidates', seen['user'])


if __name__ == '__main__':
    unittest.main()
