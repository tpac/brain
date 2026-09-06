"""Tests for render_rich_node() — the standard node renderer for LLM consumers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.contract import render_rich_node, NODE_FORMAT_DEFAULTS


class TestFormatNode(BrainTestBase):
    needs_embedder = False

    def _make_node(self, **kwargs):
        """Create a node and return its full ID."""
        defaults = dict(type='rule', title='Test node', content='Test content')
        defaults.update(kwargs)
        result = self.brain.remember(**defaults)
        return result['id']

    def _add_edge(self, source_id, target_id, relation='related_to',
                  weight=0.8, description=''):
        """Insert an edge using the new multi-relation model."""
        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        dal.add_relation(source_id, target_id, relation, description, weight)

    def _render(self, node_id, config=None):
        """Fetch node + render: the two-step pattern replacing format_node()."""
        node = self.brain.get_node(node_id)
        if not node:
            return None
        return render_rich_node(node, config)

    def _stamp_relation(self, source_id, target_id, ts, column='created_at'):
        """Backdate (or clear) a relation's timestamp on the (source, target) pair."""
        self.brain.conn.execute(
            "UPDATE edge_relations SET %s = ? WHERE edge_id = "
            "(SELECT edge_id FROM edges WHERE source_id = ? AND target_id = ?)" % column,
            (ts, source_id, target_id))
        self.brain.conn.commit()

    def _edge_line(self, node_id, relation, config=None):
        """The rendered edge line carrying `relation` (absolute dates by default)."""
        out = self._render(node_id, config)
        return next(l for l in out.split('\n') if l.startswith('    [') and relation in l)

    # ── Basic rendering ──

    def test_full_node(self):
        """Full node with core fields renders header + content + keywords."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres for reliability.',
            keywords='db postgres sql', confidence=0.9, locked=True)
        out = self._render(nid)
        self.assertIsNotNone(out)
        self.assertIn('[decision]', out)
        self.assertIn('"Use Postgres"', out)
        # Confidence display rooted out 2026-05-31: show_confidence defaults
        # off (the field is dormant — read by no ranking path). Default render
        # must NOT show it. Same strictness, opposite contract.
        self.assertNotIn('conf:', out)
        self.assertIn('locked', out)
        self.assertIn('We chose Postgres', out)
        # Keywords column dropped in schema v28; render block removed.
        # Asserting absence (same strictness, opposite contract).
        self.assertNotIn('Keywords:', out)

    def test_confidence_renders_when_enabled(self):
        """Confidence is hidden by default but still renders when a caller
        explicitly opts in via show_confidence=True (the dormant field's
        opt-in path stays covered after the 2026-05-31 default flip)."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres.',
            confidence=0.9)
        node = self.brain.get_node(nid)
        out = render_rich_node(node, {'show_confidence': True})
        self.assertIn('conf:0.9', out)

    def test_nonexistent_node_returns_none(self):
        """Render returns None for an ID that doesn't exist."""
        out = self._render('nonexistent-id-12345678')
        self.assertIsNone(out)

    # ── Header format ──

    def test_header_includes_id_prefix(self):
        """Header shows first 8 chars of node ID."""
        nid = self._make_node(type='lesson', title='Header test')
        out = self._render(nid)
        self.assertIn('id:%s' % nid[:8], out)

    def test_header_unlocked_no_locked_flag(self):
        """Unlocked nodes don't show ', locked' in header."""
        nid = self._make_node(locked=False)
        out = self._render(nid)
        self.assertNotIn('locked', out)

    def test_header_encoding_source(self):
        """encoding_source appears in header when set."""
        nid = self._make_node(encoding_source='encoder:sonnet')
        out = self._render(nid)
        self.assertIn('src:encoder:sonnet', out)

    # ── Content truncation ──

    def test_content_limit_truncates(self):
        """content_limit config truncates long content."""
        long_content = 'A' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid, config={'content_limit': 50})
        # _truncate caps at <= limit chars INCLUDING the ellipsis
        # (s[:limit-1] + '…'), so a 500-char body renders as 49 A's + '…'.
        self.assertIn('A' * 49 + '…', out)
        self.assertNotIn('A' * 50, out)

    def test_no_content_limit_shows_full(self):
        """Default (content_limit=None) shows full content."""
        long_content = 'B' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid)
        self.assertIn('B' * 500, out)

    # ── Metadata ──

    def test_metadata_situation(self):
        """Situation from node_metadata_kv (canonical store) is rendered.

        v24+: situation lives in node_metadata_kv, not the removed
        node_embeddings table. Stronger than the old test — verifies
        both the canonical storage location AND the rendered output.
        """
        sit = 'When choosing a database for OLTP workloads'
        nid = self._make_node(situation=sit)
        # Verify storage location — situation must be in kv, not elsewhere
        kv_row = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id=? AND key='situation'",
            (nid,)).fetchone()
        self.assertIsNotNone(kv_row, 'situation should be stored in node_metadata_kv')
        self.assertEqual(kv_row[0], sit, 'kv value must match the written situation exactly')
        # Verify render
        out = self._render(nid)
        self.assertIn('Situation: When choosing a database', out)
        self.assertIn(sit, out)  # full exact-match, stronger than substring

    def test_metadata_reasoning(self):
        """Reasoning from metadata_kv is rendered."""
        nid = self._make_node(reasoning='Postgres has better JSON support than MySQL')
        out = self._render(nid)
        self.assertIn('Reasoning:', out)
        self.assertIn('Postgres has better JSON support', out)

    def test_correction_edge_annotations(self):
        """A `corrects` edge surfaces correction context on both endpoints.

        correction_enrich walks correction_improvement-aspect edges
        (corrects, supersedes, reframes, ...). The renderer annotates
        the corrector's view with 'Corrects:' and the corrected node's
        view with 'Updated by:'.
        """
        original_id = self._make_node(title='Use MySQL')
        correction_id = self._make_node(title='Use Postgres instead')
        # Edge: correction_id corrects original_id
        self.brain.connect_typed(
            source_id=correction_id, target_id=original_id,
            relation='corrects', weight=0.5,
            description='Postgres reliability beats MySQL for this workload',
            encoding_source='test:correction_edge_annotations')

        # Corrector's view: 'Corrects:' the original
        out_corrector = self._render(correction_id)
        self.assertIn('Corrects:', out_corrector)
        self.assertIn('Use MySQL', out_corrector)
        # Corrected node's view: 'Updated by:' the correction
        out_original = self._render(original_id)
        self.assertIn('Updated by:', out_original)
        self.assertIn('Use Postgres', out_original)

    # ── Edges ──

    def test_edges_shown(self):
        """Edges appear with target title and relation."""
        nid = self._make_node(title='Source node')
        target_id = self._make_node(type='mechanism', title='Target node')
        self._add_edge(nid, target_id, relation='depends_on', weight=0.9,
                       description='runtime dependency')
        out = self._render(nid)
        self.assertIn('Edges:', out)
        self.assertIn('"Target node"', out)
        self.assertIn('depends_on', out)
        self.assertIn('runtime dependency', out)

    def test_edge_limit_config(self):
        """edge_limit config caps the number of edges shown."""
        nid = self._make_node(title='Hub node')
        for i in range(6):
            tid = self._make_node(title='Spoke %d' % i)
            self._add_edge(nid, tid, weight=0.5 + i * 0.01)
        # Default limit is 5 — should show 5 of 6
        # Edge lines rendered as '    [type id:XX date] ...' (4-space indent + bracket)
        out = self._render(nid)
        edge_lines = [l for l in out.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines), 5)
        # With limit 2
        out2 = self._render(nid, config={'edge_limit': 2})
        edge_lines2 = [l for l in out2.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines2), 2)

    def test_edge_line_age_is_the_relations_not_the_neighbors(self):
        """The age in an edge line's bracket is when THIS relation was written,
        never the neighbor node's age — which used to sit there unlabeled and
        read as the edge's."""
        nid = self._make_node(title='Owner')
        old_nbr = self._make_node(title='Ancient neighbor')
        self.brain.conn.execute("UPDATE nodes SET created_at = ? WHERE id = ?",
                                ('2020-01-01T00:00:00+00:00', old_nbr))
        self._add_edge(nid, old_nbr, relation='grounds', description='written today')
        edge_line = self._edge_line(nid, 'grounds')   # absolute dates
        self.assertNotIn('2020', edge_line)
        from servers.clock import iso_now
        self.assertIn(iso_now()[:10], edge_line)

    def test_get_node_excludes_noise_relations_for_every_reader(self):
        """The read exclusion lives in get_node (registry structural_exclusions
        = the noise aspect), so every reader of `connections` — Anchor, the
        recall surface, the encoder catalog, the healer, consolidation — is
        noise-free without a filter of its own. A pair that carries a noise
        relation AND a semantic one keeps the semantic line; a pair that is
        noise only disappears; the compat `relation` is never a noise verb."""
        self.assertIn('community_member', self.brain.aspects.structural_exclusions)
        nid = self._make_node(title='Member')
        comm = self._make_node(type='community', title='A community')
        peer = self._make_node(title='Peer')
        self._add_edge(comm, nid, relation='community_member', weight=0.9)
        self._add_edge(nid, peer, relation='co_anchored', weight=0.9,
                       description='shared episodic anchor')
        self._add_edge(nid, peer, relation='extends', weight=0.5,
                       description='the semantic claim')
        node = self.brain.get_node(nid)
        self.assertEqual([c['id'] for c in node['connections']], [peer])
        (conn,) = node['connections']
        self.assertEqual([r['relation'] for r in conn['relations']], ['extends'])
        self.assertEqual(conn['relation'], 'extends')
        out = render_rich_node(node)
        self.assertNotIn('community_member', out)
        self.assertNotIn('co_anchored', out)
        self.assertIn('this extends "Peer" — the semantic claim', out)

    def test_communities_ride_as_their_own_line_where_a_format_opts_in(self):
        """Community membership is a `communities` attachment on the canonical
        pull and renders as ONE `Communities:` line — never as edge lines
        (community_member is noise-excluded from connections). Off by default
        and for the encoder catalog; on for Anchor's get_nodes formats and
        the recall surface Anchor reads."""
        from servers.contract import GET_NODES_SMALL_FORMAT, GET_NODES_FULL_FORMAT
        from servers.scales.s1.encode_contract import S1_NODE_CONFIG
        from servers.scales.s1.surface_contract import (
            SURFACE_ARC_FORMAT, HAIKU_FORMAT, resolve_surface_format)
        nid = self._make_node(title='Member')
        comm = self._make_node(type='community', title='A community')
        self._add_edge(comm, nid, relation='community_member', weight=0.9)
        node = self.brain.get_node(nid)
        self.assertEqual(node['communities'], [{'id': comm, 'title': 'A community'}])
        self.assertEqual(node['connections'], [])
        # the same member pulled in one batch WITH its community keeps its
        # placement — the lookup decides which endpoint is the community by
        # the node's type, not by which id the caller asked for
        batch = self.brain.get_node([nid, comm])
        self.assertEqual(batch[nid]['communities'], [{'id': comm, 'title': 'A community'}])
        self.assertEqual(batch[comm]['communities'], [])
        line = '  Communities: "A community" (id:%s)' % comm[:8]
        self.assertNotIn('Communities:', render_rich_node(node))
        self.assertNotIn('Communities:', render_rich_node(node, S1_NODE_CONFIG))
        for cfg in (GET_NODES_SMALL_FORMAT, GET_NODES_FULL_FORMAT, HAIKU_FORMAT,
                    resolve_surface_format(SURFACE_ARC_FORMAT, 1000)):
            out = render_rich_node(node, cfg)
            self.assertIn(line, out)
            self.assertNotIn('community_member', out)
        # a node in no community renders no empty line
        lonely = self._make_node(title='Lonely')
        self.assertEqual(self.brain.get_node(lonely)['communities'], [])
        self.assertNotIn('Communities:', render_rich_node(
            self.brain.get_node(lonely), GET_NODES_SMALL_FORMAT))

    def test_flat_weights_break_ties_by_relation_recency_and_the_cut_says_so(self):
        """Weights are flat in production (0.5/0.6 everywhere), so the top-N
        cut used to be a tie broken by SQL row order. get_node orders equal
        weights by the relation's created_at, newest first, and Anchor's
        small/balanced formats say when the cut dropped edges: 'Edges (6 of 7)'.
        (The encoder catalog says it too, through the view policy's cfg —
        pinned in test_encoder_view.)"""
        from servers.contract import GET_NODES_SMALL_FORMAT, GET_NODES_BALANCED_FORMAT
        hub = self._make_node(title='Hub')
        spokes = [self._make_node(title='Spoke %d' % i) for i in range(7)]
        for i, s in enumerate(spokes):
            self._add_edge(hub, s, relation='extends', weight=0.6,
                           description='claim %d' % i)
            # the relation's birth: spoke 0 oldest ... spoke 6 newest
            self._stamp_relation(hub, s, '2026-01-%02dT00:00:00+00:00' % (i + 1))
        node = self.brain.get_node(hub)
        self.assertEqual([c['id'] for c in node['connections']], spokes[::-1])
        out = render_rich_node(node, GET_NODES_BALANCED_FORMAT)     # limit 6
        self.assertIn('  Edges (6 of 7):', out)
        self.assertIn('claim 6', out)
        self.assertNotIn('claim 0', out)          # the oldest is the one cut
        # a limit that does not cut (8) keeps the bare header
        self.assertIn('  Edges:\n', render_rich_node(node, GET_NODES_SMALL_FORMAT))
        # the recency that orders is the recency the line prints: a claim
        # REPAIRED today outranks its untouched siblings even though it was
        # born first
        self._stamp_relation(hub, spokes[0], '2026-09-01T00:00:00+00:00', column='updated_at')
        self.assertEqual(self.brain.get_node(hub)['connections'][0]['id'], spokes[0])
        # a higher weight still wins over recency
        heavy = self._make_node(title='Heavy')
        self._add_edge(hub, heavy, relation='grounds', weight=0.9, description='heavy')
        self._stamp_relation(hub, heavy, '2020-01-01T00:00:00+00:00')
        self.assertEqual(self.brain.get_node(hub)['connections'][0]['id'], heavy)
        # a pair's weight is the max over its SURVIVING relations: a heavy
        # noise relation on a weak pair must not lift it above semantic peers
        noisy = self._make_node(title='Noisy')
        self._add_edge(hub, noisy, relation='extends', weight=0.5, description='weak claim')
        self._add_edge(hub, noisy, relation='co_anchored', weight=0.95)
        conns = self.brain.get_node(hub)['connections']
        self.assertEqual(conns[-1]['id'], noisy)
        self.assertEqual(conns[-1]['weight'], 0.5)

    def test_edge_line_age_is_the_repair_when_the_claim_changed(self):
        """A description repaired in place used to keep its birth date on the
        line — the age lied after exactly the repair we want. edge_relations
        now carries updated_at, stamped only when the claim changes
        (description / weight via the upsert, the verb via rename); a no-op
        re-connect leaves it NULL and the line keeps the honest birth date."""
        from servers.clock import iso_now
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title='Neighbor')
        # seed through the same door the assertions exercise, so the
        # re-connect below is a true no-op whatever weight it resolves
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='asserted 9.6.0', encoding_source='test')
        self._stamp_relation(nid, nbr, '2020-01-01T00:00:00+00:00')

        def _row():
            return self.brain.conn.execute(
                "SELECT er.relation, er.updated_at FROM edge_relations er JOIN edges e "
                "ON e.edge_id = er.edge_id WHERE e.source_id = ? AND e.target_id = ? "
                "AND er.archived = 0", (nid, nbr)).fetchone()

        # untouched: birth date on the line, no stamp
        self.assertIsNone(_row()[1])
        self.assertIn('2020', self._edge_line(nid, 'gaps_in'))
        # a re-connect that changes nothing is a true no-op
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='asserted 9.6.0', encoding_source='test')
        self.assertIsNone(_row()[1])
        # the repair stamps, and the line's age is the repair
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='moved to 9.7.2', encoding_source='test')
        self.assertTrue(_row()[1])
        line = self._edge_line(nid, 'gaps_in')
        self.assertNotIn('2020', line)
        self.assertIn(iso_now()[:10], line)
        self.assertIn('moved to 9.7.2', line)
        # renaming the verb is a claim change too
        self._stamp_relation(nid, nbr, None, column='updated_at')
        self.brain.revise_edge(nid, nbr, 'gaps_in', new_relation='closes',
                               encoding_source='test')
        relation, stamped = _row()
        self.assertEqual(relation, 'closes')
        self.assertTrue(stamped)

    def test_edge_lines_one_per_relation_descriptions_whole(self):
        """A pair carrying several relations renders one line per relation,
        each with its own description untruncated — a reader may copy it
        verbatim as a swap's `old`."""
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title='Neighbor')
        long_desc = 'x' * 250 + ' the load-bearing tail'
        self._add_edge(nid, nbr, relation='extends', description=long_desc)
        self._add_edge(nid, nbr, relation='grounds', description='second claim')
        out = self._render(nid)
        edge_lines = [l for l in out.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines), 2)
        self.assertTrue(any(l.endswith(' — ' + long_desc) for l in edge_lines), edge_lines)
        self.assertTrue(any(l.endswith(' — second claim') for l in edge_lines))
        self.assertTrue(all(' | ' not in l for l in edge_lines))
        # incoming direction still reads actor-first
        out_nbr = self._render(nbr)
        self.assertIn('"Owner" extends this — ' + long_desc, out_nbr)

    def test_differential_project_mark_on_mismatch(self):
        """cfg['scope']: foreign project renders the ⚠ mark, the generic
        'Project:' KV line is suppressed."""
        nid = self._make_node(title='Foreign fact', project='exco')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertIn('⚠ From another project: exco', out)
        self.assertNotIn('Project: exco', out)

    def test_differential_project_silent_on_match(self):
        """Same-project node renders NO project line at all in differential
        mode — a same-project line on a one-project corpus is noise."""
        nid = self._make_node(title='Home fact', project='brain')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)
        self.assertNotIn('Project:', out)

    def test_differential_project_neutral_on_unscoped_node(self):
        """A node with no project provenance is never marked foreign —
        unknown is neutral, matching the scope lane semantics."""
        nid = self._make_node(title='Unscoped fact')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)

    def test_differential_counterpart_mark_on_mismatch(self):
        """The counterpart dimension marks through the SAME central
        scope_marks path — adding a dimension re-threads nothing."""
        nid = self._make_node(title='Other-speaker fact', counterpart='Dana')
        out = self._render(nid, {'scope': {'project': 'brain',
                                           'counterpart': 'Ada'}})
        self.assertIn('⚠ Learned with another counterpart: Dana', out)
        self.assertNotIn('Counterpart: Dana', out)

    def test_differential_counterpart_silent_on_match(self):
        nid = self._make_node(title='Same-speaker fact', counterpart='Ada')
        out = self._render(nid, {'scope': {'counterpart': 'Ada'}})
        self.assertNotIn('another counterpart', out)
        self.assertNotIn('Counterpart:', out)

    def test_legacy_render_keeps_generic_project_line(self):
        """Callers that don't declare a scope keep the pre-existing generic
        KV render — no information loss for unwired consumers."""
        nid = self._make_node(title='Legacy view', project='exco',
                              counterpart='Ada')
        out = self._render(nid)
        self.assertIn('Project: exco', out)
        self.assertNotIn('From another project', out)
        # counterpart is differential-ONLY: its value is the install default
        # (identical on every node), so the generic KV line is pure noise
        # and stays suppressed even for undeclared callers.
        self.assertNotIn('Counterpart:', out)

if __name__ == '__main__':
    unittest.main()
