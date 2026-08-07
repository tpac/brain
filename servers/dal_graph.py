"""
brain — Data Access Layer (DAL): graph (edges & relations, brain.db)

GraphDAL owns the v22 two-table edge model: `edges` (physical row per
pair — edge_id PK, source_id/target_id, aggregate weight) and
`edge_relations` (multiple semantic relations per edge_id FK). Split out
of dal.py, which holds the rest of the brain.db-backed classes (nodes,
vectors, metadata, ...).

Usage in brain.py:
    from servers.dal_graph import GraphDAL

    self._graph = GraphDAL(self.conn)
"""

import sqlite3
from typing import Any, Dict, List, Optional

from .clock import iso_now
from .db_backends.sqlite import commit_unless_batched


# ═══════════════════════════════════════════════════════════════
# GRAPH QUERY CONTRACT
# ═══════════════════════════════════════════════════════════════
# Every edge-reading method in GraphDAL conforms to this shape and
# these defaults. Centralizes what the rest of the code can assume.
# When we change what "an edge" means, we change it HERE once, and
# every consumer inherits the update.

# Canonical edge-row shape returned by GraphDAL reads. Node-centric:
# each row is a (owner → neighbor) relationship from the queried
# node's perspective. Matches get_neighbors output.
EDGE_ROW_SHAPE = {
    # Neighbor node fields (the node on the OTHER side of the edge)
    'id':                 'str  — neighbor node_id',
    'type':               'str  — neighbor node type',
    'title':              'str  — neighbor title',
    'content_summary':    'str  — neighbor content summary (may be None)',
    'confidence':         'float',
    'locked':             'int (0|1)',
    'created_at':         'str ISO',
    'revised_at':         'str ISO or None',
    # Edge metadata
    'edge_id':            'str — stable pair hash',
    'relation':           'str — typed relation name',
    'edge_description':   'str — relation description',
    'weight':             'float — edge-aggregate weight',
    'direction':          "str — 'outgoing' | 'incoming' from queried node",
    # Optional (present on richer methods)
    'last_accessed':      'str ISO',
    'access_count':       'int',
    'emotion':            'float',
    'emotion_label':      'str',
    'last_strengthened':  'str ISO',
    'co_access_count':    'int',
    'content_preview':    'str — substr of content when caller requests',
}

# Relations considered noise for semantic edge queries. Default-excluded
# by callers that want knowledge edges only. Override per-call when you
# need co_accessed (fatigue) or emergent_bridge (auto-links).
# community_member is NOT in this default — it's real thematic context,
# just not migrated by consolidation (see ABSORB_EXCLUDED_RELATIONS).
DEFAULT_EXCLUDED_RELATIONS = frozenset(['co_accessed', 'emergent_bridge'])
# Minimum edge-description length to feed the edge_context embedding. Single
# source of truth shared by GraphDAL.get_edge_descriptions_for (the text
# producer) and VectorDAL.find_missing (the backfill candidate filter) — the
# two MUST agree, or find_missing queues edgeless/short-desc nodes that yield
# no text, and they starve the edged nodes out of the backfill batch forever.
EDGE_CONTEXT_MIN_DESC_LENGTH = 10

# Relations absorb() must NOT migrate to the survivor. Community placement
# is the community unit's judged decision (affinity gate ≥0.25 + encoder
# accept/reject + drift detection re-evaluation) — a merge inheriting the
# absorbed node's membership would bypass all three. The absorbed node is
# archived, so its membership edge dies with it (dangling-edge restorer);
# the survivor gets (re-)placed through the normal community cycle, scored
# on the semantic edges the absorb just enriched. Audit 2026-06-12: the
# consolidation prompt + the comment above stated this exclusion as fact
# while the code migrated everything — this constant makes it true.
ABSORB_EXCLUDED_RELATIONS = frozenset(['community_member'])

# When `include_archived=False` is the default, every edge-reading method
# filters `archived = 0` in its WHERE clause. v25 added the column;
# this contract is the reason the filter lives in one place.


def _relation_not_in_clause(values):
    """Build an `AND relation NOT IN (?,?...)` SQL fragment + its param list for
    exempting relations from an edge-archival UPDATE. Returns ('', []) when
    empty. Single source for the survivor_lineage exemption shared by
    delete_node_edges + archive_dangling_edges, so the clause shape can't drift
    between them."""
    vals = list(values or ())
    if not vals:
        return '', []
    return 'AND relation NOT IN (%s)' % ','.join('?' * len(vals)), vals


class GraphDAL:
    """Access layer for brain.db graph tables: edges + edge_relations.

    ALL edge SQL lives here. When we move to in-memory graph, swap this
    implementation — nothing else changes. Every edge-reading method
    honors the GRAPH QUERY CONTRACT above:
      - Returns EDGE_ROW_SHAPE dicts (node-centric, neighbor fields flat)
      - Defaults include_archived=False (v25 soft-archive filter)
      - Accepts exclude_relations set to drop noise (see
        DEFAULT_EXCLUDED_RELATIONS for the standard noise set)

    Raises on invalid args — no silent empty returns masking bad calls.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def count_total(self) -> int:
        """Count total edges."""
        row = self.conn.execute('SELECT COUNT(*) FROM edges').fetchone()
        return row[0] if row else 0

    def get_edge_id(self, source_id: str, target_id: str) -> Optional[str]:
        """Get edge_id for a pair (checks both directions)."""
        row = self.conn.execute(
            'SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)
        ).fetchone()
        return row[0] if row else None

    def get_neighbors(self, node_id: str, limit: int = 8,
                      exclude_relations: set = None,
                      exclude_node_ids: set = None,
                      include_archived: bool = False,
                      content_preview_chars: int = 0) -> List[Dict[str, Any]]:
        """Get neighbors with node + edge + relation data (EDGE_ROW_SHAPE).

        Single-direction storage: queries both directions, flags each as
        outgoing/incoming. Relations from edge_relations via edge_id JOIN.

        Args:
            node_id: Node to find neighbors of. Empty → raises ValueError.
            limit: Max neighbors (ordered by edge weight desc).
            exclude_relations: Relation types to skip. None → no exclusion.
                Pass DEFAULT_EXCLUDED_RELATIONS for the standard noise set.
            exclude_node_ids: Node IDs to skip (already visited in traversal).
            include_archived: if False (default), filters er.archived=0.
                Enables forensic/recovery queries when True.
            content_preview_chars: if > 0, adds `content_preview` field to
                each row — substr(content, 1, N). Default 0 skips content
                to keep result size small.
        """
        if not node_id:
            raise ValueError("get_neighbors: node_id required")

        where_parts = ["n.archived = 0"]
        params = [node_id, node_id, node_id, node_id]

        if not include_archived:
            where_parts.append("er.archived = 0")

        if exclude_node_ids:
            placeholders = ",".join("?" * len(exclude_node_ids))
            where_parts.append("n.id NOT IN (%s)" % placeholders)
            params.extend(exclude_node_ids)

        if exclude_relations:
            placeholders = ",".join("?" * len(exclude_relations))
            where_parts.append("er.relation NOT IN (%s)" % placeholders)
            params.extend(exclude_relations)

        params.append(limit)
        where_clause = " AND ".join(where_parts)

        preview_col = ''
        if content_preview_chars and content_preview_chars > 0:
            preview_col = ', substr(n.content, 1, %d) as content_preview' % int(content_preview_chars)

        rows = self.conn.execute("""
            SELECT
                n.id, n.type, n.title, n.content_summary, n.confidence,
                n.revised_at, n.created_at, n.last_accessed, n.access_count,
                n.locked, n.emotion, n.emotion_label,
                er.relation, er.weight, er.description,
                e.last_strengthened, e.co_access_count, e.edge_id,
                CASE WHEN e.source_id = ? THEN 'outgoing' ELSE 'incoming' END as direction
                {preview_col}
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?)
            AND {where_clause}
            ORDER BY e.weight DESC
            LIMIT ?
        """.format(preview_col=preview_col, where_clause=where_clause),
            params).fetchall()

        out = []
        for r in rows:
            row = {
                'id': r[0], 'type': r[1], 'title': r[2], 'content_summary': r[3],
                'confidence': r[4], 'revised_at': r[5], 'created_at': r[6],
                'last_accessed': r[7], 'access_count': r[8], 'locked': r[9],
                'emotion': r[10], 'emotion_label': r[11],
                'relation': r[12] or '', 'weight': r[13],
                'edge_description': r[14], 'last_strengthened': r[15],
                'co_access_count': r[16], 'edge_id': r[17],
                'direction': r[18],
            }
            if preview_col:
                row['content_preview'] = r[19] or ''
            out.append(row)
        return out

    # ──────────────────────────────────────────────────────────────
    # v25 consolidated edge-read API — see GRAPH QUERY CONTRACT above.
    # Every method defaults include_archived=False.
    # ──────────────────────────────────────────────────────────────

    def nodes_touched_by_relations(self, relations, include_archived: bool = False):
        """Set of node IDs that participate (as source or target) in any edge
        with one of the given relations.

        Used by the 'Unreviewed Node' suppression pattern — any unit can ask
        "which nodes have already been seen by this kind of edge?" without
        writing raw SQL. Defaults to active (non-archived) edge_relations.
        """
        rels = list(relations)
        if not rels:
            return set()
        ph = ','.join('?' * len(rels))
        archived_clause = '' if include_archived else ' AND er.archived = 0'
        rows = self.conn.execute("""
            SELECT DISTINCT node_id FROM (
                SELECT e.source_id AS node_id FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE er.relation IN (%s)%s
                UNION
                SELECT e.target_id AS node_id FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE er.relation IN (%s)%s
            )
        """ % (ph, archived_clause, ph, archived_clause),
            rels + rels).fetchall()
        return {r[0] for r in rows}

    def get_neighbors_bulk(self, node_ids,
                           exclude_relations=None,
                           include_archived: bool = False):
        """Bulk flat-row version of get_neighbors — one query for many owners.

        Returns dict {owner_id: [row_dict, ...]} where each row_dict has
        EDGE_ROW_SHAPE (one entry per edge_relation, not grouped). Use this
        when the caller iterates relations individually; use
        get_connections_bulk when the caller wants relations grouped per
        (owner, neighbor).

        Defaults to DEFAULT_EXCLUDED_RELATIONS when exclude_relations is None
        (drops co_accessed, emergent_bridge). Pass an empty set to include all.
        """
        ids = list(node_ids)
        if not ids:
            return {}

        if exclude_relations is None:
            exclude_relations = DEFAULT_EXCLUDED_RELATIONS

        owner_ph = ",".join("?" * len(ids))
        where_parts = ["n.archived = 0"]
        params = list(ids) + list(ids) + list(ids)

        if not include_archived:
            where_parts.append("er.archived = 0")

        if exclude_relations:
            rel_ph = ",".join("?" * len(exclude_relations))
            where_parts.append("er.relation NOT IN (%s)" % rel_ph)
            params.extend(exclude_relations)

        where_clause = " AND ".join(where_parts)

        rows = self.conn.execute("""
            SELECT
                CASE WHEN e.source_id IN ({owner_ph}) THEN e.source_id ELSE e.target_id END AS owner_id,
                n.id, n.type, n.title, n.content_summary, n.confidence,
                n.revised_at, n.created_at, n.last_accessed, n.access_count,
                n.locked, n.emotion, n.emotion_label,
                er.relation, er.weight, er.description,
                e.last_strengthened, e.co_access_count, e.edge_id,
                CASE WHEN e.source_id IN ({owner_ph}) THEN 'outgoing' ELSE 'incoming' END as direction
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id IN ({owner_ph}) THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id IN ({owner_ph}) OR e.target_id IN ({owner_ph}))
            AND {where_clause}
            ORDER BY e.weight DESC
        """.format(owner_ph=owner_ph, where_clause=where_clause),
            params + list(ids) + list(ids)).fetchall()

        result = {owner: [] for owner in ids}
        for r in rows:
            owner = r[0]
            if owner not in result:
                # Edge row where owner matched join but not in our list — skip
                continue
            result[owner].append({
                'id': r[1], 'type': r[2], 'title': r[3], 'content_summary': r[4],
                'confidence': r[5], 'revised_at': r[6], 'created_at': r[7],
                'last_accessed': r[8], 'access_count': r[9], 'locked': r[10],
                'emotion': r[11], 'emotion_label': r[12],
                'relation': r[13] or '', 'weight': r[14],
                'edge_description': r[15], 'last_strengthened': r[16],
                'co_access_count': r[17], 'edge_id': r[18],
                'direction': r[19],
            })
        return result

    def get_connections_bulk(self, node_ids,
                             exclude_relations=None,
                             include_relations=None,
                             include_archived: bool = False,
                             include_neighbor_archived: bool = False):
        """Grouped neighbor fetch — multiple relations per (owner, neighbor)
        collapsed into a single entry with a `relations` list.

        The rich-node builder in brain_recall needs this shape: one entry
        per unique (owner, neighbor) pair, carrying aggregate edge weight
        and all relations on that pair.

        Args:
            node_ids: owner node ids to walk from
            exclude_relations: relations to skip (defaults to noise-relation list).
                Ignored if include_relations is set.
            include_relations: when set, ONLY relations in this iterable are
                returned. Use for aspect-scoped walks (e.g. correction-aspect
                relations only). Mutually exclusive with exclude_relations —
                if include_relations is provided, exclude_relations is ignored.
            include_archived: include archived edge_relations rows
            include_neighbor_archived: include edges whose neighbor node is archived

        Returns dict {owner_id: [connection_dict, ...]} where each
        connection_dict has:
            id, type, title, created_at, revised_at, confidence, locked,
            weight, direction, relations: [{relation, description, weight}, ...]

        Raises ValueError on empty node_ids.
        """
        ids = list(node_ids)
        if not ids:
            raise ValueError("get_connections_bulk: node_ids is empty")

        id_ph = ','.join('?' * len(ids))

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        neighbor_archived_clause = (
            'AND n1.archived = 0 AND n2.archived = 0'
            if not include_neighbor_archived else ''
        )
        rel_clause = ''
        rel_params = []
        if include_relations is not None:
            inc_list = list(include_relations)
            if not inc_list:
                # Empty whitelist → no edges match. Return empty grouping.
                return {nid: [] for nid in ids}
            rel_ph = ','.join('?' * len(inc_list))
            rel_clause = 'AND er.relation IN (%s)' % rel_ph
            rel_params = inc_list
        else:
            if exclude_relations is None:
                exclude_relations = DEFAULT_EXCLUDED_RELATIONS
            if exclude_relations:
                rel_ph = ','.join('?' * len(exclude_relations))
                rel_clause = 'AND er.relation NOT IN (%s)' % rel_ph
                rel_params = list(exclude_relations)

        sql = """
            SELECT e.source_id, e.target_id, e.weight,
                   er.relation, er.description, er.weight as rel_weight,
                   n1.id, n1.type, n1.title, n1.created_at, n1.revised_at,
                   n1.confidence, n1.locked,
                   n2.id, n2.type, n2.title, n2.created_at, n2.revised_at,
                   n2.confidence, n2.locked
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n1 ON n1.id = e.target_id
            JOIN nodes n2 ON n2.id = e.source_id
            WHERE (e.source_id IN ({id_ph}) OR e.target_id IN ({id_ph}))
              {archived_clause}
              {neighbor_archived_clause}
              {rel_clause}
        """.format(
            id_ph=id_ph,
            archived_clause=archived_clause,
            neighbor_archived_clause=neighbor_archived_clause,
            rel_clause=rel_clause,
        )

        rows = self.conn.execute(sql, ids + ids + rel_params).fetchall()

        owner_set = set(ids)
        grouped = {nid: {} for nid in ids}

        for row in rows:
            src, tgt = row[0], row[1]
            agg_weight = row[2]
            rel = row[3] or 'related'
            desc = row[4] or ''
            rel_weight = row[5] if row[5] is not None else agg_weight
            n1 = {'id': row[6], 'type': row[7], 'title': row[8],
                  'created_at': row[9], 'revised_at': row[10],
                  'confidence': row[11], 'locked': row[12] == 1}
            n2 = {'id': row[13], 'type': row[14], 'title': row[15],
                  'created_at': row[16], 'revised_at': row[17],
                  'confidence': row[18], 'locked': row[19] == 1}
            relation_entry = {'relation': rel, 'description': desc,
                              'weight': rel_weight}

            if src in owner_set and tgt != src:
                entry = grouped[src].setdefault(n1['id'], {
                    **n1, 'weight': agg_weight, 'direction': 'outgoing',
                    'relations': [],
                })
                entry['relations'].append(relation_entry)

            if tgt in owner_set and src != tgt:
                entry = grouped[tgt].setdefault(n2['id'], {
                    **n2, 'weight': agg_weight, 'direction': 'incoming',
                    'relations': [],
                })
                entry['relations'].append(relation_entry)

        return {owner: list(nbrs.values()) for owner, nbrs in grouped.items()}

    def bulk_archive_relations(self, where_sql, params, archived_by, *,
                               null_embeddings, recompute_weight,
                               exempt_relations=()):
        """THE one soft-archive flip for edge_relations rows.

        Every archiving writer routes here (remove_relation,
        delete_node_edges, archive_dangling_edges, decay_edges' prune arm) —
        one UPDATE shape, one observed-truth return, so the flip semantics
        can't drift between callers (ruled 2026-08-03, node 482ef98e).

        SELECT with the UPDATE's exact predicate first, then UPDATE — same
        connection, same transaction, under brain.write_lock — so the
        returned [edge_id, relation] pairs ARE what the UPDATE flipped,
        never an approximation. Already-archived rows and exempt relations
        never appear in the return.

        Policy is EXPLICIT per caller, never defaulted:
          null_embeddings: drop the stored embedding with the flip (archived
              edges are never read; revive re-embeds async). The dangling
              sweep keeps False — its historical behavior, preserved exactly.
          recompute_weight: refresh the edges aggregate row per flipped
              edge. Single-edge and prune callers True/at-caller; bulk node
              sweeps False (aggregate rows of an archived node are unread).
          exempt_relations: relations that must survive the sweep
              (survivor_lineage: absorbed_into).

        archived_at is ISO-T via clock.iso_now() — the one write-side format.
        Does NOT commit: callers own commit_unless_batched (decay calls this
        per-relation inside one pass; a commit here would split its batch).
        """
        exempt_clause, exempt = _relation_not_in_clause(exempt_relations)
        base = 'archived = 0 AND (%s) %s' % (where_sql, exempt_clause)
        bind = list(params) + exempt
        flipped = [[r[0], r[1]] for r in self.conn.execute(
            'SELECT edge_id, relation FROM edge_relations WHERE ' + base,
            bind).fetchall()]
        if not flipped:
            return flipped
        embed_cols = (', embedding = NULL, embedding_model = NULL'
                      if null_embeddings else '')
        self.conn.execute(
            'UPDATE edge_relations '
            'SET archived = 1, archived_at = ?, archived_by = ?%s '
            'WHERE %s' % (embed_cols, base),
            [iso_now(), archived_by] + bind)
        if recompute_weight:
            for eid in {f[0] for f in flipped}:
                self._update_aggregate_weight(eid)
        return flipped

    # The dangling selection: every ACTIVE relation on an edge whose source
    # or target node is archived. Edge-level on purpose — an archived
    # endpoint makes every relation on that edge dangling.
    _DANGLING_WHERE = """edge_id IN (
                 SELECT er.edge_id FROM edge_relations er
                 JOIN edges e ON e.edge_id = er.edge_id
                 JOIN nodes n_src ON n_src.id = e.source_id
                 JOIN nodes n_tgt ON n_tgt.id = e.target_id
                 WHERE er.archived = 0
                   AND (n_src.archived = 1 OR n_tgt.archived = 1)
               )"""

    def archive_dangling_edges(self, archived_by: str,
                               exempt_relations=()) -> Dict[str, Any]:
        """Archive active edge_relations rows whose source or target node is archived.

        Invariant restorer: the brain's rule is `Archive edges alongside nodes —
        no dangling edges after committing`. Historical leak paths
        (pre-April-2026 archive_node deletion bug, mid-migration races) can
        leave active edges pointing at archived nodes; this method scrubs them.

        Commits its own flip (commit_unless_batched) — it had NO commit until
        2026-08-06, so its writes rode open until some later writer committed,
        and any trace emitted after it was orphanable by construction (the
        emitter's in_transaction gate kept firing on Healer cycles).

        Args:
            archived_by: encoding_source-style tag for the archive action
                (e.g. 's2:healer', 'migration:cleanup_2026_05_16').
            exempt_relations: relations that are SUPPOSED to span an archived
                endpoint and must NOT be scrubbed — the survivor-redirect link
                `absorbed_into` (resolve_live walks it; in a chain A→B→C its
                target is itself archived). The caller sources these from
                `brain.aspects.relations_in(['survivor_lineage'])` so the
                taxonomy, not this method, owns the list. DAL stays
                aspect-agnostic — it just takes the strings.

        Returns {'archived': int, 'edge_relations': [[edge_id, relation], ...]}
        — the pairs the UPDATE actually flipped, for the caller's trace rows.
        (Was a bare rowcount int until 2026-08-06.)
        """
        flipped = self.bulk_archive_relations(
            self._DANGLING_WHERE, [], archived_by,
            null_embeddings=False, recompute_weight=False,
            exempt_relations=exempt_relations)
        commit_unless_batched(self.conn)
        return {'archived': len(flipped), 'edge_relations': flipped}

    def reconcile_community_membership(self,
                                       encoding_source='s2:community_repair'):
        """Restorer: back-fill `community_member` edges for ORPHANED communities.

        A community declares its members in two places that can silently
        diverge: the `community_members` metadata string AND the actual
        `community_member` edges. The community encoder (Haiku) sometimes
        creates the node + metadata but omits the edge field entirely — or
        used the retired `connections=` param (dropped by remember()'s guard).
        The node then claims N members with ZERO edges: a structural
        inconsistency nothing else catches, because the declared list is the
        only diffable intent (community is the only encoder that records its
        expected structure as data). See also `archive_dangling_edges` — same
        per-cycle integrity-restorer pattern.

        Scope is deliberately the ZERO-edge case only. A community with SOME
        member edges is left alone: a partial gap is far more likely intentional
        drift (a member was disconnected) than omission, and re-adding from the
        possibly-stale metadata would resurrect a removed member. Omission is
        all-or-nothing at the encoder (one `connect_to` field for all members),
        so it always presents as zero edges — exactly what this targets.

        Idempotent: once edges exist the community is skipped. Archived/missing
        declared members are skipped (legitimate drift, not omission).

        Caller must hold brain.write_lock (writes via add_relation).
        Returns {communities_healed, edges_backfilled, details: [(cid, n), ...]}.
        """
        import re
        # 1. Declared members per live community (id -> {member_id: label}).
        #    Anchor the id match to a segment start (^ or comma) so an 8-hex
        #    token inside a member's TITLE can't be mistaken for a member id.
        declared = {}
        for cid, val in self.conn.execute(
                "SELECT kv.node_id, kv.value FROM node_metadata_kv kv "
                "JOIN nodes n ON n.id = kv.node_id "
                "WHERE kv.key = 'community_members' "
                "AND n.type = 'community' AND n.archived = 0").fetchall():
            members = {}
            for mt in re.finditer(r'(?:^|,)\s*([0-9a-f]{8})\s*:\s*([^,]*)',
                                  val or ''):
                members[mt.group(1)] = mt.group(2).strip()
            if members:
                declared[cid] = members
        if not declared:
            return {'communities_healed': 0, 'edges_backfilled': 0,
                    'details': [], 'edge_ids': []}

        # 2. Communities that ALREADY have >=1 active community_member edge,
        #    in EITHER direction — skip them (partial gap == drift, not
        #    omission). get_community_members reads membership both ways
        #    (historical mix of community->member and legacy member->community
        #    edges), so this orphan check must too — else a legacy-direction
        #    community is falsely flagged as orphan every cycle.
        edged = set()
        for src, tgt in self.conn.execute(
                "SELECT e.source_id, e.target_id FROM edges e "
                "JOIN edge_relations er ON er.edge_id = e.edge_id "
                "WHERE er.relation = 'community_member' "
                "AND er.archived = 0").fetchall():
            edged.add(src)
            edged.add(tgt)

        orphans = {cid: ms for cid, ms in declared.items() if cid not in edged}
        if not orphans:
            return {'communities_healed': 0, 'edges_backfilled': 0,
                    'details': [], 'edge_ids': []}

        # 3. Which declared members are LIVE (skip archived/missing targets —
        #    a declared member that was archived is drift, not an omission).
        all_members = sorted({m for ms in orphans.values() for m in ms})
        live = set()
        for i in range(0, len(all_members), 400):
            chunk = all_members[i:i + 400]
            rows = self.conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 AND id IN (%s)"
                % ','.join('?' * len(chunk)), chunk).fetchall()
            live.update(r[0] for r in rows)

        # 4. Back-fill the gap on orphaned communities only.
        healed = edges = 0
        details = []
        edge_ids = []
        for cid, members in orphans.items():
            # Skip self: community_members occasionally echoes the community's
            # own id, and add_relation has no self-edge guard (the LLM path
            # does, via _apply_connect_to exclude_self).
            live_missing = [(m, lbl) for m, lbl in members.items()
                            if m in live and m != cid]
            if not live_missing:
                continue
            for mid, label in live_missing:
                desc = ((label or 'community member')
                        + ' — member edge restored by membership '
                          'reconciliation')[:200]
                res = self.add_relation(cid, mid, 'community_member',
                                        description=desc, weight=0.6,
                                        encoding_source=encoding_source)
                edges += 1
                if res.get('edge_id'):
                    edge_ids.append(res['edge_id'])
            healed += 1
            details.append((cid, len(live_missing)))
        # edge_ids ride to the community unit's delta — this is direct-DAL
        # (the mutation emitter can't see it), so the S2 story records the
        # backfill itself (ruled 2026-08-04, plan step 10).
        return {'communities_healed': healed, 'edges_backfilled': edges,
                'details': details, 'edge_ids': edge_ids}

    def has_edge_between(self, source_ids, target_ids,
                         relations=None,
                         include_archived: bool = False) -> bool:
        """Existence check — is there any edge with these relations between
        any node in source_ids and any node in target_ids?

        Used by correction/tension detection, bridge-count guards.
        Raises ValueError if either set is empty.
        """
        src_list = list(source_ids)
        tgt_list = list(target_ids)
        if not src_list or not tgt_list:
            raise ValueError(
                "has_edge_between: source_ids and target_ids must both be non-empty")

        src_ph = ','.join('?' * len(src_list))
        tgt_ph = ','.join('?' * len(tgt_list))
        params = src_list + tgt_list

        rel_clause = ''
        if relations:
            rel_list = list(relations)
            rel_ph = ','.join('?' * len(rel_list))
            rel_clause = 'AND er.relation IN (%s)' % rel_ph
            params += rel_list

        archived_clause = '' if include_archived else 'AND er.archived = 0'

        sql = """
            SELECT 1 FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE e.source_id IN (%s) AND e.target_id IN (%s)
              %s
              %s
            LIMIT 1
        """ % (src_ph, tgt_ph, rel_clause, archived_clause)

        return self.conn.execute(sql, params).fetchone() is not None

    def get_community_members(self, community_id: str,
                              include_archived: bool = False,
                              require_active_member: bool = True):
        """Members of a community via community_member edges.

        Walks both directions of the edge to handle the historical mix
        where some edges point node→community and others community→node.
        Returns neighbor node dicts (subset of EDGE_ROW_SHAPE:
        id, type, title, created_at, confidence, locked).

        Raises ValueError if community_id is empty.
        """
        if not community_id:
            raise ValueError("get_community_members: community_id required")

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        member_archived_clause = 'AND member.archived = 0' if require_active_member else ''

        sql = """
            SELECT DISTINCT member.id, member.type, member.title,
                            member.created_at, member.confidence, member.locked
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes member ON member.id = CASE
                WHEN e.source_id = ? THEN e.target_id
                ELSE e.source_id END
            WHERE er.relation = 'community_member'
              AND (e.source_id = ? OR e.target_id = ?)
              AND member.type != 'community'
              %s
              %s
        """ % (archived_clause, member_archived_clause)

        rows = self.conn.execute(sql, (community_id, community_id, community_id)).fetchall()
        return [{
            'id': r[0], 'type': r[1], 'title': r[2],
            'created_at': r[3], 'confidence': r[4], 'locked': r[5],
        } for r in rows]

    def get_members_bulk(self, community_ids, include_archived: bool = False):
        """Members of MANY communities via community_member edges, batched.

        Bulk sibling of get_community_members: same bidirectional walk (the
        historical community->member and legacy member->community mix) and the
        same EDGE_ROW_SHAPE subset, but for a list of communities in one pass.
        Returns {community_id: [member dict, ...]}; communities with no member
        edges are simply absent (caller treats as empty). DISTINCT collapses a
        member reachable via edges in both directions.
        """
        ids = [c for c in (community_ids or []) if c]
        if not ids:
            return {}
        archived_clause = '' if include_archived else 'AND er.archived = 0'
        out = {}
        # Chunk to stay within SQLite's bind-variable limit.
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            placeholders = ','.join('?' * len(chunk))
            sql = """
                SELECT DISTINCT c.id AS community_id,
                                member.id, member.type, member.title,
                                member.created_at, member.confidence, member.locked
                FROM nodes c
                JOIN edges e ON (e.source_id = c.id OR e.target_id = c.id)
                JOIN edge_relations er ON er.edge_id = e.edge_id
                    AND er.relation = 'community_member' %s
                JOIN nodes member ON member.id = CASE
                    WHEN e.source_id = c.id THEN e.target_id
                    ELSE e.source_id END
                    AND member.archived = 0 AND member.type != 'community'
                WHERE c.id IN (%s) AND c.type = 'community' AND c.archived = 0
            """ % (archived_clause, placeholders)
            for r in self.conn.execute(sql, chunk).fetchall():
                out.setdefault(r[0], []).append({
                    'id': r[1], 'type': r[2], 'title': r[3],
                    'created_at': r[4], 'confidence': r[5], 'locked': r[6],
                })
        return out

    def get_communities_for(self, node_ids,
                            include_archived: bool = False,
                            require_active_community: bool = True):
        """Reverse of get_community_members: for each given node, list the
        communities it belongs to via community_member edges.

        Returns dict {node_id: [{id, title}, ...]} — symmetric to
        get_community_members's shape. Used by consolidation_decoder to
        enrich clusters with their community placement.

        Raises ValueError on empty node_ids.
        """
        ids = list(node_ids)
        if not ids:
            raise ValueError("get_communities_for: node_ids is empty")

        id_ph = ','.join('?' * len(ids))
        archived_clause = '' if include_archived else 'AND er.archived = 0'
        community_clause = 'AND n.archived = 0' if require_active_community else ''

        sql = """
            SELECT
                CASE WHEN e.source_id IN ({id_ph}) THEN e.source_id
                     ELSE e.target_id END as member,
                CASE WHEN e.source_id IN ({id_ph}) THEN e.target_id
                     ELSE e.source_id END as community,
                n.title
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE
                WHEN e.source_id IN ({id_ph}) THEN e.target_id
                ELSE e.source_id END
            WHERE (e.source_id IN ({id_ph}) OR e.target_id IN ({id_ph}))
              AND er.relation = 'community_member'
              AND n.type = 'community'
              {archived_clause}
              {community_clause}
        """.format(
            id_ph=id_ph,
            archived_clause=archived_clause,
            community_clause=community_clause,
        )

        rows = self.conn.execute(sql, ids * 5).fetchall()

        from collections import defaultdict
        membership = defaultdict(list)
        for member_id, comm_id, comm_title in rows:
            membership[member_id].append({'id': comm_id, 'title': comm_title})
        return dict(membership)

    def count_by_relation(self, include_archived: bool = False):
        """Edge count grouped by relation type.

        Returns dict {relation: count}, ordered by count desc. Used by
        integrity_audit, edge_families, health_check.
        """
        where = '' if include_archived else 'WHERE archived = 0'
        rows = self.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM edge_relations %s "
            "GROUP BY relation ORDER BY cnt DESC" % where
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_edge_descriptions_for(self, node_id: str,
                                  min_length: int = EDGE_CONTEXT_MIN_DESC_LENGTH,
                                  exclude_relations=None,
                                  include_archived: bool = False,
                                  limit: int = 5):
        """Return meaningful edge descriptions for a node's edges.

        Feeds edge_context embedding in _compute_group_vectors. Filters
        out short/empty descriptions (below min_length) and noise relations.
        Default exclusion: DEFAULT_EXCLUDED_RELATIONS + 'community_member'
        (structural, not semantic text).

        Returns list[str] of descriptions. Raises ValueError if node_id empty.
        """
        if not node_id:
            raise ValueError("get_edge_descriptions_for: node_id required")
        if exclude_relations is None:
            exclude_relations = DEFAULT_EXCLUDED_RELATIONS | {'community_member'}

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        rel_clause = ''
        if exclude_relations:
            rel_ph = ','.join('?' * len(exclude_relations))
            rel_clause = 'AND er.relation NOT IN (%s)' % rel_ph

        sql = """
            SELECT er.description FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE (e.source_id = ? OR e.target_id = ?)
              %s
              %s
              AND er.description IS NOT NULL
              AND length(er.description) > ?
            ORDER BY e.weight DESC
            LIMIT ?
        """ % (archived_clause, rel_clause)

        params = [node_id, node_id]
        if exclude_relations:
            params += list(exclude_relations)
        params += [min_length, limit]

        rows = self.conn.execute(sql, params).fetchall()
        return [r[0] for r in rows if r[0]]

    def count_node_edges(self, node_id: str, min_weight: float = 0.1,
                         relations=None,
                         include_archived: bool = False) -> int:
        """Count edges touching a node (both directions).

        Args:
            node_id: node whose edges to count. Empty → raises ValueError.
            min_weight: edges-table weight floor. Default 0.1.
            relations: iterable of relation names — if provided, only count
                edges that carry at least one of these relations (via JOIN).
                None → count the aggregate edges table, relation-agnostic.
            include_archived: if False (default) and `relations` is set,
                filters er.archived=0. Ignored when relations is None
                (aggregate edges has no archived column).

        Why two paths: when you don't care about relation, counting from
        `edges` directly is cheap. When you do, the JOIN is needed.
        """
        if not node_id:
            raise ValueError("count_node_edges: node_id required")

        if relations:
            rel_list = list(relations)
            rel_ph = ','.join('?' * len(rel_list))
            archived_clause = '' if include_archived else 'AND er.archived = 0'
            row = self.conn.execute(
                'SELECT COUNT(DISTINCT er.edge_id) FROM edges e '
                'JOIN edge_relations er ON er.edge_id = e.edge_id '
                'WHERE (e.source_id = ? OR e.target_id = ?) '
                'AND e.weight >= ? '
                'AND er.relation IN (%s) '
                '%s' % (rel_ph, archived_clause),
                [node_id, node_id, min_weight] + rel_list
            ).fetchone()
            return row[0] if row else 0

        row = self.conn.execute(
            'SELECT COUNT(*) FROM edges WHERE (source_id = ? OR target_id = ?) AND weight >= ?',
            (node_id, node_id, min_weight)
        ).fetchone()
        return row[0] if row else 0

    # get_edge_count removed 2026-05-30 (DAL cleanup Phase 0) — exact dup of
    # count_total; both were dead (brain._get_edge_count uses raw SQL, to be
    # routed through count_total in Phase 3).
    # get_well_connected + get_random_walk_neighbors removed 2026-05-30 — the
    # consolidation/promotion + random-walk paths that used them are retired
    # (brain_connections._random_walk, their only kin, was also dead).

    # --- Writes ---

    # create_edge removed 2026-05-30 (DAL cleanup Phase 0) — DEPRECATED since the
    # Hebbian path moved to recall_write_queue via add_relation; 0 callers (its
    # docstring's claimed brain_recall caller was already gone). Use
    # brain.connect_typed() (write-path embed hook) for all edge creation.

    # strengthen_edge REMOVED 2026-05-18 (Phase 8 of bg_writer migration).
    # Was a deprecated read-modify-write helper used only by the old
    # brain_recall._hebbian_strengthen mixin, which Phase 5 deleted.
    # Hebbian strengthening now uses atomic UPSERT inside
    # recall_write_queue._apply_hebbian_pairs via add_relation.

    def delete_node_edges(self, node_id: str,
                          archived_by: str = 'delete_node_edges',
                          exempt_relations=()) -> list:
        """Soft-archive all edge_relations touching a node (v25).

        Single source for "archive a node's edges" — archive_node routes here
        (passing its real `archived_by`) instead of duplicating the SQL.

        Commit is gated on self.conn.in_batch (commit_unless_batched) — a no-op
        inside a brain_batch envelope, a real commit standalone.

        Was a hard DELETE prior to v25 — the asymmetry with node archive
        destroyed edge provenance forever. Now sets archived=1 on the
        relations and leaves the edges aggregate row intact.

        Returns the [edge_id, relation] pairs the UPDATE actually flipped —
        observed truth for the caller's trace metadata. NOT the full edge
        list: already-archived rows and `exempt_relations` misses are
        excluded, so the return can never claim the deliberately-exempted
        absorbed_into redirect was archived. Count = len(return).
        (The pre-UPDATE SELECT shares the UPDATE's predicate and, on the
        archive_node path, its write transaction. A hypothetical standalone
        call with no prior DML would SELECT in autocommit — a concurrent
        writer could then slip a row between the two statements.)

        Args:
            archived_by: encoding_source-style tag (e.g. 's2:consolidation').
            exempt_relations: relations that must outlive the node — the
                survivor-redirect link `absorbed_into`, or the resolve_live
                chain breaks. Caller sources these from
                `brain.aspects.relations_in(['survivor_lineage'])`; the DAL
                stays aspect-agnostic. hard_delete_node_edges removes
                everything regardless (a deleted endpoint leaves no chain).
        """
        edge_ids = [r[0] for r in self.conn.execute(
            'SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)
        ).fetchall()]

        # NULL the stored embedding on archive — same pattern node archive
        # uses (DELETE FROM node_enrichments). Archived edges are never read
        # by spread/select_edges (every read filters archived=0), so the blob
        # is dead weight; a later revive via add_relation Branch 3 re-embeds
        # async. Symmetric with nodes.
        flipped = []
        for i in range(0, len(edge_ids), 500):
            chunk = edge_ids[i:i + 500]
            flipped.extend(self.bulk_archive_relations(
                'edge_id IN (%s)' % ','.join('?' * len(chunk)), chunk,
                archived_by, null_embeddings=True, recompute_weight=False,
                exempt_relations=exempt_relations))

        commit_unless_batched(self.conn)
        return flipped

    def hard_delete_node_edges(self, node_id: str) -> int:
        """HARD-delete every edge touching a node — both the `edge_relations`
        rows and the `edges` aggregate rows. For the node delete-cascade: a hard
        node delete must leave no edge or relation rows. Contrast
        delete_node_edges, which SOFT-archives (archived=1) to preserve history
        for a still-live node. Returns the count of `edges` rows removed."""
        if not node_id:
            return 0
        edge_ids = [r[0] for r in self.conn.execute(
            'SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)).fetchall()]
        if edge_ids:
            ph = ','.join('?' * len(edge_ids))
            self.conn.execute(
                'DELETE FROM edge_relations WHERE edge_id IN (%s)' % ph, edge_ids)
        n = self.conn.execute(
            'DELETE FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)).rowcount
        commit_unless_batched(self.conn)
        return n

    def decay_edges(self) -> Dict[str, Any]:
        """Apply exponential decay to auto-generated edge relations.

        Commit is gated on self.conn.in_batch (commit_unless_batched).

        Operates on edge_relations via edge_id.
        When a relation's weight drops below threshold, that relation is removed.
        If an edge has no relations left, the physical edge is also removed.

        Formula: new_weight = weight * 0.5^(hours_since_created / half_life)

        Returns: {decayed: int, pruned: int,
                  by_type: {relation: {decayed, pruned}},
                  pruned_edges: [[edge_id, relation], ...]}  # observed flips,
                  # for the caller's per-edge trace rows
        """
        from .brain_constants import EDGE_TYPES, EDGE_PRUNE_THRESHOLD

        total_decayed = 0
        total_pruned = 0
        by_type = {}
        pruned_edges = []

        for relation, config in EDGE_TYPES.items():
            if not config.get('decays'):
                continue

            half_life = config['halfLife']

            # Apply decay on active edge_relations only (v25)
            self.conn.execute("""
                UPDATE edge_relations
                SET weight = weight * power(0.5,
                    (julianday('now') - julianday(created_at)) * 24.0 / ?)
                WHERE relation = ?
                  AND archived = 0
                  AND created_at IS NOT NULL
                  AND (julianday('now') - julianday(created_at)) * 24.0 > 0
            """, (half_life, relation))
            decayed = self.conn.execute('SELECT changes()').fetchone()[0]

            # Soft-archive relations below threshold (v25 — was DELETE) via
            # the shared flip primitive: embedding NULLed with the flip,
            # aggregate weight recomputed per flipped edge, and the flipped
            # [edge_id, relation] pairs kept — the caller emits one trace
            # row per pruned relation (per-edge rows ruled 2026-08-03, no
            # rollup shape). The weight-decay UPDATE above stays outside the
            # primitive by design — it flips nothing.
            flipped = self.bulk_archive_relations(
                'relation = ? AND weight < ?',
                [relation, EDGE_PRUNE_THRESHOLD], 'decay_pruned',
                null_embeddings=True, recompute_weight=True)
            pruned = len(flipped)
            pruned_edges.extend(flipped)

            if decayed or pruned:
                by_type[relation] = {'decayed': decayed, 'pruned': pruned}
                total_decayed += decayed
                total_pruned += pruned

        commit_unless_batched(self.conn)
        return {'decayed': total_decayed, 'pruned': total_pruned,
                'by_type': by_type, 'pruned_edges': pruned_edges}

    # --- edge_relations (multi-relation semantic layer via edge_id) ---

    @staticmethod
    def _generate_edge_id(source_id, target_id):
        """Deterministic edge ID from source+target pair."""
        import hashlib
        h = hashlib.md5((source_id + ':' + target_id).encode()).hexdigest()[:8]
        return 'edg_' + h

    # Sentinel for distinguishing "field not specified" (preserve existing)
    # from "explicit value passed" (replace). Plain default values can't
    # express this distinction.
    _UNSET = object()

    def add_relation(self, source_id, target_id, relation,
                     description=_UNSET, weight=_UNSET, encoding_source=_UNSET):
        """Upsert a relation on an edge pair. Creates the physical edge if needed.

        Stage 1B contract — field-preserving upsert + lifecycle audit via traces.

        Commit is gated on self.conn.in_batch (commit_unless_batched): a no-op
        when a wider transaction owns the envelope (brain_batch, or the
        bg_writer queue drain that opens BEGIN IMMEDIATE around a batch of
        pairs and commits once), a real commit standalone. The owner flips
        conn.in_batch — letting add_relation self-commit inside would break
        atomicity (earlier writes persist while a later failure rolls back
        only the most-recent statements).

        Three branches by row state for (edge_id, relation):
          - No row              → INSERT with passed values + sensible defaults
          - Active row exists   → field-preserving UPDATE (only specified fields update)
          - Archived row exists → REVIVE: archived=0, fresh created_at, all fields
                                  reset to passed values + defaults (semantic
                                  fresh row; PK forces row reuse, trace events
                                  capture the lifecycle). Deltas carry an extra
                                  `archived: 1→0` row naming the revival; the
                                  field deltas keep old=None — an empty `old` is
                                  the documented "just created" signal, and a
                                  revive IS a semantic fresh row.

        Auto-strengthen behavior dropped (Stage 1B Option α). Hebbian co-access
        weight bumps are applied off the recall hot path by
        recall_write_queue._apply_hebbian_pairs — encoder-explicit connect
        calls are now idempotent.

        Field-preservation rule (active row branch): caller passes _UNSET
        (the default) for a field → existing value preserved. Caller passes
        an explicit value → that value replaces existing.

        Returns:
            {'edge_id': str,
             'created': bool,              # new INSERT or revived archive
             'revived_from_archive': bool, # subset of created
             'updated': bool,              # active row had specified fields update
             'deltas': [{'field', 'old', 'new'}, ...],  # for trace emission
             'warnings': []}               # reserved for future warnings

        Raises ValueError if source or target node doesn't exist.
        """
        ts = iso_now()

        # Resolve defaults for fields that get a value-or-default (used in
        # INSERT and revive branches; active-update preserves unspecified).
        desc_specified = (description is not GraphDAL._UNSET)
        weight_specified = (weight is not GraphDAL._UNSET)
        es_specified = (encoding_source is not GraphDAL._UNSET)
        desc_value = description if desc_specified else ''
        weight_value = weight if weight_specified else 0.5
        es_value = encoding_source if es_specified else ''

        # Validate both nodes exist
        for nid, label in [(source_id, 'source'), (target_id, 'target')]:
            exists = self.conn.execute(
                'SELECT 1 FROM nodes WHERE id = ?', (nid,)).fetchone()
            if not exists:
                raise ValueError("Cannot create edge: %s node '%s' does not exist" % (
                    label, nid[:12]))

        # Find or create the physical edge (check both directions)
        edge_id = self.get_edge_id(source_id, target_id)

        if not edge_id:
            # Create new physical edge row
            edge_id = self._generate_edge_id(source_id, target_id)
            self.conn.execute(
                'INSERT OR IGNORE INTO edges '
                '(edge_id, source_id, target_id, weight, co_access_count, last_strengthened, created_at) '
                'VALUES (?, ?, ?, ?, 0, ?, ?)',
                (edge_id, source_id, target_id, weight_value, ts, ts))

        # Look up this (edge_id, relation) pair. PK is (edge_id, relation),
        # so at most one row exists — may be active or archived.
        existing = self.conn.execute(
            'SELECT description, weight, encoding_source, archived '
            'FROM edge_relations WHERE edge_id = ? AND relation = ?',
            (edge_id, relation)
        ).fetchone()

        result = {
            'edge_id': edge_id,
            'created': False,
            'revived_from_archive': False,
            'updated': False,
            'deltas': [],
            'warnings': [],
        }

        # Birth-deltas for the INSERT and revive branches — one list, so the
        # two "semantic fresh row" branches cannot drift apart. old=None on
        # every field is the documented "just created" signal.
        def _birth_deltas():
            return [
                {'field': 'description', 'old': None, 'new': desc_value},
                {'field': 'weight', 'old': None, 'new': weight_value},
                {'field': 'encoding_source', 'old': None, 'new': es_value},
            ]

        if existing is None:
            # Branch 1: No row → INSERT
            self.conn.execute(
                'INSERT INTO edge_relations '
                '(edge_id, relation, description, weight, encoding_source, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (edge_id, relation, desc_value, weight_value, es_value, ts))
            result['created'] = True
            result['deltas'] = _birth_deltas()

        elif existing[3] == 0:
            # Branch 2: Active row → field-preserving UPDATE
            old_desc, old_weight, _old_es, _archived = existing
            updates = {}
            if desc_specified and description != old_desc:
                updates['description'] = description
                result['deltas'].append({
                    'field': 'description', 'old': old_desc, 'new': description})
            if weight_specified and weight != old_weight:
                updates['weight'] = weight
                result['deltas'].append({
                    'field': 'weight', 'old': old_weight, 'new': weight})
            # encoding_source is the CREATOR mark — set once at birth (the INSERT
            # and revive branches), never rewritten on an active-row update. A
            # connect re-touch (anchor re-asserting an edge, the encoder
            # re-linking one) updates description/weight but must NOT relabel who
            # created the edge: the column is a denormalized cache of the
            # creation event the trace log recorded, so a later overwrite would
            # make it drift from that event. Deliberate re-attribution is
            # rename_relation's job (it always rewrites encoding_source), not the
            # connect upsert's — so es_specified is intentionally ignored here.

            if updates:
                set_clause = ', '.join('%s = ?' % k for k in updates)
                self.conn.execute(
                    'UPDATE edge_relations SET %s '
                    'WHERE edge_id = ? AND relation = ?' % set_clause,
                    [*updates.values(), edge_id, relation])
                result['updated'] = True
            # If no specified fields differ → true no-op (no SQL write).

        else:
            # Branch 3: Archived row → revive with fresh state
            # Semantic 'fresh row': all fields reset to passed values + defaults.
            # Schema PK forces row reuse; trace events tell the lifecycle.
            old_desc, old_weight, old_es, _archived = existing
            self.conn.execute(
                'UPDATE edge_relations '
                'SET archived = 0, archived_at = NULL, archived_by = NULL, '
                '    description = ?, weight = ?, encoding_source = ?, '
                '    created_at = ? '
                'WHERE edge_id = ? AND relation = ?',
                (desc_value, weight_value, es_value, ts, edge_id, relation))
            result['created'] = True
            result['revived_from_archive'] = True
            # The archived flip is the one delta that distinguishes a revive
            # from a plain create in the trace row — nothing emitted it before.
            result['deltas'] = ([{'field': 'archived', 'old': 1, 'new': 0}]
                                + _birth_deltas())

        # Update aggregate weight + last_strengthened on physical edge
        self._update_aggregate_weight(edge_id)
        self.conn.execute(
            'UPDATE edges SET last_strengthened = ? WHERE edge_id = ?', (ts, edge_id))
        commit_unless_batched(self.conn)

        # Enqueue for temporal extraction AND async edge re-embedding when the
        # description (part of compose_edge_text) changed. enqueue_edge() is a
        # cheap set.add; the embed_queue worker runs backfill_entity_dates +
        # backfill_edge_embeddings. Lazy import avoids a module-load cycle.
        _desc_changed = any(d.get('field') == 'description' for d in result['deltas'])
        if result['created'] or _desc_changed:
            # Invalidate the stored embedding so the worker re-embeds. New rows
            # are already NULL; only an existing row whose description changed
            # needs explicit NULLing.
            if _desc_changed and not result['created']:
                self.conn.execute(
                    'UPDATE edge_relations SET embedding = NULL, embedding_model = NULL '
                    'WHERE edge_id = ? AND relation = ?', (edge_id, relation))
                commit_unless_batched(self.conn)
            self._enqueue_edge_embed(edge_id, 'add_relation')

        return result

    @staticmethod
    def _enqueue_edge_embed(edge_id, origin):
        """Enqueue an edge for async re-embedding, loudly on failure.

        The enqueue is a set.add — failure is exotic (lock contention, import
        collapse). No brain reference at DAL level, so a stderr line is the
        loudest available channel; was bare `except: pass` pre-migration.
        """
        try:
            from . import embed_queue
            embed_queue.enqueue_edge(edge_id)
        except Exception as _eq_err:
            try:
                import sys as _sys
                print('[GraphDAL.%s] enqueue_edge failed: %s'
                      % (origin, _eq_err), file=_sys.stderr)
            except Exception:
                pass

    def get_relations(self, edge_id, include_archived: bool = False):
        """Get active relations for an edge by edge_id.

        Returns list of dicts: [{relation, description, weight, encoding_source, created_at}, ...]
        include_archived=True surfaces archived rows too (forensics / recovery).
        """
        where = 'WHERE edge_id = ?'
        if not include_archived:
            where += ' AND archived = 0'
        rows = self.conn.execute(
            'SELECT relation, description, weight, encoding_source, created_at '
            'FROM edge_relations %s '
            'ORDER BY weight DESC' % where,
            (edge_id,)
        ).fetchall()
        return [{'relation': r[0], 'description': r[1] or '',
                 'weight': r[2], 'encoding_source': r[3] or '',
                 'created_at': r[4]}
                for r in rows]

    def remove_relation(self, source_id, target_id, relation, archived_by: str = 'unknown'):
        """Soft-archive a specific relation on a pair (v25).

        Commit is gated on self.conn.in_batch (commit_unless_batched) — a no-op
        inside the brain_batch `disconnect` envelope, a real commit standalone.

        Flips archived=1 on the matching row. Previously hard-DELETEd; the
        change preserves edge history for recovery. The edges aggregate row
        stays regardless — reads filter via edge_relations joins.

        Returns OBSERVED truth, from the flip primitive's same-predicate
        SELECT — never a fabricated
        flip: {'edge_id', 'relation', 'flipped', 'deltas'}. flipped=False (empty
        deltas) means the row was already archived or never existed; a caller
        emitting traces from `deltas` therefore cannot record an archive that
        didn't happen. edge_id is None when no physical edge exists at all.
        """
        edge_id = self.get_edge_id(source_id, target_id)
        result = {'edge_id': edge_id, 'relation': relation,
                  'flipped': False, 'deltas': []}
        if not edge_id:
            return result

        # Shared flip primitive; embedding NULLed with the flip (revive via
        # add_relation Branch 3 re-embeds). recompute_weight=False because
        # this caller recomputes UNCONDITIONALLY below — its historical
        # behavior refreshes the aggregate even on a no-op flip.
        flipped = self.bulk_archive_relations(
            'edge_id = ? AND relation = ?', [edge_id, relation], archived_by,
            null_embeddings=True, recompute_weight=False)
        if flipped:
            result['flipped'] = True
            result['deltas'] = [{'field': 'archived', 'old': 0, 'new': 1}]

        # Recompute aggregate weight from remaining active relations
        self._update_aggregate_weight(edge_id)
        commit_unless_batched(self.conn)
        return result

    def rename_relation(self, edge_id: str, old_relation: str,
                        new_relation: str, encoding_source: str) -> None:
        """Rename a relation on an edge in place — updates the matching row's
        relation + encoding_source. No weight recompute: a rename changes neither
        weights nor the active-relation count. Commit gated on self.conn.in_batch.

        The relation string is part of compose_edge_text, so the stored embedding
        is now stale — NULL it here (storage-only invalidation, DAL-appropriate)
        and enqueue the edge for async re-embed by the embed_queue worker. Callers
        (reclassify, revise_edge) stay embedding-ignorant; the worker owns the
        actual re-embed via Brain.backfill_edge_embeddings.
        """
        self.conn.execute(
            "UPDATE edge_relations SET relation = ?, encoding_source = ?, "
            "embedding = NULL, embedding_model = NULL "
            "WHERE edge_id = ? AND relation = ?",
            (new_relation, encoding_source, edge_id, old_relation))
        commit_unless_batched(self.conn)
        self._enqueue_edge_embed(edge_id, 'rename_relation')

    def _update_aggregate_weight(self, edge_id):
        """Set edges.weight to max weight across ACTIVE relation rows.

        Archived relations do not contribute — they're history, not signal.
        When all relations on an edge are archived, edges.weight is
        explicitly set to 0 so weight-based reads (min_weight filters,
        get_well_connected) skip the orphan edges row. The row itself
        persists for edge_id stability; reads that JOIN edge_relations
        with archived=0 already get zero rows regardless.

        No silent no-op — the weight is always written (0 or max), so the
        edges row state reflects the truth of its active relations.
        """
        row = self.conn.execute(
            'SELECT MAX(weight) FROM edge_relations '
            'WHERE edge_id = ? AND archived = 0',
            (edge_id,)
        ).fetchone()
        new_weight = row[0] if row and row[0] is not None else 0.0
        self.conn.execute(
            'UPDATE edges SET weight = ? WHERE edge_id = ?',
            (new_weight, edge_id))


