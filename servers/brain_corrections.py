"""
brain — BrainCorrections Mixin

Walks correction-aspect edges (corrects, supersedes, reframes, resolves, ...)
to attach corrector context to any set of node ids. Every canonical node pull
(brain.get_node) attaches `_corrections` via this method.

If you're writing a new fetcher that bypasses brain.get_node, decide
explicitly whether you need corrections and call brain.correction_enrich(ids)
if you do. Don't re-implement the aspect-edge walk.

Mixed into the Brain class via multiple inheritance. References self.conn,
self.aspects, self._log_error — provided by Brain.__init__.
"""

import sys
from typing import Any, Dict, Iterable, List


class BrainCorrectionsMixin:
    """Correction-aspect graph walks for Brain."""

    def correction_enrich(self, node_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Find corrections for a set of nodes via the correction_improvement aspect.

        Walks edge_relations whose `relation` is in
        `self.aspects.correction_improvement.edge_relations` (corrects,
        supersedes, reframes, resolves, addresses, fixes, ...). Bidirectional —
        both incoming and outgoing edges.

        Returns dict {node_id: [correction_dict, ...]} keyed by BOTH the
        8-char short form AND the full id, so callers indexing by either form
        resolve.

        Each correction_dict carries the HEAVY payload — renderer slices it
        per consumer (HAIKU_FORMAT: balanced, ENCODER/HEALER_FORMAT: full).

            id              (str)  neighbor short id
            title           (str)  neighbor title
            type            (str)  neighbor node type
            direction       (str)  'corrects' (this node is the corrector) |
                                   'corrected_by' (this node was corrected)
            relation        (str)  specific aspect verb
            edge_description (str) the edge's `why` text
            content         (str)  neighbor's full content (renderer slices)
            reasoning       (str)  neighbor's reasoning metadata
            user_raw_quote  (str)  neighbor's user_raw_quote metadata
            anchor_raw_quote (str) neighbor's anchor_raw_quote metadata

        Loud-by-default: any failure inside the aspect walk gets logged via
        self._log_error AND returns an empty dict so the calling pipeline
        (get_node, traverse, future fetchers) degrades gracefully instead of
        taking the whole query down.
        """
        if not node_ids:
            return {}

        try:
            return self._correction_enrich_impl(node_ids)
        except Exception as e:
            try:
                self._log_error(
                    'correction_enrich', e,
                    'aspect-edge walk failed for %d ids' % len(list(node_ids)))
            except Exception:
                # _log_error itself failing is rare (DB / rate limit) — don't
                # cascade; print so it shows in daemon.log.
                print('[correction_enrich] error + _log_error failed: %r' % e,
                      file=sys.stderr, flush=True)
            return {}

    def _correction_enrich_impl(self, node_ids):
        """Body of correction_enrich — separated so the public method can wrap
        it in a single try/except without indentation pyramid.
        """
        from .dal import NodeDAL, GraphDAL
        from .dal_metadata import MetadataDAL

        conn = self.conn
        ndal = NodeDAL(conn)
        full_to_short = {}
        full_ids = []
        for nid in node_ids:
            if not nid:
                continue
            full = ndal.resolve_id(nid) if len(str(nid)) < 16 else nid
            if full and full not in full_to_short:
                full_to_short[full] = full[:8]
                full_ids.append(full)
        if not full_ids:
            return {}

        correction_relations = tuple(self.aspects.correction_improvement.edge_relations)
        if not correction_relations:
            return {}

        graph_dal = GraphDAL(conn)
        connections = graph_dal.get_connections_bulk(
            full_ids,
            include_relations=correction_relations,
            include_archived=False,
            include_neighbor_archived=False)

        neighbor_ids = set()
        for _owner_id, conns in connections.items():
            for c in conns:
                neighbor_ids.add(c['id'])

        if not neighbor_ids:
            return {}

        # Keys must mirror render_corrections's heavy-mode render list, otherwise
        # the renderer reads a key the data layer never fetched.
        naked_by_id = ndal.get_bulk(list(neighbor_ids))
        meta_dal = MetadataDAL(conn)
        meta_by_id = meta_dal.get_fields_bulk(
            list(neighbor_ids),
            ['reasoning', 'user_raw_quote', 'anchor_raw_quote'])

        corrections = {}
        for owner_full, conns in connections.items():
            owner_short = full_to_short[owner_full]
            bucket = []
            for c in conns:
                neighbor_id = c['id']
                naked = naked_by_id.get(neighbor_id) or {}
                meta = meta_by_id.get(neighbor_id) or {}
                content = naked.get('content') or ''
                ntype = naked.get('type') or c.get('type') or ''
                title = c.get('title') or naked.get('title') or ''
                edge_dir = c.get('direction')  # 'outgoing' | 'incoming'
                direction = 'corrects' if edge_dir == 'outgoing' else 'corrected_by'
                for rel in c.get('relations', []) or []:
                    bucket.append({
                        'id': neighbor_id[:8],
                        'title': title,
                        'type': ntype,
                        'direction': direction,
                        'relation': rel.get('relation') or '',
                        'edge_description': rel.get('description') or '',
                        'content': content,
                        'reasoning': meta.get('reasoning') or '',
                        'user_raw_quote': meta.get('user_raw_quote') or '',
                        'anchor_raw_quote': meta.get('anchor_raw_quote') or '',
                    })

            if not bucket:
                continue

            # Dedup on (neighbor_id, direction, relation) — defensive against
            # any future change in get_connections_bulk grouping.
            seen = set()
            deduped = []
            for item in bucket:
                key = (item['id'], item['direction'], item['relation'])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            corrections[owner_short] = deduped
            corrections[owner_full] = deduped

        return corrections
