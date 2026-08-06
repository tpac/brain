"""
brain — BrainRemember Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE
from .dal import VectorDAL
from .dal_graph import ABSORB_EXCLUDED_RELATIONS
from .clock import iso_cutoff, iso_now
from .brain_constants import (
    ENRICHMENT_NEIGHBOR_COUNT,
    ENRICHMENT_PROMPT_TEMPLATE,
)
from typing import Any, Dict, List, Optional, Set
import json
import math
import re
import sys
import time

from .brain_constants import (
    TFIDF_STOP_WORDS,
)


# ── connect_to catalog-title matching (deterministic, no vectors) ──

_TITLE_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _title_tokens(text):
    """Normalize a title to its token sequence: NFKD → strip combining marks
    → lowercase → split on every non-alphanumeric run (hyphen/underscore/
    em-dash/percent variance vanishes; numbers and hashes survive — they're
    the distinctive tokens). No stemming, no stopwords: predictability IS
    the safety property of the near-title matcher.

    The diacritic strip mirrors FTS5 unicode61's remove_diacritics default —
    the two tokenizers must agree or a probe goes dead: unicode61 indexes
    'über' as 'uber', and a bare [a-z0-9] scan would emit the probe 'ber',
    which matches nothing (bug 69c2cbab #4 — tokenizer correspondence)."""
    import unicodedata
    decomposed = unicodedata.normalize('NFKD', text or '')
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return _TITLE_TOKEN_RE.findall(stripped.lower())


def _token_edit_distance(a, b, cap):
    """Levenshtein over token SEQUENCES (order-sensitive on purpose — a
    reordered title is not 'a bit off'). Early-exits at `cap`: any distance
    beyond it returns cap+1 — callers only need 'too far', not how far."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a):
        cur = [i + 1]
        row_min = cur[0]
        for j, tb in enumerate(b):
            c = min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ta != tb))
            cur.append(c)
            row_min = min(row_min, c)
        if row_min > cap:
            return cap + 1
        prev = cur
    return min(prev[-1], cap + 1)


class BrainRememberMixin:
    """Remember methods for Brain."""

    # ═══════════════════════════════════════════════════════════════
    # Unified metadata storage — single path for remember() and revise()
    # ═══════════════════════════════════════════════════════════════

    # Fields that are control parameters, not node metadata.
    # These are consumed by remember()/revise() logic and should never be stored.
    # `connections` and `auto_connect` are RETIRED remember() params (the params
    # were removed 2026-06-18): `connect_to` replaced store-time edge creation,
    # and the co_accessed-on-remember behavior `auto_connect` gated was deleted
    # 2026-05-31. Both names stay here as legacy-caller swallow guards —
    # `validate_field` silently accepts unknown kwargs, so a stray
    # `connections=` / `auto_connect=` (still passed by some test call sites)
    # would otherwise land as junk KV metadata. A stray `connections=` is ALSO
    # logged loudly in remember() (event 'remember_connections_retired'), since
    # it once had a store-time side effect worth tracking; `auto_connect` was a
    # pure toggle, so swallowing it silently is harmless.
    _CONTROL_FIELDS = frozenset({
        'connections', 'auto_connect',
        'reason', 'updates', 'connect_to',
    })

    def _store_node_metadata(self, node_id: str, fields: Dict[str, Any],
                             caller: str = 'unknown') -> int:
        """Store metadata fields for a node. Single path for all write operations.

        Routes each field to the correct storage:
          - STRUCTURAL_FIELDS → already on nodes table (skip)
          - situation → node_metadata_kv (canonical; _situation embedding
            derived later by backfill)
          - PROMOTED metadata_kv fields → node_metadata_kv
          - Emergent/unknown fields → node_metadata_kv
          - Control fields → skip silently (connections, auto_connect, etc.)

        Warns on fields that don't match any storage path.

        Returns count of fields stored.
        """
        from .contract import STRUCTURAL_FIELDS, PROMOTED_FIELDS

        kv_fields = {}
        stored = 0

        for field, value in fields.items():
            # Control params — consumed by callers, never stored
            if field in self._CONTROL_FIELDS:
                continue

            # Structural — already on nodes table, handled by INSERT/UPDATE
            if field in STRUCTURAL_FIELDS:
                continue

            # Empty values — skip
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Situation lives in node_metadata_kv as of v24 — alongside
            # question, reasoning, etc. The embedding (BLOB) is generated
            # later by backfill_vectors() reading from kv.
            #
            # Pass raw values through to set_many — its _encode_value handles
            # str / list / dict / primitives consistently. Doing str(value)
            # here would str()-ify lists into Python repr (`"['a','b']"`)
            # which isn't JSON-parseable. Aspects (Step 5b) need clean lists.
            if field == 'situation':
                kv_fields['situation'] = value
                continue

            # Promoted metadata_kv field — store in KV
            if field in PROMOTED_FIELDS:
                pf = PROMOTED_FIELDS[field]
                if pf.get('store') == 'metadata_kv':
                    kv_fields[field] = value
                    continue
                # Promoted field with different store — log error so it's visible
                self._log_error('store_metadata_unhandled',
                                ValueError('field "%s" (store=%s) not handled' % (
                                    field, pf.get('store'))),
                                '%s: node %s' % (caller, node_id[:8]))
                continue

            # Emergent field — any unknown field goes to metadata_kv
            kv_fields[field] = value

        if kv_fields:
            try:
                count = self._meta_kv.set_many(node_id, kv_fields)
                stored += count
            except Exception as _e:
                self._log_error('store_metadata_kv', _e,
                                '%s: KV for %s (%d fields)' % (
                                    caller, node_id[:8], len(kv_fields)))

        return stored

    # ═══════════════════════════════════════════════════════════════
    # Unified archive — single path for all archive operations
    # ═══════════════════════════════════════════════════════════════

    def _resync_vector_cache(self, where: str) -> None:
        """Restore cache == DB after a rollback. Cache mutations inside a
        batch envelope are eager (the DAL drops cache rows when it deletes DB
        rows); a rollback restores the DB but not the cache — without this,
        a rolled-back revise/archive/absorb leaves a LIVE node invisible to
        cache-served scans until restart (the inverse of the 2026-07-17
        healer bug). reload() is one SELECT; rollback is the rare loud path,
        so the cost is fine. Plain VectorDAL has no cache — nothing to do."""
        reload_fn = getattr(self._vec_dal, 'reload', None)
        if reload_fn is None:
            return
        try:
            reload_fn()
        except Exception as _e:
            self._log_error('vector_cache_resync', _e, where)

    def _deindex_node(self, node_id: str, include_tfidf: bool = True) -> int:
        """Remove a node from the search indexes — the single de-indexing path
        shared by archive_node and delete_node_cascade, so the two can never
        again cover different index subsets (the drift class this consolidation
        kills). Returns enrichment rows deleted (trace metadata).

        include_tfidf distinguishes the two callers' correct behavior:
          - cascade (hard delete, default True): drop EVERYTHING — enrichment
            vectors (+cache), tfidf/node_vectors, FTS5.
          - archive (soft delete, include_tfidf=False): drop the expensive
            embeddings + FTS row, but KEEP the tfidf rows. Deleting them would
            (a) inflate doc_freq — TfIdfDAL.delete_for_node removes node_vectors
            without decrementing the per-term counts, skewing idf on every
            archive — and (b) strip lexical reachability that include_archived
            forensics rely on. The node row survives; archived=0 filters keep
            it out of live recall.

        Composes with batch envelopes: each store's delete gates its commit on
        conn.in_batch. Fts5DAL.delete tolerates a missing table and logs its
        own failures (loud-by-default lives there); the sqlite_master probe
        just avoids a stderr line on test DBs that lack the virtual table."""
        vectors_deleted = self._vec_dal.delete_for_node(node_id)
        if include_tfidf:
            self._tfidf.delete_for_node(node_id)
        has_fts5 = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone() is not None
        if has_fts5:
            self._fts.delete(node_id)   # self-guarding: logs internally
        self._maybe_commit()            # Fts5DAL.delete doesn't self-commit
        return vectors_deleted

    def delete_node_cascade(self, node_id: str) -> None:
        """Hard-delete a node and EVERY child-table row, routed through the
        owning DALs — the one place that knows all child tables. Replaces the
        hand-rolled NodeDAL.purge, which deleted only node_enrichments /
        node_metadata_kv / edges / nodes and LEAKED node_vectors +
        node_source_refs (orphan rows on every purge). Irreversible — use
        archive_node() for soft delete.

        Atomic: the deletes run inside one connection-level batch envelope so a
        mid-cascade failure leaves the node intact rather than half-deleted
        (matching purge's old single-commit behaviour, now complete). Composes
        safely if called inside an outer batch — it defers the commit to the
        owner via the saved in_batch state."""
        if not node_id:
            return
        prior = self.conn.in_batch
        self.conn.in_batch = True
        try:
            self._deindex_node(node_id)   # enrichments (+cache), tfidf, FTS5
            self._meta_kv.delete_all(node_id)          # node_metadata_kv
            self._graph.hard_delete_node_edges(node_id)  # edges + edge_relations
            self._source_refs.delete_source_refs(node_id)    # node_source_refs
            self._nodes.delete(node_id)                # nodes
            if not prior:
                self.conn.commit()  # commit-ok: single atomic node-delete cascade
        except Exception:
            if not prior:
                self.conn.rollback()
                self._resync_vector_cache('delete_node_cascade rollback')
            raise
        finally:
            self.conn.in_batch = prior

    def archive_exempt_relations(self) -> tuple:
        """Relations exempt from dangling-edge archival — the survivor-redirect
        links (survivor_lineage aspect: absorbed_into) that must outlive the
        nodes they leave. Single source for archive_node + the Healer; the DAL
        stays aspect-agnostic and just takes the strings.

        LOUD on empty: survivor_lineage is a REQUIRED aspect, so an empty result
        means the registry is unset (init failed — brain.py leaves self.aspects
        unset by design) or the working copy predates the aspect and hasn't
        self-healed yet. Either way the exemption is DISABLED and the
        dangling-edge reaper will scrub absorbed_into redirect edges — a silent
        correctness failure, so we log it (rate-limited) instead of letting it
        pass quietly or raising AttributeError mid-archive."""
        rels = ()
        if getattr(self, 'aspects', None) is not None:
            rels = tuple(self.aspects.relations_in(['survivor_lineage']))
        if not rels:
            self._log_error(
                'survivor_lineage_exempt_empty',
                ValueError('survivor_lineage aspect missing/empty'),
                'absorbed_into archival exemption DISABLED — redirect edges '
                'will be reaped by the dangling-edge sweep')
        return rels

    def archive_node(self, node_id: str, archived_by: str,
                     reason: str = '', survivor_id: str = None,
                     extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Archive a node. Single path for all callers.

        What it does:
          1. Guards: rejects locked/critical nodes
          2. Sets archived=1, updated_at=now
          3. Stores audit metadata: archived_by, archived_reason, archived_at
          4. Soft-archives edge_relations (v25 — archived=1 preserves history
             for future recovery; edges aggregate row stays for edge_id stability)
          5. Deletes vectors from node_enrichments (embeddings are expensive to keep)
          6. Removes from FTS5 index

        Args:
            node_id: Node to archive.
            archived_by: Who is archiving. Convention: "s2:consolidation",
                         "s2:community_detection", "hook:integrity", "anchor", etc.
            reason: Human-readable reason for the archive.
            survivor_id: Live node this one's content survives in (absorb,
                         supersession). Writes the `_sys_archived_survivor_id`
                         pointer resolve_live walks + the `absorbed_into`
                         lineage edge. A first-class parameter, NOT an extra
                         key — a misspelled extra key silently loses lineage
                         (the `superseded_by` bug, 2026-07-30).
            extra: Optional dict of additional AUDIT metadata to store.
                   Never carries behavior; survivor-looking keys are refused
                   loudly (see tripwire below).

        Returns:
            Dict with ok=True/False and details.
        """
        ts = iso_now()

        # Fetch node
        row = self.conn.execute(
            'SELECT id, locked, critical, title, type FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'ok': False, 'error': 'Node not found', 'node_id': node_id}

        full_id, locked, critical, title, node_type = row

        # Guard: never archive locked or critical nodes
        if locked or critical:
            flag = 'locked' if locked else 'critical'
            self._log_error('archive_guarded',
                            ValueError('Cannot archive %s node' % flag),
                            '%s tried to archive %s "%s"' % (
                                archived_by, node_id[:8], (title or '')[:40]))
            return {'ok': False, 'error': 'Cannot archive %s node' % flag,
                    'node_id': node_id}

        # 1. Set archived=1
        self.conn.execute(
            'UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?',
            (ts, full_id))

        # 2. Store audit metadata (_sys_ prefix = system fields, filtered from LLM rendering)
        audit = {
            '_sys_archived_by': archived_by,
            '_sys_archived_reason': reason or 'no reason provided',
            '_sys_archived_at': ts,
        }
        if survivor_id:
            audit['_sys_archived_survivor_id'] = str(survivor_id)
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                # Tripwire: survivor semantics must go through the survivor_id
                # PARAMETER — a survivor-looking extra key is the exact
                # misspelling that silently orphaned 9 handoff nodes
                # (`superseded_by`, 2026-07-30). Refuse it loudly rather than
                # store a pointer resolve_live will never walk.
                if k in ('survivor_id', 'superseded_by', 'survivor',
                         'consolidated_into', 'absorbed_into'):
                    self._log_warning(
                        'archive_survivor_key_in_extra',
                        "extra key %r on archive of %s looks like survivor "
                        "lineage — dropped; pass survivor_id= instead"
                        % (k, node_id[:8]), 'archived_by=%s' % archived_by)
                    continue
                audit['_sys_archived_%s' % k] = str(v)
        try:
            self._meta_kv.set_many(full_id, audit)
        except Exception as _e:
            self._log_error('archive_metadata', _e,
                            'storing audit for %s' % node_id[:8])

        # 3–5 run inside ONE connection batch envelope. delete_node_edges and
        # add_relation self-commit via commit_unless_batched, so without this a
        # STANDALONE archive (consolidation orphan-heal, hook:integrity — paths
        # that don't set conn.in_batch) would commit the node+edges and then a
        # failure in the vector/FTS cleanup would leave a half-archived node:
        # archived=1 but still in FTS5 → it resurfaces in recall (the exact
        # dead-node-leak class). Composes with an outer batch (absorb): defers
        # the commit/rollback to the owner via the saved in_batch state.
        # Mirrors _delete's cascade envelope.
        prior = self.conn.in_batch
        self.conn.in_batch = True
        try:
            # 3. Soft-archive edge_relations touching this node (v25) via the
            # DAL — single source, no inline SQL. The survivor-redirect
            # relations (survivor_lineage aspect: absorbed_into) are EXEMPT —
            # they must outlive the node so the resolve_live chain A→B→C
            # survives B's own archival.
            edges_deleted = self._graph.delete_node_edges(
                full_id, archived_by=archived_by,
                exempt_relations=self.archive_exempt_relations())

            # 3b. Survivor-redirect edge. When this archive carries a survivor
            # (the absorb op, consolidation merges, handoff supersession all
            # route here via the survivor_id parameter), record
            # absorbed→survivor as a first-class `absorbed_into` edge,
            # multi-homed in correction_improvement (correction_enrich walks it)
            # + survivor_lineage (the redirect/archival-exempt role). Written
            # AFTER the soft-archive above (which exempts it) so it lands and
            # stays archived=0. `_sys_archived_survivor_id` is still written
            # (step 2 audit) as the backfill source + resolve_live's read path.
            if survivor_id and survivor_id != full_id:
                try:
                    self._graph.add_relation(
                        full_id, survivor_id, 'absorbed_into',
                        description=reason or 'absorbed into %s' % survivor_id[:8],
                        encoding_source=archived_by)
                except Exception as _e:
                    self._log_error('archive_absorbed_into_edge', _e,
                                    'absorbed_into %s -> %s' % (
                                        full_id[:8], str(survivor_id)[:8]))

            # 4. De-index INSIDE the archive transaction — atomic with the
            # archived=1 flag: a de-index failure rolls the WHOLE archive back
            # (the caller sees a clean failure, no half-archived node, no lost
            # trace). Soft delete → include_tfidf=False keeps the lexical rows.
            # The eager cache drop is covered by resync-on-rollback below; when
            # archive runs inside an outer envelope, the deletes join the outer
            # commit and that envelope's rollback resyncs.
            vectors_deleted = self._deindex_node(full_id, include_tfidf=False)

            if not prior:
                self.conn.commit()  # commit-ok: single atomic archive
        except Exception:
            if not prior:
                self.conn.rollback()
                self._resync_vector_cache('archive_node rollback')
            raise
        finally:
            self.conn.in_batch = prior

        # 7. Trace event — S3 + dashboards see who archived what.
        # Tracing must never block the archive itself, but a failure
        # here is real audit data loss — log it so we know.
        try:
            self._trace_dal.append(
                chain_id='archive-%s' % full_id[:8],
                scale='s0', event_type='delta', ref_type='tool_result',
                summary='archived %s by %s' % (full_id[:8], archived_by),
                metadata={
                    'node_id': full_id,
                    'title': (title or '')[:80],
                    'type': node_type,
                    'archived_by': archived_by,
                    'reason': reason,
                    'edges_deleted': edges_deleted,
                    'vectors_deleted': vectors_deleted,
                })
        except Exception as _e:
            self._log_error('archive_trace', _e,
                            'trace write for archived %s' % full_id[:8])

        return {
            'ok': True,
            'node_id': full_id,
            'title': (title or '')[:60],
            'type': node_type,
            'archived_by': archived_by,
            'reason': reason,
            'edges_deleted': edges_deleted,
            'vectors_deleted': vectors_deleted,
        }

    def absorb(self, survivor_id: str, absorbed_id: str,
               content: str = None, reason: str = '', updates=None,
               prune_edges=None, drop_fields=None,
               archived_by: str = 'anchor', **kwargs) -> Dict[str, Any]:
        """Lossless merge: fold absorbed_id INTO survivor_id, then archive absorbed.

        Transfer-by-default. Everything the caller doesn't deliberately override
        moves structurally — so a merge can't silently drop information the way
        the imperative revise+connect+archive dance does (preservation audit,
        node 988de522). The caller hand-writes only the synthesis (`content`)
        and names what to drop (`prune_edges`, `drop_fields`).

        Transfers:
          - source_refs  → union onto survivor (INSERT OR IGNORE dedups)
          - edges        → re-point absorbed's external edges to survivor,
                           upsert-dedup, drop the absorbed<->survivor intra edge;
                           noise relations excluded by get_connections_bulk default
          - access_count → survivor += absorbed (usage history is additive)
          - metadata KV  → fill keys survivor LACKS from absorbed; survivor wins;
                           `_sys_` keys skipped
          - ANY field    → caller overrides via `content`, an `updates` dict, OR
                           field kwargs (title=, confidence=, situation=, type=,
                           critical=, ...) — the SAME shape as revise(). Applied
                           through revise() AFTER the auto-transfers, so explicit
                           overrides win over KV-fill. Untouched fields keep the
                           survivor's value; absorbed's content persists on the
                           archived husk. Auto-transfers above cover only the
                           unambiguously-additive dimensions; everything
                           judgment-laden (confidence, title, ...) is explicit.
        Then archive_node(absorbed, survivor_id=...) for provenance.

        Guards: absorbed must be archivable (NOT locked/critical) — the type
        constraint that makes "a locked node is always the survivor" structural.
        survivor must exist and be unarchived; survivor != absorbed.

        Atomicity: the merge is several writes, so it is all-or-nothing. In a
        batch we nest a SAVEPOINT (a mid-merge failure rolls back ONLY our writes,
        not the whole batch); standalone we flip conn.in_batch so the composed DAL
        writers don't commit mid-merge, then commit once at the end — or roll back
        on ANY failure (a raise, a failed override-revise, or a refused archive).
        No partial merge ever commits.
        """
        if survivor_id == absorbed_id:
            return {'ok': False, 'error': 'survivor and absorbed are the same node'}

        rows = {r[0]: r for r in self.conn.execute(
            'SELECT id, locked, critical, archived, access_count '
            'FROM nodes WHERE id IN (?, ?)',
            (survivor_id, absorbed_id)).fetchall()}
        if survivor_id not in rows:
            return {'ok': False, 'error': 'survivor not found', 'node_id': survivor_id}
        if absorbed_id not in rows:
            return {'ok': False, 'error': 'absorbed not found', 'node_id': absorbed_id}
        if rows[survivor_id][3]:
            return {'ok': False, 'error': 'survivor is archived', 'node_id': survivor_id}

        a_locked, a_critical, a_archived, a_access = rows[absorbed_id][1:]
        if a_locked or a_critical:
            flag = 'locked' if a_locked else 'critical'
            self._log_warning(
                'absorb_guarded',
                'Cannot absorb %s node %s into %s — absorbed must be archivable' % (
                    flag, absorbed_id[:8], survivor_id[:8]),
                'absorb refused: the absorbed node is %s (only the survivor may '
                'be locked/critical)' % flag)
            return {'ok': False, 'error': 'Cannot absorb %s node' % flag,
                    'absorbed_id': absorbed_id}
        if a_archived:
            return {'ok': False, 'error': 'absorbed node is already archived',
                    'absorbed_id': absorbed_id}

        # Caller field overrides — revise() shape: content / updates / kwargs.
        field_updates = dict(updates or {})
        field_updates.update(kwargs)
        if content is not None:
            field_updates['content'] = content

        report = {'ok': True, 'survivor_id': survivor_id,
                  'absorbed_id': absorbed_id, 'absorbed_archived': False}

        # ── Atomicity envelope (see docstring) ──
        was_batch = self.conn.in_batch
        if was_batch:
            self.conn.execute('SAVEPOINT absorb_sp')
        self.conn.in_batch = True   # composed DAL writers must NOT commit mid-merge
        success = False
        try:
            # 1. source_refs — union onto survivor
            absorbed_refs = self._source_refs.get_source_refs(absorbed_id)
            report['source_refs_added'] = (
                self._source_refs.add_source_refs(survivor_id, absorbed_refs)
                if absorbed_refs else 0)

            # 2. edges — re-point absorbed's external edges to survivor,
            # preserving each relation's weight + description.
            # community_member never migrates: placement is the community
            # unit's judged decision, not a merge side effect (see
            # ABSORB_EXCLUDED_RELATIONS in dal_graph.py, imported at module top).
            prune = set(prune_edges or [])
            conns = self._graph.get_connections_bulk(
                [absorbed_id]).get(absorbed_id, [])
            # Each migration's add_relation result (edge_id + deltas) is kept —
            # the dispatch caller shapes them into the mutation manifest so the
            # emitter can trace the re-pointed edges. Discarded before step 7.
            migrated_edges = []
            for c in conns:
                neighbor = c['id']
                if neighbor == survivor_id:
                    continue  # intra-pair edge dies with the absorbed node
                outgoing = c.get('direction') == 'outgoing'
                for rel in c.get('relations', []):
                    relation = rel.get('relation')
                    if (not relation or relation in prune
                            or relation in ABSORB_EXCLUDED_RELATIONS):
                        continue
                    src, tgt = ((survivor_id, neighbor) if outgoing
                                else (neighbor, survivor_id))
                    kw = {'encoding_source': archived_by}
                    if rel.get('description'):
                        kw['description'] = rel['description']
                    if rel.get('weight') is not None:
                        kw['weight'] = rel['weight']
                    res = self._graph.add_relation(src, tgt, relation, **kw)
                    migrated_edges.append({
                        'source_id': src, 'target_id': tgt,
                        'relation': relation,
                        'edge_id': res.get('edge_id') or '',
                        'deltas': res.get('deltas') or [],
                        'warnings': list(res.get('warnings') or []),
                    })
            report['edges_migrated'] = len(migrated_edges)
            report['migrated_edges'] = migrated_edges

            # 3. access_count — additive (usage history)
            if a_access:
                self.conn.execute(
                    'UPDATE nodes SET access_count = access_count + ? WHERE id = ?',
                    (a_access, survivor_id))
            report['access_count_added'] = a_access or 0

            # 4 + 5. Fill KV the survivor LACKS from absorbed, then apply caller
            # overrides — through ONE revise() call so embedding-bearing fields
            # (situation, ...) get re-embedded and there is a single field-write
            # path. Caller overrides win (field_updates last); _sys_ never moves.
            drop = set(drop_fields or [])
            kv = self._meta_kv.get_all_bulk([survivor_id, absorbed_id])
            s_kv, a_kv = kv.get(survivor_id, {}), kv.get(absorbed_id, {})
            fill = {k: v for k, v in a_kv.items()
                    if not k.startswith('_sys_') and k not in drop
                    and k not in field_updates
                    and (v or '').strip() and not (s_kv.get(k) or '').strip()}
            # Voice fields are the exception to survivor-wins-drop. A distinctive
            # operator/Anchor quote on the absorbed node is meaning paraphrase
            # can't recover; dropping it because the survivor already has a
            # (different) quote is real signal loss. When BOTH carry a quote and
            # they're genuinely distinct, merge-append instead of dropping the
            # absorbed one. (survivor-lacks is already covered by `fill`; caller
            # override via field_updates and `drop` still win below.)
            voice = {}
            for vf in ('user_raw_quote', 'anchor_raw_quote'):
                if vf in drop or vf in field_updates:
                    continue
                s_val = (s_kv.get(vf) or '').strip()
                a_val = (a_kv.get(vf) or '').strip()
                if (s_val and a_val and a_val != s_val
                        and a_val not in s_val and s_val not in a_val):
                    voice[vf] = '%s\n\n%s' % (s_val, a_val)
            merged = {**fill, **voice, **field_updates}
            rev_failed = False
            if merged:
                rev = self.revise(survivor_id, updates=merged,
                                  reason=reason or 'absorb %s' % absorbed_id[:8])
                report['voice_merged'] = sorted(voice.keys())
                report['fields_filled'] = sorted(fill.keys())
                report['fields_revised'] = sorted(field_updates.keys())
                # Survivor-revise deltas ride the report so the dispatch caller
                # can manifest a node_revised row for the merge (emitter input);
                # discarded before step 7.
                report['deltas'] = rev.get('deltas', [])
                if rev.get('warnings'):
                    report['revise_warnings'] = rev['warnings']
                if rev.get('error'):
                    report['ok'] = False
                    report['revise_error'] = rev['error']
                    rev_failed = True

            # 6. archive the absorbed node — SKIPPED if the override-revise failed,
            # so a failed synthesis never destroys the source.
            if not rev_failed:
                arch = self.archive_node(
                    absorbed_id, archived_by=archived_by,
                    reason=reason or 'absorbed into %s' % survivor_id[:8],
                    survivor_id=survivor_id)
                report['absorbed_archived'] = arch.get('ok', False)
                if arch.get('ok'):
                    success = True
                else:
                    report['ok'] = False
                    report['archive_error'] = arch.get('error')
        except Exception:
            self._absorb_unwind(was_batch)
            raise

        if success:
            if was_batch:
                self.conn.execute('RELEASE absorb_sp')
            self.conn.in_batch = was_batch
            self._maybe_commit()
        else:
            # guard/override/archive failure mid-merge — undo every transfer so
            # no half-merge is ever committed.
            self._absorb_unwind(was_batch)
        return report

    def _absorb_unwind(self, was_batch: bool) -> None:
        """Roll back an in-flight absorb. In a batch: roll the SAVEPOINT back
        (the rest of the batch survives). Standalone: rollback the implicit txn.
        Always restores conn.in_batch."""
        try:
            if was_batch:
                self.conn.execute('ROLLBACK TO absorb_sp')
                self.conn.execute('RELEASE absorb_sp')
            else:
                self.conn.rollback()
        finally:
            # In finally, not the try body: absorb archives the absorbed node
            # mid-SAVEPOINT (eager cache drop), so the cache must be resynced
            # even if the ROLLBACK TO itself throws (e.g. a lost savepoint) —
            # otherwise the dropped rows persist and a live node goes
            # cache-invisible until restart.
            self._resync_vector_cache('absorb unwind')
            self.conn.in_batch = was_batch

    def _tfidf_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF: expand CamelCase, lowercase, remove stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (length > 2, non-stopword)
        """
        if not text:
            return []

        # Split CamelCase before lowercasing: "UserDashboard" → "User Dashboard"
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        expanded = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', expanded)

        # Lowercase, remove non-alphanumeric (keep hyphens, dots), split
        tokens = expanded.lower()
        tokens = re.sub(r'[^a-z0-9\s\-\.]', ' ', tokens)
        tokens = re.split(r'[\s\-\.]+', tokens)

        # Filter: length > 2, not stopword, remove trailing non-alphanumeric
        result = []
        for w in tokens:
            w = re.sub(r'[^a-z0-9]', '', w)
            if len(w) > 2 and w not in TFIDF_STOP_WORDS:
                result.append(w)

        return result

    def _compute_tf(self, text: str) -> Dict[str, float]:
        """
        Compute term frequency vector (augmented TF formula).

        Args:
            text: Text to analyze

        Returns:
            Dict of term→TF value (0-1)
        """
        tokens = self._tfidf_tokenize(text)
        if not tokens:
            return {}

        # Count term frequencies
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Augmented TF: 0.5 + 0.5 * (count / max_freq)
        max_freq = max(freq.values()) if freq else 1
        tf = {}
        for term, count in freq.items():
            tf[term] = 0.5 + 0.5 * (count / max_freq)

        return tf

    def _store_tfidf_vector(self, node_id: str, title: str, content: Optional[str]):
        """
        Store TF-IDF vector for a node (title + content).

        Args:
            node_id: Node ID
            title: Node title
            content: Node content (optional)

        Note: previously accepted a `keywords` argument that contributed
        to the TF-IDF text. Dropped 2026-05-24 — the auto-extracted
        keywords field was a downstream tokenizer dump producing
        near-duplicate noise (idiotic./idiotic, r1r10/r1-r10), which
        actively hurt precision. Title + content is the cleaner signal.
        """
        full_text = ' '.join(filter(None, [title, content]))
        tf = self._compute_tf(full_text)
        # TfIdfDAL.store_tf_vector replaces the node's vectors and bumps doc_freq
        # per term; it commits via commit_unless_batched (same batch gate
        # _maybe_commit used), so this stays batch-atomic inside brain_batch.
        self._tfidf.store_tf_vector(node_id, tf)

    # _tfidf_score (single-node cosine) removed 2026-05-30 (DAL cleanup Phase 3b)
    # — 0 callers repo-wide (recall uses _batch_tfidf_scores). The live batch
    # scorer below is a read on the recall hot path; its TfIdfDAL migration is
    # deferred to Phase 4 (reads) with a decode_funnel before/after.

    def _batch_tfidf_scores(self, query_terms: List[str], node_ids: List[str]) -> Dict[str, float]:
        """
        Batch compute TF-IDF scores for multiple nodes (efficient).

        Args:
            query_terms: Tokenized query
            node_ids: List of node IDs to score

        Returns:
            Dict of node_id→score
        """
        if not query_terms or not node_ids:
            return {}

        total_docs = self._get_node_count()
        if total_docs == 0:
            return {}

        # Precompute IDF for all query terms
        idf_map = {}
        for term in set(query_terms):
            # `or 1`: an absent term defaults to df=1 (get_doc_freq returns 0),
            # preserving the original IDF. Existing terms always have count>=1,
            # so `or 1` only ever rewrites the absent-term case.
            df = self._tfidf.get_doc_freq(term) or 1
            idf_map[term] = math.log((total_docs + 1) / (df + 1)) + 1

        # Build query vector
        query_vec = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0) + 1

        q_max = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] /= q_max

        # Query norm (constant for all docs)
        query_norm_sq = 0
        for term, q_val in query_vec.items():
            idf = idf_map.get(term, 1)
            query_norm_sq += (q_val * idf) ** 2

        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0:
            return {}

        # Get all matching vectors in one query (term+node filtered)
        unique_terms = list(set(query_terms))

        # Group by node_id
        node_term_maps = {}
        for node_id, term, tf in self._tfidf.get_tf_vectors_for(unique_terms, node_ids):
            if node_id not in node_term_maps:
                node_term_maps[node_id] = {}
            node_term_maps[node_id][term] = tf

        # Compute similarity for each node
        scores = {}
        for node_id in node_ids:
            node_term_map = node_term_maps.get(node_id)
            if not node_term_map:
                scores[node_id] = 0
                continue

            dot_product = 0
            doc_norm_sq = 0

            for term, tf_val in node_term_map.items():
                idf = idf_map.get(term, 1)
                d_val = tf_val * idf
                q_val = (query_vec.get(term, 0) or 0) * idf
                dot_product += q_val * d_val
                doc_norm_sq += d_val * d_val

            doc_norm = math.sqrt(doc_norm_sq)
            scores[node_id] = dot_product / (query_norm * doc_norm) if doc_norm > 0 else 0

        return scores

    def _rebuild_tfidf_index(self):
        """Rebuild TF-IDF index for all existing (non-archived) nodes.

        Builds the TF-IDF text from title + content only (the keywords
        column was dropped 2026-05-24 along with the broken
        auto-extractor).
        """
        # Route the index writes through TfIdfDAL (clear_all + per-node
        # store_tf_vector). Each of those gates its commit on conn.in_batch;
        # we own the envelope here, so flip in_batch for the duration to keep
        # the whole rebuild a SINGLE commit (else N+1 commits — one per node).
        # Save/restore makes it correct even if a caller ever nests this in a
        # wider batch; _maybe_commit() then defers to that parent.
        # (The node fetch stays raw — it's a `nodes` read, migrated in Phase 4.)
        was_batch = self.conn.in_batch
        self.conn.in_batch = True
        try:
            self._tfidf.clear_all()
            cursor = self.conn.execute(
                'SELECT id, title, content FROM nodes WHERE archived = 0')
            for node_id, title, content in cursor.fetchall():
                full_text = ' '.join(filter(None, [title, content]))
                self._tfidf.store_tf_vector(node_id, self._compute_tf(full_text))
        finally:
            self.conn.in_batch = was_batch
        self._maybe_commit()

    def remember(self, type: str, title: str, content: Optional[str] = None,
                 locked: bool = False,
                 emotion: float = 0, emotion_label: str = 'neutral',
                 emotion_source: str = 'auto',
                 # `project` param removed 2026-07-03 — provenance is
                 # system-stamped at the write boundary and rides
                 # **extra_fields into node_metadata_kv (PROMOTED_FIELDS).
                 confidence: float = 1.0,
                 personal: Optional[str] = None,
                 personal_context: Optional[str] = None,
                 critical: bool = False,
                 encoding_source: Optional[str] = None,
                 situation: Optional[str] = None,
                 source_turn_id: Optional[str] = None,
                 evolution_status: Optional[str] = None,
                 # Promoted metadata fields (stored in node_metadata_kv)
                 reasoning: Optional[str] = None,
                 user_raw_quote: Optional[str] = None,
                 anchor_raw_quote: Optional[str] = None,
                 # `correction_of` parameter removed 2026-05-17 — corrections
                 # tracked via correction_improvement-aspect edges
                 # (corrects/supersedes/reframes/...). See render_corrections()
                 # + correction_enrich() for the read path.
                 correction_pattern: Optional[str] = None,
                 source_context: Optional[str] = None,
                 confidence_rationale: Optional[str] = None,
                 alternatives: Optional[List[Dict[str, str]]] = None,
                 change_impacts: Optional[List[Dict[str, str]]] = None,
                 source_attribution: Optional[str] = None,
                 scope: Optional[str] = None,
                 connect_to: Optional[List[Any]] = None,
                 # v29 / Phase B: source_refs anchors the node to trace events.
                 # Sparse by design (1-3 refs typical). Each ref is an 8-char
                 # hex trace_event.id. Persisted via SourceRefDAL.add_source_refs
                 # → node_source_refs join table. Legacy integer ids are
                 # coerced to canonical hex by the DAL.
                 source_refs: Optional[List[str]] = None,
                 ctx=None,
                 **extra_fields) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Accepts ALL contract fields. Core fields go to the nodes table,
        promoted fields go to node_metadata_kv/node_enrichments, and any
        unknown fields are stored as emergent metadata in node_metadata_kv.

        Returns:
            Dict with id, type, title, and related_nodes (top 5 similar existing nodes).
        """
        # Validate personal flag
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            personal = None

        # Constitution: only intentional (anchor) encoding can create locked nodes.
        # All automated sources (encoder, idle, hook) must earn permanence.
        # encoding_source convention: "category:process" e.g. "encoder:sonnet", "idle:redistribution"
        if encoding_source and not encoding_source.startswith('anchor') and locked:
            locked = False

        node_id = self._generate_id(type)
        ts = self.now()

        # `connections` was a store-time edge param, retired 2026-06-18 —
        # `connect_to` fully replaced it. _CONTROL_FIELDS still swallows it so
        # it can't pollute node metadata, but a caller still passing real edge
        # data is a bug we want to SEE, not silently drop. Log it loudly (errors
        # table) — this is the write boundary every remember-path (remember /
        # remember_batch / direct) flows through. Guard on truthiness, not key
        # presence: a stray `connections=None` / `[]` is not an edge-creation
        # attempt, so don't fire a spurious retirement error for it.
        _legacy = extra_fields.get('connections')
        if _legacy:
            _n = len(_legacy) if isinstance(_legacy, (list, tuple)) else '?'
            self._log_error(
                'remember_connections_retired',
                ValueError('remember(connections=...) is retired — use connect_to'),
                'node %s dropped %s legacy connection(s)' % (node_id[:8], _n))

        # ══════════════════════════════════════════════════════════════
        # v6: AUTO-ENRICHMENT — make every node rich by default
        # The brain's data was shallow because rich encoding required
        # extra effort. Now remember() fills in what it can automatically.
        # ══════════════════════════════════════════════════════════════

        # Auto-set confidence by type if caller left it at default
        # TYPE_CONFIDENCE from brain_constants defines how reliable each type tends to be
        if confidence == 1.0:  # default = unset by caller
            confidence = TYPE_CONFIDENCE.get(type, 0.70)

        # v4: Fixed personal nodes are always locked — their whole point is permanence
        if personal == 'fixed':
            locked = True

        # v5: Auto-generate content summary for tiered recall
        content_summary = self._generate_summary(title, content)

        # INSERT into nodes table
        from .brain_constants import CURRENT_ENCODING_VERSION
        self.conn.execute(
            '''INSERT INTO nodes
               (id, type, title, content, content_summary,
                activation, stability, locked, confidence,
                recency_score, emotion, emotion_label, emotion_source,
                personal, personal_context, encoding_version, encoding_source,
                evolution_status, source_turn_id,
                last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1.0, 1.0, ?, ?, 1.0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (node_id, type, title, content, content_summary,
             1 if locked else 0, confidence,
             emotion, emotion_label, emotion_source,
             personal, personal_context, CURRENT_ENCODING_VERSION,
             encoding_source or 'anchor',
             evolution_status, source_turn_id,
             ts, ts, ts)
        )
        self._maybe_commit()

        # Encode-time voice fidelity validation (Phase B+ structural backup).
        # A/B testing of v20 surfaced a speaker-misattribution failure mode:
        # when the conversation has operator-asks-question + anchor-articulates-
        # principle shape, Sonnet's voice-attribution logic matches content to
        # field rather than speaker to field — it can land identical strings in
        # BOTH user_raw_quote and anchor_raw_quote. Prompt teaching alone didn't
        # catch this (failed across v19, v20.0, v20.1 on the same corpus).
        # Loud at write boundary so the error surfaces every time it happens.
        # Non-blocking: write proceeds, but the encoder errors table records the
        # violation for retrospective audit + future S2Healer-driven cleanup.
        if (user_raw_quote and anchor_raw_quote
                and user_raw_quote.strip() == anchor_raw_quote.strip()
                and len(user_raw_quote.strip()) > 0):
            self._log_error(
                'voice_fidelity_identical_strings',
                ValueError(
                    "user_raw_quote == anchor_raw_quote on node %s "
                    "(type=%s, title=%r). Voice fields are for different "
                    "speakers; identical strings indicate Sonnet matched "
                    "content-to-field rather than speaker-to-field. "
                    "Quote: %r" % (
                        node_id, type, (title or '')[:80],
                        user_raw_quote.strip()[:160])),
                'encode-time voice fidelity check')

        # Store all metadata via unified path — promoted, emergent, and extra fields.
        _meta_fields = {}
        # Promoted fields passed as explicit args
        for _name, _val in [
            ('reasoning', reasoning), ('user_raw_quote', user_raw_quote),
            ('anchor_raw_quote', anchor_raw_quote),
            ('correction_pattern', correction_pattern), ('source_context', source_context),
            ('confidence_rationale', confidence_rationale), ('scope', scope),
            ('source_attribution', source_attribution), ('situation', situation),
        ]:
            if _val is not None:
                _meta_fields[_name] = _val
        # Extra fields from callers (community metadata, any emergent fields)
        _meta_fields.update(extra_fields)
        if _meta_fields:
            self._store_node_metadata(node_id, _meta_fields, caller='remember')

        # v5.2: Critical flag requires operator approval — don't set directly
        if critical:
            self._add_pending_critical(node_id, title)

        # v5: Build TF-IDF vector for this node (from title + content)
        try:
            self._store_tfidf_vector(node_id, title, content)
        except Exception as e:
            self._log_error('tfidf_vector_store', e, 'storing TF-IDF vector for node %s' % node_id[:12])

        # v9: Sync FTS5 full-text search index. Keywords column is scheduled
        # for removal in schema v28; for now FTS5 still has the column so we
        # pass empty string. Once v28 lands, Fts5DAL.upsert signature drops it.
        try:
            from .dal import Fts5DAL
            self._fts.upsert(node_id, title, content or '', '')
            self._maybe_commit()
        except Exception as e:
            self._log_error('fts5_sync_remember', e, 'syncing FTS5 for node %s' % node_id[:12])

        # Vector computation handled by the embed_queue worker — this node
        # is marked dirty and will be embedded within ~5s. S2 Heal catches
        # anything that slips through on crash.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
            embedding_queued = True
        except Exception as e:
            self._log_error('embed_enqueue_remember', e, 'enqueue %s' % node_id[:12])
            embedding_queued = False

        # Recall-on-create is gated OFF since embedding moved to the async
        # embed_queue worker: the old gate keyed on synchronous embedding
        # (`embedding_stored`), which no longer happens inline. The recall below
        # uses query text (not the new node's vector) so it COULD run — re-enable
        # only if the per-write recall latency is acceptable. Kept off to preserve
        # behavior after the async-embedding change. (flagged 2026-06-04)
        recall_on_create_enabled = False

        # v29 / Phase B: persist source_refs to node_source_refs join table.
        # SourceRefDAL.add_source_refs coerces legacy int ids → 8-char hex,
        # uses INSERT OR IGNORE (first-write-wins, re-encode is safe),
        # preserves the encoder's write order via `position`. Empty/None
        # input is a no-op. Failures are logged but don't fail the write —
        # invalid refs degrade gracefully at recall (S2Healer cleans
        # dangling refs in a future pass).
        # Edges co_anchored creates are surfaced so the dispatch handler can
        # pop them off the agent-facing payload. They are NOT traced —
        # co_anchored is noise-aspect (ruled 2026-08-04), same coverage rule
        # as emergent_bridge. The graph edge itself is still first-class.
        co_anchored_made = []
        if source_refs:
            try:
                self._source_refs.add_source_refs(node_id, source_refs)
            except Exception as e:
                self._log_error(
                    'source_refs_persist', e,
                    'persisting source_refs for node %s (%d refs)' % (
                        node_id[:12], len(source_refs)))

            # Step 7 / decision 15: co_anchored auto-edge. When this node's
            # source_refs overlap with any existing node's refs, write a
            # structural co_anchored edge to each sibling. The graph layer
            # is the signal — no score boost, no magnitude to guess.
            # Excluded from candidate cosine ranking at brain_recall.py:334
            # alongside co_accessed. Sparse refs (1-3) × small cohort →
            # negligible cost.
            try:
                graph_dal = self._graph
                siblings: set = set()
                for tid in source_refs:
                    if not isinstance(tid, str):
                        continue
                    for sibling_id in self._source_refs.get_nodes_referencing(tid):
                        if sibling_id != node_id:
                            siblings.add(sibling_id)
                for sibling_id in siblings:
                    _r = graph_dal.add_relation(
                        node_id, sibling_id, 'co_anchored',
                        description='shared episodic anchor',
                        encoding_source='dispatch:co_anchored',
                    )
                    co_anchored_made.append({
                        'src_id': node_id, 'target_id': sibling_id,
                        'relation': 'co_anchored',
                        'edge_id': (_r or {}).get('edge_id'),
                        'deltas': (_r or {}).get('deltas', [])})
            except Exception as e:
                self._log_error(
                    'co_anchored_autoedge', e,
                    'co_anchored auto-edge for node %s (%d refs)' % (
                        node_id[:12], len(source_refs)))

        # connect_to: title-resolved typed edges. When called standalone (not
        # from a batch), there are no siblings — only catalog fallback applies.
        # Inside remember_batch / brain_batch, connect_to is popped from the
        # spec BEFORE this call and processed AFTER all siblings are created.
        connect_to_result = None
        if connect_to:
            # Capture the outcome so the caller sees what linked and why an
            # edge didn't — connect_to is fail-soft, but no longer silent to
            # the caller (it used to surface only on the dashboard).
            connect_to_result = self._apply_connect_to(
                node_id, connect_to, sibling_map=None,
                encoding_source=encoding_source or 'anchor')

        # co_accessed-on-remember REMOVED (2026-05-31). It connected a new node
        # to recently-written nodes by temporal write-adjacency — pre-Phase-5
        # noise (no judge selection) — and, as a side effect, MATERIALIZED each
        # pair's physical edge with an arbitrary direction (source = the newer
        # node). Because the model stores one physical direction per pair (v22),
        # any later SEMANTIC edge drawn on that pair (depends_on, supersedes,
        # corrects, ...) inherited that accidental direction via get_edge_id.
        # co_accessed is now created ONLY on recall, from judge-selected surface
        # picks (recall_write_queue Hebbian path), where it carries real
        # co-activation signal. See correction node 2f344177.

        # v11: Emergent bridging at store-time
        bridges = []
        try:
            bridges = self._bridge_at_store_time(node_id)
        except Exception as e:
            self._log_error('bridge_at_store', e, 'emergent bridging for node %s' % node_id[:12])

        # v5: Track encoding for heartbeat + segment tracking.
        # Per-session via the ctx passed in by the caller (dispatch handlers
        # load ctx from args.session_id). Falls back to the deprecated
        # `self.session_id` singleton when no ctx is provided — non-MCP
        # callers (seed_pack, migrations, healer) take this path and don't
        # care about parallel-session attribution.
        _ctx_for_attr = ctx
        if _ctx_for_attr is None and self.session_id:
            try:
                _ctx_for_attr = self.get_or_create_session(self.session_id)
            except Exception:
                _ctx_for_attr = None
        if _ctx_for_attr is not None:
            try:
                self.record_remember(_ctx_for_attr)
                # ctx mutation persists via daemon autosave loop —
                # _ctx_for_attr is the cached instance on brain so the
                # autosave picks it up.
            except Exception as e:
                self._log_error('record_remember', e, 'tracking encoding for heartbeat')
        try:
            _sid = _ctx_for_attr.session_id if _ctx_for_attr else ''
            self.add_to_segment(node_id, _sid)
        except Exception as e:
            self._log_error('add_to_segment', e, 'tracking node %s in conversation segment' % node_id[:12])

        # v6: Generate enrichment prompt for Claude to fill in.
        # The brain recalls neighbors and builds a structured prompt.
        # If enrichments are provided inline (from a previous enrich() call), store them.
        # Otherwise, return the prompt so Claude can fill it in.
        enrichment_prompt = None
        enrichment_stored = 0
        try:
            enrichment_prompt = self._build_enrichment_prompt(node_id, title, content)
        except Exception as e:
            print(f'[brain] V5 enrichment prompt failed for {node_id}: {e}', file=sys.stderr)

        # message_stream escalation REMOVED 2026-04-05 — encoding reads from traces

        # _store_node_metadata removed 2026-04-13 — old table, KV handles this via _store_metadata_kv above.

        # v9: Recall-on-create — return related nodes so caller can connect immediately
        related_nodes = []
        try:
            from .pipeline_contract import ENCODING_AGENT
            if recall_on_create_enabled:
                recall_result = self.recall(query='%s %s' % (title, (content or '')[:ENCODING_AGENT['recall_on_create_query_limit']]), limit=ENCODING_AGENT['recall_on_create_limit'] + 1, source='internal')
                for r in recall_result.get('results', []):
                    if r.get('id') != node_id:
                        related_nodes.append({
                            'id': r.get('id', ''),
                            'type': r.get('type', ''),
                            'title': r.get('title', ''),
                            'content': (r.get('content', '') or '')[:ENCODING_AGENT['recall_on_create_content_limit']],
                            'confidence': r.get('confidence', 0),
                            'score': round(r.get('effective_activation', 0), 3),
                        })
                    if len(related_nodes) >= ENCODING_AGENT['recall_on_create_limit']:
                        break
        except Exception as e:
            self._log_error('remember_recall_on_create', e, 'recall-on-create for %s' % node_id[:8])

        result = {
            'id': node_id,
            'type': type,
            'title': title,
            'embedding_queued': embedding_queued,
            'enrichment_prompt': enrichment_prompt,
            'related_nodes': related_nodes,
        }
        # connect_to_result is present ONLY when connect_to was passed — keeps
        # the response clean for the common no-edge case. `related_nodes`
        # (similar existing nodes) and `connect_to_result` (edges you asked to
        # create) are DIFFERENT things — don't read one as the other.
        if connect_to_result is not None:
            result['connect_to_result'] = connect_to_result
        # co_anchored edges this remember() materialized (src-tagged, with
        # edge_id+deltas) — the dispatch handler pops these off the payload;
        # noise-aspect, never traced.
        if co_anchored_made:
            result['co_anchored_made'] = co_anchored_made
        return result

    # ═══════════════════════════════════════════════════════════════
    # v8: revise() — Encoding IS updating existing knowledge
    # ═══════════════════════════════════════════════════════════════

    def revise(self, node_id: str, content: str = None, reason: str = '',
               updates: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Update fields on an existing node. Per-field replace semantics.

        Three ways to call (all equivalent):
          revise(node_id, content="new text", reason="why")
          revise(node_id, updates={"confidence": 0.9}, reason="why")
          revise(node_id, situation="When debugging", reason="adding situation")

        Behavior contract:
        - Immutable fields ({id, created_at, locked}) are skipped with a
          warning. Other fields in the same call still process; the skipped
          field surfaces in the result dict's `warnings` list.
        - Specified fields are REPLACED with the passed value.
        - Unspecified fields are PRESERVED (only the keys you pass are touched).
        - Returns deltas in the result dict — caller (typically daemon_dispatch)
          emits a trace event with these deltas as the canonical revision
          history. There is no per-node history blob; query traces instead.

        After any revision: re-embeds, re-indexes TF-IDF, updates timestamps.
        """
        # Merge updates from all sources
        all_updates = dict(updates or {})
        all_updates.update(kwargs)
        if content:
            all_updates['content'] = content

        if not all_updates:
            return {'error': 'No updates provided', 'node_id': node_id}

        # Capture the FULL field set NOW for vector invalidation. `all_updates`
        # gets mutated below (content is popped, etc.), so we need the
        # original set or the invalidation step misses fields.
        fields_changed_for_invalidation = set(all_updates.keys())

        # v29 / Phase B: source_refs is a join-table field, not a node column.
        # Pop it before the field classification so it doesn't land in
        # node_metadata_kv as an extra field. Persist via replace_source_refs
        # AFTER the existence check (node_id must exist for the FK).
        # Per the unified 2-class revise contract (decision 995ffeb1) and
        # EPISODIC-REFERENCES.md §6.2: when source_refs is PRESENT in the
        # revise payload, REPLACE the entire list. When ABSENT, preserve.
        # Use a sentinel to distinguish "key absent" from "explicit empty list".
        _SR_ABSENT = object()
        new_source_refs = all_updates.pop('source_refs', _SR_ABSENT)

        # Fetch existing node
        row = self.conn.execute(
            'SELECT id, type, title, content, archived FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'error': 'Node not found', 'node_id': node_id}
        if row[4] == 1:
            return {'error': 'Cannot revise archived node', 'node_id': node_id}

        existing_id, node_type, title, old_content, _ = row
        old_content = old_content or ''
        ts = self.now()

        # ── Field classification ──
        # Top-level fields live on the nodes table (updatable via SQL).
        # Immutable fields are silently skipped with a warning.
        # `project` removed 2026-07-03 — provenance lives in node_metadata_kv
        # (PROMOTED_FIELDS) and revise strips it at the write boundary anyway.
        NODES_TABLE_FIELDS = {
            'title', 'type', 'confidence', 'emotion',
            'emotion_label', 'personal', 'personal_context',
            'critical', 'evolution_status', 'encoding_source',
            'archived',  # allows revise(archived=True) for consolidation
        }
        IMMUTABLE = {'id', 'created_at', 'locked'}

        # ── Filter skipped fields (immutable, locked-archive) ──
        # Skipped fields don't write to nodes/KV/SQL and don't appear in
        # deltas. They surface in the return dict's `warnings` list so
        # callers can detect partial-success without parsing logs.
        # Any field that is neither a nodes column nor immutable falls through
        # to node_metadata_kv as an extra field (the generic extra-fields path,
        # same as remember()). A legacy/unknown field (e.g. the v28-dropped
        # `keywords`) is therefore stored as KV, not skipped — no longer a crash
        # (keywords is out of NODES_TABLE_FIELDS, so no SELECT/UPDATE of the gone
        # column) and intentionally not special-cased.
        skipped_fields = []  # list of (field, reason)
        writable = {}
        for field, value in all_updates.items():
            if field in IMMUTABLE:
                self._log_error('revise_immutable',
                                ValueError('Cannot revise immutable field: %s' % field),
                                'node %s attempted to revise %s' % (node_id[:8], field))
                skipped_fields.append((field, 'immutable'))
                continue
            if field == 'archived' and value:
                lock_row = self.conn.execute(
                    'SELECT locked, critical FROM nodes WHERE id = ?',
                    (node_id,)).fetchone()
                if lock_row and (lock_row[0] or lock_row[1]):
                    self._log_error('revise_archive_locked',
                                    ValueError('Cannot archive locked/critical node'),
                                    'node %s' % node_id[:8])
                    skipped_fields.append((field, 'locked_or_critical'))
                    continue
            writable[field] = value

        # ── Capture old values for delta computation (before any write) ──
        # Used by callers (typically dispatch) to emit trace events with
        # field-level history. Replaces the old _sys_revision_history blob.
        old_values = {}
        if 'content' in writable:
            old_values['content'] = old_content

        top_level_to_capture = [k for k in writable if k in NODES_TABLE_FIELDS]
        if top_level_to_capture:
            cols = ', '.join(top_level_to_capture)
            old_row = self.conn.execute(
                'SELECT %s FROM nodes WHERE id = ?' % cols, (node_id,)
            ).fetchone()
            if old_row:
                for i, k in enumerate(top_level_to_capture):
                    old_values[k] = old_row[i]

        kv_to_capture = [
            k for k in writable
            if k not in NODES_TABLE_FIELDS and k != 'content'
        ]
        if kv_to_capture:
            kv_old = self._meta_kv.get_fields(node_id, kv_to_capture)
            for k in kv_to_capture:
                old_values[k] = kv_old.get(k)  # None if not previously set

        # Content: replace with new value (history lives in trace deltas now,
        # not the legacy _sys_revision_history KV blob).
        new_content = old_content
        if 'content' in writable:
            new_content = writable.pop('content')

        # Build SQL UPDATE for all fields.
        # Always update: content, content_summary, updated_at, revised_at.
        set_parts = ['content = ?', 'content_summary = ?', 'updated_at = ?', 'revised_at = ?']
        params = [new_content, self._generate_summary(title, new_content), ts, ts]

        for field, value in writable.items():
            if field in NODES_TABLE_FIELDS:
                set_parts.append('%s = ?' % field)
                params.append(value)
                if field == 'title':
                    title = value  # track for re-embed

        params.append(node_id)
        self.conn.execute(
            'UPDATE nodes SET %s WHERE id = ?' % ', '.join(set_parts), params)
        self._maybe_commit()

        # Store metadata via unified path — handles promoted, emergent, situation.
        # Only writable (non-skipped) fields get persisted.
        if writable:
            self._store_node_metadata(node_id, writable, caller='revise')

        # Vector invalidation: when a source field changes, the corresponding
        # embedding vector becomes stale. Delete the affected rows so the
        # embed_queue's backfill scan re-embeds from the updated text.
        # WITHOUT this, VectorDAL.find_missing() skips the row (it exists)
        # and the vector keeps encoding outdated text indefinitely. Title
        # changes invalidate the title slot too — collected via SQL UPDATE
        # above and added to the field set here.
        #
        # Failure here is a CORRECTNESS issue (recall serves stale embeddings
        # until next backfill cycle). We log loudly AND surface the failure
        # in the return dict so callers can detect partial-success — silent
        # swallow would hide drift indefinitely.
        vector_invalidation_failed = False
        try:
            from .pipeline_contract import vectors_affected_by
            invalidated_vectors = set()
            for field in fields_changed_for_invalidation:
                invalidated_vectors |= vectors_affected_by(field)
            if invalidated_vectors:
                # ONE call: typed DB delete + exactly-mirrored cache drop
                # (CachedVectorDAL) or DB-only (plain VectorDAL) — same
                # signature, structural parity. The old two-call shape here
                # (raw SQL + separate whole-node cache drop) was the
                # 2026-07-17 healer-invisibility bug.
                self._vec_dal.delete_for_node(
                    node_id, vector_types=invalidated_vectors)
        except Exception as e:
            vector_invalidation_failed = True
            self._log_error('revise_vector_invalidate', e,
                            'invalidating vectors for %s — STALE EMBEDDINGS '
                            'will be served by recall until next backfill '
                            'cycle catches up' % node_id[:8])

        # Vector (re)computation handled by the embed_queue worker — revisions
        # mark the node dirty so stale text→vector pairs get refreshed within ~5s.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
        except Exception as e:
            self._log_error('embed_enqueue_revise', e, 'enqueue %s' % node_id[:12])

        # Re-index TF-IDF from title + new_content (keywords column dropped
        # 2026-05-24 along with the broken auto-extractor)
        try:
            self._store_tfidf_vector(node_id, title, new_content)
        except Exception as e:
            self._log_error("revise_tfidf", e, "Failed to re-index TF-IDF for %s" % node_id[:8])

        # v9: Re-sync FTS5 full-text search index (keywords scheduled for
        # removal in schema v28; empty string until then)
        try:
            from .dal import Fts5DAL
            self._fts.upsert(node_id, title, new_content, '')
            self._maybe_commit()
        except Exception as e:
            self._log_error("fts5_sync_revise", e, "syncing FTS5 for %s" % node_id[:8])

        # pending_consolidation table dropped 2026-04-05

        # message_stream escalation REMOVED 2026-04-05

        # ── VERIFICATION: read-back to confirm writes landed ──
        verification_failures = []

        # Verify nodes table fields
        readback = self._nodes.get_naked_node(node_id)
        if readback:
            for field in list(writable.keys()):
                if field in readback:
                    expected = writable[field]
                    actual = readback.get(field)
                    if actual != expected and str(actual) != str(expected):
                        verification_failures.append(field)
            # Content was popped from `writable` earlier; verify separately
            # (REPLACE semantic — readback must equal new_content exactly).
            # Use old_values to detect that content was actually a write target —
            # `content` named-arg path AND `updates={'content': ...}` path both
            # populate old_values['content'], so this catches either.
            if 'content' in old_values:
                actual_content = readback.get('content') or ''
                if actual_content != new_content:
                    verification_failures.append('content')

        # Situation embedding deferred to backfill — no inline verification needed

        # Vector invalidation failure surfaces here too — same severity as a
        # missed field write, since recall correctness depends on it.
        if vector_invalidation_failed:
            verification_failures.append('vector_invalidation')

        verified = len(verification_failures) == 0

        # ── Build deltas for trace event emission ──
        # Caller (typically daemon_dispatch) uses these to write a single
        # node_revised trace event capturing what changed in this call.
        deltas = []
        for field, new_val in writable.items():
            old_val = old_values.get(field)
            if old_val != new_val:
                deltas.append({
                    'field': field,
                    'old': old_val,
                    'new': new_val,
                })
        # Content was popped from `writable` earlier; check separately.
        if 'content' in old_values and old_values['content'] != new_content:
            deltas.append({
                'field': 'content',
                'old': old_values['content'],
                'new': new_content,
            })

        # Warnings surface skipped fields without requiring log parsing.
        warnings = []
        for field, _reason in skipped_fields:
            if _reason == 'immutable':
                warnings.append('immutable field skipped: %s' % field)
            elif _reason == 'locked_or_critical':
                warnings.append('archive blocked (locked/critical): %s' % field)

        # fields_updated: what was actually written. Excludes skipped fields.
        # Includes 'content' if it was passed (popped from writable earlier).
        fields_updated = list(writable.keys())
        if 'content' in old_values:
            fields_updated.append('content')

        # v29 / Phase B: persist source_refs with field-level REPLACE semantics
        # (decision 995ffeb1 — unified revise contract; §6.2). When the key was
        # absent in the payload, preserve existing refs (no-op). When present —
        # even as an empty list — replace the entire list.
        source_refs_replaced = None  # None=untouched, int=count after replace
        if new_source_refs is not _SR_ABSENT:
            try:
                source_refs_replaced = self._source_refs.replace_source_refs(
                    node_id, new_source_refs or [])
                fields_updated.append('source_refs')
                deltas.append({
                    'field': 'source_refs',
                    'op': 'replace',
                    'count_after': source_refs_replaced,
                })
            except Exception as e:
                self._log_error(
                    'source_refs_persist_revise', e,
                    'replacing source_refs on revise %s' % node_id[:12])

            # Step 7: refresh co_anchored cohort after REPLACE. New refs may
            # bring new siblings; old siblings whose shared refs are now gone
            # become stale edges that S2Healer archives (§10.6 Responsibility 2).
            # add_relation is idempotent — existing edges are field-preserving
            # no-ops.
            try:
                graph_dal = self._graph
                siblings: set = set()
                for tid in (new_source_refs or []):
                    if not isinstance(tid, str):
                        continue
                    for sibling_id in self._source_refs.get_nodes_referencing(tid):
                        if sibling_id != node_id:
                            siblings.add(sibling_id)
                for sibling_id in siblings:
                    graph_dal.add_relation(
                        node_id, sibling_id, 'co_anchored',
                        description='shared episodic anchor',
                        encoding_source='dispatch:co_anchored',
                    )
            except Exception as e:
                self._log_error(
                    'co_anchored_autoedge_revise', e,
                    'co_anchored auto-edge on revise %s' % node_id[:12])

        return {
            'id': node_id,
            'type': writable.get('type', node_type),
            'title': title,
            'revised_at': ts,
            'content_length': len(new_content),
            'fields_updated': fields_updated,
            'deltas': deltas,
            'warnings': warnings,
            'verified': verified,
            'verification_failures': verification_failures if not verified else [],
            'pending_resolved': 0,
            'source_refs_replaced': source_refs_replaced,
        }

    # ═══════════════════════════════════════════════════════════════
    # v5.2: Critical node approval flow
    # Critical nodes get force-surfaced at boot and boosted in recall.
    # Setting critical=1 requires explicit operator approval.
    # ═══════════════════════════════════════════════════════════════

    def _add_pending_critical(self, node_id: str, title: str):
        """Add a node to the pending critical approvals list."""
        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            pending = _json.loads(pending_json) if pending_json else []
            pending.append({
                'node_id': node_id,
                'title': title,
                'requested_at': self.now()
            })
            self.set_config('pending_critical_approvals', _json.dumps(pending))
        except Exception as e:
            self._log_error('_add_pending_critical', e, 'adding pending critical approval')

    def mark_critical(self, node_id: str, reason: str = '') -> Dict[str, Any]:
        """Propose a node as critical. Does NOT set the flag — requires operator approval via revise().

        Args:
            node_id: The node to mark as critical
            reason: Why this node is critical (for the operator to review)

        Returns:
            Dict with node_id, status='pending', reason
        """
        # Verify node exists (title is reused in the pending entry below)
        node_title = self._nodes.get_title(node_id)
        if node_title is None:
            return {'error': f'Node {node_id} not found'}

        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            pending = _json.loads(pending_json) if pending_json else []
            # Don't duplicate
            existing_ids = {p['node_id'] for p in pending}
            if node_id not in existing_ids:
                pending.append({
                    'node_id': node_id,
                    'title': node_title,
                    'reason': reason,
                    'requested_at': self.now()
                })
                self.set_config('pending_critical_approvals', _json.dumps(pending))
        except Exception as e:
            self._log_error('mark_critical', e, 'adding pending critical approval')
            return {'error': str(e)}

        return {'node_id': node_id, 'status': 'pending', 'reason': reason}

    # approve_critical removed 2026-04-13 — never wired to MCP, direct DB write.

    def get_pending_critical(self) -> List[Dict[str, Any]]:
        """Get all pending critical approval requests."""
        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            return _json.loads(pending_json) if pending_json else []
        except Exception as e:
            self._log_error('get_pending_critical', e, 'parsing pending critical approvals JSON')
            return []

    def backfill_summaries(self, batch_size: int = 50) -> Dict[str, Any]:
        """Generate content_summary for existing nodes that lack one. Run during idle."""
        cur = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE content IS NOT NULL AND content != '' AND content_summary IS NULL LIMIT ?",
            (batch_size,)
        )
        rows = cur.fetchall()
        count = 0
        for node_id, title, content in rows:
            summary = self._generate_summary(title, content)
            if summary:
                self.conn.execute(
                    "UPDATE nodes SET content_summary = ? WHERE id = ?",
                    (summary, node_id)
                )
                count += 1
        if count:
            self._maybe_commit()
        return {'backfilled': count, 'remaining': len(rows) - count}


    # _store_node_metadata removed 2026-04-13 — old node_metadata table dropped.

    def remember_rich(self, type: str, title: str, content: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Backward-compatible wrapper — remember() now handles all fields directly."""
        return self.remember(type=type, title=title, content=content, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    # connect_to resolution — sibling-aware, sequencing-agnostic
    # ═══════════════════════════════════════════════════════════════

    def _resolve_connect_to_entry(self, entry, sibling_map=None,
                                  exclude_self=None, exclude_ids=None):
        """Resolve a connect_to entry to (target_id, relation_pairs, reason).

        sibling_map: {lowercased_title: node_id} for nodes created in the
                     same batch. Sibling matching is CASE-INSENSITIVE —
                     keys are lowered when the map is built (see
                     remember_batch sibling_map construction) AND lowered
                     again at lookup. Sibling exact-match (case-insensitive)
                     wins over the catalog-title pass — NEW wins on title
                     collision. If you mean an existing catalog node, use
                     revise on its id, not duplicate-title remember.
                     (Catalog matching is _match_catalog_title — token-exact
                     or bounded near-title, deterministic, no vectors.)
        exclude_self: source node_id; resolution to this id is treated as a
                      self-reference and rejected.

        Returns (target_id, [(relation, description), ...], None) on success,
        or (None, [], reason) on any failure — `reason` is a short caller-facing
        string so the failure can surface in the write response, not just the
        dashboard. All failures also log loudly via _log_error — no silent skips.
        """
        # Parse entry shape (string or dict)
        if isinstance(entry, str):
            title_query = entry
            relation_pairs = [('related', '')]
        elif isinstance(entry, dict):
            title_query = entry.get('title', '')
            if not title_query:
                reason = "connect_to entry missing 'title' field"
                self._log_error(
                    'connect_to_invalid', ValueError(reason),
                    'entry=%s' % str(entry)[:200])
                return None, [], reason
            if not isinstance(title_query, str):
                # A non-empty non-string title (int, list, ...) clears the
                # falsy guard above but would crash the .strip()/.lower()/regex
                # calls downstream — that AttributeError escapes _apply_connect_to
                # and rolls back the whole batch. Reject loudly instead.
                reason = ("connect_to entry 'title' must be a string, got %s"
                          % type(title_query).__name__)
                self._log_error(
                    'connect_to_invalid', TypeError(reason),
                    'entry=%s' % str(entry)[:200])
                return None, [], reason
            if isinstance(entry.get('relations'), list):
                relation_pairs = []
                for r in entry['relations']:
                    if not isinstance(r, dict):
                        continue
                    rel = r.get('relation', 'related')
                    desc = r.get('why', r.get('description', ''))
                    relation_pairs.append((rel, desc))
                if not relation_pairs:
                    reason = "connect_to relations array is empty or malformed"
                    self._log_error(
                        'connect_to_invalid', ValueError(reason),
                        'entry=%s' % str(entry)[:200])
                    return None, [], reason
            else:
                rel = entry.get('relation', 'related')
                desc = entry.get('why', entry.get('description', ''))
                relation_pairs = [(rel, desc)]
        else:
            reason = ("connect_to entry must be str or dict, got %s"
                      % type(entry).__name__)
            self._log_error(
                'connect_to_invalid', TypeError(reason),
                'entry=%s' % str(entry)[:200])
            return None, [], reason

        target_id = None

        # Pass 0: ID-shape pre-check. The encoder sometimes passes an 8+ char
        # hex ID in the `title` field when it really means "connect to this
        # specific known node by id" — e.g. when an ID was visible in the
        # conversation (recalled context, prior tool result, surfaced trace).
        # Without this check, sibling-map and fuzzy-title both miss because
        # neither matches an opaque hash. Resolve via id-prefix lookup; if
        # found, prefer it over the title-based passes. Log a soft warning
        # so we can see how often the encoder does this (signal for prompt
        # tuning, not a hard error).
        import re as _re
        if _re.fullmatch(r'[0-9a-fA-F]{8,}', title_query.strip()):
            try:
                row = self.conn.execute(
                    'SELECT id FROM nodes WHERE id LIKE ? LIMIT 2',
                    (title_query.strip().lower() + '%',)).fetchall()
                if len(row) == 1:
                    target_id = row[0][0]
                elif len(row) > 1:
                    # Ambiguous prefix — log and fall through to title path.
                    self._log_error(
                        'connect_to_id_prefix_ambiguous',
                        ValueError(
                            "connect_to title looked like an id but matched "
                            "multiple nodes; falling back to title search"),
                        'prefix=%s matches=%d' % (title_query[:16], len(row)))
            except Exception as e:
                self._log_error(
                    'connect_to_id_lookup_failed', e,
                    'id-prefix lookup for %r' % title_query[:80])
                # fall through to the title path

        # Pass 1: sibling map (NEW wins on title collision)
        if not target_id and sibling_map:
            target_id = sibling_map.get(title_query.lower())

        # Pass 2: catalog title — deterministic token matching only (exact =
        # distance 0 at any length; near = distance ≤ NEAR_TITLE_MAX_OPS behind
        # floor/uniqueness/margin gates). Vectors are deliberately absent from
        # the write path (decision 2026-07-30): a wrong edge outlives a missing
        # one. find_node_by_title (interactive fuzzy search) is NOT called here.
        ambiguous_reason = None
        if not target_id:
            try:
                target_id, ambiguous_reason = self._match_catalog_title(
                    title_query, exclude_ids=exclude_ids)
            except Exception as e:
                reason = "title lookup failed: %s" % str(e)[:120]
                self._log_error(
                    'connect_to_failed', e,
                    'catalog title match for %r' % title_query[:80])
                return None, [], reason

        # Self-reference guard
        if target_id and exclude_self and target_id == exclude_self:
            reason = "connect_to would create self-edge"
            self._log_error(
                'connect_to_self', ValueError(reason),
                'node=%s title=%s' % (exclude_self[:8], title_query[:80]))
            return None, [], reason

        # Unresolved — neither sibling nor catalog matched
        if not target_id:
            reason = ambiguous_reason or (
                "title %r resolved to no node (pass the node id, the "
                "sibling's exact title, or the exact catalog title)"
                % title_query[:80])
            self._log_error(
                'connect_to_unresolved',
                ValueError("connect_to title resolved to nothing"),
                'title=%s%s' % (title_query[:80],
                                ' | %s' % ambiguous_reason if ambiguous_reason
                                else ''))
            return None, [], reason

        return target_id, relation_pairs, None

    def _title_candidate_rows(self, tokens, limit=500):
        """The ONE lexical candidate door for title→node matching: probe the
        FTS5 index TITLE-SCOPED (`title:"tok"*` per probe, OR'd, prefix-
        tolerant) and hydrate (id, title) for the live matches. Shared by
        _match_catalog_title (write path) and find_node_by_title's lexical
        pass (interactive) — neither scans the nodes table.

        The column scope is load-bearing for the pigeonhole recall guarantee:
        an unscoped MATCH ranks title-hits against every node that merely
        MENTIONS a probe token in content, and bm25 ORDER BY + LIMIT then
        makes the pool a relevance cut that can rank a qualifying title out
        (bug 69c2cbab #1). Title-only matching makes the pool exactly "titles
        carrying a probe token" — but recall is still only guaranteed when
        that pool fits `limit`, so saturation is SURFACED, never assumed away.

        Returns (rows, saturated): rows are live (id, title) pairs;
        saturated=True means the FTS pool hit `limit` and recall can no
        longer be assumed — the write path must refuse rather than match.
        Empty rows when FTS has nothing (or is unavailable — Fts5DAL.search
        already logs that loud)."""
        ids = self._fts.search(' '.join(tokens), limit=limit, prefix=True,
                               column='title', min_token_len=1)
        if not ids:
            return [], False
        saturated = len(ids) >= limit
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id, title FROM nodes WHERE archived = 0 '
            'AND id IN (%s)' % ph, ids).fetchall()
        return rows, saturated

    def _match_catalog_title(self, title_query, exclude_ids=None):
        """Deterministic catalog-title match for connect_to (write path).

        Token-sequence Levenshtein against live titles:
          • distance 0 — exact normalized match, accepted at any length
            (a verbatim copy is unambiguous intent);
          • distance 1..NEAR_TITLE_MAX_OPS — the bounded-tolerance rung:
            requires ≥ NEAR_TITLE_MIN_TOKENS distinct query tokens, a UNIQUE
            best candidate, and the runner-up ≥ NEAR_TITLE_MARGIN ops further
            out. Acceptance logs loud (connect_to_near_title) with both
            strings — tolerance is visible, never silent.
        Ambiguity (two candidates tied at best, duplicates at distance 0, or
        a photo-finish inside the margin) REFUSES rather than picks.

        Candidate generation rides the FTS5 title index (_title_candidate_rows
        — the ONE indexed door every lexical title consumer shares): a title
        within K ops of the query misses at most K query tokens, so probing
        the K+1 longest tokens (pigeonhole) keeps recall while the index keeps
        it off a table scan. FTS rows are written synchronously in remember()
        and revise(), so just-created and just-renamed nodes are immediately
        matchable — none of the async-vector blindness the old fuzzy path had.

        Returns (target_id, None) on acceptance, (None, reason) on an
        ambiguity refusal (caller surfaces the reason), (None, None) when
        nothing is within the bar.

        exclude_ids: node ids removed from candidacy entirely (batch-level
        connect_to passes its own creations — a sibling sharing the target's
        title must not tie the true catalog node into an ambiguity refusal).
        """
        from .contract import (NEAR_TITLE_MAX_OPS, NEAR_TITLE_MIN_TOKENS,
                               NEAR_TITLE_MARGIN)
        q = _title_tokens(title_query)
        if not q:
            return None, None
        # Margin needs to see runner-up distances up to best+MARGIN, so the
        # DP cap extends past MAX_OPS; beyond the cap everything is "too far".
        cap = NEAR_TITLE_MAX_OPS + NEAR_TITLE_MARGIN - 1
        # Pigeonhole must cover the CAP, not just the acceptance bar: the
        # margin gate reasons about runners out to distance `cap`, and a
        # runner missing every probe is invisible — which silently turns a
        # photo-finish refusal into an acceptance (bug 69c2cbab #2). cap+1
        # probes ⇒ any title within `cap` ops shares at least one.
        probes = sorted(set(q), key=lambda t: (-len(t), t))
        probes = probes[:cap + 1]
        rows, saturated = self._title_candidate_rows(probes)
        if saturated:
            # The FTS pool hit its limit: candidates may have been cut, so
            # neither uniqueness nor margin can be trusted. Refuse loudly —
            # same posture as ambiguity (never guess at a write boundary).
            reason = ("title candidate pool saturated — recall guarantee "
                      "cannot hold; pass the node id")
            self._log_warning(
                'connect_to_pool_saturated',
                'probe pool hit limit for query %r — refusing'
                % title_query[:90])
            return None, reason
        scored = []
        skip = exclude_ids or ()
        for nid, title in rows:
            if nid in skip:
                continue
            d = _token_edit_distance(q, _title_tokens(title), cap)
            if d <= cap:
                scored.append((d, nid, title))
        within = [s for s in scored if s[0] <= NEAR_TITLE_MAX_OPS]
        if not within:
            return None, None
        within.sort(key=lambda s: s[0])
        best_d, best_id, best_title = within[0]
        tied = [s for s in within if s[0] == best_d]
        if len(tied) > 1:
            reason = ("title matches %d nodes at distance %d (%s) — refusing "
                      "ambiguous match; pass the node id"
                      % (len(tied), best_d,
                         ', '.join(s[1][:8] for s in tied[:4])))
            return None, reason
        if best_d == 0:
            return best_id, None
        # Near acceptance gates: length floor + margin to the runner-up.
        if len(set(q)) < NEAR_TITLE_MIN_TOKENS:
            return None, None
        runner_d = min((s[0] for s in scored if s[1] != best_id),
                       default=cap + 1)
        if runner_d - best_d < NEAR_TITLE_MARGIN:
            reason = ("near-title photo-finish (best=%d, runner-up=%d) — "
                      "refusing; pass the node id" % (best_d, runner_d))
            return None, reason
        self._log_warning(
            'connect_to_near_title',
            'near-title accepted at distance %d: query=%r matched=%r (%s)'
            % (best_d, title_query[:90], best_title[:90], best_id[:8]))
        return best_id, None

    def _apply_connect_to(self, src_id, connect_to_spec, sibling_map=None,
                          encoding_source=None, exclude_ids=None):
        """Resolve and create edges for each connect_to entry from src_id.

        Each entry is independent — failures on one don't affect others.
        All failures log loudly; the function never raises and never blocks
        the surrounding write path.

        exclude_ids: batch-created node ids, removed from CATALOG candidacy
        in the title match (sibling resolution stays Pass 1, exact-only, by
        design). Without this, a just-created sibling sitting 1-2 ops from a
        real catalog target ties it into an ambiguity refusal — or wins the
        edge outright (bug 69c2cbab #3). Batch callers pass their created
        set; the single-node remember() path has no siblings to exclude.

        encoding_source: provenance for the edges minted here — the creating
        node's resolved source (e.g. 'anchor' for a direct-MCP remember,
        'encoder:sonnet' for the Scribe). These edges hang off a node created
        in the same write, so they share its provenance. Passed through to
        connect_typed; None means preserve-existing (the legacy behavior, which
        on a fresh edge fell to the '' default — the bug this closes). The MCP
        proxy can't reach these edges (they're minted inside remember()), so the
        source is threaded here from the caller rather than stamped at the proxy.

        Returns a dict:
            {'created': [{'src_id', 'target_id', 'relation', 'edge_id', 'deltas'}, ...],
             'failed':  [{'title', 'reason'}, ...]}
        Each `created` entry is self-contained (src_id tagged here, so callers
        don't re-derive it) and carries the edge_id + add_relation deltas so the
        caller can emit a directional `edge_relation_revised` trace (connect_to
        edges used to be the one edge path that emitted nothing). Every failure
        carries a caller-facing `reason` so the write response can show WHY an
        edge didn't form — not just a count, and not only on the dashboard.
        Callers derive counts via len(). The encoder still gets its
        "tried N, failed M" signal from these lengths.

        Input flexibility: a JSON-string that parses to a list is accepted
        (callers sometimes stringify the array) — the coercion is visible via
        a `failed` entry only when the string does NOT parse to a list.
        """
        created = []
        failed = []
        if not connect_to_spec:
            return {'created': created, 'failed': failed}

        # Lenient coercion: accept a JSON-stringified list. Postel's law —
        # absorb the common "I stringified the array" slip, but surface it if
        # the string isn't actually a list.
        if isinstance(connect_to_spec, str):
            try:
                import json as _json
                parsed = _json.loads(connect_to_spec)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                connect_to_spec = parsed
            else:
                reason = ("connect_to must be a list (or a JSON string that "
                          "parses to one); got an unparseable str")
                self._log_error('connect_to_invalid', TypeError(reason),
                                'src=%s' % src_id[:8])
                failed.append({'title': str(connect_to_spec)[:80], 'reason': reason})
                return {'created': created, 'failed': failed}

        if not isinstance(connect_to_spec, list):
            reason = ("connect_to must be a list, got %s"
                      % type(connect_to_spec).__name__)
            self._log_error('connect_to_invalid', TypeError(reason),
                            'src=%s' % src_id[:8])
            failed.append({'title': str(connect_to_spec)[:80], 'reason': reason})
            return {'created': created, 'failed': failed}

        for entry in connect_to_spec:
            title_query = entry.get('title', entry) if isinstance(entry, dict) else entry
            target_id, relation_pairs, reason = self._resolve_connect_to_entry(
                entry, sibling_map=sibling_map, exclude_self=src_id,
                exclude_ids=exclude_ids)
            if target_id is None:
                # Resolution failed — already logged via _log_error. Surface
                # the reason so the caller can self-correct in the moment.
                failed.append({'title': str(title_query)[:80], 'reason': reason})
                continue
            for rel, desc in relation_pairs:
                try:
                    edge_res = self.connect_typed(src_id, target_id, relation=rel,
                                       weight=0.6, description=desc,
                                       encoding_source=encoding_source)
                    created.append({'src_id': src_id, 'target_id': target_id,
                                    'relation': rel,
                                    'edge_id': (edge_res or {}).get('edge_id'),
                                    'deltas': (edge_res or {}).get('deltas', [])})
                except Exception as e:
                    reason = "connect_typed failed: %s" % str(e)[:120]
                    failed.append({'title': str(title_query)[:80], 'reason': reason})
                    self._log_error(
                        'connect_to_failed', e,
                        'src=%s target=%s rel=%s' % (
                            src_id[:8], target_id[:8], rel))
        return {'created': created, 'failed': failed}

    def remember_batch(self, nodes: List[Dict],
                        connect_to: Optional[List[str]] = None,
                        ctx=None) -> Dict[str, Any]:
        """Create multiple nodes in one call. Each node uses the same contract as remember().

        Args:
            nodes: List of dicts, each with the same fields remember() accepts
                   (type, title, content, keywords, situation, reasoning, etc.)
            connect_to: List of catalog targets (node id or exact/near title —
                        deterministic ladder, no vectors) to connect all new
                        nodes to

        Returns:
            {nodes_created, results: [{id, title, related_nodes}], connections_created}

        Note: pairwise auto-connect of new nodes in the batch was removed
        (2026-05-24). It used to write `related_to` edges with empty
        descriptions between every pair of created nodes — semantic noise
        that confused recall and accumulated as `related_to`-with-no-why
        pollution. If callers want intra-batch edges, pass them
        explicitly via per-node `connect_to`. Episodically-bonded nodes
        (same encoding event) now share that lineage via source_refs +
        the future `co_anchored` edge (decision 15 in
        EPISODIC-REFERENCES.md), not a synthetic relation.
        """
        results = []
        created_ids = []
        connections_created = 0

        # Pass 1: create all nodes. Pop per-node connect_to BEFORE calling
        # remember() so it doesn't fire there with an empty sibling_map —
        # we'll process them all together once siblings exist (Pass 2).
        # Build sibling_map: lowercased title → node_id for sequencing-
        # agnostic resolution (B can connect_to A even if declared first).
        sibling_map = {}
        deferred_connects = []  # [(node_id, ct_spec)]
        # node_id → the node's resolved encoding_source, so connect_to edges
        # (resolved in pass 2, below) inherit the provenance of the node they
        # hang off — 'anchor' for direct-MCP, 'encoder:sonnet' for the Scribe.
        node_sources = {}
        for spec in nodes:
            if isinstance(spec, dict):
                ct_spec = spec.pop('connect_to', None)
            else:
                ct_spec = None
            result = self.remember(**spec, ctx=ctx)
            results.append(result)
            if result.get('id'):
                created_ids.append(result['id'])
                node_sources[result['id']] = spec.get('encoding_source') or 'anchor'
                title = (spec.get('title') or '').lower()
                if title:
                    sibling_map[title] = result['id']
                if ct_spec:
                    deferred_connects.append((result['id'], ct_spec))

        # Pass 2: resolve per-node connect_to with full sibling_map populated.
        # Collect each made edge (with its src_id) so the dispatch handler can
        # emit a directional edge_relation_revised trace per edge.
        connect_to_failed = []  # [{title, reason}] across all nodes
        connect_to_made = []    # [{src_id, target_id, relation, edge_id, deltas}]
        created_set = set(created_ids)
        for src_id, ct_spec in deferred_connects:
            r = self._apply_connect_to(
                src_id, ct_spec, sibling_map=sibling_map,
                encoding_source=node_sources.get(src_id, 'anchor'),
                exclude_ids=created_set)
            connections_created += len(r['created'])
            connect_to_made.extend(r['created'])  # entries are src-tagged by _apply_connect_to
            connect_to_failed.extend(r['failed'])

        # Batch-level connect_to: same edge from EVERY created node to one
        # catalog target. Resolution rides the SAME deterministic ladder as
        # per-node entries (_resolve_connect_to_entry: id → catalog exact/near;
        # sibling_map deliberately absent — siblings are excluded here by
        # contract). Failures are loud and reported, never silently skipped
        # (the pre-2026-07-30 fuzzy version continued without a word).
        if connect_to:
            for entry in connect_to:
                target_id, relation_pairs, fail_reason = (
                    self._resolve_connect_to_entry(entry, sibling_map=None,
                                                   exclude_ids=created_set))
                if not target_id:
                    connect_to_failed.append(
                        {'title': entry.get('title', '') if isinstance(entry, dict)
                                  else str(entry)[:80],
                         'reason': fail_reason})
                    continue
                if target_id in created_set:
                    continue  # sibling — batch-level targets catalog only

                for node_id in created_ids:
                    for rel, desc in relation_pairs:
                        try:
                            edge_res = self.connect_typed(node_id, target_id, relation=rel,
                                              weight=0.6, description=desc,
                                              encoding_source=node_sources.get(node_id, 'anchor'))
                            connections_created += 1
                            connect_to_made.append({
                                'src_id': node_id, 'target_id': target_id,
                                'relation': rel,
                                'edge_id': (edge_res or {}).get('edge_id'),
                                'deltas': (edge_res or {}).get('deltas', [])})
                        except Exception as _e:
                            self._log_error('batch_connect_to', _e, 'connecting %s → %s' % (node_id[:8], target_id[:8]))

        return {
            'nodes_created': len(created_ids),
            'results': results,
            'connections_created': connections_created,
            'connect_to_made': connect_to_made,
            'connect_to_failures': len(connect_to_failed),
            'connect_to_failed': connect_to_failed,
        }

    def revise_batch(self, revisions: List[Dict]) -> Dict[str, Any]:
        """Revise multiple nodes in one call. Each revision uses the same
        contract as revise() — per-field replace, immutable fields skipped
        with warning, deltas captured for trace history.

        Args:
            revisions: List of dicts, each with:
                - node_id (required): ID of node to revise
                - reason (required): why this revision
                - content, situation, reasoning, etc.: any revisable field

        Example:
            revise_batch(revisions=[
                {"node_id": "abc123", "content": "Judge now runs in daemon", "reason": "architecture changed"},
                {"node_id": "def456", "situation": "When debugging daemon connectivity", "reason": "adding situation"},
                {"node_id": "ghi789", "reasoning": "updated — encoder uses node catalog", "reason": "encoder v3.2"},
            ])

        Returns:
            {revised: count,
             results: [{node_id, status, error?, deltas?, warnings?}]}

            Per-result `deltas` and `warnings` mirror what revise() returns —
            callers (typically dispatch) use them to emit one trace event per
            revised node.
        """
        results = []
        revised_count = 0

        for spec in revisions:
            node_id = spec.get('node_id')
            if not node_id:
                results.append({'error': 'missing node_id', 'status': 'skipped'})
                continue

            reason = spec.get('reason', '')
            content = spec.get('content')

            # Extract field updates. Excludes node_id/reason/content (handled
            # explicitly) AND the dispatch control keys — mirroring standalone
            # revise()'s DISPATCH_KEYS reservation. `encoding_source` in
            # particular is the immutable CREATOR mark, not a revisable field:
            # the dispatch layer injects it for trace attribution, but a revise
            # must never write it onto the node, or a bulk-edit would silently
            # relabel who created the memory. (chain_id/session_id/_caller_session
            # are trace/identity control, never node fields.)
            _REVISE_CONTROL = ('node_id', 'reason', 'content', 'encoding_source',
                               'chain_id', 'session_id', '_caller_session')
            updates = {k: v for k, v in spec.items()
                       if k not in _REVISE_CONTROL and v is not None}

            try:
                result = self.revise(node_id=node_id, content=content,
                                     reason=reason, updates=updates)
                if result.get('error'):
                    results.append({'node_id': node_id, 'status': 'error', 'error': result['error']})
                else:
                    results.append({
                        'node_id': node_id,
                        'status': 'revised',
                        'deltas': result.get('deltas', []),
                        'warnings': result.get('warnings', []),
                    })
                    revised_count += 1
            except Exception as e:
                self._log_error('revise_batch', e, 'revising %s' % node_id[:8])
                results.append({'node_id': node_id, 'status': 'error', 'error': str(e)})

        return {
            'revised': revised_count,
            'results': results,
        }

    # validate_node removed 2026-04-13 — old node_metadata table dropped.

    def _generate_summary(self, title: str, content: Optional[str] = None) -> Optional[str]:
        """Generate a content_summary (max 200 chars) for tiered recall.

        Returns first sentence of content, or first 200 chars if no sentence boundary.
        Returns None if content is empty or very short (title suffices).
        """
        if not content or len(content) < 30:
            return None
        # First sentence
        period_idx = content.find('. ')
        if 0 < period_idx < 200:
            return content[:period_idx + 1]
        # First 200 chars with ellipsis
        if len(content) > 200:
            return content[:197] + '...'
        return content

    def _bridge_at_store_time(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Detect bridge opportunities at store-time.
        Returns array of bridges created.
        """
        max_bridges = self.get_config('bridge_max_per_remember', 2)
        candidates = self._find_bridge_candidates(node_id, limit=max_bridges)
        created = []

        for c in candidates:
            bridge = self._create_bridge(node_id, c['targetId'], c.get('sharedTitles', ''))
            if bridge:
                created.append(bridge)

        return created

    def set_personal(self, node_id: str, personal: str,
                     personal_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a node as personal information.

        Args:
            node_id: Node to mark
            personal: 'fixed' (permanent fact, auto-locks), 'fluid' (evolving truth,
                      10x slower decay), 'contextual' (depends on conditions), or
                      None to remove personal flag
            personal_context: For contextual nodes — when/where this applies
                              (e.g. "during technical sprints", "at work")

        Returns:
            Dict with node_id, personal, locked status
        """
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            return {'error': f'Invalid personal flag: {personal}. Use fixed/fluid/contextual/None.'}

        ts = self.now()

        # Fixed personal nodes are always locked
        if personal == 'fixed':
            self.conn.execute(
                'UPDATE nodes SET personal = ?, personal_context = ?, locked = 1, updated_at = ? WHERE id = ?',
                (personal, personal_context, ts, node_id)
            )
        else:
            self.conn.execute(
                'UPDATE nodes SET personal = ?, personal_context = ?, updated_at = ? WHERE id = ?',
                (personal, personal_context, ts, node_id)
            )
        self._maybe_commit()

        # Fetch updated node
        cursor = self.conn.execute(
            'SELECT title, locked, personal, personal_context FROM nodes WHERE id = ?',
            (node_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        return {
            'node_id': node_id,
            'title': row[0],
            'locked': row[1] == 1,
            'personal': row[2],
            'personal_context': row[3],
        }

    def get_personal_nodes(self, personal_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all personal nodes, optionally filtered by type.

        Args:
            personal_type: 'fixed', 'fluid', 'contextual', or None for all personal nodes

        Returns:
            List of personal node dicts
        """
        if personal_type:
            cursor = self.conn.execute(
                'SELECT id, type, title, content, personal, personal_context, locked FROM nodes WHERE personal = ? AND archived = 0 ORDER BY updated_at DESC',
                (personal_type,)
            )
        else:
            cursor = self.conn.execute(
                'SELECT id, type, title, content, personal, personal_context, locked FROM nodes WHERE personal IS NOT NULL AND archived = 0 ORDER BY updated_at DESC'
            )

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'personal': row[4],
                'personal_context': row[5], 'locked': row[6] == 1,
            })
        return results

    # ═══════════════════════════════════════════════════════════════
    # v6: Multi-vector enrichment (Embedding Migration to LLM)
    # The brain builds a structured prompt with neighbors.
    # Claude (or a local LLM) fills in Q/A/B/K.
    # Each is embedded and stored in node_enrichments.
    # ═══════════════════════════════════════════════════════════════

    def _build_enrichment_prompt(self, node_id: str, title: str,
                                  content: Optional[str] = None) -> Optional[str]:
        """Build the V5 structured enrichment prompt for a node.

        Finds neighbors via edges, formats them, and returns the prompt
        for Claude (or local LLM) to fill in.

        Returns None if node has no neighbors (nothing to anchor to).
        """
        try:
            graph_dal = self._graph
            neighbors = graph_dal.get_neighbors(
                node_id, limit=ENRICHMENT_NEIGHBOR_COUNT
            )
            if not neighbors:
                return None

            neighbor_lines = []
            for nb in neighbors:
                kw = nb.get('keywords', '') or ''
                kw_short = ', '.join(kw.split()[:5]) if kw else 'none'
                neighbor_lines.append(
                    f"- {nb['title'][:80]} ({nb['type']}, keywords: {kw_short})"
                )

            content_preview = (content or '')[:200]
            prompt = ENRICHMENT_PROMPT_TEMPLATE.format(
                neighbors='\n'.join(neighbor_lines),
                title=title,
                content=content_preview,
            )
            return prompt
        except Exception as e:
            print(f'[brain] _build_enrichment_prompt failed: {e}', file=sys.stderr)
            return None

    def store_enrichments(self, node_id: str, question: Optional[str] = None,
                          anchor: Optional[str] = None, bridge: Optional[str] = None,
                          keywords: Optional[str] = None) -> Dict[str, Any]:
        """Store enrichment vectors for a node (called after Claude fills in the prompt).

        Each non-None enrichment text is embedded and stored in node_enrichments.
        Returns count of enrichments stored and any errors.
        """
        vdal = self._vec_dal
        stored = 0
        errors = []

        enrichments = {
            'question': question,
            'anchor': anchor,
            'bridge': bridge,
            'keywords': keywords,
        }

        for vtype, text in enrichments.items():
            if not text or not text.strip():
                continue
            text = text.strip()
            try:
                blob = None
                if embedder.is_ready():
                    blob = embedder.embed_document(text)
                vdal.store(node_id, vtype, text, blob,
                           model=embedder.stats.get('model_name', 'unknown') if embedder.is_ready() else 'none')
                stored += 1
            except Exception as e:
                errors.append(f'{vtype}: {str(e)[:100]}')
                self._log_error('store_enrichment', e, '%s/%s' % (node_id[:8], vtype))

        return {
            'node_id': node_id,
            'enrichments_stored': stored,
            'errors': errors if errors else None,
        }

    def get_enrichment_coverage(self) -> Dict[str, Any]:
        """Get vector coverage stats."""
        try:
            return self._vec_dal.get_coverage_stats()
        except Exception as e:
            return {'error': str(e)}

    def find_node_by_title(self, title_query: str, threshold: float = 0.75,
                           top_k: int = 1,
                           session_id: str = '') -> Optional[Dict[str, Any]]:
        """Find a node by fuzzy title matching using embedding similarity.

        Embeds the query, scans all node embeddings, returns the best match(es)
        above threshold with a content snippet so the caller can verify
        correctness.

        Args:
            title_query: Title to search for (fuzzy)
            threshold: Minimum similarity (0.0-1.0). Default 0.75 is conservative
                       to prevent false matches. Lower to 0.6 for broader search.
            top_k: Return top K matches (default 1 = best match only)

        Returns: {id, title, type, similarity, content_snippet} or None.
                 If top_k > 1, returns list of matches.

        Note: prior to schema v28 this also returned a `keywords` field
        carrying the auto-extracted tokenizer dump. That column was
        dropped; verification now relies on content_snippet + title alone.
        """
        scored = {}  # id → result dict, dedup by node

        # Path 1: lexical — FTS-indexed candidates (_title_candidate_rows,
        # the same door the connect_to write path uses; no table scan),
        # verified by in-order token containment: every query token appears
        # in the title, in order, as a substring — the semantics the old
        # `LIKE %w%w%` full scan implemented, minus its cross-word false
        # positives and minus the scan.
        q_tokens = _title_tokens(title_query)
        text_ids = []
        if q_tokens:
            # Interactive lookup is best-effort: saturation just bounds the
            # lexical pool (Path 2 embeddings backstop); only the WRITE path
            # must refuse on it.
            candidate_rows, _saturated = self._title_candidate_rows(q_tokens)
            for nid, title in candidate_rows:
                t_lower = title.lower()
                pos = 0
                for tok in q_tokens:
                    pos = t_lower.find(tok, pos)
                    if pos == -1:
                        break
                    pos += len(tok)
                else:
                    text_ids.append(nid)
        if text_ids:
            ph = ','.join('?' * len(text_ids))
            for nid, title, ntype, snippet in self.conn.execute(
                    "SELECT id, title, type, SUBSTR(content, 1, 200) "
                    "FROM nodes WHERE id IN (%s)" % ph, text_ids).fetchall():
                scored[nid] = {
                    "id": nid, "title": title, "type": ntype,
                    "similarity": 0.95,  # text match = high confidence
                    "content_snippet": snippet or "",
                }

        # Path 2: Embedding similarity — semantic fallback
        if embedder.is_ready() and len(scored) < top_k:
            query_vec = embedder.embed_query(title_query)
            if query_vec:
                rows = self.conn.execute(
                    "SELECT ne.node_id, ne.embedding, n.title, n.type, "
                    "SUBSTR(n.content, 1, 200) as snippet "
                    "FROM node_enrichments ne JOIN nodes n ON ne.node_id = n.id "
                    "WHERE ne.vector_type = '_primary' AND n.archived = 0"
                ).fetchall()
                for node_id, emb_blob, title, ntype, snippet in rows:
                    if not emb_blob or node_id in scored:
                        continue
                    sim = embedder.cosine_similarity(query_vec, emb_blob)
                    if sim >= threshold:
                        scored[node_id] = {
                            "id": node_id, "title": title, "type": ntype,
                            "similarity": round(sim, 3),
                            "content_snippet": snippet or "",
                        }

        # Scope veil: fuzzy title search is ambient DISCOVERY (a probe
        # phrase enumerates the corpus with content snippets), not the
        # sanctioned reach-for-a-known-id — walled nodes never match.
        # Sessionless callers get the outward-only veil.
        _veil = self.scope_veil(session_id or '')
        if _veil:
            scored = {nid: v for nid, v in scored.items() if nid not in _veil}

        results = sorted(scored.values(), key=lambda x: x["similarity"], reverse=True)

        if top_k == 1:
            return results[0] if results else None
        return results[:top_k]

