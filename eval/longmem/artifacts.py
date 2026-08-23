"""Eval artifacts dumper — captures per-item analysis bytes outside the brain.

WHY THIS EXISTS
---------------
Eval runs are expensive (wall time + API spend). When a failure looks
suspicious you should be able to answer "did the scout extract this? did
the encoder receive it? at what rank did this node land?" WITHOUT
re-running the eval. That requires durable artifacts captured at the
moment they exist.

This module is the durable-artifacts layer. After each item completes
the harness calls the dumper; the dumper reads through the brain's
existing surfaces (DAL connections, no new APIs in `servers/`) and
writes a self-contained bundle to disk. Brain code is unchanged.

WHAT GETS WRITTEN — per item, under
`eval/longmem/reports/{run_name}/items/{qid}/`:

    meta.json          {qid, axis, run_name, dates, timings, ingest stats}
    traces.jsonl       every trace_event (S0/S1/S2/etc) — full metadata
    nodes.jsonl        the run's node delta (node_created traces → get_node),
                       full content + KV + corrections; seeds excluded
    edges.jsonl        edge relations touching the delta, with weights +
                       descriptions (noise relations excluded)
    interactions.jsonl every interaction version this brain saw
    recall.json        query phase — query, top-N candidates with scores,
                       selected, dropped, context, classifier evidence
                       augmented with rank-of-fact-bearing-node
    result.json        the harness result dict

The artifact set is scale-agnostic: traces.jsonl captures any scale's
events, nodes.jsonl captures any node type. Adding S0 or S2 evals later
needs no schema change here — only new analyzer functions.

WHAT DOES *NOT* GET WRITTEN
- The brain.db itself (use `--keep_dbs` if you need to re-run queries
  against the same brain — but for post-hoc analysis, the dumps suffice).
- The embedder model (large, reproducible).

COST
- ~1 MB per item on disk.
- ~200 ms extra per item to dump.
- No API cost. No additional LLM calls.

EXTENDING TO OTHER SCALES
- For S2 evals (e.g. "did consolidation merge cluster X?"):
    dumper.checkpoint('pre_unit', brain)
    # run unit
    dumper.checkpoint('post_unit', brain)
  Each checkpoint snapshots nodes + edges under the given prefix,
  enabling before/after diffs.
- For S0/multi-turn evals: the existing per-item structure works; just
  call dump_meta + dump_traces + dump_nodes + dump_recall as relevant.

DOCUMENTATION: eval/ARTIFACTS.md (read first if you're new here).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalArtifactsDumper:
    """Per-item artifacts dumper.

    Construct once per item, call dump_* methods at the appropriate moments
    in the eval lifecycle. Files are written incrementally so a partial
    failure preserves whatever was dumped before the crash.

    Caller is responsible for the order — typical sequence:
        1. dump_meta(...)               # immediately after item starts
        2. dump_interactions(brain)     # snapshot interactions early
        3. (run ingest)
        4. dump_traces(brain)           # all events written so far
        5. dump_nodes(brain)            # graph at end-of-ingest
        6. dump_edges(brain)            # connections at end-of-ingest
        7. (run query + answer)
        8. dump_recall(...)             # query-phase artifact
        9. dump_result(result)          # final harness result

    For multi-checkpoint evals (S2 etc.), call dump_nodes / dump_edges
    multiple times with the `prefix` argument:
        dumper.dump_nodes(brain, prefix='pre_consolidation')
        # run consolidation
        dumper.dump_nodes(brain, prefix='post_consolidation')
    """

    def __init__(self, run_name: str, qid: str,
                 reports_root: Optional[str] = None) -> None:
        if reports_root is None:
            reports_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'reports')
        self.run_dir = Path(reports_root) / run_name / 'items' / qid
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.qid = qid
        self._t0 = time.time()

    # ─── path helpers ────────────────────────────────────────────────

    def path(self, name: str) -> Path:
        return self.run_dir / name

    # ─── individual dumpers ──────────────────────────────────────────

    def dump_meta(self, axis: str, question: str, gold: str,
                  question_date: Optional[str], haystack_dates: List[str],
                  haystack_session_ids: List[str],
                  haystack_turn_count: int) -> None:
        """One-shot metadata dump. Write at item start."""
        meta = {
            'run_name': self.run_name,
            'qid': self.qid,
            'axis': axis,
            'question': question,
            'gold': gold,
            'question_date': question_date,
            'haystack_dates': haystack_dates,
            'haystack_session_ids': haystack_session_ids,
            'haystack_turn_count': haystack_turn_count,
            'started_at': time.time(),
        }
        self._write_json('meta.json', meta)

    def dump_interactions(self, brain) -> None:
        """Dump which prompt+config every boundary actually RAN, plus every
        registered override version present in this brain.

        Two records per line-kind, because a version dump alone answers the
        retrospective question ("which encoder produced these nodes?") only
        while every boundary has a row. Under the override model most names
        have NO row and run their code default, so a rows-only dump reports
        nothing for exactly the boundaries that did the work. `kind:
        'effective'` carries the resolved fingerprint — the content address of
        what ran, whether it came from a row or from code.
        """
        try:
            listing = brain.list_interactions()
        except Exception as e:
            self._write_error('interactions.jsonl', e)
            return

        with open(self.path('interactions.jsonl'), 'w') as f:
            from servers.interaction_defaults import INTERACTION_DEFAULTS
            for name in sorted(INTERACTION_DEFAULTS):
                stamp = brain.get_interaction_stamp(name)
                f.write(json.dumps({
                    'kind': 'effective', 'name': name,
                    'fingerprint': stamp['fingerprint'],
                    'source': stamp['source'],
                    'version': stamp['version'],
                }) + '\n')
            for entry in listing:
                name = entry['name']
                for v in brain.list_interaction_versions(name):
                    row = brain.get_interaction(name, version=v['version']) or {}
                    f.write(json.dumps({
                        'kind': 'version', 'name': name,
                        'version': v['version'],
                        'template': row.get('template') or '',
                        'parameters': self._safe_json(row.get('parameters')),
                        'created_at': row.get('created_at'),
                        'created_by': v.get('created_by'),
                        'active': entry.get('active_version') == v['version'],
                        'active_set_by': entry.get('active_set_by'),
                    }) + '\n')

    TRACE_DUMP_LIMIT = 200000

    def dump_traces(self, brain) -> None:
        """Dump every trace_event (any scale, any event_type), through the
        trace query API — which flags truncation, where the raw dump would
        just end.

        No `interaction_id` column: it is an install-local rowid that nothing
        JOINs, and the K-provenance it used to stand for now travels in the
        event metadata as `interaction_fingerprint` — content-addressed, so it
        stays meaningful in a frozen corpus where the row it pointed at is
        gone.
        """
        try:
            out = brain.query_traces(hours=None, limit=self.TRACE_DUMP_LIMIT)
        except Exception as e:
            self._write_error('traces.jsonl', e)
            return

        rows = out.get('events') or []
        if out.get('truncated'):
            # Its own marker, not `.error`: a truncated dump is real data that
            # stops early, and a consumer must be able to tell that from a
            # dump that never happened. Per-item eval brains are fresh, so
            # thousands of rows — hitting this means the dumper ran against
            # something far bigger than one item.
            with open(self.path('traces.jsonl.truncated'), 'w') as f:
                f.write('trace dump hit the %d-row limit — this is a prefix '
                        'of the run, not the run\n' % self.TRACE_DUMP_LIMIT)

        with open(self.path('traces.jsonl'), 'w') as f:
            for r in sorted(rows, key=lambda e: e['id']):
                f.write(json.dumps(r) + '\n')

    @staticmethod
    def _delta_node_ids(brain) -> Dict[str, str]:
        """The run's node delta: {node_id: 'created' | 'revised'}, in trace
        order (created first, then revised-only).

        'created' = a node_created trace row; 'revised' = a node_revised row
        for a node the run did NOT create — a pre-existing (seed) node the
        encoder revised or absorbed into. Without the revised half, gold
        content written into a seed node would be invisible to every
        gold-bearing scan downstream.

        Per-item eval brains are fresh (`create_fresh_eval_brain(wipe=True)`
        everywhere live), so the whole logs DB is this run: no session filter.
        Scoping by session_id would MISS nodes — items ingest many haystack
        sessions and S2 units carry no session at all. Seeds load at Brain
        init, outside dispatch, so they never appear in the delta.

        Loud on truncation: a clipped delta would silently undercount the
        very set every downstream metric is computed over.
        """
        delta: Dict[str, str] = {}
        for ref_type, op in (('node_created', 'created'),
                             ('node_revised', 'revised')):
            res = brain.query_traces(ref_type=ref_type, hours=None,
                                     limit=10000)
            if res.get('truncated'):
                raise RuntimeError('%s delta truncated: %s'
                                   % (ref_type, res['truncated']))
            # get_by_ref_type returns newest-first; reverse to trace order.
            for ev in reversed(res.get('events', [])):
                nid = ev.get('ref_id') or ''
                if nid and nid not in delta:
                    delta[nid] = op
        return delta

    def dump_nodes(self, brain, prefix: str = '') -> None:
        """Dump the run's node delta with full content + KV + corrections.

        Sourced from the brain, not raw SQL (B2 ruling,
        docs/EVAL-BRAIN-PATH-MIGRATION.md): which nodes exist because this
        run ran = the node_created + node_revised trace delta (each record
        carries `delta_op`: 'created', or 'revised' for a pre-existing node
        the run mutated — gold absorbed into a seed stays scannable); content
        = brain.get_node, the canonical pull (corrections walked, KV
        attached). The untouched seed pack is excluded by construction —
        where the old snapshot SQL dumped it wholesale and inflated every
        per-item metric by the seed-pack size. Created-then-archived nodes
        (S2 consolidation) stay visible with archived=true.

        File: nodes{prefix}.jsonl — one node per line, creation order. Use
        `prefix` for before/after checkpoints in multi-stage evals.
        """
        suffix = f'_{prefix}' if prefix else ''
        try:
            delta = self._delta_node_ids(brain)
            nodes = brain.get_node(list(delta)) if delta else {}
        except Exception as e:
            self._write_error(f'nodes{suffix}.jsonl', e)
            return

        with open(self.path(f'nodes{suffix}.jsonl'), 'w') as f:
            for nid, op in delta.items():
                node = nodes.get(nid)
                if not node:
                    continue
                rec = dict(node)
                rec['delta_op'] = op
                rec['kv'] = rec.pop('_metadata', None) or {}
                # Edges live in edges.jsonl — don't duplicate them per node.
                rec.pop('connections', None)
                f.write(json.dumps(rec, default=str) + '\n')

    def dump_edges(self, brain, prefix: str = '') -> None:
        """Dump edge relations touching the run's node delta.

        Sourced from the brain's exposed edge read — the connections
        brain.get_node attaches (GraphDAL stays behind it). Created nodes
        only — a revised seed's pre-existing seed↔seed edges are not run
        behavior. One row per (source, target, relation); an edge between
        two run nodes appears under both owners, deduped here. Noise
        relations (co_accessed, emergent_bridge) are excluded by the
        canonical read — they are recall/remember mechanics, not encoder
        behavior, which is what the consumers of this file measure.

        File: edges{prefix}.jsonl.
        """
        suffix = f'_{prefix}' if prefix else ''
        try:
            ids = [nid for nid, op in self._delta_node_ids(brain).items()
                   if op == 'created']
            nodes = brain.get_node(ids) if ids else {}
        except Exception as e:
            self._write_error(f'edges{suffix}.jsonl', e)
            return

        seen = set()
        with open(self.path(f'edges{suffix}.jsonl'), 'w') as f:
            for nid in ids:
                node = nodes.get(nid)
                if not node:
                    continue
                for conn in node.get('connections') or []:
                    outgoing = conn.get('direction') == 'outgoing'
                    src_id, tgt_id = ((nid, conn.get('id')) if outgoing
                                      else (conn.get('id'), nid))
                    src_title, tgt_title = (
                        (node.get('title'), conn.get('title')) if outgoing
                        else (conn.get('title'), node.get('title')))
                    for rel in conn.get('relations') or []:
                        key = (src_id, tgt_id, rel.get('relation'))
                        if key in seen:
                            continue
                        seen.add(key)
                        rec = {
                            'relation': rel.get('relation'),
                            'description': rel.get('description'),
                            'relation_weight': rel.get('weight'),
                            'source_id': src_id, 'target_id': tgt_id,
                            'edge_weight': conn.get('weight'),
                            'source_title': src_title,
                            'target_title': tgt_title,
                        }
                        f.write(json.dumps(rec, default=str) + '\n')

    def dump_recall(self, query_session_id: str, query: str,
                    candidates: List[Dict[str, Any]],
                    selected: List[Any], dropped: List[Any],
                    context: str,
                    classifier_evidence: Optional[Dict[str, Any]] = None,
                    answerer_response: Optional[Dict[str, Any]] = None,
                    tool_trace: Optional[List[Any]] = None,
                    surface_variant: str = "",
                    ) -> None:
        """Dump the query-phase artifact.

        Args:
            query_session_id: the session id used for the query
            query: the question text passed into recall
            candidates: list of {id, title, score, type, source} dicts
                from the recall trace (top-N before surface filters)
            selected: surface-selected node IDs (whatever shape the trace
                gave us — list of strings or dicts)
            dropped: dropped nodes from surface
            context: the additionalContext that reached the answerer
            classifier_evidence: evidence dict produced by classify_failure
                (gold_in_brain, recall_candidates_count, etc.) — augmented
                here with `fact_node_rank_in_candidates` if computable.
            answerer_response: {hypothesis, abstained, has_context,
                tokens_in, tokens_out, elapsed_ms} from the answerer.
        """
        # Compute the rank of any fact-bearing nodes inside the candidate list.
        # This is the missing piece in the original classifier — it tells you
        # whether SURFACE_MISS is a true surface failure (fact in candidates,
        # surface skipped) or a recall ranker failure (fact buried below
        # the candidate cutoff).
        fact_node_ranks: Dict[str, int] = {}
        if classifier_evidence and classifier_evidence.get('gold_in_brain', {}).get('matches'):
            cand_ids = [c.get('id', '')[:8] for c in candidates]
            for match in classifier_evidence['gold_in_brain']['matches']:
                short_id = (match.get('node_id') or '')[:8]
                if short_id and short_id in cand_ids:
                    fact_node_ranks[short_id] = cand_ids.index(short_id) + 1
                elif short_id:
                    fact_node_ranks[short_id] = -1  # not in top-N

        rec = {
            'query_session_id': query_session_id,
            'query': query,
            'candidates': candidates,
            'candidate_count': len(candidates),
            'selected': selected,
            'dropped': dropped,
            'context': context,
            'context_chars': len(context or ''),
            'classifier_evidence': classifier_evidence,
            'fact_node_ranks_in_candidates': fact_node_ranks,
            'answerer_response': answerer_response,
            # Denormalized from traces.jsonl (surface_selected K event) so post-hoc
            # tool-usage analysis doesn't have to re-parse traces. v4 surface
            # always writes empty list; v5 agentic writes the per-call tool_use
            # sequence (recall_topical / recall_recent / etc.).
            'tool_trace': tool_trace or [],
            'surface_variant': surface_variant,
        }
        self._write_json('recall.json', rec)

    def dump_result(self, result: Dict[str, Any]) -> None:
        """Mirror the harness result dict to the artifacts dir."""
        self._write_json('result.json', result)

    def dump_agent_calls(self, session_ids: List[str]) -> Dict[str, int]:
        """Collect per-call encoder + surface artifacts into agent_calls/.

        Encoder prompts and judge results live in the item brain's
        {BRAIN_DB_DIR}/payloads/{date}/{chain}/ (the payload recorder — the
        eval is a sanctioned direct reader; chains are s1e-{sid[:8]}-{stop}
        and s1r-{sid[:8]}-{stop}). The surface-selected files are operational
        state, still under BRAIN_TMP_DIR.

        Together with interactions.jsonl (which captures the system prompt
        version active at each call), this gives us full replay-ability:
        take any saved agent_call, swap in a new system prompt from a future
        registered version, re-run client.messages.create(...) and observe
        the new behavior — no need to re-run the eval pipeline.

        Returns: {'encoder_calls': N, 'surface_calls': M, 'errors': E}
        """
        import shutil
        from eval.longmem.fresh_brain import capture_files_for
        # Set per-run by fresh_brain.create_fresh_eval_brain (to the item
        # brain dir), so concurrent runs can't cross-copy.
        db_dir = os.environ.get('BRAIN_DB_DIR', '')
        out_dir = self.run_dir / 'agent_calls'
        out_dir.mkdir(exist_ok=True)
        encoder_n = 0
        surface_n = 0
        errors = 0

        for sid in (session_ids or []):
            # Encoder payloads — prompt + round_payload per encoding window;
            # name by chain dir + file so stops/rounds stay distinguishable.
            # capture_files_for owns the recorder layout — the one eval-side
            # reader (kind='' = every payload kind on the chain).
            for src in capture_files_for(db_dir, sid, prefix='s1e', kind=''):
                try:
                    name = '%s__%s' % (os.path.basename(os.path.dirname(src)),
                                       os.path.basename(src))
                    shutil.copy2(src, out_dir / name)
                    encoder_n += 1
                except Exception:
                    errors += 1
            # Judge payloads — agentic surface output per recall.
            for src in capture_files_for(db_dir, sid, prefix='s1r', kind=''):
                try:
                    name = '%s__%s' % (os.path.basename(os.path.dirname(src)),
                                       os.path.basename(src))
                    shutil.copy2(src, out_dir / name)
                    surface_n += 1
                except Exception:
                    errors += 1
            # (The per-query surface-selected tmp file was retired with the
            # co_accessed family — the surfaced ids live in the
            # surface_selected trace, which dump_recall already records.)
        stats = {'encoder_calls': encoder_n, 'surface_calls': surface_n,
                 'errors': errors}
        self._write_json('agent_calls/_manifest.json', stats)
        return stats

    # ─── lower-level helpers ─────────────────────────────────────────

    @staticmethod
    def _safe_json(s: Optional[str]) -> Any:
        if not s:
            return None
        if isinstance(s, (dict, list)):
            return s
        try:
            return json.loads(s)
        except Exception:
            return s  # raw string fallback — never lose data

    def _write_json(self, name: str, obj: Any) -> None:
        with open(self.path(name), 'w') as f:
            json.dump(obj, f, indent=2, default=str)

    def _write_error(self, name: str, err: Exception) -> None:
        with open(self.path(name + '.error'), 'w') as f:
            f.write(f'{type(err).__name__}: {err}\n')


# ─── reading helpers (for analyzer) ──────────────────────────────────

def load_artifacts(run_name: str, qid: str,
                   reports_root: Optional[str] = None) -> Dict[str, Any]:
    """Load all artifact files for one item into a single dict.

    Convenience for analysis scripts. Missing files become None — never
    raises so callers can do partial inspection.
    """
    if reports_root is None:
        reports_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'reports')
    base = Path(reports_root) / run_name / 'items' / qid

    out: Dict[str, Any] = {'qid': qid, 'run_name': run_name, 'dir': str(base)}

    for json_name in ['meta.json', 'recall.json', 'result.json']:
        p = base / json_name
        out[json_name.replace('.json', '')] = (
            json.loads(p.read_text()) if p.exists() else None)

    for jsonl_name in ['traces.jsonl', 'nodes.jsonl', 'edges.jsonl',
                       'interactions.jsonl']:
        p = base / jsonl_name
        if p.exists():
            out[jsonl_name.replace('.jsonl', '')] = [
                json.loads(line) for line in p.read_text().splitlines() if line.strip()
            ]
        else:
            out[jsonl_name.replace('.jsonl', '')] = None

    return out


def list_items(run_name: str, reports_root: Optional[str] = None) -> List[str]:
    """List all qids that have artifacts under a run."""
    if reports_root is None:
        reports_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'reports')
    items_dir = Path(reports_root) / run_name / 'items'
    if not items_dir.exists():
        return []
    return sorted(d.name for d in items_dir.iterdir() if d.is_dir())
