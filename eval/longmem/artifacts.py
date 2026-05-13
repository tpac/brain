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
    nodes.jsonl        every active node with full content + KV + connections
    edges.jsonl        every edge_relation with weights + descriptions
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
        """Dump every interaction version present in this brain.

        Captures which prompt versions the encoder/surfacer/etc. used —
        critical for retrospective ("which version of the encoder produced
        these nodes?").
        """
        try:
            rows = brain.logs_conn.execute(
                "SELECT id, name, version, template, parameters, "
                "created_at, created_by, parent_version "
                "FROM interactions ORDER BY name, version"
            ).fetchall()
        except Exception as e:
            self._write_error('interactions.jsonl', e)
            return

        with open(self.path('interactions.jsonl'), 'w') as f:
            for r in rows:
                rec = {
                    'id': r[0], 'name': r[1], 'version': r[2],
                    'template': r[3] or '',
                    'parameters': self._safe_json(r[4]),
                    'created_at': r[5], 'created_by': r[6],
                    'parent_version': r[7],
                }
                f.write(json.dumps(rec) + '\n')

    def dump_traces(self, brain) -> None:
        """Dump every trace_event (any scale, any event_type)."""
        try:
            rows = brain.logs_conn.execute(
                "SELECT id, chain_id, scale, event_type, ref_type, ref_id, "
                "summary, metadata, session_id, interaction_id, created_at "
                "FROM trace_events ORDER BY id"
            ).fetchall()
        except Exception as e:
            self._write_error('traces.jsonl', e)
            return

        with open(self.path('traces.jsonl'), 'w') as f:
            for r in rows:
                rec = {
                    'id': r[0], 'chain_id': r[1], 'scale': r[2],
                    'event_type': r[3], 'ref_type': r[4], 'ref_id': r[5],
                    'summary': r[6] or '',
                    'metadata': self._safe_json(r[7]),
                    'session_id': r[8], 'interaction_id': r[9],
                    'created_at': r[10],
                }
                f.write(json.dumps(rec) + '\n')

    def dump_nodes(self, brain, prefix: str = '') -> None:
        """Dump every active (non-archived) node with full content + KV.

        File: nodes{prefix}.jsonl — one node per line. Use `prefix` for
        before/after checkpoints in multi-stage evals.
        """
        suffix = f'_{prefix}' if prefix else ''
        try:
            node_rows = brain.conn.execute(
                "SELECT id, type, title, content, keywords, activation, "
                "stability, access_count, locked, archived, critical, "
                "recency_score, emotion, emotion_label, emotion_source, "
                "project, confidence, personal, personal_context, "
                "evolution_status, resolved_at, resolved_by, due_date, "
                "content_summary, source_attribution, scope, "
                "encoding_version, encoding_source, revised_at, "
                "source_turn_id, last_accessed, created_at, updated_at "
                "FROM nodes WHERE archived = 0 ORDER BY created_at"
            ).fetchall()
        except Exception as e:
            self._write_error(f'nodes{suffix}.jsonl', e)
            return

        node_cols = [
            'id', 'type', 'title', 'content', 'keywords', 'activation',
            'stability', 'access_count', 'locked', 'archived', 'critical',
            'recency_score', 'emotion', 'emotion_label', 'emotion_source',
            'project', 'confidence', 'personal', 'personal_context',
            'evolution_status', 'resolved_at', 'resolved_by', 'due_date',
            'content_summary', 'source_attribution', 'scope',
            'encoding_version', 'encoding_source', 'revised_at',
            'source_turn_id', 'last_accessed', 'created_at', 'updated_at',
        ]

        # Pre-load KV by node_id for efficient join
        kv_by_node: Dict[str, Dict[str, str]] = {}
        try:
            for nid, key, val in brain.conn.execute(
                    "SELECT node_id, key, value FROM node_metadata_kv"
            ).fetchall():
                kv_by_node.setdefault(nid, {})[key] = val
        except Exception:
            pass

        with open(self.path(f'nodes{suffix}.jsonl'), 'w') as f:
            for row in node_rows:
                rec = dict(zip(node_cols, row))
                rec['kv'] = kv_by_node.get(rec['id'], {})
                f.write(json.dumps(rec, default=str) + '\n')

    def dump_edges(self, brain, prefix: str = '') -> None:
        """Dump every active edge_relation row, joined with edges + node titles.

        File: edges{prefix}.jsonl. One edge_relation per line — multiple
        relations per (source, target) pair appear as separate rows
        (Stage 1B model — each edge can carry multiple relations).
        """
        suffix = f'_{prefix}' if prefix else ''
        try:
            rows = brain.conn.execute(
                "SELECT er.edge_id, er.relation, er.description, er.weight, "
                "er.encoding_source, er.decay_rate, er.created_at, "
                "e.source_id, e.target_id, e.weight, e.co_access_count, "
                "src.title, tgt.title "
                "FROM edge_relations er "
                "JOIN edges e ON er.edge_id = e.edge_id "
                "JOIN nodes src ON e.source_id = src.id "
                "JOIN nodes tgt ON e.target_id = tgt.id "
                "WHERE er.archived = 0 AND src.archived = 0 AND tgt.archived = 0 "
                "ORDER BY er.created_at"
            ).fetchall()
        except Exception as e:
            self._write_error(f'edges{suffix}.jsonl', e)
            return

        with open(self.path(f'edges{suffix}.jsonl'), 'w') as f:
            for r in rows:
                rec = {
                    'edge_id': r[0],
                    'relation': r[1], 'description': r[2],
                    'relation_weight': r[3], 'encoding_source': r[4],
                    'decay_rate': r[5], 'created_at': r[6],
                    'source_id': r[7], 'target_id': r[8],
                    'edge_weight': r[9], 'co_access_count': r[10],
                    'source_title': r[11], 'target_title': r[12],
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
