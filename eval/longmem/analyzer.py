"""Eval analyzer — reads artifact bundles and produces deep failure-mode reports.

WHY
---
The harness's failure_classifier produces directional labels (ENCODE_MISS /
SURFACE_MISS / RECALL_MISS / ANSWER_MISS) based on a keyword scan + s1r
trace. These labels are useful but not authoritative — for example, the
keyword scan can't distinguish:

    encoder_filtered_by_design  vs  encoder_tried_but_missed
    recall_ranker_buried_it     vs  surface_skipped_it

The artifact bundle has the data needed for that distinction. This module
walks artifacts and produces a richer label + actionable evidence.

USAGE
-----
    from eval.longmem.analyzer import analyze_failure
    report = analyze_failure('eval_a_failures_deep_2026_05_10', '58470ed2')
    print(report['markdown'])

REFINED BUCKETS (subdivisions of the original four)
---------------------------------------------------
ENCODE_MISS variants:
    encoder_filtered     — encoder explicitly decided 0 nodes (e.g. "no stake")
    encoder_no_extract   — scouts emitted nothing relevant; encoder had nothing
    encoder_partial      — encoder wrote nodes but gold-bearing fact not encoded
    encoder_paraphrased  — gold fact MAY be encoded under different terms (manual review)

SURFACE_MISS variants:
    surface_skipped      — fact-bearing node IS in candidates, surface didn't pick
    surface_empty_ctx    — surface returned nothing (across-the-board abstention)

RECALL_MISS variants:
    ranker_buried        — fact-bearing node NOT in top-N candidates
    selection_diluted    — fact-bearing node in candidates but selected siblings missed it
    no_query_signal      — recall returned zero candidates entirely

ANSWER_MISS variants:
    answerer_overcautious — context had the fact, answerer abstained
    answerer_perspective  — context had it, answerer interpreted question differently
    answerer_other        — context had it, answerer wrong for unclear reasons
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a script — `./dev python3 eval/longmem/analyzer.py ...`
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.longmem.artifacts import load_artifacts


# ─── helpers — gold-fact recognition in different layers ──────────────

def _extract_gold_terms(gold: str, min_len: int = 4) -> List[str]:
    """Extract distinctive terms from gold answer.

    Drops common stop words + short tokens. Used to find the gold-bearing
    node and to scan scout outputs.
    """
    if not gold:
        return []
    stop = {
        'the', 'a', 'an', 'and', 'or', 'but', 'so', 'this', 'that', 'with',
        'for', 'from', 'into', 'about', 'is', 'are', 'was', 'were', 'be',
        'have', 'has', 'had', 'will', 'can', 'may', 'i', 'you', 'we', 'my',
        'your', 'our', 'they', 'them', 'their', 'it', 'its', 'in', 'on',
        'of', 'to', 'as', 'at', 'by', 'not', 'no', 'do', 'does', 'did',
        'said', 'says', 'mentioned', 'mention', 'enough', 'how', 'what',
        'when', 'where', 'who', 'which', 'why',
    }
    words = re.findall(r"[A-Za-z0-9$.]+", gold.lower())
    return [w for w in words if len(w) >= min_len and w not in stop]


def _node_text(node: Dict[str, Any]) -> str:
    """Concat all searchable text fields of a node into one lower-case string."""
    parts = [
        node.get('title') or '',
        node.get('content') or '',
        node.get('keywords') or '',
        node.get('content_summary') or '',
    ]
    kv = node.get('kv') or {}
    # Both voice-field generations: bundles dumped before schema v31 carry
    # user_raw_quote/anchor_raw_quote; post-rename bundles the their_/my_ pair.
    for k in ('situation', 'reasoning', 'their_raw_quote', 'my_raw_quote',
              'user_raw_quote', 'anchor_raw_quote',
              'event_description', 'value', 'entity', 'question'):
        v = kv.get(k)
        if v:
            parts.append(str(v))
    return ' '.join(p.lower() for p in parts if p)


def _gold_scan_terms(gold: str) -> List[str]:
    """Terms for the gold scan: this module's strict extractor, falling back
    to the CLASSIFIER's (digits, 2-3-letter uppercase, ≥4-char words) so
    short/numeric golds ("220", "AI", "6 PM") stay scannable exactly where
    the classifier can scan them. Single source for the fallback — a local
    copy of that rule set is how the two modules' verdicts drifted apart."""
    from eval.longmem.classifier import _extract_key_terms
    terms = _extract_gold_terms(gold)
    if terms:
        return terms
    terms = _extract_key_terms(gold)
    # The classifier's own guard (classifier.py, scan gate): a single
    # sub-2-char term ("3") substring-matches everything — no verdict.
    if len(terms) == 1 and len(terms[0]) < 2:
        return []
    return terms


def _gold_scan_basis(gold: str) -> str:
    """How _find_gold_bearing_nodes will search this gold: 'terms',
    'phrase' (exact substring of the whole gold — gate shared with the
    classifier via PHRASE_SCAN_MIN_CHARS), or 'unscannable'."""
    from eval.longmem.classifier import PHRASE_SCAN_MIN_CHARS
    if _gold_scan_terms(gold):
        return 'terms'
    if len((gold or '').strip()) >= PHRASE_SCAN_MIN_CHARS:
        return 'phrase'
    return 'unscannable'


def _find_gold_bearing_nodes(nodes: List[Dict[str, Any]], gold: str,
                             ) -> List[Dict[str, Any]]:
    """Find nodes whose text contains all distinctive gold terms.

    Stricter than the harness's classifier scan because it requires ALL
    gold terms in one node (not just the title/content fields).

    Short/numeric golds ("220", "6 PM") extract zero >=4-char terms —
    returning [] for them used to render "gold NOT in any node (encoder
    gap)" for golds that were never actually searched (32/163 on-disk items
    contradicted the classifier's own scan). Falls back to an exact phrase
    match on the whole gold, mirroring the classifier; 'unscannable' golds
    return [] and the refined bucket says so instead of blaming the encoder.
    """
    basis = _gold_scan_basis(gold)
    if basis == 'unscannable':
        return []
    terms = _gold_scan_terms(gold)
    phrase = (gold or '').lower().strip()
    out = []
    for n in nodes:
        text = _node_text(n)
        hit = all(t in text for t in terms) if terms else phrase in text
        if hit:
            out.append({
                'id': n['id'],
                'title': n.get('title') or '',
                'type': n.get('type') or '',
                'encoding_source': n.get('encoding_source') or '',
                'matched_terms': terms or [phrase],
            })
    return out


# ─── per-trace walkers ────────────────────────────────────────────────

def _walk_scouts(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract per-scout metrics + candidates from trace events.

    Returns one row per scout invocation:
        {scout: 'facts', turn_window_idx: 1, scanned: {...},
         candidates: [...], passed: N}
    """
    out = []
    for t in traces:
        if t.get('ref_type') == 'scout_findings':
            md = t.get('metadata') or {}
            out.append({
                'scout': md.get('scout', '?'),
                'candidates': md.get('candidate_handles', []),
                'errors': md.get('errors', []),
                'warnings': md.get('warnings', []),
                'created_at': t.get('created_at', ''),
            })
        elif t.get('ref_type') == 'scout_input':
            md = t.get('metadata') or {}
            out.append({
                'scout': md.get('scout', '?'),
                'scanned': md.get('scanned', {}),
                'latency_ms': md.get('latency_ms', 0),
                'is_input_only': True,
                'created_at': t.get('created_at', ''),
            })
    return sorted(out, key=lambda r: r['created_at'])


def _walk_encoder(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract encoder runs with journal + actions count.

    Returns one row per s1e cycle:
        {actions: N, write_actions: N, rounds: N, inputs_processed: N,
         journal_entry: '...', outcomes: {...}, created_at: '...'}
    """
    out = []
    for t in traces:
        if t.get('ref_type') == 'encoding_run':
            md = t.get('metadata') or {}
            out.append({
                'actions': md.get('actions', 0),
                'write_actions': md.get('write_actions', 0),
                'rounds': md.get('rounds', 0),
                'inputs_processed': md.get('inputs_processed', 0),
                'outcomes': md.get('outcomes', {}),
                'journal_entry': md.get('journal_entry', ''),
                'rejection_skipped': md.get('rejection_skipped', 0),
                'created_at': t.get('created_at', ''),
            })
    return sorted(out, key=lambda r: r['created_at'])


def _candidate_rank(candidates: List[Dict[str, Any]],
                    node_id: str) -> Optional[int]:
    """Return 1-based rank of node_id in candidates (None if absent)."""
    short = (node_id or '')[:8]
    for i, c in enumerate(candidates):
        if (c.get('id') or '')[:8] == short:
            return i + 1
    return None


# ─── refined bucket logic ─────────────────────────────────────────────

def _refine_bucket(bundle: Dict[str, Any]
                   ) -> Tuple[str, Dict[str, Any]]:
    """Compute a refined bucket label from full artifact data.

    Returns (refined_bucket, evidence_dict). Items that passed return
    ('passed', {...}) so callers can distinguish pass-through from failure.
    """
    meta = bundle.get('meta') or {}
    nodes = bundle.get('nodes')
    traces = bundle.get('traces')
    recall = bundle.get('recall')
    result = bundle.get('result')

    # Pass-through: don't run failure refinement on items that succeeded.
    # The harness's judge already decided this; refining further is noise.
    if result is not None and result.get('correct') is True:
        return 'passed', {
            'original_bucket': None,
            'note': 'Item passed the judge — no failure to refine.',
            'hypothesis': result.get('hypothesis', '')[:200],
        }

    # load_artifacts returns None for a MISSING file (an empty nodes.jsonl is
    # [] — a real "encoder created nothing" answer). A missing file must not
    # refine as encoder_no_extract / no_query_signal — or, with result.json
    # gone, turn a passing item into a failure. Same tri-state report.py's
    # _gold_in_brain_for_item already keeps for this class.
    missing = [name for name, val in (('nodes', nodes), ('traces', traces),
                                      ('recall', recall), ('result', result))
               if val is None]
    if missing:
        return 'artifacts_missing', {
            'original_bucket': (result or {}).get('failure_bucket'),
            'missing_files': missing,
            'note': 'artifact files absent — item not measurable; check '
                    '*.jsonl.error sidecars in the item dir',
        }

    result = result or {}
    recall = recall or {}
    gold = meta.get('gold', '') or result.get('answer_gold', '')
    original_bucket = result.get('failure_bucket')

    encoder_runs = _walk_encoder(traces)
    scouts = _walk_scouts(traces)

    # Find gold-bearing nodes via deeper search (any text field, all KV).
    bearing = _find_gold_bearing_nodes(nodes, gold)

    evidence: Dict[str, Any] = {
        'original_bucket': original_bucket,
        'gold_terms': _gold_scan_terms(gold),
        'gold_scan_basis': _gold_scan_basis(gold),
        'gold_bearing_nodes': bearing,
        'encoder_run_count': len(encoder_runs),
        'encoder_total_actions': sum(r['actions'] for r in encoder_runs),
        # Keep full journals for keyword detection; markdown will truncate per row.
        'encoder_journals_full': [
            r['journal_entry'] for r in encoder_runs if r['journal_entry']
        ],
        'encoder_journal_summary': [
            r['journal_entry'][:240] for r in encoder_runs if r['journal_entry']
        ],
        'scout_summary': [
            f"{s['scout']}: {len(s.get('candidates', []))} candidates"
            for s in scouts if not s.get('is_input_only')
        ],
        'recall_candidate_count': recall.get('candidate_count', 0),
        'recall_selected_count': len(recall.get('selected', [])),
        'recall_context_chars': recall.get('context_chars', 0),
    }

    # Compute rank of any bearing node in the recall candidates
    candidates = recall.get('candidates', []) or []
    if bearing and candidates:
        evidence['fact_node_ranks'] = {}
        for b in bearing:
            r = _candidate_rank(candidates, b['id'])
            evidence['fact_node_ranks'][b['id'][:8]] = r if r else 'NOT_IN_TOP_N'

    # Refined buckets
    if evidence['gold_scan_basis'] == 'unscannable':
        # Gold too short to search either way ("3") — an encoder-gap verdict
        # here would be a coin flip dressed as a measurement.
        return 'gold_unscannable', evidence

    if not bearing:
        # Gold fact NOT in any node — refine the encode-side miss
        if evidence['encoder_total_actions'] == 0:
            # Encoder ran but encoded nothing — was this a deliberate decision?
            # Search FULL journals (not truncated 240-char summaries) for explicit
            # filtering language — the journal entries can be 1-2K chars and the
            # "decision: zero nodes" line is often near the end.
            journal = ' '.join(evidence.get('encoder_journals_full') or []).lower()
            if any(kw in journal for kw in [
                'zero nodes', 'encoding anything here would be padding',
                'no new knowledge', 'no personal stake', 'nothing genuinely new',
                'declined to encode', 'not worth encoding', 'no decision',
                'decision: zero', 'no operator voice', 'routine academic',
                'one-shot', 'one shot task', 'nothing to encode',
            ]):
                return 'encoder_filtered', evidence
            return 'encoder_no_extract', evidence
        else:
            # Encoder ran AND wrote nodes, just not gold-bearing
            return 'encoder_partial', evidence

    # Gold fact IS in some node — diagnose the recall/surface side
    if not candidates:
        return 'no_query_signal', evidence

    ranks = [_candidate_rank(candidates, b['id']) for b in bearing]
    in_top = any(r is not None for r in ranks)

    if not in_top:
        # Fact-bearing node exists but didn't make it to candidates
        return 'ranker_buried', evidence

    # Fact-bearing node IS in top-N
    selected_ids = recall.get('selected', []) or []
    selected_short = set()
    for s in selected_ids:
        if isinstance(s, str):
            selected_short.add(s[:8])
        elif isinstance(s, dict) and s.get('id'):
            selected_short.add(s['id'][:8])
    bearing_in_selected = any(b['id'][:8] in selected_short for b in bearing)

    if not bearing_in_selected:
        # In candidates, NOT selected
        if evidence['recall_selected_count'] == 0:
            return 'surface_empty_ctx', evidence
        return 'surface_skipped', evidence

    # Bearing node WAS selected → context had it; answerer didn't use it
    answerer = recall.get('answerer_response') or {}
    if answerer.get('abstained'):
        return 'answerer_overcautious', evidence
    return 'answerer_other', evidence


# ─── markdown report ─────────────────────────────────────────────────

def _markdown(bundle: Dict[str, Any], refined: str,
              evidence: Dict[str, Any]) -> str:
    """Render a human-readable markdown report for a single failure."""
    meta = bundle.get('meta') or {}
    result = bundle.get('result') or {}
    recall = bundle.get('recall') or {}

    lines = []
    header = 'Pass-through' if refined == 'passed' else 'Failure analysis'
    lines.append(f"# {header} — `{meta.get('qid', '?')}`")
    lines.append('')
    lines.append(f"**Axis:** {meta.get('axis', '?')}")
    lines.append(f"**Original bucket:** `{evidence.get('original_bucket', '-')}`")
    lines.append(f"**Refined bucket:** `{refined}`")
    lines.append('')
    lines.append(f"**Question:** {meta.get('question', '')[:240]}")
    lines.append(f"**Gold:** {meta.get('gold', '')[:240]}")
    lines.append(f"**Hypothesis:** {result.get('hypothesis', '')[:300]}")
    lines.append('')

    # Pass-through items: short report. The encoder/recall sections require
    # failure-specific evidence keys that aren't populated for passed items.
    if refined == 'passed':
        lines.append('Item passed the judge. No failure analysis needed.')
        lines.append('')
        return '\n'.join(lines)

    if refined == 'artifacts_missing':
        lines.append(f"⚠ Artifact files missing: "
                     f"{', '.join(evidence.get('missing_files', []))} — item "
                     f"not measurable; check *.jsonl.error sidecars in the "
                     f"item dir.")
        lines.append('')
        return '\n'.join(lines)

    lines.append('## Encoder')
    lines.append(f"- Runs: {evidence['encoder_run_count']}, "
                 f"total actions: {evidence['encoder_total_actions']}")
    if evidence['encoder_journal_summary']:
        lines.append('- Encoder journal:')
        for j in evidence['encoder_journal_summary']:
            lines.append(f"  > {j}")
    lines.append('')

    lines.append('## Scouts')
    for s in evidence['scout_summary']:
        lines.append(f"- {s}")
    lines.append('')

    lines.append('## Encoded knowledge — gold-fact match')
    bearing = evidence.get('gold_bearing_nodes') or []
    if evidence.get('gold_scan_basis') == 'unscannable':
        lines.append('- ⚠ gold too short to scan — no encoder verdict '
                     'possible (not an encoder gap)')
    elif not bearing:
        if evidence.get('gold_scan_basis') == 'phrase':
            lines.append('- ❌ No node carries the gold phrase '
                         '(short gold — exact-substring scan)')
        else:
            lines.append(f"- ❌ No node carries all distinctive gold terms: "
                         f"`{evidence.get('gold_terms', [])}`")
    else:
        lines.append(f"- ✅ {len(bearing)} node(s) carry the gold terms:")
        for b in bearing:
            lines.append(f"  - `{b['id'][:8]}` [{b['type']}] from "
                         f"{b['encoding_source']}: {b['title'][:80]}")
    lines.append('')

    lines.append('## Recall')
    lines.append(f"- Candidates: {evidence['recall_candidate_count']}")
    lines.append(f"- Selected: {evidence['recall_selected_count']}")
    lines.append(f"- Context: {evidence['recall_context_chars']} chars")
    if 'fact_node_ranks' in evidence:
        lines.append(f"- Fact-node ranks in candidates: {evidence['fact_node_ranks']}")
    lines.append('')

    return '\n'.join(lines)


# ─── public API ───────────────────────────────────────────────────────

def analyze_failure(run_name: str, qid: str,
                    reports_root: Optional[str] = None) -> Dict[str, Any]:
    """Produce the deep-analysis report for one failure.

    Returns a dict with: bucket_refined, evidence, markdown.
    """
    bundle = load_artifacts(run_name, qid, reports_root=reports_root)
    refined, evidence = _refine_bucket(bundle)
    md = _markdown(bundle, refined, evidence)
    return {
        'qid': qid,
        'run_name': run_name,
        'bucket_refined': refined,
        'evidence': evidence,
        'markdown': md,
    }


def analyze_run(run_name: str, reports_root: Optional[str] = None
                ) -> Dict[str, Any]:
    """Run analyze_failure on every item in a run; aggregate by refined bucket."""
    from eval.longmem.artifacts import list_items
    qids = list_items(run_name, reports_root=reports_root)

    results = []
    by_refined: Dict[str, int] = {}
    bucket_drift: List[Dict[str, str]] = []

    for qid in qids:
        try:
            r = analyze_failure(run_name, qid, reports_root=reports_root)
        except Exception as e:
            results.append({'qid': qid, 'error': str(e)})
            continue
        results.append(r)
        rb = r['bucket_refined']
        by_refined[rb] = by_refined.get(rb, 0) + 1
        orig = r['evidence'].get('original_bucket')
        if orig and orig != rb:
            bucket_drift.append({'qid': qid, 'original': orig, 'refined': rb})

    return {
        'run_name': run_name,
        'item_count': len(qids),
        'by_refined_bucket': by_refined,
        'bucket_drift': bucket_drift,
        'items': results,
    }


def write_run_report(run_name: str, out_path: Optional[str] = None,
                     reports_root: Optional[str] = None) -> str:
    """Write a single-file markdown report for a run to disk.

    Returns the path written.
    """
    if reports_root is None:
        reports_root = str(Path(__file__).resolve().parent / 'reports')
    if out_path is None:
        out_path = str(Path(reports_root) / run_name / 'analysis.md')

    summary = analyze_run(run_name, reports_root=reports_root)

    lines: List[str] = []
    lines.append(f"# Deep failure analysis — {run_name}")
    lines.append('')
    lines.append(f"**Items analyzed:** {summary['item_count']}")
    lines.append('')
    lines.append('## Refined bucket distribution')
    lines.append('')
    for bucket, count in sorted(summary['by_refined_bucket'].items(),
                                key=lambda x: -x[1]):
        lines.append(f"- `{bucket}`: {count}")
    lines.append('')

    if summary['bucket_drift']:
        lines.append('## Bucket drift (original → refined)')
        lines.append('')
        lines.append('| qid | original | refined |')
        lines.append('|---|---|---|')
        for d in summary['bucket_drift']:
            lines.append(f"| `{d['qid']}` | `{d['original']}` | `{d['refined']}` |")
        lines.append('')

    lines.append('## Per-item analyses')
    lines.append('')
    for item in summary['items']:
        if 'error' in item:
            lines.append(f"### `{item['qid']}` — analysis failed")
            lines.append(f"```\n{item['error']}\n```\n")
        else:
            lines.append(item['markdown'])
            lines.append('---\n')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text('\n'.join(lines))
    return out_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('run_name', help='Run name (matches reports/{run_name})')
    p.add_argument('--qid', default=None, help='Just this qid (if set, prints to stdout)')
    p.add_argument('--out', default=None, help='Output path for run-level report')
    args = p.parse_args()

    if args.qid:
        r = analyze_failure(args.run_name, args.qid)
        print(r['markdown'])
        print()
        print(f"refined bucket: {r['bucket_refined']}")
    else:
        path = write_run_report(args.run_name, out_path=args.out)
        print(f"wrote {path}")
