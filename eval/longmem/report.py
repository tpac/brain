"""Friendly markdown report generator for LongMemEval runs.

Consumes the harness's scored JSON and emits a human-readable, actionable
report focused on WHERE TO INVEST to improve results.

Also renders a "Per-item breakdown" section so you can scan, per item:
the question, the gold answer, our hypothesis, whether the gold-bearing
fact was actually written to the brain (via analyzer's scan), and the
judge's verdict + comparison + reasoning. This is the answer to "are we
recalling the answer, or is the fact missing from the brain entirely?".
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


BUCKET_LABELS = {
    "ENCODE_MISS": "encode — gold fact not in any node (brain scan confirmed absent)",
    "RECALL_MISS": "recall — fact is in brain but didn't land in delivered context",
    "SURFACE_MISS": "surface — candidates existed, but surfacer returned nothing",
    "ANSWER_MISS": "answer — context had the fact, answerer didn't use it",
    # PARTIAL_RECALL kept for backward compat when reading older runs.
    "PARTIAL_RECALL": "partial recall (legacy) — context delivered but specific fact missing",
}

BUCKET_INVEST_PROMPT = {
    "ENCODE_MISS": (
        "The fact never reached the brain — brain scan finds no node carrying it. "
        "This is the scribe's job: either the encoder skipped the turn, or paraphrased "
        "the detail out, or the specific handle (number, name, phrase) was lost to "
        "abstraction. Scouts (facts/quote/temporal) target this directly."
    ),
    "RECALL_MISS": (
        "The fact is in the brain but didn't land in the context delivered to the "
        "answerer. Either zero candidates scored, the right node didn't rank top-8, "
        "or the context was dominated by adjacent-but-wrong nodes. Evidence.relevant_to_gold "
        "shows the semantic ranking; if the right node's score is low, tune "
        "embeddings/weights. If the score is high but wasn't selected, surfacer issue."
    ),
    "SURFACE_MISS": (
        "Candidates existed, but the surfacer returned zero selections or empty context. "
        "Narrow case — usually a surfacer prompt regression or an over-aggressive "
        "abstain path. Inspect the surfacer trace for this item."
    ),
    "ANSWER_MISS": (
        "Context contained the fact, answerer still failed. Abstention threshold "
        "too aggressive OR context too noisy (too many neighbors diluting signal). "
        "Review the answerer prompt."
    ),
    "PARTIAL_RECALL": (
        "Legacy bucket — re-run with current classifier to split into ENCODE_MISS "
        "(fact not in brain) or RECALL_MISS (in brain, didn't land in context)."
    ),
}


def _axis_pass_rate(graded: List[Dict]) -> Dict[str, Dict[str, Any]]:
    by_axis = defaultdict(lambda: {"correct": 0, "total": 0, "items": []})
    for r in graded:
        axis = r.get("axis", "unknown")
        by_axis[axis]["total"] += 1
        if r.get("correct"):
            by_axis[axis]["correct"] += 1
        by_axis[axis]["items"].append(r)
    return dict(by_axis)


def _failures_by_bucket(graded: List[Dict]) -> Dict[str, List[Dict]]:
    buckets = defaultdict(list)
    for r in graded:
        if r.get("correct"):
            continue
        b = r.get("failure_bucket", "UNCLASSIFIED")
        buckets[b].append(r)
    return dict(buckets)


def _render_item_line(r: Dict) -> str:
    qid = r.get("question_id", "?")[:12]
    axis = r.get("axis", "?")
    q = (r.get("question") or "")[:90]
    reason = (r.get("failure_reason") or "").strip()
    return f"- `{qid}` [{axis}] — {q}\n    → {reason}"


def _gold_in_brain_for_item(run_name: str, r: Dict,
                            reports_root: str) -> Optional[bool]:
    """Was the gold-bearing fact actually written into the brain for this item?

    Loads the per-item artifact bundle (nodes.jsonl) and runs the analyzer's
    deeper scan — looks across title/content/keywords + every relevant KV
    field. Returns True/False, or None if artifacts aren't available
    (e.g. old run without per-item dumps).
    """
    try:
        from eval.longmem.artifacts import load_artifacts
        from eval.longmem.analyzer import (_find_gold_bearing_nodes,
                                           _gold_scan_basis)
    except Exception:
        return None

    qid = r.get("question_id")
    vidx = r.get("variance_idx")
    if not qid:
        return None
    artifact_qid = qid if vidx is None else f"{qid}-r{vidx}"
    try:
        bundle = load_artifacts(run_name, artifact_qid, reports_root=reports_root)
    except Exception:
        return None
    nodes = bundle.get("nodes")
    if nodes is None:
        # File missing → artifacts unavailable. An EMPTY nodes.jsonl is a
        # real answer now that it holds the run delta: the encoder created
        # nothing, so the gold is definitionally not in the brain → False.
        return None
    gold = r.get("answer_gold") or ""
    if isinstance(gold, list):
        gold = ", ".join(str(x) for x in gold)
    if _gold_scan_basis(str(gold)) == 'unscannable':
        # Gold too short to search either way — "not measurable", not
        # "not in any node (encoder gap)".
        return None
    try:
        bearing = _find_gold_bearing_nodes(nodes, str(gold))
    except Exception:
        return None
    return bool(bearing)


def _truncate(s: Any, n: int) -> str:
    """Stringify and clip; turn None into '—' for readable cells."""
    if s is None:
        return "—"
    text = str(s) if not isinstance(s, str) else s
    text = text.replace("\n", " ").strip()
    return (text[:n] + "…") if len(text) > n else text


def _investment_ranking(buckets: Dict[str, List[Dict]]) -> List[Dict]:
    """Rank buckets by failure count; each bucket gets a specific recommendation."""
    ranked = []
    for bucket, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        axes = defaultdict(int)
        for r in items:
            axes[r.get("axis", "unknown")] += 1
        axis_breakdown = ", ".join(f"{a}×{n}" for a, n in sorted(axes.items(), key=lambda x: -x[1]))
        ranked.append({
            "bucket": bucket,
            "failure_count": len(items),
            "axes": axis_breakdown,
            "recommendation": BUCKET_INVEST_PROMPT.get(bucket, "(no recommendation for this bucket)"),
        })
    return ranked


def render_report(run_json_path: str) -> str:
    """Render a markdown report next to the JSON run report. Returns the .md path."""
    with open(run_json_path) as f:
        report = json.load(f)

    # Prefer judge-rewritten results when present (scored_*.json from
    # eval/longmem/judge.py grade_run). Falls back to inline-graded results
    # from the harness when re-rendering a run_*.json directly.
    source = report.get("graded_results") or report.get("results", [])
    graded = [r for r in source if "correct" in r]
    # Items that crashed before grading (harness writes {question_id, error}
    # dicts) — kept out of the score but rendered, so a 10-item run with 6
    # crashes can't read as "4/4 = 100%".
    crashed = [r for r in source if "correct" not in r]
    total = len(graded)
    correct = sum(1 for r in graded if r["correct"])
    score = correct / total if total else 0

    by_axis = _axis_pass_rate(graded)
    failures = _failures_by_bucket(graded)
    investments = _investment_ranking(failures)

    run_name = report.get("run_name") or ""
    reports_root = str(Path(run_json_path).resolve().parent)
    axis_stats = report.get("axis_stats") or {}
    per_qid_stats = report.get("per_qid_stats") or {}
    variance_n = (report.get("config") or {}).get("variance", 1) or 1

    # Always recompute comparison_counts from the graded source. The top-level
    # field may be stale (e.g. after re-grading a run with a different judge),
    # so trust the items, not the cached summary.
    comparison_counts: Dict[str, int] = {}
    for r in graded:
        cmp_name = r.get("comparison")
        if cmp_name:
            comparison_counts[cmp_name] = comparison_counts.get(cmp_name, 0) + 1

    lines: List[str] = []
    lines.append(f"# LongMemEval — {run_name or '?'}")
    lines.append("")
    lines.append(f"**Overall: {correct}/{total} = {score:.0%}** "
                 f"(wall clock {report.get('total_ms', 0)/1000:.1f}s"
                 f"{', variance=' + str(variance_n) if variance_n > 1 else ''})")
    if crashed:
        ids = ", ".join(str(r.get("question_id", "?")) for r in crashed[:10])
        lines.append("")
        lines.append(f"⚠ **{len(crashed)} item(s) crashed before grading** "
                     f"(excluded from the score): {ids}")
    lines.append("")

    # ── By axis ──
    lines.append("## By axis")
    lines.append("")
    has_stddev = any(s.get("stddev", 0) for s in axis_stats.values()) if axis_stats else False
    if has_stddev:
        lines.append("| Axis | Pass | Mean | Stddev | n |")
        lines.append("|---|---|---|---|---|")
        for axis in sorted(by_axis):
            stats = by_axis[axis]
            s = axis_stats.get(axis, {})
            rate = stats["correct"] / stats["total"] if stats["total"] else 0
            mark = "✓" if stats["correct"] == stats["total"] else ("✗" if stats["correct"] == 0 else "~")
            lines.append(f"| {mark} {axis} | {stats['correct']}/{stats['total']} | "
                         f"{s.get('mean', rate):.0%} | "
                         f"{s.get('stddev', 0):.2f} | "
                         f"{s.get('n', stats['total'])} |")
    else:
        lines.append("| Axis | Pass | Rate |")
        lines.append("|---|---|---|")
        for axis in sorted(by_axis):
            stats = by_axis[axis]
            rate = stats["correct"] / stats["total"] if stats["total"] else 0
            mark = "✓" if stats["correct"] == stats["total"] else ("✗" if stats["correct"] == 0 else "~")
            lines.append(f"| {mark} {axis} | {stats['correct']}/{stats['total']} | {rate:.0%} |")
    lines.append("")

    # ── Judge comparison distribution ──
    # Captures the "we sometimes do better than gold" signal from the new
    # judge — distinguish equivalent / hypothesis_better / partial / wrong /
    # gold_ambiguous instead of just YES/NO.
    if comparison_counts:
        lines.append("## Judge comparisons")
        lines.append("")
        lines.append("| Comparison | Count |")
        lines.append("|---|---|")
        for cmp_name, n in sorted(comparison_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{cmp_name}` | {n} |")
        lines.append("")

    # ── Per-qid variance (only when variance > 1) ──
    if per_qid_stats:
        lines.append("## Per-item variance")
        lines.append("")
        lines.append("Same item run multiple times — high stddev = flaky item.")
        lines.append("")
        lines.append("| qid | Mean | Stddev | n |")
        lines.append("|---|---|---|---|")
        for qid, s in sorted(per_qid_stats.items(), key=lambda kv: kv[1]["stddev"], reverse=True):
            lines.append(f"| `{qid}` | {s['mean']:.0%} | {s['stddev']:.2f} | {s['n']} |")
        lines.append("")

    # ── Failures by bucket ──
    if failures:
        lines.append("## Where we're losing")
        lines.append("")
        for bucket in ("ENCODE_MISS", "RECALL_MISS", "SURFACE_MISS",
                       "ANSWER_MISS", "PARTIAL_RECALL", "UNCLASSIFIED"):
            items = failures.get(bucket, [])
            if not items:
                continue
            lines.append(f"### {bucket} ({len(items)}) — {BUCKET_LABELS.get(bucket, bucket)}")
            lines.append("")
            for r in items:
                lines.append(_render_item_line(r))
            lines.append("")

    # ── Investment recommendations ──
    if investments:
        lines.append("## Where to invest (ranked by impact)")
        lines.append("")
        for i, inv in enumerate(investments, 1):
            lines.append(f"**{i}. {inv['bucket']}** — {inv['failure_count']} "
                         f"failure{'s' if inv['failure_count'] != 1 else ''} "
                         f"({inv['axes']})")
            lines.append("")
            lines.append(f"    {inv['recommendation']}")
            lines.append("")

    # ── Perf summary ──
    if graded:
        total_ingest = sum(r.get("ingest_ms", 0) for r in graded)
        total_s1r = sum(r.get("query_s1r_ms", 0) for r in graded)
        total_answer = sum(r.get("answer_ms", 0) for r in graded)
        lines.append("## Perf")
        lines.append("")
        lines.append(f"- Ingest total: {total_ingest/1000:.1f}s ({total_ingest/len(graded)/1000:.1f}s/item)")
        lines.append(f"- Query S1R:    {total_s1r/1000:.1f}s ({total_s1r/len(graded)/1000:.1f}s/item)")
        lines.append(f"- Answerer:     {total_answer/1000:.1f}s ({total_answer/len(graded)/1000:.1f}s/item)")
        lines.append("")

    # ── Per-item breakdown — Q vs Gold vs Hypothesis vs gold-in-brain ──
    # The answer to "are we recalling the answer, or is the fact missing
    # from the brain?". Loads each item's artifact bundle to ask the
    # analyzer if any node carries the gold-bearing terms.
    if graded:
        lines.append("## Per-item breakdown")
        lines.append("")
        lines.append("For each item: turns ingested, question, gold answer, "
                     "our hypothesis, whether the gold-bearing fact is in the "
                     "brain at all (analyzer scan of all nodes), and judge "
                     "reasoning. **Gold-in-brain ✗ + verdict ✗ = encoder gap. "
                     "Gold-in-brain ✓ + verdict ✗ = recall/surface/answer gap.**")
        lines.append("")

        # Group by qid so variance replicates render together
        by_qid_results: Dict[str, List[Dict]] = defaultdict(list)
        for r in graded:
            by_qid_results[r.get("question_id", "?")].append(r)

        for qid in sorted(by_qid_results.keys()):
            for r in by_qid_results[qid]:
                axis = r.get("axis", "?")
                ingest = r.get("ingest") or {}
                turns = ingest.get("user_turns") or ingest.get("turns") or "?"
                vidx = r.get("variance_idx")
                vlabel = f" r{vidx}" if vidx is not None else ""
                hdr = f"### `{qid}`{vlabel}  ax=`{axis}`  turns={turns}"
                lines.append(hdr)
                lines.append("")
                lines.append(f"- **Q:** {_truncate(r.get('question'), 240)}")
                lines.append(f"- **Gold:** {_truncate(r.get('answer_gold'), 240)}")
                verdict_mark = "✓" if r.get("correct") else "✗"
                cmp_name = r.get("comparison") or "—"
                lines.append(f"- **Ours ({verdict_mark} {cmp_name}):** "
                             f"{_truncate(r.get('hypothesis'), 300)}")
                gib = _gold_in_brain_for_item(run_name, r, reports_root)
                if gib is True:
                    gib_mark = "✓ encoded"
                elif gib is False:
                    gib_mark = "✗ NOT in any node (encoder gap)"
                else:
                    gib_mark = "? (no artifacts)"
                lines.append(f"- **Gold in brain:** {gib_mark}")
                reasoning = r.get("judge_reasoning") or r.get("failure_reason") or ""
                if reasoning:
                    lines.append(f"- **Why:** {_truncate(reasoning, 280)}")
                lines.append("")

    md_text = "\n".join(lines) + "\n"
    md_path = run_json_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(md_text)
    return md_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="path to run_<name>.json from harness.py")
    args = parser.parse_args()
    print(render_report(args.report))
