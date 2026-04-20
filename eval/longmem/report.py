"""Friendly markdown report generator for LongMemEval runs.

Consumes the harness's scored JSON and emits a human-readable, actionable
report focused on WHERE TO INVEST to improve results.
"""
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


BUCKET_LABELS = {
    "ENCODE_MISS": "encode — the answer never made it into the brain",
    "RECALL_MISS": "recall — encoded, but didn't score into the top-25 candidates",
    "SURFACE_MISS": "surface — in candidates, but the surfacer passed it over",
    "PARTIAL_RECALL": "partial recall — context delivered but the specific fact wasn't in it",
    "ANSWER_MISS": "answer — the fact was in context, but the answerer didn't use it",
}

BUCKET_INVEST_PROMPT = {
    "ENCODE_MISS": (
        "Look at the S1E encoder — was the detail in the conversation timeline, "
        "or did the encoder skip the turn? If detail was present, tighten the "
        "prompt's attention to that question type. If absent, the encoder gate "
        "(every 5 turns) may be missing context — consider wider message windows."
    ),
    "RECALL_MISS": (
        "The node exists but scoring didn't rank it. Most likely: query terms "
        "don't match title/situation embeddings (embedding mismatch) or "
        "keyword fallback too weak. Check cosine scores in the candidates "
        "list — if the relevant node is near the boundary, tune z-weights."
    ),
    "SURFACE_MISS": (
        "The surfacer (Haiku) had the node in candidates and rejected it. "
        "That's a surfacer prompt / judgment issue. Review the surfacer "
        "interaction prompt for this axis — it's dropping signal it should keep."
    ),
    "PARTIAL_RECALL": (
        "The surfacer delivered adjacent/general nodes but not the one carrying "
        "the specific fact — either the specific fact wasn't encoded as its own "
        "node (encoder abstraction bias — tune S1E to keep specifics), OR the "
        "specific node exists but didn't score into the top candidates "
        "(recall scoring gap for fact-oriented queries)."
    ),
    "ANSWER_MISS": (
        "Context contained the fact, answerer still failed. Abstention threshold "
        "too aggressive OR context too noisy (too many neighbors diluting signal). "
        "Review the answerer prompt."
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

    graded = [r for r in report.get("results", []) if "correct" in r]
    total = len(graded)
    correct = sum(1 for r in graded if r["correct"])
    score = correct / total if total else 0

    by_axis = _axis_pass_rate(graded)
    failures = _failures_by_bucket(graded)
    investments = _investment_ranking(failures)

    lines: List[str] = []
    lines.append(f"# LongMemEval — {report.get('run_name', '?')}")
    lines.append("")
    lines.append(f"**Overall: {correct}/{total} = {score:.0%}** "
                 f"(wall clock {report.get('total_ms', 0)/1000:.1f}s)")
    lines.append("")

    # ── By axis ──
    lines.append("## By axis")
    lines.append("")
    lines.append("| Axis | Pass | Rate |")
    lines.append("|---|---|---|")
    for axis in sorted(by_axis):
        stats = by_axis[axis]
        rate = stats["correct"] / stats["total"] if stats["total"] else 0
        mark = "✓" if stats["correct"] == stats["total"] else ("✗" if stats["correct"] == 0 else "~")
        lines.append(f"| {mark} {axis} | {stats['correct']}/{stats['total']} | {rate:.0%} |")
    lines.append("")

    # ── Failures by bucket ──
    if failures:
        lines.append("## Where we're losing")
        lines.append("")
        for bucket in ("ENCODE_MISS", "RECALL_MISS", "SURFACE_MISS",
                       "PARTIAL_RECALL", "ANSWER_MISS", "UNCLASSIFIED"):
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
