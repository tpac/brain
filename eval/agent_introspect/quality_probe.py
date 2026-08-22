"""Quality probe — compare what v17 vs v18 actually encode.

Same cohort, same scouts, same model, same runtime config — only the
system prompt varies. For each (prompt, qid, trial) we run the encoder
once via `replay_one` and walk the resulting tool_use blocks to compute
structural quality dimensions:

  Nodes:
    - count, mean content length, mean situation length
    - type distribution (which types Sonnet reaches for)
    - field coverage: situation, reasoning, their_raw_quote, my_raw_quote,
      keywords, event_time

  Provenance (Fix 2 target):
    - source_attribution coverage (% of nodes flagged)
    - anchor_unconfirmed count
    - open-type node count (uncertainty preserved as a first-class node)

  Edges:
    - total count, edges per node
    - relation distribution
    - correction-aspect relations (corrects, supersedes, reframes, resolves,
      reframed_by, addresses, ...) — Fix 1 indirectly should increase these
      via better atom separation
    - `why` field non-empty coverage

  Operator-revealing types (Fix 1 target):
    - personal_context, active_thread, preference, situation count
    - % of nodes that are operator-revealing

The aggregate report shows per-prompt means and (where useful) per-qid
splits, so you can see WHERE the prompt change moves the metric.

USE
    ./dev python3 -m eval.agent_introspect.quality_probe \\
        --qids 09ba9854_abs,54026fce,2311e44b,71017276,cc5ded98 \\
        --prompts v17=/tmp/s1e_v17.md,v18=/tmp/s1e_v18_candidate.md \\
        --trials 3 \\
        --out eval/longmem/reports/quality_probe_v17_vs_v18.md
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import load_env, write_report, write_json  # noqa: E402
from eval.agent_introspect.encoder_replay import replay_one  # noqa: E402


# Aspects: relations that indicate correction. Source-of-truth in
# servers/scales/s2/aspects_v1.json under the correction_improvement aspect.
# Mirrored here for the probe so it doesn't depend on a live brain.
CORRECTION_RELATIONS: Set[str] = {
    "corrects", "supersedes", "reframes", "resolves", "addresses",
    "fixes", "refines", "clarifies", "updates", "replaces", "amends",
    "revisits", "reinterprets", "contradicts", "rebuts", "challenges",
    "evolves", "matures", "deprecates", "obsoletes",
    "reframed_by", "updated_by",
}

# Operator-revealing node types added/emphasized by Fix 1.
OPERATOR_REVEALING_TYPES: Set[str] = {
    "personal_context", "active_thread", "preference",
    "situation", "interest",
}

# Voice fields that anchor a node to verbatim source content.
VOICE_FIELDS = {"their_raw_quote", "my_raw_quote"}


def _extract_nodes(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull all `remember_batch` nodes from action list."""
    out = []
    for a in actions:
        if a.get("tool") == "remember_batch":
            for n in (a.get("input") or {}).get("nodes") or []:
                out.append(n)
    return out


def _extract_edges(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull all edges from remember_batch (connect_to within nodes) AND
    connect_batch (top-level connections). Returns list of
    {relation, why, source_inferred, target_inferred}."""
    out = []
    for a in actions:
        if a.get("tool") == "remember_batch":
            for n in (a.get("input") or {}).get("nodes") or []:
                title = (n.get("title") or "")[:40]
                for c in n.get("connect_to") or []:
                    out.append({
                        "relation": (c.get("relation") or "").strip(),
                        "why": (c.get("why") or "").strip(),
                        "source": title,
                        "target": (c.get("target_id") or
                                   c.get("target_title") or "?")[:40],
                    })
        elif a.get("tool") == "connect_batch":
            for c in (a.get("input") or {}).get("connections") or []:
                out.append({
                    "relation": (c.get("relation") or "").strip(),
                    "why": (c.get("why") or "").strip(),
                    "source": (c.get("source_id") or
                               c.get("source") or "?")[:40],
                    "target": (c.get("target_id") or
                               c.get("target") or "?")[:40],
                })
    return out


def _analyze_one_run(r: Dict[str, Any]) -> Dict[str, Any]:
    """Compute structural quality metrics from one replay_one() result."""
    actions = r.get("actions", [])
    nodes = _extract_nodes(actions)
    edges = _extract_edges(actions)

    # Type distribution
    types = Counter(n.get("type") or "?" for n in nodes)
    op_revealing = sum(1 for n in nodes
                       if (n.get("type") or "") in OPERATOR_REVEALING_TYPES)
    open_nodes = sum(1 for n in nodes if (n.get("type") or "") == "open")

    # Field coverage on nodes
    fields = {
        "situation": sum(1 for n in nodes if (n.get("situation") or "").strip()),
        "reasoning": sum(1 for n in nodes if (n.get("reasoning") or "").strip()),
        "their_raw_quote": sum(1 for n in nodes
                              if (n.get("their_raw_quote") or "").strip()),
        "my_raw_quote": sum(1 for n in nodes
                                if (n.get("my_raw_quote") or "").strip()),
        "keywords": sum(1 for n in nodes
                        if (n.get("keywords") or "").strip()),
        "event_time": sum(1 for n in nodes
                          if (n.get("event_time") or "").strip()),
    }

    # Provenance: source_attribution and anchor_unconfirmed flagging
    src_attr = sum(1 for n in nodes
                   if (n.get("source_attribution") or "").strip())
    anchor_unconf = sum(1 for n in nodes
                        if "anchor_unconfirmed" in
                        str(n.get("source_attribution") or ""))

    # Content depth
    content_lens = [len(n.get("content") or "") for n in nodes]
    situation_lens = [len(n.get("situation") or "") for n in nodes]

    # Edges
    rel_counts = Counter(e["relation"] or "?" for e in edges)
    correction_edges = sum(1 for e in edges
                           if e["relation"] in CORRECTION_RELATIONS)
    why_nonempty = sum(1 for e in edges if e["why"])

    return {
        # Aggregate
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "edges_per_node": (len(edges) / len(nodes)) if nodes else 0,
        # Types
        "type_distribution": dict(types),
        "unique_types": len(types),
        "operator_revealing_count": op_revealing,
        "operator_revealing_pct": (op_revealing / len(nodes) * 100) if nodes else 0,
        "open_nodes": open_nodes,
        # Field coverage (counts)
        "field_coverage_counts": fields,
        "voice_coverage_pct": (
            (fields["their_raw_quote"] + fields["my_raw_quote"]) /
            (len(nodes) * 2) * 100) if nodes else 0,
        "situation_coverage_pct": (
            (fields["situation"] / len(nodes) * 100) if nodes else 0),
        "reasoning_coverage_pct": (
            (fields["reasoning"] / len(nodes) * 100) if nodes else 0),
        # Provenance
        "source_attribution_count": src_attr,
        "anchor_unconfirmed_count": anchor_unconf,
        "provenance_pct": (src_attr / len(nodes) * 100) if nodes else 0,
        # Content depth
        "mean_content_len": (sum(content_lens) / len(content_lens)) if content_lens else 0,
        "mean_situation_len": (sum(situation_lens) / len(situation_lens)) if situation_lens else 0,
        # Edges
        "relation_distribution": dict(rel_counts),
        "correction_edges": correction_edges,
        "correction_edges_pct": (correction_edges / len(edges) * 100) if edges else 0,
        "edge_why_coverage_pct": (why_nonempty / len(edges) * 100) if edges else 0,
        # Raw token info from replay_one
        "stop_reason": r.get("stop_reason"),
        "tokens_in": r.get("tokens_in"),
        "tokens_out": r.get("tokens_out"),
        "call_ms": r.get("call_ms"),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/sd across a slice of analyzed runs."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0
    def _sd(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    # Merge type distributions
    merged_types: Counter = Counter()
    merged_rels: Counter = Counter()
    for r in rows:
        merged_types.update(r.get("type_distribution") or {})
        merged_rels.update(r.get("relation_distribution") or {})

    out = {"n": n}
    for k in ("nodes_count", "edges_count", "edges_per_node", "unique_types",
              "operator_revealing_count", "operator_revealing_pct", "open_nodes",
              "voice_coverage_pct", "situation_coverage_pct",
              "reasoning_coverage_pct", "source_attribution_count",
              "anchor_unconfirmed_count", "provenance_pct",
              "mean_content_len", "mean_situation_len", "correction_edges",
              "correction_edges_pct", "edge_why_coverage_pct",
              "tokens_in", "tokens_out", "call_ms"):
        out[f"mean_{k}"] = _mean(k)
        out[f"sd_{k}"] = _sd(k)
    out["type_distribution"] = dict(merged_types)
    out["relation_distribution"] = dict(merged_rels)
    out["stop_reasons"] = sorted({r.get("stop_reason", "?") for r in rows})
    return out


def _format_report(prompts: Dict[str, str], qids: List[str],
                   trials: int, rows: List[Dict[str, Any]]) -> str:
    """Side-by-side markdown report."""
    out: List[str] = []
    out.append("# Quality probe — prompt comparison\n")
    out.append(f"Prompts: {list(prompts.keys())}\n")
    out.append(f"qids: {qids} · trials per (prompt, qid): {trials}\n")
    out.append("")

    # ── Section 1: aggregate per prompt (across all qids/trials) ──────
    out.append("## Aggregate per prompt — all qids × all trials\n")
    out.append("| metric | " + " | ".join(prompts.keys()) + " |")
    out.append("|---|" + "|".join("---:" for _ in prompts) + "|")
    metric_keys = [
        ("mean_nodes_count", "nodes/run", "{:.1f}"),
        ("mean_edges_count", "edges/run", "{:.1f}"),
        ("mean_edges_per_node", "edges/node", "{:.2f}"),
        ("mean_unique_types", "unique types/run", "{:.1f}"),
        ("mean_operator_revealing_count", "operator-revealing nodes", "{:.1f}"),
        ("mean_operator_revealing_pct", "% operator-revealing", "{:.0f}%"),
        ("mean_open_nodes", "open nodes (uncertainty)", "{:.1f}"),
        ("mean_source_attribution_count", "src_attribution flagged nodes", "{:.1f}"),
        ("mean_anchor_unconfirmed_count", "anchor_unconfirmed nodes", "{:.1f}"),
        ("mean_provenance_pct", "% nodes w/ src_attribution", "{:.0f}%"),
        ("mean_voice_coverage_pct", "% voice (user+anchor) covered", "{:.0f}%"),
        ("mean_situation_coverage_pct", "% nodes w/ situation", "{:.0f}%"),
        ("mean_reasoning_coverage_pct", "% nodes w/ reasoning", "{:.0f}%"),
        ("mean_correction_edges", "correction-aspect edges/run", "{:.1f}"),
        ("mean_correction_edges_pct", "% edges that are correction-aspect", "{:.0f}%"),
        ("mean_edge_why_coverage_pct", "% edges w/ non-empty why", "{:.0f}%"),
        ("mean_mean_content_len", "mean content len (chars)", "{:.0f}"),
        ("mean_mean_situation_len", "mean situation len (chars)", "{:.0f}"),
        ("mean_call_ms", "mean wall-time (ms)", "{:.0f}"),
        ("mean_tokens_out", "mean output tokens", "{:.0f}"),
    ]
    agg_by_prompt = {p: _aggregate([r for r in rows if r["prompt"] == p])
                     for p in prompts}
    for key, label, fmt in metric_keys:
        cells = [fmt.format(agg_by_prompt[p].get(key, 0)) for p in prompts]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")

    # ── Section 2: type distributions ──────────────────────────────────
    out.append("## Type distribution (counts across all trials)\n")
    all_types = sorted(set().union(*[set(agg_by_prompt[p].get("type_distribution") or {})
                                     for p in prompts]))
    out.append("| type | " + " | ".join(prompts.keys()) + " |")
    out.append("|---|" + "|".join("---:" for _ in prompts) + "|")
    for t in all_types:
        cells = [str(agg_by_prompt[p].get("type_distribution", {}).get(t, 0))
                 for p in prompts]
        marker = " 🆕" if t in OPERATOR_REVEALING_TYPES else ""
        out.append(f"| `{t}`{marker} | " + " | ".join(cells) + " |")
    out.append("")

    # ── Section 3: relation distributions ─────────────────────────────
    out.append("## Edge-relation distribution (counts across all trials)\n")
    all_rels = sorted(set().union(*[set(agg_by_prompt[p].get("relation_distribution") or {})
                                    for p in prompts]))
    out.append("| relation | " + " | ".join(prompts.keys()) + " |")
    out.append("|---|" + "|".join("---:" for _ in prompts) + "|")
    for rel in all_rels:
        cells = [str(agg_by_prompt[p].get("relation_distribution", {}).get(rel, 0))
                 for p in prompts]
        marker = " 🔧" if rel in CORRECTION_RELATIONS else ""
        out.append(f"| `{rel}`{marker} | " + " | ".join(cells) + " |")
    out.append("")

    # ── Section 4: per-qid mean nodes & flags ─────────────────────────
    out.append("## Per-qid means (nodes, op-revealing, provenance flags)\n")
    out.append("| qid | metric | " + " | ".join(prompts.keys()) + " |")
    out.append("|---|---|" + "|".join("---:" for _ in prompts) + "|")
    for qid in qids:
        for metric_key, label, fmt in [
            ("mean_nodes_count", "nodes", "{:.1f}"),
            ("mean_operator_revealing_count", "operator-revealing", "{:.1f}"),
            ("mean_source_attribution_count", "src_attr flagged", "{:.1f}"),
            ("mean_open_nodes", "open (uncertainty)", "{:.1f}"),
        ]:
            cells = []
            for p in prompts:
                qid_rows = [r for r in rows
                            if r["prompt"] == p and r.get("qid") == qid]
                agg = _aggregate(qid_rows)
                cells.append(fmt.format(agg.get(metric_key, 0)))
            out.append(f"| `{qid}` | {label} | " + " | ".join(cells) + " |")
    out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qids", required=True, help="comma-separated qids")
    p.add_argument("--prompts", required=True,
                   help="comma-separated label=path entries, e.g. v17=/tmp/s1e_v17.md,v18=/tmp/s1e_v18.md")
    p.add_argument("--trials", type=int, default=3,
                   help="trials per (prompt, qid)")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="encoder max_tokens (default 8000 — matches the bump decision)")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out", default=None,
                   help="markdown output path (also writes .json next to it)")
    args = p.parse_args()

    load_env()

    # Parse --prompts label=path,label=path
    prompts: Dict[str, str] = {}
    for chunk in args.prompts.split(","):
        if "=" not in chunk:
            raise ValueError(f"--prompts entry missing '=': {chunk}")
        label, path = chunk.split("=", 1)
        prompts[label.strip()] = Path(path.strip()).read_text()

    qids = [q.strip() for q in args.qids.split(",") if q.strip()]

    # Fresh brain for scout dispatch
    tmpdir = tempfile.mkdtemp(prefix="quality_probe_")
    os.environ["BRAIN_DB_DIR"] = tmpdir
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    brain = create_fresh_eval_brain(path=tmpdir, wipe=True)

    # Build task list
    tasks: List[Tuple[str, str, int]] = []
    for prompt_label in prompts:
        for qid in qids:
            for trial in range(args.trials):
                tasks.append((prompt_label, qid, trial))

    print(f"[quality_probe] running {len(tasks)} tasks "
          f"({len(prompts)} prompts × {len(qids)} qids × {args.trials} trials) "
          f"with parallel={args.parallel}, max_tokens={args.max_tokens}",
          flush=True)

    rows: List[Dict[str, Any]] = []

    def _do(prompt_label, qid, trial):
        try:
            r = replay_one(brain, qid, prompts[prompt_label],
                           model=args.model, max_tokens=args.max_tokens)
        except Exception as e:
            return {"prompt": prompt_label, "qid": qid, "trial": trial,
                    "error": f"{type(e).__name__}: {e}"}
        analyzed = _analyze_one_run(r)
        analyzed["prompt"] = prompt_label
        analyzed["qid"] = qid
        analyzed["trial"] = trial
        return analyzed

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(_do, *t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            r = fut.result()
            if "error" in r:
                print(f"  [{i+1}/{len(tasks)}] {r['prompt']:<6} qid={r['qid']:<14} "
                      f"t={r['trial']} ERR: {r['error'][:60]}", flush=True)
            else:
                print(f"  [{i+1}/{len(tasks)}] {r['prompt']:<6} qid={r['qid']:<14} "
                      f"t={r['trial']} nodes={r['nodes_count']:>2} "
                      f"edges={r['edges_count']:>2} "
                      f"op_rev={r['operator_revealing_count']:>2} "
                      f"src_attr={r['source_attribution_count']:>2} "
                      f"open={r['open_nodes']:>2} "
                      f"corr_edge={r['correction_edges']:>2}", flush=True)
            rows.append(r)

    rows.sort(key=lambda r: (r["prompt"], r.get("qid", ""), r["trial"]))

    # Print final aggregate
    print("\n" + "=" * 90)
    print(f"{'prompt':<8} {'nodes':>7} {'edges':>7} {'op_rev':>8} {'src_attr':>9} "
          f"{'open':>5} {'corr_e':>7} {'voice%':>8} {'sit%':>7} {'reason%':>9}")
    print("-" * 90)
    for prompt_label in prompts:
        agg = _aggregate([r for r in rows if r["prompt"] == prompt_label])
        if not agg or agg.get("n") == 0:
            continue
        print(f"{prompt_label:<8} "
              f"{agg['mean_nodes_count']:>4.1f}±{agg['sd_nodes_count']:>2.1f} "
              f"{agg['mean_edges_count']:>4.1f}±{agg['sd_edges_count']:>2.1f} "
              f"{agg['mean_operator_revealing_count']:>5.1f}±{agg['sd_operator_revealing_count']:>2.1f} "
              f"{agg['mean_source_attribution_count']:>6.1f}±{agg['sd_source_attribution_count']:>2.1f} "
              f"{agg['mean_open_nodes']:>5.1f} "
              f"{agg['mean_correction_edges']:>5.1f}±{agg['sd_correction_edges']:>1.1f} "
              f"{agg['mean_voice_coverage_pct']:>5.0f}%   "
              f"{agg['mean_situation_coverage_pct']:>4.0f}%  "
              f"{agg['mean_reasoning_coverage_pct']:>6.0f}%")

    if args.out:
        md = _format_report(prompts, qids, args.trials, rows)
        write_report(Path(args.out), md)
        write_json(Path(args.out).with_suffix(".json"),
                   {"prompts": list(prompts.keys()), "qids": qids,
                    "trials": args.trials, "rows": rows})


if __name__ == "__main__":
    main()
