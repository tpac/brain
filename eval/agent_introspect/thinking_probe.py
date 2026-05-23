"""Thinking-budget probe — compare encoder runtime configs head-to-head.

Same prompt, same conversation, multiple runtime configurations:

  bump_8k          — max_tokens=8000, no extended thinking
  bump_10k         — max_tokens=10000, no extended thinking
  thinking_4k      — max_tokens=12000 with budget_tokens=4000 (extended thinking)
  thinking_6k      — max_tokens=14000 with budget_tokens=6000

For each (config, qid, trial), captures:
  - input/cache/output tokens (Anthropic accounting)
  - thinking-block text length (sanity-check the thinking budget)
  - elapsed wall-time
  - tool_use count + nodes emitted (does the model get to act?)
  - stop_reason
  - estimated USD cost (Anthropic Sonnet 4.6 pricing)

Used to answer: "Does extended thinking buy us cleaner reasoning at the
same or lower cost than bumping max_tokens?"

USE
    ./dev python3 -m eval.agent_introspect.thinking_probe \\
        --qids 09ba9854_abs,54026fce \\
        --prompt /tmp/s1e_v18_candidate.md \\
        --trials 3 \\
        --out eval/longmem/reports/thinking_probe_v18.md

NOTE: each call faithfully replicates the encoder's runtime config — same
cache_control breakpoints (system+tools 1h, user_content 5m), same tool
definitions. The only thing varying between configs is max_tokens and
the `thinking` parameter.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import load_env, write_report, write_json  # noqa: E402
from eval.agent_introspect.encoder_replay import (  # noqa: E402
    _build_user_content, _load_conversation, _run_scouts,
)


# Anthropic Sonnet 4.6 pricing (per million tokens, USD).
PRICING = {
    "input": 3.0,
    "cache_create": 3.75,    # 1.25× input
    "cache_read": 0.30,      # 0.1× input
    "output": 15.0,          # also charges for thinking tokens
}


# Named runtime configurations. Add new ones here; the probe will run them.
DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {"label": "bump_8k",      "max_tokens": 8000,  "thinking_budget": 0},
    {"label": "bump_10k",     "max_tokens": 10000, "thinking_budget": 0},
    {"label": "thinking_4k",  "max_tokens": 12000, "thinking_budget": 4000},
    {"label": "thinking_6k",  "max_tokens": 14000, "thinking_budget": 6000},
]


def _estimate_cost(usage: Dict[str, int]) -> float:
    """Anthropic cost estimate from a usage dict."""
    inp = usage.get("input_tokens", 0) or 0
    cc = usage.get("cache_creation_input_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    return (inp / 1e6 * PRICING["input"]
            + cc / 1e6 * PRICING["cache_create"]
            + cr / 1e6 * PRICING["cache_read"]
            + out / 1e6 * PRICING["output"])


def _count_outputs(content) -> Dict[str, int]:
    """Inspect the response content. Returns counts of {tool_use, nodes_emitted,
    thinking_text_chars, text_chars}."""
    tu_n = 0
    nodes_n = 0
    thinking_chars = 0
    text_chars = 0
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            tu_n += 1
            if block.name == "remember_batch":
                nodes_n += len((block.input or {}).get("nodes", []))
        elif btype == "thinking" and hasattr(block, "thinking"):
            thinking_chars += len(block.thinking)
        elif btype == "text" and hasattr(block, "text"):
            text_chars += len(block.text)
    return {
        "tool_uses": tu_n,
        "nodes_emitted": nodes_n,
        "thinking_text_chars": thinking_chars,
        "text_chars": text_chars,
    }


def _one_call(client, system_prompt: str, user_content: str, tools: List[Dict],
              cfg: Dict[str, Any], model: str) -> Dict[str, Any]:
    """One Sonnet call under the given runtime config. Returns full result row."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": cfg["max_tokens"],
        "system": [{"type": "text", "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": user_content,
             "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        ]}],
        "tools": tools,
    }
    if cfg.get("thinking_budget", 0):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": cfg["thinking_budget"],
        }
        # Extended thinking with tool_use: temperature MUST be 1.0 (the API
        # rejects other values for safety reasons documented in the
        # extended-thinking guide).
        kwargs["temperature"] = 1.0

    t0 = time.time()
    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:
        return {
            "label": cfg["label"], "error": f"{type(e).__name__}: {e}",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    elapsed_ms = int((time.time() - t0) * 1000)
    counts = _count_outputs(resp.content)
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(
            resp.usage, "cache_read_input_tokens", 0) or 0,
    }
    cost = _estimate_cost(usage)
    return {
        "label": cfg["label"],
        "max_tokens": cfg["max_tokens"],
        "thinking_budget": cfg.get("thinking_budget", 0),
        "elapsed_ms": elapsed_ms,
        "stop_reason": resp.stop_reason,
        **usage,
        **counts,
        "cost_usd": cost,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean / stddev / counts across a slice of rows."""
    if not rows:
        return {}
    n = len(rows)
    def _m(key: str) -> float:
        vals = [r.get(key, 0) or 0 for r in rows if "error" not in r]
        return sum(vals) / len(vals) if vals else 0.0
    def _sd(key: str) -> float:
        vals = [r.get(key, 0) or 0 for r in rows if "error" not in r]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return {
        "trials": n,
        "errors": sum(1 for r in rows if "error" in r),
        "mean_elapsed_s": _m("elapsed_ms") / 1000,
        "sd_elapsed_s": _sd("elapsed_ms") / 1000,
        "mean_input": _m("input_tokens"),
        "mean_cache_create": _m("cache_creation_input_tokens"),
        "mean_cache_read": _m("cache_read_input_tokens"),
        "mean_output": _m("output_tokens"),
        "mean_thinking_chars": _m("thinking_text_chars"),
        "mean_tool_uses": _m("tool_uses"),
        "mean_nodes": _m("nodes_emitted"),
        "sd_nodes": _sd("nodes_emitted"),
        "mean_cost_usd": _m("cost_usd"),
        "stop_reasons": sorted({r.get("stop_reason", "?") for r in rows
                                if "error" not in r}),
    }


def _format_report(prompt_path: str, configs: List[Dict[str, Any]],
                   qids: List[str], rows: List[Dict[str, Any]]) -> str:
    """Markdown comparison report."""
    out: List[str] = []
    out.append(f"# Thinking probe — `{prompt_path}`\n")
    out.append(f"Configs tested: {[c['label'] for c in configs]}\n")
    out.append(f"qids: {qids} · trials per config: "
               f"{max(1, len(rows) // (len(configs) * len(qids)))}\n")
    out.append("")

    out.append("## Aggregate per config (across all qids/trials)\n")
    out.append("| config | mean_s | mean_in | cache_cr | cache_rd | mean_out "
               "| thinking_ch | mean_nodes (sd) | mean_cost | stop_reasons |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for cfg in configs:
        cfg_rows = [r for r in rows if r["label"] == cfg["label"]]
        agg = _aggregate(cfg_rows)
        if not agg:
            out.append(f"| {cfg['label']} | (no data) |")
            continue
        out.append(
            f"| `{cfg['label']}` "
            f"| {agg['mean_elapsed_s']:.1f}±{agg['sd_elapsed_s']:.1f} "
            f"| {agg['mean_input']:.0f} "
            f"| {agg['mean_cache_create']:.0f} "
            f"| {agg['mean_cache_read']:.0f} "
            f"| {agg['mean_output']:.0f} "
            f"| {agg['mean_thinking_chars']:.0f} "
            f"| {agg['mean_nodes']:.1f}±{agg['sd_nodes']:.1f} "
            f"| ${agg['mean_cost_usd']:.4f} "
            f"| {','.join(agg['stop_reasons'])} |"
        )
    out.append("")

    out.append("## Per-qid breakdown\n")
    for qid in qids:
        out.append(f"### `{qid}`\n")
        out.append("| config | s | in | cc | cr | out | think | nodes | $ | stop |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for cfg in configs:
            qid_cfg_rows = [r for r in rows
                            if r["label"] == cfg["label"] and r.get("qid") == qid]
            for r in qid_cfg_rows:
                if "error" in r:
                    out.append(f"| `{cfg['label']}` | — | — | — | — | — | — | — | — | ERR: {r['error'][:40]} |")
                    continue
                out.append(
                    f"| `{cfg['label']}` "
                    f"| {r['elapsed_ms']/1000:.1f} "
                    f"| {r['input_tokens']} "
                    f"| {r['cache_creation_input_tokens']} "
                    f"| {r['cache_read_input_tokens']} "
                    f"| {r['output_tokens']} "
                    f"| {r['thinking_text_chars']} "
                    f"| {r['nodes_emitted']} "
                    f"| ${r['cost_usd']:.4f} "
                    f"| {r['stop_reason']} |"
                )
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qids", required=True,
                   help="comma-separated qids")
    p.add_argument("--prompt", required=True,
                   help="path to s1e prompt file (system prompt under test)")
    p.add_argument("--trials", type=int, default=3,
                   help="trials per (config, qid)")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out", default=None,
                   help="markdown output path (also writes .json next to it)")
    p.add_argument("--configs-json", default=None,
                   help="optional JSON file overriding the default configs")
    args = p.parse_args()

    load_env()

    # Load configs
    if args.configs_json:
        configs = json.loads(Path(args.configs_json).read_text())
    else:
        configs = DEFAULT_CONFIGS

    qids = [q.strip() for q in args.qids.split(",") if q.strip()]

    # Build fresh brain (for scouts)
    tmpdir = tempfile.mkdtemp(prefix="thinking_probe_")
    os.environ["BRAIN_DB_DIR"] = tmpdir
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    brain = create_fresh_eval_brain(path=tmpdir, wipe=True)

    # Per-qid: build user_content once (scouts are deterministic, same across configs+trials)
    print(f"[thinking_probe] building per-qid user_content via scouts...", flush=True)
    per_qid: Dict[str, str] = {}
    for qid in qids:
        conv = _load_conversation(qid)
        scout = _run_scouts(brain, conv["turns"], conv["conversation_now"])
        user_content = _build_user_content(conv, scout["report"])
        per_qid[qid] = user_content
        print(f"  {qid}: user_content {len(user_content)} chars", flush=True)

    system_prompt = Path(args.prompt).read_text()
    print(f"[thinking_probe] system_prompt {len(system_prompt)} chars from "
          f"{args.prompt}", flush=True)

    from servers.scales.s1.encode import _get_tool_schemas
    tools = _get_tool_schemas()

    import anthropic
    client = anthropic.Anthropic()

    # Build the task list
    tasks = []
    for cfg in configs:
        for qid in qids:
            for trial in range(args.trials):
                tasks.append((cfg, qid, trial))

    print(f"[thinking_probe] running {len(tasks)} tasks "
          f"({len(configs)} configs × {len(qids)} qids × {args.trials} trials) "
          f"with parallel={args.parallel}", flush=True)

    rows: List[Dict[str, Any]] = []
    def _do(cfg, qid, trial):
        r = _one_call(client, system_prompt, per_qid[qid], tools, cfg, args.model)
        r["qid"] = qid
        r["trial"] = trial
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(_do, *t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            r = fut.result()
            if "error" in r:
                print(f"  [{i+1}/{len(tasks)}] {r['label']:<14} qid={r['qid']:<14} "
                      f"t={r['trial']} ERR: {r['error'][:60]}", flush=True)
            else:
                print(f"  [{i+1}/{len(tasks)}] {r['label']:<14} qid={r['qid']:<14} "
                      f"t={r['trial']} "
                      f"{r['elapsed_ms']/1000:>5.1f}s "
                      f"in={r['input_tokens']:>5} "
                      f"out={r['output_tokens']:>4} "
                      f"think={r['thinking_text_chars']:>5}ch "
                      f"nodes={r['nodes_emitted']:>2} "
                      f"${r['cost_usd']:.4f} "
                      f"stop={r['stop_reason']}",
                      flush=True)
            rows.append(r)

    rows.sort(key=lambda r: (r.get("qid", ""), r["label"], r["trial"]))

    # Final summary
    print()
    print(f"{'='*100}")
    print(f"{'config':<14} {'mean_s':>8} {'in':>6} {'cc':>5} {'cr':>5} "
          f"{'out':>5} {'think':>6} {'nodes':>10} {'cost':>9} stop_reasons")
    print(f"{'-'*100}")
    for cfg in configs:
        agg = _aggregate([r for r in rows if r["label"] == cfg["label"]])
        if not agg:
            continue
        print(f"{cfg['label']:<14} "
              f"{agg['mean_elapsed_s']:>5.1f}±{agg['sd_elapsed_s']:>2.1f}s "
              f"{agg['mean_input']:>6.0f} "
              f"{agg['mean_cache_create']:>5.0f} "
              f"{agg['mean_cache_read']:>5.0f} "
              f"{agg['mean_output']:>5.0f} "
              f"{agg['mean_thinking_chars']:>6.0f} "
              f"{agg['mean_nodes']:>5.1f}±{agg['sd_nodes']:>3.1f} "
              f"${agg['mean_cost_usd']:>7.4f} "
              f"{','.join(agg['stop_reasons'])}")

    # Render
    if args.out:
        md = _format_report(args.prompt, configs, qids, rows)
        write_report(Path(args.out), md)
        write_json(Path(args.out).with_suffix(".json"),
                   {"prompt": args.prompt, "configs": configs, "qids": qids,
                    "rows": rows})


if __name__ == "__main__":
    main()
