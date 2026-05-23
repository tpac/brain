"""Content-level diff between two encoder prompts on a single qid.

Answers content questions the structural quality_probe can't:
  1. Did we lose facts? — Side-by-side list of node titles + content.
  2. Are ranges preserved as ranges? — Searches node content for fuzzy/range
     patterns ("around X", "X-Y", "X to Y") that came from the conversation,
     and checks whether the encoded value matches or collapses to a point.
  3. Are we revising? — Counts revise_batch + connect_batch calls and lists
     the revision targets / edge relations.

Single qid per run. For broader cohort coverage, use quality_probe.

USE
    ./dev python3 -m eval.agent_introspect.content_diff \\
        --qid 09ba9854_abs \\
        --prompts v17=/tmp/s1e_v17.md,v18=/tmp/s1e_v18_candidate.md \\
        --trials 2 \\
        --out eval/longmem/reports/content_diff_bus.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import load_env, write_report  # noqa: E402
from eval.agent_introspect.encoder_replay import (  # noqa: E402
    _load_conversation, replay_one,
)


# Heuristic detector for range/fuzzy values in source text.
RANGE_PATTERNS = [
    re.compile(r"¥[\d,]+\s*[-–]\s*¥?[\d,]+", re.I),
    re.compile(r"\$\d+\s*[-–]\s*\$?\d+", re.I),
    re.compile(r"\baround\s+[\d,]+[^.]*", re.I),
    re.compile(r"\bapprox(?:imately)?\s+[\d,]+[^.]*", re.I),
    re.compile(r"\bstarting\s+from\s+(?:around\s+)?[\d,]+", re.I),
    re.compile(r"\b\d+\s*[-–]\s*\d+\s*(?:minutes?|hours?|usd|jpy)", re.I),
]


def _extract_actions(r: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Group actions by tool."""
    out = {"remember_batch": [], "revise_batch": [],
           "connect_batch": [], "brain_batch": [], "other": []}
    for a in r.get("actions", []):
        bucket = a.get("tool", "other")
        if bucket not in out:
            bucket = "other"
        out[bucket].append(a)
    return out


def _flatten_nodes(actions: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    out = []
    for a in actions["remember_batch"]:
        for n in (a.get("input") or {}).get("nodes") or []:
            out.append(n)
    # brain_batch may also embed remember ops
    for a in actions["brain_batch"]:
        for op in (a.get("input") or {}).get("operations") or []:
            if op.get("op") == "remember":
                out.append(op)
    return out


def _flatten_revisions(actions: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    out = []
    for a in actions["revise_batch"]:
        for rev in (a.get("input") or {}).get("revisions") or []:
            out.append(rev)
    for a in actions["brain_batch"]:
        for op in (a.get("input") or {}).get("operations") or []:
            if op.get("op") == "revise":
                out.append(op)
    return out


def _flatten_edges(actions: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    out = []
    for a in actions["remember_batch"]:
        for n in (a.get("input") or {}).get("nodes") or []:
            title = (n.get("title") or "")[:40]
            for c in n.get("connect_to") or []:
                out.append({"relation": c.get("relation"),
                            "why": c.get("why", ""),
                            "source_title": title,
                            "target": (c.get("target_id") or
                                       c.get("target_title") or "?")})
    for a in actions["connect_batch"]:
        for c in (a.get("input") or {}).get("connections") or []:
            out.append({"relation": c.get("relation"),
                        "why": c.get("why", ""),
                        "source_title": c.get("source_id") or c.get("source", "?"),
                        "target": c.get("target_id") or c.get("target", "?")})
    return out


def _detect_ranges_in_conversation(conv: Dict[str, Any]) -> List[Dict[str, str]]:
    """Find every range/fuzzy value mentioned in the haystack."""
    hits = []
    for t in conv.get("turns") or []:
        text = t.get("text", "")
        for pat in RANGE_PATTERNS:
            for m in pat.finditer(text):
                hits.append({
                    "turn": t.get("turn_id"),
                    "role": t.get("role"),
                    "match": m.group().strip(),
                    "context": text[max(0, m.start() - 30):
                                    min(len(text), m.end() + 60)].strip(),
                })
    # Dedupe
    seen = set()
    out = []
    for h in hits:
        key = (h["turn"], h["match"])
        if key not in seen:
            out.append(h)
            seen.add(key)
    return out


def _node_carries_range(node: Dict[str, Any], range_text: str) -> bool:
    """Does the node's content/title preserve the range, or collapse to a point?"""
    range_text_l = range_text.lower().strip()
    # Pull out the two endpoints (very rough: first two numbers).
    nums = re.findall(r"\d[\d,]*", range_text_l)
    if len(nums) < 2:
        return True  # Single number — not a range. Don't penalize.
    a, b = nums[0], nums[1]

    blob = " ".join([
        (node.get("title") or ""),
        (node.get("content") or ""),
        (node.get("situation") or ""),
        (node.get("reasoning") or ""),
    ]).lower()

    # The encoded node carries the range if BOTH endpoints appear in the blob.
    return (a in blob) and (b in blob)


def _format_report(qid: str, prompts: Dict[str, str],
                   conv: Dict[str, Any],
                   per_prompt_runs: Dict[str, List[Dict[str, Any]]]
                   ) -> str:
    out: List[str] = []
    out.append(f"# Content diff — `{qid}`\n")
    out.append(f"**Question:** {conv['question']}\n")
    out.append(f"**Gold:** {conv['gold']}\n")
    out.append("")

    # ── Section 1: revision behaviour ────────────────────────────────
    out.append("## 1. Are we revising?\n")
    out.append("| prompt | trial | remember nodes | revise calls | connect_batch | brain_batch |")
    out.append("|---|---|---:|---:|---:|---:|")
    for prompt_label, runs in per_prompt_runs.items():
        for run_idx, r in enumerate(runs):
            actions = _extract_actions(r)
            n_nodes = len(_flatten_nodes(actions))
            n_revs = len(_flatten_revisions(actions))
            n_conn = len(actions["connect_batch"])
            n_bb = len(actions["brain_batch"])
            out.append(f"| `{prompt_label}` | {run_idx} | {n_nodes} | "
                       f"{n_revs} | {n_conn} | {n_bb} |")
    out.append("")
    out.append("(Single-shot replay has no prior nodes to revise — revise_batch "
               "calls would happen only in multi-window production runs. "
               "Connect_batch + brain_batch capture relational work.)\n")

    # ── Section 2: range preservation ────────────────────────────────
    out.append("## 2. Range / fuzzy value preservation\n")
    ranges = _detect_ranges_in_conversation(conv)
    if not ranges:
        out.append("No range/fuzzy patterns detected in the haystack.\n")
    else:
        out.append(f"{len(ranges)} range/fuzzy values detected in haystack:\n")
        for h in ranges[:25]:
            out.append(f"- [t{h['turn']} {h['role']}] `{h['match']}` "
                       f"({h['context'][:80]}…)")
        out.append("")
        out.append("### Per-prompt: how many of these did the encoder preserve as ranges?\n")
        out.append("| prompt | trial | ranges in source | ranges preserved | "
                   "collapsed-to-point |")
        out.append("|---|---|---:|---:|---:|")
        for prompt_label, runs in per_prompt_runs.items():
            for run_idx, r in enumerate(runs):
                actions = _extract_actions(r)
                nodes = _flatten_nodes(actions)
                preserved = 0
                collapsed = 0
                for rng in ranges:
                    found = False
                    for n in nodes:
                        if _node_carries_range(n, rng["match"]):
                            found = True
                            break
                    if found:
                        preserved += 1
                    else:
                        # Check if the encoder mentioned the range topic but didn't preserve the range
                        topic_first_num = re.findall(r"\d", rng["match"])
                        if topic_first_num:
                            for n in nodes:
                                blob = " ".join([(n.get("title") or ""),
                                                 (n.get("content") or "")]).lower()
                                if topic_first_num[0] in blob:
                                    collapsed += 1
                                    break
                out.append(f"| `{prompt_label}` | {run_idx} | {len(ranges)} | "
                           f"{preserved} | {collapsed} |")
        out.append("")

    # ── Section 3: facts side-by-side ────────────────────────────────
    out.append("## 3. Did we lose facts? — Node titles side-by-side\n")
    for prompt_label, runs in per_prompt_runs.items():
        out.append(f"### `{prompt_label}` — {len(runs)} trials\n")
        for run_idx, r in enumerate(runs):
            actions = _extract_actions(r)
            nodes = _flatten_nodes(actions)
            out.append(f"**Trial {run_idx} ({len(nodes)} nodes):**\n")
            for n in nodes:
                src_attr = n.get("source_attribution", "")
                src_marker = f" 🚩{src_attr}" if src_attr else ""
                title = (n.get("title") or "")[:120]
                ntype = n.get("type") or "?"
                out.append(f"  - `{ntype}` — {title}{src_marker}")
            out.append("")

    # ── Section 4: full node content (one representative trial per prompt) ──
    out.append("## 4. Full node content — first trial per prompt\n")
    for prompt_label, runs in per_prompt_runs.items():
        if not runs:
            continue
        out.append(f"### `{prompt_label}` — trial 0 full content\n")
        actions = _extract_actions(runs[0])
        nodes = _flatten_nodes(actions)
        for i, n in enumerate(nodes, 1):
            out.append(f"**Node {i}: `{n.get('type','?')}` — {n.get('title','')}**\n")
            if n.get("content"):
                out.append(f"- content: {n['content']}")
            if n.get("situation"):
                out.append(f"- situation: {n['situation']}")
            if n.get("user_raw_quote"):
                out.append(f"- user_raw_quote: \"{n['user_raw_quote']}\"")
            if n.get("anchor_raw_quote"):
                out.append(f"- anchor_raw_quote: \"{n['anchor_raw_quote']}\"")
            if n.get("source_attribution"):
                out.append(f"- source_attribution: `{n['source_attribution']}`")
            if n.get("connect_to"):
                for c in n["connect_to"]:
                    out.append(f"- ↪ `{c.get('relation','?')}` → "
                               f"{c.get('target_title') or c.get('target_id','?')}"
                               f" (why: {c.get('why','')[:120]})")
            out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qid", required=True)
    p.add_argument("--prompts", required=True,
                   help="label=path,label=path")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    load_env()

    prompts: Dict[str, str] = {}
    for chunk in args.prompts.split(","):
        if "=" not in chunk:
            raise ValueError(f"--prompts entry missing '=': {chunk}")
        label, path = chunk.split("=", 1)
        prompts[label.strip()] = Path(path.strip()).read_text()

    tmpdir = tempfile.mkdtemp(prefix="content_diff_")
    os.environ["BRAIN_DB_DIR"] = tmpdir
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    brain = create_fresh_eval_brain(path=tmpdir, wipe=True)

    conv = _load_conversation(args.qid)

    import concurrent.futures
    per_prompt_runs: Dict[str, List[Dict[str, Any]]] = {p: [] for p in prompts}

    tasks = [(label, trial) for label in prompts for trial in range(args.trials)]

    def _do(label, trial):
        return label, replay_one(brain, args.qid, prompts[label],
                                  model=args.model, max_tokens=args.max_tokens)

    print(f"[content_diff] {args.qid}: {len(tasks)} replays "
          f"({len(prompts)} prompts × {args.trials} trials)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(_do, *t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            label, r = fut.result()
            per_prompt_runs[label].append(r)
            n_nodes = sum(len((a.get("input") or {}).get("nodes") or [])
                          for a in r.get("actions", [])
                          if a.get("tool") == "remember_batch")
            print(f"  {label}: {n_nodes} nodes, {len(r.get('actions',[]))} actions, "
                  f"{r.get('call_ms','?')}ms", flush=True)

    md = _format_report(args.qid, prompts, conv, per_prompt_runs)
    if args.out:
        write_report(Path(args.out), md)
    else:
        print(md)


if __name__ == "__main__":
    main()
