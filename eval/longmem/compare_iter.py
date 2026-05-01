"""Compare results across multiple iter_* runs.

Reads hypotheses_{run_name}.jsonl files for each run and prints:
  1. Per-axis summary table (variant × axis)
  2. Per-item table — which items moved between variants
  3. Failure-mode shifts: who flipped from pass→fail and vice versa
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPORT_DIR = Path("eval/longmem/reports")


def load_run(run_name):
    """Load a run's per-item results.

    Reads run_{run_name}.json (the report file) which has the full result
    dicts (correct, abstained, has_context, etc.). Falls back to
    hypotheses_{run_name}.jsonl for the bare-bones shape if needed.
    """
    rpath = REPORT_DIR / f"run_{run_name}.json"
    if rpath.exists():
        try:
            data = json.loads(rpath.read_text())
            results = data.get("results") or data.get("items") or []
            if results:
                return {r["question_id"]: r for r in results}
        except Exception:
            pass

    # Fallback: hypotheses jsonl (minimal — no correct flag)
    path = REPORT_DIR / f"hypotheses_{run_name}.jsonl"
    if not path.exists():
        return None
    items = {}
    for line in path.open():
        r = json.loads(line)
        items[r["question_id"]] = r
    return items


def axis_of(qid, item, oracle):
    """Determine axis from oracle data — same logic as harness."""
    if qid.endswith("_abs"):
        return "abstention"
    qt = oracle.get(qid, {}).get("question_type", "?")
    if "single-session" in qt:
        return "info_extraction"
    if qt == "multi-session":
        return "multi_session"
    if qt == "temporal-reasoning":
        return "temporal"
    if qt == "knowledge-update":
        return "knowledge_update"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run_name list (e.g. iter_A_baseline iter_D_l4 ...)")
    parser.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    args = parser.parse_args()

    oracle = {it["question_id"]: it for it in json.load(open(args.oracle))}

    runs = {}
    for r in args.runs:
        d = load_run(r)
        if d is None:
            print(f"[compare] {r}: hypotheses file not found, skipping", file=sys.stderr)
            continue
        runs[r] = d

    if not runs:
        print("no runs loaded", file=sys.stderr)
        sys.exit(1)

    # Common item set across all variants (only fully-completed compare cleanly)
    common = set.intersection(*(set(rs.keys()) for rs in runs.values()))
    print(f"# Compare across {len(runs)} variants on {len(common)} common items\n")

    # Per-axis table
    print("## Per-axis pass rates\n")
    print("| Axis | " + " | ".join(runs.keys()) + " |")
    print("|---|" + "|".join("---" for _ in runs) + "|")

    axis_buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # axis → run → [pass, total]
    for qid in common:
        ax = axis_of(qid, oracle.get(qid, {}), oracle)
        for run, items in runs.items():
            r = items[qid]
            axis_buckets[ax][run][1] += 1
            if r.get("correct"):
                axis_buckets[ax][run][0] += 1

    for ax in sorted(axis_buckets.keys()):
        cells = []
        for run in runs:
            p, t = axis_buckets[ax][run]
            cells.append(f"{p}/{t}")
        print(f"| {ax} | " + " | ".join(cells) + " |")

    # Overall
    print(f"| **overall** | ", end="")
    for run in runs:
        p = sum(items.get("correct", False) for items in runs[run].values() if items["question_id"] in common)
        t = len(common)
        print(f"**{p}/{t} ({100*p/t:.0f}%)** | ", end="")
    print()

    # Per-item table
    print("\n## Per-item")
    print("| qid | axis | " + " | ".join(runs.keys()) + " | question |")
    print("|---|---|" + "|".join("---" for _ in runs) + "|---|")
    for qid in sorted(common):
        ax = axis_of(qid, oracle.get(qid, {}), oracle)
        cells = []
        for run in runs:
            r = runs[run][qid]
            mark = "✓" if r.get("correct") else "✗"
            if r.get("abstained"):
                mark += "·abs"
            elif not r.get("has_context"):
                mark += "·no-ctx"
            cells.append(mark)
        q = oracle.get(qid, {}).get("question", "")[:60]
        print(f"| `{qid[:14]}` | {ax} | " + " | ".join(cells) + f" | {q} |")

    # Movement: items that flipped
    print("\n## Flips vs first run")
    first = list(runs.keys())[0]
    for run in list(runs.keys())[1:]:
        gained = []
        lost = []
        for qid in common:
            a = runs[first][qid].get("correct")
            b = runs[run][qid].get("correct")
            if a != b:
                if b and not a:
                    gained.append(qid)
                elif a and not b:
                    lost.append(qid)
        print(f"\n### {first} → {run}")
        print(f"- gained: {len(gained)} {[q[:14] for q in gained]}")
        print(f"- lost:   {len(lost)} {[q[:14] for q in lost]}")


if __name__ == "__main__":
    main()
