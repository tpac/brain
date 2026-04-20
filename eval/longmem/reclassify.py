"""Re-bucket a saved run report without re-running the eval.

Uses the stored `failure_evidence` (candidate count, selected IDs, ctx chars)
to compute the correct bucket with the current `_bucket()` logic. Also
regenerates the Haiku one-liner reason so it's aligned to the new bucket.

Usage:
  ./dev python3 -m eval.longmem.reclassify eval/longmem/reports/run_XXX.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def rebucket(ev: dict, item: dict = None) -> str:
    """Reproduce classifier._bucket() logic using the stored evidence.

    Note: PARTIAL_RECALL vs ANSWER_MISS distinction needs the context text,
    which isn't stored in evidence — so we can only detect it when re-reading
    live traces. Post-hoc reclassify gives ANSWER_MISS for ctx>0 cases.
    """
    if not ev.get("query_fired"):
        return "RECALL_MISS" if ev.get("relevant_to_gold") else "ENCODE_MISS"

    n_cand = ev.get("recall_candidates_count", 0)
    n_sel = len(ev.get("selected_ids", []))
    ctx_chars = ev.get("context_chars", 0)
    relevant = ev.get("relevant_to_gold", [])

    if n_cand == 0:
        return "ENCODE_MISS" if not relevant else "RECALL_MISS"
    if n_sel == 0 or ctx_chars == 0:
        return "SURFACE_MISS"
    return "ANSWER_MISS"


def reclassify_report(run_json_path: str, regen_reasons: bool = True) -> str:
    """Read run JSON, fix buckets, regen reasons, write a new scored JSON + MD.

    Emits a sibling file with `_reclassified` suffix so the original is preserved.
    """
    # Load env
    from pathlib import Path
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                key, val = k.strip(), v.strip().strip('"').strip("'")
                if not os.environ.get(key):
                    os.environ[key] = val

    with open(run_json_path) as f:
        report = json.load(f)

    from eval.longmem.classifier import _reason

    bucket_fixes = 0
    for item in report.get("results", []):
        if item.get("correct", True):
            continue
        ev = item.get("failure_evidence")
        if not ev:
            continue
        new_bucket = rebucket(ev)
        old_bucket = item.get("failure_bucket")
        if new_bucket != old_bucket:
            bucket_fixes += 1
            item["failure_bucket"] = new_bucket
            if regen_reasons:
                item["failure_reason"] = _reason(
                    item["question"], item["answer_gold"],
                    item["hypothesis"], new_bucket, ev)

    # Recompute failure_buckets summary
    by_bucket = {}
    for r in report.get("results", []):
        if not r.get("correct", True) and r.get("failure_bucket"):
            by_bucket[r["failure_bucket"]] = by_bucket.get(r["failure_bucket"], 0) + 1
    report["failure_buckets"] = by_bucket

    out_json = run_json_path.replace(".json", "_reclassified.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    from eval.longmem.report import render_report
    md_path = render_report(out_json)

    print(f"[reclassify] bucket fixes: {bucket_fixes}")
    print(f"[reclassify] json → {out_json}")
    print(f"[reclassify] md   → {md_path}")
    return md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="path to run_<name>.json")
    parser.add_argument("--no-reasons", action="store_true",
                        help="keep original reasons (skip Haiku regen)")
    args = parser.parse_args()
    reclassify_report(args.report, regen_reasons=not args.no_reasons)
