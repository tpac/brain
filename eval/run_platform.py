#!/usr/bin/env python3
"""Unified runner for the eval platform.

Usage:
    # Run all suites
    python3 eval/run_platform.py

    # Run specific suite
    python3 eval/run_platform.py --suite encoding
    python3 eval/run_platform.py --suite e2e

    # Save results
    python3 eval/run_platform.py --save
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Eval Platform Runner")
    parser.add_argument("--suite", choices=["encoding", "e2e", "all"], default="all")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--category", help="Filter conversations to category")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    all_results = {}
    t0 = time.time()

    # ── Encoding Suite (needs ANTHROPIC_API_KEY) ──
    if args.suite in ("encoding", "all"):
        from eval.encoding_suite import run_suite as run_encoding, print_results as print_encoding
        results = run_encoding(model=args.model, category=args.category,
                              max_workers=args.workers, verbose=verbose)
        all_results["encoding"] = results
        if verbose:
            print_encoding(results)

    # ── E2E Suite (needs ANTHROPIC_API_KEY) ──
    if args.suite in ("e2e", "all"):
        from eval.e2e_suite import run_suite as run_e2e, print_results as print_e2e
        results = run_e2e(model=args.model, category=args.category,
                         max_workers=min(args.workers, 2), verbose=verbose)
        all_results["e2e"] = results
        if verbose:
            print_e2e(results)

    elapsed = time.time() - t0

    if verbose:
        print("=" * 70)
        print("PLATFORM COMPLETE — %.0fs total" % elapsed)
        print("=" * 70)

    if args.save:
        output_dir = str(ROOT / "eval" / "results")
        os.makedirs(output_dir, exist_ok=True)
        filename = "platform_%s.json" % time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": elapsed,
                "suites": all_results,
            }, f, indent=2, default=str)
        print("Full results saved: %s" % path)


if __name__ == "__main__":
    main()
