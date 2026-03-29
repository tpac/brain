#!/usr/bin/env python3
"""Capability: Noise resistance.

Tests that the encoding agent does NOT create nodes for casual, trivial,
or meta-conversational exchanges. A good encoder knows when to stay quiet.

PASS: Nothing encoded (or only context markers)
FAIL: Noise nodes created ("user said ok", "discussed next steps", etc.)
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.capabilities.base import CapabilityTest


class NoiseResistanceTest(CapabilityTest):
    capability_name = "noise_resistance"


def run(fixture_path: str = None, model: str = "claude-sonnet-4-6",
        verbose: bool = True, scenario_filter: str = None):
    """Run all noise resistance scenarios."""
    if fixture_path is None:
        fixture_path = str(Path(__file__).parent.parent / "fixtures" / "capability_brain.db")

    test = NoiseResistanceTest(fixture_path=fixture_path, model=model)

    scenario_dir = Path(__file__).parent.parent / "scenarios" / "noise_resistance"
    scenario_files = sorted(scenario_dir.glob("*.json"))

    if scenario_filter:
        scenario_files = [f for f in scenario_files if scenario_filter in f.name]

    if not scenario_files:
        print("No noise resistance scenarios found in %s" % scenario_dir)
        return []

    if verbose:
        print("=" * 60)
        print("NOISE RESISTANCE CAPABILITY TEST")
        print("=" * 60)
        print("Scenarios: %d, Model: %s" % (len(scenario_files), model))
        print()

    scores = []
    for sf in scenario_files:
        if verbose:
            print("[%s]" % sf.stem)
        try:
            score = test.run_scenario(str(sf), verbose=verbose)
            scores.append(score)
        except Exception as e:
            if verbose:
                print("  ERROR: %s" % e)
            scores.append({"scenario_id": sf.stem, "verdict": "ERROR", "error": str(e)})

    if verbose:
        print()
        passed = sum(1 for s in scores if (getattr(s, 'verdict', None) or (s.get('verdict') if isinstance(s, dict) else '')) == "PASS")
        print("Results: %d/%d passed" % (passed, len(scores)))

    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--scenario", help="Filter to specific scenario")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    scores = run(model=args.model, verbose=not args.quiet, scenario_filter=args.scenario)
