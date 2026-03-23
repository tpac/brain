"""Benchmark: Precision lifecycle — table-driven vs config-slot evaluation rates.

Simulates multi-turn conversations with realistic failure modes:
- Normal flow (recall → response → followup)
- Rapid messages (multiple recalls before response)
- Session interruption (compaction, session end mid-chain)
- Hook timeout (response track doesn't fire)
- Empty responses (Claude gives short answer)

Compares:
  1. Config-slot handoff (old): single last_recall_log_id / last_evaluated_recall_id
  2. Table-driven (new): query recall_log for pending work

All imports verified: Brain (servers/brain.py), RecallPrecision (servers/brain_precision.py),
LogsDAL (servers/dal.py). Method signatures match existing implementations.

Sacred system benchmark — run BEFORE and AFTER any precision pipeline change.

Usage:
    python tests/bench_precision_lifecycle.py
    python tests/bench_precision_lifecycle.py --verbose
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.brain_precision import RecallPrecision
from servers.dal import LogsDAL


# ── Simulation scenarios ──

SCENARIOS = [
    {
        "name": "normal_5_turns",
        "description": "5 normal turns — recall, response, followup, repeat",
        "turns": [
            {"type": "recall", "query": "auth flow", "node_count": 3},
            {"type": "response", "text": "Based on the auth decisions, Clerk magic links are the way to go."},
            {"type": "followup", "text": "That makes sense. Should we also add SSO?"},
            {"type": "recall", "query": "SSO options", "node_count": 2},
            {"type": "response", "text": "For SSO, Clerk supports Google and Apple out of the box."},
            {"type": "followup", "text": "Perfect, let's use Google SSO."},
            {"type": "recall", "query": "budget screen", "node_count": 4},
            {"type": "response", "text": "The budget screen uses a slider component, not text input."},
            {"type": "followup", "text": "Right, the slider. Can we add preset amounts too?"},
            {"type": "recall", "query": "creative upload", "node_count": 2},
            {"type": "response", "text": "Creative screen defaults to Upload tab."},
            {"type": "followup", "text": "Ok, but what about AI generate?"},
            {"type": "recall", "query": "onboarding fields", "node_count": 3},
            {"type": "response", "text": "Onboarding has exactly two fields: name and URL."},
            {"type": "followup", "text": "Good, keep it simple."},
        ],
        "expected_evaluations": 5,
    },
    {
        "name": "rapid_messages",
        "description": "3 recalls before any response — tests queue handling",
        "turns": [
            {"type": "recall", "query": "auth flow", "node_count": 3},
            {"type": "recall", "query": "SSO options", "node_count": 2},
            {"type": "recall", "query": "budget screen", "node_count": 4},
            {"type": "response", "text": "Here is a comprehensive answer covering auth, SSO, and budget."},
            {"type": "followup", "text": "Great overview, that covers everything."},
        ],
        "expected_evaluations_config": 1,
        "expected_evaluations_table": 1,
    },
    {
        "name": "session_interruption",
        "description": "Compaction mid-chain — response track never fires for second recall",
        "turns": [
            {"type": "recall", "query": "pricing model", "node_count": 3},
            {"type": "response", "text": "The pricing has three tiers for different customer segments."},
            {"type": "followup", "text": "Makes sense for the MVP."},
            {"type": "recall", "query": "payment integration", "node_count": 2},
            {"type": "interrupt"},
            {"type": "recall", "query": "dashboard layout", "node_count": 3},
            {"type": "response", "text": "Dashboard uses a responsive grid layout with charts."},
            {"type": "followup", "text": "Can we add real-time charts?"},
        ],
        "expected_evaluations_config": 1,
        "expected_evaluations_table": 2,
    },
    {
        "name": "empty_responses",
        "description": "Claude gives short responses — response track can't evaluate",
        "turns": [
            {"type": "recall", "query": "error handling", "node_count": 3},
            {"type": "response", "text": "Ok"},
            {"type": "followup", "text": "Can you elaborate on the error format?"},
            {"type": "recall", "query": "API format", "node_count": 2},
            {"type": "response", "text": "The API returns JSON errors with code and message fields for all endpoints."},
            {"type": "followup", "text": "Perfect, that is what I needed."},
        ],
        "expected_evaluations_config": 1,
        "expected_evaluations_table": 1,
    },
    {
        "name": "hook_timeout",
        "description": "Response track times out on 2 of 3 recalls",
        "turns": [
            {"type": "recall", "query": "auth flow", "node_count": 3},
            {"type": "response", "text": "Auth uses Clerk with magic links for a streamlined experience."},
            {"type": "followup", "text": "Good approach for the MVP."},
            {"type": "recall", "query": "SSO config", "node_count": 2},
            {"type": "timeout"},
            {"type": "followup", "text": "Actually let us skip SSO for now."},
            {"type": "recall", "query": "onboarding", "node_count": 3},
            {"type": "timeout"},
            {"type": "followup", "text": "Keep onboarding simple please."},
        ],
        "expected_evaluations_config": 1,
        "expected_evaluations_table": 1,
    },
]


def create_test_brain():
    """Create a fresh brain for simulation."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, 'brain.db')
    brain = Brain(db_path)
    for i in range(10):
        brain.remember(type="decision", title="Test decision %d" % i,
                       content="Content for decision %d" % i)
    return brain


def simulate_config_slot(brain, scenario):
    """Simulate old config-slot handoff behavior."""
    precision = RecallPrecision(brain.logs_conn, brain.conn)
    session_id = "ses_bench_%s" % scenario["name"]
    evaluations = 0
    last_recall_log_id = None
    last_evaluated_recall_id = None

    for turn in scenario["turns"]:
        if turn["type"] == "recall":
            log_id = precision.log_recall(
                session_id=session_id, query=turn["query"],
                returned_ids=["node_%d" % i for i in range(turn["node_count"])],
                recalled_titles={"node_%d" % i: "Title %d" % i for i in range(turn["node_count"])},
                recalled_snippets={"node_%d" % i: "Snippet %d" % i for i in range(turn["node_count"])})
            last_recall_log_id = log_id

        elif turn["type"] == "response":
            if last_recall_log_id and len(turn["text"]) >= 20:
                precision.evaluate_response(last_recall_log_id, turn["text"])
                last_evaluated_recall_id = last_recall_log_id
                last_recall_log_id = None

        elif turn["type"] == "followup":
            if last_evaluated_recall_id:
                precision.evaluate_followup(last_evaluated_recall_id, turn["text"])
                evaluations += 1
                last_evaluated_recall_id = None

        elif turn["type"] == "interrupt":
            last_recall_log_id = None
            last_evaluated_recall_id = None

        elif turn["type"] == "timeout":
            last_recall_log_id = None

    return evaluations


def simulate_table_driven(brain, scenario):
    """Simulate new table-driven lifecycle."""
    dal = LogsDAL(brain.logs_conn)
    precision = RecallPrecision(brain.logs_conn, brain.conn, logs_dal=dal)
    session_id = "ses_tbl_%s" % scenario["name"]
    evaluations = 0

    for turn in scenario["turns"]:
        if turn["type"] == "recall":
            pending = dal.get_pending_followups(session_id, limit=5)
            for p in pending:
                precision.evaluate_followup(p['id'], "(evaluated by next recall)")
                evaluations += 1

            precision.log_recall(
                session_id=session_id, query=turn["query"],
                returned_ids=["node_%d" % i for i in range(turn["node_count"])],
                recalled_titles={"node_%d" % i: "Title %d" % i for i in range(turn["node_count"])},
                recalled_snippets={"node_%d" % i: "Snippet %d" % i for i in range(turn["node_count"])})

        elif turn["type"] == "response":
            pending_id = dal.get_pending_response(session_id)
            if pending_id and len(turn["text"]) >= 20:
                precision.evaluate_response(pending_id, turn["text"])

        elif turn["type"] == "followup":
            pending = dal.get_pending_followups(session_id, limit=5)
            for p in pending:
                precision.evaluate_followup(p['id'], turn["text"])
                evaluations += 1

        elif turn["type"] == "interrupt":
            pass  # Table survives

        elif turn["type"] == "timeout":
            pass  # Row stays at LOGGED stage

    # End-of-session sweep
    pending = dal.get_pending_followups(session_id, limit=5)
    for p in pending:
        precision.evaluate_followup(p['id'], "(end-of-session sweep)")
        evaluations += 1

    return evaluations


def run_benchmark(verbose=False):
    """Run all scenarios, compare config-slot vs table-driven."""
    print("=" * 70)
    print("Precision Lifecycle Benchmark — Config-Slot vs Table-Driven")
    print("=" * 70)
    print()

    total_config = 0
    total_table = 0
    total_recalls = 0

    for scenario in SCENARIOS:
        brain_config = create_test_brain()
        brain_table = create_test_brain()

        n_recalls = sum(1 for t in scenario["turns"] if t["type"] == "recall")
        total_recalls += n_recalls

        t0 = time.perf_counter()
        config_evals = simulate_config_slot(brain_config, scenario)
        config_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        table_evals = simulate_table_driven(brain_table, scenario)
        table_ms = (time.perf_counter() - t0) * 1000

        expected_config = scenario.get("expected_evaluations_config", scenario.get("expected_evaluations", n_recalls))
        expected_table = scenario.get("expected_evaluations_table", scenario.get("expected_evaluations", n_recalls))

        total_config += config_evals
        total_table += table_evals

        config_ok = config_evals >= expected_config
        table_ok = table_evals >= expected_table

        print(f"  [{scenario['name']}] {scenario['description']}")
        print(f"    Recalls: {n_recalls}")
        print(f"    Config-slot: {config_evals}/{n_recalls} eval (expected >={expected_config}) "
              f"{'PASS' if config_ok else 'FAIL'} [{config_ms:.0f}ms]")
        print(f"    Table-driven: {table_evals}/{n_recalls} eval (expected >={expected_table}) "
              f"{'PASS' if table_ok else 'FAIL'} [{table_ms:.0f}ms]")
        print()

        brain_config.close()
        brain_table.close()

    config_rate = total_config * 100 // total_recalls if total_recalls else 0
    table_rate = total_table * 100 // total_recalls if total_recalls else 0

    print(f"{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total recalls: {total_recalls}")
    print(f"  Config-slot: {total_config}/{total_recalls} ({config_rate}%)")
    print(f"  Table-driven: {total_table}/{total_recalls} ({table_rate}%)")
    print(f"  Improvement: +{table_rate - config_rate}pp")
    print()
    target = int(total_recalls * 0.8)
    passed = table_rate >= 80
    print(f"  Table-driven target (>=80%): {total_table}>={target} -> {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")

    return {'total_recalls': total_recalls, 'config_rate': config_rate,
            'table_rate': table_rate, 'passed': passed}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    results = run_benchmark(verbose=args.verbose)
    return 0 if results['passed'] else 1


if __name__ == '__main__':
    exit(main())
