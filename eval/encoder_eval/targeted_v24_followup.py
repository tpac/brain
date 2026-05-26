"""Follow-up to targeted_v24_eval — candidate-only on 5 new items.

Tom's directive: see how the candidate stack (v24 + scout v7 + scout v4)
behaves on a broader sample. No need to re-run baseline; v22's behavior
on these items is already known from the 50-cell v22_vs_v19_longmem_5per
run (eval/encoder_eval/reports/v22_vs_v19_longmem_5per_20260525_212645).

Same per-cell brain pattern as targeted_v24_eval — interactions overridden
in the local eval brain, production daemon untouched.

Usage:
    ./dev python3 -m eval.encoder_eval.targeted_v24_followup
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from servers.daemon_client import send_command
from servers.brain import Brain
from eval.longmem.harness import _item_axis
from eval.longmem.fresh_brain import create_fresh_eval_brain, per_item_brain_dir
from eval.longmem.replay import replay_item, query_brain
from eval.longmem.answerer import answer_question
from eval.longmem.judge import judge_one
from eval.encoder_eval.quality_probes import run_all_probes
from eval.encoder_eval.targeted_v24_eval import (
    apply_interaction_override,
    fetch_template,
    run_cell as _run_cell_with_arm,
)


CANDIDATE_VERSIONS = {
    's1e': 24, 's1_scout_facts': 7, 's1_scout_quote': 4,
}

# Five new items, one per axis. None overlap with targeted_v24_eval's 5.
# v22-baseline results known from 50-cell run (used for comparison post-hoc).
ITEMS_QIDS = [
    '3fe836c9',           # multi_session — v22 ✓ (pre-approval delta)
    'gpt4_93159ced_abs',  # abstention   — v22 ✗ (Google/NovaTech answerer-mismatch)
    'cc539528',           # info_extr.   — v22 ✓ (where v19 wrote 0 nodes)
    'cc5ded98',           # know_update  — v22 ✓ (coding exercises hours)
    'gpt4_d31cdae3',      # temporal     — v22 ✓ (narrator trip ordering)
]


def main():
    run_name = f"v24_followup_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / 'eval' / 'encoder_eval' / 'reports' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell = out_dir / 'per_cell.jsonl'

    print(f"\nrun_name: {run_name}", flush=True)
    print(f"arm:      candidate {list(CANDIDATE_VERSIONS.values())}", flush=True)

    templates = {}
    for name, version in CANDIDATE_VERSIONS.items():
        templates[(name, version)] = fetch_template(name, version)
        print(f"  fetched {name} v{version}: {len(templates[(name, version)])} chars",
              flush=True)

    with open(ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json') as f:
        oracle = json.load(f)
    items = [it for it in oracle if it.get('question_id') in ITEMS_QIDS]
    missing = set(ITEMS_QIDS) - {i['question_id'] for i in items}
    assert not missing, f"missing items: {missing}"
    items.sort(key=lambda x: ITEMS_QIDS.index(x['question_id']))

    write_lock = threading.Lock()
    out_fh = open(per_cell, 'a')

    def writer(cell):
        with write_lock:
            out_fh.write(json.dumps(cell, default=str) + '\n')
            out_fh.flush()
        mark = '✓' if cell['correct'] else '✗'
        cov = cell['probes']['source_refs_coverage']
        print(f"  [{mark}] {cell['axis']:18s} {cell['item_id']:<22s} "
              f"nodes={cov.get('nodes_encoded', 0):>2d} "
              f"refs={cov.get('coverage_pct', 0):>5.1f}% "
              f"({cell['ingest_ms']/1000:.0f}s)",
              flush=True)

    # Patch ARMS into the shared run_cell so the arm label is meaningful
    from eval.encoder_eval import targeted_v24_eval as t24
    arm_label = 'candidate_v24+7+4'
    if arm_label not in t24.ARMS:
        t24.ARMS[arm_label] = CANDIDATE_VERSIONS

    try:
        print(f"\n=== {arm_label} on {len(items)} new items ===", flush=True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = [pool.submit(_run_cell_with_arm, arm_label, templates, item, run_name)
                    for item in items]
            for fut in as_completed(futs):
                try:
                    cell = fut.result()
                except Exception as e:
                    print(f"  CELL RAISED: {e!r}", flush=True)
                    continue
                writer(cell)
    finally:
        out_fh.close()

    print(f"\nDone. {per_cell}", flush=True)


if __name__ == '__main__':
    main()
