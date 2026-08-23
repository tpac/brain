"""Targeted v24 + scout v7/v4 eval — baseline (v22/v5/v2) vs candidate (v24/v7/v4).

Tests c2ac3c61 (the multi_session precision-refinement failure) + 4 v22-passing
items across axes to verify no regression. Per-cell brain has all three
interactions overridden in its own interactions table — production daemon is
untouched.

Usage:
    ./dev python3 -m eval.encoder_eval.targeted_v24_eval
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
from tests.interaction_override import override_interaction


# Two arms: 3 interactions per arm, version-pinned per cell
ARMS = {
    'baseline_v22+5+2': {
        's1e': 22, 's1_scout_facts': 5, 's1_scout_quote': 2,
    },
    'candidate_v24+7+4': {
        's1e': 24, 's1_scout_facts': 7, 's1_scout_quote': 4,
    },
}

# c2ac3c61 = the multi_session precision-refinement target case.
# Others = v22-passing items spanning 3 axes — regression checks.
ITEMS_QIDS = [
    'c2ac3c61',       # multi_session — TARGET (v22 ✗)
    '5025383b',       # multi_session — regression check (v22 ✓)
    '60159905',       # multi_session — regression check (v22 ✓, count-style)
    'ce6d2d27',       # knowledge_update — regression check (v22 ✓)
    'bbf86515',       # temporal — regression check (v22 ✓)
]


def fetch_template(name: str, version: int) -> str:
    r = send_command('get_interaction', {'name': name, 'version': version})
    return r['result']['template']


def run_cell(arm_name: str, templates: dict, item: dict, run_name: str) -> dict:
    qid = item['question_id']
    arm_versions = ARMS[arm_name]
    arm_run_name = f"{run_name}-{arm_name}"
    item_db = per_item_brain_dir(qid, run_name=arm_run_name)
    brain = create_fresh_eval_brain(path=item_db, wipe=True)
    # Apply all three overrides to this eval brain's interaction table
    for name, version in arm_versions.items():
        override_interaction(brain, name, template=templates[(name, version)])

    t0 = time.time()
    ingest_session_id = f"ingest-{qid}"
    ingest_stats = replay_item(
        brain, ingest_session_id, item['haystack_sessions'],
        haystack_dates=item.get('haystack_dates'),
        log_prefix=f"[{arm_name}/{qid}]")
    ingest_ms = int((time.time() - t0) * 1000)

    q_result = query_brain(brain, item['question'], item.get('question_date'))
    a_result = answer_question(
        item['question'], q_result['additional_context'],
        item.get('question_date'))
    j = judge_one(item['question'], item['answer'], a_result['hypothesis'])

    brain.close()
    probe_brain = Brain(db_path=os.path.join(item_db, 'brain.db'))
    try:
        probes = run_all_probes(probe_brain, item)
    finally:
        probe_brain.conn.close()
        probe_brain.logs_conn.close()

    return {
        'arm': arm_name,
        'versions': arm_versions,
        'item_id': qid,
        'axis': _item_axis(item),
        'correct': j['correct'],
        'judge_raw': j['raw'],
        'judge_reasoning': j.get('reasoning', ''),
        'hypothesis': a_result['hypothesis'],
        'gold': item['answer'],
        'question': item['question'],
        'ingest_stats': ingest_stats,
        'probes': probes,
        'ingest_ms': ingest_ms,
        'brain_db': item_db,
    }


def main():
    run_name = f"v24_targeted_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / 'eval' / 'encoder_eval' / 'reports' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell = out_dir / 'per_cell.jsonl'

    # Materialize templates
    print(f"\nrun_name: {run_name}", flush=True)
    print(f"out_dir : {out_dir}", flush=True)
    templates: dict = {}
    for arm_versions in ARMS.values():
        for name, version in arm_versions.items():
            key = (name, version)
            if key not in templates:
                templates[key] = fetch_template(name, version)
                print(f"  fetched {name} v{version}: {len(templates[key])} chars",
                      flush=True)

    # Load items
    with open(ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json') as f:
        oracle = json.load(f)
    items = [it for it in oracle if it.get('question_id') in ITEMS_QIDS]
    missing = set(ITEMS_QIDS) - {i['question_id'] for i in items}
    assert not missing, f"missing items: {missing}"
    # Re-order to match ITEMS_QIDS for stable per_cell ordering
    items.sort(key=lambda x: ITEMS_QIDS.index(x['question_id']))

    write_lock = threading.Lock()
    out_fh = open(per_cell, 'a')

    def writer(cell):
        with write_lock:
            out_fh.write(json.dumps(cell, default=str) + '\n')
            out_fh.flush()
        mark = '✓' if cell['correct'] else '✗'
        cov = cell['probes']['source_refs_coverage']
        print(f"  [{mark}] {cell['arm']:20s} {cell['axis']:18s} "
              f"{cell['item_id']:<22s} "
              f"nodes={cov.get('nodes_encoded', 0):>2d} "
              f"refs={cov.get('coverage_pct', 0):>5.1f}% "
              f"({cell['ingest_ms']/1000:.0f}s)",
              flush=True)

    try:
        for arm_name in ARMS.keys():
            print(f"\n=== ARM: {arm_name} {list(ARMS[arm_name].values())} ===",
                  flush=True)
            with ThreadPoolExecutor(max_workers=5) as pool:
                futs = [pool.submit(run_cell, arm_name, templates, item, run_name)
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
    print(f"To view: jq 'select(.arm) | "
          f'"\\(.arm)\\t\\(.item_id)\\t\\(.correct)\\t\\(.hypothesis[0:120])"\' '
          f"{per_cell}", flush=True)


if __name__ == '__main__':
    main()
