"""Surface replay probe — swap the surface system prompt, hold dynamic content.

For each item in a saved eval run:
  1. Load the per-item brain.db (saved via --keep_dbs)
  2. Reconstruct candidates_data from recall.json + nodes.jsonl
  3. Load operator's query + recent_messages from oracle
  4. Register a candidate surface prompt as the active 'surface' interaction
  5. Call _call_surface (handles v4 single-shot or v5 agentic tool-loop)
  6. Capture selection + tool_trace + raw output

The dynamic content (query, Frame, candidates) is what the original surface
saw. Only the static instruction (system prompt) is swapped. This is the
"Coverage" probe slot the agent_introspect family had open, applied to surface.

USE
    ./dev python3 -m eval.agent_introspect.surface_replay \\
        --run-dir eval/longmem/reports/ab_armB_v15_11_v5_<TS> \\
        --qids 8e91e7d9,54026fce,37f165cf,55241a1f \\
        --prompt eval/surface_v6_prompt.txt \\
        --variant v5_agentic \\
        --out eval/longmem/reports/surface_replay_v6.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load_oracle_item(qid: str) -> Dict[str, Any]:
    oracle_path = ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json'
    for item in json.loads(oracle_path.read_text()):
        if item.get('question_id') == qid:
            return item
    raise KeyError(f'qid {qid} not in oracle')


def _candidates_from_recall(run_dir: Path, qid: str) -> List[Dict[str, Any]]:
    """Reconstruct candidates_data shape from recall.json + nodes.jsonl.

    surface.py expects candidates_data as a list of dicts with at minimum:
      id, title, score, type, content, situation, reasoning, keywords
    recall.json has the ordered list with score; nodes.jsonl has the bodies.
    """
    rec = json.loads((run_dir / 'items' / qid / 'recall.json').read_text())
    node_index: Dict[str, Dict[str, Any]] = {}
    for line in (run_dir / 'items' / qid / 'nodes.jsonl').open():
        n = json.loads(line)
        node_index[n['id']] = n

    cands = []
    for c in rec.get('candidates') or []:
        cid_prefix = c.get('id', '')
        # Find the full node by prefix match
        full = next((node_index[full_id] for full_id in node_index
                     if full_id.startswith(cid_prefix)), None)
        if full is None:
            continue
        kv = full.get('kv') or {}
        cands.append({
            'id': full['id'],
            'title': full.get('title', ''),
            'type': full.get('type', ''),
            'content': full.get('content', ''),
            'situation': kv.get('situation', ''),
            'reasoning': kv.get('reasoning', ''),
            'keywords': full.get('keywords', ''),
            'score': c.get('score', 0.0),
            'match': c.get('score', 0.0),
            'conf': full.get('confidence', 0.5),
            'source': 'cosine',
        })
    return cands


def replay_one(run_dir: Path, qid: str, system_prompt: str,
                variant: str = 'v5_agentic') -> Dict[str, Any]:
    """Run a single surface call with a candidate prompt against the saved brain."""
    rec = json.loads((run_dir / 'items' / qid / 'recall.json').read_text())
    meta = json.loads((run_dir / 'items' / qid / 'meta.json').read_text())

    # Load brain.db (it was the AgentsContext path used at eval time; the
    # per-item dir is what the harness used with BRAIN_DB_DIR)
    brain_db_path = rec.get('brain_db_path') or None
    # Newer artifacts may not store this; result.json has it
    if not brain_db_path:
        result_path = run_dir / 'items' / qid / 'result.json'
        if result_path.exists():
            r = json.loads(result_path.read_text())
            brain_db_path = r.get('brain_db_path')
    if not brain_db_path or not os.path.exists(brain_db_path):
        return {'qid': qid, 'error': f'brain_db_path missing or not on disk: {brain_db_path}'}

    os.environ['BRAIN_DB_DIR'] = brain_db_path
    os.environ['BRAIN_SURFACE_VARIANT'] = variant
    # Replays must never write into the production capture corpus —
    # _call_surface captures by default (surface_capture.py).
    os.environ['BRAIN_SURFACE_CAPTURE'] = 'off'

    from servers.brain import Brain
    brain_db_file = os.path.join(brain_db_path, 'brain.db')
    brain = Brain(brain_db_file)

    # Register candidate prompt as a new 'surface' version + activate it.
    # 8192 (Haiku's per-call max) vs production's 2048. v6 prompts Haiku
    # to write justifications per pick; 2048 truncated 54026fce in round 1,
    # 4096 truncated 75832dbd in round 2. Use Haiku's full ceiling.
    from tests.interaction_override import override_interaction
    override_interaction(
        brain, 'surface', template=system_prompt,
        parameters={'model': 'claude-haiku-4-5', 'max_tokens': 8192},
        set_by='surface_replay')

    # Reconstruct dynamic content
    candidates_data = _candidates_from_recall(run_dir, qid)
    query = rec.get('query', '')
    # recent_messages = the haystack conversation (as the surface saw it
    # at query time, the surface gets the recent operator turns)
    oracle_item = _load_oracle_item(qid)
    recent_messages = []
    for sess in oracle_item.get('haystack_sessions', []):
        for t in sess:
            recent_messages.append({'role': t['role'], 'content': t['content']})
    # Frame — use empty for replay; would normally be brain.filter_nodes(...)
    # but the saved brain has the same nodes the original surface saw, so a
    # Frame rebuild here would not change the comparison
    frame = ''

    # Make the call. Use _call_surface which handles variant routing internally.
    from servers.scales.s1.surface import _call_surface

    t0 = time.time()
    try:
        # 5-tuple: _call_surface now also returns run-cost telemetry (token
        # counts + elapsed_ms + rounds). This replay computes its own
        # elapsed_ms below; the telemetry is absorbed but unused here.
        surfaced, used_prompt, max_tokens, interaction_id, _telemetry = _call_surface(
            brain, candidates_data, query, recent_messages,
            session_id=f'replay-{qid}', result={}, frame=frame)
        elapsed_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        return {'qid': qid, 'error': f'{type(e).__name__}: {e}',
                'elapsed_ms': elapsed_ms}
    finally:
        try: brain.close()
        except Exception: pass

    selected = surfaced.get('selected', []) if isinstance(surfaced, dict) else []
    tool_trace = (getattr(brain, '_surface_tool_traces', {}) or {}).get(
        f'replay-{qid}') or []

    return {
        'qid': qid,
        'question': meta.get('question'),
        'gold': meta.get('gold'),
        'axis': meta.get('axis'),
        'variant': variant,
        'selected_count': len(selected),
        'selected': selected,
        'tool_trace': tool_trace,
        'elapsed_ms': elapsed_ms,
        'original_selected': rec.get('selected') or [],
        'original_tool_trace': rec.get('tool_trace') or [],
    }


def render_md(rows: List[Dict[str, Any]], prompt_path: str) -> str:
    out = [f'# Surface replay — `{prompt_path}`', '']
    out.append('Compares replay (this prompt) vs original (from recall.json) per item.')
    out.append('')

    total_replay = sum(r.get('selected_count', 0) for r in rows if 'error' not in r)
    total_orig = sum(len(r.get('original_selected') or []) for r in rows if 'error' not in r)
    out.append(f'**Cohort selections:** replay={total_replay}, original={total_orig}')
    out.append('')

    for r in rows:
        if 'error' in r:
            out.append(f'## `{r["qid"]}` — ERROR: {r["error"]}')
            out.append('')
            continue
        out.append(f'## `{r["qid"]}` ({r.get("axis","-")})')
        out.append('')
        out.append(f'**Q:** {r.get("question","")}')
        out.append(f'**Gold:** {(r.get("gold","") or "")[:200]}')
        out.append(f'**Timing:** {r["elapsed_ms"]}ms')
        out.append('')
        out.append(f'**Original selected** ({len(r.get("original_selected") or [])}):')
        for s in (r.get('original_selected') or [])[:8]:
            if isinstance(s, dict):
                out.append(f'  - {s.get("id","-")[:8]} mode={s.get("mode","-")} why={(s.get("why","") or "")[:80]}')
            else:
                out.append(f'  - {str(s)[:40]}')
        out.append('')
        out.append(f'**Replay selected** ({r["selected_count"]}):')
        for s in r.get('selected', [])[:8]:
            if isinstance(s, dict):
                out.append(f'  - {s.get("id","-")[:8]} mode={s.get("mode","-")} why={(s.get("why","") or "")[:80]}')
            else:
                out.append(f'  - {str(s)[:40]}')
        out.append('')
        # Tool trace
        rounds = r.get('tool_trace') or []
        if rounds:
            out.append(f'**Replay tool_trace:** {len(rounds)} rounds')
            for ri, rnd in enumerate(rounds):
                if not isinstance(rnd, dict):
                    continue
                calls = rnd.get('tool_calls') or []
                out.append(f'  Round {ri}: stop={rnd.get("stop_reason")} calls={len(calls)}')
                for c in calls:
                    args = c.get('args') or {}
                    out.append(f'    - {c.get("tool")}({args}) -> {c.get("result_count")} results')
        else:
            out.append('**Replay tool_trace:** (none)')
        out.append('')
        out.append('---')
        out.append('')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--qids', required=True, help='comma-separated qids to replay')
    p.add_argument('--prompt', required=True, help='path to candidate surface prompt')
    p.add_argument('--variant', default='v5_agentic',
                   help='BRAIN_SURFACE_VARIANT to set: v4 or v5_agentic')
    p.add_argument('--out', default=None, help='markdown output path')
    args = p.parse_args()

    # Load .env so the Anthropic key is available
    envf = ROOT / '.env'
    if envf.exists():
        for line in envf.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

    system_prompt = Path(args.prompt).read_text()
    run_dir = Path(args.run_dir)
    qids = [q.strip() for q in args.qids.split(',') if q.strip()]

    rows = []
    for i, qid in enumerate(qids, start=1):
        print(f'[{i}/{len(qids)}] replaying {qid} ...', flush=True)
        r = replay_one(run_dir, qid, system_prompt, variant=args.variant)
        if 'error' in r:
            print(f'  FAILED: {r["error"]}', flush=True)
        else:
            print(f'  → selected={r["selected_count"]} (orig {len(r.get("original_selected") or [])}) '
                  f'tools={len(r.get("tool_trace") or [])} rounds {r["elapsed_ms"]}ms', flush=True)
        rows.append(r)

    md = render_md(rows, args.prompt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f'wrote {args.out}', flush=True)
    else:
        print(md)


if __name__ == '__main__':
    main()
