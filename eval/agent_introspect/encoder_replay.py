"""Encoder replay probe — run a single encoder call with a candidate prompt.

The unbuilt "coverage" probe from AGENT-INTROSPECTION.md, finally built.
For diagnostic-level prompt iteration:

  - Load a conversation from the eval oracle
  - Run scouts to produce real muster_ctx
  - Call Sonnet ONCE with the candidate prompt + scout report + tool defs
  - Inspect tool_use blocks (no execution — read-only)

This is FAST (~30s per item) because there's no haystack ingest loop,
no surface, no answerer, no classifier, no brain writes. Just one
encoder Sonnet call with the exact prompt under test, on the exact
conversation Sonnet would see.

USE
    ./dev python3 -m eval.agent_introspect.encoder_replay \\
        --qids gpt4_85da3956,982b5123,71017276,0bb5a684 \\
        --prompt eval/prompts/s1e_v15_10.txt \\
        --out eval/longmem/reports/temporal_compare_2026_05_13/replay_v15_10.md

Reports: per-item tool_use sequence, with event_time emission
highlighted for each remember_batch node.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import load_env, write_report


def _load_conversation(qid: str) -> Dict[str, Any]:
    """Return {meta, turns, conversation_now}."""
    oracle_path = ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json'
    oracle = json.loads(oracle_path.read_text())
    item = next((i for i in oracle if i['question_id'] == qid), None)
    if item is None:
        raise ValueError(f'qid {qid} not in oracle')
    turns = []
    tid = 0
    for sess_idx, sess in enumerate(item.get('haystack_sessions', [])):
        for t in sess:
            turns.append({
                'turn_id': f't{tid}',
                'role': t.get('role', ''),
                'text': t.get('content', ''),
                'session_idx': sess_idx,
            })
            tid += 1
    # conversation_now = last haystack session date
    dates = item.get('haystack_dates') or []
    conv_now = dates[-1].split(' ')[0].replace('/', '-') if dates else None
    return {
        'qid': qid,
        'question': item.get('question'),
        'gold': item.get('answer'),
        'question_date': item.get('question_date'),
        'turns': turns,
        'conversation_now': conv_now,
    }


def _run_scouts(brain, turns, conversation_now: str) -> Dict[str, Any]:
    """Run muster on the full conversation, return scout outputs dict.

    Matches encode.py's call: pass messages with 'role'/'content'/'id',
    plus the catalog-rendered string + ID set (empty for fresh brains)
    and a session_id/counter pair (synthetic — replay isn't tied to a
    real session).
    """
    from servers.scales.s1.scouts.muster import build_muster_context, run_muster
    messages = [{'role': t['role'], 'content': t['text'], 'id': t['turn_id']}
                for t in turns]
    ctx = build_muster_context(
        brain=brain,
        messages=messages,
        session_id='replay-session',
        counter=0,
        catalog_rendered='(empty — fresh brain)',
        catalog_node_ids=set(),
        session_context='',
        current_date=conversation_now,
    )
    formatted_report, scout_outputs, metrics = run_muster(ctx)
    return {'report': formatted_report, 'outputs': scout_outputs, 'metrics': metrics}


def _build_user_content(conv: Dict[str, Any], scout_report: str) -> str:
    """Approximate the encoder's user_content: catalog stub + conversation + scout report.

    Faithful to encode.py::_build_user_content shape, minus catalog (eval starts
    from empty brain) and journal (no prior turns).
    """
    lines = []
    lines.append(f"## Current date\n{conv['conversation_now']}")
    lines.append('')
    lines.append(f"## Node catalog\n(empty — fresh brain)")
    lines.append('')
    lines.append('## Conversation')
    for t in conv['turns']:
        speaker = 'OPERATOR' if t['role'] == 'user' else 'ANCHOR'
        lines.append(f"\n[turn {t['turn_id']}, {speaker}]")
        lines.append(t['text'])
    lines.append('')
    lines.append('## Scout reports')
    lines.append('')
    lines.append(scout_report)
    return '\n'.join(lines)


def _format_action(idx: int, block) -> Dict[str, Any]:
    """Render a Sonnet tool_use block into an inspectable dict."""
    return {
        'idx': idx,
        'tool': block.name,
        'input': block.input,
    }


def _summarize_remember_batch(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull node summaries (type, title, event_time, fields-set) from a remember_batch action."""
    out = []
    nodes = (action.get('input') or {}).get('nodes') or []
    for n in nodes:
        kv = {k: v for k, v in n.items() if k not in ('type', 'title', 'content',
                                                       'situation', 'connect_to')}
        out.append({
            'type': n.get('type'),
            'title': (n.get('title') or '')[:80],
            'event_time': kv.get('event_time'),
            'has_content': bool(n.get('content')),
            'has_situation': bool(n.get('situation')),
            'kv_fields': [k for k in kv.keys() if k != 'event_time'],
            'connects': len(n.get('connect_to') or []),
        })
    return out


def replay_one(brain, qid: str, system_prompt: str,
                model: str = 'claude-sonnet-4-6',
                max_tokens: int = 8000) -> Dict[str, Any]:
    """Run a single encoder call on the full conversation and capture tool_use.

    Note: production encoder runs in windowed calls (every ~5 stops) with
    4000 max_tokens per call. The single-shot replay asks Sonnet to encode
    the WHOLE haystack in one call — bump max_tokens to 8000 to compensate.
    This is a diagnostic ("what does v15.10 anchor for this conversation?"),
    not a perfect production replay; expect Sonnet to produce a denser
    single-batch encoding than the windowed pipeline would.
    """
    conv = _load_conversation(qid)
    t_scout = time.time()
    scout_data = _run_scouts(brain, conv['turns'], conv['conversation_now'])
    scout_ms = int((time.time() - t_scout) * 1000)

    user_content = _build_user_content(conv, scout_data['report'])

    # Build encoder tool defs from the live registry
    from servers.scales.s1.encode import _get_tool_schemas
    tools = _get_tool_schemas()

    import anthropic
    client = anthropic.Anthropic()
    t_call = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_content}],
        tools=tools,
    )
    call_ms = int((time.time() - t_call) * 1000)

    actions = []
    text_parts = []
    for i, block in enumerate(resp.content):
        if getattr(block, 'type', None) == 'tool_use':
            actions.append(_format_action(i, block))
        elif getattr(block, 'type', None) == 'text':
            text_parts.append(block.text)

    return {
        'qid': qid,
        'question': conv['question'],
        'gold': conv['gold'],
        'conversation_now': conv['conversation_now'],
        'scout_ms': scout_ms,
        'call_ms': call_ms,
        'tokens_in': resp.usage.input_tokens,
        'tokens_out': resp.usage.output_tokens,
        'stop_reason': resp.stop_reason,
        'actions': actions,
        'final_text': '\n'.join(text_parts),
        'scout_metrics': scout_data['metrics'],
    }


def render_report(results: List[Dict[str, Any]], prompt_path: str) -> str:
    lines = [f'# Encoder replay — {prompt_path}', '']
    total_nodes = 0
    total_with_et = 0
    for r in results:
        lines.append(f'## `{r["qid"]}` (conversation_now={r["conversation_now"]})')
        lines.append('')
        lines.append(f'**Q:** {r["question"]}')
        lines.append(f'**Gold:** {r["gold"]}')
        lines.append(f'**Timing:** scouts={r["scout_ms"]}ms · encoder={r["call_ms"]}ms · '
                     f'tokens={r["tokens_in"]}→{r["tokens_out"]}')
        lines.append(f'**Stop reason:** {r["stop_reason"]} · {len(r["actions"])} tool calls')
        lines.append('')

        item_nodes = 0
        item_with_et = 0
        for a in r['actions']:
            tool = a['tool']
            if tool == 'remember_batch':
                summaries = _summarize_remember_batch(a)
                lines.append(f'### `remember_batch` ({len(summaries)} nodes)')
                lines.append('')
                lines.append('| # | type | event_time | title | content? | situation? | kv | edges |')
                lines.append('|---|---|---|---|:---:|:---:|---|---:|')
                for i, n in enumerate(summaries, start=1):
                    et = n.get('event_time') or '—'
                    if n.get('event_time'):
                        item_with_et += 1
                    item_nodes += 1
                    lines.append(f"| {i} | `{n['type']}` | {et} | {n['title']!r} | "
                                 f"{'✓' if n['has_content'] else '✗'} | "
                                 f"{'✓' if n['has_situation'] else '✗'} | "
                                 f"{','.join(n['kv_fields']) or '—'} | {n['connects']} |")
                lines.append('')
            elif tool == 'revise_batch':
                revs = (a.get('input') or {}).get('revisions') or []
                lines.append(f'### `revise_batch` ({len(revs)} revisions)')
                for r2 in revs:
                    nid = (r2.get('node_id') or '')[:8]
                    fields = [k for k in r2 if k not in ('node_id',)]
                    et_change = r2.get('event_time')
                    lines.append(f'  - node_id={nid} fields={fields}'
                                 + (f' event_time={et_change}' if et_change else ''))
                lines.append('')
            elif tool == 'connect_batch':
                conns = (a.get('input') or {}).get('connections') or []
                lines.append(f'### `connect_batch` ({len(conns)} edges)')
                for c in conns[:8]:
                    lines.append(f'  - {(c.get("source") or "?")[:30]} '
                                 f'--{c.get("relation","?")}--> '
                                 f'{(c.get("target") or "?")[:30]}')
                lines.append('')
            else:
                lines.append(f'### `{tool}` action')
                lines.append('```')
                lines.append(json.dumps(a.get('input'), default=str, indent=2)[:600])
                lines.append('```')
                lines.append('')

        et_rate = (item_with_et / item_nodes * 100) if item_nodes else 0
        lines.append(f'**Per-item event_time rate:** {item_with_et}/{item_nodes} ({et_rate:.0f}%)')
        if r.get('final_text'):
            lines.append('')
            lines.append('**Encoder closing text:**')
            lines.append('> ' + r['final_text'].replace('\n', '\n> ')[:600])
        lines.append('')
        lines.append('---')
        lines.append('')

        total_nodes += item_nodes
        total_with_et += item_with_et

    cohort_rate = (total_with_et / total_nodes * 100) if total_nodes else 0
    lines.insert(1, f'**Cohort event_time rate:** {total_with_et}/{total_nodes} '
                    f'({cohort_rate:.0f}%) across {len(results)} items')
    lines.insert(2, '')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qids', required=True,
                   help='comma-separated qids to replay')
    p.add_argument('--prompt', required=True,
                   help='path to s1e prompt file (system prompt)')
    p.add_argument('--out', default=None, help='markdown output path')
    p.add_argument('--model', default='claude-sonnet-4-6')
    p.add_argument('--max-tokens', type=int, default=8000,
                   help='per-call max_tokens (default 8000 vs production 4000 '
                        'because single-call replay encodes the full conversation)')
    p.add_argument('--parallel', type=int, default=1,
                   help='concurrent Sonnet calls (default 1 = sequential). Each '
                        'call is ~90s for a 24-turn haystack; parallel=4 fans '
                        'them out so a 4-item replay finishes in ~95s wall '
                        'instead of ~6 min.')
    args = p.parse_args()

    load_env()

    # Need a brain for interaction lookup + scout dispatch. fresh_brain
    # creates a clean eval brain (seeded interactions, no production data).
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    import tempfile, os, shutil
    tmpdir = tempfile.mkdtemp(prefix='encoder_replay_')
    os.environ['BRAIN_DB_DIR'] = tmpdir
    brain = create_fresh_eval_brain(path=tmpdir, wipe=True)

    # Register the candidate prompt as the active s1e
    from tests.interaction_override import override_interaction
    system_prompt_raw = Path(args.prompt).read_text()
    override_interaction(brain, 's1e', template=system_prompt_raw,
                         parameters={'max_tokens': 4000, 'max_rounds': 5},
                         set_by='encoder_replay')
    try:
        system_prompt = system_prompt_raw
        qids = [q.strip() for q in args.qids.split(',') if q.strip()]
        results = []

        def _one(qid):
            try:
                r = replay_one(brain, qid, system_prompt,
                                 model=args.model, max_tokens=args.max_tokens)
                nodes_n = sum(len((a.get('input') or {}).get('nodes') or [])
                              for a in r['actions'] if a['tool'] == 'remember_batch')
                et_n = 0
                for a in r['actions']:
                    if a['tool'] == 'remember_batch':
                        for n in (a.get('input') or {}).get('nodes') or []:
                            if n.get('event_time'):
                                et_n += 1
                print(f'[{qid}] → {len(r["actions"])} tool calls, {nodes_n} nodes, '
                      f'{et_n} with event_time ({r["call_ms"]}ms, '
                      f'{r["tokens_in"]}→{r["tokens_out"]} tok)', flush=True)
                return r
            except Exception as e:
                print(f'[{qid}] FAILED: {type(e).__name__}: {e}', flush=True)
                return {'qid': qid, 'error': str(e)}

        if args.parallel > 1 and len(qids) > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f'[replay] launching {len(qids)} items with {args.parallel} workers',
                  flush=True)
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                # preserve qid order in results
                fut_by_qid = {qid: pool.submit(_one, qid) for qid in qids}
                results = [fut_by_qid[qid].result() for qid in qids]
        else:
            for i, qid in enumerate(qids, start=1):
                print(f'[{i}/{len(qids)}] {qid} ...', flush=True)
                results.append(_one(qid))

        md = render_report([r for r in results if 'error' not in r], args.prompt)
        if args.out:
            write_report(Path(args.out), md)
        else:
            print(md)
    finally:
        try:
            brain.save()
            brain.close()
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
