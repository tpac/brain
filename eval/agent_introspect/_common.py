"""Shared helpers across agent_introspect probes."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


SONNET_MODEL = 'claude-sonnet-4-5'
OPUS_MODEL = 'claude-opus-4-8'  # independent stronger scorer (avoids Sonnet-grades-Sonnet bias)
DEFAULT_MAX_TOKENS = 2000


def load_env():
    """Best-effort .env loader so the script works outside the daemon."""
    if os.environ.get('ANTHROPIC_API_KEY'):
        return
    envf = Path('.env')
    if envf.exists():
        for line in envf.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def call_sonnet(system: str, user: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = SONNET_MODEL,
                 temperature: float = None) -> Dict[str, Any]:
    """One stateless Sonnet call. Returns dict with text + usage + elapsed.

    Pass temperature=0 for an evaluator/grader so scoring noise doesn't add to
    whatever variance the thing being scored already has.
    """
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        **({'temperature': temperature} if temperature is not None else {}),
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    text = ''
    for block in resp.content:
        if hasattr(block, 'text'):
            text += block.text
    return {
        'text': text,
        'tokens_in': resp.usage.input_tokens,
        'tokens_out': resp.usage.output_tokens,
        'elapsed_ms': elapsed_ms,
        'model': model,
    }


def format_actions_for_review(action_details: List[Dict[str, Any]]) -> str:
    """Render encoder action_details into a human-readable block for
    Sonnet to inspect. One entry per tool call."""
    lines = []
    for i, ad in enumerate(action_details, start=1):
        tool = ad.get('tool', '?')
        lines.append(f'\n— Action {i}: {tool} —')
        node_ids = ad.get('node_ids') or []
        if node_ids:
            lines.append(f'  resulted in node_ids: {node_ids}')
        input_data = ad.get('input') or {}
        if tool == 'remember_batch' and 'nodes' in input_data:
            for j, n in enumerate(input_data['nodes'], start=1):
                lines.append(f'  node[{j}]: type={n.get("type","?")!r} title={n.get("title","")!r}')
                # Show all top-level keys to make field presence visible
                for k in sorted(n.keys()):
                    if k in ('type', 'title'):
                        continue
                    v = n[k]
                    if isinstance(v, str):
                        v_show = v if len(v) < 200 else v[:197] + '...'
                        lines.append(f'    {k}: {v_show!r}')
                    elif isinstance(v, list):
                        lines.append(f'    {k}: list[{len(v)}]')
                        for sub in v[:3]:
                            lines.append(f'      - {json.dumps(sub, default=str)[:200]}')
                    else:
                        lines.append(f'    {k}: {json.dumps(v, default=str)[:200]}')
        elif tool == 'brain_batch' and 'operations' in input_data:
            for j, op in enumerate(input_data['operations'], start=1):
                op_name = op.get('op', '?')
                lines.append(f'  op[{j}]: {op_name} {json.dumps({k:v for k,v in op.items() if k != "op"}, default=str)[:300]}')
        elif tool == 'revise_batch' and 'revisions' in input_data:
            for j, r in enumerate(input_data['revisions'], start=1):
                lines.append(f'  revision[{j}]: node_id={r.get("node_id","?")[:8]} {json.dumps(r, default=str)[:300]}')
        else:
            lines.append(f'  input: {json.dumps(input_data, default=str)[:400]}')
    return '\n'.join(lines)


def load_item_artifact(run_dir: str, qid: str) -> Dict[str, Any]:
    """Load meta + traces + nodes for an eval item. Returns dict with:
      meta, action_details (list across all s1e runs), conversation,
      encoder_prompt_active_version.
    """
    item_path = Path(run_dir) / 'items' / qid
    meta = json.loads((item_path / 'meta.json').read_text())

    # Extract conversation from the oracle (only meta carries qid)
    oracle_path = Path('eval/longmem/data/longmemeval_oracle.json')
    conversation = []
    if oracle_path.exists():
        for item in json.loads(oracle_path.read_text()):
            if item['question_id'] == qid:
                for sess_idx, sess in enumerate(item.get('haystack_sessions', [])):
                    for turn in sess:
                        conversation.append({
                            'role': turn.get('role',''),
                            'content': turn.get('content',''),
                            'session_idx': sess_idx,
                        })
                break

    # Collect every encoding_run action_details
    action_details = []
    for line in (item_path / 'traces.jsonl').open():
        t = json.loads(line)
        if t.get('event_type') == 'delta' and t.get('ref_type') == 'encoding_run':
            m = t.get('metadata') or {}
            if isinstance(m, str):
                try: m = json.loads(m)
                except: m = {}
            for ad in (m.get('action_details') or []):
                action_details.append(ad)

    # Resolve which s1e prompt was active (from interactions.jsonl)
    encoder_prompt = ''
    try:
        for line in (item_path / 'interactions.jsonl').open():
            ix = json.loads(line)
            if ix.get('name') == 's1e':
                # Take the latest version
                if not encoder_prompt or ix.get('version', 0) > getattr(load_item_artifact, '_v', 0):
                    encoder_prompt = ix.get('template', '')
    except FileNotFoundError:
        pass

    return {
        'qid': qid,
        'meta': meta,
        'conversation': conversation,
        'action_details': action_details,
        'encoder_prompt': encoder_prompt,
    }


def build_context_block(meta: Dict[str, Any]) -> str:
    """Render the temporal context the encoder ran in.

    Encoder uses conversation_now (= the haystack session's date for eval
    items) to resolve relative phrases. Without this, auditors cannot
    judge anchoring rules — 'is event_time the right date for X' is
    undefined when the encoder's notion of "today" is unknown.

    Used by every agent-introspect probe whose audit involves date
    resolution or temporal rule compliance. Shared here so probes don't
    drift on what "temporal context" means.
    """
    haystack_dates = (meta or {}).get('haystack_dates') or []
    # conversation_now = haystack session date (the date the encoder
    # treated as "today" when ingesting these turns). For multi-session
    # haystacks the encoder re-anchors per session; for the eval cohort
    # most items are single-session.
    conv_now = haystack_dates[-1] if haystack_dates else '(unknown)'
    question_date = (meta or {}).get('question_date', '(unknown)')
    return (f'conversation_now (encoder treats this as "today"): {conv_now}\n'
            f'haystack_dates (all sessions in chronological order): '
            f'{haystack_dates or "(unknown)"}\n'
            f'question_date (when the eval question was asked, AFTER '
            f'ingest): {question_date}')


def write_report(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f'wrote {path}')


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f'wrote {path}')
