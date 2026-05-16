"""Multi-angle review of saved surface runs.

For each (qid, run_dir):
  1. Reconstructs the inputs the surface saw (prompt + candidates + query).
  2. Loads what actually happened (selection, tool trace, answerer
     hypothesis, judge verdict, gold).
  3. Asks Sonnet for a structured outside-reviewer critique on 6 axes.
  4. Asks Haiku to re-explain its own choice given the same inputs.

Output: markdown report with per-item analysis.

USE
    ./dev python3 -m eval.agent_introspect.surface_review \\
        --run-dir eval/longmem/reports/v15_11_v6_postfix \\
        --qids 54026fce,gpt4_385a5000,8e91e7d9,37f165cf,60bf93ed_abs \\
        --prompt eval/surface_v6_prompt.txt \\
        --out eval/longmem/reports/surface_review_v6_postfix.md
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


REVIEWER_MODEL = "claude-sonnet-4-5"
INTROSPECT_MODEL = "claude-haiku-4-5"


def _load_oracle_item(qid: str) -> Dict[str, Any]:
    oracle_path = ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json'
    for item in json.loads(oracle_path.read_text()):
        if item.get('question_id') == qid:
            return item
    raise KeyError(qid)


def _load_item_artifacts(run_dir: Path, qid: str) -> Dict[str, Any]:
    """Load every artifact for one item."""
    item_dir = run_dir / 'items' / qid
    if not item_dir.exists():
        raise FileNotFoundError(f"item dir missing: {item_dir}")

    out: Dict[str, Any] = {'qid': qid}
    out['meta'] = json.loads((item_dir / 'meta.json').read_text())
    out['result'] = json.loads((item_dir / 'result.json').read_text())
    out['recall'] = json.loads((item_dir / 'recall.json').read_text())

    nodes: Dict[str, Dict[str, Any]] = {}
    for line in (item_dir / 'nodes.jsonl').open():
        n = json.loads(line)
        nodes[n['id']] = n
    out['nodes_by_id'] = nodes

    # Candidates with full node content (the 25 the surface saw)
    cands = []
    for c in out['recall'].get('candidates') or []:
        cid_prefix = c.get('id', '')
        full = next((nodes[full_id] for full_id in nodes
                     if full_id.startswith(cid_prefix)), None)
        kv = (full or {}).get('kv') or {}
        cands.append({
            'id': c.get('id'),
            'title': (full or {}).get('title') or '',
            'type': (full or {}).get('type') or '',
            'content': (full or {}).get('content') or '',
            'score': c.get('score', 0.0),
            'situation': kv.get('situation') or '',
        })
    out['candidates'] = cands

    return out


def _format_candidates_block(cands: List[Dict[str, Any]], k: int = 25) -> str:
    """Format candidates the same way the surface saw them (approximately)."""
    lines = []
    for i, c in enumerate(cands[:k], start=1):
        lines.append(
            f"#{i} [{c['id'][:8]}] type={c['type']} score={c['score']:.2f}\n"
            f"   title: {c['title']}\n"
            f"   content: {(c['content'] or '')[:200]}"
        )
    return "\n".join(lines)


def _build_review_context(artifacts: Dict[str, Any]) -> str:
    """Compose the full context the Sonnet reviewer sees."""
    m = artifacts['meta']
    r = artifacts['result']
    rec = artifacts['recall']

    # Source of truth for selections is recall.json — result.json only
    # populates failure_evidence.selected_ids on failed items, so passing
    # items would otherwise look like "selected 0" to the reviewer.
    selected_ids = rec.get('selected') or []
    selected_details = []
    for sid in selected_ids:
        # selected list may be either full IDs or 8-char prefixes
        sid_prefix = sid[:8] if isinstance(sid, str) else ''
        match = next((c for c in artifacts['candidates']
                       if c['id'].startswith(sid_prefix)), None)
        if match:
            selected_details.append(
                f"  - {match['id'][:8]} ({match['type']}): {match['title']}"
            )

    tool_trace_text = ""
    for round_rec in (rec.get('tool_trace') or []):
        calls = round_rec.get('tool_calls') or []
        if not calls:
            tool_trace_text += (
                f"  Round {round_rec.get('round')}: "
                f"stop_reason={round_rec.get('stop_reason')}, no tool calls\n"
            )
        for call in calls:
            tool_trace_text += (
                f"  Round {round_rec.get('round')}: "
                f"{call.get('tool')}({call.get('args')}) -> "
                f"{call.get('result_count')} results\n"
            )
    if not tool_trace_text:
        tool_trace_text = "  (no tools fired)"

    return f"""=== QUESTION ===
{m.get('question')}

Q-date: {m.get('question_date')}
Axis: {m.get('axis')}

=== GOLD ANSWER ===
{m.get('gold')}

=== HAYSTACK SUMMARY ===
{len(m.get('haystack_session_ids', []))} session(s),
{m.get('haystack_turn_count', '?')} total turns.

=== WHAT SURFACE SAW (25 CANDIDATES) ===
{_format_candidates_block(artifacts['candidates'])}

=== WHAT SURFACE DID ===
Tool trace:
{tool_trace_text}

Selected ({len(selected_ids)}):
{chr(10).join(selected_details) if selected_details else '  (nothing selected)'}

=== WHAT ANSWERER PRODUCED ===
{r.get('hypothesis')}

abstained: {r.get('abstained')}
has_context: {r.get('has_context')}

=== JUDGE VERDICT ===
correct: {r.get('correct')}
failure_bucket: {r.get('failure_bucket') or '(passed)'}
failure_reason: {r.get('failure_reason') or '(passed)'}
"""


SONNET_SYSTEM = """You are an outside code reviewer auditing the behavior of a memory-retrieval pipeline. The pipeline has three layers:

  1. SURFACE (Haiku) — selects 3-5 nodes from ~25 cosine candidates, optionally augmenting via fetch tools.
  2. ANSWERER (Haiku) — composes a final answer from the selected nodes. Three valid response patterns: ANSWER (direct or simple-inference), PARTIAL (have related material, name what's missing), ABSTAIN (no relevant material).
  3. JUDGE (Sonnet) — scores against a gold answer string.

You will see one item: the question, the gold answer, the 25 candidates surface saw, what surface did (tool trace + selections), the answerer's hypothesis, and the judge's verdict.

Your job: critique on 8 axes. Be specific, cite candidate IDs and exact phrases. Be honest about whether the gold is fair.

Output sections, each 2-4 sentences:

1. **Prompt clarity** — was the surface system prompt clear enough to guide Haiku correctly here? Any contradictions or ambiguities you can name?
2. **Candidate quality** — were the 25 cosine candidates good for this question? Any that surface should obviously have picked?
3. **Selection logic** — given the candidates, was the surface's choice (including any tool calls) reasonable?
4. **Source neutrality** — if surface fired tools, were the selections balanced between original cosine candidates and tool-fetched ones, or did Haiku show a bias either way? Cite candidate IDs and their source.
5. **Answerer inference** — if the answerer abstained or partially-qualified, was there a simple inference (date comparison, sum, count, pick-earlier/larger) that would have produced a valid ANSWER? Did the answerer apply it or skip it?
6. **Answerer fit** — did the answerer's response shape (answer / partial / abstain) match what the candidates supported?
7. **Judge alignment** — is the judge's verdict fair given the haystack and the model's actual answer? Or is the gold overreaching / corrupt?
8. **Improvement direction** — what specific change to prompt, code, or eval would help cases like this?"""


HAIKU_INTROSPECT_SYSTEM = """You are Anchor's surface. You were just shown a query and 25 candidates from the brain. You returned a selection. Now reflect on your reasoning.

You will see: your system prompt, the query, the 25 candidates, your tool calls if any, and your final selection.

Walk through your reasoning. Be honest:
- What topic did you read in the operator's message?
- Which candidates did you consider seriously? Which did you dismiss?
- If you fired tools, why? What did you expect them to return?
- If you selected zero, what made you abstain?
- If you selected some, what made those the strongest picks?

3-6 sentences. First-person. Direct."""


def _call_anthropic(model: str, system: str, user: str, max_tokens: int = 2048) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if hasattr(b, "text")).strip()


def review_one(run_dir: Path, qid: str, surface_prompt: str,
                introspect: bool = True) -> Dict[str, Any]:
    """Build context, send to Sonnet, optionally to Haiku."""
    artifacts = _load_item_artifacts(run_dir, qid)
    ctx = _build_review_context(artifacts)

    t0 = time.time()
    sonnet_review = _call_anthropic(REVIEWER_MODEL, SONNET_SYSTEM, ctx,
                                     max_tokens=2048)
    sonnet_ms = int((time.time() - t0) * 1000)

    haiku_introspect = ""
    haiku_ms = 0
    if introspect:
        # For introspection, give Haiku its own prompt + the context (incl
        # selections so it knows "what it did"). It's a re-explanation,
        # not a re-run.
        intro_user = (
            f"SYSTEM PROMPT YOU WERE GIVEN:\n---\n{surface_prompt}\n---\n\n"
            f"WHAT YOU SAW AND DID:\n{ctx}"
        )
        t0 = time.time()
        haiku_introspect = _call_anthropic(INTROSPECT_MODEL,
                                            HAIKU_INTROSPECT_SYSTEM,
                                            intro_user, max_tokens=1024)
        haiku_ms = int((time.time() - t0) * 1000)

    return {
        'qid': qid,
        'question': artifacts['meta'].get('question'),
        'gold': artifacts['meta'].get('gold'),
        'verdict': 'PASS' if artifacts['result'].get('correct') else 'FAIL',
        'failure_bucket': artifacts['result'].get('failure_bucket'),
        'hypothesis': artifacts['result'].get('hypothesis'),
        'sonnet_review': sonnet_review,
        'sonnet_ms': sonnet_ms,
        'haiku_introspect': haiku_introspect,
        'haiku_ms': haiku_ms,
        'context_chars': len(ctx),
    }


def render_md(rows: List[Dict[str, Any]], prompt_path: str) -> str:
    out = [f'# Surface multi-angle review — `{prompt_path}`', '']
    out.append('Sonnet outside critique + Haiku self-explanation, per item.')
    out.append('')

    passes = sum(1 for r in rows if r.get('verdict') == 'PASS')
    out.append(f'**Reviewed:** {len(rows)} items, {passes} passing, '
                f'{len(rows) - passes} failing')
    out.append('')

    for r in rows:
        verdict_label = (f"**{r['verdict']}**"
                         if r['verdict'] == 'PASS'
                         else f"**FAIL** ({r.get('failure_bucket')})")
        out.append(f"## `{r['qid']}` — {verdict_label}")
        out.append('')
        out.append(f"**Q:** {r.get('question')}")
        out.append(f"**Gold:** {(r.get('gold') or '')[:300]}")
        out.append(f"**Hypothesis:** {(r.get('hypothesis') or '')[:400]}")
        out.append('')
        out.append('### Sonnet outside review')
        out.append(r.get('sonnet_review') or '(no review)')
        out.append('')
        out.append('### Haiku self-explanation')
        out.append(r.get('haiku_introspect') or '(no introspection)')
        out.append('')
        out.append('---')
        out.append('')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--qids', required=True, help='comma-separated qids')
    p.add_argument('--prompt', required=True, help='surface prompt path (for Haiku intro)')
    p.add_argument('--no-introspect', action='store_true',
                   help='skip the Haiku self-explanation call')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    # Load env so ANTHROPIC_API_KEY is reachable
    try:
        from servers.scales.dispatch import load_env
        load_env()
    except Exception:
        pass

    surface_prompt = Path(args.prompt).read_text()
    run_dir = Path(args.run_dir)
    qids = [q.strip() for q in args.qids.split(',') if q.strip()]

    rows = []
    for i, qid in enumerate(qids, start=1):
        print(f'[{i}/{len(qids)}] reviewing {qid} ...', flush=True)
        try:
            r = review_one(run_dir, qid, surface_prompt,
                            introspect=not args.no_introspect)
            print(f'  -> verdict={r["verdict"]} '
                  f'sonnet={r["sonnet_ms"]}ms '
                  f'haiku={r["haiku_ms"]}ms', flush=True)
            rows.append(r)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}', flush=True)
            rows.append({'qid': qid, 'error': str(e)})

    md = render_md(rows, args.prompt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f'wrote {args.out}', flush=True)
    else:
        print(md)


if __name__ == '__main__':
    main()
