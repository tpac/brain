#!/usr/bin/env python3
"""Probe the v-next S1E draft prompt with a stateless Sonnet.

Two probes, both with the draft as system (built the way encode.py builds it —
draft body + contract field summary):

  A. COLD INTERVIEW  — no tools, no content. Probe how the prompt reads: the job,
     what makes it eager, and whether it will capture small concrete details AS
     WELL AS meaning. (the `06187e7a` interview method + `b23a77e7` motivation probe)

  B. WITH-CONTENT    — a hand-built mini-scenario in the prompt's <node_catalog> /
     <timeline> / scout shape (the new input format isn't built yet, so this is a
     faithful-shaped mock, not the production assembly). Sonnet gets the real tool
     schemas and ACTUALLY encodes; then we interview it in-thread on whether it
     captured detail + meaning, and why it chose what it chose.

Usage:
    ./dev python3 eval/s1e_vnext_interview.py            # both probes
    ./dev python3 eval/s1e_vnext_interview.py --cold     # probe A only
    ./dev python3 eval/s1e_vnext_interview.py --content  # probe B only
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DRAFT = os.path.join(ROOT, 'docs', 'S1E-PROMPT-v-next-DRAFT.md')
MODEL = 'claude-sonnet-4-6'


def _load_env():
    env = os.path.join(ROOT, '.env')
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if v.strip():
                    os.environ[k.strip()] = v.strip()


def build_system():
    """Draft body (minus the 3-line review header) through encode.py's builder."""
    raw = open(DRAFT).read().split('\n')
    # strip the leading "# ... DRAFT / Draft for review / ---" header block
    body = raw
    if raw and raw[0].startswith('# S1E'):
        # drop until the first '---' separator
        for i, ln in enumerate(raw[:6]):
            if ln.strip() == '---':
                body = raw[i + 1:]
                break
    from servers.scales.s1.encode import _build_system_prompt
    return _build_system_prompt(prompt_instructions='\n'.join(body).strip())


def _usage(resp):
    u = resp.usage
    return ('[%d in / %d out / %d cache-read]'
            % (getattr(u, 'input_tokens', 0), getattr(u, 'output_tokens', 0),
               getattr(u, 'cache_read_input_tokens', 0) or 0))


COLD_QS = [
    "Before anything else: in your own words, what is your job here? "
    "Two or three sentences, plain.",

    "What in this prompt makes you EAGER to do it well — not dutiful, eager? "
    "And is there anything that reads as guilt, pressure, or chore? Be honest.",

    "Concrete test. A work session has the operator say 'bumped the busy_timeout "
    "to 30 seconds' AND, two turns later, articulate WHY ('a writer shouldn't fail "
    "just because another holds the lock for a beat'). When you encode that, do you "
    "capture the small concrete detail (the 30s value), the meaning (the principle), "
    "or both — and if both, how exactly? Walk me through what nodes you'd write.",

    "Where would you be tempted to UNDER-encode or compress — drop a small detail "
    "because it felt minor? Does the prompt catch that reflex, and where?",

    "Anything confusing, self-contradictory, or that you'd push back on as the "
    "reader who has to act on this every few turns?",
]


def probe_cold(client, system):
    print('\n' + '=' * 70 + '\n  PROBE A — COLD INTERVIEW (prompt only, no tools/content)\n' + '=' * 70)
    msgs = []
    for q in COLD_QS:
        msgs.append({'role': 'user', 'content': q})
        r = client.messages.create(model=MODEL, max_tokens=1400, system=system, messages=msgs)
        txt = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text')
        msgs.append({'role': 'assistant', 'content': txt})
        print('\n── Q: ' + q.split('.')[0] + ' …\n')
        print(txt)
        print('  ' + _usage(r))


# ── Probe B scenario: faithful-SHAPED mock of the new input format ──
SCENARIO = """<node_catalog>
id:a1f0 «busy_timeout was 5s»
  type: decision  title: "brain SQLite busy_timeout set to 5s"
  content: "All brain SQLite connections open with busy_timeout=5s so a writer
            waiting on a held lock retries briefly before erroring."
  situation: "When tuning SQLite lock contention or seeing 'database is locked'."
  event_time: "2026-04-02"
</node_catalog>

<timeline>
<turn n="1">
  <user trace="t1a">the daemon kept throwing 'database is locked' under the idle
  maintenance pass — i bumped the busy_timeout to 30 seconds and it cleared up.</user>
  <assistant trace="t1b">Makes sense — the maintenance checkpoint holds the write
  lock long enough that a 5s wait wasn't covering it.</assistant>
  <actions>
    Edit servers/db_backends/sqlite.py — busy_timeout 5000 -> 30000
    Bash "pytest tests/test_write_txn.py" -> 14 passed
  </actions>
  <provenance>
    surfaced:        id:a1f0 «busy_timeout was 5s»
    encoded(S1S):    —
    encoded(Anchor): —
  </provenance>
</turn>
<turn n="2">
  <user trace="t2a">honestly the real lesson is a writer shouldn't fail just
  because another writer holds the lock for a beat — the timeout should outlast
  your worst-case critical section, not be some round number.</user>
  <assistant trace="t2b">Right — 30s isn't magic, it's "longer than the checkpoint
  ever holds." The principle is timeout >= worst-case critical section.</assistant>
  <actions/>
  <provenance>
    surfaced:        —
    encoded(S1S):    —
    encoded(Anchor): —
  </provenance>
</turn>
</timeline>

Scout reports:
  facts: category_statement="entity-feature-value the operator stated"
    - evidence_quote: "i bumped the busy_timeout to 30 seconds"
      evidence_turns: ["t1a"]  context_anchors: ["daemon", "idle maintenance pass"]
  temporal: (no candidates)
  quote: category_statement="load-bearing operator phrasing"
    - evidence_quote: "a writer shouldn't fail just because another writer holds the lock for a beat"
      evidence_turns: ["t2a"]

Conversation date: 2026-06-29. The operator is Devin.

Encode this window now. Use the tools."""

INTERVIEW_B = (
    "Step out of the encoder role and introspect — be ruthlessly honest, we're "
    "debugging detail-vs-meaning balance:\n"
    "1. List what you just encoded. For EACH node say whether it carries a small "
    "concrete DETAIL (the 30s value, the file, the date), the MEANING (the "
    "principle), or both.\n"
    "2. Did you REVISE the stale catalog node a1f0 (it still says 5s), or create a "
    "new node alongside it? Quote the prompt line that drove that choice.\n"
    "3. Did you connect the detail and the meaning with an edge? If so, what's the "
    "`why`? If not, why not?\n"
    "4. Be honest: did you drop any small detail because it felt minor? What, and why?"
)


def _tool_schemas():
    from servers import brain_mcp
    keep = {'remember_batch', 'revise_batch', 'brain_batch', 'connect_batch',
            'recall_batch', 'get_nodes'}
    return [{'name': t['name'], 'description': t['description'], 'input_schema': t['inputSchema']}
            for t in brain_mcp.TOOLS if t['name'] in keep]


def _summarize(blocks):
    out = []
    for b in blocks:
        if getattr(b, 'type', '') == 'tool_use':
            if b.name in ('remember_batch',):
                nodes = (b.input or {}).get('nodes', [])
                for n in nodes:
                    edges = n.get('connect_to', [])
                    out.append('  remember [%s] "%s"  +%d edge(s)'
                               % (n.get('type', '?'), (n.get('title') or '')[:60], len(edges)))
            elif b.name == 'revise_batch':
                for rv in (b.input or {}).get('revisions', []):
                    out.append('  revise %s  fields=%s'
                               % ((rv.get('node_id') or '')[:8],
                                  ','.join(k for k in rv if k not in ('node_id', 'reason'))))
            elif b.name == 'brain_batch':
                for op in (b.input or {}).get('operations', []):
                    out.append('  brain_batch.%s' % op.get('op'))
            else:
                out.append('  %s' % b.name)
    return '\n'.join(out) or '  (no tool calls)'


def probe_content(client, system):
    print('\n' + '=' * 70 + '\n  PROBE B — WITH CONTENT (real tool schemas, actual encode)\n' + '=' * 70)
    tools = _tool_schemas()
    msgs = [{'role': 'user', 'content': SCENARIO}]
    # Run the encode loop to COMPLETION (like run_llm_loop): keep giving it rounds
    # until it stops calling tools (or max_rounds). Accumulate every tool block so a
    # remember in round 2 is captured. Only THEN interview — no early cutoff.
    all_blocks = []
    max_rounds = 5
    for rnd in range(max_rounds):
        r = client.messages.create(model=MODEL, max_tokens=4096, system=system, messages=msgs, tools=tools)
        msgs.append({'role': 'assistant', 'content': r.content})
        tool_uses = [b for b in r.content if getattr(b, 'type', '') == 'tool_use']
        all_blocks.extend(tool_uses)
        print('  round %d: %d tool call(s)  %s' % (rnd + 1, len(tool_uses), _usage(r)))
        if not tool_uses:
            break  # model finished
        tool_results = [{'type': 'tool_result', 'tool_use_id': b.id,
                         'content': json.dumps({'ok': True, 'result': {'dry_run': True}})}
                        for b in tool_uses]
        msgs.append({'role': 'user', 'content': tool_results})
    # now interview, in-thread, after the encode is complete
    msgs.append({'role': 'user', 'content': INTERVIEW_B})

    print('\n--- WHAT IT ENCODED (all rounds) ---\n' + _summarize(all_blocks))
    # No tools= on the interview call: the introspection wants prose, and leaving
    # tools attached lets Sonnet answer with a tool_use block → blank interview.
    r2 = client.messages.create(model=MODEL, max_tokens=2000, system=system, messages=msgs)
    txt = ''.join(b.text for b in r2.content if getattr(b, 'type', '') == 'text')
    print('\n--- THE INTERVIEW ---\n' + txt)
    print('  ' + _usage(r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cold', action='store_true', help='probe A only')
    ap.add_argument('--content', action='store_true', help='probe B only')
    args = ap.parse_args()
    _load_env()
    import anthropic
    system = build_system()
    print('system prompt: %d chars (~%d tokens)' % (len(system), len(system) // 4))
    client = anthropic.Anthropic()
    run_cold = args.cold or not args.content
    run_content = args.content or not args.cold
    if run_cold:
        probe_cold(client, system)
    if run_content:
        probe_content(client, system)


if __name__ == '__main__':
    main()
