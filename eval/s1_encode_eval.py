#!/usr/bin/env python3
"""Encoding Prompt Eval — tests S1 encoder quality.

Runs the encoder against a real conversation from the brain and measures:
- Token count (prompt + output)
- Rounds and time
- Actions: creates, revises, connects, recalls (extra)
- Field richness per created node
- Whether encoder uses catalog node IDs vs extra recall calls

Uses IsolatedBrain — production databases are never touched.

Usage:
    python3 eval/encoding_prompt_eval_v32.py
    python3 eval/encoding_prompt_eval_v32.py --session <session_id>
"""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.isolated_brain import IsolatedBrain


def get_tool_schemas(tool_set='new'):
    """S1 encoding tool subset."""
    from servers import brain_mcp
    if tool_set == 'old':
        names = {
            'recall', 'find_node_by_title', 'get_node',
            'remember', 'remember_batch', 'revise', 'revise_batch', 'connect',
            'record_divergence', 'learn_vocabulary',
        }
    else:
        names = {
            'remember_batch', 'revise_batch',
            'brain_batch', 'connect_batch',
            'recall_batch', 'get_nodes',
        }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in names]


def gather_messages(brain, session_id, limit=20):
    """Get messages from trace_events (S0 turns) — matches production pipeline."""
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    try:
        turns = brain._trace_dal.get_session_turns(session_id, limit=limit)
        if turns:
            for i, t in enumerate(turns):
                t['id'] = 'turn-%d' % i
                t['content'] = (t.get('content', '') or '')[:ENCODING_AGENT['message_content_limit']]
            return turns
    except Exception as e:
        print('[eval] TRACE READ ERROR: %s' % e)

    # Fallback: try message_stream (for older sessions)
    from servers.scales.s1.encode_contract import ENCODING_AGENT as cfg
    try:
        rows = brain.logs_conn.execute(
            "SELECT id, role, content, signal_type, timestamp, recalled_raw, judge_output "
            "FROM message_stream WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [{"id": r[0], "role": r[1],
                 "content": (r[2] or "")[:cfg['message_content_limit']],
                 "signal": r[3], "timestamp": r[4],
                 "recalled_raw": r[5], "judge_output": r[6]}
                for r in reversed(rows)]
    except Exception:
        return []


def build_content(brain, messages, counter, session_id):
    """Build encoder prompt using production code path."""
    from servers.scales.s1.encode import _build_user_content
    return _build_user_content(brain, messages, counter, session_id)


def build_system_prompt(prompt_file=None):
    """Build system prompt. If prompt_file given, use that instead of production."""
    from servers.scales.s1.encode import _build_system_prompt
    if prompt_file:
        with open(prompt_file) as f:
            prompt_text = f.read()
        return _build_system_prompt(prompt_instructions=prompt_text)
    return _build_system_prompt()


def run_encoding(client, system_prompt, user_content, tools, brain):
    """Run one encoding pass. Reads execute, writes are dry run."""
    t0 = time.time()
    api_messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096,
        system=system_prompt, messages=api_messages, tools=tools)

    all_actions = []
    rounds = 0
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    final_text = ''

    for rounds in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]
        if text_blocks:
            final_text = text_blocks[-1].text

        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            all_actions.append({"tool": tu.name, "input": tu.input})

            read_tools = {'recall', 'find_node_by_title', 'get_node', 'recall_batch', 'get_nodes'}
            if tu.name in read_tools:
                from servers.daemon_dispatch import COMMAND_TABLE
                entry = COMMAND_TABLE.get(tu.name)
                if entry:
                    result = entry.handler(brain, tu.input, [])
                    from servers import brain_mcp
                    result_text = brain_mcp._format_result(tu.name, result.get("result", {})) if result.get("ok") else "ERROR"
                else:
                    result_text = "Unknown tool"
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
            else:
                # Dry run — simulate success
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                    "content": '{"ok": true, "result": {"id": "dryrun_%04d", "status": "captured"}}' % len(all_actions)})

        api_messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        api_messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            system=system_prompt, messages=api_messages, tools=tools)
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

    elapsed = time.time() - t0

    # Categorize actions
    creates = [a for a in all_actions if a["tool"] in ("remember_batch",)]
    revises = [a for a in all_actions if a["tool"] in ("revise", "revise_batch")]
    connects = [a for a in all_actions if a["tool"] == "connect"]
    recalls = [a for a in all_actions if a["tool"] in ("recall", "find_node_by_title", "get_node")]

    # Count individual nodes created
    node_count = 0
    rich_fields = []
    for a in creates:
        nodes = a["input"].get("nodes", [])
        node_count += len(nodes)
        for n in nodes:
            filled = [k for k in n if k not in ("type", "title", "content", "keywords") and n[k]]
            rich_fields.append(filled)

    # Count revisions
    revision_count = 0
    for a in revises:
        if a["tool"] == "revise_batch":
            revision_count += len(a["input"].get("revisions", []))
        else:
            revision_count += 1

    return {
        "rounds": rounds + 1, "elapsed": elapsed,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_chars": len(user_content),
        "node_count": node_count, "revision_count": revision_count,
        "connect_count": len(connects), "recall_count": len(recalls),
        "rich_fields": rich_fields,
        "actions": all_actions,
        "final_text": final_text,
    }


def pick_session(brain, requested_session=None):
    """Find the best session to test against."""
    if requested_session:
        return requested_session

    # Try trace_events first (newer sessions)
    try:
        rows = brain._trace_dal.conn.execute(
            "SELECT session_id, COUNT(*) as c "
            "FROM trace_events WHERE scale = 's1' AND event_type = 'O' AND ref_type = 'recall' "
            "GROUP BY session_id HAVING c >= 3 "
            "ORDER BY MAX(created_at) DESC LIMIT 1"
        ).fetchall()
        if rows:
            return rows[0][0]
    except Exception:
        pass

    # Fallback: message_stream
    try:
        rows = brain.logs_conn.execute(
            "SELECT session_id, COUNT(*) as c, "
            "SUM(CASE WHEN judge_output IS NOT NULL AND judge_output != '(no selection)' THEN 1 ELSE 0 END) as judged "
            "FROM message_stream WHERE role='user' "
            "GROUP BY session_id HAVING judged >= 3 "
            "ORDER BY MAX(timestamp) DESC LIMIT 1"
        ).fetchall()
        if rows:
            return rows[0][0]
    except Exception:
        pass

    return None


def print_result(label, result):
    """Print a single eval result."""
    print("\n  --- %s ---" % label)
    print("  %-25s %s" % ("Rounds", result["rounds"]))
    print("  %-25s %.1fs" % ("Time", result["elapsed"]))
    print("  %-25s %d" % ("Prompt chars", result["prompt_chars"]))
    print("  %-25s %d" % ("Input tokens", result["input_tokens"]))
    print("  %-25s %d" % ("Output tokens", result["output_tokens"]))
    print("  %-25s %d" % ("Total tokens", result["total_tokens"]))
    print("  %-25s %d" % ("Nodes created", result["node_count"]))
    print("  %-25s %d" % ("Nodes revised", result["revision_count"]))
    print("  %-25s %d" % ("Connections made", result["connect_count"]))
    print("  %-25s %d" % ("Extra recall calls", result["recall_count"]))

    if result["rich_fields"]:
        all_fields_used = set()
        for fields in result["rich_fields"]:
            all_fields_used.update(fields)
        richness = sum(len(f) for f in result["rich_fields"]) / len(result["rich_fields"]) if result["rich_fields"] else 0
        print("  %-25s %.1f" % ("Avg rich fields/node", richness))
        print("  %-25s %s" % ("Fields used", sorted(all_fields_used)))

    # KPI
    rounds_ok = result["rounds"] <= 2
    recalls_ok = result["recall_count"] == 0
    nodes_ok = result["node_count"] >= 1
    print("  KPIs: rounds=%d [%s] recalls=%d [%s] nodes=%d [%s]" % (
        result["rounds"], "OK" if rounds_ok else "FAIL",
        result["recall_count"], "OK" if recalls_ok else "FAIL",
        result["node_count"], "OK" if nodes_ok else "FAIL"))


def main():
    import anthropic
    # Load .env for API key
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

    requested_session = None
    compare_prompt = None
    if '--session' in sys.argv:
        idx = sys.argv.index('--session')
        if idx + 1 < len(sys.argv):
            requested_session = sys.argv[idx + 1]
    if '--compare' in sys.argv:
        idx = sys.argv.index('--compare')
        if idx + 1 < len(sys.argv):
            compare_prompt = sys.argv[idx + 1]

    with IsolatedBrain() as env:
        brain = env.brain
        client = anthropic.Anthropic()
        tools_new = get_tool_schemas('new')
        tools_old = get_tool_schemas('old')

        system_prompt = build_system_prompt()
        print("System prompt: %d chars (~%d tokens)" % (len(system_prompt), len(system_prompt) // 4))
        print("New tools: %d (%s)" % (len(tools_new), ', '.join(t['name'] for t in tools_new)))
        print("Old tools: %d (%s)" % (len(tools_old), ', '.join(t['name'] for t in tools_old)))
        print("Brain: %d nodes (isolated)" % env.node_count())

        session_id = pick_session(brain, requested_session)
        if not session_id:
            print("No suitable session found. Need a session with 3+ judge outputs.")
            return

        messages = gather_messages(brain, session_id)
        user_msgs = [m["content"][:80] for m in messages if m.get("role") == "user"]
        print("\nSession: %s" % session_id[:16])
        print("Messages: %d (%d turns)" % (len(messages), len(user_msgs)))
        print("Topics: %s" % " | ".join(user_msgs[:4]))

        user_content = build_content(brain, messages, 5, session_id)
        print("\nPrompt: %d chars (~%d tokens)" % (len(user_content), len(user_content) // 4))

        # Run A (current prompt)
        print("\n" + "=" * 60)
        print("RUNNING: current prompt")
        print("=" * 60)
        result_a = run_encoding(client, system_prompt, user_content, tools_new, brain)
        print_result("CURRENT (new prompt + new tools)", result_a)

        # Run B (comparison: old prompt + old tools)
        result_b = None
        if compare_prompt:
            system_prompt_b = build_system_prompt(prompt_file=compare_prompt)
            print("\n" + "=" * 60)
            print("RUNNING: old prompt + old tools")
            print("=" * 60)
            result_b = run_encoding(client, system_prompt_b, user_content, tools_old, brain)
            print_result("OLD (old prompt + old tools)", result_b)

            # Side-by-side
            print("\n" + "=" * 60)
            print("COMPARISON")
            print("=" * 60)
            print("  %-25s %12s %12s" % ("", "Current", "Compare"))
            for k in ["rounds", "total_tokens", "node_count", "revision_count", "recall_count"]:
                print("  %-25s %12s %12s" % (k, result_a[k], result_b[k]))

        # Journal output (current)
        if result_a.get("final_text"):
            print("\n  --- ENCODER OUTPUT ---")
            for line in result_a["final_text"].split('\n'):
                print("  %s" % line)


if __name__ == "__main__":
    main()
