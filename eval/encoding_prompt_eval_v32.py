#!/usr/bin/env python3
"""Encoding Prompt Eval — v3 (inline nodes) vs v3.2 (node catalog + references).

Tests the new split format: rich node catalog at top, ID references in timeline.
Runs both formats against the same conversation and compares:
- Token count (prompt size)
- Rounds and time
- Actions: creates, revises, connects
- Field richness
- Whether the encoder uses catalog node IDs for connections

Uses IsolatedBrain — production databases are never touched.

Usage:
    python3 eval/encoding_prompt_eval_v32.py
"""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.isolated_brain import IsolatedBrain


def get_tool_schemas():
    from servers import brain_mcp
    ENCODING_TOOLS = {
        'recall', 'find_node_by_title', 'get_node',
        'remember', 'remember_batch', 'revise', 'connect',
        'record_divergence', 'learn_vocabulary',
        'remember_lesson', 'remember_mechanism',
        'remember_mental_model', 'remember_impact',
        'remember_convention',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]


def gather_messages(brain, session_id, limit=10):
    """Get messages with judge_output from message_stream."""
    from servers.pipeline_contract import ENCODING_AGENT
    rows = brain.logs_conn.execute(
        "SELECT id, role, content, signal_type, timestamp, recalled_raw, judge_output "
        "FROM message_stream WHERE session_id = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    return [{"id": r[0], "role": r[1],
             "content": (r[2] or "")[:ENCODING_AGENT['message_content_limit']],
             "signal": r[3], "timestamp": r[4],
             "recalled_raw": r[5], "judge_output": r[6]}
            for r in reversed(rows)]


def build_v3_content(brain, messages, counter, session_id):
    """V3 format: inline judge output per turn (current production)."""
    from servers.pipeline_contract import ENCODING_AGENT

    journal_key = 'encoding_journal_%s' % session_id
    journal = brain.get_config(journal_key, '') or 'First run.'
    prev_context = brain.get_config('session_context', '') or ''

    timeline = ""
    turn_num = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "user":
            turn_num += 1
            user_content = (m.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
            timeline += "[TURN %d]\nUSER: \"%s\"\n" % (turn_num, user_content)

            judge_output = m.get("judge_output")
            if judge_output and judge_output != '(no selection)':
                timeline += "BRAIN SURFACED (judge-selected):\n%s\n" % judge_output
            elif judge_output == '(no selection)':
                timeline += "BRAIN SURFACED: (none relevant)\n"
            else:
                timeline += "BRAIN SURFACED: (no recall data)\n"

            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                asst = (messages[i + 1].get("content") or "")[:ENCODING_AGENT['message_display_limit']]
                timeline += "ASSISTANT: \"%s\"\n" % asst
                i += 1
            timeline += "\n"
        i += 1

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Encoding Journal\n%s\n\n" % journal
    if prev_context:
        content += "### Session Context\n%s\n\n" % prev_context
    content += "### Conversation Timeline\n\n%s\n" % timeline
    return content


def build_v32_content(brain, messages, counter, session_id):
    """V3.2 format: node catalog at top + ID references in timeline."""
    from servers.encoding_agent import _build_user_content
    return _build_user_content(brain, messages, counter, session_id)


def run_encoding(client, system_prompt, user_content, tools, label, brain):
    """Run one encoding pass. Reads execute, writes are dry run."""
    print("\n  --- %s ---" % label)
    print("  Prompt: %d chars (~%d tokens)" % (len(user_content), len(user_content) // 4))

    t0 = time.time()
    api_messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096,
        system=system_prompt, messages=api_messages, tools=tools)

    all_actions = []
    rounds = 0
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    for rounds in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            all_actions.append({"tool": tu.name, "input": tu.input})

            read_tools = {'recall', 'find_node_by_title', 'get_node'}
            if tu.name in read_tools:
                from servers.daemon_dispatch import COMMAND_TABLE
                entry = COMMAND_TABLE.get(tu.name)
                if entry:
                    result = entry.handler(brain, tu.input, [])
                    from servers import brain_mcp
                    result_text = brain_mcp._format_result(tu.name, result.get("result", {})) if result.get("ok") else "ERROR"
                else:
                    result_text = "Unknown"
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
            else:
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                    "content": '{"ok": true, "result": {"id": "dryrun_%04d", "status": "captured"}}'  % len(all_actions)})

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

    remembers = [a for a in all_actions if a["tool"] in ("remember", "remember_batch", "remember_lesson", "remember_mechanism", "remember_mental_model", "remember_impact", "remember_convention")]
    revises = [a for a in all_actions if a["tool"] == "revise"]
    connects = [a for a in all_actions if a["tool"] == "connect"]
    recalls = [a for a in all_actions if a["tool"] in ("recall", "find_node_by_title", "get_node")]

    print("  Results: %d rounds, %.1fs" % (rounds + 1, elapsed))
    print("  Tokens: %d in, %d out" % (input_tokens, output_tokens))
    print("  Actions: %d create, %d revise, %d connect, %d recall" % (
        len(remembers), len(revises), len(connects), len(recalls)))

    for a in remembers:
        inp = a["input"]
        if a["tool"] == "remember_batch":
            nodes = inp.get("nodes", [])
            print("  BATCH: %d nodes" % len(nodes))
            for n in nodes:
                filled = [k for k in n if k not in ("type", "title", "content", "keywords") and n[k]]
                print("    [%s] %s — fields: %s" % (n.get("type", "?"), n.get("title", "?")[:50], filled or "(basic)"))
        else:
            filled = [k for k in inp if k not in ("type", "title", "content", "keywords") and inp[k]]
            print("  [%s] %s — fields: %s" % (inp.get("type", "?"), inp.get("title", "?")[:50], filled or "(basic)"))

    return {
        "rounds": rounds + 1, "elapsed": elapsed,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "remembers": len(remembers), "revises": len(revises),
        "connects": len(connects), "recalls": len(recalls),
        "actions": all_actions,
    }


def main():
    import anthropic

    with IsolatedBrain() as env:
        brain = env.brain
        client = anthropic.Anthropic()
        tools = get_tool_schemas()

        # Load system prompt (same for both variants)
        from servers.encoding_agent import _build_system_prompt
        system_prompt = _build_system_prompt()

        print("System prompt: %d chars" % len(system_prompt))
        print("Brain: %d nodes" % env.node_count())
        print("DB dir: %s (isolated)" % env.db_dir)

        # Pick session with most judge_output data
        sessions = brain.logs_conn.execute(
            "SELECT session_id, COUNT(*) as c, "
            "SUM(CASE WHEN judge_output IS NOT NULL AND judge_output != '(no selection)' THEN 1 ELSE 0 END) as judged "
            "FROM message_stream WHERE role='user' "
            "GROUP BY session_id HAVING judged >= 3 "
            "ORDER BY MAX(timestamp) DESC LIMIT 1"
        ).fetchall()

        if not sessions:
            print("No sessions with 3+ judge outputs found. Need judge data for v3.2 comparison.")
            return

        session_id, msg_count, judged = sessions[0]
        print("\nSession: %s (%d messages, %d with judge output)" % (session_id[:12], msg_count, judged))

        messages = gather_messages(brain, session_id, limit=10)
        user_msgs = [m["content"][:80] for m in messages if m.get("role") == "user"]
        print("Topics: %s" % " | ".join(user_msgs[:3]))

        # Build both prompt variants
        v3_content = build_v3_content(brain, messages, 1, session_id)
        v32_content = build_v32_content(brain, messages, 1, session_id)

        print("\n" + "=" * 60)
        print("V3 (inline nodes):  %d chars (~%d tokens)" % (len(v3_content), len(v3_content) // 4))
        print("V3.2 (catalog+ref): %d chars (~%d tokens)" % (len(v32_content), len(v32_content) // 4))
        print("Delta: %+d chars (%+.0f%%)" % (
            len(v32_content) - len(v3_content),
            (len(v32_content) - len(v3_content)) / len(v3_content) * 100 if v3_content else 0))
        print("=" * 60)

        r1 = run_encoding(client, system_prompt, v3_content, tools, "V3 (inline)", brain)
        r2 = run_encoding(client, system_prompt, v32_content, tools, "V3.2 (catalog+ref)", brain)

        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print("  %-20s %12s %12s" % ("", "V3", "V3.2"))
        print("  %-20s %12d %12d" % ("Prompt chars", len(v3_content), len(v32_content)))
        print("  %-20s %12d %12d" % ("Input tokens", r1["input_tokens"], r2["input_tokens"]))
        print("  %-20s %12d %12d" % ("Output tokens", r1["output_tokens"], r2["output_tokens"]))
        print("  %-20s %12d %12d" % ("Rounds", r1["rounds"], r2["rounds"]))
        print("  %-20s %12.1f %12.1f" % ("Time (s)", r1["elapsed"], r2["elapsed"]))
        print("  %-20s %12d %12d" % ("Creates", r1["remembers"], r2["remembers"]))
        print("  %-20s %12d %12d" % ("Revises", r1["revises"], r2["revises"]))
        print("  %-20s %12d %12d" % ("Connects", r1["connects"], r2["connects"]))
        print("  %-20s %12d %12d" % ("Recalls (extra)", r1["recalls"], r2["recalls"]))


if __name__ == "__main__":
    main()
