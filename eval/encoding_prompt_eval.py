#!/usr/bin/env python3
"""Encoding Prompt A/B Eval — compare v1 vs v2 encoding prompts.

Runs both prompts against the same conversation slices and compares:
- Number of tool calls (remember, revise, connect)
- Types chosen
- Fields filled (situation, reasoning, user_raw_quote, open fields)
- Quality: richness of content, connections made
- Questions asked (v2 feature)

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/encoding_prompt_eval.py
"""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))

import anthropic
from servers.brain import Brain
from servers.pipeline_contract import ENCODING_AGENT
from servers.brain_voice import BrainVoice


def _load_env():
    env_path = os.path.join(str(Path(__file__).resolve().parent.parent), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip()


def load_prompt(path):
    with open(path) as f:
        return f.read()


def gather_conversation(brain, session_id, limit=20):
    """Get conversation exchanges from message_stream."""
    rows = brain.logs_conn.execute(
        "SELECT role, content, signal_type, timestamp "
        "FROM message_stream WHERE session_id = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    return [{"role": r[0], "content": (r[1] or "")[:2000], "signal": r[2], "timestamp": r[3]}
            for r in reversed(rows)]


def gather_recall_context(brain, messages):
    """Independent recall based on conversation topics."""
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    query = " ".join(msg[:200] for msg in user_msgs[-3:])
    result = brain.recall(query=query, limit=5)
    results = result.get("results", [])
    if not results:
        return ""
    lines = []
    for r in results:
        c = {"id": r.get("id", ""), "type": r.get("type", ""),
             "title": r.get("title", ""), "content": r.get("content", ""),
             "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
             "revised_at": r.get("revised_at"), "created_at": r.get("created_at"),
             "_graph": r.get("_graph", {})}
        BrainVoice.format_node_deep(c, lines, conn=brain.conn, max_d1=3, max_d2=2, max_d3=1)
    return "\n".join(lines)


def build_user_content(messages, recall_context, counter):
    """Build the user message for the encoding agent."""
    msg_text = ""
    for m in messages:
        role = (m.get("role") or "?").upper()
        content = (m.get("content") or "")[:600]
        msg_text += "[%s]: %s\n\n" % (role, content)

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Previous State\nFirst run.\n\n"
    content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(messages), msg_text)
    if recall_context:
        content += "### Brain Context\n\n%s\n" % recall_context
    else:
        content += "### Brain Context\nNo recall data available.\n\n"
    return content


def get_tool_schemas():
    """Get encoding agent tool schemas."""
    from servers import brain_mcp
    ENCODING_TOOLS = {
        'recall', 'find_node_by_title', 'get_node',
        'remember', 'revise', 'connect',
        'record_divergence', 'learn_vocabulary',
        'remember_lesson', 'remember_mechanism',
        'remember_mental_model', 'remember_impact',
        'remember_convention',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]


def run_encoding(client, system_prompt, user_content, tools, label, brain=None):
    """Run one encoding pass. Reads execute against real brain, writes are dry run."""
    print("\n=== %s ===" % label)
    print("System prompt: %d chars" % len(system_prompt))
    print("User content: %d chars" % len(user_content))

    t0 = time.time()
    api_messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096,
        system=system_prompt, messages=api_messages, tools=tools)

    all_actions = []
    all_text = []
    rounds = 0

    for rounds in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text" and b.text.strip()]

        if text_blocks:
            all_text.extend(text_blocks)

        if not tool_uses:
            break

        # Execute reads (recall, find, get) against real brain.
        # Writes (remember, revise, connect) are captured but NOT executed.
        tool_results = []
        for tu in tool_uses:
            action = {"tool": tu.name, "input": tu.input}
            all_actions.append(action)

            read_tools = {'recall', 'find_node_by_title', 'get_node'}
            if tu.name in read_tools:
                # Real execution — agent gets actual brain data
                from servers.daemon_dispatch import COMMAND_TABLE
                entry = COMMAND_TABLE.get(tu.name)
                if entry:
                    result = entry.handler(brain, tu.input, [])
                    from servers import brain_mcp
                    result_text = brain_mcp._format_result(tu.name, result.get("result", {})) if result.get("ok") else "ERROR: %s" % result.get("error", "?")
                else:
                    result_text = "Unknown tool"
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": result_text
                })
            else:
                # Write tools: capture but don't execute
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": '{"ok": true, "result": {"id": "dryrun_%04d", "status": "captured (dry run)"}}'  % len(all_actions)
                })

        api_messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        api_messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            system=system_prompt, messages=api_messages, tools=tools)

    # Final text
    final_text = [b.text for b in response.content if b.type == "text" and b.text.strip()]
    all_text.extend(final_text)

    elapsed = time.time() - t0

    # Analyze
    remembers = [a for a in all_actions if a["tool"] in ("remember", "remember_lesson", "remember_mechanism", "remember_mental_model", "remember_impact", "remember_convention")]
    revises = [a for a in all_actions if a["tool"] == "revise"]
    connects = [a for a in all_actions if a["tool"] == "connect"]
    recalls = [a for a in all_actions if a["tool"] in ("recall", "find_node_by_title", "get_node")]

    print("\nResults (%d rounds, %.1fs):" % (rounds + 1, elapsed))
    print("  Actions: %d remember, %d revise, %d connect, %d recall" % (
        len(remembers), len(revises), len(connects), len(recalls)))

    # Types used
    types = [a["input"].get("type", "?") for a in remembers]
    print("  Types: %s" % (types if types else "(none)"))

    # Fields filled
    for i, a in enumerate(remembers):
        inp = a["input"]
        filled = [k for k in inp.keys() if k not in ("type", "title", "content", "keywords") and inp[k]]
        print("  Node %d: [%s] %s" % (i + 1, inp.get("type", "?"), inp.get("title", "?")[:60]))
        print("    content: %d chars" % len(inp.get("content", "")))
        if filled:
            print("    extra fields: %s" % ", ".join(filled))
        else:
            print("    extra fields: (none)")

    for i, a in enumerate(revises):
        inp = a["input"]
        print("  Revise %d: node=%s reason=%s" % (i + 1, inp.get("node_id", "?")[:8], inp.get("reason", "?")[:50]))

    for i, a in enumerate(connects):
        inp = a["input"]
        print("  Connect %d: %s -[%s]-> %s" % (i + 1, inp.get("source_id", "?")[:8], inp.get("relation", "?"), inp.get("target_id", "?")[:8]))

    # Questions (v2 feature)
    questions = [t for t in all_text if '?' in t]
    if questions:
        print("  Questions:")
        for q in questions:
            print("    %s" % q[:200])

    return {
        "actions": all_actions,
        "remembers": remembers,
        "revises": revises,
        "connects": connects,
        "recalls": recalls,
        "text": all_text,
        "rounds": rounds + 1,
        "elapsed": elapsed,
        "types": types,
    }


def main():
    _load_env()
    brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))
    client = anthropic.Anthropic()
    tools = get_tool_schemas()

    project_dir = str(Path(__file__).resolve().parent.parent)
    v1_prompt = load_prompt(os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent.md'))
    v2_prompt = load_prompt(os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent-v2.md'))

    # v1 production appends contract field summary
    try:
        from servers.contract import generate_field_summary
        v1_prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception:
        pass

    # Pick 3 conversation slices from different topics
    sessions = brain.logs_conn.execute(
        "SELECT session_id, COUNT(*) as c FROM message_stream "
        "GROUP BY session_id HAVING c >= 20 ORDER BY MAX(timestamp) DESC LIMIT 3"
    ).fetchall()

    if not sessions:
        print("No sessions with 20+ messages found")
        return

    for session_id, msg_count in sessions:
        print("\n" + "=" * 80)
        print("SESSION: %s (%d messages)" % (session_id[:12], msg_count))
        print("=" * 80)

        messages = gather_conversation(brain, session_id, limit=20)
        recall_context = gather_recall_context(brain, messages)
        user_content = build_user_content(messages, recall_context, 1)

        # Show conversation summary
        user_msgs = [m["content"][:100] for m in messages if m.get("role") == "user"]
        print("Topics: %s" % " | ".join(user_msgs[:3]))

        r1 = run_encoding(client, v1_prompt, user_content, tools, "V1 (current)", brain=brain)
        r2 = run_encoding(client, v2_prompt, user_content, tools, "V2 (new)", brain=brain)

        # Compare
        print("\n--- COMPARISON ---")
        print("  V1: %d remember, %d revise, %d connect | types: %s" % (
            len(r1["remembers"]), len(r1["revises"]), len(r1["connects"]), r1["types"]))
        print("  V2: %d remember, %d revise, %d connect | types: %s" % (
            len(r2["remembers"]), len(r2["revises"]), len(r2["connects"]), r2["types"]))

        v1_fields = sum(len([k for k in a["input"] if k not in ("type", "title", "content", "keywords") and a["input"][k]]) for a in r1["remembers"])
        v2_fields = sum(len([k for k in a["input"] if k not in ("type", "title", "content", "keywords") and a["input"][k]]) for a in r2["remembers"])
        print("  V1 extra fields total: %d | V2 extra fields total: %d" % (v1_fields, v2_fields))
        print("  V1 time: %.1fs | V2 time: %.1fs" % (r1["elapsed"], r2["elapsed"]))
        if r2.get("text"):
            print("  V2 text output (questions?): %s" % r2["text"][0][:200] if r2["text"] else "(none)")


if __name__ == "__main__":
    main()
