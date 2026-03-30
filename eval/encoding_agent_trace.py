#!/usr/bin/env python3
"""Encoding Agent Full Trace — profile every action the LLM takes.

Runs the encoding agent against real conversation data with LIVE brain access
(reads work, writes are dry-run). Captures:
- Every Sonnet API round (input tokens, output tokens, time)
- Every tool call: name, input, output, success/failure, time
- Every error the agent encounters and how it recovers
- What it tried to do, what it fell back to
- Text reasoning between tool calls

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/encoding_agent_trace.py
"""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))

# Load API key
env_path = os.path.join(str(Path(__file__).resolve().parent.parent), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

import anthropic
from servers.brain import Brain
from servers.pipeline_contract import ENCODING_AGENT
from servers.brain_voice import BrainVoice


def gather_messages(brain, session_id, limit=20):
    rows = brain.logs_conn.execute(
        "SELECT role, content, signal_type, timestamp "
        "FROM message_stream WHERE session_id = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    return [{"role": r[0], "content": (r[1] or "")[:2000], "signal": r[2], "timestamp": r[3]}
            for r in reversed(rows)]


def gather_recall_context(brain, messages):
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


def build_prompt(brain, messages, recall_context, counter):
    from servers.pipeline_contract import ENCODING_AGENT
    msg_text = ""
    for m in messages:
        role = (m.get("role") or "?").upper()
        content = (m.get("content") or "")[:600]
        msg_text += "[%s]: %s\n\n" % (role, content)

    previous_state = brain.get_config('encoding_agent_state', '') or 'First run.'
    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Previous State\n%s\n\n" % previous_state
    content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(messages), msg_text)
    if recall_context:
        content += "### Brain Context\n\n%s\n" % recall_context
    else:
        content += "### Brain Context\nNo recall data available.\n\n"
    return content


def get_tools():
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


def main():
    brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))
    client = anthropic.Anthropic()

    # Load system prompt
    project_dir = str(Path(__file__).resolve().parent.parent)
    prompt_path = os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent.md')
    with open(prompt_path) as f:
        system_prompt = f.read()

    tools = get_tools()

    # Get most recent session with enough messages
    sessions = brain.logs_conn.execute(
        "SELECT session_id, COUNT(*) as c FROM message_stream "
        "GROUP BY session_id HAVING c >= 10 ORDER BY MAX(timestamp) DESC LIMIT 1"
    ).fetchall()

    if not sessions:
        print("No sessions with 10+ messages")
        return

    session_id = sessions[0][0]
    messages = gather_messages(brain, session_id)

    print("=" * 70)
    print("ENCODING AGENT FULL TRACE")
    print("=" * 70)
    print("Session: %s (%d messages)" % (session_id[:12], len(messages)))
    print("System prompt: %d chars" % len(system_prompt))
    print("Tools: %d available" % len(tools))
    print()

    # Gather context
    t0 = time.time()
    recall_context = gather_recall_context(brain, messages)
    recall_ms = int((time.time() - t0) * 1000)
    print("[PREP] Recall context: %d chars in %dms" % (len(recall_context), recall_ms))

    user_content = build_prompt(brain, messages, recall_context, 99)
    print("[PREP] User content: %d chars" % len(user_content))
    print()

    # Run the agent
    api_messages = [{"role": "user", "content": user_content}]
    total_input_tokens = 0
    total_output_tokens = 0
    all_actions = []
    errors = []

    for round_num in range(10):
        print("-" * 50)
        print("[ROUND %d] Calling Sonnet..." % (round_num + 1))

        t1 = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            system=system_prompt, messages=api_messages, tools=tools)
        api_ms = int((time.time() - t1) * 1000)

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        print("[ROUND %d] %dms | in:%d out:%d | stop:%s" % (
            round_num + 1, api_ms, response.usage.input_tokens,
            response.usage.output_tokens, response.stop_reason))

        # Process response blocks
        tool_uses = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print("[THINK] %s" % block.text[:300])
                if len(block.text) > 300:
                    print("  ... (%d chars total)" % len(block.text))
            elif block.type == "tool_use":
                tool_uses.append(block)

        if not tool_uses:
            print("[DONE] No tool calls — agent finished")
            break

        # Execute each tool call
        tool_results = []
        for tu in tool_uses:
            t2 = time.time()

            # Summarize input
            input_summary = {}
            for k, v in tu.input.items():
                if isinstance(v, str) and len(v) > 100:
                    input_summary[k] = v[:100] + "..."
                else:
                    input_summary[k] = v

            read_tools = {'recall', 'find_node_by_title', 'get_node'}
            write_tools = {'remember', 'revise', 'connect', 'record_divergence',
                          'learn_vocabulary', 'remember_lesson', 'remember_mechanism',
                          'remember_mental_model', 'remember_impact', 'remember_convention'}

            is_read = tu.name in read_tools
            is_write = tu.name in write_tools

            if is_read:
                # Execute against real brain
                try:
                    from servers.daemon_dispatch import COMMAND_TABLE
                    entry = COMMAND_TABLE.get(tu.name)
                    if entry:
                        result = entry.handler(brain, tu.input, [])
                        from servers import brain_mcp
                        if result.get("ok"):
                            result_text = brain_mcp._format_result(tu.name, result.get("result", {}))
                            status = "OK"
                        else:
                            result_text = "ERROR: %s" % result.get("error", "?")
                            status = "ERROR"
                            errors.append({
                                "round": round_num + 1,
                                "tool": tu.name,
                                "input": input_summary,
                                "error": result.get("error", "?")
                            })
                    else:
                        result_text = "Unknown tool: %s" % tu.name
                        status = "UNKNOWN"
                except Exception as e:
                    result_text = "EXCEPTION: %s" % e
                    status = "EXCEPTION"
                    errors.append({
                        "round": round_num + 1,
                        "tool": tu.name,
                        "input": input_summary,
                        "error": str(e)
                    })
            else:
                # Dry run for writes
                result_text = '{"ok": true, "result": {"id": "dryrun_%d", "status": "captured (dry run)"}}'  % len(all_actions)
                status = "DRY_RUN"

            tool_ms = int((time.time() - t2) * 1000)

            action = {
                "round": round_num + 1,
                "tool": tu.name,
                "input": input_summary,
                "status": status,
                "ms": tool_ms,
                "output_chars": len(result_text),
            }
            all_actions.append(action)

            # Log
            marker = "📖" if is_read else "✏️" if is_write else "?"
            print("  %s [%s] %s → %s (%dms, %d chars)" % (
                marker, tu.name,
                json.dumps(input_summary, default=str)[:120],
                status, tool_ms, len(result_text)))

            if status == "ERROR":
                print("    ⚠️ ERROR: %s" % result_text[:200])

            # Show what recall found (abbreviated)
            if tu.name == 'recall' and status == "OK" and 'No results' not in result_text:
                lines = result_text.split('\n')
                node_lines = [l for l in lines if l.startswith('[')]
                for nl in node_lines[:3]:
                    print("    found: %s" % nl[:80])

            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result_text,
            })

        # Build next API message
        api_messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        api_messages.append({"role": "user", "content": tool_results})

    # Summary
    print()
    print("=" * 70)
    print("TRACE SUMMARY")
    print("=" * 70)

    reads = [a for a in all_actions if a["status"] == "OK"]
    writes = [a for a in all_actions if a["status"] == "DRY_RUN"]
    errs = [a for a in all_actions if a["status"] in ("ERROR", "EXCEPTION")]

    print("Total rounds: %d" % (round_num + 1))
    print("Total tokens: %d in + %d out = %d" % (total_input_tokens, total_output_tokens, total_input_tokens + total_output_tokens))
    print("Total actions: %d (%d reads, %d writes, %d errors)" % (
        len(all_actions), len(reads), len(writes), len(errs)))
    print()

    # Tool usage breakdown
    tool_counts = {}
    for a in all_actions:
        key = a["tool"]
        if key not in tool_counts:
            tool_counts[key] = {"count": 0, "ok": 0, "error": 0, "dry": 0, "total_ms": 0}
        tool_counts[key]["count"] += 1
        tool_counts[key]["total_ms"] += a["ms"]
        if a["status"] == "OK": tool_counts[key]["ok"] += 1
        elif a["status"] == "DRY_RUN": tool_counts[key]["dry"] += 1
        else: tool_counts[key]["error"] += 1

    print("Tool usage:")
    for tool, stats in sorted(tool_counts.items(), key=lambda x: -x[1]["count"]):
        print("  %-25s %dx (ok:%d dry:%d err:%d) avg:%dms" % (
            tool, stats["count"], stats["ok"], stats["dry"], stats["error"],
            stats["total_ms"] // max(1, stats["count"])))

    if errors:
        print()
        print("ERRORS ENCOUNTERED:")
        for e in errors:
            print("  Round %d [%s]: %s" % (e["round"], e["tool"], e["error"][:100]))
            print("    Input: %s" % json.dumps(e["input"], default=str)[:120])

    # Check: what writes would the agent have made?
    print()
    print("WRITES (dry run):")
    for a in writes:
        inp = a["input"]
        if a["tool"] in ("remember", "remember_lesson", "remember_mechanism",
                         "remember_mental_model", "remember_impact", "remember_convention"):
            print("  CREATE [%s] %s" % (inp.get("type", "?"), inp.get("title", "?")[:60]))
            filled = [k for k in inp if k not in ("type", "title", "content", "keywords") and inp[k]]
            if filled:
                print("    fields: %s" % ", ".join(filled))
        elif a["tool"] == "revise":
            print("  REVISE %s reason=%s" % (inp.get("node_id", "?")[:8], inp.get("reason", "?")[:60]))
        elif a["tool"] == "connect":
            print("  CONNECT %s -[%s]-> %s" % (
                inp.get("source_id", "?")[:8], inp.get("relation", "?"),
                inp.get("target_id", "?")[:8]))
        elif a["tool"] == "learn_vocabulary":
            print("  VOCAB %s → %s" % (inp.get("term", "?"), inp.get("maps_to", "?")))


if __name__ == "__main__":
    main()
