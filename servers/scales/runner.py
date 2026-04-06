"""Scale runner infrastructure — background thread lifecycle for scale agents.

Every scale agent (S1 encode, S2 session encode, future scales) follows
the same lifecycle:
1. Create read-only Brain instance
2. Create dispatch function (reads local, writes via TCP)
3. Call the scale's run function
4. Write delta trace
5. Release lock, close brain

This module provides the generic lifecycle. Scale-specific logic lives
in each scale's module (scales/s1/encode.py, scales/s2/encode.py, etc.).
"""

import time
import threading

from .dispatch import make_scale_dispatch, daemon_tcp_send


def run_in_background(name, brain_db_path, session_id, counter, lock,
                      run_fn, encoding_source='encoder:sonnet',
                      trace_scale='s1', trace_chain_fn=None):
    """Run a scale agent in a background thread.

    Args:
        name: Scale name for logging (e.g. 's1e', 's2')
        brain_db_path: Path to brain.db
        session_id: Session ID from SessionContext
        counter: Stop counter value
        lock: threading.Lock for mutual exclusion (one agent at a time)
        run_fn: Scale's run function: run_fn(brain, dispatch_fn, counter, session_id) -> dict
        encoding_source: encoding_source value for new nodes
        trace_scale: Scale for delta trace ('s1', 's2', etc.)
        trace_chain_fn: Function(session_id, counter) -> chain_id for delta trace.
                        If None, no delta trace is written (scale writes its own).
    """
    def _thread_fn():
        t0 = time.time()
        read_brain = None
        try:
            print("[%s] STARTING (counter=%d)" % (name, counter), flush=True)
            from servers.brain import Brain
            read_brain = Brain(brain_db_path)

            dispatch = make_scale_dispatch(read_brain, encoding_source=encoding_source)

            result = run_fn(read_brain, dispatch, counter, session_id)
            elapsed_ms = int((time.time() - t0) * 1000)
            actions = result.get('actions', 0) if isinstance(result, dict) else 0
            print("[%s] DONE: %d actions in %dms" % (name, actions, elapsed_ms), flush=True)

            # Write delta trace if chain function provided
            if trace_chain_fn:
                try:
                    chain_id = trace_chain_fn(session_id, counter)
                    action_lines = []
                    for a in (result.get('action_details', []) if isinstance(result, dict) else []):
                        action_lines.append('%s: %s' % (a.get('tool', ''), a.get('summary', '')))
                    daemon_tcp_send('trace_append', {
                        'chain_id': chain_id,
                        'scale': trace_scale,
                        'event_type': 'delta',
                        'ref_type': 'encoding_run',
                        'ref_id': str(counter),
                        'summary': '%d actions in %dms:\n%s\n---\n%s' % (
                            actions, elapsed_ms,
                            '\n'.join(action_lines) if action_lines else '(no actions)',
                            (result.get('final_text', '') or '')[:2000]),
                        'session_id': session_id,
                    })
                except Exception as e:
                    print('[%s] TRACE ERROR (delta): %s' % (name, e), flush=True)

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            print("[%s] FAILED after %dms: %s" % (name, elapsed_ms, e), flush=True)
        finally:
            if read_brain:
                try:
                    read_brain.close()
                except Exception:
                    pass
            lock.release()

    threading.Thread(target=_thread_fn, daemon=True, name=name).start()


def run_llm_loop(client, model, max_tokens, max_rounds, system_prompt,
                 user_content, tools, dispatch_fn, log_fn=None):
    """Generic LLM tool loop — call model, process tool_use, dispatch, repeat.

    Used by all scale encode agents. Scale-specific logic is in what
    system_prompt, user_content, and tools contain — the loop is identical.

    Args:
        client: anthropic.Anthropic() instance
        model: Model ID (e.g. 'claude-sonnet-4-6')
        max_tokens: Max output tokens
        max_rounds: Max tool-use rounds before stopping
        system_prompt: System prompt string
        user_content: User content string
        tools: List of tool schema dicts
        dispatch_fn: function(cmd, args) -> dict for tool execution
        log_fn: Optional function(msg) for logging

    Returns:
        dict with: rounds, actions, write_actions, action_details, final_text, profile
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    WRITE_TOOLS = {
        'remember', 'remember_batch', 'revise', 'revise_batch',
        'connect',
    }

    t0 = time.time()
    profile = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    api_messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system_prompt, messages=api_messages, tools=tools)
    _step("llm_r0")

    actions = []
    rounds = 0

    for rounds in range(max_rounds):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            result = dispatch_fn(tu.name, tu.input)
            from servers import brain_mcp
            if result.get("ok"):
                result_text = brain_mcp._format_result(tu.name, result.get("result", {}))
            else:
                result_text = "ERROR: %s" % result.get("error", "Unknown")
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result_text,
            })
            action_summary = tu.input.get("title", tu.input.get("query",
                tu.input.get("node_id", "")))[:60]
            actions.append({"tool": tu.name, "summary": action_summary})
            _log("  [%s] %s" % (tu.name, action_summary))

        api_messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        api_messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system_prompt, messages=api_messages, tools=tools)
        _step("llm_r%d" % (rounds + 1))

    final_text = "".join(b.text for b in response.content if b.type == "text")
    write_actions = [a for a in actions if a['tool'] in WRITE_TOOLS]

    return {
        "rounds": rounds + 1,
        "actions": len(actions),
        "write_actions": len(write_actions),
        "action_details": write_actions,
        "final_text": final_text[:2000] if final_text else '',
        "profile": profile,
    }
