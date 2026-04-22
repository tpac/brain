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
            # skip_embedder=True: background threads don't embed directly.
            # Writes go through daemon TCP (single-writer rule) where the
            # daemon's main thread handles embedding. Loading ONNX in a
            # background thread causes inter-thread spinning on macOS.
            read_brain = Brain(brain_db_path, skip_embedder=True)

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
                    # Build structured metadata for S2 consumption
                    created_ids = []
                    revised_ids = []
                    connected_pairs = []
                    for a in (result.get('action_details', []) if isinstance(result, dict) else []):
                        tool = a.get('tool', '')
                        nids = a.get('node_ids', [])
                        if tool in ('remember', 'remember_batch'):
                            created_ids.extend(nids)
                        elif tool in ('revise', 'revise_batch'):
                            revised_ids.extend(nids)
                        elif tool in ('connect', 'connect_batch'):
                            connected_pairs.extend(nids)

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
                        'metadata': {
                            'created': created_ids,
                            'revised': revised_ids,
                            'connected': connected_pairs,
                            'elapsed_ms': elapsed_ms,
                        },
                        'session_id': session_id,
                    })
                except Exception as e:
                    print('[%s] TRACE ERROR (delta): %s' % (name, e), flush=True)
                    if read_brain:
                        try:
                            read_brain._log_error(
                                'scale_runner_trace_write', e,
                                '%s delta trace write failed' % name)
                        except Exception:
                            pass

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            print("[%s] FAILED after %dms: %s" % (name, elapsed_ms, e), flush=True)
            # Background thread crash — the scale encoder silently stopped
            # producing. Surface so operators see recurring failures the
            # same way they see S2 coordinator crashes.
            if read_brain:
                try:
                    read_brain._log_error(
                        'scale_runner_thread_crash', e,
                        '%s thread died after %dms' % (name, elapsed_ms))
                except Exception:
                    pass
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

    Prompt caching (enabled always): `cache_control` markers on system and
    on the initial user message. System is 1h TTL for cross-call reuse
    within long eval runs; user message is 5m TTL so the 2nd turn (after
    tool result) hits the cache reliably. Net ~15-25% per-scribe cost
    saving, ~40-50% input token reduction. Cache usage is reported via
    cache_creation_tokens / cache_read_tokens in the return dict.

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
        dict with: rounds, actions, write_actions, action_details,
                   read_calls, final_text, profile, cache_*_tokens
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    WRITE_TOOLS = {
        'remember', 'remember_batch', 'revise', 'revise_batch',
        'connect', 'brain_batch',
    }

    t0 = time.time()
    profile = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation = 0
    total_cache_read = 0

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    truncations = []

    def _track_usage(resp, round_num):
        nonlocal total_input_tokens, total_output_tokens
        nonlocal total_cache_creation, total_cache_read
        if hasattr(resp, 'usage'):
            total_input_tokens += getattr(resp.usage, 'input_tokens', 0)
            total_output_tokens += getattr(resp.usage, 'output_tokens', 0)
            total_cache_creation += getattr(resp.usage, 'cache_creation_input_tokens', 0) or 0
            total_cache_read += getattr(resp.usage, 'cache_read_input_tokens', 0) or 0
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            out_used = getattr(resp.usage, 'output_tokens', 0) if hasattr(resp, 'usage') else '?'
            truncations.append({
                'round': round_num,
                'output_tokens': out_used,
                'max_tokens': max_tokens,
            })
            _log("WARNING: max_tokens hit (round %d, %s/%d output tokens) — response truncated" % (
                round_num, out_used, max_tokens))

    # BP1 — tools + system cached at 1h TTL. System prompt is byte-identical
    # across every call within a prompt version; 1h TTL keeps a whole eval or
    # a long chat session warm. Writes cost 2× on the first call of the hour;
    # amortizes after ~2 reads.
    system_param = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }]

    def _create_message(msgs):
        """Create message with streaming to avoid timeout on large contexts."""
        with client.messages.stream(
                model=model, max_tokens=max_tokens,
                system=system_param, messages=msgs, tools=tools) as stream:
            return stream.get_final_message()

    # BP2 — entire turn-1 user content cached at 5m TTL. Within a single
    # run_llm_loop invocation, turn 2 re-sends this exact block and reads
    # from cache (always a hit, 0 seconds after turn 1 writes). Across
    # calls this usually misses (journal/timeline shift) — but the within-
    # call win is ~6k tokens on turn 2 which is where latency matters.
    api_messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": user_content,
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }],
    }]
    response = _create_message(api_messages)
    _track_usage(response, 0)
    _step("llm_r0")

    actions = []
    rounds = 0

    def _dispatch_tool_uses(response_obj):
        """Dispatch every tool_use block in response. Append to actions.
        Returns a tool_results list usable as a user-role message block.
        """
        tool_uses = [b for b in response_obj.content if b.type == "tool_use"]
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
            result_ids = []
            if result.get("ok"):
                r = result.get("result", {})
                if isinstance(r, dict):
                    if r.get("id"):
                        result_ids.append(r["id"])
                    for item in r.get("results", []):
                        if isinstance(item, dict):
                            if item.get("id"):
                                result_ids.append(item["id"])
                            elif isinstance(item.get("result"), dict) and item["result"].get("id"):
                                result_ids.append(item["result"]["id"])
                elif isinstance(r, list):
                    for item in r:
                        if isinstance(item, dict) and item.get("id"):
                            result_ids.append(item["id"])
            actions.append({"tool": tu.name, "summary": action_summary,
                            "node_ids": result_ids,
                            "input": tu.input})
            _log("  [%s] %s" % (tu.name, action_summary))
        return tool_results, tool_uses

    for rounds in range(max_rounds):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results, _ = _dispatch_tool_uses(response)

        api_messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        api_messages.append({"role": "user", "content": tool_results})
        response = _create_message(api_messages)
        _track_usage(response, rounds + 1)
        _step("llm_r%d" % (rounds + 1))

    final_text = "".join(b.text for b in response.content if b.type == "text")
    write_actions = [a for a in actions if a['tool'] in WRITE_TOOLS]
    read_calls = [a for a in actions if a['tool'] not in WRITE_TOOLS]

    # Total "billed as fresh input" = uncached input. cache_read_input_tokens
    # is read from cache at 0.1× cost; cache_creation_input_tokens is written
    # at 1.25× (5min TTL) or 2× (1h TTL) cost. Report all three so the
    # operator can see hit ratio over many runs.
    total_cache_pool = total_cache_creation + total_cache_read
    hit_rate = (total_cache_read / total_cache_pool * 100) if total_cache_pool else 0.0
    _log("Rounds: %d | Actions: %d (writes: %d, reads: %d) | Tokens: %d fresh / %d cached-read / %d cached-write / %d out | hit=%.0f%% | Profile: %s" % (
        rounds + 1, len(actions), len(write_actions), len(read_calls),
        total_input_tokens, total_cache_read, total_cache_creation, total_output_tokens,
        hit_rate,
        ', '.join('%s=%dms' % (n, t) for n, t in profile)))

    return {
        "rounds": rounds + 1,
        "actions": len(actions),
        "write_actions": len(write_actions),
        "action_details": write_actions,
        "read_calls": read_calls,
        "final_text": final_text[:2000] if final_text else '',
        "profile": profile,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_creation_tokens": total_cache_creation,
        "cache_read_tokens": total_cache_read,
        "truncations": truncations,
    }
