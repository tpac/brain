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

from .dispatch import make_scale_dispatch


# Hard upper bound on any single Anthropic SDK call (S1 surface, S1
# encode, S2 encoders, scouts). The SDK default is roughly 600s but is
# measured against time.monotonic(), which does NOT advance while the
# process is suspended (macOS sleep). A call started right before sleep
# can therefore hang indefinitely after wake. The autosave loop's wall-
# clock gap detector triggers a daemon restart on detected suspend
# events; this constant bounds normal-mode hangs (slow API, throttled
# response, etc.) so a stuck call doesn't tie up a worker forever.
# Community encoder round 2 on cold-cache batches can legitimately take
# ~218s; 600s leaves headroom without inviting silence.
ANTHROPIC_CLIENT_TIMEOUT = 600.0


# The created/revised/archived node-lifecycle split is no longer re-derived
# here from tool names. Each dispatch write handler returns the authoritative
# `affected` dict (it knows its op + has the brain result, incl. connect_to
# edges the old tool-name heuristic couldn't see). The runner just reads it off
# the dispatch return — see _dispatch_tool_uses. Edges are not in `affected`;
# they're directional edge_relation_revised traces emitted by the handlers.


# Canonical token-usage telemetry field names — the keys run_llm_loop returns,
# base._call_llm returns, and build_delta_metadata accepts. Defined once so the
# SDK-attribute mapping (below) and the encoders' cross-batch accumulator
# (IntegrationUnit._sum_telemetry) read the same field set and can't drift.
USAGE_FIELDS = ('input_tokens', 'output_tokens',
                'cache_read_tokens', 'cache_creation_tokens')


def read_usage(response):
    """Map an Anthropic response's `.usage` onto USAGE_FIELDS (all int, 0 when
    the attribute or the whole usage object is absent).

    Single source for the SDK attribute names so run_llm_loop's per-round
    tracking and base._call_llm's single-shot path read them identically — an
    SDK field rename is fixed in one place. `read_usage(None)` returns the
    all-zero dict (the no-response / pre-call telemetry baseline)."""
    usage = getattr(response, 'usage', None)
    return {
        'input_tokens':          getattr(usage, 'input_tokens', 0) or 0,
        'output_tokens':         getattr(usage, 'output_tokens', 0) or 0,
        'cache_read_tokens':     getattr(usage, 'cache_read_input_tokens', 0) or 0,
        'cache_creation_tokens': getattr(usage, 'cache_creation_input_tokens', 0) or 0,
    }


def run_in_background(name, brain_db_path, session_id, counter, lock,
                      run_fn, encoding_source='encoder:sonnet', on_complete=None):
    """Run a scale agent in a background thread.

    Args:
        name: Scale name for logging (e.g. 's1e', 's2')
        brain_db_path: Path to brain.db
        session_id: Session ID from SessionContext
        counter: Stop counter value
        lock: threading.Lock for mutual exclusion (one agent at a time)
        run_fn: Scale's run function: run_fn(brain, dispatch_fn, counter, session_id) -> dict
        encoding_source: encoding_source value for new nodes
        on_complete: optional callback(write_actions: int) invoked AFTER a
            successful run, in this background thread. The run executes against
            a throwaway read_brain (writes go via TCP), so a caller that needs
            to record on ITS OWN brain (e.g. S1E counting encode runs toward the
            S2 gate on brain.activity) passes a closure over that brain here —
            same process, so the closure is valid across the thread. Receives
            write_actions so the caller can gate on "actually wrote material".
            Never called if run_fn raised.

    The delta trace is written by the scale's own run_fn via build_delta_metadata
    (the unified shape). This wrapper does NOT write one — a previous version did,
    producing a SECOND, brain_batch-blind `encoding_run` delta per cycle (the
    structured node-lifecycle split was always empty). That legacy writer was
    removed; the runner only owns thread lifecycle now.
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

            # Completion hook — runs on the CALLER's brain (closure), not the
            # throwaway read_brain. Gated on write_actions so it reflects real
            # written material. Guarded so a callback error never crashes the
            # thread or masks the run.
            if on_complete is not None:
                try:
                    write_actions = (result.get('write_actions', 0)
                                     if isinstance(result, dict) else 0)
                    on_complete(write_actions)
                except Exception as _oce:
                    if read_brain:
                        try:
                            read_brain._log_error(
                                'scale_runner_on_complete', _oce,
                                '%s on_complete callback failed' % name)
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


def run_unit_in_background(unit, name, lock, on_complete=None):
    """Run an in-process integration unit on a daemon worker thread.

    The in-process counterpart to run_in_background: the unit runs on the
    daemon's OWN brain and writes through its `_make_encoder_dispatch` (direct,
    under brain.write_lock; vectors via the async embed_queue) — no throwaway
    Brain copy, no TCP round-trip. Used by S1 Scribe now that it's converged
    onto the in-process IntegrationUnit pattern the S2 units already use.

    Owns only thread lifecycle: it runs unit.run(), invokes on_complete with the
    run's write_actions (so callers can gate on "actually wrote material" — e.g.
    the S2 activity counter), and releases `lock` in finally so a crash can't
    wedge the encoder. Mirrors run_in_background's contract; the caller transfers
    lock ownership to this thread exactly as before.
    """
    def _thread_fn():
        t0 = time.time()
        try:
            print("[%s] STARTING" % name, flush=True)
            result = unit.run()
            elapsed_ms = int((time.time() - t0) * 1000)
            actions = result.get('actions', 0) if isinstance(result, dict) else 0
            print("[%s] DONE: %d actions in %dms" % (name, actions, elapsed_ms), flush=True)

            if on_complete is not None:
                try:
                    write_actions = (result.get('write_actions', 0)
                                     if isinstance(result, dict) else 0)
                    on_complete(write_actions)
                except Exception as _oce:
                    try:
                        unit.brain._log_error(
                            'scale_runner_on_complete', _oce,
                            '%s on_complete callback failed' % name)
                    except Exception:
                        pass
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            print("[%s] FAILED after %dms: %s" % (name, elapsed_ms, e), flush=True)
            # Background thread crash — the encoder silently stopped producing.
            # Surface it the same way run_in_background does.
            try:
                unit.brain._log_error(
                    'scale_runner_thread_crash', e,
                    '%s thread died after %dms' % (name, elapsed_ms))
            except Exception:
                pass
        finally:
            lock.release()

    threading.Thread(target=_thread_fn, daemon=True, name=name).start()


def run_llm_loop(client, model, max_tokens, max_rounds, system_prompt,
                 user_content, tools, dispatch_fn, log_fn=None,
                 user_preamble=None, get_nodes_config=None):
    """Generic LLM tool loop — call model, process tool_use, dispatch, repeat.

    Used by all scale encode agents. Scale-specific logic is in what
    system_prompt, user_content, and tools contain — the loop is identical.

    Prompt caching: up to 3 `cache_control` markers placed by stability:
      - System (1h TTL): byte-identical across every call within a prompt
        version. Whole eval / long session keeps system warm.
      - User preamble (1h TTL, optional): stable instructions/format
        reminders moved to the START of the user content. When provided,
        creates a cross-call cached block that joins system in the cache.
      - User content body (5m TTL): the dynamic per-cycle content
        (catalog, journal, timeline). Cached within a single run so
        round-2 hits cache reliably.

    Args:
        client: anthropic.Anthropic() instance
        model: Model ID (e.g. 'claude-sonnet-4-6')
        max_tokens: Max output tokens
        max_rounds: Max tool-use rounds before stopping
        system_prompt: System prompt string
        user_content: User content string (the dynamic body)
        tools: List of tool schema dicts
        dispatch_fn: function(cmd, args) -> dict for tool execution
        log_fn: Optional function(msg) for logging
        user_preamble: Optional stable string to prefix the user content with
            its own 1h cache breakpoint. Use for instructions/format that
            don't change per call. None disables (single-block user content).
        get_nodes_config: Optional render_rich_node config dict. When set, the
            caller's `get_nodes` tool results render through render_rich_node
            with THIS config at every batch size — overriding _format_result's
            batch-size heuristic and (critically) its <=3-node raw-JSON escape
            hatch, which dumps full _corrections and can explode an encoder's
            context. None = default batch-size-driven rendering. Consumers with
            tight token budgets (S2 encoders) pass their own lean config.

    Returns:
        dict with: rounds, actions, write_actions, action_details,
                   read_calls, final_text, profile, cache_*_tokens
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    WRITE_TOOLS = {
        'remember', 'remember_batch', 'revise', 'revise_batch',
        'connect', 'connect_batch', 'brain_batch',
    }

    t0 = time.time()
    profile = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation = 0
    total_cache_read = 0

    # Per-round diagnostic data so we can answer "where did r1's 100s go".
    # Each entry: {round, ttft_ms, total_ms, output_tokens, input_tokens,
    # cache_read, cache_creation}. ttft_ms isolates server prefill (largely
    # invariant in generation cost) from the actual token-by-token output.
    per_round_stats = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    truncations = []

    def _track_usage(resp, round_num, ttft_ms=None, total_ms=None):
        nonlocal total_input_tokens, total_output_tokens
        nonlocal total_cache_creation, total_cache_read
        u = read_usage(resp)
        in_this_round = u['input_tokens']
        out_this_round = u['output_tokens']
        cr_this_round = u['cache_read_tokens']
        cw_this_round = u['cache_creation_tokens']
        total_input_tokens += in_this_round
        total_output_tokens += out_this_round
        total_cache_creation += cw_this_round
        total_cache_read += cr_this_round
        per_round_stats.append({
            'round': round_num,
            'ttft_ms': ttft_ms,
            'total_ms': total_ms,
            'output_tokens': out_this_round,
            'input_tokens': in_this_round,
            'cache_read': cr_this_round,
            'cache_creation': cw_this_round,
        })
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            _log("WARNING: max_tokens hit (round %d, %s/%d output tokens) — response truncated" % (
                round_num, out_this_round, max_tokens))
            truncations.append({
                'round': round_num,
                'output_tokens': out_this_round,
                'max_tokens': max_tokens,
            })

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
        """Create message with streaming. Returns (final_message, ttft_ms,
        total_ms). ttft_ms is the time from request issue to the first
        server event — isolates "server prefill / cache lookup / wait"
        from token-by-token generation time. None if iteration fails
        (we fall back to the unmeasured path so a diagnostic glitch
        can't kill an encoder cycle)."""
        request_t0 = time.time()
        ttft_ms = None
        try:
            with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_param, messages=msgs, tools=tools) as stream:
                # Iterate raw events so we can timestamp the first server
                # signal. SDK's stream object exposes a synchronous iter
                # that yields RawMessageStreamEvent objects — message_start,
                # content_block_start, content_block_delta, content_block_stop,
                # message_delta, message_stop. We only need the first event
                # to know when output began.
                for _event in stream:
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - request_t0) * 1000)
                final_msg = stream.get_final_message()
        except Exception:
            # Defensive fallback — if event iteration breaks for any reason,
            # re-issue without timing. Costs an extra API call only on bug,
            # never on happy path. Better than nuking the encoder cycle.
            with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_param, messages=msgs, tools=tools) as stream:
                final_msg = stream.get_final_message()
            ttft_ms = None
        total_ms = int((time.time() - request_t0) * 1000)
        return final_msg, ttft_ms, total_ms

    # User content. When `user_preamble` is provided, it forms a stable 1h
    # block that joins system in the cache (cross-call reuse). The dynamic
    # `user_content` body keeps a 5m breakpoint so within a single
    # run_llm_loop, round-2 always reads from cache.
    user_blocks = []
    if user_preamble:
        # BP-stable-user — 1h cached, byte-identical across calls
        user_blocks.append({
            "type": "text",
            "text": user_preamble,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })
    # BP-dynamic-user — 5m cached, the per-cycle content
    user_blocks.append({
        "type": "text",
        "text": user_content,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    })
    api_messages = [{"role": "user", "content": user_blocks}]
    response, ttft_ms, total_ms = _create_message(api_messages)
    _track_usage(response, 0, ttft_ms=ttft_ms, total_ms=total_ms)
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
                result_text = brain_mcp._format_result(
                    tu.name, result.get("result", {}),
                    get_nodes_config=get_nodes_config)
            else:
                result_text = "ERROR: %s" % result.get("error", "Unknown")
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result_text,
            })
            action_summary = tu.input.get("title", tu.input.get("query",
                tu.input.get("node_id", "")))[:60]
            result_ids = []
            # Authoritative node-lifecycle split — the dispatch handler returned
            # it as a TOP-LEVEL `affected` (sibling of `result`), computed where
            # the op + brain result are both known. build_delta_metadata
            # aggregates created/revised/archived over these; S2 reads them.
            affected = result.get("affected") if result.get("ok") else None
            affected = affected if isinstance(affected, dict) else {}
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
                            "created": affected.get("created") or [],
                            "revised": affected.get("revised") or [],
                            "archived": affected.get("archived") or [],
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
        response, ttft_ms, total_ms = _create_message(api_messages)
        _track_usage(response, rounds + 1, ttft_ms=ttft_ms, total_ms=total_ms)
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

    # Per-round breakdown — answers "where did r1's wall-clock go?"
    # Format: r0 ttft=12000ms gen=75798ms out=4521tok in=6432 cr=28135 cw=28135
    # ttft separates server prefill from generation. If gen >> output/50tok-per-sec,
    # there's slow generation; if ttft >> gen, prefill dominates (large context).
    for rd in per_round_stats:
        ttft = rd.get('ttft_ms')
        tot = rd.get('total_ms') or 0
        gen = (tot - ttft) if (ttft is not None and tot) else None
        out_tok = rd.get('output_tokens') or 0
        gen_rate = (out_tok * 1000.0 / gen) if (gen and gen > 0) else None
        _log("  r%d ttft=%s gen=%s out=%dtok rate=%s in=%d cr=%d cw=%d" % (
            rd['round'],
            '%dms' % ttft if ttft is not None else '?',
            '%dms' % gen if gen is not None else '?',
            out_tok,
            ('%.0ftok/s' % gen_rate) if gen_rate is not None else '?',
            rd.get('input_tokens') or 0,
            rd.get('cache_read') or 0,
            rd.get('cache_creation') or 0))

    return {
        "rounds": rounds + 1,
        "actions": len(actions),
        "write_actions": len(write_actions),
        "action_details": write_actions,
        "read_calls": read_calls,
        # Full text — build_delta_metadata caps the TRACE loudly, and
        # _save_journal / _save_session_context self-cap. Pre-truncating here
        # made the trace's loud-truncation marker dead (silent drop upstream).
        "final_text": final_text or '',
        "profile": profile,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_creation_tokens": total_cache_creation,
        "cache_read_tokens": total_cache_read,
        "truncations": truncations,
    }
