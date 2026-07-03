"""Scale runner infrastructure — background-thread lifecycle + the LLM tool loop.

Scale agents (S1 Scribe, the S2 units, future scales) run IN-PROCESS on the
daemon's brain on a background worker thread, so a slow encode never blocks the
hook / coordinator that launched it. `run_unit_in_background` owns that thread
lifecycle (run the unit, fire on_complete, release the lock); the unit writes
through its own `_make_encoder_dispatch` under brain.write_lock. The generic
`run_llm_loop` (the call → tool_use → dispatch loop with prompt caching) lives
here too. Scale-specific logic lives in each scale's module (scales/s1/scribe.py
+ encode.py, scales/s2/*).
"""

import time
import threading
import os
import json


# Full-prompt capture (eval/observability). OFF unless BRAIN_PROMPT_CAPTURE_DIR
# is set — production never sets it, so this is a no-op on the live path. When
# set, run_llm_loop dumps the LITERAL per-round payload (system + messages) so
# the actual prompt is recoverable without an unfaithful rebuild. The seq is a
# process-monotonic tiebreaker so two captures can never overwrite each other,
# even if the (label, round) key ever collides. See docs/S1-SCRIBE-REDESIGN.md.
_CAPTURE_SEQ = 0
_CAPTURE_SEQ_LOCK = threading.Lock()


def _next_capture_seq():
    global _CAPTURE_SEQ
    with _CAPTURE_SEQ_LOCK:
        _CAPTURE_SEQ += 1
        return _CAPTURE_SEQ


# Hard upper bound on any single Anthropic SDK call (S1 surface, S1
# encode, S2 encoders, scouts). The SDK default is roughly 600s but is
# measured against time.monotonic(), which does NOT advance while the
# process is suspended (macOS sleep). A call started right before sleep
# can therefore hang indefinitely after wake. A post-sleep hang is
# recovered reactively (ensure_daemon at session start / the MCP health
# monitor during a session, both force-restarting via launchctl
# kickstart -k); this constant bounds normal-mode hangs (slow API,
# throttled response, etc.) so a stuck call doesn't tie up a worker forever.
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
# SDK-attribute mapping (read_usage) and the accumulator (sum_usage, reused by
# run_llm_loop, IntegrationUnit._accumulate_run, and the surface loop) read the
# same field set and can't drift.
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


def sum_usage(total, usage):
    """Accumulate one `usage` dict's USAGE_FIELDS into `total`, in place
    (defensive on missing/None), and return `total`.

    The single token-accumulator primitive — one definition reused by
    run_llm_loop's per-round tracking, IntegrationUnit's per-batch run
    accumulation, and the surface agentic loop, so the "sum the four token
    fields" loop exists exactly once. Start `total` from read_usage(None) (the
    all-zero baseline) for a clean dict of the right keys."""
    for f in USAGE_FIELDS:
        total[f] = total.get(f, 0) + (usage.get(f) or 0)
    return total


def run_unit_in_background(unit, name, lock, on_complete=None):
    """Run an in-process integration unit on a daemon worker thread.

    The unit runs on the daemon's OWN brain and writes through its
    `_make_encoder_dispatch` (direct, under brain.write_lock; vectors via the
    async embed_queue). The background thread keeps a slow encode from blocking
    the hook / coordinator that launched it. Used by S1 Scribe and any future
    in-process scale unit. (Replaced the legacy out-of-process path — a
    throwaway Brain copy writing back over TCP.)

    Owns only thread lifecycle: it runs unit.run(), invokes on_complete with the
    run's write_actions (so callers can gate on "actually wrote material" — e.g.
    the S2 activity counter), and releases `lock` in finally so a crash can't
    wedge the encoder. The caller acquires the lock and transfers ownership to
    this thread.
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
            # Surface it so operators see recurring failures (same as the S2
            # coordinator's crash logging).
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
                 user_preamble=None, get_nodes_config=None, capture_label=None,
                 effort=None):
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

        effort: Optional API effort level ('low'|'medium'|'high'|'max').
            Passed as output_config={'effort': ...} on every request. None
            (default) omits output_config entirely — the API default (high).
            Comes from the encoder's interaction `parameters` JSON (the
            K-store), so it's A/B-able per prompt version without code change.

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
    # Token totals as a dict over USAGE_FIELDS, accumulated via the shared
    # sum_usage primitive. read_usage(None) is the all-zero baseline of the
    # right keys; the per-round breakdown derives from per_round_stats.
    usage_total = read_usage(None)

    # Per-round diagnostic data so we can answer "where did r1's 100s go".
    # Each entry: {round, ttft_ms, total_ms} + USAGE_FIELDS (read_usage's keys).
    # ttft_ms isolates server prefill (largely invariant in generation cost)
    # from the actual token-by-token output.
    per_round_stats = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    truncations = []

    def _track_usage(resp, round_num, ttft_ms=None, total_ms=None):
        u = read_usage(resp)
        sum_usage(usage_total, u)
        per_round_stats.append({'round': round_num, 'ttft_ms': ttft_ms,
                                'total_ms': total_ms, **u})
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            _log("WARNING: max_tokens hit (round %d, %s/%d output tokens) — response truncated" % (
                round_num, u['output_tokens'], max_tokens))
            truncations.append({
                'round': round_num,
                'output_tokens': u['output_tokens'],
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

    # Full-prompt capture: dumps the literal payload (system + messages) per
    # round when BRAIN_PROMPT_CAPTURE_DIR is set AND a caller labels the run.
    # No-op otherwise. Never raises into the encode path — capture failure must
    # not break a cycle. `_cap_round` counts rounds within THIS loop; the
    # process-monotonic seq + pid make the filename collision-proof.
    _capture_dir = os.environ.get('BRAIN_PROMPT_CAPTURE_DIR')
    _cap_round = [0]

    def _capture_payload(msgs):
        if not (_capture_dir and capture_label):
            return
        try:
            os.makedirs(_capture_dir, exist_ok=True)
            seq = _next_capture_seq()
            fn = os.path.join(_capture_dir, "%s-r%d-%d-%05d.json" % (
                capture_label, _cap_round[0], os.getpid(), seq))
            with open(fn, 'w') as f:
                json.dump({
                    "label": capture_label,
                    "round": _cap_round[0],
                    "seq": seq,
                    "model": model,
                    "effort": effort,              # None = API default (high)
                    "system": system_prompt,       # full text, not a length
                    "messages": msgs,              # full, every content block
                    "tools": [t.get("name") for t in (tools or [])],
                }, f, ensure_ascii=False)
        except Exception as _e:
            (log_fn or (lambda *_a: None))("[capture] prompt dump failed: %s" % _e)
        finally:
            _cap_round[0] += 1

    def _create_message(msgs):
        """Create message with streaming. Returns (final_message, ttft_ms,
        total_ms). ttft_ms is the time from request issue to the first
        server event — isolates "server prefill / cache lookup / wait"
        from token-by-token generation time. None if iteration fails
        (we fall back to the unmeasured path so a diagnostic glitch
        can't kill an encoder cycle)."""
        _capture_payload(msgs)   # literal per-round payload; no-op unless capturing
        request_t0 = time.time()
        ttft_ms = None
        # effort rides in output_config; None omits it (API default = high).
        extra = {"output_config": {"effort": effort}} if effort else {}
        try:
            with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_param, messages=msgs, tools=tools,
                    **extra) as stream:
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
                    system=system_param, messages=msgs, tools=tools,
                    **extra) as stream:
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
            _tu_t0 = time.time()
            result = dispatch_fn(tu.name, tu.input)
            latency_ms = int((time.time() - _tu_t0) * 1000)
            error = None if result.get("ok") else result.get("error", "Unknown")
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
            # Per-tool result_count — how many result items the call returned
            # (read hits; write op-results). Self-contained count off the result
            # shape; doesn't touch the result_ids parse above. recall_batch
            # returns a list of {query, results} groups, so sum the nested hits
            # rather than count queries; a flat list (get_nodes) counts itself.
            result_count = 0
            if result.get("ok"):
                _r = result.get("result")
                if isinstance(_r, dict):
                    _rl = _r.get("results")
                    result_count = len(_rl) if isinstance(_rl, list) else (1 if _r.get("id") else 0)
                elif isinstance(_r, list):
                    if _r and all(isinstance(x, dict) and isinstance(x.get("results"), list)
                                  for x in _r):
                        result_count = sum(len(x["results"]) for x in _r)
                    else:
                        result_count = len(_r)
            # Per-tool detail (latency_ms/result_count/error) rides on every
            # action record, so action_details (writes) + read_calls (reads) —
            # already threaded into build_delta_metadata by the run_llm_loop
            # encoders — gain the same per-call observability Surface's
            # tool_trace has. `input` is the args.
            actions.append({"tool": tu.name, "summary": action_summary,
                            "node_ids": result_ids,
                            "created": affected.get("created") or [],
                            "revised": affected.get("revised") or [],
                            "archived": affected.get("archived") or [],
                            "input": tu.input,
                            "latency_ms": latency_ms,
                            "result_count": result_count,
                            "error": error})
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
    total_cache_pool = usage_total['cache_creation_tokens'] + usage_total['cache_read_tokens']
    hit_rate = (usage_total['cache_read_tokens'] / total_cache_pool * 100) if total_cache_pool else 0.0
    _log("Rounds: %d | Actions: %d (writes: %d, reads: %d) | Tokens: %d fresh / %d cached-read / %d cached-write / %d out | hit=%.0f%% | Profile: %s" % (
        rounds + 1, len(actions), len(write_actions), len(read_calls),
        usage_total['input_tokens'], usage_total['cache_read_tokens'],
        usage_total['cache_creation_tokens'], usage_total['output_tokens'],
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
            rd.get('cache_read_tokens') or 0,
            rd.get('cache_creation_tokens') or 0))

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
        # USAGE_FIELDS keys match the four token return keys exactly.
        **usage_total,
        "truncations": truncations,
    }
